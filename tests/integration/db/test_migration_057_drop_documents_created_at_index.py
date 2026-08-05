"""``057_drop_documents_created_at_index`` — PostgreSQL lane.

The SQLite lane is
``tests/unit/db/test_migration_057_drop_documents_created_at_index.py`` and
carries the coverage rationale; it needs no server and runs on every PR.
This module drives the same lifecycle against a real server, which is a
genuinely different path rather than the same test with a different DSN:
only here does the revision's Postgres branch execute its
``SET LOCAL lock_timeout = '5s'``, and only here can the restored index be
read back out of ``pg_indexes`` as the definition Postgres actually built.

No test here touches the shared dev database. Both classes own a throwaway
one (``tests/test_helpers/pg_scratch_db.py``). That is deliberate rather
than tidy: these tests downgrade and re-upgrade, and CI runs the integration
job with ``--timeout-method=thread``, which kills the process outright so
``finally`` blocks do not run. A rewind-then-restore against the shared
database would strand it at a stale revision on timeout, and every later test
in the serial job would fail against a stale schema with the real cause
several tests back. Owning the database bounds the worst case to one leaked,
uniquely named database.

Run explicitly (the shell may leak a different URL)::

    KHORA_DATABASE_URL="postgresql://khora:khora@localhost:5434/khora" \
        UV_NO_SYNC=1 uv run pytest \
        tests/integration/db/test_migration_057_drop_documents_created_at_index.py \
        -o addopts="" --no-cov -q
"""

from __future__ import annotations

import asyncio
import threading
import time
import warnings
from collections.abc import Iterator

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import create_async_engine

from tests.test_helpers.documents_created_at_index import (
    BELOW_ORIGIN_REVISION,
    HEAD_REVISION,
    INDEX_NAME,
    ORIGIN_REVISION,
    PREV_REVISION,
    UNCONDITIONAL_DROP_SQL,
    indexed_columns,
    make_config,
)
from tests.test_helpers.pg_scratch_db import pg_reachable, scratch_database, sqlstates

#: Postgres ``lock_not_available`` — what a ``lock_timeout`` trip raises.
LOCK_NOT_AVAILABLE = "55P03"

#: The revision sets ``lock_timeout = '5s'``. Bounds are deliberately loose:
#: the lower one only has to exclude "failed instantly for another reason",
#: the upper one only has to exclude "waited indefinitely".
LOCK_TIMEOUT_SECONDS = 5.0

pytestmark = [pytest.mark.integration]

skip_no_pg = pytest.mark.skipif(
    not pg_reachable(),
    reason="PostgreSQL not reachable (run `make dev` first)",
)


async def _index_definition(url: str) -> str | None:
    """The ``pg_indexes.indexdef`` for the index under test, or None."""
    engine = create_async_engine(url)
    try:
        async with engine.connect() as conn:
            result = await conn.execute(
                sa.text("SELECT indexdef FROM pg_indexes WHERE tablename = 'documents' AND indexname = :name"),
                {"name": INDEX_NAME},
            )
            return result.scalar()
    finally:
        await engine.dispose()


async def _execute(url: str, statement: str) -> None:
    engine = create_async_engine(url)
    try:
        async with engine.begin() as conn:
            await conn.exec_driver_sql(statement)
    finally:
        await engine.dispose()


async def _replay_unconditional_drop(url: str) -> None:
    """Run 009's literal ``DROP INDEX`` and roll it back.

    Index DDL is transactional on Postgres, so the rollback restores the
    schema exactly even when the statement succeeds.
    """
    engine = create_async_engine(url)
    try:
        async with engine.connect() as conn:
            trans = await conn.begin()
            try:
                await conn.exec_driver_sql(UNCONDITIONAL_DROP_SQL)
            except Exception as exc:  # pragma: no cover - the failure this test exists to catch
                raise AssertionError(
                    f"the unconditional `{UNCONDITIONAL_DROP_SQL}` that {ORIGIN_REVISION}'s "
                    f"downgrade() issues would fail after downgrading past 057: {exc}. "
                    f"Migration 057's downgrade() must recreate {INDEX_NAME}."
                ) from exc
            finally:
                await trans.rollback()
    finally:
        await engine.dispose()


@pytest.fixture(scope="module")
def scratch_db_url() -> Iterator[str]:
    """A throwaway Postgres database at head, shared by the lifecycle tests.

    Module-scoped because bringing a fresh database through the whole chain
    is the expensive part; each test that rewinds restores it to head in a
    ``finally``. That restore is intra-module hygiene only — the database is
    this module's own, so a killed process cannot strand anything shared.
    """
    with scratch_database("mig057_lifecycle") as url:
        command.upgrade(make_config(url), HEAD_REVISION)
        yield url


@skip_no_pg
class TestMigration057IndexLifecycle:
    def test_head_does_not_carry_the_index(self, scratch_db_url: str) -> None:
        assert asyncio.run(_index_definition(scratch_db_url)) is None, (
            f"{INDEX_NAME} survived the upgrade to {HEAD_REVISION}"
        )

    def test_downgrade_restores_the_index_and_re_upgrade_removes_it(self, scratch_db_url: str) -> None:
        cfg = make_config(scratch_db_url)

        try:
            command.downgrade(cfg, PREV_REVISION)
            indexdef = asyncio.run(_index_definition(scratch_db_url))

            assert indexdef is not None, f"downgrade did not restore {INDEX_NAME}"
            # Single column, and that column is created_at. A restore over the
            # wrong column would satisfy a name-only check while covering
            # nothing — and would then be permanently accepted by every later
            # ``IF NOT EXISTS``.
            assert indexed_columns(indexdef) == ["created_at"], indexdef
        finally:
            # Leave the module's database back at head for the sibling tests.
            command.upgrade(cfg, HEAD_REVISION)

        assert asyncio.run(_index_definition(scratch_db_url)) is None, "re-upgrade did not remove the index again"

    def test_post_downgrade_state_satisfies_the_unconditional_drop_at_the_origin(self, scratch_db_url: str) -> None:
        """After downgrading past 057, 009's bare ``DROP INDEX`` still works.

        Migration 009's ``downgrade()`` removes this index with a bare
        ``op.drop_index(...)`` — no ``if_exists`` — so it raises if the index
        is absent. Rather than infer that from the index listing, this
        replays the exact statement against the post-downgrade database
        inside a transaction that is rolled back, so the check is the real
        thing and leaves no trace.
        """
        cfg = make_config(scratch_db_url)

        try:
            command.downgrade(cfg, PREV_REVISION)
            asyncio.run(_replay_unconditional_drop(scratch_db_url))
        finally:
            command.upgrade(cfg, HEAD_REVISION)

    def test_upgrade_succeeds_when_the_index_is_already_absent(self) -> None:
        """``if_exists=True`` on the drop, exercised rather than asserted.

        Two real database shapes reach 057 without this index: one whose
        operator dropped it on migration 054's explicit advice to measure
        rather than assume, and one built by the deprecated ``create_all``
        path, which never creates it because the ORM does not declare it.

        Owns its own database rather than rewinding the module's: it has to
        mutate the schema at an intermediate revision, and leaving that
        mutation visible to a sibling test would couple them.
        """
        with scratch_database("mig057_absent") as url:
            cfg = make_config(url)
            command.upgrade(cfg, PREV_REVISION)
            assert asyncio.run(_index_definition(url)) is not None, (
                f"precondition: {INDEX_NAME} exists at {PREV_REVISION}"
            )
            asyncio.run(_execute(url, f"DROP INDEX {INDEX_NAME}"))

            command.upgrade(cfg, HEAD_REVISION)

            assert asyncio.run(_index_definition(url)) is None

    def test_downgrade_succeeds_when_the_index_is_already_present(self) -> None:
        """``if_not_exists=True`` on the restore, exercised rather than asserted.

        The revision's operator runbook for downgrading a large table is to
        pre-build the index with ``CREATE INDEX CONCURRENTLY`` and let the
        migration's plain ``CREATE INDEX`` fall through as a no-op — which is
        only a safe instruction if the create really tolerates the index
        already being there. The pre-build is spelled non-concurrently here
        because ``CONCURRENTLY`` cannot run inside a transaction and the
        distinction is irrelevant to what this test checks.
        """
        with scratch_database("mig057_present") as url:
            cfg = make_config(url)
            command.upgrade(cfg, HEAD_REVISION)
            asyncio.run(_execute(url, f"CREATE INDEX {INDEX_NAME} ON documents (created_at)"))

            command.downgrade(cfg, PREV_REVISION)

            indexdef = asyncio.run(_index_definition(url))
            assert indexdef is not None
            assert indexed_columns(indexdef) == ["created_at"], indexdef


class _DocumentsLockHolder:
    """Holds ``ACCESS EXCLUSIVE`` on ``documents`` from another connection.

    Runs its own event loop on its own thread, because ``command.upgrade``
    must be called from OUTSIDE a running loop (the bundled ``env.py`` drives
    the migration with ``asyncio.run``). The lock therefore cannot be held by
    the calling thread's loop.
    """

    def __init__(self, url: str) -> None:
        self._url = url
        self._acquired = threading.Event()
        self._release = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        async def hold() -> None:
            engine = create_async_engine(self._url)
            try:
                async with engine.connect() as conn:
                    trans = await conn.begin()
                    try:
                        await conn.exec_driver_sql("LOCK TABLE documents IN ACCESS EXCLUSIVE MODE")
                        self._acquired.set()
                        await asyncio.to_thread(self._release.wait)
                    finally:
                        await trans.rollback()
            finally:
                await engine.dispose()

        try:
            asyncio.run(hold())
        except BaseException as exc:  # noqa: BLE001 - surfaced to the test thread below
            self._error = exc
        finally:
            # Unblock the waiter even if acquisition failed, so a broken
            # holder surfaces as an assertion rather than a hang.
            self._acquired.set()

    def __enter__(self) -> _DocumentsLockHolder:
        self._thread.start()
        assert self._acquired.wait(timeout=30), "lock holder never acquired ACCESS EXCLUSIVE on documents"
        assert self._error is None, f"lock holder failed before acquiring: {self._error!r}"
        return self

    def __exit__(self, *_exc: object) -> None:
        self._release.set()
        self._thread.join(timeout=30)
        # Raised unconditionally, and deliberately even when the body already
        # failed. A holder that dies AFTER setting ``_acquired`` releases the
        # lock early, so the upgrade under test may have succeeded or failed
        # for reasons that have nothing to do with ``lock_timeout`` — the body's
        # own verdict (typically ``DID NOT RAISE``) would point at the
        # migration when the fixture is at fault. Any in-flight exception stays
        # visible on ``__context__``.
        if self._error is not None:
            raise AssertionError(
                f"the lock holder died while it was supposed to be holding ACCESS EXCLUSIVE on "
                f"documents: {self._error!r}. The lock was released early, so this test's verdict "
                f"about lock_timeout is meaningless — fix the fixture before reading the result."
            ) from self._error


def _upgrade_bounded(cfg: Config, revision: str, *, timeout: float) -> BaseException | None:
    """Run ``command.upgrade`` on a worker thread, bounded by *timeout*.

    Returns the exception it raised, or ``None`` if it completed.

    The bound is the whole point, and a plain call cannot provide it. In the
    exact scenario this module exists to catch — ``SET LOCAL lock_timeout``
    not in effect — the ``DROP INDEX`` waits on the conflicting lock forever.
    CI's ``--timeout-method=thread`` resolves that by calling ``os._exit(1)``,
    and the integration job is serial (``-n 0``), so the process dies taking
    with it every later integration test, the pytest summary, the JUnit XML
    and ``coverage-integration.xml``. The build would be red either way, but
    the diagnosis and the rest of the suite's results would not survive.

    Running on a fresh thread also satisfies this module's standing constraint
    that ``command.upgrade`` be called outside a running event loop: a new
    thread has no loop of its own.

    The worker is a daemon and is deliberately not joined on the timeout path.
    It is blocked inside the driver and cannot be interrupted, but it does not
    leak past the test: ``scratch_database.__exit__`` calls ``_drop_database``,
    which issues ``pg_terminate_backend`` for every backend on that datname.
    """
    captured: list[BaseException] = []

    def run() -> None:
        try:
            command.upgrade(cfg, revision)
        except BaseException as exc:  # noqa: BLE001 - handed back to the calling thread
            captured.append(exc)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout=timeout)

    if worker.is_alive():
        pytest.fail(
            f"`SET LOCAL lock_timeout` IS NOT IN EFFECT. The upgrade was still blocked on the "
            f"conflicting ACCESS EXCLUSIVE lock after {timeout:.0f}s, with the revision's timeout "
            f"set to {LOCK_TIMEOUT_SECONDS:.0f}s, so it would wait indefinitely in production "
            f"instead of failing fast. `SET LOCAL` is a silent no-op outside a transaction — check "
            f"that env.py still configures `transaction_per_migration=True`, and that nothing has "
            f"wrapped this revision's body in an `autocommit_block()`, which would take the DDL "
            f"back out of the transaction the setting is scoped to."
        )

    return captured[0] if captured else None


@skip_no_pg
class TestMigration057LockTimeout:
    """The Postgres branch's ``SET LOCAL lock_timeout`` actually takes effect.

    This is the one claim in the revision that fails SILENTLY when wrong.
    ``SET LOCAL`` outside a transaction is a no-op with a warning, so a
    revision that lost its transaction demarcation would keep the statement,
    keep passing every other test in this module, and block indefinitely
    behind a real lock in production instead of failing in 5s.

    Deterministic by construction, with no timing race in the setup: the
    competing transaction takes ``ACCESS EXCLUSIVE`` and never releases it
    until the test is over, so the ``DROP INDEX`` can never win the lock. The
    only question is whether it gives up, and when.

    The precondition is gated elsewhere —
    ``tests/unit/test_migration_bundling.py::test_configures_transaction_per_migration``
    pins ``transaction_per_migration=True``, which is what puts ``SET LOCAL``
    in scope for this revision's DDL. That test would catch the demarcation
    flip directly; this one catches the observable consequence, including the
    case where someone wraps 057's body in an ``autocommit_block()`` and
    voids the timeout without touching env.py.

    **Failure mode if the timeout is NOT in effect:** ``_upgrade_bounded``
    converts it into a named ``pytest.fail`` rather than letting the upgrade
    block. That matters more than it reads — see that helper for why an
    unbounded wait would take the whole integration job down with it.

    **If this fails on the SQLSTATE assertion rather than the raise**, check
    the attribute before concluding the timeout is broken: asyncpg reports
    ``55P03`` on ``.sqlstate`` and leaves ``.pgcode`` as ``None``, and
    ``env.py`` normalizes Postgres URLs to ``postgresql+asyncpg``. The
    ``sqlstates()`` helper reads ``.sqlstate`` across the whole ``orig`` /
    ``__cause__`` / ``__context__`` chain for that reason. A test asserting on
    the wrong attribute fails identically to a timeout that never fired.
    """

    def test_upgrade_gives_up_instead_of_blocking_behind_a_conflicting_lock(self) -> None:
        with scratch_database("mig057_lock") as url:
            cfg = make_config(url)
            command.upgrade(cfg, PREV_REVISION)
            assert asyncio.run(_index_definition(url)) is not None, (
                f"precondition: {INDEX_NAME} must exist at {PREV_REVISION} so the drop has work to do"
            )

            with _DocumentsLockHolder(url):
                started = time.monotonic()
                error = _upgrade_bounded(cfg, HEAD_REVISION, timeout=LOCK_TIMEOUT_SECONDS * 6)
                elapsed = time.monotonic() - started

            assert error is not None, (
                "the upgrade SUCCEEDED while a conflicting ACCESS EXCLUSIVE lock was held, which "
                "should be impossible - the DROP INDEX cannot have run. Suspect the lock holder "
                "released early, or that the revision no longer touches documents."
            )
            assert LOCK_NOT_AVAILABLE in sqlstates(error), (
                f"expected SQLSTATE {LOCK_NOT_AVAILABLE} (lock_not_available) from the "
                f"lock_timeout trip, got {sorted(sqlstates(error))}: {error}"
            )
            # Lower bound only. It excludes "failed instantly for an unrelated
            # reason" — a raise at ~0s would carry the right SQLSTATE only by
            # coincidence, and this is what distinguishes a timeout that fired
            # from one that never armed.
            #
            # There is deliberately no upper bound here. An earlier version
            # asserted `elapsed < LOCK_TIMEOUT_SECONDS * 6` and commented it as
            # excluding "waited far longer than the timeout" — but in the only
            # scenario where the wait is unbounded, control never returns from
            # the upgrade, so that assertion was unreachable by construction.
            # `_upgrade_bounded` enforces the upper bound where it can actually
            # be enforced, and reports it as a named failure.
            assert elapsed > LOCK_TIMEOUT_SECONDS * 0.5, (
                f"the upgrade failed after only {elapsed:.1f}s, well short of the "
                f"{LOCK_TIMEOUT_SECONDS:.0f}s timeout - it likely failed for a reason other than "
                f"the lock_timeout trip, and the SQLSTATE match above is then a coincidence"
            )

            # The revision aborted, so its transaction rolled back and the
            # index is still there. This is what makes the failure retryable
            # rather than a half-applied revision.
            assert asyncio.run(_index_definition(url)) is not None, (
                "the aborted revision left the index dropped - the failure was not atomic"
            )


@skip_no_pg
@pytest.mark.slow
class TestMigration057DowngradeWalk:
    """End-to-end downgrade walk past the revision that created the index.

    Walking dozens of migrations backwards drops real tables, so this owns
    its database outright.

    Marked ``slow`` for the same reason migration 054's equivalent is: one
    test does CREATE DATABASE, the full chain up, a ~48-step walk back down,
    and DROP DATABASE. The default local run excludes it via ``-m "not
    slow"`` in ``pyproject.toml``, while CI's integration selection
    (``-m "integration and not filter_conformance"``) does NOT exclude
    ``slow``, so it still runs where the coverage is wanted. The SQLite lane
    walks the same path cheaply on every run, so nothing is lost locally.
    """

    def test_downgrade_walks_past_the_index_origin_without_error(self) -> None:
        with scratch_database("mig057") as url:
            cfg = make_config(url)
            command.upgrade(cfg, HEAD_REVISION)
            assert asyncio.run(_index_definition(url)) is None, "fresh database did not reach the dropped state"

            try:
                command.downgrade(cfg, BELOW_ORIGIN_REVISION)
            except Exception as exc:
                # Distinguish the failure this test is about from an
                # unrelated, pre-existing broken downgrade elsewhere in the
                # chain. Only the former is evidence against this change.
                if INDEX_NAME in str(exc):
                    raise AssertionError(
                        f"downgrading past {ORIGIN_REVISION} failed, naming {INDEX_NAME}: {exc}. "
                        f"Migration 057's downgrade() must restore {INDEX_NAME}, because "
                        f"{ORIGIN_REVISION}'s downgrade() drops it unconditionally."
                    ) from exc
                # Escape hatch taken. Announce it loudly rather than letting a
                # bare `s` in the progress line read as "fine".
                #
                # CI runs pytest without ``-rs``, so a skip REASON never reaches
                # the log - only the character. A permanently-skipping test that
                # reports green is worse than no test, and this one has in fact
                # been skipping (confirmed in the first green run of this
                # branch, and migration 054's equivalent skips identically, so
                # the cause predates this migration).
                #
                # ``warnings.warn`` is the channel that survives: pytest prints
                # its warnings summary even under ``-q``, which is how CI runs.
                # Verified by experiment, not assumed.
                reason = (
                    f"POSTGRES DOWNGRADE WALK DID NOT ACTUALLY RUN. The chain broke below "
                    f"{HEAD_REVISION} before reaching {ORIGIN_REVISION}, with an error that does "
                    f"not name {INDEX_NAME} - so it is not attributable to this migration and "
                    f"this test cannot make its assertion. What is NOT covered by this run: "
                    f"057's downgrade restore against PostgreSQL. What still covers it: the "
                    f"SQLite lane's walk, which has no escape hatch and fails hard. "
                    f"Underlying error: {type(exc).__name__}: {exc}"
                )
                warnings.warn(reason, stacklevel=2)
                pytest.skip(reason)

            assert asyncio.run(_index_definition(url)) is None, (
                f"{INDEX_NAME} should not exist below {ORIGIN_REVISION}, which is the revision that creates it"
            )
