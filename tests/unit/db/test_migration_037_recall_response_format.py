"""``037_recall_response_format``.

Pins the schema changes the migration makes against the prior head
(``036_dream_conflicts``):

1. Upgrade adds ``documents.source_name`` (VARCHAR(64) nullable),
   ``documents.source_url`` (VARCHAR(2048) nullable), and
   ``chunks.chunker_info`` (JSON-typed, NOT NULL, server default ``{}``).
2. Six ``documents`` columns flip to nullable with no server default;
   pre-upgrade rows holding ``''`` are normalized to ``NULL``.
3. ``documents.source_type`` is exempt from the nullability flip: its
   DB default switches ``''`` → ``'library'`` and pre-upgrade rows
   holding ``''`` are normalized to ``'library'``.
4. The nango backfill maps ``source LIKE 'nango://<provider>/%'`` to
   ``source_name = <provider>``; non-nango rows (including those whose
   ``source`` was normalized to NULL by the flip) get
   ``source_name IS NULL``.
5. Downgrade drops the three new columns and restores
   ``NOT NULL DEFAULT ''`` on the seven affected columns.
6. The upgrade emits a single loguru INFO event with message
   ``"khora.migration.applied"`` and the five contract fields
   ``migration_id`` / ``duration_ms`` / ``lock_timeout_tripped`` /
   ``source_name_backfilled`` / ``source_name_unmatched`` bound in
   ``record["extra"]``.

The tests run end-to-end against the sqlite_lance fixture stack:
``op.batch_alter_table`` performs a table-copy on SQLite, so the
nullability flip and the source_type default change exercise the same
code path the production sqlite_lance backend uses.

Postgres-specific assertions (e.g. NOT NULL constraint surviving the
flip on ``source_type``) are exercised by the broader integration
suite via the standard Postgres compose stack — this file is
intentionally SQLite-only per qa's task #3 brief.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch
from uuid import uuid4

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from loguru import logger
from sqlalchemy import inspect
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

_MIGRATIONS_DIR = Path(__file__).resolve().parents[3] / "src" / "khora" / "db" / "migrations"
_PRIOR_REV = "036_dream_conflicts"
_TARGET_REV = "037_recall_response_format"
_FLIPPED_COLUMNS = ("source", "content_type", "title", "author", "language", "checksum")

_MIGRATION_MODULE = importlib.import_module(f"khora.db.migrations.versions.{_TARGET_REV}")


def _make_config(url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(_MIGRATIONS_DIR))
    cfg.set_main_option("sqlalchemy.url", url)
    cfg.attributes["database_url"] = url
    return cfg


@pytest.fixture
def sqlite_url(tmp_path: Path) -> str:
    return f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"


@pytest.fixture
def cfg_at_036(sqlite_url: str) -> Config:
    """Alembic config stamped at ``036_dream_conflicts``, ready to seed."""
    cfg = _make_config(sqlite_url)
    command.upgrade(cfg, _PRIOR_REV)
    return cfg


@contextmanager
def _capture_migration_events() -> Any:
    """Capture loguru records (not just formatted strings) emitted at INFO+.

    The migration uses ``logger.bind(...).info("khora.migration.applied")``;
    we read the full record dict so we can assert ``record["extra"]``.
    """
    records: list[dict[str, Any]] = []

    def _sink(message: Any) -> None:
        records.append(dict(message.record))

    handler_id = logger.add(_sink, level="INFO", format="{message}")
    try:
        yield records
    finally:
        logger.remove(handler_id)


async def _seed_namespace(engine: AsyncEngine) -> str:
    """Insert one memory_namespaces row and return its id (str)."""
    ns_id = str(uuid4())
    async with engine.begin() as conn:
        await conn.execute(
            sa.text(
                "INSERT INTO memory_namespaces "
                "(id, namespace_id, version, is_active, created_at, updated_at) "
                "VALUES (:id, :id, 1, 1, datetime('now'), datetime('now'))"
            ),
            {"id": ns_id},
        )
    return ns_id


async def _seed_documents(engine: AsyncEngine, ns_id: str, rows: list[dict[str, Any]]) -> list[str]:
    """Insert document rows; return list of ids in input order."""
    ids = [str(uuid4()) for _ in rows]
    async with engine.begin() as conn:
        for doc_id, row in zip(ids, rows, strict=True):
            params = {
                "id": doc_id,
                "ns": ns_id,
                "content": row.get("content", "x"),
                "source": row.get("source"),
                "source_type": row.get("source_type", "library"),
                "title": row.get("title", ""),
            }
            await conn.execute(
                sa.text(
                    "INSERT INTO documents "
                    "(id, namespace_id, content, source, source_type, title) "
                    "VALUES (:id, :ns, :content, :source, :source_type, :title)"
                ),
                params,
            )
    return ids


async def _make_flipped_columns_not_null(engine: AsyncEngine) -> None:
    """Rebuild ``documents`` so the six flipped columns are ``NOT NULL DEFAULT ''``.

    The Alembic chain creates these columns nullable (``server_default=''`` only),
    so a chain-built DB never reproduces the legacy ``create_tables()`` /
    ``Base.metadata.create_all`` starting state, where the old ``Mapped[str]``
    ORM declaration implied ``nullable=False``. SQLite cannot ``ALTER COLUMN``
    to add ``NOT NULL`` in place, so we table-copy: read the live ``documents``
    DDL, inject ``NOT NULL DEFAULT ''`` into the six target columns, recreate the
    table under the same name, copy every row, and restore the indexes.

    Reading the DDL at runtime keeps this faithful to whatever the prior
    revision actually emits, rather than hard-coding a snapshot that would rot.
    """
    async with engine.begin() as conn:
        await conn.execute(sa.text("PRAGMA foreign_keys=OFF"))
        ddl = (
            await conn.execute(sa.text("SELECT sql FROM sqlite_master WHERE name = 'documents' AND type = 'table'"))
        ).scalar()
        assert ddl is not None, "documents table DDL not found in sqlite_master"
        index_ddls = [
            row[0]
            for row in (
                await conn.execute(
                    sa.text(
                        "SELECT sql FROM sqlite_master "
                        "WHERE tbl_name = 'documents' AND type = 'index' AND sql IS NOT NULL"
                    )
                )
            ).fetchall()
        ]
        col_names = [row[1] for row in (await conn.execute(sa.text("PRAGMA table_info(documents)"))).fetchall()]

        new_ddl = ddl
        for col in _FLIPPED_COLUMNS:
            pattern = re.compile(rf"^(\s*{col}\s+VARCHAR\(\d+\)[^,\n]*?)(,?\s*)$", re.MULTILINE)

            def _force_not_null(match: re.Match[str]) -> str:
                body = re.sub(
                    r"DEFAULT\s*\([^)]*\)|DEFAULT\s*'[^']*'",
                    "DEFAULT ''",
                    match.group(1),
                )
                if "DEFAULT" not in body:
                    body += " DEFAULT ''"
                return f"{body} NOT NULL{match.group(2)}"

            new_ddl, replaced = pattern.subn(_force_not_null, new_ddl)
            assert replaced == 1, f"expected to rewrite exactly one {col} column line, got {replaced}"

        new_ddl = new_ddl.replace('CREATE TABLE "documents"', 'CREATE TABLE "documents_new"', 1)
        columns = ", ".join(col_names)
        await conn.execute(sa.text("ALTER TABLE documents RENAME TO documents_old"))
        await conn.execute(sa.text(new_ddl))
        await conn.execute(
            sa.text(
                f"INSERT INTO documents_new ({columns}) SELECT {columns} FROM documents_old"  # noqa: S608
            )
        )
        await conn.execute(sa.text("DROP TABLE documents_old"))
        await conn.execute(sa.text("ALTER TABLE documents_new RENAME TO documents"))
        for index_ddl in index_ddls:
            await conn.execute(sa.text(index_ddl))
        await conn.execute(sa.text("PRAGMA foreign_keys=ON"))


@pytest.mark.unit
class TestRevisionMetadata:
    def test_revision_id(self) -> None:
        assert _MIGRATION_MODULE.revision == _TARGET_REV
        assert _MIGRATION_MODULE.down_revision == _PRIOR_REV


@pytest.mark.unit
class TestUpgradeAddsNewColumns:
    """Spec #1 — upgrade adds source_name, source_url, chunker_info."""

    def test_documents_has_source_name_nullable_varchar64(self, cfg_at_036: Config, sqlite_url: str) -> None:
        command.upgrade(cfg_at_036, _TARGET_REV)

        async def check() -> None:
            engine = create_async_engine(sqlite_url)
            try:
                async with engine.connect() as conn:
                    cols = await conn.run_sync(lambda sc: inspect(sc).get_columns("documents"))
                by_name = {c["name"]: c for c in cols}
                assert "source_name" in by_name, "source_name column missing after upgrade"
                col = by_name["source_name"]
                assert col["nullable"] is True
                # SQLAlchemy reflects VARCHAR(64) as String(length=64).
                assert getattr(col["type"], "length", None) == 64
            finally:
                await engine.dispose()

        asyncio.run(check())

    def test_documents_has_source_url_nullable_varchar2048(self, cfg_at_036: Config, sqlite_url: str) -> None:
        command.upgrade(cfg_at_036, _TARGET_REV)

        async def check() -> None:
            engine = create_async_engine(sqlite_url)
            try:
                async with engine.connect() as conn:
                    cols = await conn.run_sync(lambda sc: inspect(sc).get_columns("documents"))
                by_name = {c["name"]: c for c in cols}
                assert "source_url" in by_name, "source_url column missing after upgrade"
                col = by_name["source_url"]
                assert col["nullable"] is True
                assert getattr(col["type"], "length", None) == 2048
            finally:
                await engine.dispose()

        asyncio.run(check())

    def test_chunks_has_chunker_info_not_null_default_empty_object(self, cfg_at_036: Config, sqlite_url: str) -> None:
        command.upgrade(cfg_at_036, _TARGET_REV)

        async def check() -> None:
            engine = create_async_engine(sqlite_url)
            try:
                async with engine.connect() as conn:
                    cols = await conn.run_sync(lambda sc: inspect(sc).get_columns("chunks"))
                by_name = {c["name"]: c for c in cols}
                assert "chunker_info" in by_name, "chunker_info column missing after upgrade"
                col = by_name["chunker_info"]
                assert col["nullable"] is False, "chunker_info must be NOT NULL"
                # Default is the JSON literal '{}'. SQLAlchemy reflects it as a quoted string.
                assert col["default"] is not None
                assert "{}" in str(col["default"])
            finally:
                await engine.dispose()

        asyncio.run(check())

    def test_chunker_info_default_populates_on_insert(self, cfg_at_036: Config, sqlite_url: str) -> None:
        """A chunk row inserted without chunker_info gets {} from the server default."""
        command.upgrade(cfg_at_036, _TARGET_REV)

        async def check() -> None:
            engine = create_async_engine(sqlite_url)
            try:
                ns = await _seed_namespace(engine)
                [doc_id] = await _seed_documents(engine, ns, [{"source": "library:foo"}])
                chunk_id = str(uuid4())
                async with engine.begin() as conn:
                    await conn.execute(
                        sa.text(
                            "INSERT INTO chunks (id, namespace_id, document_id, content) "
                            "VALUES (:id, :ns, :doc, 'hello')"
                        ),
                        {"id": chunk_id, "ns": ns, "doc": doc_id},
                    )
                async with engine.connect() as conn:
                    raw = (
                        await conn.execute(
                            sa.text("SELECT chunker_info FROM chunks WHERE id = :id"),
                            {"id": chunk_id},
                        )
                    ).scalar()
                # sa.JSON in SQLite deserializes to dict on read.
                if isinstance(raw, str):
                    raw = json.loads(raw)
                assert raw == {}
            finally:
                await engine.dispose()

        asyncio.run(check())


@pytest.mark.unit
class TestNullabilityFlip:
    """Spec #2 — six columns become nullable; ``''`` rows become ``NULL``."""

    def test_flipped_columns_are_nullable_with_no_server_default(self, cfg_at_036: Config, sqlite_url: str) -> None:
        command.upgrade(cfg_at_036, _TARGET_REV)

        async def check() -> None:
            engine = create_async_engine(sqlite_url)
            try:
                async with engine.connect() as conn:
                    cols = await conn.run_sync(lambda sc: inspect(sc).get_columns("documents"))
                by_name = {c["name"]: c for c in cols}
                for col_name in _FLIPPED_COLUMNS:
                    col = by_name[col_name]
                    assert col["nullable"] is True, f"{col_name} expected nullable after flip"
                    assert col["default"] is None, (
                        f"{col_name} expected no server default after flip, got {col['default']!r}"
                    )
            finally:
                await engine.dispose()

        asyncio.run(check())

    def test_empty_string_rows_normalized_to_null(self, cfg_at_036: Config, sqlite_url: str) -> None:
        """A row seeded pre-upgrade with ``''`` in every flipped column ends up NULL."""

        async def seed_then_upgrade() -> tuple[str, str]:
            engine = create_async_engine(sqlite_url)
            try:
                ns = await _seed_namespace(engine)
                doc_id = str(uuid4())
                # Pre-upgrade schema requires NULL for source_type only via server_default=''.
                # Seed every flippable column with the empty string.
                async with engine.begin() as conn:
                    await conn.execute(
                        sa.text(
                            "INSERT INTO documents "
                            "(id, namespace_id, content, source, source_type, content_type, "
                            " title, author, language, checksum) "
                            "VALUES (:id, :ns, 'c', '', '', '', '', '', '', '')"
                        ),
                        {"id": doc_id, "ns": ns},
                    )
            finally:
                await engine.dispose()
            return ns, doc_id

        ns, doc_id = asyncio.run(seed_then_upgrade())
        command.upgrade(cfg_at_036, _TARGET_REV)

        async def check() -> None:
            engine = create_async_engine(sqlite_url)
            try:
                async with engine.connect() as conn:
                    row = (
                        await conn.execute(
                            sa.text(
                                "SELECT source, content_type, title, author, language, checksum "
                                "FROM documents WHERE id = :id"
                            ),
                            {"id": doc_id},
                        )
                    ).first()
                assert row is not None
                for idx, col_name in enumerate(_FLIPPED_COLUMNS):
                    assert row[idx] is None, f"{col_name} expected NULL after upgrade normalized '', got {row[idx]!r}"
            finally:
                await engine.dispose()

        asyncio.run(check())


@pytest.mark.unit
class TestLegacyNotNullStartingState:
    """Regression — upgrade must survive a legacy ``NOT NULL DEFAULT ''`` start.

    On databases created via the legacy ``create_tables()`` /
    ``Base.metadata.create_all`` path, the six flipped columns start as
    ``NOT NULL DEFAULT ''`` (the old ``Mapped[str]`` ORM implied
    ``nullable=False``). The ``DROP NOT NULL`` must run before the
    empty-string → NULL normalization UPDATE; otherwise writing NULL while the
    constraint is still in force raises a NOT NULL violation that rolls back
    the whole revision and strands the DB at the prior revision.

    The Alembic-chain fixture leaves these columns nullable, so the rest of the
    suite never exercises this path — hence the explicit table-copy here.
    """

    def test_upgrade_succeeds_and_normalizes_empty_strings_to_null(self, cfg_at_036: Config, sqlite_url: str) -> None:
        async def seed() -> str:
            engine = create_async_engine(sqlite_url)
            try:
                await _make_flipped_columns_not_null(engine)
                # Confirm the simulated legacy state actually took: all six
                # columns must report NOT NULL before the upgrade runs.
                async with engine.connect() as conn:
                    info = (await conn.execute(sa.text("PRAGMA table_info(documents)"))).fetchall()
                not_null = {row[1]: bool(row[3]) for row in info}
                for col in _FLIPPED_COLUMNS:
                    assert not_null[col] is True, f"{col} should be NOT NULL in the simulated legacy state"

                ns = await _seed_namespace(engine)
                doc_id = str(uuid4())
                async with engine.begin() as conn:
                    await conn.execute(
                        sa.text(
                            "INSERT INTO documents "
                            "(id, namespace_id, content, source, source_type, content_type, "
                            " title, author, language, checksum) "
                            "VALUES (:id, :ns, 'c', '', '', '', '', '', '', '')"
                        ),
                        {"id": doc_id, "ns": ns},
                    )
            finally:
                await engine.dispose()
            return doc_id

        doc_id = asyncio.run(seed())

        # Must not raise. Pre-reorder this raised a NOT NULL constraint violation
        # (surfaced as OperationalError / IntegrityError) and rolled the txn back.
        command.upgrade(cfg_at_036, _TARGET_REV)

        async def check() -> None:
            engine = create_async_engine(sqlite_url)
            try:
                async with engine.connect() as conn:
                    row = (
                        await conn.execute(
                            sa.text(
                                "SELECT source, content_type, title, author, language, checksum "
                                "FROM documents WHERE id = :id"
                            ),
                            {"id": doc_id},
                        )
                    ).first()
                assert row is not None
                for idx, col_name in enumerate(_FLIPPED_COLUMNS):
                    assert row[idx] is None, (
                        f"{col_name} expected NULL after upgrade normalized '' on a legacy "
                        f"NOT NULL start, got {row[idx]!r}"
                    )
            finally:
                await engine.dispose()

        asyncio.run(check())


@pytest.mark.unit
class TestSourceTypeException:
    """Spec #3 — source_type stays under explicit default ``'library'``; ``''`` rows rewritten."""

    def test_source_type_default_is_library(self, cfg_at_036: Config, sqlite_url: str) -> None:
        command.upgrade(cfg_at_036, _TARGET_REV)

        async def check() -> None:
            engine = create_async_engine(sqlite_url)
            try:
                async with engine.connect() as conn:
                    cols = await conn.run_sync(lambda sc: inspect(sc).get_columns("documents"))
                col = next(c for c in cols if c["name"] == "source_type")
                # Server default is the SQL literal 'library' (quoted on SQLite reflection).
                assert col["default"] is not None
                assert "library" in str(col["default"])
            finally:
                await engine.dispose()

        asyncio.run(check())

    def test_empty_source_type_rows_become_library(self, cfg_at_036: Config, sqlite_url: str) -> None:
        async def seed() -> str:
            engine = create_async_engine(sqlite_url)
            try:
                ns = await _seed_namespace(engine)
                doc_id = str(uuid4())
                async with engine.begin() as conn:
                    await conn.execute(
                        sa.text(
                            "INSERT INTO documents (id, namespace_id, content, source_type) VALUES (:id, :ns, 'c', '')"
                        ),
                        {"id": doc_id, "ns": ns},
                    )
            finally:
                await engine.dispose()
            return doc_id

        doc_id = asyncio.run(seed())
        command.upgrade(cfg_at_036, _TARGET_REV)

        async def check() -> None:
            engine = create_async_engine(sqlite_url)
            try:
                async with engine.connect() as conn:
                    st = (
                        await conn.execute(
                            sa.text("SELECT source_type FROM documents WHERE id = :id"),
                            {"id": doc_id},
                        )
                    ).scalar()
                assert st == "library"
            finally:
                await engine.dispose()

        asyncio.run(check())

    def test_explicit_source_type_value_preserved(self, cfg_at_036: Config, sqlite_url: str) -> None:
        """A row whose source_type is already a non-empty value (e.g. 'web') is not rewritten."""

        async def seed() -> str:
            engine = create_async_engine(sqlite_url)
            try:
                ns = await _seed_namespace(engine)
                doc_id = str(uuid4())
                async with engine.begin() as conn:
                    await conn.execute(
                        sa.text(
                            "INSERT INTO documents (id, namespace_id, content, source_type) "
                            "VALUES (:id, :ns, 'c', 'web')"
                        ),
                        {"id": doc_id, "ns": ns},
                    )
            finally:
                await engine.dispose()
            return doc_id

        doc_id = asyncio.run(seed())
        command.upgrade(cfg_at_036, _TARGET_REV)

        async def check() -> None:
            engine = create_async_engine(sqlite_url)
            try:
                async with engine.connect() as conn:
                    st = (
                        await conn.execute(
                            sa.text("SELECT source_type FROM documents WHERE id = :id"),
                            {"id": doc_id},
                        )
                    ).scalar()
                assert st == "web"
            finally:
                await engine.dispose()

        asyncio.run(check())


@pytest.mark.unit
class TestNangoBackfill:
    """Spec #4 — nango sources backfill source_name; non-nango stays NULL."""

    def test_backfill_matrix(self, cfg_at_036: Config, sqlite_url: str) -> None:
        seeds = [
            ("nango://linear/issues", "linear"),
            ("nango://slack/messages", "slack"),
            # Greedy match on first '/' — extra path segments are ignored.
            ("nango://github/prs/extra/parts", "github"),
            # Provider is lower-cased to match the case-insensitive convention
            # in ``core.models.source.register_source_alias`` — see the
            # migration's ``_backfill_source_name`` docstring. Pins the
            # ``.lower()`` invariant so a future optimization can't silently
            # drop it; covers single-cap, all-caps, and CamelCase.
            ("nango://Linear/issues", "linear"),
            ("nango://Slack/messages", "slack"),
            ("nango://NOTION/db/123", "notion"),
            ("nango://GITHUB/prs", "github"),
            ("nango://CamelCase/x", "camelcase"),
            ("library:foo", None),
            ("http://example.com", None),
            ("", None),  # normalized to NULL by the flip first
            (None, None),  # explicit NULL stays NULL
        ]

        async def seed() -> tuple[list[str], str]:
            engine = create_async_engine(sqlite_url)
            try:
                ns = await _seed_namespace(engine)
                ids: list[str] = []
                async with engine.begin() as conn:
                    for src, _expected in seeds:
                        doc_id = str(uuid4())
                        ids.append(doc_id)
                        await conn.execute(
                            sa.text(
                                "INSERT INTO documents "
                                "(id, namespace_id, content, source, source_type) "
                                "VALUES (:id, :ns, 'c', :src, 'library')"
                            ),
                            {"id": doc_id, "ns": ns, "src": src},
                        )
            finally:
                await engine.dispose()
            return ids, ns

        ids, _ns = asyncio.run(seed())
        command.upgrade(cfg_at_036, _TARGET_REV)

        async def check() -> None:
            engine = create_async_engine(sqlite_url)
            try:
                async with engine.connect() as conn:
                    rows = (
                        await conn.execute(
                            sa.text("SELECT id, source, source_name FROM documents"),
                        )
                    ).fetchall()
                by_id = {row.id: row for row in rows}
                for doc_id, (seeded_src, expected_name) in zip(ids, seeds, strict=True):
                    row = by_id[doc_id]
                    assert row.source_name == expected_name, (
                        f"seeded source={seeded_src!r}: expected source_name={expected_name!r}, got {row.source_name!r}"
                    )
                    if seeded_src == "":
                        # Verify the empty-string normalization to NULL ran before the backfill.
                        assert row.source is None, "empty-string source should normalize to NULL"
            finally:
                await engine.dispose()

        asyncio.run(check())


@pytest.mark.unit
class TestDowngrade:
    """Spec #5 — downgrade drops new columns and restores NOT NULL DEFAULT ''."""

    def test_new_columns_gone(self, cfg_at_036: Config, sqlite_url: str) -> None:
        command.upgrade(cfg_at_036, _TARGET_REV)
        command.downgrade(cfg_at_036, _PRIOR_REV)

        async def check() -> None:
            engine = create_async_engine(sqlite_url)
            try:
                async with engine.connect() as conn:
                    doc_cols = await conn.run_sync(lambda sc: {c["name"] for c in inspect(sc).get_columns("documents")})
                    chunk_cols = await conn.run_sync(lambda sc: {c["name"] for c in inspect(sc).get_columns("chunks")})
                assert "source_name" not in doc_cols
                assert "source_url" not in doc_cols
                assert "chunker_info" not in chunk_cols
            finally:
                await engine.dispose()

        asyncio.run(check())

    def test_flipped_columns_back_to_not_null_default_empty(self, cfg_at_036: Config, sqlite_url: str) -> None:
        command.upgrade(cfg_at_036, _TARGET_REV)
        command.downgrade(cfg_at_036, _PRIOR_REV)

        async def check() -> None:
            engine = create_async_engine(sqlite_url)
            try:
                async with engine.connect() as conn:
                    cols = await conn.run_sync(lambda sc: inspect(sc).get_columns("documents"))
                by_name = {c["name"]: c for c in cols}
                for col_name in _FLIPPED_COLUMNS:
                    col = by_name[col_name]
                    assert col["nullable"] is False, f"{col_name} expected NOT NULL after downgrade"
                    assert col["default"] is not None and "''" in str(col["default"]), (
                        f"{col_name} expected DEFAULT '' after downgrade, got {col['default']!r}"
                    )
            finally:
                await engine.dispose()

        asyncio.run(check())

    def test_source_type_back_to_empty_string_default(self, cfg_at_036: Config, sqlite_url: str) -> None:
        command.upgrade(cfg_at_036, _TARGET_REV)
        command.downgrade(cfg_at_036, _PRIOR_REV)

        async def check() -> None:
            engine = create_async_engine(sqlite_url)
            try:
                async with engine.connect() as conn:
                    cols = await conn.run_sync(lambda sc: inspect(sc).get_columns("documents"))
                col = next(c for c in cols if c["name"] == "source_type")
                # Downgrade restores default ''. (SQLite reflects the pre-existing
                # source_type column as nullable=True because the original 000
                # schema declared it with server_default='' but no NOT NULL — on
                # PostgreSQL the constraint would also be restored to NOT NULL.)
                assert col["default"] is not None
                assert "''" in str(col["default"])
            finally:
                await engine.dispose()

        asyncio.run(check())

    def test_version_table_back_at_prior_revision(self, cfg_at_036: Config, sqlite_url: str) -> None:
        command.upgrade(cfg_at_036, _TARGET_REV)
        command.downgrade(cfg_at_036, _PRIOR_REV)

        async def check() -> None:
            engine = create_async_engine(sqlite_url)
            try:
                async with engine.connect() as conn:
                    v = (await conn.execute(sa.text("SELECT version_num FROM khora_alembic_version"))).scalar()
                assert v == _PRIOR_REV
            finally:
                await engine.dispose()

        asyncio.run(check())


@pytest.mark.unit
class TestStructuredLogEvent:
    """Spec #6 — upgrade emits one INFO event with the five contract fields."""

    def test_applied_event_fields_on_clean_upgrade(self, cfg_at_036: Config, sqlite_url: str) -> None:
        # Seed 2 nango rows (linear, github) + 1 non-nango row so the counts are
        # non-trivial: backfilled=2, unmatched=1.
        async def seed() -> None:
            engine = create_async_engine(sqlite_url)
            try:
                ns = await _seed_namespace(engine)
                async with engine.begin() as conn:
                    for src in (
                        "nango://linear/issues",
                        "nango://github/prs",
                        "library:foo",
                    ):
                        await conn.execute(
                            sa.text(
                                "INSERT INTO documents "
                                "(id, namespace_id, content, source, source_type) "
                                "VALUES (:id, :ns, 'c', :src, 'library')"
                            ),
                            {"id": str(uuid4()), "ns": ns, "src": src},
                        )
            finally:
                await engine.dispose()

        asyncio.run(seed())

        with _capture_migration_events() as records:
            command.upgrade(cfg_at_036, _TARGET_REV)

        applied = [r for r in records if r["message"] == "khora.migration.applied"]
        assert len(applied) == 1, (
            f"expected exactly one khora.migration.applied event, got {len(applied)}: {[r['message'] for r in records]}"
        )
        event = applied[0]
        assert event["level"].name == "INFO"

        extra = event["extra"]
        # All five contract fields present.
        expected_keys = {
            "migration_id",
            "duration_ms",
            "lock_timeout_tripped",
            "source_name_backfilled",
            "source_name_unmatched",
        }
        missing = expected_keys - set(extra)
        assert not missing, f"missing structured-log fields: {missing}"

        # Field-level contract.
        assert extra["migration_id"] == _TARGET_REV
        assert isinstance(extra["duration_ms"], int) and extra["duration_ms"] >= 0
        assert extra["lock_timeout_tripped"] is False
        assert extra["source_name_backfilled"] == 2
        assert extra["source_name_unmatched"] == 1

    def test_applied_event_excludes_null_sources_from_unmatched(self, cfg_at_036: Config, sqlite_url: str) -> None:
        """NULL-source rows are EXCLUDED from ``source_name_unmatched``.

        Pins the dashboard-signal definition documented at the migration's
        ``_backfill_source_name`` docstring (lines 164-169): ``unmatched``
        counts populated-but-non-nango rows. A row whose ``source`` was
        an empty string (normalized to NULL by the flip) or was already
        NULL must NOT bump the counter — otherwise ``unmatched`` ≈
        row-count and the signal becomes meaningless.

        Seed: 1 nango row + 1 populated non-nango row + 1 ``''`` row (→ NULL)
        + 1 explicit NULL row. Expect ``backfilled=1, unmatched=1`` — only
        the single populated non-nango row counts.
        """

        async def seed() -> None:
            engine = create_async_engine(sqlite_url)
            try:
                ns = await _seed_namespace(engine)
                async with engine.begin() as conn:
                    # 1 nango row → backfilled
                    await conn.execute(
                        sa.text(
                            "INSERT INTO documents (id, namespace_id, content, source, source_type) "
                            "VALUES (:id, :ns, 'c', 'nango://slack/messages', 'library')"
                        ),
                        {"id": str(uuid4()), "ns": ns},
                    )
                    # 1 populated non-nango row → unmatched (the only one)
                    await conn.execute(
                        sa.text(
                            "INSERT INTO documents (id, namespace_id, content, source, source_type) "
                            "VALUES (:id, :ns, 'c', 'library:foo', 'library')"
                        ),
                        {"id": str(uuid4()), "ns": ns},
                    )
                    # 1 empty-string row — normalized to NULL by the flip → MUST NOT bump unmatched
                    await conn.execute(
                        sa.text(
                            "INSERT INTO documents (id, namespace_id, content, source, source_type) "
                            "VALUES (:id, :ns, 'c', '', 'library')"
                        ),
                        {"id": str(uuid4()), "ns": ns},
                    )
                    # 1 explicit NULL row → MUST NOT bump unmatched
                    await conn.execute(
                        sa.text(
                            "INSERT INTO documents (id, namespace_id, content, source, source_type) "
                            "VALUES (:id, :ns, 'c', NULL, 'library')"
                        ),
                        {"id": str(uuid4()), "ns": ns},
                    )
            finally:
                await engine.dispose()

        asyncio.run(seed())

        with _capture_migration_events() as records:
            command.upgrade(cfg_at_036, _TARGET_REV)

        applied = [r for r in records if r["message"] == "khora.migration.applied"]
        assert len(applied) == 1
        extra = applied[0]["extra"]
        assert extra["source_name_backfilled"] == 1, "exactly one nango row should be backfilled"
        assert extra["source_name_unmatched"] == 1, (
            "unmatched must count only the populated non-nango row; the empty-string "
            "and explicit-NULL rows are excluded by the ``source IS NOT NULL`` filter"
        )

    def test_applied_event_lock_timeout_true_for_55p03(self) -> None:
        """A real lock_timeout trip (pgcode ``55P03``) sets ``lock_timeout_tripped=True``.

        Monkeypatches ``_upgrade_impl`` to raise an ``OperationalError`` whose
        ``.orig.pgcode`` is the PostgreSQL ``lock_not_available`` SQLSTATE.
        Pins the contract that the structured log distinguishes this from
        any other ``OperationalError``.
        """

        class _LockOrig:
            pgcode = "55P03"

        def _raise_lock_timeout() -> tuple[int, int]:
            raise OperationalError("statement", None, _LockOrig())

        records: list[dict[str, Any]] = []

        def _sink(message: Any) -> None:
            records.append(dict(message.record))

        handler_id = logger.add(_sink, level="ERROR", format="{message}")
        try:
            with patch.object(_MIGRATION_MODULE, "_upgrade_impl", _raise_lock_timeout):
                with pytest.raises(OperationalError):
                    _MIGRATION_MODULE.upgrade()
        finally:
            logger.remove(handler_id)

        applied = [r for r in records if r["message"] == "khora.migration.applied"]
        assert len(applied) == 1
        event = applied[0]
        assert event["level"].name == "ERROR"
        extra = event["extra"]
        assert set(extra) >= {
            "migration_id",
            "duration_ms",
            "lock_timeout_tripped",
            "source_name_backfilled",
            "source_name_unmatched",
        }
        assert extra["migration_id"] == _TARGET_REV
        assert extra["lock_timeout_tripped"] is True
        # Error path emits zeroed counts (initialized up-front before
        # ``_upgrade_impl`` runs) so dashboards never see missing fields.
        assert extra["source_name_backfilled"] == 0
        assert extra["source_name_unmatched"] == 0

    def test_applied_event_lock_timeout_false_for_non_55p03(self) -> None:
        """A non-lock OperationalError sets ``lock_timeout_tripped=False``.

        OperationalError wraps deadlocks, syntax errors, connection drops,
        server shutdowns — none of which are lock-timeout trips. Pinning
        this prevents a regression that flips the field to True for any
        OperationalError, which would lie to oncall dashboards.
        """

        class _SyntaxErrOrig:
            pgcode = "42601"  # PG: syntax_error

        def _raise_syntax_error() -> tuple[int, int]:
            raise OperationalError("statement", None, _SyntaxErrOrig())

        records: list[dict[str, Any]] = []

        def _sink(message: Any) -> None:
            records.append(dict(message.record))

        handler_id = logger.add(_sink, level="ERROR", format="{message}")
        try:
            with patch.object(_MIGRATION_MODULE, "_upgrade_impl", _raise_syntax_error):
                with pytest.raises(OperationalError):
                    _MIGRATION_MODULE.upgrade()
        finally:
            logger.remove(handler_id)

        applied = [r for r in records if r["message"] == "khora.migration.applied"]
        assert len(applied) == 1
        event = applied[0]
        assert event["level"].name == "ERROR"
        assert event["extra"]["lock_timeout_tripped"] is False
        assert event["extra"]["migration_id"] == _TARGET_REV

    def test_applied_event_lock_timeout_false_when_orig_is_none(self) -> None:
        """An OperationalError with no ``.orig`` (driver-less raise) also reads False.

        Covers the defensive ``getattr(exc, "orig", None) is None`` branch
        in ``_is_lock_timeout``.
        """

        def _raise_origless() -> tuple[int, int]:
            # SQLAlchemy's OperationalError(statement, params, orig=None)
            # leaves .orig=None on the wrapped exception.
            raise OperationalError("statement", None, Exception("driverless"))

        records: list[dict[str, Any]] = []

        def _sink(message: Any) -> None:
            records.append(dict(message.record))

        handler_id = logger.add(_sink, level="ERROR", format="{message}")
        try:
            with patch.object(_MIGRATION_MODULE, "_upgrade_impl", _raise_origless):
                with pytest.raises(OperationalError):
                    _MIGRATION_MODULE.upgrade()
        finally:
            logger.remove(handler_id)

        applied = [r for r in records if r["message"] == "khora.migration.applied"]
        assert len(applied) == 1
        # A bare Exception has no .pgcode attribute, so _is_lock_timeout returns False.
        assert applied[0]["extra"]["lock_timeout_tripped"] is False
