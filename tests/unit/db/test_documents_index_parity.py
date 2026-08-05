"""The ``documents`` index set at head is exactly what the ORM declares.

The existing drift gate (``tests/test_helpers/schema_drift.py``) is
**one-directional, ORM -> live**: it checks that every index ``Base.metadata``
declares was actually built, and ignores live-only objects by construction.
That direction cannot see an index the chain builds and the ORM does not
declare — which is precisely what ``ix_documents_created_at`` was for 47
revisions, and why it never appeared in any baseline ledger there.

This module closes that blind spot for one table by asserting **set
equality** in both directions. It runs the real chain on a throwaway SQLite
file with no server, so it executes on every CI job and for contributors
without Docker.

Scoped to ``documents`` deliberately
------------------------------------
The equivalent assertion on ``chunks`` fails today: ``chunks`` carries **five**
built-but-undeclared indexes at SQLite head — ``ix_chunks_created_at``,
``ix_chunks_ns_created``, ``ix_chunks_source_ts``,
``ix_chunks_last_accessed_at`` and ``ix_chunks_ns_temporal`` — three of them
from migration 009. Postgres adds more still (029's BRIN, 031's two).

Note the fifth, because a reflection-based audit reports only four: migration
017 builds ``ix_chunks_ns_temporal`` over
``(namespace_id, COALESCE(source_timestamp, created_at))``, and SQLAlchemy's
SQLite dialect skips expression-based indexes during reflection with a
``SAWarning`` rather than an error. It is in ``sqlite_master``; it is not in
``sa.inspect(...).get_indexes(...)``. That is why the live side below reads
``sqlite_master`` directly.

Those are all out of scope here — nothing has measured them as a problem, and
widening this test to cover them would either fail the suite or need an
allowlist, which is the ledger shape this test exists to avoid. If they are
ever reconciled, this module is the place to widen.

What this does NOT cover
------------------------
The SQLite leg only. A ``documents`` index built inside a
``dialect.name == "postgresql"`` branch is invisible here — the chain never
creates it on SQLite, so it cannot show up as extra.

That is not hypothetical, and the reason there is no integration-lane twin is
NOT that such indexes are absent. Two exist today, and neither is
ORM-declared:

* ``ix_documents_ns_session`` (migration 031), and
* ``ix_documents_graph_mirror_pending`` (migration 051)

both created with ``CREATE INDEX CONCURRENTLY`` behind an
``if not _is_postgres(): return`` guard. So a Postgres twin of this test
would **fail today**, for exactly the reason ``chunks`` is excluded above —
it would need the same scoping decision, not merely a different fixture.
Writing it is a separate piece of work with its own judgement call about
which of those two to declare and which to accept.

For completeness: migrations 020 / 021 / 022 also branch on the dialect, but
both arms build the same index — the branch only swaps ``postgresql_where``
for ``sqlite_where`` on a partial-index predicate — so those are genuinely
covered here.

**Names only.** This compares index *names*, not the columns behind them, so
an index rebuilt over the wrong columns under the right name passes here. That
dimension is the sibling gate's: ``schema_drift.collect_drift`` compares
reflected ``column_names`` against the declared columns and reports
``wrong_index_columns``. The two are complementary — that one checks identity
for declared indexes, this one checks the *set* in both directions — and
neither subsumes the other.

Expression-based indexes ARE covered, but only because the live side reads
``sqlite_master``. A reflection-based comparison would silently miss them —
see ``_live_index_names`` for why, and ``ix_chunks_ns_temporal`` for the
shape. If that function is ever "simplified" back to
``sa.inspect(...).get_indexes(...)``, this paragraph becomes false and the
gap reopens with no test failing to announce it.

Reading a failure
-----------------
* Extra live index — a migration built an index nobody declared. Either
  declare it in ``DocumentModel.__table_args__`` or drop it in a migration.
  Do not add an exemption here.
* Missing live index — the ORM declares something the chain does not build.
  That is also caught by the ORM -> live drift gate, which carries a baseline
  ledger; this module has none on purpose.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from khora.db.models import Base
from tests.test_helpers.schema_drift import upgrade

pytestmark = pytest.mark.unit

TABLE = "documents"


def _declared_index_names() -> set[str]:
    """ORM-declared ``documents`` indexes the SQLite chain is expected to build.

    Deliberately NOT ``schema_drift.index_invisible_on_sqlite``, and the
    divergence is the point rather than an oversight. That rule exempts two
    things: ``postgresql_using`` indexes, and expression-based ones. The
    second exemption exists only because SQLAlchemy *reflection* cannot see
    expression indexes — and this module's live side reads ``sqlite_master``,
    which can. Importing the shared rule here would exempt on the declared
    side something the live side reports, so an ORM-declared expression index
    would come back as ``built but not declared`` and tell the reader to
    declare it where it is already declared. Narrowing the rule makes the two
    sides agree by construction instead.

    ``postgresql_using`` stays exempt: those name a Postgres access method,
    and whether the SQLite arm of a migration builds anything under that name
    is a property of the migration, not of the declaration. Note this leaves
    the same latent asymmetry for that case — a ``postgresql_using`` index the
    SQLite chain builds as a plain b-tree (migration 004 does exactly this on
    another table) would report as undeclared. Neither case can fire today:
    ``documents`` declares nine indexes, none ``postgresql_using`` and none
    expression-based. The HNSW / GIN / BRIN indexes usually cited as the
    obstacle to a comparison like this are on ``chunks`` and ``entities``, and
    never on ``documents`` — which is precisely how ``ix_documents_created_at``
    stayed hidden for 47 revisions.
    """
    return {
        index.name
        for index in Base.metadata.tables[TABLE].indexes
        if index.name and not index.dialect_kwargs.get("postgresql_using")
    }


def _live_index_names(db_path: Path) -> set[str]:
    """Every ``documents`` index in the built schema, minus SQLite's implicit ones.

    Read out of ``sqlite_master`` rather than through
    ``sa.inspect(...).get_indexes(...)``, and that choice is load-bearing:
    SQLAlchemy's SQLite dialect **skips expression-based indexes during
    reflection**, emitting a ``SAWarning`` rather than an error. An
    expression index the chain builds and the ORM never declares would
    therefore be invisible to a reflection-based comparison — which is
    exactly the drift class this module exists to catch, and exactly the
    shape of ``chunks``' ``ix_chunks_ns_temporal`` (migration 017, raw
    ``op.execute`` in both dialect arms). ``sqlite_master`` is ground truth
    and has no such blind spot.

    ``sqlite_autoindex_*`` entries are created by SQLite itself to back
    UNIQUE / PRIMARY KEY constraints. They have no ORM index counterpart by
    construction, so including them would make the comparison unsatisfiable.

    One consequence, deliberately accepted: the declared side still exempts
    expression-based indexes (via ``index_invisible_on_sqlite``) because
    reflection could not see them, while this side now can. If a future
    ``documents`` index is declared expression-based in the ORM, the two
    sides will disagree and this test fails. That is the right way round —
    a loud failure prompting someone to revisit the exemption beats the
    silent false pass the reflection-based version gave. No ``documents``
    index is expression-based or ``postgresql_using`` today, so the
    exemption matches nothing here and the two sides are exactly aligned.
    """
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute("SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = ?", (TABLE,)).fetchall()
    finally:
        con.close()
    return {name for (name,) in rows if name and not name.startswith("sqlite_autoindex_")}


def test_documents_index_set_at_head_matches_the_orm_declaration(tmp_path: Path) -> None:
    db_path = tmp_path / "head.db"
    upgrade(f"sqlite:///{db_path}")

    declared = _declared_index_names()
    live = _live_index_names(db_path)

    # Guard the guard: an empty set on either side would make the equality
    # below trivially true if the chain or the declaration silently produced
    # nothing.
    assert declared, "the ORM declares no documents indexes - this test would prove nothing"
    assert live, "the chain built no documents indexes - this test would prove nothing"

    assert live == declared, (
        f"documents index set drifted from the ORM declaration.\n"
        f"  built but not declared: {sorted(live - declared)}\n"
        f"  declared but not built: {sorted(declared - live)}\n"
        f"Declare the extras in DocumentModel.__table_args__ or drop them in a migration; "
        f"write a migration for the missing ones. Do not add an exemption to this test."
    )
