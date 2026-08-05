"""Reflect the live schema and diff it against the ORM declaration.

Used by the two drift gates: ``tests/unit/test_migration_drift.py``
(SQLite leg) and ``tests/integration/db/test_schema_drift_pg.py``
(PostgreSQL leg). Both run the full Alembic chain to head, reflect what
was actually built, and compare it against ``Base.metadata``.

Why reflection and not ``alembic.autogenerate.compare_metadata``
---------------------------------------------------------------
``compare_metadata`` renders every column type through the bound
dialect's compiler *before* it produces any diff entries. ``Base.metadata``
carries ``JSONB`` columns, so on the SQLite leg it raises
``UnsupportedCompilationError: Compiler <SQLiteTypeCompiler> can't render
element of type JSONB`` and never returns a diff to filter. The failure is
in the render step, so it cannot be allowlisted away. The SQLite leg has to
work without a database server (it is a unit test and gates every PR), so
the gate reflects the built schema and diffs the dimensions we care about
explicitly.

Gate direction: ORM subset-of live
----------------------------------
Only objects declared in the ORM are checked for presence in the live
schema. Live-only objects are ignored by construction — that is why the
Postgres-only migrations that build indexes with no ORM counterpart need
no allowlist entry.

What this gate does NOT check
-----------------------------
A green run means "every ORM table, column and index exists, every index
covers the declared columns in the declared left-to-right order, and every
declared NOT NULL is installed". It does **not** mean the schemas match.
Known gaps, each of which can hide a real defect:

* **Index sort direction.** The comparison uses ``index.columns``, which
  flattens ``mention_count.desc()`` to ``mention_count``. A ``DESC`` index
  rebuilt ``ASC`` passes. Read "declared order" above as the order of the
  columns, not their sort order.
* **Index uniqueness.** An index that stopped enforcing ``unique=True``
  passes — a data-integrity drift, not a cosmetic one.
* **Partial-index predicates.** ``postgresql_where`` is not compared, so
  an index whose WHERE clause drifted still passes.
* **Column types.** ``String(64)`` vs ``String(512)``, ``JSONB`` vs
  ``JSON``. Not hypothetical here: migration 042 was a column widening.
* **Server defaults.** A wrong ``server_default`` is the exact mechanism
  behind the empty-string ``source_type`` rows migration 037 had to
  rewrite.
* **Foreign keys and CHECK constraints.** Neither is inspected at all.

Adding a dimension is a few lines in ``collect_drift`` plus its baseline
ledger; do it when a real drift motivates it, not speculatively.

Index identity is name AND column list
--------------------------------------
Matching on name alone is not enough, because ``storage/optimize.py``
creates indexes with ``CREATE INDEX IF NOT EXISTS`` and migration 055
creates one with ``if_not_exists=True``. Those two definitions are
independent string literals in two files with nothing pinning them
together. If a database already carries an index with the right *name* over
the wrong *columns*, every ``IF NOT EXISTS`` from then on silently accepts
it — so a name-only check would report green forever on a permanently wrong
index. Comparing the reflected ``column_names`` against the declared
columns closes that for every index at once.

An ORM index is matched against reflected *indexes* only. A same-named
unique *constraint* does not satisfy it: accepting that spelling would be
presence-only (a constraint carries no comparable column list through the
inspector), which is exactly the name-only check the paragraph above
argues against. Both unique ORM indexes declared today
(``ix_documents_namespace_external_id_unique``,
``idx_namespace_stable_active``) are written as ``CREATE UNIQUE INDEX`` by
their migrations on both dialects, so nothing takes that path anyway.

Dialect exemptions are rules, not name lists
--------------------------------------------
A name list goes stale silently the moment someone adds an index. The two
exemptions are therefore derived from the declaration itself:

* An index is invisible to SQLite reflection iff it declares
  ``postgresql_using`` (a Postgres-only access method — the SQLite branch
  of the chain never builds it) or is expression-based (SQLAlchemy's SQLite
  dialect skips those during reflection with a ``SAWarning``).
* A column is absent from the SQLite schema iff its type is ``Vector`` or
  ``TSVECTOR`` — on the embedded stack LanceDB owns the vectors and FTS5
  owns full-text search, so the chain never adds those columns.
The index rule deliberately **over**-exempts. Migration 004 builds
``ix_temporal_edges_occurred_brin`` as a plain b-tree on SQLite even though
the ORM declares ``postgresql_using="brin"``, so the SQLite leg could in
principle check it and does not. Exempting it costs a check the Postgres
leg still performs; narrowing the rule to inspect each declaration's SQLite
branch would couple the rule to migration internals, which is how a rule
turns back into a name list.

Neither rule fires on the Postgres leg, which is why the Postgres leg sees
drift the SQLite leg cannot — see ``PG_ONLY_INDEX_BASELINE``. That is the
point of having a second leg at all.

One thing the gate does NOT paper over
--------------------------------------
Three ORM indexes are **config-conditional**: migrations 002 / 007 / 024
build ``ix_chunks_embedding_hnsw``, ``ix_entities_embedding_hnsw`` and
``ix_chronicle_events_embedding_hnsw`` only when
``full_precision_hnsw_supported()`` holds, i.e. at or below pgvector's
``VECTOR_HNSW_MAX_DIM`` ceiling for the ``vector`` HNSW opclass. Above it
the chain builds migration 018's ``halfvec`` expression index instead, which
the ORM does not declare.

There is deliberately **no exemption rule** for them. An exemption would
also hide a genuine regression — an hnsw index the chain stopped building
for some unrelated reason — at every dimension above the ceiling. Instead
each leg pins the dimension it migrates at (see
``_MIGRATION_EMBEDDING_DIMENSION`` on the Postgres leg), which makes the
gate's claim well-defined: *at the pinned dimension, every ORM index the
chain can build, it does build*. ``TestConfigConditionalIndexes`` in the
SQLite leg is the tripwire if that pin ever moves above the ceiling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import TSVECTOR

from khora.db.migrations._schema_config import DEFAULT_EMBEDDING_DIMENSION, VECTOR_HNSW_MAX_DIM
from khora.db.models import Base

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "src" / "khora" / "db" / "migrations"


def make_alembic_config(url: str) -> Config:
    """Build a programmatic Alembic Config pointing at the bundled migrations."""
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    cfg.set_main_option("sqlalchemy.url", url)
    cfg.attributes["database_url"] = url
    return cfg


def upgrade(url: str, revision: str = "head") -> None:
    """Run the bundled Alembic chain against ``url`` up to ``revision``."""
    command.upgrade(make_alembic_config(url), revision)


# ---------------------------------------------------------------------------
# Exemption rules
# ---------------------------------------------------------------------------


def index_invisible_on_sqlite(index: sa.Index) -> bool:
    """True when SQLite cannot be expected to report this ORM index.

    Two cases: a Postgres-only access method (``postgresql_using``), which
    the SQLite branch of the chain never builds, and an expression-based
    index, which SQLAlchemy's SQLite reflection skips.
    """
    if index.dialect_kwargs.get("postgresql_using"):
        return True
    return any(not isinstance(expr, sa.Column) for expr in index.expressions)


def column_absent_on_sqlite(column: sa.Column) -> bool:
    """True when the SQLite chain deliberately does not create this column."""
    return isinstance(column.type, (Vector, TSVECTOR))


# ---------------------------------------------------------------------------
# The diff
# ---------------------------------------------------------------------------


@dataclass
class SchemaDrift:
    """ORM declarations with no counterpart in the live schema.

    Every entry is a ``"<table>.<name>"`` string so the sets compare
    cleanly against a literal baseline ledger and read plainly in a
    failure message.
    """

    missing_tables: set[str] = field(default_factory=set)
    missing_columns: set[str] = field(default_factory=set)
    missing_indexes: set[str] = field(default_factory=set)
    # Index exists under the declared name but over different columns (or in
    # a different order). See the "Index identity" note in the module
    # docstring — this is the dimension that keeps ``IF NOT EXISTS`` honest.
    wrong_index_columns: set[str] = field(default_factory=set)
    nullable_in_live: set[str] = field(default_factory=set)


def collect_drift(inspector: sa.Inspector, *, sqlite: bool) -> SchemaDrift:
    """Diff ``Base.metadata`` against the schema ``inspector`` reflects.

    ``sqlite=True`` applies the two dialect exemption rules above. On the
    Postgres leg pass ``sqlite=False`` and nothing is exempt — including the
    three config-conditional hnsw indexes, which callers keep honest by
    pinning the dimension they migrate at rather than by exempting them here.
    """
    drift = SchemaDrift()
    live_tables = set(inspector.get_table_names())

    for table_name, table in Base.metadata.tables.items():
        if table_name not in live_tables:
            drift.missing_tables.add(table_name)
            continue

        live_indexes = {idx["name"]: list(idx["column_names"] or []) for idx in inspector.get_indexes(table_name)}
        for index in table.indexes:
            if sqlite and index_invisible_on_sqlite(index):
                continue
            qualified = f"{table_name}.{index.name}"
            if index.name in live_indexes:
                declared = [col.name for col in index.columns]
                if live_indexes[index.name] != declared:
                    drift.wrong_index_columns.add(qualified)
            else:
                drift.missing_indexes.add(qualified)

        live_columns = {col["name"]: col for col in inspector.get_columns(table_name)}
        for column in table.columns:
            qualified = f"{table_name}.{column.name}"
            live = live_columns.get(column.name)
            if live is None:
                if sqlite and column_absent_on_sqlite(column):
                    continue
                drift.missing_columns.add(qualified)
                continue
            # ORM subset-of live: a NOT NULL the ORM promises must be
            # installed. The reverse (live stricter than the ORM) is not
            # this gate's business.
            if not column.nullable and live["nullable"]:
                drift.nullable_in_live.add(qualified)

    return drift


def assert_ratchet(actual: set[str], baseline: frozenset[str], label: str) -> None:
    """Assert the drift set matches its baseline ledger exactly, both ways.

    ``actual - baseline`` is new drift: a declaration the chain stopped
    building. ``baseline - actual`` is a stale ledger entry: someone fixed
    a drift and left the line behind.

    The second direction is what makes this a ratchet rather than a
    snapshot. Without it the ledger only ever grows and any failure could
    be silenced by appending a line.
    """
    new = actual - baseline
    assert not new, (
        f"New {label} drift, not in the baseline ledger: {sorted(new)}. "
        f"The ORM declares these but the migration chain does not build them. "
        f"Write a migration — do not append to the ledger."
    )
    stale = baseline - actual
    assert not stale, (
        f"Stale {label} baseline entries: {sorted(stale)}. "
        f"These are no longer drifting — delete the lines from the ledger in "
        f"tests/test_helpers/schema_drift.py."
    )


# ---------------------------------------------------------------------------
# Baseline ledgers
# ---------------------------------------------------------------------------
#
# Pre-existing drift, shared by both legs. Every line is a real drift that
# predates this gate: the ORM declares it, the migration chain does not build
# it. The ledgers exist so the gate can be turned on today without an 88-line
# migration; they are asserted in BOTH directions by ``assert_ratchet`` so
# they can only shrink.
#
# To fix one: write a migration and DELETE the line. Leaving the line behind
# fails the gate. Appending a line to silence a failure is the one thing this
# design is built to prevent.
#
# The two legs share these ledgers for the drift both legs can see. A shared
# frozenset asserted in both directions structurally CANNOT hold an entry that
# one leg exempts: the exempting leg never produces it in ``actual``, so the
# stale direction fires. Drift that only one leg can observe therefore needs
# its own delta set (see ``PG_ONLY_INDEX_BASELINE``), unioned in by that leg
# only. Do not loosen the shared ledgers to accommodate it.

# ORM indexes no migration creates. Two of the three
# (``ix_relationships_*``) are in the ``optimize_storage()`` catch-up list, so
# a database an operator ran it on has them; ``ix_memory_namespaces_namespace_id``
# is created by nothing at all — not the chain, not the catch-up list.
# ``ix_documents_namespace_source_type`` is deliberately absent — migration 055
# creates it, so removing that migration must make the gate fail by name.
INDEX_BASELINE = frozenset(
    {
        "memory_namespaces.ix_memory_namespaces_namespace_id",
        "relationships.ix_relationships_namespace_type",
        "relationships.ix_relationships_target_source",
    }
)

# Un-migrated ORM indexes that ONLY the Postgres leg can observe, because the
# SQLite leg exempts them under ``index_invisible_on_sqlite``. Unioned into
# INDEX_BASELINE by the Postgres module alone.
#
# ``ix_entities_namespace_mentions`` is declared with ``mention_count.desc()``,
# which makes it expression-based and therefore SQLite-exempt. No migration
# creates it — its only creator is the ``optimize_storage()`` catch-up list,
# which is a runtime opt-in and not part of the chain. It is a real drift,
# visible on one leg only.
PG_ONLY_INDEX_BASELINE = frozenset(
    {
        "entities.ix_entities_namespace_mentions",
    }
)

# ORM columns declared ``nullable=False`` that the chain built NULLABLE.
# Overwhelmingly columns carrying a Python-side ``default=`` and no explicit
# ``nullable`` argument: the declarative layer infers NOT NULL from the
# non-Optional annotation, ``op.create_table`` in the migration did not.
# ``documents.source_type`` (migration 055) and ``documents.created_at``
# (migration 056) are deliberately absent for the same reason as the index
# above: both are now built NOT NULL by the chain, so ledgering either would
# make the gate green whether or not its migration exists. Note
# ``documents.updated_at`` below stays — 056 flips ``created_at`` only.
NULLABILITY_BASELINE = frozenset(
    {
        "chronicle_events.created_at",
        "chunks.chunk_index",
        "chunks.created_at",
        "chunks.embedding_model",
        "chunks.end_char",
        "chunks.metadata",
        "chunks.start_char",
        "chunks.token_count",
        "documents.chunk_count",
        "documents.entity_count",
        "documents.metadata",
        "documents.size_bytes",
        "documents.status",
        "documents.updated_at",
        "entities.attributes",
        "entities.confidence",
        "entities.created_at",
        "entities.description",
        "entities.embedding_model",
        "entities.entity_type",
        "entities.mention_count",
        "entities.metadata",
        "entities.source_chunk_ids",
        "entities.source_document_ids",
        "entities.updated_at",
        "episodes.created_at",
        "episodes.description",
        "episodes.embedding_model",
        "episodes.entity_ids",
        "episodes.metadata",
        "episodes.occurred_at",
        "episodes.source_chunk_ids",
        "episodes.source_document_ids",
        "episodes.updated_at",
        "expertise_definitions.config",
        "expertise_definitions.created_at",
        "expertise_definitions.description",
        "expertise_definitions.is_active",
        "expertise_definitions.metadata",
        "expertise_definitions.updated_at",
        "expertise_definitions.version",
        "memory_events.actor_type",
        "memory_events.data",
        "memory_events.metadata",
        "memory_events.timestamp",
        "memory_events.version",
        "memory_facts.created_at",
        "memory_facts.updated_at",
        "memory_namespaces.config_overrides",
        "memory_namespaces.created_at",
        "memory_namespaces.metadata",
        "memory_namespaces.sync_checkpoints",
        "memory_namespaces.updated_at",
        "permissions.created_at",
        "permissions.metadata",
        "permissions.updated_at",
        "relationships.confidence",
        "relationships.created_at",
        "relationships.description",
        "relationships.metadata",
        "relationships.properties",
        "relationships.relationship_type",
        "relationships.source_chunk_ids",
        "relationships.source_document_ids",
        "relationships.updated_at",
        "relationships.weight",
        "sync_checkpoints.created_at",
        "sync_checkpoints.metadata",
        "sync_checkpoints.updated_at",
        "temporal_edges.confidence",
        "temporal_edges.created_at",
        "temporal_edges.description",
        "temporal_edges.ingested_at",
        "temporal_edges.is_valid",
        "temporal_edges.metadata",
        "temporal_edges.properties",
        "temporal_edges.relationship_type",
        "temporal_edges.source_chunk_ids",
        "temporal_edges.source_document_ids",
        "time_nodes.created_at",
        "time_nodes.edge_count",
        "time_nodes.entity_count",
        "time_nodes.metadata",
        "time_nodes.updated_at",
    }
)

__all__ = [
    "DEFAULT_EMBEDDING_DIMENSION",
    "INDEX_BASELINE",
    "MIGRATIONS_DIR",
    "NULLABILITY_BASELINE",
    "PG_ONLY_INDEX_BASELINE",
    "VECTOR_HNSW_MAX_DIM",
    "SchemaDrift",
    "assert_ratchet",
    "collect_drift",
    "column_absent_on_sqlite",
    "index_invisible_on_sqlite",
    "make_alembic_config",
    "upgrade",
]
