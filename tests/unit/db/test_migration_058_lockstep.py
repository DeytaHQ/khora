"""#1574 — migration 058 and the pgvector store must install the SAME function.

``khora_chunks`` is not part of the Alembic-managed schema: it is created at
runtime by ``PgVectorTemporalStore.connect()``, which issues its own
``CREATE OR REPLACE FUNCTION khora_chunks_content_tsv_trigger()`` on every boot.
Migration ``058_khora_chunks_title_fts`` issues one too. Both target the same
function, so **whichever runs last wins** — and if the two bodies ever drift, the
formula a deployment ends up with depends on boot order (migration first, or a
store ``connect()`` first). That is not a difference a test elsewhere would
catch: both statements succeed either way, and the only symptom is a lexical
index that is weighted on some deployments and not others.

The duplication itself is deliberate — a migration is a frozen snapshot, so it
must not ``import`` the runtime constant. This module is the seam that keeps the
two copies honest without coupling them, and it is the reason the runtime SQL
was lifted out of ``connect()`` into a module-level constant in the first place.

No database: this is a string comparison. It belongs in the unit lane so it runs
on every PR, including for contributors without Docker.
"""

from __future__ import annotations

import importlib

import pytest

from khora.storage.temporal.pgvector import _TSV_FUNCTION_SQL as RUNTIME_TSV_FUNCTION_SQL

pytestmark = pytest.mark.unit

_REVISION = "058_khora_chunks_title_fts"
_MIGRATION = importlib.import_module(f"khora.db.migrations.versions.{_REVISION}")

MIGRATION_TSV_FUNCTION_SQL: str = _MIGRATION._TSV_FUNCTION_SQL
MIGRATION_TSV_FUNCTION_SQL_CONTENT_ONLY: str = _MIGRATION._TSV_FUNCTION_SQL_CONTENT_ONLY


def test_upgrade_function_body_is_byte_identical_to_the_runtime_copy() -> None:
    """The convergence contract, asserted the only way that is meaningful.

    Byte-for-byte rather than normalized: whitespace inside a ``plpgsql`` body
    is cosmetic to Postgres but a normalized comparison would let the two copies
    diverge in the parts that ARE cosmetic, and nobody would then notice the
    edit that changed one of them substantively.
    """
    assert MIGRATION_TSV_FUNCTION_SQL == RUNTIME_TSV_FUNCTION_SQL, (
        "migration 058 and khora/storage/temporal/pgvector.py install different "
        "khora_chunks_content_tsv_trigger() bodies; whichever runs last wins, so "
        "the installed formula would depend on boot order. Change both together."
    )


def test_both_copies_weight_title_above_content() -> None:
    """The shared body is the #1574 formula, not merely a matching pair.

    Equality alone would still pass if both copies were reverted together. These
    are the three fragments the feature actually needs: the ``'A'`` label on
    ``title``, the ``'B'`` label on ``content``, and the ``coalesce`` that stops
    a NULL title annihilating the whole concatenation.
    """
    for name, sql in (
        ("runtime", RUNTIME_TSV_FUNCTION_SQL),
        ("migration", MIGRATION_TSV_FUNCTION_SQL),
    ):
        assert "setweight(to_tsvector('english', coalesce(NEW.title, '')), 'A')" in sql, name
        assert "setweight(to_tsvector('english', NEW.content), 'B')" in sql, name


def test_downgrade_restores_the_unweighted_content_only_formula() -> None:
    """058's ``downgrade()`` must reinstall the pre-#1574 body, not a variant.

    A downgrade that left ``setweight`` in place would keep the labels while
    claiming to have removed them — and the ranking vector the store passes is
    chosen on the assumption that a downgraded database has unlabelled (``D``)
    content tokens.
    """
    assert "NEW.content_tsv := to_tsvector('english', NEW.content);" in MIGRATION_TSV_FUNCTION_SQL_CONTENT_ONLY
    assert "setweight" not in MIGRATION_TSV_FUNCTION_SQL_CONTENT_ONLY
    assert "NEW.title" not in MIGRATION_TSV_FUNCTION_SQL_CONTENT_ONLY


def test_both_directions_replace_the_same_function() -> None:
    """Upgrade and downgrade must name one function, or a downgrade orphans one."""
    signature = "CREATE OR REPLACE FUNCTION khora_chunks_content_tsv_trigger() RETURNS trigger"
    assert MIGRATION_TSV_FUNCTION_SQL.startswith(signature)
    assert MIGRATION_TSV_FUNCTION_SQL_CONTENT_ONLY.startswith(signature)
    assert RUNTIME_TSV_FUNCTION_SQL.startswith(signature)


def test_revision_pins_the_expected_predecessor() -> None:
    """Cheap chain guard: 058 sits directly on top of 057.

    The PG lane builds the schema at ``down_revision`` and then upgrades one
    step, so a silently re-pointed ``down_revision`` would make that test
    exercise a different chain position while still passing.
    """
    assert _MIGRATION.revision == _REVISION
    assert _MIGRATION.down_revision == "057_drop_documents_created_at_index"
