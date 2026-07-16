"""Bi-temporal columns are now ACTIVE on the read path (#888 -> #970 -> #1272).

History: migrations 033/034 added ``valid_to`` / ``invalidated_at`` /
``invalidated_by`` (plus ``WHERE invalidated_at IS NULL`` partial indexes) on
``entities`` / ``relationships`` / ``memory_facts`` / ``chronicle_events``. They
were RESERVED scaffolding - written by dream-apply on the PG side, never
filtered on read - and this module was a tripwire that FAILED if a read filter
landed before the Neo4j tombstone-mirror existed, because a PG-only filter would
diverge from the (unfiltered) graph reads.

The mirror landed in #1272: dream apply now mirrors its PG tombstones onto the
graph ``valid_until`` (post-commit, reconciler-backed), and the read filter is
applied to BOTH stores in lockstep. The reserve is over - so the old tripwire is
RETIRED and inverted. This module now asserts the read filters are PRESENT (the
contract is now "filter on read", not "never filter on read"), widened to cover
the ``valid_to`` (relationship prune) and chronicle-side columns the original
guard called out for the widen-then-retire handoff.

See ``https://docs.deyta.ai/khora/storage-backends`` (bi-temporal section) and #970.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# tests/unit/ -> project root -> src/khora
_SRC = Path(__file__).resolve().parents[2] / "src" / "khora"
_PGVECTOR = _SRC / "storage" / "backends" / "pgvector.py"
_NEO4J = _SRC / "storage" / "backends" / "neo4j.py"


@pytest.mark.unit
def test_pgvector_list_paths_filter_soft_deletes() -> None:
    """The pgvector recall list paths now hide soft-deleted rows (#1272).

    Widened from the original guard to cover all three soft-delete columns:
    ``valid_until`` (entity dedupe), ``valid_to`` (relationship prune) and
    ``invalidated_at`` (relationship self-loop), via the shared filter helpers.
    """
    text = _PGVECTOR.read_text(encoding="utf-8")
    # The shared filter helpers exist and encode the three columns.
    assert "_entity_live_filter" in text
    assert "_relationship_live_filter" in text
    assert "EntityModel.valid_until" in text
    assert "RelationshipModel.valid_to.is_(None)" in text
    assert "RelationshipModel.invalidated_at.is_(None)" in text
    # And the list paths apply them.
    assert text.count("_entity_live_filter()") >= 1
    assert text.count("_relationship_live_filter()") >= 1


@pytest.mark.unit
def test_neo4j_read_paths_filter_valid_until_unconditionally() -> None:
    """The Neo4j read paths now filter ``valid_until`` unconditionally (#1272).

    The mirror folds PG ``valid_to`` / ``invalidated_at`` / ``valid_until`` onto
    the single graph ``valid_until``; recall hides it in lockstep with PG so the
    two stores agree on the live set.
    """
    text = _NEO4J.read_text(encoding="utf-8")
    # Both list paths carry the unconditional valid_until read filter.
    pattern = re.compile(r"valid_until\s+IS\s+NULL\s+OR\s+\S*valid_until\s*>\s*\$now", re.IGNORECASE)
    matches = pattern.findall(text)
    # One for list_entities (e.valid_until), one for list_relationships (r.valid_until).
    assert len(matches) >= 2, f"expected >=2 unconditional valid_until read filters, found {len(matches)}"
