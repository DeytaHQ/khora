"""Guard: bi-temporal columns are RESERVED, never filtered on read (#888).

Migrations 033/034 added ``valid_to`` / ``invalidated_at`` / ``invalidated_by``
columns plus ``WHERE invalidated_at IS NULL`` partial indexes on ``entities``,
``relationships``, ``memory_facts`` and ``chronicle_events``. Per ADR-003
(option B) these columns are reserved scaffolding: written only by dream-apply
on the PostgreSQL side, and NOT filtered on the ingest / recall read paths.

Adding a read-side ``WHERE invalidated_at IS NULL`` filter before a Neo4j
tombstone-mirror exists would make pg-side reads diverge from graph-side reads
(the latent hazard the issue tracks). This test is a tripwire: if a future PR
adds such a filter to a read path, CI fails here with a pointer to #888 so the
divergence is caught before it ships.

The scan is scoped to the read paths (``engines/`` and ``storage/backends/``).
``src/khora/dream/`` is explicitly allowlisted - it is the legitimate writer
and its idempotency guards use ``invalidated_at IS NULL`` by design.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# tests/unit/ -> project root -> src/khora
_SRC = Path(__file__).resolve().parents[2] / "src" / "khora"

# Read paths to scan. The dream package (the legitimate writer) is NOT in this
# list, so it is excluded by construction; see the module docstring.
_READ_PATH_DIRS = [
    _SRC / "engines",
    _SRC / "storage" / "backends",
]

# Matches the dangerous read-side soft-delete predicate in any spacing/case,
# e.g. ``invalidated_at IS NULL`` or ``invalidated_at  is   null``. This is the
# SQL column from migrations 033/034 - distinct from the Neo4j node-versioning
# property ``version_valid_to`` (a different column that is safe to filter on).
_FILTER_PATTERN = re.compile(r"\binvalidated_at\b\s+is\s+null", re.IGNORECASE)


@pytest.mark.unit
def test_no_premature_invalidated_at_read_filter() -> None:
    offenders: list[str] = []
    for root in _READ_PATH_DIRS:
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for lineno, line in enumerate(text.splitlines(), start=1):
                if _FILTER_PATTERN.search(line):
                    offenders.append(f"{path}:{lineno}: {line.strip()}")

    assert not offenders, (
        "Found a `WHERE invalidated_at IS NULL` read filter on a bi-temporal "
        "column in the ingest/recall read path. These columns (migrations "
        "033/034) are RESERVED: written only by dream-apply (pg-side), never "
        "filtered on read, until the Neo4j tombstone-mirror lands - otherwise "
        "pg/neo4j reads diverge. See #888 and ADR-003. Offending lines:\n" + "\n".join(offenders)
    )
