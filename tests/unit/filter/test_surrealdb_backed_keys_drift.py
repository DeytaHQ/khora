"""Drift gate: the surrealdb backed-key set tracks the temporal_chunk schema — ``@internal``.

The SurrealDB recall path can only push a predicate over a system key the
``temporal_chunk`` table actually backs with a real column. The store declares that
queryable set as :data:`~khora.storage.temporal.surrealdb._BACKED_SYSTEM_KEYS`
(``{"occurred_at", "created_at"}``) and the compiler refuses (raising) any predicate
over a system key outside it.

That declared set is hand-maintained, but its SOURCE OF TRUTH is the schema: the only
:data:`SYSTEM_KEYS` members that are real columns on the SCHEMAFULL ``temporal_chunk``
table. This test parses the schema DDL the store actually runs and asserts the two
stay in lock-step, so a future schema edit (adding a system-key column, or dropping
``occurred_at``) that is not mirrored into ``_BACKED_SYSTEM_KEYS`` — or vice versa —
fails LOUDLY here instead of silently letting an unbacked predicate through
(drop-every-row bug) or rejecting a now-backed key.

Note: ``source_system`` / ``author`` / ``channel`` / ``tags`` are columns on the
table but are NOT :data:`SYSTEM_KEYS`, so the ``SYSTEM_KEYS ∩ columns`` intersection
naturally excludes them — the gate is about the filterable system-key surface, not
every physical column.
"""

from __future__ import annotations

import re

import pytest

from khora.filter.model import SYSTEM_KEYS
from khora.storage.temporal.surrealdb import _BACKED_SYSTEM_KEYS, _TEMPORAL_CHUNK_SCHEMA

pytestmark = pytest.mark.unit

# Every ``DEFINE FIELD IF NOT EXISTS <name> ON temporal_chunk`` column name in the
# schema DDL the store runs on ``connect()``.
_SCHEMA_FIELD_RE = re.compile(r"DEFINE FIELD IF NOT EXISTS (\w+) ON temporal_chunk")


def _schema_columns() -> frozenset[str]:
    """The set of column names defined on the temporal_chunk table by the store DDL."""
    return frozenset(_SCHEMA_FIELD_RE.findall(_TEMPORAL_CHUNK_SCHEMA))


def test_backed_system_keys_are_exactly_occurred_at_and_created_at() -> None:
    # The declared backed (queryable) system-key set is exactly the two datetime
    # columns. A regression that adds/drops a key is caught here.
    assert _BACKED_SYSTEM_KEYS == frozenset({"occurred_at", "created_at"})


def test_backed_system_keys_match_schema_columns() -> None:
    # The drift gate: the declared backed system-key set MUST equal the SYSTEM_KEYS
    # members that are actually columns on the temporal_chunk table. If a schema
    # migration adds (or removes) a system-key column without updating
    # _BACKED_SYSTEM_KEYS — or _BACKED_SYSTEM_KEYS drifts from the schema — these two
    # sets diverge and this assertion fails, naming the offending keys.
    schema_backed = SYSTEM_KEYS & _schema_columns()
    assert _BACKED_SYSTEM_KEYS == schema_backed, (
        f"surrealdb backed system keys {sorted(_BACKED_SYSTEM_KEYS)} drifted from temporal_chunk schema "
        f"columns {sorted(schema_backed)} — update _BACKED_SYSTEM_KEYS to match the schema"
    )


def test_backed_set_is_a_subset_of_system_keys() -> None:
    # The backed set only ever names real SYSTEM_KEYS members — a typo or a stray
    # non-system column (e.g. ``source_system``) would not be a filterable key.
    assert _BACKED_SYSTEM_KEYS <= SYSTEM_KEYS


def test_unbacked_keys_are_the_eight_remaining_system_keys() -> None:
    # The complement within SYSTEM_KEYS is the eight denormalized document keys the
    # gate rejects — pin it so a key silently migrating between backed/unbacked is
    # caught. (Guards the partition: backed ∪ unbacked == SYSTEM_KEYS, disjoint.)
    unbacked = SYSTEM_KEYS - _BACKED_SYSTEM_KEYS
    assert unbacked == frozenset(
        {
            "source_type",
            "source_name",
            "source_url",
            "source_timestamp",
            "external_id",
            "content_type",
            "source",
            "title",
        }
    )
    assert _BACKED_SYSTEM_KEYS | unbacked == SYSTEM_KEYS
    assert _BACKED_SYSTEM_KEYS & unbacked == frozenset()
