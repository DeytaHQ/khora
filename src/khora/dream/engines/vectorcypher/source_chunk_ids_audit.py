"""Read-only audit of ``Entity.source_chunk_ids`` integrity and shape (#659).

For every entity in a namespace this op reports:

* per-entity: the array length and the count of UUIDs that no longer map
  to a live ``chunks`` row (dead references — likely chunk deletions
  that the online path didn't fan out to all entities that mentioned
  them);
* per-namespace: the p50/p90/p99/max array length and the total
  dead-UUID count;
* the top-K entities by ``source_chunk_ids`` length — those are the GC
  candidates for the Phase 2.3 mutation op.

Pure SELECTs. Zero LLM calls. Zero writes.

Two SQL paths share the same Python shape:

* **PostgreSQL** uses ``unnest(source_chunk_ids)`` + ``LEFT JOIN chunks``
  so the dead-UUID count is computed in the database — large arrays
  never round-trip into Python.
* **SQLite** stores ``source_chunk_ids`` as a JSON-text column (see
  ``migration 000_initial_schema._uuid_array_type``); we read the raw
  text, parse to a list of UUIDs in Python, and check membership
  against the namespace's chunk-id set.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from time import perf_counter
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from sqlalchemy import text

from khora.dream.plan import DreamOp, OpKind

if TYPE_CHECKING:
    from khora.storage.coordinator import StorageCoordinator


_PHASE = "audit"
_DECISION = "audit_complete"


async def plan_vectorcypher_source_chunk_ids_audit(
    namespace_id: UUID,
    *,
    coordinator: StorageCoordinator,
    top_k_offenders: int = 20,
) -> DreamOp:
    """Audit ``Entity.source_chunk_ids`` arrays for dead UUIDs + length distribution.

    Args:
        namespace_id: The stable namespace identifier. Resolved to the
            active row-level id via the coordinator before any SELECT
            runs.
        coordinator: Storage coordinator — must have a SQL backend
            (``relational`` / ``vector`` / ``event_store``) so a session
            can be opened.
        top_k_offenders: Cap on the ``top_offenders`` list in the report.
            Defaults to 20.

    Returns:
        A :class:`DreamOp` whose ``outputs[0]`` is a dict with keys:

        * ``total_entities`` — entities considered in the namespace.
        * ``total_dead_uuids`` — sum of dead-UUID counts across all
          entities.
        * ``length_p50`` / ``length_p90`` / ``length_p99`` /
          ``length_max`` — percentiles + max of array length.
        * ``top_offenders`` — list of
          ``{entity_id, name, length, dead_uuid_count}`` dicts ordered
          by ``length`` descending. Truncated to ``top_k_offenders``.

        The op carries ``decision="audit_complete"`` always (this op
        only reports; it never short-circuits).

    Read-only. Performs ``SELECT`` queries only — no mutations, no LLM
    calls.
    """
    started_at = datetime.now(UTC)
    started_perf = perf_counter()

    resolved_id = await coordinator.resolve_namespace(namespace_id)

    rows = await _collect_entity_rows(coordinator, resolved_id)

    lengths = [r["length"] for r in rows]
    total_entities = len(rows)
    total_dead_uuids = sum(r["dead_uuid_count"] for r in rows)

    p50, p90, p99, max_len = _length_percentiles(lengths)

    rows_sorted = sorted(rows, key=lambda r: r["length"], reverse=True)
    top_offenders = [
        {
            "entity_id": str(r["entity_id"]),
            "name": r["name"],
            "length": r["length"],
            "dead_uuid_count": r["dead_uuid_count"],
        }
        for r in rows_sorted[:top_k_offenders]
    ]

    report: dict[str, Any] = {
        "total_entities": total_entities,
        "total_dead_uuids": total_dead_uuids,
        "length_p50": p50,
        "length_p90": p90,
        "length_p99": p99,
        "length_max": max_len,
        "top_offenders": top_offenders,
    }

    duration_ms = (perf_counter() - started_perf) * 1000.0

    return DreamOp(
        op_id=uuid4(),
        phase=_PHASE,
        op_type=OpKind.VECTORCYPHER_SOURCE_CHUNK_IDS_AUDIT,
        inputs=({"namespace_id": str(namespace_id), "top_k_offenders": top_k_offenders},),
        outputs=(report,),
        decision=_DECISION,
        rationale="",
        started_at=started_at,
        duration_ms=duration_ms,
        namespace_id=namespace_id,
    )


# ---------------------------------------------------------------------------
# SQL paths
# ---------------------------------------------------------------------------


async def _collect_entity_rows(
    coordinator: StorageCoordinator,
    resolved_namespace_id: UUID,
) -> list[dict[str, Any]]:
    """Return per-entity audit rows for the namespace.

    Each row carries ``entity_id``, ``name``, ``length`` (array
    cardinality) and ``dead_uuid_count`` (UUIDs in
    ``source_chunk_ids`` that do not appear in ``chunks.id``).
    """
    async with coordinator.transaction() as txn:
        session = txn.session
        dialect = session.bind.dialect.name if session.bind is not None else ""

        if dialect == "postgresql":
            return await _collect_postgres(session, resolved_namespace_id)
        return await _collect_sqlite(session, resolved_namespace_id)


async def _collect_postgres(session: Any, resolved_namespace_id: UUID) -> list[dict[str, Any]]:
    """PostgreSQL path: array unnest + LEFT JOIN, dead count computed in-DB."""
    sql = text(
        """
        WITH dead AS (
            SELECT
                e.id AS entity_id,
                COUNT(*) FILTER (WHERE c.id IS NULL) AS dead_count
            FROM entities AS e
            JOIN LATERAL unnest(e.source_chunk_ids) AS cid ON TRUE
            LEFT JOIN chunks AS c
                   ON c.id = cid AND c.namespace_id = e.namespace_id
            WHERE e.namespace_id = :ns
            GROUP BY e.id
        )
        SELECT
            e.id AS entity_id,
            e.name AS name,
            COALESCE(cardinality(e.source_chunk_ids), 0) AS length,
            COALESCE(d.dead_count, 0) AS dead_count
        FROM entities AS e
        LEFT JOIN dead AS d ON d.entity_id = e.id
        WHERE e.namespace_id = :ns
        """
    )
    result = await session.execute(sql, {"ns": resolved_namespace_id})
    return [
        {
            "entity_id": row.entity_id,
            "name": row.name,
            "length": int(row.length or 0),
            "dead_uuid_count": int(row.dead_count or 0),
        }
        for row in result
    ]


async def _collect_sqlite(session: Any, resolved_namespace_id: UUID) -> list[dict[str, Any]]:
    """SQLite path: parse JSON-text arrays in Python, anti-join in memory."""
    ns_param = _sqlite_namespace_param(resolved_namespace_id)

    chunk_rows = await session.execute(
        text("SELECT id FROM chunks WHERE namespace_id = :ns"),
        {"ns": ns_param},
    )
    live_chunk_ids: set[UUID] = set()
    for row in chunk_rows:
        cid = row.id if hasattr(row, "id") else row[0]
        if cid is None:
            continue
        try:
            live_chunk_ids.add(UUID(str(cid)))
        except (TypeError, ValueError):
            continue

    entity_rows = await session.execute(
        text("SELECT id, name, source_chunk_ids FROM entities WHERE namespace_id = :ns"),
        {"ns": ns_param},
    )

    out: list[dict[str, Any]] = []
    for row in entity_rows:
        entity_id = row.id if hasattr(row, "id") else row[0]
        name = row.name if hasattr(row, "name") else row[1]
        raw = row.source_chunk_ids if hasattr(row, "source_chunk_ids") else row[2]

        chunk_ids = _parse_sqlite_uuid_list(raw)
        dead = sum(1 for cid in chunk_ids if cid not in live_chunk_ids)
        out.append(
            {
                "entity_id": UUID(str(entity_id)) if not isinstance(entity_id, UUID) else entity_id,
                "name": name,
                "length": len(chunk_ids),
                "dead_uuid_count": dead,
            }
        )
    return out


def _sqlite_namespace_param(resolved_namespace_id: UUID) -> str:
    """SQLite stores UUIDs as 32-char hex without dashes — match that.

    See ``khora.storage.backends.sqlite_lance._helpers.uuid_to_text``.
    """
    return resolved_namespace_id.hex


def _parse_sqlite_uuid_list(value: Any) -> list[UUID]:
    """Decode the SQLite JSON-text representation of a UUID array."""
    if value is None:
        return []
    if isinstance(value, list):
        raw_items = value
    else:
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return []
        if not isinstance(parsed, list):
            return []
        raw_items = parsed

    out: list[UUID] = []
    for item in raw_items:
        try:
            out.append(UUID(str(item)))
        except (TypeError, ValueError):
            continue
    return out


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------


def _length_percentiles(lengths: list[int]) -> tuple[int, int, int, int]:
    """Return (p50, p90, p99, max) of ``lengths``; zeros when empty."""
    if not lengths:
        return 0, 0, 0, 0
    sorted_lengths = sorted(lengths)
    return (
        _percentile(sorted_lengths, 0.50),
        _percentile(sorted_lengths, 0.90),
        _percentile(sorted_lengths, 0.99),
        sorted_lengths[-1],
    )


def _percentile(sorted_values: list[int], q: float) -> int:
    """Nearest-rank percentile — matches ``statistics.quantiles``' floor behaviour.

    Returns an int because array lengths are int. Empty input returns
    0; single-value input returns that value.
    """
    if not sorted_values:
        return 0
    if len(sorted_values) == 1:
        return sorted_values[0]
    # Nearest-rank: rank = ceil(q * N), 1-indexed.
    rank = max(1, int(q * len(sorted_values) + 0.999999))
    rank = min(rank, len(sorted_values))
    return sorted_values[rank - 1]
