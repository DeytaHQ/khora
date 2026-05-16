"""Plan rewrites of ``Entity.source_chunk_ids`` to drop dead UUIDs (#662).

Phase 2.3 mutation op, **dry-run only in v0.14** — apply mode raises
``NotImplementedError`` and is tracked in #649 phase 4 / #668.

For every entity whose ``source_chunk_ids`` array contains UUIDs that no
longer resolve to a live ``chunks`` row (the online ``forget`` /
``forget_session`` cascade missed the graph back-pointer), this op emits
one :class:`DreamOp` carrying:

* ``inputs[0] = {entity_id, before_length, dead_uuids}``
* ``outputs[0] = {after_array, after_length}``
* ``decision = "planned"``

Entities with zero dead refs (or fewer than ``min_dead``) are not
emitted. ``apply`` would feed the survivor arrays into
``backends/neo4j.py:reset_entity_source_chunk_ids_batch``; that wiring
lands in v0.15.

Two SQL paths share the same Python shape — same split as the Phase 1.5
audit op (``source_chunk_ids_audit``):

* **PostgreSQL** uses ``unnest(source_chunk_ids) WITH ORDINALITY`` so
  the live/dead split is computed in the database and only the dead
  UUIDs (plus the survivor array) round-trip into Python.
* **SQLite** stores ``source_chunk_ids`` as a JSON-text column; we read
  the raw text, parse to a list of UUIDs in Python, and partition
  against the namespace's live chunk-id set.
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


_PHASE = "mutation"
_DECISION = "planned"


async def plan_vectorcypher_source_chunk_ids_gc(
    namespace_id: UUID,
    *,
    coordinator: StorageCoordinator,
    min_dead: int = 1,
    mode: str = "dry-run",
) -> tuple[DreamOp, ...]:
    """Plan ``Entity.source_chunk_ids`` rewrites that drop dead UUIDs.

    Args:
        namespace_id: The stable namespace identifier. Resolved to the
            active row-level id via the coordinator before any SELECT
            runs.
        coordinator: Storage coordinator — must have a SQL backend
            (``relational`` / ``vector`` / ``event_store``) so a session
            can be opened.
        min_dead: Threshold below which an entity is not emitted as a
            candidate. Defaults to 1 — every entity with at least one
            dead reference is planned. Values ``< 1`` are clamped to 1.
        mode: ``"dry-run"`` (the only supported value in v0.14). Any
            other value raises ``NotImplementedError``.

    Returns:
        A tuple of :class:`DreamOp` instances, one per entity with at
        least ``min_dead`` dead UUIDs. Each carries
        ``inputs[0] = {entity_id, before_length, dead_uuids}``,
        ``outputs[0] = {after_array, after_length}``, and
        ``decision="planned"``. Empty when no entity meets the threshold.

    Raises:
        NotImplementedError: When ``mode != "dry-run"``. Apply mode
            lands in v0.15 — see #649 phase 4 / #668.

    Read-only. Performs ``SELECT`` queries only — no mutations, no LLM
    calls.
    """
    if mode != "dry-run":
        raise NotImplementedError("apply mode lands in v0.15 — see #649 phase 4 / #668")

    threshold = max(1, min_dead)
    resolved_id = await coordinator.resolve_namespace(namespace_id)

    rows = await _collect_dead_refs(coordinator, resolved_id)

    ops: list[DreamOp] = []
    for row in rows:
        dead_uuids: list[UUID] = row["dead_uuids"]
        if len(dead_uuids) < threshold:
            continue
        after_array: list[UUID] = row["after_array"]
        before_length: int = row["before_length"]
        entity_id: UUID = row["entity_id"]

        started_perf = perf_counter()
        started_at = datetime.now(UTC)
        duration_ms = (perf_counter() - started_perf) * 1000.0

        ops.append(
            DreamOp(
                op_id=uuid4(),
                phase=_PHASE,
                op_type=OpKind.VECTORCYPHER_SOURCE_CHUNK_IDS_GC,
                inputs=(
                    {
                        "entity_id": str(entity_id),
                        "before_length": before_length,
                        "dead_uuids": [str(u) for u in dead_uuids],
                    },
                ),
                outputs=(
                    {
                        "after_array": [str(u) for u in after_array],
                        "after_length": len(after_array),
                    },
                ),
                decision=_DECISION,
                rationale=(
                    f"Drop {len(dead_uuids)} dead chunk UUID(s) from "
                    f"source_chunk_ids ({before_length} -> {len(after_array)})."
                ),
                started_at=started_at,
                duration_ms=duration_ms,
                namespace_id=namespace_id,
            )
        )
    return tuple(ops)


# ---------------------------------------------------------------------------
# SQL paths
# ---------------------------------------------------------------------------


async def _collect_dead_refs(
    coordinator: StorageCoordinator,
    resolved_namespace_id: UUID,
) -> list[dict[str, Any]]:
    """Return per-entity dead-UUID + survivor-array rows for the namespace.

    Each row carries ``entity_id``, ``before_length``, ``dead_uuids``
    (UUIDs in ``source_chunk_ids`` with no matching ``chunks`` row),
    and ``after_array`` (the survivor UUIDs in their original order).
    """
    async with coordinator.transaction() as txn:
        session = txn.session
        dialect = session.bind.dialect.name if session.bind is not None else ""

        if dialect == "postgresql":
            return await _collect_postgres(session, resolved_namespace_id)
        return await _collect_sqlite(session, resolved_namespace_id)


async def _collect_postgres(session: Any, resolved_namespace_id: UUID) -> list[dict[str, Any]]:
    """PostgreSQL path: array unnest WITH ORDINALITY, partition in-DB."""
    sql = text(
        """
        WITH unrolled AS (
            SELECT
                e.id AS entity_id,
                COALESCE(cardinality(e.source_chunk_ids), 0) AS before_length,
                u.cid AS cid,
                u.ord AS ord,
                c.id IS NULL AS is_dead
            FROM entities AS e
            JOIN LATERAL unnest(e.source_chunk_ids) WITH ORDINALITY AS u(cid, ord) ON TRUE
            LEFT JOIN chunks AS c
                   ON c.id = u.cid AND c.namespace_id = e.namespace_id
            WHERE e.namespace_id = :ns
        )
        SELECT
            entity_id,
            MAX(before_length) AS before_length,
            COALESCE(
                ARRAY_AGG(cid ORDER BY ord) FILTER (WHERE is_dead),
                ARRAY[]::uuid[]
            ) AS dead_uuids,
            COALESCE(
                ARRAY_AGG(cid ORDER BY ord) FILTER (WHERE NOT is_dead),
                ARRAY[]::uuid[]
            ) AS after_array
        FROM unrolled
        GROUP BY entity_id
        HAVING bool_or(is_dead)
        """
    )
    result = await session.execute(sql, {"ns": resolved_namespace_id})
    out: list[dict[str, Any]] = []
    for row in result:
        out.append(
            {
                "entity_id": row.entity_id,
                "before_length": int(row.before_length or 0),
                "dead_uuids": [UUID(str(u)) for u in (row.dead_uuids or [])],
                "after_array": [UUID(str(u)) for u in (row.after_array or [])],
            }
        )
    return out


async def _collect_sqlite(session: Any, resolved_namespace_id: UUID) -> list[dict[str, Any]]:
    """SQLite path: parse JSON-text arrays in Python, partition in memory."""
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
        text("SELECT id, source_chunk_ids FROM entities WHERE namespace_id = :ns"),
        {"ns": ns_param},
    )

    out: list[dict[str, Any]] = []
    for row in entity_rows:
        entity_id = row.id if hasattr(row, "id") else row[0]
        raw = row.source_chunk_ids if hasattr(row, "source_chunk_ids") else row[1]

        chunk_ids = _parse_sqlite_uuid_list(raw)
        if not chunk_ids:
            continue
        dead = [cid for cid in chunk_ids if cid not in live_chunk_ids]
        if not dead:
            continue
        survivors = [cid for cid in chunk_ids if cid in live_chunk_ids]
        out.append(
            {
                "entity_id": UUID(str(entity_id)) if not isinstance(entity_id, UUID) else entity_id,
                "before_length": len(chunk_ids),
                "dead_uuids": dead,
                "after_array": survivors,
            }
        )
    return out


def _sqlite_namespace_param(resolved_namespace_id: UUID) -> str:
    """SQLite stores UUIDs as 32-char hex without dashes — match that."""
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
