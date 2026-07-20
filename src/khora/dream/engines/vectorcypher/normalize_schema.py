"""Vectorcypher schema-drift normalization (#673, Phase 5.4 of #649).

Operator-driven rename of ``entity_type`` and ``relationship_type``
columns across a namespace. **Never auto-derives the mapping** — the
mapping is operator policy supplied via
:class:`khora.dream.config.DreamConfig.normalize_schema_mapping`.

Pairs with the Phase 5.3 contradiction-detection workflow (#672): the
contradiction triage queue is the natural source of mapping
recommendations; this op is what the operator runs after inspecting that
queue and deciding which renames to apply.

**Coordinated-release warning.** Type-name strings are part of the
consumer contract (``khora-cli``, ``khora-explorer``). Running this op
on a production namespace requires lockstep updates to any downstream
consumer that pattern-matches on ``entity_type`` / ``relationship_type``
values. See ``https://docs.deyta.ai/khora``.

Stability: **internal**. The op kind string
``vectorcypher_normalize_schema`` is part of :class:`khora.dream.OpKind`
and stable as an identifier; the planner / apply signatures and
inputs/outputs shape may evolve through Phase 5.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from sqlalchemy import text

from khora.core.models.event import EventType, MemoryEvent
from khora.dream.engines.vectorcypher._uuid_bind import uuid_bind
from khora.dream.plan import DreamOp, OpKind
from khora.dream.result import UndoRecord
from khora.telemetry import trace_span

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from khora.dream.config import DreamConfig
    from khora.storage.coordinator import StorageCoordinator


_PHASE = "normalize"
_OP_SPAN = "khora.dream.vectorcypher.normalize_schema"
_PAGE_SIZE = 1000


async def plan_vectorcypher_normalize_schema(
    namespace_id: UUID,
    *,
    coordinator: StorageCoordinator,
    config: DreamConfig,
) -> list[DreamOp]:
    """Plan entity-type / relationship-type renames for ``namespace_id``.

    Reads every entity and relationship in the namespace; for each row
    whose ``entity_type`` / ``relationship_type`` is a key in
    ``config.normalize_schema_mapping`` emits a rename target. Returns:

    - ``[]`` when ``config.normalize_schema_enabled`` is False.
    - ``[DreamOp(decision="insufficient_input")]`` when the mapping is
      empty — operator policy required, never auto-derived.
    - ``[DreamOp(decision="planned")]`` carrying the full target list
      otherwise (one op per plan; the apply handler iterates internally).

    Args:
        namespace_id: Namespace to scan.
        coordinator: Storage coordinator (DI for tests).
        config: Dream config with the operator-supplied mapping.

    Returns:
        Zero or one :class:`DreamOp`. Multiple ops would split the
        rename across separate undo files; a single op keeps the apply
        transactional.
    """
    if not config.normalize_schema_enabled:
        return []

    mapping = dict(config.normalize_schema_mapping or {})

    started_at = datetime.now(UTC)

    with trace_span(
        _OP_SPAN,
        namespace_id=str(namespace_id),
        mapping_size=len(mapping),
    ) as span:
        if not mapping:
            span.set_attribute("entity_rename_count", 0)
            span.set_attribute("relationship_rename_count", 0)
            return [
                DreamOp(
                    op_id=uuid4(),
                    phase=_PHASE,
                    op_type=OpKind.VECTORCYPHER_NORMALIZE_SCHEMA,
                    inputs=({"mapping": {}},),
                    outputs=(),
                    decision="insufficient_input",
                    rationale=(
                        "No operator mapping supplied — schema normalization "
                        "is operator policy and refuses to run with an empty "
                        "mapping. Populate "
                        "DreamConfig.normalize_schema_mapping with explicit "
                        "old_type -> new_type rules and re-run."
                    ),
                    started_at=started_at,
                    namespace_id=namespace_id,
                )
            ]

        entity_renames = await _collect_entity_renames(coordinator, namespace_id, mapping)
        relationship_renames = await _collect_relationship_renames(coordinator, namespace_id, mapping)
        span.set_attribute("entity_rename_count", len(entity_renames))
        span.set_attribute("relationship_rename_count", len(relationship_renames))

    outputs: dict[str, Any] = {
        "entity_renames": entity_renames,
        "relationship_renames": relationship_renames,
    }

    return [
        DreamOp(
            op_id=uuid4(),
            phase=_PHASE,
            op_type=OpKind.VECTORCYPHER_NORMALIZE_SCHEMA,
            inputs=({"mapping": mapping},),
            outputs=(outputs,),
            decision="planned",
            rationale=(
                f"Operator-supplied mapping: rename "
                f"{len(entity_renames)} entities and "
                f"{len(relationship_renames)} relationships across "
                f"{len(mapping)} type(s)."
            ),
            started_at=started_at,
            namespace_id=namespace_id,
        )
    ]


async def _collect_entity_renames(
    coordinator: StorageCoordinator,
    namespace_id: UUID,
    mapping: dict[str, str],
) -> list[dict[str, str]]:
    """Page through entities and emit one rename target per matched row."""
    renames: list[dict[str, str]] = []
    offset = 0
    while True:
        batch = await coordinator.list_entities(namespace_id, limit=_PAGE_SIZE, offset=offset)
        if not batch:
            break
        for entity in batch:
            new_type = mapping.get(entity.entity_type)
            if new_type is None or new_type == entity.entity_type:
                continue
            renames.append(
                {
                    "id": str(entity.id),
                    "old_type": entity.entity_type,
                    "new_type": new_type,
                }
            )
        if len(batch) < _PAGE_SIZE:
            break
        offset += _PAGE_SIZE
    return renames


async def _collect_relationship_renames(
    coordinator: StorageCoordinator,
    namespace_id: UUID,
    mapping: dict[str, str],
) -> list[dict[str, str]]:
    """Page through relationships and emit one rename target per matched row."""
    renames: list[dict[str, str]] = []
    offset = 0
    while True:
        batch = await coordinator.list_relationships(namespace_id, limit=_PAGE_SIZE, offset=offset)
        if not batch:
            break
        for rel in batch:
            new_type = mapping.get(rel.relationship_type)
            if new_type is None or new_type == rel.relationship_type:
                continue
            renames.append(
                {
                    "id": str(rel.id),
                    "old_type": rel.relationship_type,
                    "new_type": new_type,
                }
            )
        if len(batch) < _PAGE_SIZE:
            break
        offset += _PAGE_SIZE
    return renames


# ---------------------------------------------------------------------------
# Apply handler
# ---------------------------------------------------------------------------


async def apply_vectorcypher_normalize_schema(
    op: DreamOp,
    *,
    coordinator: StorageCoordinator,
    session: AsyncSession | None,
) -> UndoRecord:
    """Execute one planned ``vectorcypher_normalize_schema`` op.

    For each entry in ``op.outputs[0]["entity_renames"]`` /
    ``["relationship_renames"]``:

      1. Re-read the row's current type via SELECT. If it already matches
         ``new_type`` (replay), skip the UPDATE and the event emission —
         the op is idempotent.
      2. Issue ``UPDATE entities SET entity_type=:new ... WHERE id=:id``
         (or the relationships variant).
      3. Dispatch one :class:`MemoryEvent` of type ``ENTITY_UPDATED`` /
         ``RELATIONSHIP_UPDATED`` via ``coordinator.dispatch_hook``.

    Idempotent on replay: rows whose stored type already equals
    ``new_type`` are silently skipped. Caller (orchestrator) owns the
    transaction; this handler does NOT commit. The undo record carries
    only the rows that were actually rewritten — replaying the swapped
    mapping reverses the rename.

    Args:
        op: Planned op with ``outputs[0]`` containing ``entity_renames``
            and ``relationship_renames`` lists.
        coordinator: Storage coordinator — used only for hook dispatch.
        session: Orchestrator-owned async session. ``None`` falls back
            to a no-op (embedded backends without a SQL surface).

    Returns:
        :class:`UndoRecord` whose ``before`` carries the applied
        renames. Top-level keys are ``entity_renames`` and
        ``relationship_renames`` — never ``chunk_id`` (safety floor).
    """
    outputs = op.outputs[0] if op.outputs else {}
    entity_renames = list(outputs.get("entity_renames") or [])
    relationship_renames = list(outputs.get("relationship_renames") or [])
    applied_at = datetime.now(UTC)

    if session is None:
        # Embedded backend without a SQL session — nothing to write.
        return UndoRecord(
            op_id=op.op_id,
            op_type=str(op.op_type),
            before={"entity_renames": [], "relationship_renames": []},
            applied_at=applied_at,
        )

    applied_entity_renames = await _apply_entity_renames(
        session=session,
        coordinator=coordinator,
        renames=entity_renames,
        namespace_id=op.namespace_id,
        applied_at=applied_at,
    )
    applied_relationship_renames = await _apply_relationship_renames(
        session=session,
        coordinator=coordinator,
        renames=relationship_renames,
        namespace_id=op.namespace_id,
        applied_at=applied_at,
    )

    return UndoRecord(
        op_id=op.op_id,
        op_type=str(op.op_type),
        before={
            "entity_renames": applied_entity_renames,
            "relationship_renames": applied_relationship_renames,
        },
        applied_at=applied_at,
    )


async def _apply_entity_renames(
    *,
    session: AsyncSession,
    coordinator: StorageCoordinator,
    renames: list[dict[str, Any]],
    namespace_id: UUID | None,
    applied_at: datetime,
) -> list[dict[str, str]]:
    applied: list[dict[str, str]] = []
    bind_uuid = uuid_bind(session)
    for entry in renames:
        eid = UUID(str(entry["id"]))
        old_type = str(entry["old_type"])
        new_type = str(entry["new_type"])
        current = await _read_entity_type(session, eid)
        if current is None or current == new_type:
            # Already-renamed or missing — idempotent skip.
            continue
        await session.execute(
            text("UPDATE entities SET entity_type = :new_type, updated_at = :ts WHERE id = :id"),
            {"new_type": new_type, "ts": applied_at, "id": bind_uuid(eid)},
        )
        applied.append({"id": str(eid), "old_type": old_type, "new_type": new_type})
        if namespace_id is not None:
            await _emit_entity_updated(coordinator, namespace_id, eid, old_type, new_type)
    return applied


async def _apply_relationship_renames(
    *,
    session: AsyncSession,
    coordinator: StorageCoordinator,
    renames: list[dict[str, Any]],
    namespace_id: UUID | None,
    applied_at: datetime,
) -> list[dict[str, str]]:
    applied: list[dict[str, str]] = []
    bind_uuid = uuid_bind(session)
    for entry in renames:
        rid = UUID(str(entry["id"]))
        old_type = str(entry["old_type"])
        new_type = str(entry["new_type"])
        current = await _read_relationship_type(session, rid)
        if current is None or current == new_type:
            continue
        await session.execute(
            text("UPDATE relationships SET relationship_type = :new_type, updated_at = :ts WHERE id = :id"),
            {"new_type": new_type, "ts": applied_at, "id": bind_uuid(rid)},
        )
        applied.append({"id": str(rid), "old_type": old_type, "new_type": new_type})
        if namespace_id is not None:
            await _emit_relationship_updated(coordinator, namespace_id, rid, old_type, new_type)
    return applied


async def _read_entity_type(session: AsyncSession, eid: UUID) -> str | None:
    """Return the stored entity_type for ``eid`` or ``None`` if absent."""
    result = await session.execute(
        text("SELECT id, entity_type FROM entities WHERE id = :id"),
        {"id": uuid_bind(session)(eid)},
    )
    row = result.first()
    if row is None:
        return None
    return getattr(row, "entity_type", None)


async def _read_relationship_type(session: AsyncSession, rid: UUID) -> str | None:
    result = await session.execute(
        text("SELECT id, relationship_type FROM relationships WHERE id = :id"),
        {"id": uuid_bind(session)(rid)},
    )
    row = result.first()
    if row is None:
        return None
    return getattr(row, "relationship_type", None)


async def _emit_entity_updated(
    coordinator: StorageCoordinator,
    namespace_id: UUID,
    entity_id: UUID,
    old_type: str,
    new_type: str,
) -> None:
    event = MemoryEvent(
        namespace_id=namespace_id,
        event_type=EventType.ENTITY_UPDATED,
        resource_type="entity",
        resource_id=entity_id,
        data={
            "old_type": old_type,
            "new_type": new_type,
            "source": "dream.vectorcypher.normalize_schema",
        },
    )
    await coordinator.dispatch_hook(event)


async def _emit_relationship_updated(
    coordinator: StorageCoordinator,
    namespace_id: UUID,
    relationship_id: UUID,
    old_type: str,
    new_type: str,
) -> None:
    event = MemoryEvent(
        namespace_id=namespace_id,
        event_type=EventType.RELATIONSHIP_UPDATED,
        resource_type="relationship",
        resource_id=relationship_id,
        data={
            "old_type": old_type,
            "new_type": new_type,
            "source": "dream.vectorcypher.normalize_schema",
        },
    )
    await coordinator.dispatch_hook(event)


__all__ = [
    "apply_vectorcypher_normalize_schema",
    "plan_vectorcypher_normalize_schema",
]
