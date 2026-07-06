"""Replace-path graph-mirror payloads for the #1430 reconciler.

``StorageCoordinator.replace_document_extraction`` commits Postgres (new
chunks + document COMPLETED) first, then mirrors the change to the graph
backend (entity retire / remap / strip / upsert) outside any transaction.
A graph failure in that window leaves a durable PG/graph divergence that
#884 made observable (a degradation on ``RememberResult``) but not
recoverable - the split persisted until the next successful replace.

This module is the replace-path equivalent of ``khora.dream.graph_mirror``
(#1272): it serializes the exact graph plan the failed mirror was about to
apply into a JSON payload persisted on ``documents.graph_mirror_pending``
(migration 051), and replays it later. The replay is idempotent:

* retire verbs match by id and skip already-retired rows;
* remap / strip verbs remove ids that are simply absent on a re-run;
* entity upsert MERGEs on ``(namespace_id, name, entity_type)``;
* relationship create dedup-merges per #1320.

So a payload can be applied any number of times and converges the graph to
the state PG already committed.
"""

from __future__ import annotations

from dataclasses import fields
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from khora.core.models import Entity, Relationship

if TYPE_CHECKING:
    from khora.storage.coordinator import StorageCoordinator

REPLACE_MIRROR_PAYLOAD_VERSION = 1

# Entity / Relationship dataclass fields persisted in the payload.
# ``source_documents`` is a read-time projection (never persisted) and is
# deliberately excluded.
_ENTITY_UUID_FIELDS = frozenset({"id", "namespace_id"})
_ENTITY_UUID_LIST_FIELDS = frozenset({"source_document_ids", "source_chunk_ids"})
_ENTITY_DATETIME_FIELDS = frozenset({"valid_from", "valid_until", "created_at", "updated_at"})
_EXCLUDED_FIELDS = frozenset({"source_documents"})

_REL_UUID_FIELDS = frozenset({"id", "namespace_id", "source_entity_id", "target_entity_id"})


def _to_json_value(value: Any) -> Any:
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, list):
        return [_to_json_value(v) for v in value]
    return value


def _model_to_json(obj: Entity | Relationship) -> dict[str, Any]:
    return {
        f.name: _to_json_value(getattr(obj, f.name))
        for f in fields(obj)
        if f.name not in _EXCLUDED_FIELDS and not f.name.startswith("_")
    }


def _coerce(raw: dict[str, Any], *, uuid_fields: frozenset[str], datetime_fields: frozenset[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if value is None:
            out[key] = None
        elif key in uuid_fields:
            out[key] = UUID(str(value))
        elif key in _ENTITY_UUID_LIST_FIELDS:
            out[key] = [UUID(str(v)) for v in value]
        elif key in datetime_fields:
            out[key] = datetime.fromisoformat(str(value))
        else:
            out[key] = value
    return out


def _entity_from_json(raw: dict[str, Any]) -> Entity:
    known = {f.name for f in fields(Entity) if f.init}
    data = _coerce(
        {k: v for k, v in raw.items() if k in known},
        uuid_fields=_ENTITY_UUID_FIELDS,
        datetime_fields=_ENTITY_DATETIME_FIELDS,
    )
    return Entity(**data)


def _relationship_from_json(raw: dict[str, Any]) -> Relationship:
    known = {f.name for f in fields(Relationship) if f.init}
    data = _coerce(
        {k: v for k, v in raw.items() if k in known},
        uuid_fields=_REL_UUID_FIELDS,
        datetime_fields=_ENTITY_DATETIME_FIELDS,
    )
    return Relationship(**data)


def build_replace_mirror_payload(
    *,
    old_document_id: UUID,
    entity_retirement_rows: list[dict[str, str]],
    relationship_retirement_rows: list[dict[str, Any]],
    entity_survivor_remap_rows: list[dict[str, str]],
    relationship_survivor_remap_rows: list[dict[str, str]],
    entity_survivor_strip_ids: list[UUID],
    relationship_survivor_strip_ids: list[UUID],
    net_new_entities: list[Entity],
    net_new_relationships: list[Relationship],
    exception: BaseException,
) -> dict[str, Any]:
    """Serialize the graph plan for ``documents.graph_mirror_pending``.

    The reconciler replays from this JSON payload rather than recomputing:
    the net-new entities/relationships came from an LLM extraction that is
    not durably stored anywhere else once the mirror fails, and the
    retire / remap / strip rows were computed against the pre-replace graph
    state that later mutations would obscure.
    """
    return {
        "version": REPLACE_MIRROR_PAYLOAD_VERSION,
        "old_document_id": str(old_document_id),
        "failed_at": datetime.now(UTC).isoformat(),
        "exception": type(exception).__name__,
        # Already JSON-safe (all str) - persisted verbatim.
        "entity_retirement_rows": list(entity_retirement_rows),
        "relationship_retirement_rows": [
            {
                "relationship_id": str(row["relationship_id"]),
                "old_doc_id": str(row["old_doc_id"]),
                "retired_at": row["retired_at"].isoformat()
                if isinstance(row["retired_at"], datetime)
                else str(row["retired_at"]),
            }
            for row in relationship_retirement_rows
        ],
        "entity_survivor_remap_rows": list(entity_survivor_remap_rows),
        "relationship_survivor_remap_rows": list(relationship_survivor_remap_rows),
        "entity_survivor_strip_ids": [str(u) for u in entity_survivor_strip_ids],
        "relationship_survivor_strip_ids": [str(u) for u in relationship_survivor_strip_ids],
        "net_new_entities": [_model_to_json(e) for e in net_new_entities],
        "net_new_relationships": [_model_to_json(r) for r in net_new_relationships],
    }


async def apply_replace_mirror_payload(
    coordinator: StorageCoordinator,
    payload: dict[str, Any],
    *,
    namespace_id: UUID,
) -> dict[str, int]:
    """Reconciler entry: replay a persisted replace graph-mirror payload.

    Runs the same verbs in the same order as the original graph phase of
    ``replace_document_extraction`` (retire -> remap -> strip -> upsert ->
    create). Entity upsert and relationship create go through the
    coordinator so the vector-side entity rows heal alongside the graph
    (#868 ordering). Raises on the first failing verb - the caller keeps
    the marker queued and retries later.
    """
    graph = coordinator._graph
    if graph is None:
        raise RuntimeError("Graph backend not configured")

    # Boundary validation: the payload comes back from the database and could
    # have been written by a newer khora (e.g. after a rollback). Refuse to
    # run graph mutations against an unknown schema - the marker stays queued
    # and surfaces as a reconcile degradation.
    version = payload.get("version")
    if version != REPLACE_MIRROR_PAYLOAD_VERSION:
        raise ValueError(f"Unsupported replace graph-mirror payload version: {version!r}")

    old_document_id = UUID(str(payload["old_document_id"]))

    entities_retired = 0
    entity_retirement_rows = payload.get("entity_retirement_rows") or []
    if entity_retirement_rows:
        entities_retired = await graph.retire_orphaned_entities_batch(entity_retirement_rows)

    relationships_retired = 0
    relationship_retirement_rows = [
        {
            "relationship_id": UUID(str(row["relationship_id"])),
            "old_doc_id": UUID(str(row["old_doc_id"])),
            "retired_at": datetime.fromisoformat(str(row["retired_at"])),
        }
        for row in payload.get("relationship_retirement_rows") or []
    ]
    if relationship_retirement_rows:
        relationships_retired = await graph.retire_orphaned_relationships_batch(
            relationship_retirement_rows, namespace_id=namespace_id
        )

    entity_survivor_remap_rows = payload.get("entity_survivor_remap_rows") or []
    relationship_survivor_remap_rows = payload.get("relationship_survivor_remap_rows") or []
    if entity_survivor_remap_rows or relationship_survivor_remap_rows:
        await graph.remap_source_document_ids_batch(
            entity_survivors=entity_survivor_remap_rows,
            relationship_survivors=relationship_survivor_remap_rows,
            namespace_id=namespace_id,
        )

    entity_strip_ids = [UUID(str(u)) for u in payload.get("entity_survivor_strip_ids") or []]
    if entity_strip_ids:
        await graph.remove_document_from_entity_sources_batch(entity_strip_ids, old_document_id, namespace_id)
    relationship_strip_ids = [UUID(str(u)) for u in payload.get("relationship_survivor_strip_ids") or []]
    if relationship_strip_ids:
        await graph.remove_document_from_relationship_sources_batch(
            relationship_strip_ids, old_document_id, namespace_id
        )

    net_new_entities = [_entity_from_json(raw) for raw in payload.get("net_new_entities") or []]
    entities_upserted = 0
    if net_new_entities:
        entities_upserted = len(await coordinator.upsert_entities_batch(namespace_id, net_new_entities))

    net_new_relationships = [_relationship_from_json(raw) for raw in payload.get("net_new_relationships") or []]
    relationships_created = 0
    if net_new_relationships:
        relationships_created = len(await coordinator.create_relationships_batch(net_new_relationships))

    return {
        "entities_retired": entities_retired,
        "relationships_retired": relationships_retired,
        "entities_upserted": entities_upserted,
        "relationships_created": relationships_created,
    }
