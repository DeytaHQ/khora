"""Shared forget-cascade cleanup for entities and relationships.

When a document is forgotten, the entities and relationships extracted from
it must be cleaned up across whatever store(s) actually hold them. Pre-#923
this was anchored on the Neo4j-only ``fetch_document_extraction_state`` and
silently no-op'd on every non-Neo4j stack (SurrealDB, Memgraph, Neptune, AGE,
sqlite_lance graph), leaving orphaned entities behind while reporting success.

The fix re-anchors cleanup on ``source_document_ids`` refcounting, which every
backend that stores entities carries:

- For each entity/relationship whose ``source_document_ids`` lists the
  forgotten document, strip that document id from the array.
- If the array becomes empty (no other document references it) the
  entity/relationship is an orphan and is hard-deleted.
- Shared entities (still referenced by another document) survive; only their
  source list shrinks. This mirrors the Neo4j path's ``source_document_count``
  semantics - never hard-delete a still-referenced entity.

Routing: entities live in pgvector's ``entities`` table on pg-backed stacks
(with or without a Neo4j graph) and in the sqlite_lance / SurrealDB / Memgraph
/ Neptune / AGE graph adapter tables on the other stacks. We pick whichever
store exposes the list + delete + source-strip primitives as the authoritative
store, compute the orphan/survivor split there, then opportunistically mirror
the deletes to the other store (e.g. Neo4j nodes alongside pgvector rows) so
the graph stays consistent.

Selecting the candidate set: when the primary store exposes a
source-document-scoped lookup (``list_entities_by_source_document`` /
``list_relationships_by_source_document`` on pgvector and Neo4j) we prefer it -
it returns exactly the records that reference the forgotten document. Backends
without that lookup fall back to a bounded full-namespace scan + Python filter;
if that scan hits ``_SCAN_LIMIT`` we emit a ``Degradation`` so a possible
un-cleaned-orphan tail is never a silent miss.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from loguru import logger

from khora.core.diagnostics import Degradation
from khora.telemetry.metrics import metric_counter

_FORGET_DEGRADED_COUNTER = metric_counter(
    "khora.forget.cascade.degraded_total",
    description=(
        "Forget-cascade entity/relationship cleanups that could not run because "
        "no configured store exposed the required primitives. Labels: reason. "
        "No namespace label (cardinality rule)."
    ),
)

# Cap how many rows we scan per namespace when splitting orphans from
# survivors on backends without a source-document-scoped lookup. Mirrors the
# list_* limits used by the coordinator fallback path.
_SCAN_LIMIT = 100_000


def _can_list_and_delete(store: Any) -> bool:
    return store is not None and all(
        callable(getattr(store, name, None))
        for name in ("list_entities", "list_relationships", "delete_entities_batch", "delete_relationships_batch")
    )


async def cascade_forget_extraction(
    *,
    graph: Any,
    vector: Any,
    document_id: UUID,
    namespace_id: UUID,
    engine: str,
) -> list[Degradation]:
    """Drop / decrement entities and relationships extracted from a document.

    Returns a list of ``Degradation`` records. An empty list means cleanup
    ran (or there was simply nothing to clean). A non-empty list means the
    cleanup could not be performed in full - the caller should treat the
    forget as partially degraded rather than silently successful.
    """
    # Prefer the vector store (pgvector) as the authoritative refcount holder
    # on graph-backed stacks; fall back to the graph store (sqlite_lance,
    # SurrealDB, Memgraph, Neptune, AGE) when the vector adapter has no entity
    # tables.
    if _can_list_and_delete(vector):
        primary, mirror = vector, graph
    elif _can_list_and_delete(graph):
        primary, mirror = graph, vector
    else:
        _FORGET_DEGRADED_COUNTER.add(1, {"reason": "no_store_with_primitives"})
        logger.warning(
            "{}: forget cascade could not clean entities for document {} - "
            "no configured store exposes list/delete primitives",
            engine,
            document_id,
        )
        return [
            Degradation(
                component="forget_cascade",
                reason="no_store_with_primitives",
                detail=(
                    "Neither the vector nor the graph backend exposes the "
                    "entity/relationship cleanup primitives; orphan entities "
                    "may be left behind."
                ),
            )
        ]

    degradations: list[Degradation] = []

    entities, ent_deg = await _candidate_entities(primary, document_id, namespace_id, engine)
    relationships, rel_deg = await _candidate_relationships(primary, document_id, namespace_id, engine)
    degradations.extend(ent_deg)
    degradations.extend(rel_deg)

    orphan_ent_ids = [e.id for e in entities if _is_orphan(e, document_id)]
    survive_ent_ids = [e.id for e in entities if _is_survivor(e, document_id)]
    orphan_rel_ids = [r.id for r in relationships if _is_orphan(r, document_id)]
    survive_rel_ids = [r.id for r in relationships if _is_survivor(r, document_id)]

    if orphan_ent_ids:
        await primary.delete_entities_batch(orphan_ent_ids, namespace_id=namespace_id)
        await _delete_entities(mirror, orphan_ent_ids, namespace_id)
    if orphan_rel_ids:
        await primary.delete_relationships_batch(orphan_rel_ids, namespace_id=namespace_id)
        await _delete_relationships(mirror, orphan_rel_ids, namespace_id)

    if survive_ent_ids:
        degradations.extend(await _strip_entities(primary, survive_ent_ids, document_id, namespace_id, engine))
        degradations.extend(await _strip_entities(mirror, survive_ent_ids, document_id, namespace_id, engine))
    if survive_rel_ids:
        degradations.extend(await _strip_relationships(primary, survive_rel_ids, document_id, namespace_id, engine))
        degradations.extend(await _strip_relationships(mirror, survive_rel_ids, document_id, namespace_id, engine))

    logger.debug(
        "{}: forget cascade for document {} removed {} orphan entities / {} orphan relationships, "
        "stripped {} surviving entities / {} surviving relationships",
        engine,
        document_id,
        len(orphan_ent_ids),
        len(orphan_rel_ids),
        len(survive_ent_ids),
        len(survive_rel_ids),
    )
    return degradations


async def _candidate_entities(
    primary: Any, document_id: UUID, namespace_id: UUID, engine: str
) -> tuple[list[Any], list[Degradation]]:
    """Entities referencing ``document_id``, preferring a source-scoped lookup."""
    fn = getattr(primary, "list_entities_by_source_document", None)
    if callable(fn):
        return await fn(namespace_id, document_id), []
    results = await primary.list_entities(namespace_id, limit=_SCAN_LIMIT)
    return results, _scan_cap_degradation(results, "entities", document_id, engine)


async def _candidate_relationships(
    primary: Any, document_id: UUID, namespace_id: UUID, engine: str
) -> tuple[list[Any], list[Degradation]]:
    """Relationships referencing ``document_id``, preferring a source-scoped lookup."""
    fn = getattr(primary, "list_relationships_by_source_document", None)
    if callable(fn):
        return await fn(namespace_id, document_id), []
    results = await primary.list_relationships(namespace_id, limit=_SCAN_LIMIT)
    return results, _scan_cap_degradation(results, "relationships", document_id, engine)


def _scan_cap_degradation(results: list[Any], kind: str, document_id: UUID, engine: str) -> list[Degradation]:
    if len(results) < _SCAN_LIMIT:
        return []
    _FORGET_DEGRADED_COUNTER.add(1, {"reason": "scan_cap_hit"})
    logger.warning(
        "{}: forget cascade {} scan hit the {}-row cap for document {} - orphans beyond the cap may be left behind",
        engine,
        kind,
        _SCAN_LIMIT,
        document_id,
    )
    return [
        Degradation(
            component="forget_cascade",
            reason="scan_cap_hit",
            detail=(
                f"The {kind} scan hit the {_SCAN_LIMIT}-row cap; the primary store has no "
                "source-document-scoped lookup so orphans beyond the cap may be left behind."
            ),
        )
    ]


def _is_orphan(record: Any, document_id: UUID) -> bool:
    sources = record.source_document_ids or []
    return document_id in sources and len(sources) == 1


def _is_survivor(record: Any, document_id: UUID) -> bool:
    sources = record.source_document_ids or []
    return document_id in sources and len(sources) > 1


# The cleanup primitives have three method-name / signature shapes across
# stores; these adapters dispatch to whichever the store exposes:
#   - pgvector:        delete_entities_batch / remove_document_from_entity_sources(ids, doc, ns)
#   - Neo4j:           delete_entities_batch / remove_document_from_entity_sources_batch(ids, doc, ns)
#   - GraphBackendBase fallback (SurrealDB/Memgraph/Neptune/AGE/sqlite_lance):
#                      delete_entities_batch / strip_document_from_entities(ids, doc, namespace_id=ns)


async def _delete_entities(store: Any, ids: list[UUID], namespace_id: UUID) -> None:
    fn = getattr(store, "delete_entities_batch", None)
    if callable(fn):
        await fn(ids, namespace_id=namespace_id)


async def _delete_relationships(store: Any, ids: list[UUID], namespace_id: UUID) -> None:
    fn = getattr(store, "delete_relationships_batch", None)
    if callable(fn):
        await fn(ids, namespace_id=namespace_id)


async def _strip_entities(
    store: Any, ids: list[UUID], document_id: UUID, namespace_id: UUID, engine: str
) -> list[Degradation]:
    # Prefer native bulk methods (pgvector / Neo4j) over the per-record
    # GraphBackendBase fallback, which every graph backend also inherits.
    if store is None:
        return []
    try:
        if callable(getattr(store, "remove_document_from_entity_sources_batch", None)):
            await store.remove_document_from_entity_sources_batch(ids, document_id, namespace_id)
        elif callable(getattr(store, "remove_document_from_entity_sources", None)):
            await store.remove_document_from_entity_sources(ids, document_id, namespace_id)
        elif callable(getattr(store, "strip_document_from_entities", None)):
            await store.strip_document_from_entities(ids, document_id, namespace_id=namespace_id)
        elif callable(getattr(store, "delete_entities_batch", None)):
            # Store can list+delete but has no source-strip primitive. Do NOT
            # silently skip - surface a degradation so the survivor's stale
            # source_document_ids is a tracked gap, not an invisible one.
            return _strip_unsupported("entities", document_id, engine)
    except Exception as exc:  # noqa: BLE001
        return _strip_failed("entities", document_id, engine, exc)
    return []


async def _strip_relationships(
    store: Any, ids: list[UUID], document_id: UUID, namespace_id: UUID, engine: str
) -> list[Degradation]:
    if store is None:
        return []
    try:
        if callable(getattr(store, "remove_document_from_relationship_sources_batch", None)):
            await store.remove_document_from_relationship_sources_batch(ids, document_id, namespace_id)
        elif callable(getattr(store, "remove_document_from_relationship_sources", None)):
            await store.remove_document_from_relationship_sources(ids, document_id, namespace_id)
        elif callable(getattr(store, "strip_document_from_relationships", None)):
            await store.strip_document_from_relationships(ids, document_id, namespace_id=namespace_id)
        elif callable(getattr(store, "delete_relationships_batch", None)):
            return _strip_unsupported("relationships", document_id, engine)
    except Exception as exc:  # noqa: BLE001
        return _strip_failed("relationships", document_id, engine, exc)
    return []


def _strip_unsupported(kind: str, document_id: UUID, engine: str) -> list[Degradation]:
    _FORGET_DEGRADED_COUNTER.add(1, {"reason": "strip_unsupported"})
    logger.warning(
        "{}: forget cascade cannot strip document {} from survivor {} - "
        "store exposes list/delete but no source-strip primitive",
        engine,
        document_id,
        kind,
    )
    return [
        Degradation(
            component="forget_cascade",
            reason="strip_unsupported",
            detail=(
                f"Store can delete {kind} but exposes no source-strip primitive; "
                "survivors keep a stale source_document_ids entry for the forgotten document."
            ),
        )
    ]


def _strip_failed(kind: str, document_id: UUID, engine: str, exc: Exception) -> list[Degradation]:
    _FORGET_DEGRADED_COUNTER.add(1, {"reason": "strip_failed"})
    logger.error(
        "{}: forget cascade failed to strip document {} from survivor {}: {}",
        engine,
        document_id,
        kind,
        exc,
        exc_info=True,
    )
    return [
        Degradation(
            component="forget_cascade",
            reason="strip_failed",
            detail=f"Stripping document {document_id} from survivor {kind} raised.",
            exception=str(exc),
        )
    ]
