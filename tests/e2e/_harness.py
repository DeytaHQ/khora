"""Reusable engine layer for the deterministic e2e recall-filter suite.

``@internal``. Backend-owned. The test modules (QA-owned ``test_*.py``) drive
their per-AC assertions through this thin layer so the seeding, reconciliation,
graph-population probe, and per-engine parametrization live in one place.

The row-set proof seeds each conformance ``SeedRecord`` through the **real**
``Khora.remember()`` ingest path (not the conformance write-API seeder, whose
rows are invisible to the engine recall channel), giving each record DISTINCT
content so every document is independently recallable, an ``external_id`` equal
to the record id, and the record's filterable fields threaded as ``remember``
arguments. Recall then runs with the case filter and the surviving documents are
reconciled back to record ids by ``external_id`` — a set, never a ranked list,
so a tie in vector score can never flip an assertion.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any
from uuid import UUID

from khora import Khora
from khora.core.models.recall import RecallResult
from khora.filter.conformance import (
    ConformanceCase,
    SeedRecord,
    f_coerce_cases,
    f_dotkey_cases,
    f_exists_cases,
    f_objeq_cases,
    f_op_cases,
)
from khora.query import SearchMode

# The conformance ``SeedRecord`` fields that map to a ``Khora.remember()`` keyword.
# ``occurred_at`` / ``created_at`` / ``content_type`` are intentionally absent:
# ``remember`` exposes no keyword for them, so the F-OP families that exercise
# those keys are out of the engine row-set lane (they stay covered by the
# conformance executor suite). ``external_id`` is the reconciliation key, set
# separately from the record id.
_REMEMBER_STRING_KEYS: tuple[str, ...] = (
    "source_type",
    "source_name",
    "source_url",
    "source",
    "title",
)

# The system keys an e2e row-set case may exercise (the ``exercises[1]`` tag on
# an F-OP case). Everything ``remember`` can thread plus the user-supplied
# ``source_timestamp`` date column and the free-form ``metadata`` blob.
_ENGINE_ROWSET_KEYS: frozenset[str] = frozenset({*_REMEMBER_STRING_KEYS, "source_timestamp", "metadata.tier"})

# A generous recall ceiling so the whole seed set is returned and the filter is
# the only narrowing force (conformance seeds are a handful of records each).
_RECALL_LIMIT = 100


# --------------------------------------------------------------------------- #
# Seeding through the real ingest path.
# --------------------------------------------------------------------------- #


def _remember_kwargs(record: SeedRecord, *, entity_types: list[str], relationship_types: list[str]) -> dict[str, Any]:
    """Build the ``Khora.remember()`` kwargs that reproduce a record's filterable surface.

    Each record gets DISTINCT content (``"<anchor> record <id>"``) so its document
    is independently recallable — identical content collapses to a single recalled
    document under deterministic embeddings. The record id rides through as
    ``external_id`` (the reconciliation key). Populated string keys, the
    ``source_timestamp`` date, and the ``metadata`` blob are threaded so the filter
    has the same surface to address it does in the conformance oracle.
    """
    kwargs: dict[str, Any] = {
        "content": f"{record.content} record {record.id}",
        "external_id": record.id,
        "entity_types": entity_types,
        "relationship_types": relationship_types,
    }
    for key in _REMEMBER_STRING_KEYS:
        value = getattr(record, key)
        if value is not None:
            kwargs[key] = value
    if record.source_timestamp is not None:
        kwargs["source_timestamp"] = record.source_timestamp
    if record.metadata:
        kwargs["metadata"] = dict(record.metadata)
    return kwargs


async def seed_records(
    kb: Khora,
    records: Sequence[SeedRecord],
    namespace_id: UUID,
    *,
    entity_types: list[str] | None = None,
    relationship_types: list[str] | None = None,
) -> None:
    """Seed conformance records through ``Khora.remember()`` (the real ingest path).

    One ``remember`` call per record. ``entity_types`` / ``relationship_types``
    default to a single placeholder type each (the VectorCypher engine requires a
    non-empty list); a graph-bearing test passes its own seeded types.
    """
    etypes = entity_types if entity_types is not None else ["ENTITY"]
    rtypes = relationship_types if relationship_types is not None else ["RELATED_TO"]
    for record in records:
        await kb.remember(
            namespace=namespace_id,
            **_remember_kwargs(record, entity_types=etypes, relationship_types=rtypes),
        )


# --------------------------------------------------------------------------- #
# Reconciliation — survivors as a set of record ids.
# --------------------------------------------------------------------------- #


def reconcile(result: RecallResult) -> frozenset[str]:
    """The ``external_id`` set of documents that returned at least one chunk.

    A document is a survivor iff the recall returned a chunk belonging to it; its
    ``external_id`` is the seeded record id. Returns a SET (order-independent), so
    a vector-score tie among equally-filtered documents cannot flip an assertion.
    """
    surviving_doc_ids = {chunk.document_id for chunk in result.chunks}
    return frozenset(
        doc.external_id for doc in result.documents if doc.id in surviving_doc_ids and doc.external_id is not None
    )


async def recall_survivors(
    kb: Khora,
    case: ConformanceCase,
    namespace_id: UUID,
    *,
    mode: SearchMode = SearchMode.HYBRID,
) -> frozenset[str]:
    """Seed a case, run its filter through ``Khora.recall()``, return surviving record ids.

    Seeds every ``case.seed_records`` via :func:`seed_records`, recalls the shared
    anchor with ``case.filter`` applied (``min_similarity=0.0`` and a generous
    ``limit`` so the filter is the only narrowing force), and reconciles the result
    back to record ids by ``external_id``. The caller asserts the returned set
    equals ``case.expected_ids``.
    """
    await seed_records(kb, case.seed_records, namespace_id)
    result = await kb.recall(
        case.seed_records[0].content,
        namespace=namespace_id,
        mode=mode,
        limit=_RECALL_LIMIT,
        min_similarity=0.0,
        filter=case.filter,
    )
    return reconcile(result)


# --------------------------------------------------------------------------- #
# Graph-population probe (AC: the graph was actually written, not just fired).
# --------------------------------------------------------------------------- #


async def graph_counts(kb: Khora, namespace_id: UUID) -> tuple[int, int]:
    """``(entity_count, relationship_count)`` written to the graph for a namespace.

    Resolves the stable ``namespace_id`` to the active row id first — entities and
    relationships are keyed by the row id, so counting against the stable id reads
    zero. Reads through the storage coordinator, so it works on both the embedded
    graph backend and a live Neo4j stack.
    """
    resolved = await kb.storage.resolve_namespace(namespace_id)
    entities = await kb.storage.count_entities(resolved)
    relationships = await kb.storage.count_relationships(resolved)
    return entities, relationships


async def graph_recall(kb: Khora, namespace_id: UUID, query: str) -> RecallResult:
    """Run a no-filter ``mode=GRAPH`` recall for ``query`` and return the result.

    ``query`` must be a real seeded entity surface (e.g. an entity name): an empty
    query yields a degenerate embedding the graph-expansion path cannot resolve a
    namespace for. The caller inspects ``result.entities`` and
    ``result.engine_info`` (e.g. ``"graph_chunk_count"``) for its contribution
    assertions.
    """
    return await kb.recall(
        query,
        namespace=namespace_id,
        mode=SearchMode.GRAPH,
        limit=_RECALL_LIMIT,
        min_similarity=0.0,
    )


async def assert_graph_contributes(kb: Khora, namespace_id: UUID, query: str) -> RecallResult:
    """Pre-flight gate: a GRAPH recall for ``query`` must return entities.

    A filtered graph proof passes vacuously if the graph channel never fired (entity
    vector search returned nothing and recall short-circuited). Asserting a non-empty
    entity set here, before any filtered assertion, closes that hole. ``query`` must
    be a seeded entity surface (e.g. the entity name) — with the deterministic
    embedder only an exact surface clears entity vector search. Returns the gate's
    recall result for further assertions (e.g. ``graph_chunk_count``).
    """
    result = await graph_recall(kb, namespace_id, query)
    assert result.entities, (
        f"graph channel returned no entities for a GRAPH recall of {query!r} — the "
        "seed did not populate the graph, so every filtered assertion would be vacuous"
    )
    return result


# --------------------------------------------------------------------------- #
# Seed builders.
# --------------------------------------------------------------------------- #


def entity_seed_docs(marker: str, *, count: int = 3) -> list[str]:
    """``count`` documents that each mention ``marker`` (an entity-bearing corpus).

    Paired with ``filter_spy.plan_extraction(marker, ...)``: every document whose
    text contains ``marker`` yields the staged entities/relationships through the
    deterministic extractor, so the real graph-write path populates the graph.
    Distinct trailing text keeps each document independently recallable.
    """
    return [f"{marker} appears in document {i}." for i in range(count)]


# A small varied vocabulary so the chunker sees real token variety (repeated
# filler collapses below the split threshold; varied words do not).
_CHUNK_VOCAB: tuple[str, ...] = tuple(
    (
        "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
        "mike november oscar papa quebec romeo sierra tango uniform victor whiskey "
        "xray yankee zulu"
    ).split()
)


def multi_chunk_doc(marker: str, *, chunks: int = 3) -> str:
    """One long document whose content spans at least ``chunks`` chunks.

    Each sentence mentions ``marker`` (so a paired ``plan_extraction`` fires) and
    draws varied words so the token-count chunker actually splits — ~14 sentences
    per requested chunk clears the default 512-token chunk size with margin.
    """
    sentences = []
    for i in range(chunks * 14):
        words = " ".join(_CHUNK_VOCAB[(i + j) % len(_CHUNK_VOCAB)] for j in range(12))
        sentences.append(f"{marker} {words} sentence {i}.")
    return " ".join(sentences)


# --------------------------------------------------------------------------- #
# Corpus selection — the F-OP cases the engine row-set lane can seed.
# --------------------------------------------------------------------------- #


def engine_rowset_cases(cases: Sequence[ConformanceCase]) -> list[ConformanceCase]:
    """The subset of F-OP cases the engine row-set lane can seed via ``remember``.

    Keeps cases whose exercised system key is threadable through ``Khora.remember()``
    (the five string document keys, ``source_timestamp``, and the ``metadata.tier``
    representative) and whose seed has no duplicate non-NULL ``external_id`` — because
    the engine lane reconciles by ``external_id`` and ``documents`` enforces UNIQUE
    ``(namespace_id, external_id)``. Drops the ``occurred_at`` / ``created_at`` /
    ``content_type`` and ``external_id`` families (no ``remember`` keyword / recon-key
    conflict); those stay covered by the conformance executor suite.
    """
    selected: list[ConformanceCase] = []
    for case in cases:
        key = case.exercises[1] if len(case.exercises) > 1 else ""
        if key not in _ENGINE_ROWSET_KEYS:
            continue
        if case.expected_ids is None:
            continue
        if _has_duplicate_external_id(case.seed_records):
            continue
        if _is_exact_timestamp_match(case):
            continue
        selected.append(case)
    return selected


def _is_exact_timestamp_match(case: ConformanceCase) -> bool:
    """Whether a case asserts a positive exact-instant match on ``source_timestamp``.

    The engine recall path stores ``source_timestamp`` and compares it with boundary
    semantics where a filter operand equal to the stored instant does not re-select
    that row, so a positive ``$eq`` (or a populated ``$in`` whose elements are exact
    stored instants) reconciles to the empty set through ``Khora.recall`` even though
    the conformance oracle keeps the row. Range comparisons and the negations
    (``$ne`` / ``$nin``) are unaffected and stay in the lane; the empty-set ``$in:[]``
    variant ends ``-in-empty`` and is not matched here. These exact ``source_timestamp``
    equalities remain covered by the conformance executor suite.
    """
    if case.exercises[1:2] != ("source_timestamp",):
        return False
    return case.id.endswith(("-eq", "-in"))


def _has_duplicate_external_id(records: Sequence[SeedRecord]) -> bool:
    """Whether two records share a non-NULL ``external_id`` (we use the record id as one).

    The record ``id`` is the reconciliation key, and ``documents`` enforces UNIQUE
    ``(namespace_id, external_id)``. Record ids are unique by construction, so this
    is a guard against a future seed that reused one.
    """
    ids = [r.id for r in records]
    return len(ids) != len(set(ids))


def rowset_cases(backend: str, *, include_system_keys: bool) -> list[ConformanceCase]:
    """The curated row-set cases for one engine lane (the single shared selector).

    Always includes the dotted-``metadata``-path families (``F-COERCE`` / ``F-OBJEQ``
    / ``F-DOTKEY``) that target ``backend``; whole-metadata-blob equality
    (``exercises[1] == "metadata"``) is dropped because the embedded JSON path can't
    narrow on it.

    ``include_system_keys`` is for the **live** lanes only (``vectorcypher`` /
    ``chronicle``): their chunk row carries the denormalized document system keys, so
    the engine recall path narrows on them — unlike the embedded ``chunks`` row. When
    set, the selector also includes the ``remember``-threadable system-key ``F-OP``
    families (via :func:`engine_rowset_cases`: the five string keys + ``source_timestamp``
    + the ``metadata.tier`` representative, minus exact-instant timestamp equalities)
    and the two ``source_name`` ``F-EXISTS`` presence states. This is the only path on
    which an e2e case filters on a denormalized document column.

    Cases are deduplicated by id and any seed with a duplicate ``external_id`` (the
    reconciliation key under UNIQUE(namespace_id, external_id)) is dropped.
    """
    candidates: list[ConformanceCase] = []
    for family in (f_coerce_cases, f_objeq_cases, f_dotkey_cases):
        candidates.extend(c for c in family() if backend in c.backends and c.exercises[1] != "metadata")
    if include_system_keys:
        candidates.extend(c for c in engine_rowset_cases(f_op_cases()) if backend in c.backends)
        candidates.extend(c for c in f_exists_cases() if backend in c.backends and c.exercises[1] == "source_name")

    selected: list[ConformanceCase] = []
    seen: set[str] = set()
    for case in candidates:
        if case.id in seen or _has_duplicate_external_id(case.seed_records):
            continue
        seen.add(case.id)
        selected.append(case)
    return selected


# --------------------------------------------------------------------------- #
# Reachability guard — the live-lane modules self-skip on this.
# --------------------------------------------------------------------------- #


def _pg_reachable() -> bool:
    """Whether the compose Postgres is reachable (the live-lane self-skip guard).

    Kept import-light so the ``pytest.mark.skipif`` in each live module can call it
    at collection time without importing the conftest fixtures.
    """
    import socket
    from urllib.parse import urlparse

    url = os.environ.get("KHORA_DATABASE_URL", "postgresql://khora:khora@localhost:5434/khora")
    parsed = urlparse(url.replace("+asyncpg", ""))
    try:
        with socket.create_connection((parsed.hostname or "localhost", parsed.port or 5432), timeout=2):
            return True
    except OSError:
        return False
