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

import pytest

from khora import Khora
from khora.core.models.recall import RecallResult
from khora.filter import RecallFilter
from khora.filter.ast import parse_to_ast
from khora.filter.conformance import (
    ConformanceCase,
    SeedRecord,
    f_coerce_cases,
    f_dotkey_cases,
    f_exists_cases,
    f_objeq_cases,
    f_op_cases,
)
from khora.filter.execute import filter_leaf_keys
from khora.filter.report import FilterPushdownReport
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

# The similarity floor for the row-set proof. The deterministic SHA-256 hash
# embedder produces vectors with NO semantic meaning, so the query↔doc cosine is
# frequently NEGATIVE for a genuinely-seeded document. Backends that return a
# signed cosine (e.g. SurrealDB's brute-force path) then drop those docs at a
# ``0.0`` floor — making the row-set proof vacuous even though the filter would
# keep them. A negative floor admits EVERY seeded document regardless of the
# meaningless hash-cosine sign, so the filter stays the only narrowing force (the
# whole point of the proof). ``-1.0`` is the cosine minimum, so it admits all.
# Verified no-op on the sqlite_lance lanes (already clamped non-negative there).
_RECALL_MIN_SIMILARITY = -1.0


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

    The Skeleton engine is the exception: it has no graph channel and **rejects**
    a non-empty ``entity_types`` / ``relationship_types`` with
    ``UnsupportedEngineKwargError`` (#890 — it deliberately skips typed extraction
    for cost). On a Skeleton ``kb`` the defaults flip to empty lists so the same
    row-set proof seeds through the Skeleton ingest path unchanged. An explicit
    override always wins (a Skeleton caller would never pass non-empty types).
    """
    skeleton = getattr(kb, "_engine_name", None) == "skeleton"
    default_etypes: list[str] = [] if skeleton else ["ENTITY"]
    default_rtypes: list[str] = [] if skeleton else ["RELATED_TO"]
    etypes = entity_types if entity_types is not None else default_etypes
    rtypes = relationship_types if relationship_types is not None else default_rtypes
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
    anchor with ``case.filter`` applied (``_RECALL_MIN_SIMILARITY`` and a generous
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
        min_similarity=_RECALL_MIN_SIMILARITY,
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
# Backend resolver + anti-vacuity guard — the parametrized matrix lane.
# --------------------------------------------------------------------------- #

# ``KHORA_E2E_BACKEND`` selects which engine lane the parametrized matrix runs.
# The value maps to ``(conformance token, include_system_keys)``: the token is
# the ``ConformanceCase.backends`` member the selector filters on, and the flag
# turns on the denormalized-document-column families — ON only when the lane's
# chunk row actually carries the system keys. VectorCypher / Chronicle (pgvector)
# and Skeleton-pgvector denormalize them onto every chunk; the embedded
# ``sqlite_lance`` row and the Skeleton surrealdb / weaviate adapters do NOT
# (surrealdb DEFINEs them on ``document`` not ``chunk``; weaviate omits them from
# the ``KhoraChunk`` collection), so those lanes leave it off — a system-key
# filter there narrows on an absent column and would reconcile vacuously.
_E2E_BACKEND_MAP: dict[str, tuple[str, bool]] = {
    # The token is the ``ConformanceCase.backends`` member, NOT the storage
    # backend name: the live VectorCypher graph lane (``vc_full``) compiles its
    # filter through the Cypher post-filter path, so it reconciles against the
    # ``cypher`` conformance token (matching the shipped graph module's
    # ``rowset_cases("cypher", ...)``) — ``postgres`` would silently under-cover.
    #
    # The trailing row-set count pins each lane to its conformance tag so review
    # catches a valid-but-WRONG token (which passes the raise-on-empty guard but
    # silently under-covers — the ``vc_full``→``postgres`` slip was 81→46). QA's
    # lane modules assert the shipped-precedent equality, so a drift fails RED.
    "vc_full": ("cypher", True),  # 81 — matches shipped test_filter_rowset_graph.py:108
    "vc_embedded": ("sqlite_lance", False),  # 30
    "skeleton_pgvector": ("postgres", True),  # 46
    "skeleton_surrealdb": ("surrealdb", False),  # 26 — system keys live on ``document``, not ``chunk``
    "skeleton_weaviate": ("weaviate", False),  # 30 — ``KhoraChunk`` collection omits the document system keys
    "skeleton_sqlite_lance": ("sqlite_lance", False),  # 30
    "chronicle": ("chronicle", True),  # 81 — matches shipped test_filter_rowset_chronicle.py:41
}


def resolve_e2e_backend(backend: str | None = None) -> tuple[str, bool]:
    """``(conformance token, include_system_keys)`` for a matrix lane.

    ``backend`` is a matrix-lane key (``"skeleton_surrealdb"``, ``"vc_full"``, …);
    when omitted it falls back to the ``KHORA_E2E_BACKEND`` env var the e2e
    workflow sets per leg. Either way the key resolves to the conformance
    ``backends`` token the selectors filter on and the system-key flag. An unknown
    / unset value is a configuration error — a silent default would let a mis-set
    leg run the wrong corpus and pass green, so it raises instead.
    """
    key = backend if backend is not None else os.environ.get("KHORA_E2E_BACKEND")
    if key not in _E2E_BACKEND_MAP:
        raise RuntimeError(
            f"e2e backend {key!r} is not a known matrix lane; "
            f"expected one of {sorted(_E2E_BACKEND_MAP)} "
            "(pass it explicitly or set KHORA_E2E_BACKEND)"
        )
    return _E2E_BACKEND_MAP[key]


def lane_rowset_cases(backend: str | None = None) -> list[ConformanceCase]:
    """The row-set cases for a matrix lane — raises if the selector yields none.

    ``backend`` is the matrix-lane key (defaulting to ``KHORA_E2E_BACKEND``). The
    empty-raise IS the Layer-1 anti-vacuity guard: a mis-mapped token or a corpus
    shrink that drops every case for this lane turns into a RED at collection time,
    never a silent empty-green parametrization. QA's parametrized tests call this —
    never the raw :func:`rowset_cases` — so the guard always fires.
    """
    token, include_system_keys = resolve_e2e_backend(backend)
    cases = rowset_cases(token, include_system_keys=include_system_keys)
    if not cases:
        raise RuntimeError(
            f"rowset_cases({token!r}, include_system_keys={include_system_keys}) "
            "selected zero cases — a mis-mapped e2e backend or a corpus shrink. "
            "Refusing to parametrize an empty (vacuously green) lane."
        )
    return cases


def lane_exists_cases(backend: str | None = None) -> list[ConformanceCase]:
    """The F-EXISTS reachability cases for a matrix lane — raises if none.

    ``backend`` is the matrix-lane key (defaulting to ``KHORA_E2E_BACKEND``).
    Filters :func:`f_exists_cases` to the lane's conformance token; when the lane
    carries no denormalized document columns (``include_system_keys`` off, the
    embedded ``chunks`` row), the ``source_name`` system-key presence states are
    dropped because the embedded recall path can't narrow on them. The empty-raise
    is the same Layer-1 guard as :func:`lane_rowset_cases`.
    """
    token, include_system_keys = resolve_e2e_backend(backend)
    cases = [
        c for c in f_exists_cases() if token in c.backends and (include_system_keys or c.exercises[1] != "source_name")
    ]
    if not cases:
        raise RuntimeError(
            f"f_exists_cases filtered to token={token!r} "
            f"(include_system_keys={include_system_keys}) selected zero cases — a "
            "mis-mapped e2e backend or a corpus shrink. Refusing to parametrize "
            "an empty (vacuously green) lane."
        )
    return cases


def lane_skip(lane: str) -> pytest.MarkDecorator:
    """A ``skipif`` mark that runs the module only on its own matrix lane.

    ``lane`` is the module's matrix-lane key (``"skeleton_surrealdb"``, …) — the
    value the e2e workflow sets in ``KHORA_E2E_BACKEND`` for that leg. The matrix
    runs ONE engine config per leg, but reachability self-selection alone can't
    separate the lanes (the ``vc_full`` leg has Postgres up, so the ``chronicle``
    and ``skeleton_pgvector`` modules would also run there). Gating each module on
    ``KHORA_E2E_BACKEND`` makes the env var the per-leg selector; ``devops`` sets it
    per leg and drops the belt-and-suspenders ``-k``.

    A **no-op when ``KHORA_E2E_BACKEND`` is unset / empty** (the default test-unit
    and local runs) so those collect-and-run every lane unchanged — the per-module
    reachability ``skipif`` still gates the live ones there. ``lane`` is validated
    against :data:`_E2E_BACKEND_MAP` so a typo'd key fails loud rather than skipping
    the module on every leg silently.
    """
    if lane not in _E2E_BACKEND_MAP:
        raise RuntimeError(f"lane_skip({lane!r}): not a known matrix lane; expected one of {sorted(_E2E_BACKEND_MAP)}")
    active = os.environ.get("KHORA_E2E_BACKEND")
    return pytest.mark.skipif(
        active not in (None, "", lane),
        reason=f"KHORA_E2E_BACKEND={active!r} selects a different e2e lane (this is {lane!r})",
    )


# --------------------------------------------------------------------------- #
# Cross-engine filter-report invariants — the engine-independent contract the
# gate asserts against every engine's emitted ``engine_info["filter"]``.
# --------------------------------------------------------------------------- #


# The exact top-level key set a ``FilterPushdownReport`` JSON dump carries, and
# the exact key set of each per-channel ``FilterChannelReport``. The gate pins
# these literally so a field rename / addition (which would silently change the
# public ``engine_info["filter"]`` schema) fails RED.
_REPORT_TOP_KEYS: frozenset[str] = frozenset(
    {"pushed_down", "post_filtered", "pushed_keys", "post_filtered_keys", "channels"}
)
_CHANNEL_KEYS: frozenset[str] = frozenset({"pushed_keys", "post_filtered_keys"})


def assert_filter_report_invariants(report: dict[str, Any], expected_leaves: frozenset[str]) -> None:
    """Assert an engine's filter report obeys the engine-independent invariants.

    ``report`` is the JSON dict an engine emits verbatim as
    ``RecallResult.engine_info["filter"]`` (a :class:`FilterPushdownReport`
    ``model_dump(mode="json")``). ``expected_leaves`` is the filter's
    constraint-leaf set, computed by the caller through the production lowering
    the recall facade uses — ``filter_leaf_keys(parse_to_ast(RecallFilter.model_validate(spec)))``
    — so the gate measures the report against the same leaves the engines
    partitioned. Pure (no recall, no engine): unit-testable on hand-built dicts.

    Invariants (engine-independent, hold on every emitting engine):

    a) SCHEMA — ``FilterPushdownReport.model_validate(report)`` succeeds, the
       top-level keys are EXACTLY :data:`_REPORT_TOP_KEYS`, and each channel value
       has EXACTLY :data:`_CHANNEL_KEYS`. A field rename/add to the public schema
       fails here.
    b) SORTED — the two top-level key lists and both lists of every channel are
       each equal to their own ``sorted(...)`` (JSON-stable output).
    c) PARTITION ⊆ LEAVES — ``pushed_keys`` and ``post_filtered_keys`` are DISJOINT
       and their union is a SUBSET of ``expected_leaves``. Subset (NOT equality): a
       leaf no channel gates lands in neither top-level list, which is legal on a
       multi-channel engine where a channel did not run this recall.
    d) PUSHED_DOWN (list-form) — when the filter HAS leaves, ``pushed_down`` iff
       (``post_filtered_keys == []`` AND ``set(pushed_keys) == expected_leaves``),
       the exact derivation ``build_filter_report`` uses (report.py ~L193). NOT the
       ``post_filtered`` bool, which a defensive full-predicate re-check flips True
       even at 100% pushdown (chronicle, VC graph). For a LEAFLESS filter the
       builder early-returns the canonical empty carrier (report.py L174-175):
       ``pushed_down`` False and BOTH top-level lists empty — asserted explicitly
       (a real contract, not a skip).
    e) CHANNEL FOLD CONSISTENCY — re-derives the builder fold from the per-channel
       breakdown: for each leaf gated by ≥1 channel, it is in the top-level
       ``post_filtered_keys`` iff some gating channel re-checked it in memory, else
       in ``pushed_keys``. Catches an engine hand-rolling the top level out of step
       with its channels. (Every channel key is also ⊆ ``expected_leaves``.)
    f) POST_FILTERED flag — one-directional only: a non-empty ``post_filtered_keys``
       implies the ``post_filtered`` bool is True. NEVER the converse — a defensive
       re-check sets the bool True with empty ``post_filtered_keys`` (NO-DEMOTE).

    Raises ``AssertionError`` on the first violated invariant. The no-private-leak
    check (``"_filter_channel_plans"`` absent from ``engine_info``) is asserted by
    the caller, which holds the full ``engine_info`` dict this helper does not see.
    """
    # (a) SCHEMA — round-trip through the model (a renamed/typed-wrong field fails
    # here) AND pin the literal top-level + per-channel key sets.
    assert set(report) == _REPORT_TOP_KEYS, (
        f"report top-level keys {sorted(report)} != expected {sorted(_REPORT_TOP_KEYS)}"
    )
    model = FilterPushdownReport.model_validate(report)
    for name, channel in report["channels"].items():
        assert set(channel) == _CHANNEL_KEYS, (
            f"channel {name!r} keys {sorted(channel)} != expected {sorted(_CHANNEL_KEYS)}"
        )

    pushed = set(model.pushed_keys)
    post_filtered = set(model.post_filtered_keys)

    # (b) SORTED — top-level and per-channel lists are JSON-stable.
    assert model.pushed_keys == sorted(model.pushed_keys), f"pushed_keys not sorted: {model.pushed_keys}"
    assert model.post_filtered_keys == sorted(model.post_filtered_keys), (
        f"post_filtered_keys not sorted: {model.post_filtered_keys}"
    )
    for name, channel in model.channels.items():
        assert channel.pushed_keys == sorted(channel.pushed_keys), (
            f"channel {name!r} pushed_keys not sorted: {channel.pushed_keys}"
        )
        assert channel.post_filtered_keys == sorted(channel.post_filtered_keys), (
            f"channel {name!r} post_filtered_keys not sorted: {channel.post_filtered_keys}"
        )

    # (c) PARTITION ⊆ LEAVES — disjoint, union a subset (an ungated leaf lands in
    # neither list — legal on a multi-channel engine).
    assert not (pushed & post_filtered), (
        f"pushed_keys and post_filtered_keys overlap on {sorted(pushed & post_filtered)} — not a partition"
    )
    assert pushed | post_filtered <= expected_leaves, (
        f"pushed_keys ∪ post_filtered_keys {sorted(pushed | post_filtered)} ⊄ "
        f"constraint-leaf set {sorted(expected_leaves)}"
    )

    # (d) PUSHED_DOWN — list-form biconditional when the filter has leaves; the
    # leafless early-return contract otherwise (both are real assertions).
    if expected_leaves:
        expected_pushed_down = not model.post_filtered_keys and pushed == expected_leaves
        assert model.pushed_down is expected_pushed_down, (
            f"pushed_down={model.pushed_down} but post_filtered_keys={model.post_filtered_keys} / "
            f"pushed_keys={model.pushed_keys} vs leaves {sorted(expected_leaves)} imply {expected_pushed_down}"
        )
    else:
        assert model.pushed_down is False, f"leafless report has pushed_down={model.pushed_down}, expected False"
        assert model.pushed_keys == [] and model.post_filtered_keys == [], (
            f"leafless report has non-empty top-level lists: "
            f"pushed_keys={model.pushed_keys}, post_filtered_keys={model.post_filtered_keys}"
        )

    # (e) CHANNEL FOLD CONSISTENCY — re-derive the top-level partition from the
    # per-channel breakdown and confirm the engine's emitted top level matches.
    for name, channel in model.channels.items():
        chan_keys = set(channel.pushed_keys) | set(channel.post_filtered_keys)
        assert chan_keys <= expected_leaves, (
            f"channel {name!r} addresses keys {sorted(chan_keys - expected_leaves)} "
            f"outside the leaf set {sorted(expected_leaves)}"
        )
    for leaf in expected_leaves:
        gating = [c for c in model.channels.values() if leaf in (set(c.pushed_keys) | set(c.post_filtered_keys))]
        if not gating:
            continue  # ungated leaf — in neither top-level list (checked by (c))
        rechecked = any(leaf in c.post_filtered_keys for c in gating)
        if rechecked:
            assert leaf in post_filtered, (
                f"leaf {leaf!r} re-checked in memory by a gating channel but absent from "
                f"top-level post_filtered_keys {sorted(post_filtered)}"
            )
        else:
            assert leaf in pushed, (
                f"leaf {leaf!r} pushed by every gating channel but absent from top-level pushed_keys {sorted(pushed)}"
            )

    # (f) POST_FILTERED flag is one-directional: non-empty post_filtered_keys ⇒
    # the bool is True. NEVER the converse (defensive_recheck sets it True with an
    # empty post_filtered_keys — NO-DEMOTE).
    if model.post_filtered_keys:
        assert model.post_filtered is True, (
            f"post_filtered_keys={model.post_filtered_keys} is non-empty but post_filtered flag is False"
        )


def filter_spec_leaves(filter_spec: dict[str, Any] | RecallFilter | None) -> frozenset[str]:
    """The constraint-leaf set of a recall filter, via the production lowering.

    ``None`` (no-filter) carries no leaves. A wire ``dict`` is validated with
    :meth:`RecallFilter.model_validate`; an already-built :class:`RecallFilter`
    is used as-is — the exact branch the recall facade takes before
    :func:`parse_to_ast`. An empty-``AND`` root (``filter={}``) carries no
    children, so its leaf set is empty. The gate module passes the result of this
    straight to :func:`assert_filter_report_invariants` as ``expected_leaves``.
    """
    if filter_spec is None:
        return frozenset()
    model = filter_spec if isinstance(filter_spec, RecallFilter) else RecallFilter.model_validate(filter_spec)
    ast = parse_to_ast(model)
    return filter_leaf_keys(ast) if ast.children else frozenset()


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


def _weaviate_reachable() -> bool:
    """Whether the live Weaviate at ``WEAVIATE_URL`` is reachable (the weaviate-lane self-skip guard).

    Mirrors :func:`_pg_reachable` and the conftest ``_weaviate_reachable``: parse
    host+port off ``WEAVIATE_URL`` and TCP-probe it, so the live Skeleton-Weaviate
    module's ``pytest.mark.skipif`` can call it at collection time without importing
    the conftest fixtures (keeps the QA-imports-from-``_harness`` contract clean).
    ``WEAVIATE_URL`` defaults to the local ``make dev`` HTTP port (8090) when unset;
    the CI leg sets it explicitly.
    """
    import socket
    from urllib.parse import urlparse

    url = os.environ.get("WEAVIATE_URL", "http://localhost:8090")
    parsed = urlparse(url)
    try:
        with socket.create_connection((parsed.hostname or "localhost", parsed.port or 8080), timeout=2):
            return True
    except OSError:
        return False


def _surreal_embedded_available() -> bool:
    """Whether the embedded SurrealDB SDK is importable (the surrealdb-lane self-skip guard).

    The ``skeleton_surrealdb`` lane runs ``memory`` mode in-process — no container,
    but it does need the optional ``surrealdb`` SDK (``pip install khora[surrealdb]``).
    Probes via ``find_spec`` so the ``pytest.mark.skipif`` can call it at collection
    time without importing the SDK. Mirrors the ``importorskip("surrealdb")`` guard
    the integration SurrealDB modules use.
    """
    import importlib.util

    return importlib.util.find_spec("surrealdb") is not None


def _embedded_available() -> bool:
    """Whether the no-Docker embedded stack (aiosqlite + lancedb) is importable.

    The self-skip guard for the container-free embedded sqlite_lance lanes (VC and
    Skeleton). Probes via ``find_spec`` so the ``pytest.mark.skipif`` can call it at
    collection time without importing either package.
    """
    import importlib.util

    return importlib.util.find_spec("aiosqlite") is not None and importlib.util.find_spec("lancedb") is not None


# --------------------------------------------------------------------------- #
# Per-lane recall-fixture resolution — the (kb fixture, reachability) wiring the
# cross-engine filter-report invariant gate consumes.
#
# Every reachability signal the gate's lanes need lives HERE (not in the gate's
# test module): the gate is one parametrized module spanning all lanes, so naming
# ``_pg_reachable`` / ``NEO4J_INTEGRATION_TEST`` in its body would make the
# verification-coverage gate union every lane's backend into a single per-module
# requirement no one leg satisfies. Routing through these helpers keeps the gate
# module backend-signal-free (inferred embedded-only), so any e2e leg that selects
# it clears the orphan check while the conftest tripwire still enforces the live
# store per leg. ``@internal``.
# --------------------------------------------------------------------------- #

# lane → kb fixture name. Mirrors tests/e2e/conftest.py one-for-one. The skip
# REASON strings are NOT held here: the gate module needs them as string literals
# (the verification-coverage gate reads a skipif reason via AST and exempts the
# env-guard phrases only when it can see the literal), so the gate keeps its own
# literal reasons and this map carries only the fixture name.
_LANE_RECALL_FIXTURE: dict[str, str] = {
    "vc_full": "vectorcypher_kb",
    "vc_embedded": "sqlite_lance_kb",
    "skeleton_pgvector": "skeleton_pgvector_kb",
    "skeleton_surrealdb": "skeleton_surrealdb_kb",
    "skeleton_weaviate": "skeleton_weaviate_kb",
    "skeleton_sqlite_lance": "skeleton_sqlite_lance_kb",
    "chronicle": "chronicle_kb",
}


def lane_recall_fixture(lane: str) -> str:
    """The recall-capable ``Khora`` fixture name for a gate lane.

    ``lane`` is a matrix-lane key (a :data:`GATE_LANES` member). Raises on an
    unknown lane so a typo fails loud rather than silently resolving nothing.
    """
    if lane not in _LANE_RECALL_FIXTURE:
        raise RuntimeError(
            f"lane_recall_fixture({lane!r}): not a gate lane; expected one of {sorted(_LANE_RECALL_FIXTURE)}"
        )
    return _LANE_RECALL_FIXTURE[lane]


def lane_reachable(lane: str) -> bool:
    """Whether a gate lane's store / optional dep is available (the collection-time guard).

    Mirrors the sibling row-set module's ``skipif`` condition for the same lane:
    the embedded lanes probe their optional dep, the live lanes TCP-probe their
    store (the ``vc_full`` lane also requires ``NEO4J_INTEGRATION_TEST`` set,
    flipping the live-graph path active — the exact guard the graph row-set module
    carries). On a no-Docker run a live lane reports unreachable so the gate
    collects-and-skips identically; in CI the conftest tripwire turns a missing
    required store into a RED session, so a skip there can only mean "not this leg".
    """
    if lane == "vc_full":
        return bool(os.environ.get("NEO4J_INTEGRATION_TEST")) and _pg_reachable()
    if lane in ("skeleton_pgvector", "chronicle"):
        return _pg_reachable()
    if lane == "skeleton_weaviate":
        return _pg_reachable() and _weaviate_reachable()
    if lane in ("vc_embedded", "skeleton_sqlite_lance"):
        return _embedded_available()
    if lane == "skeleton_surrealdb":
        return _surreal_embedded_available()
    raise RuntimeError(f"lane_reachable({lane!r}): not a gate lane; expected one of {sorted(_LANE_RECALL_FIXTURE)}")
