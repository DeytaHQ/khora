"""Shared spy helpers for the recall-filter pushdown/threading suites.

Both the embedded (``sqlite_lance``, no Docker) and the live-DB (PG+Neo4j)
filter-enforcement spy modules import from here so the *contract* they
assert is identical across backends:

* the canonical filter AST is rebuilt EXACTLY as the facade builds it
  (``RecallFilter.model_validate`` then ``parse_to_ast`` — see
  ``khora.recall`` in ``src/khora/khora.py``), and
* a path is proven to thread the filter by capturing the live
  ``filter_ast`` it received and comparing :func:`canonical_hash` against
  that expected AST.

What this helper deliberately does NOT do: inspect result rows, scores,
or ranking. The spies pin the WIRING contract (the validated AST reaches
each channel unchanged); the end-to-end row-set proof is the
filter-conformance suite's job. Keeping the two concerns apart is what
lets the no-Docker embedded spies run in the main ``test`` job.

The deterministic extractor + embedder + seed helpers are lifted from the
existing ``sqlite_lance`` integration suite so embedded and live-DB spies
share ONE entity-bearing seed (same content -> same entities -> same
vectors), making cross-backend comparisons apples-to-apples.

Not shipped as part of the khora package.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from khora.extraction.extractors.base import (
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
)
from khora.filter import RecallFilter, canonical_hash, parse_to_ast
from khora.filter.ast import FilterNode

__all__ = [
    "EMBED_DIM",
    "FilterCallRecord",
    "assert_filter_threaded",
    "expected_hash",
    "fake_embedding",
    "plan_extraction",
    "seed_corpus",
    "spy_on",
    "stub_llm",
]

# Embedded LanceDB default; the live-DB suites override their own dimension
# at the kb-fixture level and never import this constant for sizing.
EMBED_DIM = 32


# --------------------------------------------------------------------------- #
# Expected-AST oracle.
# --------------------------------------------------------------------------- #


def expected_hash(filter: dict[str, Any] | RecallFilter) -> str:
    """Return the canonical hash a correctly-threaded path must carry.

    Rebuilds the AST the SAME way the facade does — ``RecallFilter`` instance
    used as-is, dict validated via ``model_validate`` — then hashes it. This is
    the oracle the spies compare every captured ``filter_ast`` against.
    """
    recall_filter = filter if isinstance(filter, RecallFilter) else RecallFilter.model_validate(filter)
    return canonical_hash(parse_to_ast(recall_filter))


# --------------------------------------------------------------------------- #
# Call capture.
# --------------------------------------------------------------------------- #


@dataclass
class FilterCallRecord:
    """One captured call to a spied method.

    ``filter_ast`` is pulled from the call's ``filter_ast`` kwarg when present,
    else from the first positional arg that is a :class:`FilterNode` (covers
    free-function spy points like ``compile_cypher(ast, ctx)`` that take the AST
    positionally). ``canonical_hash`` is computed once at capture time so the
    assertion helper is a pure comparison.
    """

    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    filter_ast: FilterNode | None = field(default=None)
    canonical_hash: str | None = field(default=None)

    def __post_init__(self) -> None:
        ast = self.kwargs.get("filter_ast")
        if ast is None:
            ast = next((a for a in self.args if isinstance(a, FilterNode)), None)
        self.filter_ast = ast
        self.canonical_hash = canonical_hash(ast) if isinstance(ast, FilterNode) else None


def spy_on(
    monkeypatch: Any,
    target: Any,
    method_name: str,
) -> list[FilterCallRecord]:
    """Wrap ``target.method_name`` so every call is recorded, then passed through.

    Returns a live list that grows by one :class:`FilterCallRecord` per call.
    The real method still runs (and its result is returned to the caller
    unchanged), so the spy is non-invasive: the recall under test executes its
    genuine logic and we only OBSERVE what ``filter_ast`` flowed past this
    boundary.

    Handles BOTH async and SYNC targets: bound async retriever methods
    (``_vector_search_chunks`` etc.) AND module-level sync compilers
    (``compile_cypher`` / ``compile_postgres`` — pass the module as ``target``,
    e.g. ``spy_on(monkeypatch, khora.filter.compilers.cypher, "compile_cypher")``).
    The sync/async branch is selected off the wrapped callable so the wrapper
    matches the original's call convention (awaiting a sync function raises).

    Use with the real ``monkeypatch`` fixture so the patch is undone at test
    teardown.
    """
    original = getattr(target, method_name)
    records: list[FilterCallRecord] = []

    if inspect.iscoroutinefunction(original):

        @functools.wraps(original)
        async def _async_wrapper(*args: Any, **kwargs: Any) -> Any:
            records.append(FilterCallRecord(args=args, kwargs=dict(kwargs)))
            return await original(*args, **kwargs)

        monkeypatch.setattr(target, method_name, _async_wrapper)
    else:

        @functools.wraps(original)
        def _sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            records.append(FilterCallRecord(args=args, kwargs=dict(kwargs)))
            return original(*args, **kwargs)

        monkeypatch.setattr(target, method_name, _sync_wrapper)
    return records


def assert_filter_threaded(
    records: list[FilterCallRecord],
    expected: dict[str, Any] | RecallFilter,
    *,
    min_calls: int = 1,
) -> None:
    """Assert a path threaded the expected recall filter, with a vacuity guard.

    Two-part contract, both required:

    1. **Vacuity guard** — ``len(records) >= min_calls``. A path that silently
       stopped issuing the call (so nothing was ever captured) must FAIL rather
       than pass an empty all-of over zero records. This is what keeps a spy
       from going green when the channel it watches was never exercised.
    2. **Canonical-hash equality** — every captured ``filter_ast`` hashes to
       ``expected_hash(expected)``. Comparing the hash (not object identity or
       the dict) means equivalent-but-reordered ASTs still match and a dropped
       or mutated predicate is caught.

    No result/ranking inspection — by design.
    """
    want = expected_hash(expected)
    assert len(records) >= min_calls, (
        f"vacuity guard: expected >= {min_calls} spied call(s) carrying the filter, "
        f"got {len(records)} — the path under test was never exercised, so the "
        f"thread-through is unproven (not green)."
    )
    for i, rec in enumerate(records):
        assert rec.canonical_hash is not None, (
            f"call #{i} reached the channel with NO filter_ast (got args={rec.args!r} "
            f"kwargs={rec.kwargs!r}); the filter was dropped before this boundary."
        )
        assert rec.canonical_hash == want, (
            f"call #{i} carried a filter whose canonical_hash {rec.canonical_hash} "
            f"!= expected {want}; the threaded AST diverged from what the facade built."
        )


# --------------------------------------------------------------------------- #
# Deterministic extractor + embedder + seed.
# --------------------------------------------------------------------------- #


def fake_embedding(text: str, dim: int = EMBED_DIM) -> list[float]:
    """Deterministic L2-normalised vector for ``text``.

    SHA-256 the text, expand to ``dim`` floats, normalise. Same text -> same
    vector; different text -> different vector. Suitable for deterministic
    top-k ordering, NOT for semantic similarity. Shared by embedded and
    live-DB spies so identical content seeds identical vectors on both.
    """
    seed = hashlib.sha256(text.encode("utf-8")).digest()
    raw = [(seed[i % len(seed)] - 128) / 128.0 for i in range(dim)]
    norm = sum(x * x for x in raw) ** 0.5 or 1.0
    return [x / norm for x in raw]


# Content-keyed registry: a document whose text CONTAINS a registered marker
# yields that marker's ExtractionResult. Module-global so ``stub_llm`` can
# reset it per module; tests stage entries via ``plan_extraction``.
_EXTRACTION_REGISTRY: dict[str, ExtractionResult] = {}


def plan_extraction(
    marker: str,
    entities: list[tuple[str, str]],
    relationships: list[tuple[str, str, str]] | None = None,
) -> None:
    """Stage a deterministic ``ExtractionResult`` for docs containing ``marker``.

    ``entities`` are ``(name, entity_type)`` pairs; ``relationships`` are
    ``(source, target, relationship_type)`` triples. This is the hook the graph
    channels need: seed specific entity pairs + a typed edge so the Cypher /
    over-fetch paths actually traverse.
    """
    _EXTRACTION_REGISTRY[marker] = ExtractionResult(
        entities=[ExtractedEntity(name=n, entity_type=t, confidence=0.99) for n, t in entities],
        relationships=[
            ExtractedRelationship(
                source_entity=s,
                target_entity=t,
                relationship_type=rt,
                confidence=0.99,
            )
            for s, t, rt in (relationships or [])
        ],
    )


def stub_llm(monkeypatch: Any, dim: int = EMBED_DIM) -> None:
    """Patch the embedder + entity extractor to the deterministic registry.

    No ``OPENAI_API_KEY`` / network needed. Call once per test module (e.g. via
    an autouse fixture). Resets the extraction registry so stale markers from a
    prior module never leak. Embedding and extraction are patched at the class
    method level so every engine code path picks them up.

    ``dim`` sizes the deterministic vectors. The embedded LanceDB suites use the
    default (small) ``EMBED_DIM``; the live-DB suites pass ``dim=1536`` because
    the shared dev Postgres DB is migrated at 1536 (the pgvector column size is
    fixed at fresh-DB creation from ``llm.embedding_dimension``; #1260).
    """
    _EXTRACTION_REGISTRY.clear()

    async def _embed_batch(self: Any, texts: list[str]) -> list[list[float]]:
        return [fake_embedding(t, dim) for t in texts]

    async def _embed(self: Any, text: str) -> list[float]:
        return fake_embedding(text, dim)

    async def _extract_multi(self: Any, texts: list[str], **_kwargs: Any) -> list[ExtractionResult]:
        out: list[ExtractionResult] = []
        for t in texts:
            matched = next(
                (result for marker, result in _EXTRACTION_REGISTRY.items() if marker in t),
                None,
            )
            out.append(matched if matched is not None else ExtractionResult())
        return out

    monkeypatch.setattr(
        "khora.extraction.embedders.litellm.LiteLLMEmbedder.embed_batch",
        _embed_batch,
    )
    monkeypatch.setattr(
        "khora.extraction.embedders.litellm.LiteLLMEmbedder.embed",
        _embed,
    )
    monkeypatch.setattr(
        "khora.extraction.extractors.llm.LLMEntityExtractor.extract_multi",
        _extract_multi,
    )


async def seed_corpus(
    remember: Callable[..., Awaitable[Any]],
    namespace_id: UUID,
    docs: list[str | dict[str, Any]],
) -> None:
    """Seed an entity-bearing corpus via the caller's ``remember`` API.

    ``remember`` is a partial / bound callable that already pins the kb +
    per-test ``entity_types`` / ``expertise`` wiring — embedded and live-DB
    fixtures differ there, so each owns its own. We only drive content +
    namespace (+ optional metadata/title) so the SAME documents (and therefore
    the same registered entities) seed both backends. Read-only after seed;
    namespace-partitioned by the caller.

    Each doc is either:

    * a plain ``str`` → seeded as ``remember(content=doc, namespace=...)``; or
    * a ``dict`` ``{"content": str, "metadata": {...}?, "title": str?}`` → the
      ``metadata`` is threaded to ``remember(metadata=...)`` so a graph-channel
      residual-metadata predicate (e.g. ``{"metadata": {"channel": {"$eq":
      "eng"}}}``) has a plantable key to post-filter on. The bound ``remember``
      MUST NOT pre-bind ``metadata`` when dict docs are used, or the per-doc
      value collides with the partial.
    """
    for doc in docs:
        if isinstance(doc, str):
            await remember(content=doc, namespace=namespace_id)
            continue
        kwargs: dict[str, Any] = {"content": doc["content"], "namespace": namespace_id}
        if doc.get("metadata") is not None:
            kwargs["metadata"] = doc["metadata"]
        if doc.get("title") is not None:
            kwargs["title"] = doc["title"]
        await remember(**kwargs)
