"""Full-stack integration tests for the recall-filter on Chronicle + sqlite_lance.

This file drives the public ``Khora.recall(filter=...)`` API end-to-end through
the Chronicle engine over a real embedded SQLite + LanceDB store. The
engine→backend wiring threads ``filter_ast`` from the facade through Chronicle's
recall path into the deterministic post-filter, so row filtering is effective
end-to-end — these tests pin that contract on the embedded stack.

Unlike ``test_chronicle_filter_composition.py`` (which is the MOCKED, unit-level
proof of how Chronicle composes the two compiled filter halves on a
``MagicMock`` storage), this file's value-add is the LIVE corpus isolation: real
``Khora.remember()`` writes, real ``Khora.recall(filter=...)`` reads, real
sqlite_lance roundtrip. It does NOT re-mock what the composition suite already
proves; it proves the same logical narrowing survives the live write/read path.

THE THREE-PREDICATE CORPUS — real system keys
=============================================
The filter under test is ``source_name == "linear" AND occurred_at >= 2026-04-05
AND metadata.tag IN {"urgent", "release"}`` — the same three-predicate shape the
filter-conformance corpus pins as ``F-SEL-three-predicate`` (postgres leg). The
corpus here is ten records with a 5-in-scope / 5-out-of-scope split: each
out-of-scope row violates EXACTLY ONE predicate, and one in-scope row sits
EXACTLY at the date bound. Since the Chronicle recall path hydrates the
per-document ``DocumentProjection`` for doc-key filters, the predicate runs over
REAL system keys: ``source_name`` (a denormalized document key) and
``occurred_at`` (a system date key), with ``metadata.tag`` the one free-form
metadata leaf. The contract proven here is "a real-store multi-predicate AND over
real system keys narrows to exactly the in-scope set".

How each real key resolves on the live store (verified against this exact
sqlite_lance fixture, not assumed):

* ``source_name`` ($eq "linear"). When the filter carries a doc-key leaf,
  Chronicle batch-fetches the per-document ``DocumentProjection``
  (``get_document_projections_batch``) and ``_chunk_to_record`` resolves
  ``source_name`` off it, so a positive predicate bites on the value the corpus
  wrote via ``remember(source_name=...)``. The previous "doc keys resolve absent"
  limitation is fixed for filtered queries; the short-circuit means a recall with
  NO doc-key leaf still pays ZERO extra fetch (keys stay absent, as before).
* ``occurred_at`` ($gte, plain ISO operand). The embedded write path leaves the
  literal ``occurred_at`` column null and the engine derives the effective event
  time from ``source_timestamp``; the corpus seeds the date axis via
  ``remember(source_timestamp=...)``. The embedded SQLite store returns tz-NAIVE
  datetimes; the compiler aligns a naive stored value to UTC at the comparison
  boundary, so the system date key narrows cleanly rather than raising.
* ``metadata.tag`` ($in {"urgent", "release"}) — the one free-form metadata
  leaf.

ENVIRONMENT: needs the ``sqlite_lance`` extra (``pip install khora[sqlite_lance]``
→ aiosqlite + lancedb). No Docker / Postgres / Neo4j. The module self-skips when
those embedded deps are absent (e.g. the lint sandbox) and RUNS as the real gate
in the CI integration job.

How to run locally::

    uv run pytest tests/recall/test_filter_chronicle.py -m integration -v --no-cov
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

try:
    import aiosqlite  # noqa: F401
    import lancedb  # noqa: F401

    _HAS_EMBEDDED = True
except ImportError:
    _HAS_EMBEDDED = False

from khora.config import KhoraConfig
from khora.config.schema import SQLiteLanceConfig
from khora.extraction.extractors.base import ExtractionResult
from khora.extraction.skills import ExpertiseConfig
from khora.extraction.skills.base import EventExtractionConfig, FactExtractionConfig
from khora.khora import Khora
from khora.query import SearchMode
from tests.test_helpers.diagnostics import assert_no_silent_degradation

EMBED_DIM = 32  # sqlite_lance default; keeps LanceDB ANN builds cheap in tmp_path


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _HAS_EMBEDDED,
        reason="aiosqlite/lancedb not installed (pip install khora[sqlite_lance])",
    ),
]


# ---------------------------------------------------------------------------
# Deterministic embedder stub — no OPENAI_API_KEY required.
# ---------------------------------------------------------------------------
#
# Every text maps to the SAME 32-dim unit vector, so query and all chunk
# embeddings are identical → cosine similarity is 1.0 for every row. Retrieval is
# therefore filter-bound, not similarity-bound: the recall filter is the only
# thing that narrows the candidate set, which is exactly what these tests must
# isolate. (Chronicle's per-chunk event/fact extraction is disabled below, so
# only the embedder needs stubbing.)


def _unit_vector() -> list[float]:
    return [1.0] + [0.0] * (EMBED_DIM - 1)


async def _stub_embed_batch(self: Any, texts: list[str]) -> list[list[float]]:
    return [_unit_vector() for _ in texts]


async def _stub_embed(self: Any, text_in: str) -> list[float]:
    return _unit_vector()


@pytest.fixture(autouse=True)
def _patch_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "khora.extraction.embedders.litellm.LiteLLMEmbedder.embed_batch",
        _stub_embed_batch,
    )
    monkeypatch.setattr(
        "khora.extraction.embedders.litellm.LiteLLMEmbedder.embed",
        _stub_embed,
    )


# ---------------------------------------------------------------------------
# Entity-extraction stub — no OPENAI_API_KEY required.
# ---------------------------------------------------------------------------
#
# Every ``remember`` here passes ``entity_types=["PERSON"]`` /
# ``relationship_types=["KNOWS"]``, so the ingest pipeline's ``extract_entities``
# task calls ``LLMEntityExtractor.extract_multi`` — a LIVE LiteLLM call. Without
# an API key each call retries with exponential backoff before degrading to zero
# entities, which is slow and network-dependent. No test in this file asserts on
# extracted entities (the corpus filters only on source_name / occurred_at /
# metadata.tag), so stubbing the extractor to return empty results keeps these
# tests hermetic and fast while leaving the filter-narrowing contract untouched.
#
# ``extract_multi`` is the method the pipeline calls; the consumer zips its result
# one-to-one with the input chunks, so the stub returns exactly one empty
# ``ExtractionResult`` per text. (Events/facts extraction is already disabled via
# ``_no_event_fact_extraction()``, so the entity extractor is the only live LLM
# extractor left to stub.)


async def _stub_extract_multi(self: Any, texts: list[str], *args: Any, **kwargs: Any) -> list[ExtractionResult]:
    return [ExtractionResult() for _ in texts]


@pytest.fixture(autouse=True)
def _patch_entity_extractor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "khora.extraction.extractors.llm.LLMEntityExtractor.extract_multi",
        _stub_extract_multi,
    )


# ---------------------------------------------------------------------------
# Chronicle + sqlite_lance Khora fixture.
# ---------------------------------------------------------------------------


def _no_event_fact_extraction() -> ExpertiseConfig:
    """Disable Chronicle's per-chunk event/fact extraction for determinism.

    These tests exercise only the recall-filter narrowing; the event/fact
    extractors would otherwise fire LLM calls the embedder stub does not cover.
    """
    return ExpertiseConfig(
        name="chronicle-filter-integ",
        events=EventExtractionConfig(enabled=False),
        facts=FactExtractionConfig(enabled=False),
    )


@pytest.fixture
async def kb(tmp_path: Path) -> AsyncIterator[Khora]:
    """A connected Chronicle Khora bound to a per-test embedded SQLite+LanceDB.

    ``neo4j_url=None`` keeps it graph-less (Chronicle's channels are all
    SQL/LanceDB on the embedded path); ``run_migrations=True`` materializes the
    embedded schema. Reranking is OFF: it defaults to True and would lazily load
    the cross-encoder on the first recall (slow / network-dependent); it only
    reorders candidates and never adds or drops a row, so disabling it leaves the
    filter-narrowing contract under test unchanged while keeping tests hermetic.
    """
    config = KhoraConfig()
    config.storage.backend = "sqlite_lance"
    config.storage.sqlite_lance = SQLiteLanceConfig(
        db_path=str(tmp_path / "khora.db"),
        lance_path=str(tmp_path / "khora.lance"),
        embedding_dimension=EMBED_DIM,
    )
    config.llm.embedding_dimension = EMBED_DIM
    config.storage.embedding_dimension = EMBED_DIM
    config.neo4j_url = None
    config.pipelines.chunk_size = 1024
    config.query.enable_reranking = False

    instance = Khora(config, engine="chronicle", run_migrations=True)
    await instance.connect()
    try:
        yield instance
    finally:
        await instance.disconnect()


# ---------------------------------------------------------------------------
# Seed corpus — 5 in-scope, 5 out-of-scope (one predicate violation each).
# ---------------------------------------------------------------------------
#
# The three-predicate filter under test (over real keys):
#   source_name == "linear"                      (source axis, hydrated)
#   AND occurred_at >= 2026-04-05 ($date)        (date axis, system key)
#   AND metadata.tag IN {"urgent", "release"}    (metadata axis)
#
# All three axes are real keys. ``source_name`` is a denormalized
# document key the corpus writes via ``remember(source_name=channel)``; on a
# filtered (doc-key-bearing) recall Chronicle hydrates the per-document
# ``DocumentProjection`` so ``_chunk_to_record`` resolves it and the $eq bites.
# ``occurred_at`` is the system event-time key the engine derives from
# ``source_timestamp`` on the embedded write path, so the corpus seeds the date
# axis via ``remember(source_timestamp=...)``. ``metadata.tag`` is the one
# free-form metadata leaf. The in/out split is the same three-predicate shape the
# filter-conformance corpus pins as ``F-SEL-three-predicate``.
#
# Content is DISTINCT per row so each remember produces its own chunk (Chronicle
# dedupes by content checksum, so identical content would collapse to one chunk);
# the shared "alpha bravo charlie" token lets the constant-vector retrieval
# surface every row before the filter.
#
# Deliberately NOT covered here (this file is the live corpus-equivalence gate):
# the tag-is-explicitly-null, unanchored-chunk (occurred_at=None AND
# source_timestamp=None), and pgvector occurred_at=None DTO shapes. Those edge
# shapes are exercised by the mocked unit suite,
# tests/recall/test_chronicle_filter_composition.py.

_IN_BOUND = "2026-04-05T00:00:00Z"  # the occurred_at lower bound (inclusive)

# An external_id stamped on exactly one in-scope row (in_boundary) so the
# anti-masking pair (c) can prove a previously-absent doc key resolves to a real
# value post-hydration: a positive predicate on it returns that one row, a
# sentinel returns empty.
_BOUNDARY_EXTERNAL_ID = "ext-in-boundary-001"

# label -> (source_name, source_timestamp ISO, tag | None)
_IN_SCOPE: dict[str, tuple[str, str, str | None]] = {
    # All three predicates satisfied. Boundary, mid, and future event time; both
    # allowed tag values are represented.
    "in_boundary": ("linear", _IN_BOUND, "urgent"),  # exactly at the >= bound
    "in_recent": ("linear", "2026-05-01T12:00:00Z", "release"),
    "in_future": ("linear", "2026-06-01T00:00:00Z", "urgent"),
    "in_release": ("linear", "2026-04-10T00:00:00Z", "release"),
    "in_urgent": ("linear", "2026-04-20T09:30:00Z", "urgent"),
}

# Each out-of-scope row violates EXACTLY ONE predicate; every other field is
# in-scope, so a leak pins which predicate failed to bite.
_OUT_OF_SCOPE: dict[str, tuple[str | None, str, str | None]] = {
    # Predicate 1 (source_name) violated; date + tag are in-scope.
    "out_wrong_source": ("slack", "2026-05-15T00:00:00Z", "urgent"),
    # Predicate 2 (occurred_at) violated: one second before the inclusive bound.
    "out_too_old": ("linear", "2026-04-04T23:59:59Z", "release"),
    # Predicate 3 (tag) violated: tag present but outside the allowed set.
    "out_wrong_tag": ("linear", "2026-05-20T00:00:00Z", "backlog"),
    # Predicate 3 (tag) violated: tag key absent entirely.
    "out_missing_tag": ("linear", "2026-05-25T00:00:00Z", None),
    # Predicate 1 (source_name) violated via absence: source_name left UNSET at
    # write, so it hydrates absent and the positive $eq "linear" excludes it.
    "out_null_source": (None, "2026-05-30T00:00:00Z", "release"),
}


def _parse_when(when: str) -> datetime:
    """Parse a corpus ``when`` ISO string to a tz-aware UTC datetime.

    The ``Z`` suffix maps to ``+00:00``; the result feeds
    ``remember(source_timestamp=...)`` which the embedded write path derives
    ``occurred_at`` from.
    """
    return datetime.fromisoformat(when.replace("Z", "+00:00"))


# The live three-predicate recall filter — the F-SEL-three-predicate shape over real keys.
# ``occurred_at`` takes a plain ISO operand (the public ``RecallFilter`` model
# coerces it to a UTC-aware datetime; the ``{"$date": ...}`` typed literal is an
# AST-level form the dict API does not accept on a system date key), matching the
# existing system-``occurred_at`` tests below. ``source_name`` and ``occurred_at``
# are SYSTEM keys; ``metadata.tag`` is the one metadata leaf.
_RECALL_FILTER = {
    "source_name": "linear",
    "occurred_at": {"$gte": _IN_BOUND},
    "metadata.tag": {"$in": ["urgent", "release"]},
}


def _content_for(label: str) -> str:
    # Distinct, short (single-chunk) content per row, with the shared retrieval
    # token plus a stable per-row suffix so identical content never collapses.
    return f"alpha bravo charlie {label} {hashlib.sha256(label.encode()).hexdigest()[:8]}"


async def _seed(kb: Khora, namespace_id: UUID) -> dict[str, str]:
    """Remember every row and return a ``content -> label`` map.

    Each ``remember`` produces exactly one chunk (content < chunk_size). The
    distinct per-row content is the stable join key (``content -> label``), so
    assertions resolve which rows the filter returned without reaching into the
    store — the expected in-scope set is encoded by construction in ``_IN_SCOPE``.

    The source axis rides the real ``source_name`` document key (``None`` →
    omitted so the row hydrates absent) and the date axis rides
    ``source_timestamp`` (the engine derives ``occurred_at`` from it on the
    embedded write path); only ``tag`` stays in ``metadata``. ``in_boundary`` also
    carries an ``external_id`` so the anti-masking pair can prove a previously
    absent doc key resolves post-hydration.
    """
    for label, (source_name, when, tag) in {**_IN_SCOPE, **_OUT_OF_SCOPE}.items():
        metadata: dict[str, Any] = {}
        if tag is not None:
            metadata["tag"] = tag
        kwargs: dict[str, Any] = {}
        if source_name is not None:
            kwargs["source_name"] = source_name
        if label == "in_boundary":
            kwargs["external_id"] = _BOUNDARY_EXTERNAL_ID
        await kb.remember(
            content=_content_for(label),
            namespace=namespace_id,
            title=label,
            metadata=metadata,
            source_timestamp=_parse_when(when),
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
            expertise=_no_event_fact_extraction(),
            **kwargs,
        )
    return {_content_for(label): label for label in {**_IN_SCOPE, **_OUT_OF_SCOPE}}


def _labels_returned(result: Any, content_to_label: dict[str, str]) -> set[str]:
    return {content_to_label[c.content] for c in result.chunks}


# ===========================================================================
# Scenario 1 — green equivalence: the three-predicate AND returns EXACTLY the
# in-scope rows.
# ===========================================================================


async def test_filter_returns_exactly_in_scope_chunks(kb: Khora) -> None:
    """``recall(filter=...)`` narrows the live result to exactly the 5 in-scope
    rows — no out-of-scope leak.

    End-to-end proof that the engine→backend wiring filters rows on the embedded
    stack: the facade builds the AST, Chronicle hydrates the per-document
    projection (because the filter carries the ``source_name`` doc-key leaf) and
    threads the AST to the deterministic post-filter, narrowing the candidate set
    before top-k. The filter is the three-predicate shape — ``source_name``
    (hydrated doc key), ``occurred_at`` (system date key), ``metadata.tag``
    (metadata leaf). Every out-of-scope row violates exactly one predicate, so a
    leak would name the broken one.

    Also asserts the hydrated recall completes with no ADR-001 degradation: a
    failed projection fetch would append a ``chronicle.doc_hydration`` entry to
    ``engine_info["degradations"]`` and leave the doc keys absent (silently
    dropping every in-scope row), so an empty degradations list confirms the
    hydration succeeded rather than degraded into a false pass.
    """
    ns = await kb.create_namespace()
    namespace_id: UUID = ns.namespace_id

    content_to_label = await _seed(kb, namespace_id)

    # limit comfortably exceeds the corpus so the filter, not the limit, bounds
    # the result. VECTOR mode keeps retrieval purely embedding+filter (no BM25
    # keyword channel), so the filter is the only narrowing force.
    result = await kb.recall(
        "alpha bravo charlie",
        namespace=namespace_id,
        limit=20,
        mode=SearchMode.VECTOR,
        filter=_RECALL_FILTER,
    )

    returned = _labels_returned(result, content_to_label)
    in_scope = set(_IN_SCOPE)
    out_of_scope = set(_OUT_OF_SCOPE)
    assert returned == in_scope, (
        f"filter must return EXACTLY the in-scope rows; leaked={returned & out_of_scope}, missing={in_scope - returned}"
    )
    # Happy-path: the doc-key hydration succeeded (no silent degradation).
    assert_no_silent_degradation(result)


async def test_no_filter_returns_all_chunks(kb: Khora) -> None:
    """Control: with no filter the same recall returns the whole corpus.

    Proves the in-scope/out-of-scope split is a property of the FILTER, not of
    retrieval reachability — every seeded chunk is recallable absent the filter,
    so Scenario 1's narrowing is attributable to the predicate alone.
    """
    ns = await kb.create_namespace()
    namespace_id: UUID = ns.namespace_id

    content_to_label = await _seed(kb, namespace_id)

    result = await kb.recall(
        "alpha bravo charlie",
        namespace=namespace_id,
        limit=20,
        mode=SearchMode.VECTOR,
    )

    returned = _labels_returned(result, content_to_label)
    assert returned == set(content_to_label.values()), "unfiltered recall must reach every seeded chunk"


# ===========================================================================
# Doc-key hydration — anti-masking + short-circuit.
# ===========================================================================
#
# CONSTANT-VECTOR CAVEAT: every text maps to the same unit vector, so cosine is
# 1.0 for every row and the filter is the ONLY narrowing force. A trivially
# matching filter would therefore "pass" (return rows) for the WRONG reason —
# specifically the pre-hydration failure mode, where a positive doc-key predicate
# resolved its key absent and (under the $ne/absent semantics) matched everything,
# or where a positive $eq returned empty because the key never resolved. The
# positive+negative pairs below pin that the hydrated doc key resolves to a REAL
# value: a positive predicate returns exactly the matching rows (non-empty,
# correct set) and a sentinel-value predicate returns EMPTY (the key bit, it did
# not silently match-all). One pair per key: ``source_name`` (always seeded) and
# ``external_id`` (previously absent, seeded only on in_boundary).


async def test_source_name_positive_and_sentinel(kb: Khora) -> None:
    """A ``source_name`` $eq resolves to the hydrated value: positive returns the
    matching rows, a sentinel returns empty.

    Anti-masking: ``{"source_name": "linear"}`` must return exactly the rows the
    corpus seeded with ``source_name="linear"`` (every in-scope row plus the
    out-of-scope rows that violate a DIFFERENT predicate but still carry
    ``source_name="linear"``), and ``{"source_name": "no-such-source"}`` must
    return EMPTY — proving the key hydrated to a real value and the predicate bit,
    not "absent → matched all" (the pre-hydration failure mode the constant vector
    would otherwise mask).
    """
    ns = await kb.create_namespace()
    namespace_id: UUID = ns.namespace_id
    content_to_label = await _seed(kb, namespace_id)

    async def _recall(filter_: dict[str, Any]) -> set[str]:
        result = await kb.recall(
            "alpha bravo charlie",
            namespace=namespace_id,
            limit=20,
            mode=SearchMode.VECTOR,
            filter=filter_,
        )
        return _labels_returned(result, content_to_label)

    # Rows seeded with source_name="linear": all in-scope plus the out-of-scope
    # rows whose violated predicate is NOT the source (too_old, wrong_tag,
    # missing_tag all carry source_name="linear").
    linear_rows = {
        label for label, (source_name, _when, _tag) in {**_IN_SCOPE, **_OUT_OF_SCOPE}.items() if source_name == "linear"
    }
    assert await _recall({"source_name": "linear"}) == linear_rows
    assert await _recall({"source_name": "no-such-source"}) == set()


async def test_external_id_positive_and_sentinel(kb: Khora) -> None:
    """A previously-absent doc key (``external_id``) resolves post-hydration.

    ``external_id`` is seeded on exactly one row (in_boundary). The positive
    predicate returns that one row; a sentinel returns EMPTY. This exercises the
    SAME anti-masking contract as ``source_name`` but on a key that resolved
    absent before the hydration landed — proving the hydration (not just an
    always-present column) is what makes the predicate bite.
    """
    ns = await kb.create_namespace()
    namespace_id: UUID = ns.namespace_id
    content_to_label = await _seed(kb, namespace_id)

    async def _recall(filter_: dict[str, Any]) -> set[str]:
        result = await kb.recall(
            "alpha bravo charlie",
            namespace=namespace_id,
            limit=20,
            mode=SearchMode.VECTOR,
            filter=filter_,
        )
        return _labels_returned(result, content_to_label)

    assert await _recall({"external_id": _BOUNDARY_EXTERNAL_ID}) == {"in_boundary"}
    assert await _recall({"external_id": "missing"}) == set()


async def test_doc_key_filter_hydrates_metadata_filter_short_circuits(
    kb: Khora, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ENGINE's projection hydration fires for a doc-key filter and NOT for a
    metadata-only filter — the hot path pays nothing.

    Spies on the storage coordinator's ``get_document_projections_batch`` (the
    same instance ``kb.storage`` and the engine both use). The facade ALWAYS calls
    it once per recall in its document-upgrade pass (``_upgrade_recall_documents``,
    stub → full ``DocumentProjection``), so the count is the discriminator, not
    its presence:

    * a metadata-only recall (``{"metadata.tag": "urgent"}``, no doc-key leaf)
      makes exactly ONE coordinator call — the upgrade pass only; the engine's
      ``filter_leaf_keys & _DOC_PROJECTION_KEYS`` short-circuit skips its
      hydration fetch entirely (the hot path pays nothing extra).
    * a doc-key recall (``{"source_name": "linear"}``) makes exactly TWO — the
      engine's per-recall hydration fetch PLUS the upgrade pass (one fetch each,
      not N+1).
    """
    ns = await kb.create_namespace()
    namespace_id: UUID = ns.namespace_id
    await _seed(kb, namespace_id)

    real_batch = kb.storage.get_document_projections_batch
    calls: list[int] = []

    async def _spy(document_ids: list[UUID], **kwargs: Any) -> dict[UUID, Any]:
        calls.append(len(document_ids))
        return await real_batch(document_ids, **kwargs)

    monkeypatch.setattr(kb.storage, "get_document_projections_batch", _spy)

    # Metadata-only filter: no doc-key leaf → engine short-circuits, so only the
    # facade's always-on document-upgrade pass calls the coordinator (1 call).
    await kb.recall(
        "alpha bravo charlie",
        namespace=namespace_id,
        limit=20,
        mode=SearchMode.VECTOR,
        filter={"metadata.tag": "urgent"},
    )
    assert len(calls) == 1, (
        f"a metadata-only filter must NOT trigger the engine's hydration fetch "
        f"(only the facade document-upgrade pass should call the coordinator); got {len(calls)} calls"
    )

    # Doc-key filter: source_name leaf → engine hydrates (1) + facade upgrade (1).
    calls.clear()
    await kb.recall(
        "alpha bravo charlie",
        namespace=namespace_id,
        limit=20,
        mode=SearchMode.VECTOR,
        filter={"source_name": "linear"},
    )
    assert len(calls) == 2, (
        f"a doc-key filter must add the engine's hydration fetch on top of the "
        f"facade document-upgrade pass (2 calls total); got {len(calls)} calls"
    )


# ===========================================================================
# System date-key narrowing — occurred_at / created_at / source_timestamp.
# ===========================================================================
#
# The embedded SQLite store returns tz-NAIVE datetimes (its SQLAlchemy DateTime
# column is not timezone-aware), so the system date-key compare lands a naive
# stored value next to the filter's tz-aware operand. The Python compiler aligns
# both to UTC at the comparison boundary, so a system date-key predicate narrows
# the live result instead of raising. These tests pin that on the real store for
# all three system date keys, via $gte / $lte / range.
#
# occurred_at and created_at are POST-FILTER-only on Chronicle, so the $gte / $lte
# / range trio narrows purely by the post-filter compare — the cleanest exercise
# of the naive/aware alignment. source_timestamp also PUSHES DOWN into the recency
# window, whose embedded-store LanceDB pre-filter screens on the ingest-time
# created_at column; an UPPER bound there clamps any row whose created_at ("now")
# exceeds the bound, independent of source_timestamp. So source_timestamp's $gte
# lower bound narrows by the post-filter as expected, while $lte / range carrying
# an upper bound are asserted only to NOT RAISE (the tz-compare contract) — the
# pushdown clamp that bounds their exact result set is a separate Chronicle
# recency characteristic, not the compare under test.


async def test_system_occurred_at_date_filter_narrows(kb: Khora) -> None:
    """The system ``occurred_at`` date key narrows to the in-range row.

    Seeds one in-range and one out-of-range row whose effective event time is set
    via ``source_timestamp`` (the embedded write path leaves the literal
    ``occurred_at`` column null and the engine derives the effective event time
    from ``source_timestamp``). A system ``occurred_at`` ``$gte`` filter returns
    only the in-range row — the naive stored datetime is aligned to UTC before the
    compare, so it narrows rather than raising ``TypeError``.
    """
    ns = await kb.create_namespace()
    namespace_id: UUID = ns.namespace_id

    in_range = "occurred at filter probe in range"
    out_range = "occurred at filter probe out of range"
    await kb.remember(
        content=in_range,
        namespace=namespace_id,
        title="in_range",
        source_timestamp=datetime(2026, 5, 1, tzinfo=UTC),
        entity_types=["PERSON"],
        relationship_types=["KNOWS"],
        expertise=_no_event_fact_extraction(),
    )
    await kb.remember(
        content=out_range,
        namespace=namespace_id,
        title="out_range",
        source_timestamp=datetime(2020, 1, 1, tzinfo=UTC),
        entity_types=["PERSON"],
        relationship_types=["KNOWS"],
        expertise=_no_event_fact_extraction(),
    )

    result = await kb.recall(
        "occurred at filter probe",
        namespace=namespace_id,
        limit=20,
        mode=SearchMode.VECTOR,
        filter={"occurred_at": {"$gte": "2026-01-01T00:00:00Z"}},
    )

    returned = {c.content for c in result.chunks}
    assert returned == {in_range}, "system occurred_at filter must narrow to exactly the in-range row"


async def test_system_occurred_at_date_filter_lte_and_range(kb: Khora) -> None:
    """The system ``occurred_at`` date key also narrows via ``$lte`` and a range.

    ``occurred_at`` is post-filter-only (it does not push into the recency
    window), so the upper-bound and range forms narrow purely by the naive/aware
    compare. Two recent rows split at an April mid-point: ``$gte`` keeps the later
    row, ``$lte`` keeps the earlier row, an April-only range keeps neither, and a
    range that brackets both keeps both — none raising on the naive stored value.
    """
    ns = await kb.create_namespace()
    namespace_id: UUID = ns.namespace_id

    earlier = "occurred at lte probe earlier"
    later = "occurred at lte probe later"
    await kb.remember(
        content=earlier,
        namespace=namespace_id,
        title="occ_earlier",
        source_timestamp=datetime(2026, 3, 1, tzinfo=UTC),
        entity_types=["PERSON"],
        relationship_types=["KNOWS"],
        expertise=_no_event_fact_extraction(),
    )
    await kb.remember(
        content=later,
        namespace=namespace_id,
        title="occ_later",
        source_timestamp=datetime(2026, 5, 1, tzinfo=UTC),
        entity_types=["PERSON"],
        relationship_types=["KNOWS"],
        expertise=_no_event_fact_extraction(),
    )

    async def _recall(filter_: dict[str, Any]) -> set[str]:
        result = await kb.recall(
            "occurred at lte probe",
            namespace=namespace_id,
            limit=20,
            mode=SearchMode.VECTOR,
            filter=filter_,
        )
        return {c.content for c in result.chunks}

    assert await _recall({"occurred_at": {"$gte": "2026-04-01T00:00:00Z"}}) == {later}
    assert await _recall({"occurred_at": {"$lte": "2026-04-01T00:00:00Z"}}) == {earlier}
    assert await _recall({"occurred_at": {"$gte": "2026-04-01T00:00:00Z", "$lte": "2026-04-30T00:00:00Z"}}) == set()
    assert await _recall({"occurred_at": {"$gte": "2026-02-01T00:00:00Z", "$lte": "2026-06-30T00:00:00Z"}}) == {
        earlier,
        later,
    }


async def test_system_source_timestamp_date_filter_narrows(kb: Khora) -> None:
    """The system ``source_timestamp`` date key compares without raising via ``$gte`` / ``$lte`` / range.

    ``source_timestamp`` is the LITERAL column value on the record (not a COALESCE
    like ``occurred_at``), so it exercises the naive-stored compare on the raw
    column. Two recent rows (Mar / May 2026) keep both candidates inside the
    recency window.

    The ``$gte`` LOWER bound narrows by the post-filter compare — exactly the
    naive/aware alignment under test — keeping only the later row. ``$lte`` / range
    carry an UPPER bound, which ``source_timestamp`` ALSO pushes into the recency
    window; that push's embedded-store LanceDB pre-filter screens on the ingest
    ``created_at`` column ("now"), so an upper bound before "now" clamps the
    candidate set independent of ``source_timestamp``. The exact upper-bound result
    is therefore a Chronicle recency characteristic, not the tz compare — so this
    test asserts ``$lte`` / range only COMPLETE WITHOUT RAISING and stay within the
    corpus (the contract the tz fix actually owns), while the ``$gte`` lower bound
    pins the post-filter narrowing.
    """
    ns = await kb.create_namespace()
    namespace_id: UUID = ns.namespace_id

    earlier = "source timestamp probe earlier"
    later = "source timestamp probe later"
    corpus = {earlier, later}
    await kb.remember(
        content=earlier,
        namespace=namespace_id,
        title="st_earlier",
        source_timestamp=datetime(2026, 3, 1, tzinfo=UTC),
        entity_types=["PERSON"],
        relationship_types=["KNOWS"],
        expertise=_no_event_fact_extraction(),
    )
    await kb.remember(
        content=later,
        namespace=namespace_id,
        title="st_later",
        source_timestamp=datetime(2026, 5, 1, tzinfo=UTC),
        entity_types=["PERSON"],
        relationship_types=["KNOWS"],
        expertise=_no_event_fact_extraction(),
    )

    async def _recall(filter_: dict[str, Any]) -> set[str]:
        result = await kb.recall(
            "source timestamp probe",
            namespace=namespace_id,
            limit=20,
            mode=SearchMode.VECTOR,
            filter=filter_,
        )
        return {c.content for c in result.chunks}

    # $gte lower bound at the April mid-point narrows to the later row (post-filter).
    assert await _recall({"source_timestamp": {"$gte": "2026-04-01T00:00:00Z"}}) == {later}
    # $lte / range carry an upper bound — assert they don't raise and stay within
    # the corpus; their exact set is bounded by the recency pushdown, not the tz fix.
    assert await _recall({"source_timestamp": {"$lte": "2026-04-01T00:00:00Z"}}) <= corpus
    assert (
        await _recall({"source_timestamp": {"$gte": "2026-04-01T00:00:00Z", "$lte": "2026-04-30T00:00:00Z"}}) <= corpus
    )


async def test_system_created_at_date_filter_narrows(kb: Khora) -> None:
    """The system ``created_at`` date key compares without raising on the naive column.

    ``created_at`` is stamped at ingest (always "now"), so it is the naive-stored
    column whose value the test does not control. The contract pinned here is that
    the system date-key compare no longer raises ``TypeError`` and narrows
    sanely: a ``$gte`` lower bound well in the past keeps every seeded row, a
    ``$lte`` upper bound well in the past drops them all, and a range bracketing
    ingest time keeps them — all on the naive stored datetime.
    """
    ns = await kb.create_namespace()
    namespace_id: UUID = ns.namespace_id
    content_to_label = await _seed(kb, namespace_id)
    all_labels = set(content_to_label.values())

    async def _recall(filter_: dict[str, Any]) -> set[str]:
        result = await kb.recall(
            "alpha bravo charlie",
            namespace=namespace_id,
            limit=20,
            mode=SearchMode.VECTOR,
            filter=filter_,
        )
        return _labels_returned(result, content_to_label)

    # A past lower bound keeps everything; a past upper bound drops everything; a
    # range from the past to the far future brackets ingest time and keeps all.
    assert await _recall({"created_at": {"$gte": "2000-01-01T00:00:00Z"}}) == all_labels
    assert await _recall({"created_at": {"$lte": "2000-01-01T00:00:00Z"}}) == set()
    assert await _recall({"created_at": {"$gte": "2000-01-01T00:00:00Z", "$lte": "2100-01-01T00:00:00Z"}}) == all_labels


# ===========================================================================
# engine_info["filter"] — the honest FilterPushdownReport through the FULL facade.
# ===========================================================================
#
# The public ``Khora.recall(filter=...)`` path threads the AST to Chronicle, which
# emits a canonical ``FilterPushdownReport`` under ``engine_info["filter"]``; the
# facade passes it through verbatim. These tests pin that the END-TO-END public
# call surfaces the canonical report — partial-pushdown split, fully-pushed, and
# no-filter — using the real canonical leaf-key strings (``source_timestamp`` is
# the one Chronicle pushes; ``occurred_at`` / ``metadata.<x>`` are post-filter-only).


async def test_facade_filter_report_only_source_timestamp_pushes(kb: Khora) -> None:
    """A 3-leaf filter where only ``source_timestamp`` folds into the window.

    Through the full public ``recall(filter=...)``: of ``source_timestamp`` (the
    recency window's primary axis), ``occurred_at`` (event-time axis), and
    ``metadata.tag``, only ``source_timestamp`` pushes down. The surfaced report
    names exactly that split and round-trips through the canonical model.
    """
    from khora.filter import FilterPushdownReport

    ns = await kb.create_namespace()
    namespace_id: UUID = ns.namespace_id
    await _seed(kb, namespace_id)

    result = await kb.recall(
        "alpha bravo charlie",
        namespace=namespace_id,
        limit=20,
        mode=SearchMode.VECTOR,
        filter={
            "source_timestamp": {"$gte": _IN_BOUND},
            "occurred_at": {"$gte": _IN_BOUND},
            "metadata.tag": {"$in": ["urgent", "release"]},
        },
    )

    report = result.engine_info["filter"]
    FilterPushdownReport.model_validate(report)  # raises on shape drift
    assert report["pushed_keys"] == ["source_timestamp"]
    assert report["post_filtered_keys"] == ["metadata.tag", "occurred_at"]
    assert report["pushed_down"] is False
    assert report["post_filtered"] is True
    assert report["channels"] == {
        "chunks": {"pushed_keys": ["source_timestamp"], "post_filtered_keys": ["metadata.tag", "occurred_at"]}
    }


async def test_facade_filter_report_three_predicate_all_post_filtered(kb: Khora) -> None:
    """The corpus three-predicate filter pushes NOTHING (no source_timestamp leaf).

    ``_RECALL_FILTER`` constrains ``source_name`` / ``occurred_at`` /
    ``metadata.tag`` — none of which is the source_timestamp axis Chronicle pushes —
    so every leaf is post-filter-only. The report's ``pushed_keys`` is empty and all
    three leaves land in ``post_filtered_keys``.
    """
    from khora.filter import FilterPushdownReport

    ns = await kb.create_namespace()
    namespace_id: UUID = ns.namespace_id
    await _seed(kb, namespace_id)

    result = await kb.recall(
        "alpha bravo charlie",
        namespace=namespace_id,
        limit=20,
        mode=SearchMode.VECTOR,
        filter=_RECALL_FILTER,
    )

    report = result.engine_info["filter"]
    FilterPushdownReport.model_validate(report)
    assert report["pushed_keys"] == []
    assert report["post_filtered_keys"] == ["metadata.tag", "occurred_at", "source_name"]
    assert report["pushed_down"] is False
    assert report["post_filtered"] is True


async def test_facade_filter_report_no_filter_reports_nothing_narrowed(kb: Khora) -> None:
    """The no-filter public recall surfaces the canonical empty report.

    Nothing pushed, nothing post-filtered, both flags False, empty key lists, and
    the single Chronicle ``chunks`` channel present with empty lists.
    """
    from khora.filter import FilterPushdownReport

    ns = await kb.create_namespace()
    namespace_id: UUID = ns.namespace_id
    await _seed(kb, namespace_id)

    result = await kb.recall(
        "alpha bravo charlie",
        namespace=namespace_id,
        limit=20,
        mode=SearchMode.VECTOR,
    )

    report = result.engine_info["filter"]
    FilterPushdownReport.model_validate(report)
    assert report["pushed_down"] is False
    assert report["post_filtered"] is False
    assert report["pushed_keys"] == []
    assert report["post_filtered_keys"] == []
    assert report["channels"] == {"chunks": {"pushed_keys": [], "post_filtered_keys": []}}
