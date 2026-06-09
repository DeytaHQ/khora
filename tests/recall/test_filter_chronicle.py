"""Full-stack integration tests for the recall-filter on Chronicle + sqlite_lance.

This is the Chronicle counterpart to ``test_filter_skeleton_pgvector.py`` (V1):
where V1 drives the public ``Khora.recall(filter=...)`` API end-to-end through
the Skeleton engine into a real pgvector ``khora_chunks`` lake, this file drives
the SAME contract through the Chronicle engine over a real embedded SQLite +
LanceDB store. The engine→backend wiring threads ``filter_ast`` from the facade
through Chronicle's recall path into the deterministic post-filter, so row
filtering is effective end-to-end — these tests pin that contract on the
embedded stack.

Unlike ``test_chronicle_filter_composition.py`` (which is the MOCKED, unit-level
proof of how Chronicle composes the two compiled filter halves on a
``MagicMock`` storage), this file's value-add is the LIVE corpus isolation: real
``Khora.remember()`` writes, real ``Khora.recall(filter=...)`` reads, real
sqlite_lance roundtrip. It does NOT re-mock what the composition suite already
proves; it proves the same logical narrowing survives the live write/read path.

EQUIVALENCE TO V1 — same corpus + split, Chronicle-carried axes
===============================================================
V1's filter is ``source_name == "linear" AND occurred_at >= 2026-04-05 AND
metadata.tag IN {"urgent", "release"}``. This file reuses V1's shared corpus and
its exact 5-in-scope / 5-out-of-scope split (each out-of-scope row violating
EXACTLY ONE predicate, one in-scope row sitting EXACTLY at the date bound). Two
of V1's three predicate KEYS, however, do not function on the REAL Chronicle
sqlite_lance recall path, so the three-predicate AND is expressed over the axes
Chronicle actually carries on the real store — metadata keys — standing in for
V1's ``source_name`` (a denormalized document key) and ``occurred_at`` (a system
date key). The in-scope SET is identical; the contract proven here is "a
real-store multi-predicate AND narrows to exactly V1's in-scope set". The system
``occurred_at`` axis is pinned separately by a strict-xfail (below).

Why each substitution (verified against this exact sqlite_lance fixture, not
assumed):

* ``source_name`` → ``metadata.channel`` ($eq). Chronicle's recall candidates do
  NOT hydrate ``chunk.source_document`` on the recall path, so the denormalized
  document keys (``source_name``, ``source``, ``source_type``, ``title``, ...)
  all resolve ABSENT on the live post-filter record — a positive predicate on any
  of them returns empty. The engine anticipates this ("no faithful fallback
  exists", ``engine._chunk_to_record``); it is a documented limitation, not a
  bug. The source axis therefore rides a metadata key the corpus writes and the
  post-filter reads.
* ``occurred_at`` (system date key) → ``metadata.when`` ($gte, ``$date``
  operand). The effective event time IS carried on the record, but the embedded
  SQLite store returns tz-NAIVE datetimes and the shared system date-key compare
  currently raises when it puts a naive stored value next to a tz-aware operand.
  The metadata-date path normalizes both sides to UTC, so a ``$date`` predicate
  drives the date split cleanly. The system ``occurred_at`` crash is pinned by a
  strict-xfail test below so the moment the compiler normalizes tz, that test
  flips to a hard failure and signals it is time to drop the workaround.

The all-metadata filter is also why the unindexed-metadata telemetry assertion
(Scenario 2) is meaningful here: every in-scope predicate leaf is a metadata
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
from khora.extraction.skills import ExpertiseConfig
from khora.extraction.skills.base import EventExtractionConfig, FactExtractionConfig
from khora.filter import telemetry as filter_telemetry
from khora.khora import Khora
from khora.query import SearchMode

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
# The three-predicate filter under test (Chronicle-carried equivalent of V1):
#   metadata.channel == "linear"                 (stands in for V1's source_name)
#   AND metadata.when >= 2026-04-05 ($date)      (stands in for V1's occurred_at)
#   AND metadata.tag IN {"urgent", "release"}    (identical to V1)
#
# Why all three axes are METADATA keys (not V1's system / doc keys): on the
# Chronicle sqlite_lance recall DTO the document-projection keys (source_name /
# source / title / ...) resolve absent, and the SYSTEM occurred_at date compare
# raises on the embedded store's naive datetimes (pinned by the xfail below). So
# the source and date axes ride metadata — the only keys this engine both writes
# and reads on the live post-filter record. The in/out SPLIT is identical to V1's.
#
# Content is DISTINCT per row so each remember produces its own chunk (Chronicle
# dedupes by content checksum, so identical content would collapse to one chunk);
# the shared "alpha bravo charlie" token lets the constant-vector retrieval
# surface every row before the filter.
#
# Deliberately NOT covered here (this file is the live corpus-equivalence gate and
# mirrors V1's split exactly): the tag-is-explicitly-null, unanchored-chunk
# (occurred_at=None AND source_timestamp=None), and pgvector occurred_at=None DTO
# shapes. Those edge shapes are exercised by the mocked unit suite,
# tests/recall/test_chronicle_filter_composition.py.

_IN_BOUND = "2026-04-05T00:00:00Z"  # the metadata.when lower bound (inclusive)

# label -> (channel, when ISO, tag | None)
_IN_SCOPE: dict[str, tuple[str, str, str | None]] = {
    # All three predicates satisfied. Boundary, mid, and future "when"; both
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
    # Predicate 1 (channel) violated; date + tag are in-scope.
    "out_wrong_source": ("slack", "2026-05-15T00:00:00Z", "urgent"),
    # Predicate 2 (when) violated: one second before the inclusive bound.
    "out_too_old": ("linear", "2026-04-04T23:59:59Z", "release"),
    # Predicate 3 (tag) violated: tag present but outside the allowed set.
    "out_wrong_tag": ("linear", "2026-05-20T00:00:00Z", "backlog"),
    # Predicate 3 (tag) violated: tag key absent entirely.
    "out_missing_tag": ("linear", "2026-05-25T00:00:00Z", None),
    # Predicate 1 (channel) violated via absence: channel key omitted at write.
    "out_null_source": (None, "2026-05-30T00:00:00Z", "release"),
}

# The live three-predicate recall filter. ``$date`` wraps the date operand so the
# metadata-date path parses both sides to UTC-aware datetimes (sidestepping the
# naive-datetime crash a SYSTEM date-key predicate hits on the embedded store).
_RECALL_FILTER = {
    "metadata.channel": "linear",
    "metadata.when": {"$gte": {"$date": _IN_BOUND}},
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
    """
    for label, (channel, when, tag) in {**_IN_SCOPE, **_OUT_OF_SCOPE}.items():
        metadata: dict[str, Any] = {"when": when}
        if channel is not None:
            metadata["channel"] = channel
        if tag is not None:
            metadata["tag"] = tag
        await kb.remember(
            content=_content_for(label),
            namespace=namespace_id,
            title=label,
            metadata=metadata,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
            expertise=_no_event_fact_extraction(),
        )
    return {_content_for(label): label for label in {**_IN_SCOPE, **_OUT_OF_SCOPE}}


def _labels_returned(result: Any, content_to_label: dict[str, str]) -> set[str]:
    return {content_to_label[c.content] for c in result.chunks}


# ===========================================================================
# Scenario 1 — green equivalence: the three-predicate AND returns EXACTLY the
# in-scope rows, matching V1's logical in-scope set.
# ===========================================================================


async def test_filter_returns_exactly_in_scope_chunks(kb: Khora) -> None:
    """``recall(filter=...)`` narrows the live result to exactly the 5 in-scope
    rows — no out-of-scope leak.

    End-to-end proof that the engine→backend wiring filters rows on the embedded
    stack: the facade builds the AST, Chronicle threads it to the deterministic
    post-filter, and the candidate set is narrowed before top-k. Every
    out-of-scope row violates exactly one predicate, so a leak would name the
    broken one. The returned set is the SAME logical in-scope set as V1's.
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
# System date-key crash — strict xfail (DO NOT skip).
# ===========================================================================
#
# The metadata-date path the green test above rides sidesteps a real defect:
# filtering on the SYSTEM date key (``occurred_at``) over the embedded store
# compares a tz-naive stored datetime against the filter's tz-aware operand and
# raises. This test pins that defect with strict=True so that the instant the
# compiler learns to normalize tz, the test XPASSes (a hard failure) — the signal
# to delete this marker AND fold the date axis back onto the system ``occurred_at``
# key in the green test, matching V1 one-for-one.


@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason=(
        "system date-key compare on sqlite_lance compares a tz-naive stored datetime "
        "against a tz-aware operand and raises; the metadata-date path is unaffected. "
        "Drop this marker when the compiler normalizes tz on the system-key range path."
    ),
)
async def test_system_occurred_at_date_filter_narrows(kb: Khora) -> None:
    """The system ``occurred_at`` date key SHOULD narrow to the in-range row.

    Seeds one in-range and one out-of-range row whose effective event time is set
    via ``source_timestamp`` (the embedded write path leaves the literal
    ``occurred_at`` column null and the engine derives the effective event time
    from ``source_timestamp``). A system ``occurred_at`` ``$gte`` filter SHOULD
    return only the in-range row — but currently raises ``TypeError`` on the naive
    stored datetime, so this test is a strict xfail. The assertion below is the
    behaviour it WILL pin once the tz defect is fixed.
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


# ===========================================================================
# Scenario 2 — post-filter increments unindexed_metadata, with negative controls.
# ===========================================================================
#
# ``compile_python`` emits ``khora.recall.filter.unindexed_metadata`` once per
# metadata leaf at compile time, so a metadata-bearing filter recall fires it.
# The two negative controls rule out the counter firing for the wrong reason: a
# non-metadata filter (no metadata leaf) and a no-filter recall must both leave
# it silent.


class _RecordingCounter:
    """Captures ``.add(value, attributes=...)`` calls for assertions."""

    def __init__(self) -> None:
        self.adds: list[tuple[float, dict[str, Any]]] = []

    def add(self, value: float, attributes: Any = None) -> None:
        self.adds.append((value, dict(attributes or {})))


@pytest.fixture
def recording_counter(monkeypatch: pytest.MonkeyPatch) -> _RecordingCounter:
    """Replace the unindexed-metadata counter singleton with a recording fake.

    Same monkeypatch-the-singleton hook the existing recall-filter telemetry
    tests use: pre-seeding the module global makes ``record_unindexed_metadata``
    land on the fake (the lazy getter returns the already-set value).
    """
    counter = _RecordingCounter()
    monkeypatch.setattr(filter_telemetry, "_unindexed_metadata_counter", counter)
    return counter


async def test_metadata_filter_fires_unindexed_metadata(kb: Khora, recording_counter: _RecordingCounter) -> None:
    """A metadata-bearing filter recall fires ``unindexed_metadata`` on the live
    post-filter path, carrying the leaf's comparison op.

    The three-predicate filter compiles three metadata leaves to Python
    predicates, so the counter observes at least one add; assert it carries a
    real leaf op (``$eq`` from ``metadata.channel``). ``>= 1`` is robust whether
    emission is per-compile or (hypothetically) per-evaluation.
    """
    ns = await kb.create_namespace()
    namespace_id: UUID = ns.namespace_id
    await _seed(kb, namespace_id)

    await kb.recall(
        "alpha bravo charlie",
        namespace=namespace_id,
        limit=20,
        mode=SearchMode.VECTOR,
        filter=_RECALL_FILTER,
    )

    adds = recording_counter.adds
    assert len(adds) >= 1, "a metadata-bearing filter recall must fire unindexed_metadata"
    assert any(a[1].get("op") == "$eq" for a in adds), f"expected a $eq leaf op among the observations; got {adds}"


async def test_non_metadata_filter_silent_for_unindexed_metadata(
    kb: Khora, recording_counter: _RecordingCounter
) -> None:
    """Negative control: a non-metadata filter does not touch metadata, so the
    unindexed_metadata counter stays silent.

    ``{"source": {"$ne": "no-such-source"}}`` is a document-projection key with no
    metadata leaf and no datetime compare. On Chronicle's live recall DTO the
    ``source`` key resolves absent, so ``$ne`` matches every row (absent is
    "not equal") — the recall does not narrow and, crucially, does not crash. What
    matters here is that NO metadata leaf compiles, so the counter must not fire.
    This pins that Scenario 2's positive signal comes from the metadata leaves,
    not from the mere presence of a filter.
    """
    ns = await kb.create_namespace()
    namespace_id: UUID = ns.namespace_id
    await _seed(kb, namespace_id)

    await kb.recall(
        "alpha bravo charlie",
        namespace=namespace_id,
        limit=20,
        mode=SearchMode.VECTOR,
        filter={"source": {"$ne": "no-such-source"}},
    )

    assert recording_counter.adds == [], "a non-metadata filter must leave unindexed_metadata silent"


async def test_no_filter_recall_silent_for_unindexed_metadata(kb: Khora, recording_counter: _RecordingCounter) -> None:
    """Negative control: no filter → no compile → no counter."""
    ns = await kb.create_namespace()
    namespace_id: UUID = ns.namespace_id
    await _seed(kb, namespace_id)

    await kb.recall(
        "alpha bravo charlie",
        namespace=namespace_id,
        limit=20,
        mode=SearchMode.VECTOR,
    )

    assert recording_counter.adds == [], "a no-filter recall must leave unindexed_metadata silent"
