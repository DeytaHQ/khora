"""Chronicle SQLite + LanceDB integration tests.

Mirrors ``test_chronicle_pg.py`` for the embedded stack. Chronicle on
sqlite_lance was broken at the persistence layer until issue #529 (PR
#528) wired ``write_events`` / ``write_facts`` /
``query_active_facts_for_subject`` through the relational adapter when
the coordinator's vector adapter (LanceDB) doesn't carry chronicle
methods. This file is the matrix-tier regression test for that fix.

Why this exists separately from ``test_chronicle_pg.py``:
* Chronicle has four channels (semantic / BM25 / temporal / entity).
  Each lands on different adapters on the embedded path
  (vectors → LanceDB, BM25/temporal → SQLite FTS5/columns, entities →
  the SQLite graph adapter) — distinct from pgvector's single-adapter
  layout. Cross-channel behaviour needs its own integration cell.
* Issue #526 (FTS5 punctuation escaping) and issue #529 (chronicle
  persistence) both surfaced on the embedded path. The end-to-end path
  through ``Khora.recall()`` / ``Khora.remember()`` was previously
  untested on this backend.

How the LLM is mocked:
* ``LiteLLMEmbedder.embed_batch`` / ``embed`` return content-derived
  unit vectors of dimension ``EMBED_DIM=32`` (the sqlite_lance default).
* ``LLMEntityExtractor.extract_multi`` is replaced with a registry stub
  identical to the chronicle-pg / vectorcypher-sqlite_lance pattern.
* ``EventExtractor.extract_events`` and ``FactExtractor.extract_facts``
  are stubbed only when the test enables them (most tests disable
  per-chunk event/fact extraction for determinism).

How to run locally::

    uv run pytest tests/integration/matrix/test_chronicle_sqlite_lance.py \\
        -v -m integration --no-cov

No Docker / Postgres / Neo4j needed.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest

try:
    import aiosqlite  # noqa: F401
    import lancedb  # noqa: F401

    _HAS_EMBEDDED = True
except ImportError:
    _HAS_EMBEDDED = False

from khora.config import KhoraConfig
from khora.config.schema import SQLiteLanceConfig
from khora.engines.chronicle.compression import MemoryFact
from khora.engines.chronicle.events import ChronicleEvent
from khora.extraction.extractors.base import (
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
)
from khora.extraction.skills import ExpertiseConfig
from khora.extraction.skills.base import EventExtractionConfig, FactExtractionConfig
from khora.khora import Khora

EMBED_DIM = 32  # sqlite_lance default; keeps LanceDB ANN builds cheap in tmp_path


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _HAS_EMBEDDED,
        reason="aiosqlite/lancedb not installed (pip install khora[sqlite_lance])",
    ),
]


# ---------------------------------------------------------------------------
# Deterministic embedder + extractor stubs
# ---------------------------------------------------------------------------


def _embed_for(text_in: str) -> list[float]:
    """SHA-256 → 32 unit-vector entries. Same text ⇒ same vector."""
    seed = hashlib.sha256(text_in.encode("utf-8")).digest()
    raw = [(seed[i % len(seed)] - 128) / 128.0 for i in range(EMBED_DIM)]
    norm = sum(x * x for x in raw) ** 0.5 or 1.0
    return [x / norm for x in raw]


_EXTRACTION_REGISTRY: dict[str, ExtractionResult] = {}


def _plan_extraction(
    marker: str,
    entities: list[tuple[str, str]],
    relationships: list[tuple[str, str, str]] | None = None,
) -> None:
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


async def _stub_extract_multi(self: Any, texts: list[str], **_kwargs: Any) -> list[ExtractionResult]:
    out: list[ExtractionResult] = []
    for t in texts:
        matched = next(
            (r for marker, r in _EXTRACTION_REGISTRY.items() if marker in t),
            None,
        )
        out.append(matched if matched is not None else ExtractionResult())
    return out


async def _stub_embed_batch(self: Any, texts: list[str]) -> list[list[float]]:
    return [_embed_for(t) for t in texts]


async def _stub_embed(self: Any, text_in: str) -> list[float]:
    return _embed_for(text_in)


@pytest.fixture(autouse=True)
def _patch_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable the network LLM/embedder by default. Tests can stack additional
    monkeypatches (e.g. event/fact extractors) on top via ``monkeypatch``.
    """
    _EXTRACTION_REGISTRY.clear()
    monkeypatch.setattr(
        "khora.extraction.embedders.litellm.LiteLLMEmbedder.embed_batch",
        _stub_embed_batch,
    )
    monkeypatch.setattr(
        "khora.extraction.embedders.litellm.LiteLLMEmbedder.embed",
        _stub_embed,
    )
    monkeypatch.setattr(
        "khora.extraction.extractors.llm.LLMEntityExtractor.extract_multi",
        _stub_extract_multi,
    )


# ---------------------------------------------------------------------------
# Per-test embedded Khora fixture
# ---------------------------------------------------------------------------


def _no_event_fact_extraction() -> ExpertiseConfig:
    """Disable Chronicle's per-chunk event/fact extraction.

    Tests that exercise the persistence path enable it explicitly via
    ``_event_fact_extraction()`` + extractor monkeypatches.
    """
    return ExpertiseConfig(
        name="chronicle-sqlite_lance-integ",
        events=EventExtractionConfig(enabled=False),
        facts=FactExtractionConfig(enabled=False),
    )


def _event_fact_extraction() -> ExpertiseConfig:
    return ExpertiseConfig(
        name="chronicle-sqlite_lance-integ-with-extraction",
        events=EventExtractionConfig(enabled=True),
        facts=FactExtractionConfig(enabled=True),
    )


@pytest.fixture
async def kb(tmp_path: Path) -> AsyncIterator[Khora]:
    """Per-test Chronicle Khora bound to an embedded SQLite+LanceDB pair."""
    db_path = str(tmp_path / "khora.db")
    lance_path = str(tmp_path / "khora.lance")

    config = KhoraConfig()
    config.storage.backend = "sqlite_lance"
    config.storage.sqlite_lance = SQLiteLanceConfig(
        db_path=db_path,
        lance_path=lance_path,
        embedding_dimension=EMBED_DIM,
    )
    config.llm.embedding_dimension = EMBED_DIM
    config.storage.embedding_dimension = EMBED_DIM
    # Chronicle's channels are all SQL/LanceDB on embedded — no graph URL.
    config.neo4j_url = None
    config.pipelines.chunk_size = 1024
    config.pipelines.extract_entities = True
    config.pipelines.selective_extraction = False

    kb = Khora(config, engine="chronicle", run_migrations=True)
    await kb.connect()
    try:
        yield kb
    finally:
        await kb.disconnect()


@pytest.fixture
async def namespace_id(kb: Khora) -> UUID:
    ns = await kb.create_namespace()
    return ns.namespace_id


async def _remember(
    kb: Khora,
    *,
    namespace_id: UUID,
    content: str,
    title: str = "",
    expertise: ExpertiseConfig | None = None,
) -> Any:
    return await kb.remember(
        content=content,
        namespace=namespace_id,
        title=title,
        entity_types=["PERSON", "CONCEPT", "EVENT"],
        relationship_types=["KNOWS", "ATTENDED", "RELATES_TO"],
        expertise=expertise or _no_event_fact_extraction(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_chronicle_remember_recall_roundtrip(kb: Khora, namespace_id: UUID) -> None:
    """Ingest 3 docs, recall on a known token, assert the right chunk surfaces."""
    contents = [
        "Marie Curie won the Nobel Prize in Physics in 1903.",
        "Pierre Curie was a French physicist and Marie's husband.",
        "Radium and polonium were discovered by the Curies.",
    ]
    for c in contents:
        await _remember(kb, namespace_id=namespace_id, content=c)

    result = await kb.recall("Nobel Prize Physics", namespace=namespace_id, limit=3)
    assert result.chunks, "expected at least one chunk back"
    text_blob = " ".join(c.content for c in result.chunks).lower()
    assert "nobel" in text_blob


async def test_chronicle_events_and_facts_persist_via_engine(
    kb: Khora, namespace_id: UUID, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression for issue #529 at the **engine layer**.

    The fix in PR #528 wired chronicle persistence through the
    relational adapter on the embedded path. The integration test in that
    PR exercised the storage layer only (``coord.write_events`` directly).
    This test verifies the same fix end-to-end through
    ``Khora.remember()`` → chronicle engine → coordinator → storage.

    Before the fix, the engine silently dropped every event and fact
    extracted by the LLM (``AttributeError`` caught as generic Exception
    and logged as a warning).
    """
    captured_events: list[ChronicleEvent] = []
    captured_facts: list[MemoryFact] = []

    async def _stub_extract_events(
        self: Any, text: str, *, chunk_id: Any = None, namespace_id: Any = None, **_kwargs: Any
    ) -> list[ChronicleEvent]:
        ev = ChronicleEvent(
            id=uuid4(),
            chunk_id=chunk_id,
            namespace_id=namespace_id,
            subject="Marie Curie",
            verb="won",
            object="Nobel Prize",
            observation_date=datetime.now(UTC),
            referenced_date=datetime(1903, 12, 10, tzinfo=UTC),
            confidence=0.95,
            source_text=text[:200],
        )
        captured_events.append(ev)
        return [ev]

    async def _stub_extract_facts(
        self: Any, text: str, *, chunk_id: Any = None, namespace_id: Any = None, **_kwargs: Any
    ) -> list[MemoryFact]:
        fact = MemoryFact(
            id=uuid4(),
            namespace_id=namespace_id,
            subject="Marie Curie",
            predicate="won",
            object_="Nobel Prize",
            fact_text="Marie Curie won the Nobel Prize.",
            confidence=0.95,
            source_chunk_ids=[chunk_id] if chunk_id else [],
        )
        captured_facts.append(fact)
        return [fact]

    monkeypatch.setattr(
        "khora.engines.chronicle.events.EventExtractor.extract_events",
        _stub_extract_events,
    )
    monkeypatch.setattr(
        "khora.engines.chronicle.compression.FactExtractor.extract_facts",
        _stub_extract_facts,
    )

    result = await _remember(
        kb,
        namespace_id=namespace_id,
        content="Marie Curie won the Nobel Prize in Physics in 1903.",
        expertise=_event_fact_extraction(),
    )

    assert captured_events, "EventExtractor stub never ran — extraction wiring broken"
    assert captured_facts, "FactExtractor stub never ran — extraction wiring broken"

    # chronicle.engine._extract_and_persist_events catches `write_events`
    # exceptions (the #529 failure mode) and returns 0 from the helper, so the
    # `events_extracted` value in metadata is the canonical signal that
    # persistence succeeded end-to-end.
    events_persisted = result.metadata.get("events_extracted", 0)
    facts_persisted = result.metadata.get("facts_extracted", 0)
    assert events_persisted >= 1, (
        f"Issue #529 regression: extractor produced {len(captured_events)} event(s) "
        f"but engine reports {events_persisted} persisted. write_events dispatch is broken."
    )
    assert facts_persisted >= 1, (
        f"Issue #529 regression: extractor produced {len(captured_facts)} fact(s) "
        f"but engine reports {facts_persisted} persisted. write_facts dispatch is broken."
    )


def _facts_only_extraction() -> ExpertiseConfig:
    """Enable fact extraction only (events stay off for determinism)."""
    return ExpertiseConfig(
        name="chronicle-sqlite_lance-integ-facts-only",
        events=EventExtractionConfig(enabled=False),
        facts=FactExtractionConfig(enabled=True),
    )


def _stub_fact_extractor(monkeypatch: pytest.MonkeyPatch, fact_text: str) -> None:
    """Patch ``FactExtractor.extract_facts`` to emit one fact per chunk."""

    async def _stub_extract_facts(
        self: Any, text: str, *, chunk_id: Any = None, namespace_id: Any = None, **_kwargs: Any
    ) -> list[MemoryFact]:
        return [
            MemoryFact(
                id=uuid4(),
                namespace_id=namespace_id,
                subject="Marie Curie",
                predicate="won",
                object_="Nobel Prize",
                fact_text=fact_text,
                confidence=0.95,
                source_chunk_ids=[chunk_id] if chunk_id else [],
            )
        ]

    monkeypatch.setattr(
        "khora.engines.chronicle.compression.FactExtractor.extract_facts",
        _stub_extract_facts,
    )


async def test_chronicle_forget_deletes_memory_facts(
    kb: Khora, namespace_id: UUID, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression for issue #1140.

    ``memory_facts`` has no FK to chunks or documents - provenance is the
    non-FK ``source_chunk_ids`` array - so the chunks cascade from
    ``delete_document`` never touched it. Before the fix, ``forget()``
    left every fact extracted from the deleted document active with
    dangling ``source_chunk_ids``, retaining forgotten content (including
    PII distilled into ``fact_text``) indefinitely.
    """
    _stub_fact_extractor(monkeypatch, "Marie Curie won the Nobel Prize.")

    result = await _remember(
        kb,
        namespace_id=namespace_id,
        content="Marie Curie won the Nobel Prize in Physics in 1903.",
        expertise=_facts_only_extraction(),
    )
    assert result.metadata.get("facts_extracted", 0) >= 1, "test setup: no facts persisted"

    # Facts are stored under the row-level namespace id, not the stable one.
    resolved_ns = await kb.storage.resolve_namespace(namespace_id)
    facts_before = await kb.storage.query_active_facts_for_subject(resolved_ns, "Marie Curie")
    assert facts_before, "test setup: expected at least one active fact before forget"

    ok = await kb.forget(result.document_id, namespace=namespace_id)
    assert ok, "forget() must report success"

    facts_after = await kb.storage.query_active_facts_for_subject(resolved_ns, "Marie Curie")
    assert not facts_after, (
        f"Issue #1140 regression: {len(facts_after)} memory fact(s) extracted from the "
        f"forgotten document remain active with dangling source_chunk_ids"
    )


async def test_chronicle_forget_session_deletes_memory_facts(
    kb: Khora, namespace_id: UUID, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression for issue #1140 on the ``forget_session`` cascade (#620)."""
    _stub_fact_extractor(monkeypatch, "Marie Curie discovered radium.")

    session_id = uuid4()
    await kb.remember(
        content="Marie Curie discovered radium during her research in Paris.",
        namespace=namespace_id,
        entity_types=["PERSON", "CONCEPT", "EVENT"],
        relationship_types=["KNOWS", "ATTENDED", "RELATES_TO"],
        expertise=_facts_only_extraction(),
        session_id=session_id,
    )

    resolved_ns = await kb.storage.resolve_namespace(namespace_id)
    facts_before = await kb.storage.query_active_facts_for_subject(resolved_ns, "Marie Curie")
    assert facts_before, "test setup: expected at least one active fact before forget_session"

    deleted = await kb.forget_session(namespace_id, session_id)
    assert deleted == 1, f"forget_session must delete the session's document, got {deleted}"

    facts_after = await kb.storage.query_active_facts_for_subject(resolved_ns, "Marie Curie")
    assert not facts_after, (
        f"Issue #1140 regression: {len(facts_after)} memory fact(s) from the forgotten "
        f"session remain active - forget_session's #620 contract is broken one layer down"
    )


async def test_chronicle_recall_handles_punctuated_query(kb: Khora, namespace_id: UUID) -> None:
    """Regression for issue #526 at the **engine layer**.

    The PR #528 fix verified ``escape_fts5_query`` at the storage-adapter
    layer (``coord.search_fulltext_chunks``). This test pushes punctuated
    queries through the full ``Khora.recall()`` → chronicle engine path,
    catching any future regression that bypasses the escape (e.g. a new
    fusion mode that re-calls a raw FTS5 path).
    """
    await _remember(
        kb,
        namespace_id=namespace_id,
        content="Marie Curie won the Nobel Prize in Physics in 1903.",
    )

    # Each of these would have crashed prior to PR #528 if FTS5 is reached.
    for query in (
        "What did Curie win?",
        "did Curie: yes",
        "Curie (please)",
        "Curie AND Nobel",
        'say "hello" Curie',
    ):
        result = await kb.recall(query, namespace=namespace_id, limit=3)
        # chronicle returns chunks as (chunk, score) tuples; we only assert no
        # crash + iterable result here (the substantive #526 regression test
        # is at the storage layer; this catches a future fusion-mode regression
        # that bypasses escape_fts5_query).
        assert isinstance(result.chunks, list), f"recall must not crash on {query!r}"


async def test_chronicle_namespace_isolation(kb: Khora) -> None:
    """Multi-tenant correctness: recall in ns1 must not surface ns2 chunks."""
    ns1 = (await kb.create_namespace()).namespace_id
    ns2 = (await kb.create_namespace()).namespace_id

    await _remember(kb, namespace_id=ns1, content="Marie Curie won the Nobel Prize.")
    await _remember(kb, namespace_id=ns2, content="Albert Einstein won the Nobel Prize.")

    r1 = await kb.recall("Curie", namespace=ns1, limit=5)
    r2 = await kb.recall("Einstein", namespace=ns2, limit=5)

    text1 = " ".join(c.content for c in r1.chunks).lower()
    text2 = " ".join(c.content for c in r2.chunks).lower()
    assert "einstein" not in text1, "ns1 results leaked from ns2"
    assert "curie" not in text2, "ns2 results leaked from ns1"


async def test_chronicle_recall_empty_namespace(kb: Khora) -> None:
    """A fresh namespace must return an empty result, not raise."""
    empty_ns = (await kb.create_namespace()).namespace_id
    result = await kb.recall("anything", namespace=empty_ns, limit=5)
    assert result.chunks == []


async def test_chronicle_concurrent_remember(kb: Khora, namespace_id: UUID) -> None:
    """5 concurrent remember() calls land 5 chunks (entity gate doesn't drop)."""
    contents = [f"Document {i}: Marie Curie won the Nobel Prize in Physics ({i})." for i in range(5)]
    await asyncio.gather(*[_remember(kb, namespace_id=namespace_id, content=c) for c in contents])

    result = await kb.recall("Marie Curie", namespace=namespace_id, limit=10)
    assert len(result.chunks) >= 5


async def test_chronicle_recall_surfaces_entities_and_abstention_flag(kb: Khora, namespace_id: UUID) -> None:
    """Regression for issue #808.

    `Chronicle.recall()` had a one-line shadow re-assignment that wiped
    `entity_hits` to `[]` immediately after `_collect_entities` populated
    it, so `RecallResult.entities` was always empty and the
    `entities_empty` abstention flag was permanently `True` even when
    entities were extracted at ingest and the query directly named one.

    Asserts: when a corpus has known extracted entities and the recall
    query references one of them, `RecallResult.entities` is non-empty
    AND `abstention_signals['entities_empty']` is `False`.
    """
    _plan_extraction(
        "Sarah Chen",
        entities=[
            ("Sarah Chen", "PERSON"),
            ("Globex Corp", "ORGANIZATION"),
            ("expense reimbursements", "CONCEPT"),
        ],
    )
    await _remember(
        kb,
        namespace_id=namespace_id,
        content="Sarah Chen is the CFO at Globex Corp. She approves all expense reimbursements over 2000 USD.",
    )

    result = await kb.recall("What does Sarah Chen approve?", namespace=namespace_id, limit=5)

    assert result.entities, (
        "recall returned zero entities despite a corpus with three extracted entities - "
        "the entity_hits shadow-reassignment from issue #808 has regressed"
    )
    # At least one of the three planted entities must surface; the exact
    # identity depends on stub embedding similarity and isn't part of the
    # contract under test (the contract is "entity_hits flows through").
    surfaced = {e.name for e in result.entities}
    expected = {"Sarah Chen", "Globex Corp", "expense reimbursements"}
    assert surfaced & expected, f"none of the planted entities surfaced; got {surfaced!r}"

    signals = result.engine_info.get("abstention_signals", {})
    assert signals.get("entities_empty") is False, f"entities_empty must reflect actual entity count, got {signals!r}"
