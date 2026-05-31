"""Final coverage push for ``khora.pipelines.flows.ingest``.

Targets the remaining low-coverage blocks after the #753 batch:

* ``_extract_cross_chunk_relationships`` — opt-in path, no-windows
  short-circuit, no-entities window skip, extractor exception swallow,
  triple dedup, results emission.
* ``_extract_source_timestamp`` — calendar vs default ordering,
  string/datetime/ISO/date-only parsing, malformed fallthrough.
* ``_coerce_session_id`` — UUID passthrough, string parse, empty,
  malformed.
* ``stream_extract_and_embed_entities`` — empty chunks, skip-types
  fast-path on entity that bypasses embedding, batch-embed failure
  swallowed.
* ``_create_session_episodes`` — session with no timestamps skipped,
  multi-doc dedupe of entity IDs.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from khora.core.models import Chunk, Document
from khora.pipelines.flows.ingest import (
    _coerce_session_id,
    _create_session_episodes,
    _extract_cross_chunk_relationships,
    _extract_source_timestamp,
    stream_extract_and_embed_entities,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mk_chunk(ns: UUID, doc_id: UUID, *, content: str = "x", idx: int = 0) -> Chunk:
    return Chunk(
        namespace_id=ns,
        document_id=doc_id,
        content=content,
        chunk_index=idx,
        created_at=datetime.now(UTC),
    )


# ---------------------------------------------------------------------------
# _extract_cross_chunk_relationships
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestExtractCrossChunkRelationships:
    async def test_returns_empty_when_flag_disabled(self) -> None:
        chunks = [_mk_chunk(uuid4(), uuid4(), content="a", idx=0)]
        out = await _extract_cross_chunk_relationships(
            chunks,
            {},
            extractor=MagicMock(),
            extraction_context={"cross_chunk_extraction": False},
            entity_types=[],
            relationship_types=[],
        )
        assert out == []

    async def test_returns_empty_with_only_one_chunk(self) -> None:
        chunks = [_mk_chunk(uuid4(), uuid4(), content="a", idx=0)]
        out = await _extract_cross_chunk_relationships(
            chunks,
            {},
            extractor=MagicMock(),
            extraction_context={"cross_chunk_extraction": True},
            entity_types=[],
            relationship_types=[],
        )
        assert out == []

    async def test_skips_windows_with_no_entities(self) -> None:
        ns, doc_id = uuid4(), uuid4()
        chunks = [
            _mk_chunk(ns, doc_id, content="a", idx=0),
            _mk_chunk(ns, doc_id, content="b", idx=1),
        ]
        extractor = MagicMock()
        extractor.extract_multi = AsyncMock()
        out = await _extract_cross_chunk_relationships(
            chunks,
            {},  # no entities for either chunk
            extractor=extractor,
            extraction_context={"cross_chunk_extraction": True},
            entity_types=[],
            relationship_types=[],
        )
        assert out == []
        # No LLM call when there are no entities in either chunk.
        extractor.extract_multi.assert_not_awaited()

    async def test_extractor_exception_skips_window(self) -> None:
        ns, doc_id = uuid4(), uuid4()
        c0 = _mk_chunk(ns, doc_id, content="a", idx=0)
        c1 = _mk_chunk(ns, doc_id, content="b", idx=1)

        extractor = MagicMock()
        extractor.extract_multi = AsyncMock(side_effect=RuntimeError("LLM down"))
        out = await _extract_cross_chunk_relationships(
            [c0, c1],
            {c0.id: ["alice"], c1.id: ["bob"]},
            extractor=extractor,
            extraction_context={"cross_chunk_extraction": True},
            entity_types=[],
            relationship_types=[],
        )
        # Window failure swallowed — empty result, no crash.
        assert out == []
        extractor.extract_multi.assert_awaited_once()

    async def test_emits_new_relationships_and_dedupes_triples(self) -> None:
        ns, doc_id = uuid4(), uuid4()
        c0 = _mk_chunk(ns, doc_id, content="alice met bob", idx=0)
        c1 = _mk_chunk(ns, doc_id, content="bob works at acme", idx=1)
        c2 = _mk_chunk(ns, doc_id, content="alice talks to bob", idx=2)

        # Build two windows. Same triple appears in both — must dedup.
        rel = type(
            "ER",
            (),
            {
                "__init__": lambda self, **kw: self.__dict__.update(kw),
            },
        )

        def _mk_rel(src: str, tgt: str, rtype: str) -> Any:
            return rel(
                source_entity=src,
                target_entity=tgt,
                relationship_type=rtype,
                description="x",
                properties={},
                confidence=0.9,
            )

        # Window 0 (c0, c1): produces (alice, KNOWS, bob).
        # Window 1 (c1, c2): also produces (alice, KNOWS, bob) — should be deduped.
        result_0 = MagicMock(relationships=[_mk_rel("alice", "bob", "KNOWS")])
        result_1 = MagicMock(
            relationships=[
                _mk_rel("alice", "bob", "KNOWS"),  # duplicate
                _mk_rel("bob", "acme", "WORKS_FOR"),  # new
            ]
        )

        extractor = MagicMock()
        extractor.extract_multi = AsyncMock(side_effect=[[result_0], [result_1]])

        out = await _extract_cross_chunk_relationships(
            [c0, c1, c2],
            {c0.id: ["alice"], c1.id: ["bob", "acme"], c2.id: ["alice", "bob"]},
            extractor=extractor,
            extraction_context={"cross_chunk_extraction": True},
            entity_types=[],
            relationship_types=[],
        )
        # Two unique relationships from two windows.
        assert len(out) == 2
        triples = {(r.source_entity, r.target_entity, r.relationship_type) for r in out}
        assert ("alice", "bob", "KNOWS") in triples
        assert ("bob", "acme", "WORKS_FOR") in triples

    async def test_max_windows_caps_extractor_calls(self) -> None:
        ns, doc_id = uuid4(), uuid4()
        chunks = [_mk_chunk(ns, doc_id, content=f"c{i}", idx=i) for i in range(5)]
        extractor = MagicMock()
        extractor.extract_multi = AsyncMock(return_value=[MagicMock(relationships=[])])

        await _extract_cross_chunk_relationships(
            chunks,
            {c.id: ["alice"] for c in chunks},
            extractor=extractor,
            extraction_context={"cross_chunk_extraction": True},
            entity_types=[],
            relationship_types=[],
            max_windows=2,
        )
        # 2 windows max → 2 extract calls.
        assert extractor.extract_multi.await_count == 2


# ---------------------------------------------------------------------------
# _extract_source_timestamp
# ---------------------------------------------------------------------------


class TestExtractSourceTimestamp:
    def test_event_source_prefers_occurred_at(self) -> None:
        out = _extract_source_timestamp(
            {
                "source_type": "calendar",
                "sent_at": "2026-01-01T00:00:00+00:00",
                "occurred_at": "2026-02-01T00:00:00+00:00",
            }
        )
        assert out == datetime(2026, 2, 1, tzinfo=UTC)

    def test_default_source_prefers_sent_at(self) -> None:
        out = _extract_source_timestamp(
            {
                "source_type": "email",
                "sent_at": "2026-01-01T00:00:00+00:00",
                "occurred_at": "2026-02-01T00:00:00+00:00",
            }
        )
        assert out == datetime(2026, 1, 1, tzinfo=UTC)

    def test_returns_datetime_passthrough(self) -> None:
        ts = datetime(2026, 5, 18, tzinfo=UTC)
        out = _extract_source_timestamp({"sent_at": ts})
        assert out == ts

    def test_handles_iso_with_z_suffix(self) -> None:
        out = _extract_source_timestamp({"sent_at": "2026-05-18T10:00:00Z"})
        assert out is not None
        assert out.tzinfo is not None

    def test_handles_date_only_format(self) -> None:
        out = _extract_source_timestamp({"sent_at": "2026-05-18"})
        assert out == datetime(2026, 5, 18, tzinfo=UTC)

    def test_handles_iso_without_z_suffix(self) -> None:
        out = _extract_source_timestamp({"sent_at": "2026-05-18T10:00:00+00:00"})
        assert out == datetime(2026, 5, 18, 10, tzinfo=UTC)

    def test_returns_none_when_no_match(self) -> None:
        assert _extract_source_timestamp({"unrelated": "value"}) is None

    def test_skips_malformed_falls_through_to_next_field(self) -> None:
        out = _extract_source_timestamp({"sent_at": "not a date", "created_at": "2026-05-18T00:00:00+00:00"})
        assert out == datetime(2026, 5, 18, tzinfo=UTC)

    def test_skips_empty_field_value(self) -> None:
        """An empty string value falls through to the next candidate."""
        out = _extract_source_timestamp({"sent_at": "", "created_at": "2026-05-18T00:00:00+00:00"})
        assert out == datetime(2026, 5, 18, tzinfo=UTC)


# ---------------------------------------------------------------------------
# _coerce_session_id
# ---------------------------------------------------------------------------


class TestCoerceSessionId:
    def test_passes_uuid_through(self) -> None:
        sid = uuid4()
        assert _coerce_session_id(sid) == sid

    def test_parses_uuid_string(self) -> None:
        sid = uuid4()
        assert _coerce_session_id(str(sid)) == sid

    def test_none_returns_none(self) -> None:
        assert _coerce_session_id(None) is None

    def test_empty_string_returns_none(self) -> None:
        assert _coerce_session_id("") is None

    def test_malformed_returns_none(self) -> None:
        assert _coerce_session_id("not-a-uuid") is None

    def test_non_string_non_uuid_returns_none(self) -> None:
        # ``UUID(str(42))`` raises ValueError — caught.
        assert _coerce_session_id(42) is None


# ---------------------------------------------------------------------------
# stream_extract_and_embed_entities — extra branches
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStreamExtractEdges:
    async def test_empty_chunks_returns_empty_lists(self) -> None:
        embedder = MagicMock(embed_batch=AsyncMock(return_value=[]))
        ents, rels = await stream_extract_and_embed_entities(
            [],
            embedder,
            entity_types=["PERSON"],
            relationship_types=[],
        )
        assert ents == []
        assert rels == []

    async def test_batch_embed_failure_still_returns_entities(self) -> None:
        """When ``embedder.embed_batch`` raises, the entity is still surfaced (no embedding)."""
        ns = uuid4()
        doc_id = uuid4()
        chunk = _mk_chunk(ns, doc_id, content="Alice met Bob.")

        # Extraction yields one entity.
        extracted_ent = MagicMock()
        extracted_ent.name = "alice"
        extracted_ent.entity_type = "PERSON"
        extracted_ent.confidence = 0.95
        extracted_ent.description = None
        extracted_ent.attributes = {}
        extracted_ent.temporal = None

        result = MagicMock(entities=[extracted_ent], relationships=[])
        extractor = MagicMock()
        extractor.extract_multi = AsyncMock(return_value=[result])

        # Embedder fails — entities should still flow through.
        embedder = MagicMock(embed_batch=AsyncMock(side_effect=RuntimeError("embed down")))

        with patch("khora.extraction.extractors.LLMEntityExtractor", return_value=extractor):
            ents, rels = await stream_extract_and_embed_entities(
                [chunk],
                embedder,
                entity_types=["PERSON"],
                relationship_types=[],
            )
        # Entity surfaced without an embedding.
        assert len(ents) == 1
        assert ents[0].name == "alice"
        assert ents[0].embedding is None

    async def test_skip_embedding_for_low_value_entity_type(self) -> None:
        """Entities whose type is in ``skip_embedding_entity_types`` bypass the embedder."""
        ns = uuid4()
        doc_id = uuid4()
        chunk = _mk_chunk(ns, doc_id, content="metadata: page 1")

        extracted = MagicMock()
        extracted.name = "page 1"
        extracted.entity_type = "PAGE"
        extracted.confidence = 0.9
        extracted.description = None
        extracted.attributes = {}
        extracted.temporal = None

        result = MagicMock(entities=[extracted], relationships=[])
        extractor = MagicMock()
        extractor.extract_multi = AsyncMock(return_value=[result])

        embedder = MagicMock(embed_batch=AsyncMock())  # should NOT be called for the PAGE entity.

        with patch("khora.extraction.extractors.LLMEntityExtractor", return_value=extractor):
            ents, _ = await stream_extract_and_embed_entities(
                [chunk],
                embedder,
                entity_types=["PAGE"],
                relationship_types=[],
                skip_embedding_entity_types=["PAGE"],
                skip_embedding_mention_threshold=99,
            )
        assert len(ents) == 1
        # Embedder NOT called for that entity.
        embedder.embed_batch.assert_not_awaited()


# ---------------------------------------------------------------------------
# stream_extract_and_embed_entities — cross-chunk path (408-456)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStreamExtractCrossChunkPath:
    async def test_cross_chunk_extraction_flag_invokes_helper(self) -> None:
        """``extraction_context['cross_chunk_extraction']=True`` exercises the cross-chunk path."""
        ns = uuid4()
        doc_id = uuid4()
        # Two chunks where the per-chunk co-occurrence won't fire (entities live in
        # different chunks, no shared entities within a single chunk).
        c0 = _mk_chunk(ns, doc_id, content="Alice met someone.", idx=0)
        c1 = _mk_chunk(ns, doc_id, content="Acme is a company.", idx=1)

        def _mk_extracted(name: str, etype: str) -> Any:
            e = MagicMock()
            e.name = name
            e.entity_type = etype
            e.confidence = 0.95
            e.description = None
            e.attributes = {}
            e.temporal = None
            return e

        # Each chunk has one distinct entity → no per-chunk co-occurrence pair.
        result_0 = MagicMock(entities=[_mk_extracted("Alice", "PERSON")], relationships=[])
        result_1 = MagicMock(entities=[_mk_extracted("Acme", "ORGANIZATION")], relationships=[])

        extractor = MagicMock()
        extractor.extract_multi = AsyncMock(return_value=[result_0, result_1])

        # Cross-chunk helper finds one relationship not yet in existing_pairs.
        cross_rel = MagicMock()
        cross_rel.source_entity = "Alice"
        cross_rel.target_entity = "Acme"
        cross_rel.relationship_type = "WORKS_FOR"
        cross_rel.description = "from cross-chunk"
        cross_rel.confidence = 0.8
        cross_rel.properties = {}

        called: dict[str, bool] = {}

        async def _fake_cross_chunk(chunks, entities_by_chunk, extractor, ctx, **kw):
            called["yes"] = True
            return [cross_rel]

        embedder = MagicMock(embed_batch=AsyncMock(return_value=[[0.1, 0.2], [0.1, 0.2]]))

        with patch("khora.extraction.extractors.LLMEntityExtractor", return_value=extractor):
            with patch(
                "khora.pipelines.flows.ingest._extract_cross_chunk_relationships",
                new=_fake_cross_chunk,
            ):
                ents, rels = await stream_extract_and_embed_entities(
                    [c0, c1],
                    embedder,
                    entity_types=["PERSON", "ORGANIZATION"],
                    relationship_types=["WORKS_FOR"],
                    extraction_context={"cross_chunk_extraction": True},
                )

        # Helper was invoked — that's the code path under test.
        assert called.get("yes") is True
        # And the relationship was emitted.
        assert any(r.relationship_type == "WORKS_FOR" for r in rels)

    async def test_cross_chunk_skips_when_helper_returns_nothing(self) -> None:
        ns = uuid4()
        doc_id = uuid4()
        c0 = _mk_chunk(ns, doc_id, content="text", idx=0)
        c1 = _mk_chunk(ns, doc_id, content="text2", idx=1)

        ent = MagicMock()
        ent.name = "x"
        ent.entity_type = "T"
        ent.confidence = 0.95
        ent.description = None
        ent.attributes = {}
        ent.temporal = None
        result = MagicMock(entities=[ent], relationships=[])
        extractor = MagicMock()
        extractor.extract_multi = AsyncMock(return_value=[result, result])

        embedder = MagicMock(embed_batch=AsyncMock(return_value=[[0.1, 0.2]]))

        async def _empty(*a: Any, **k: Any) -> list:
            return []

        with patch("khora.extraction.extractors.LLMEntityExtractor", return_value=extractor):
            with patch("khora.pipelines.flows.ingest._extract_cross_chunk_relationships", new=_empty):
                ents, rels = await stream_extract_and_embed_entities(
                    [c0, c1],
                    embedder,
                    entity_types=["T"],
                    relationship_types=[],
                    extraction_context={"cross_chunk_extraction": True},
                )
        # No cross-chunk rels added.
        assert all(r.relationship_type != "RELATES_TO" for r in rels) or rels == []


# ---------------------------------------------------------------------------
# _create_session_episodes — additional branches
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCreateSessionEpisodesExtraBranches:
    async def test_session_with_no_timestamps_is_skipped(self) -> None:
        """If neither doc has ``source_timestamp`` nor ``created_at``, episode skipped."""
        ns = uuid4()
        storage = MagicMock()
        storage.create_episode = AsyncMock()

        doc = Document(namespace_id=ns, content="x")
        doc.metadata = {"thread_id": "thr-x"}
        doc.source_timestamp = None
        doc.created_at = None

        results = [{"document_id": str(doc.id), "entity_ids": [uuid4()], "chunk_ids": [uuid4()]}]
        created = await _create_session_episodes(
            namespace_id=ns,
            documents=[{}],
            staged_docs=[doc],
            successful_results=results,
            storage=storage,
        )
        assert created == 0
        storage.create_episode.assert_not_called()

    async def test_skips_documents_without_thread_id(self) -> None:
        ns = uuid4()
        storage = MagicMock()
        storage.create_episode = AsyncMock()

        # Doc has no thread_id → not grouped into any session.
        doc = Document(namespace_id=ns, content="x")
        doc.metadata = {}
        doc.source_timestamp = datetime(2026, 5, 13, tzinfo=UTC)

        results = [{"document_id": str(doc.id), "entity_ids": [uuid4()], "chunk_ids": [uuid4()]}]
        created = await _create_session_episodes(
            namespace_id=ns,
            documents=[{}],
            staged_docs=[doc],
            successful_results=results,
            storage=storage,
        )
        assert created == 0

    async def test_dedupes_entity_and_chunk_ids_across_session_docs(self) -> None:
        """A single entity referenced from multiple docs in the same session appears once."""
        ns = uuid4()
        storage = MagicMock()
        captured_episode: dict[str, Any] = {}

        async def _create_episode(ep: Any) -> Any:
            captured_episode["ep"] = ep
            return ep

        storage.create_episode = AsyncMock(side_effect=_create_episode)

        shared_eid = uuid4()
        shared_cid = uuid4()
        doc1 = Document(namespace_id=ns, content="a")
        doc1.metadata = {"thread_id": "thr-1"}
        doc1.source_timestamp = datetime(2026, 5, 13, 10, tzinfo=UTC)
        doc2 = Document(namespace_id=ns, content="b")
        doc2.metadata = {"thread_id": "thr-1"}
        doc2.source_timestamp = datetime(2026, 5, 13, 11, tzinfo=UTC)

        results = [
            {"document_id": str(doc1.id), "entity_ids": [shared_eid, uuid4()], "chunk_ids": [shared_cid]},
            {"document_id": str(doc2.id), "entity_ids": [shared_eid], "chunk_ids": [shared_cid, uuid4()]},
        ]
        created = await _create_session_episodes(
            namespace_id=ns,
            documents=[{}, {}],
            staged_docs=[doc1, doc2],
            successful_results=results,
            storage=storage,
        )
        assert created == 1
        ep = captured_episode["ep"]
        # 2 distinct entity ids (shared_eid + the unique one) — no duplicate.
        assert ep.entity_ids.count(shared_eid) == 1
        assert ep.metadata["message_count"] == 2
