"""Coverage push for ``khora.pipelines.flows.ingest``.

These tests mock at the storage / extractor / embedder boundary to exercise
the staging, batch-staging, session-episode, smart-resolution, batch-inference,
and embedding-backfill code paths without spinning up a real database or LLM.

Target blocks (per #695 step 2):

* Lines 52-61 (``_should_skip_entity_embedding``)
* Lines 66-81 (``_find_entity_key``)
* Lines 518 + 597-608 (``compute_checksum`` + ``_parse_temporal_date``)
* Lines 582-589 (``_coerce_session_id``)
* Lines 624-668 (``stage_document``)
* Lines 684-767 (``stage_documents_batch``)
* Lines 780-828 (``_stage_all_documents``)
* Lines 213-513 (``stream_extract_and_embed_entities``)
* Lines 1880-1951 (``_create_session_episodes``)
* Lines 2159-2207 (``run_batch_inference``)
* Lines 2234-2287 (``backfill_entity_embeddings``)

All tests run under ``pytest.mark.unit`` — no Docker, no network.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from khora.core.models import Chunk, Document, Entity, Relationship
from khora.extraction.extractors.base import (
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
    TemporalInfo,
)
from khora.pipelines.flows.ingest import (
    _coerce_session_id,
    _create_session_episodes,
    _extract_cross_chunk_relationships,
    _extract_source_timestamp,
    _find_entity_key,
    _parse_temporal_date,
    _should_skip_entity_embedding,
    _stage_all_documents,
    backfill_entity_embeddings,
    compute_checksum,
    ingest_documents,
    run_batch_inference,
    stage_document,
    stage_documents_batch,
    stream_extract_and_embed_entities,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(ns_id: UUID, doc_id: UUID, content: str = "hello world", idx: int = 0) -> Chunk:
    return Chunk(
        namespace_id=ns_id,
        document_id=doc_id,
        content=content,
        chunk_index=idx,
        created_at=datetime.now(UTC),
    )


def _make_storage(**overrides) -> MagicMock:
    """Build a MagicMock StorageCoordinator with sensible async defaults."""
    storage = MagicMock()
    storage.get_document_by_checksum = AsyncMock(return_value=None)
    storage.get_documents_by_checksums = AsyncMock(return_value={})
    storage.create_document = AsyncMock(side_effect=lambda doc: doc)
    storage.update_document = AsyncMock()
    storage.create_chunks_batch = AsyncMock()
    storage.upsert_entities_batch = AsyncMock(return_value=[])
    storage.update_entity_embeddings_batch = AsyncMock(return_value=0)
    storage.create_relationships_batch = AsyncMock(return_value=[])
    storage.list_entities = AsyncMock(return_value=[])
    storage.list_relationships = AsyncMock(return_value=[])
    storage.get_entity_by_name = AsyncMock(return_value=None)
    storage.dispatch_hook = AsyncMock()
    storage.create_episode = AsyncMock()
    # vector backend (used by backfill_entity_embeddings)
    storage.vector = MagicMock()
    storage.vector.entity_exists = AsyncMock(return_value=True)
    storage.vector.create_entity = AsyncMock()
    for key, value in overrides.items():
        setattr(storage, key, value)
    return storage


# ---------------------------------------------------------------------------
# Pure helpers (cheap to cover)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestShouldSkipEntityEmbedding:
    def _make_entity(self, entity_type: str, mention_count: int = 1) -> Entity:
        return Entity(name="X", entity_type=entity_type, mention_count=mention_count)

    def test_empty_skip_types_never_skips(self):
        e = self._make_entity("DATE")
        assert _should_skip_entity_embedding(e, [], 1) is False

    def test_unlisted_entity_type_is_not_skipped(self):
        e = self._make_entity("PERSON", mention_count=1)
        assert _should_skip_entity_embedding(e, ["DATE"], 1) is False

    def test_listed_type_with_threshold_zero_always_skips(self):
        e = self._make_entity("date", mention_count=99)  # case-insensitive
        assert _should_skip_entity_embedding(e, ["DATE"], 0) is True

    def test_threshold_one_skips_low_mention(self):
        e = self._make_entity("DATE", mention_count=1)
        assert _should_skip_entity_embedding(e, ["DATE"], 1) is True

    def test_threshold_one_keeps_repeated_mentions(self):
        e = self._make_entity("DATE", mention_count=5)
        assert _should_skip_entity_embedding(e, ["DATE"], 1) is False


@pytest.mark.unit
class TestFindEntityKey:
    def test_returns_none_when_index_empty(self):
        assert _find_entity_key("alice", {}) is None

    def test_exact_prefix_match_takes_precedence(self):
        keys = {"alice:PERSON": object(), "alicia:PERSON": object()}
        assert _find_entity_key("alice", keys) == "alice:PERSON"

    def test_fuzzy_levenshtein_match_when_no_exact(self):
        keys = {"alicia:PERSON": object()}
        match = _find_entity_key("alicea", keys)
        # "alicea" vs "alicia" is similar enough to clear the 0.7 threshold.
        assert match == "alicia:PERSON"

    def test_dissimilar_names_return_none(self):
        keys = {"zzzzzz:PERSON": object()}
        assert _find_entity_key("alice", keys) is None


@pytest.mark.unit
class TestComputeChecksum:
    def test_sha256_hex_digest_shape(self):
        digest = compute_checksum("hello")
        assert isinstance(digest, str)
        assert len(digest) == 64
        # SHA-256 is deterministic
        assert digest == compute_checksum("hello")

    def test_different_content_different_digest(self):
        assert compute_checksum("a") != compute_checksum("b")


@pytest.mark.unit
class TestParseTemporalDate:
    def test_none_returns_none(self):
        assert _parse_temporal_date(None) is None

    def test_empty_string_returns_none(self):
        assert _parse_temporal_date("") is None

    def test_iso_z_suffix_parses(self):
        dt = _parse_temporal_date("2026-05-13T14:00:00Z")
        assert dt is not None
        assert dt.isoformat() == "2026-05-13T14:00:00+00:00"

    def test_iso_with_offset(self):
        dt = _parse_temporal_date("2026-05-13T14:00:00+02:00")
        assert dt is not None
        assert dt.utcoffset() is not None

    def test_date_only(self):
        dt = _parse_temporal_date("2026-05-13")
        assert dt is not None
        assert dt.year == 2026 and dt.month == 5 and dt.day == 13

    def test_invalid_returns_none(self):
        assert _parse_temporal_date("yesterday") is None


@pytest.mark.unit
class TestCoerceSessionId:
    def test_none_returns_none(self):
        assert _coerce_session_id(None) is None

    def test_empty_string_returns_none(self):
        assert _coerce_session_id("") is None

    def test_uuid_instance_passes_through(self):
        u = uuid4()
        assert _coerce_session_id(u) == u

    def test_uuid_string_parses(self):
        u = uuid4()
        assert _coerce_session_id(str(u)) == u

    def test_invalid_string_returns_none(self):
        assert _coerce_session_id("not-a-uuid") is None


@pytest.mark.unit
class TestExtractSourceTimestamp:
    def test_event_source_prefers_occurred_at(self):
        md = {
            "source_type": "calendar",
            "sent_at": "2026-01-01T10:00:00Z",
            "occurred_at": "2026-02-01T10:00:00Z",
        }
        ts = _extract_source_timestamp(md)
        assert ts is not None
        assert ts.month == 2

    def test_default_source_falls_through_to_date_field(self):
        md = {"date": "2026-03-15"}
        ts = _extract_source_timestamp(md)
        assert ts is not None
        assert ts.day == 15

    def test_datetime_instance_returned_as_is(self):
        ref = datetime(2026, 5, 1, tzinfo=UTC)
        md = {"sent_at": ref}
        assert _extract_source_timestamp(md) == ref

    def test_malformed_value_skipped(self):
        md = {"sent_at": "not-iso", "created_at": "2026-04-01T00:00:00Z"}
        ts = _extract_source_timestamp(md)
        assert ts is not None
        assert ts.month == 4

    def test_no_matching_field_returns_none(self):
        assert _extract_source_timestamp({"unrelated": "x"}) is None


# ---------------------------------------------------------------------------
# stage_document / stage_documents_batch / _stage_all_documents
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
class TestStageDocument:
    async def test_returns_none_for_existing_checksum(self):
        ns = uuid4()
        existing = MagicMock(status="completed")
        storage = _make_storage(get_document_by_checksum=AsyncMock(return_value=existing))
        result = await stage_document({"content": "hello"}, ns, storage)
        assert result is None
        storage.create_document.assert_not_called()

    async def test_creates_new_document_with_source_timestamp(self):
        ns = uuid4()
        storage = _make_storage()
        doc = await stage_document(
            {
                "content": "alpha",
                "source": "test://x",
                "source_type": "manual",
                "title": "Alpha",
                "metadata": {"sent_at": "2026-05-13T14:00:00Z"},
            },
            ns,
            storage,
        )
        assert doc is not None
        assert doc.title == "Alpha"
        assert doc.source_timestamp is not None
        assert doc.source_timestamp.isoformat() == "2026-05-13T14:00:00+00:00"
        # created_at/updated_at stay at ingest time (the khora-ops axis) and must
        # NOT be conflated with the source event time, which lives on
        # source_timestamp (#993 bi-temporal contract).
        assert doc.created_at != doc.source_timestamp
        assert doc.updated_at != doc.source_timestamp
        storage.create_document.assert_awaited_once()

    async def test_uses_now_when_no_source_timestamp(self):
        ns = uuid4()
        storage = _make_storage()
        doc = await stage_document({"content": "beta"}, ns, storage)
        assert doc is not None
        assert doc.source_timestamp is None
        assert doc.created_at is not None

    async def test_session_id_from_custom_metadata(self):
        ns = uuid4()
        sid = uuid4()
        storage = _make_storage()
        doc = await stage_document({"content": "g", "metadata": {"session_id": str(sid)}}, ns, storage)
        assert doc is not None
        assert doc.session_id == sid

    async def test_omitted_content_type_stays_none(self):
        ns = uuid4()
        storage = _make_storage()
        doc = await stage_document({"content": "no-ct"}, ns, storage)
        assert doc is not None
        assert doc.content_type is None

    async def test_empty_content_type_normalizes_to_none(self):
        ns = uuid4()
        storage = _make_storage()
        doc = await stage_document({"content": "blank-ct", "content_type": ""}, ns, storage)
        assert doc is not None
        assert doc.content_type is None

    async def test_new_session_id_creates_document_despite_checksum_hit(self):
        """#1171: same content + new session_id must not be dropped by stage_document."""
        ns = uuid4()
        session_b = uuid4()
        existing = MagicMock(status="completed", external_id=None, session_id=uuid4())
        storage = _make_storage(get_document_by_checksum=AsyncMock(return_value=existing))
        doc = await stage_document(
            {"content": "same", "metadata": {"session_id": str(session_b)}},
            ns,
            storage,
        )
        assert doc is not None, "New session_id must create a new document, not be dropped"
        assert doc.session_id == session_b
        storage.create_document.assert_awaited_once()

    async def test_new_external_id_creates_document_despite_checksum_hit(self):
        """#1171: same content + new external_id must not be dropped by stage_document."""
        ns = uuid4()
        existing = MagicMock(status="completed", external_id="ext-a", session_id=None)
        storage = _make_storage(get_document_by_checksum=AsyncMock(return_value=existing))
        doc = await stage_document(
            {"content": "same", "external_id": "ext-b"},
            ns,
            storage,
        )
        assert doc is not None, "New external_id must create a new document, not be dropped"
        storage.create_document.assert_awaited_once()

    async def test_same_session_id_still_dedups_in_stage_document(self):
        """#1171: same content + same session_id is still a duplicate in stage_document."""
        ns = uuid4()
        session = uuid4()
        existing = MagicMock(status="completed", external_id=None, session_id=session)
        storage = _make_storage(get_document_by_checksum=AsyncMock(return_value=existing))
        result = await stage_document(
            {"content": "same", "metadata": {"session_id": str(session)}},
            ns,
            storage,
        )
        assert result is None, "Same session_id + same checksum must still be a duplicate"
        storage.create_document.assert_not_called()

    async def test_same_external_id_still_dedups_in_stage_document(self):
        """#1171: same content + same external_id is still a duplicate in stage_document."""
        ns = uuid4()
        existing = MagicMock(status="completed", external_id="ext-1", session_id=None)
        storage = _make_storage(get_document_by_checksum=AsyncMock(return_value=existing))
        result = await stage_document(
            {"content": "same", "external_id": "ext-1"},
            ns,
            storage,
        )
        assert result is None, "Same external_id + same checksum must still be a duplicate"
        storage.create_document.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
class TestStageDocumentsBatch:
    async def test_empty_input_returns_empty(self):
        ns = uuid4()
        storage = _make_storage()
        assert await stage_documents_batch([], ns, storage) == []

    async def test_creates_new_docs_for_distinct_content(self):
        ns = uuid4()
        storage = _make_storage()
        inputs = [{"content": "a"}, {"content": "b"}, {"content": "c"}]
        results = await stage_documents_batch(inputs, ns, storage)
        assert len(results) == 3
        assert all(r is not None for r in results)
        assert storage.create_document.await_count == 3

    async def test_intra_batch_duplicate_creates_once_and_copies(self):
        ns = uuid4()
        storage = _make_storage()
        inputs = [{"content": "dup"}, {"content": "dup"}, {"content": "other"}]
        results = await stage_documents_batch(inputs, ns, storage)
        assert len(results) == 3
        # Both dupes share the same Document object
        assert results[0] is results[1]
        assert results[2] is not results[0]
        # Only 2 calls (one per unique checksum)
        assert storage.create_document.await_count == 2

    async def test_existing_checksum_skipped(self):
        ns = uuid4()
        existing_doc = MagicMock(status="completed")
        # Key off the actual "skip-me" checksum, not a positional index:
        # stage_documents_batch queries `list({...})` (set-derived, order
        # unspecified), so `checksums[0]` is not deterministically "skip-me".
        skip_checksum = compute_checksum("skip-me")

        async def fake_get(ns_, checksums):
            return {skip_checksum: existing_doc}

        storage = _make_storage(get_documents_by_checksums=AsyncMock(side_effect=fake_get))
        inputs = [{"content": "skip-me"}, {"content": "new-one"}]
        results = await stage_documents_batch(inputs, ns, storage)
        assert results[0] is None
        assert results[1] is not None
        assert storage.create_document.await_count == 1


@pytest.mark.unit
@pytest.mark.asyncio
class TestStageAllDocuments:
    async def test_empty_input(self):
        storage = _make_storage()
        assert await _stage_all_documents([], uuid4(), storage) == []

    async def test_creates_every_document_regardless_of_checksum(self):
        ns = uuid4()
        storage = _make_storage()
        inputs = [
            {"content": "rewrite-me", "metadata": {"sent_at": "2026-05-13T14:00:00Z"}},
            {"content": "rewrite-me"},  # same content — would dedup in batch mode
        ]
        results = await _stage_all_documents(inputs, ns, storage)
        assert len(results) == 2
        assert all(r is not None for r in results)
        assert storage.create_document.await_count == 2

    async def test_propagates_external_id_and_extraction_hash(self):
        ns = uuid4()
        storage = _make_storage()
        results = await _stage_all_documents(
            [
                {
                    "content": "x",
                    "external_id": "ext-42",
                    "extraction_config_hash": "h" * 16,
                }
            ],
            ns,
            storage,
        )
        assert results[0].external_id == "ext-42"
        assert results[0].extraction_config_hash == "h" * 16


# ---------------------------------------------------------------------------
# stream_extract_and_embed_entities
# ---------------------------------------------------------------------------


def _stub_extraction_result(
    name: str,
    *,
    confidence: float = 0.9,
    rel_target: str | None = None,
    rel_type: str = "WORKS_FOR",
    temporal: TemporalInfo | None = None,
) -> ExtractionResult:
    e1 = ExtractedEntity(name=name, entity_type="PERSON", confidence=confidence, temporal=temporal)
    rels: list[ExtractedRelationship] = []
    if rel_target is not None:
        rels.append(
            ExtractedRelationship(
                source_entity=name,
                target_entity=rel_target,
                relationship_type=rel_type,
                confidence=confidence,
            )
        )
        # Add the target so it gets indexed
        e2 = ExtractedEntity(name=rel_target, entity_type="ORG", confidence=confidence)
        return ExtractionResult(entities=[e1, e2], relationships=rels)
    return ExtractionResult(entities=[e1], relationships=rels)


@pytest.mark.unit
@pytest.mark.asyncio
class TestStreamExtractAndEmbedEntities:
    async def test_empty_chunks_returns_empty(self):
        embedder = MagicMock(embed_batch=AsyncMock(return_value=[]))
        ents, rels = await stream_extract_and_embed_entities(
            [],
            embedder,
            entity_types=["PERSON"],
            relationship_types=["WORKS_FOR"],
        )
        assert ents == [] and rels == []

    async def test_dedupes_within_run_and_embeds_entities(self):
        ns = uuid4()
        doc_id = uuid4()
        chunks = [
            _make_chunk(ns, doc_id, "Alice works at Acme.", idx=0),
            _make_chunk(ns, doc_id, "Alice met Bob.", idx=1),
        ]
        # Same Alice across two chunks → merged into one entity.
        results = [
            _stub_extraction_result("Alice", rel_target="Acme", rel_type="WORKS_FOR"),
            _stub_extraction_result("Alice", rel_target="Bob", rel_type="KNOWS"),
        ]
        extractor = MagicMock()
        extractor.extract_multi = AsyncMock(return_value=results)
        embedder = MagicMock(embed_batch=AsyncMock(side_effect=lambda texts: [[0.1] * 4 for _ in texts]))

        with patch("khora.extraction.extractors.LLMEntityExtractor", return_value=extractor):
            ents, rels = await stream_extract_and_embed_entities(
                chunks,
                embedder,
                entity_types=["PERSON", "ORG"],
                relationship_types=["WORKS_FOR", "KNOWS"],
            )

        # Alice + Acme + Bob = 3 distinct entities, all embedded.
        names = sorted(e.name for e in ents)
        assert "alice" in names
        assert "acme" in names
        assert "bob" in names
        assert all(e.embedding is not None for e in ents)
        # mention_count for Alice should reflect the second-chunk merge
        alice = next(e for e in ents if e.name == "alice")
        assert alice.mention_count >= 2
        # At least the two extracted relationships survived. (co-occurrence
        # adds an ASSOCIATED_WITH for Alice+Acme in chunk 0 and Alice+Bob in
        # chunk 1, but the dedup against existing_pairs prevents doubling
        # the WORKS_FOR pair.)
        rel_types = {r.relationship_type for r in rels}
        assert "WORKS_FOR" in rel_types
        assert "KNOWS" in rel_types

    async def test_skips_low_confidence_entities(self):
        ns = uuid4()
        doc_id = uuid4()
        chunks = [_make_chunk(ns, doc_id, "weak signal", idx=0)]
        results = [_stub_extraction_result("ghost", confidence=0.1)]
        extractor = MagicMock(extract_multi=AsyncMock(return_value=results))
        embedder = MagicMock(embed_batch=AsyncMock(return_value=[]))

        with patch("khora.extraction.extractors.LLMEntityExtractor", return_value=extractor):
            ents, rels = await stream_extract_and_embed_entities(
                chunks,
                embedder,
                entity_types=["PERSON"],
                relationship_types=[],
            )

        # Default skill min_entity_confidence is 0.5 — entity filtered out.
        assert ents == []
        assert rels == []

    async def test_embedding_failure_keeps_entities_without_embeddings(self):
        ns = uuid4()
        doc_id = uuid4()
        chunks = [_make_chunk(ns, doc_id, "Alice met Bob.", idx=0)]
        results = [_stub_extraction_result("Alice")]
        extractor = MagicMock(extract_multi=AsyncMock(return_value=results))
        embedder = MagicMock(embed_batch=AsyncMock(side_effect=RuntimeError("LLM down")))

        with patch("khora.extraction.extractors.LLMEntityExtractor", return_value=extractor):
            ents, _ = await stream_extract_and_embed_entities(
                chunks,
                embedder,
                entity_types=["PERSON"],
                relationship_types=[],
            )

        # Entities are still returned, just without embedding vectors.
        assert any(e.embedding is None for e in ents)

    async def test_skip_embedding_entity_types(self):
        ns = uuid4()
        doc_id = uuid4()
        chunks = [_make_chunk(ns, doc_id, "On 2026-05-13 Alice spoke.", idx=0)]
        # DATE is in our skip list — should not be sent to the embedder.
        results = [
            ExtractionResult(
                entities=[
                    ExtractedEntity(name="2026-05-13", entity_type="DATE", confidence=0.9),
                    ExtractedEntity(name="Alice", entity_type="PERSON", confidence=0.9),
                ]
            )
        ]
        extractor = MagicMock(extract_multi=AsyncMock(return_value=results))

        called_texts: list[list[str]] = []

        async def fake_embed(texts):
            called_texts.append(texts)
            return [[0.1, 0.2] for _ in texts]

        embedder = MagicMock(embed_batch=AsyncMock(side_effect=fake_embed))

        with patch("khora.extraction.extractors.LLMEntityExtractor", return_value=extractor):
            ents, _ = await stream_extract_and_embed_entities(
                chunks,
                embedder,
                entity_types=["PERSON", "DATE"],
                relationship_types=[],
                skip_embedding_entity_types=["DATE"],
                skip_embedding_mention_threshold=0,
            )

        # The DATE entity is present but never embedded.
        date_entity = next((e for e in ents if e.entity_type == "DATE"), None)
        assert date_entity is not None
        assert date_entity.embedding is None
        # Only the PERSON entity made it to the embedder.
        flat = [t for batch in called_texts for t in batch]
        assert any("alice" in t.lower() for t in flat)
        assert not any("2026-05-13" in t for t in flat)


# ---------------------------------------------------------------------------
# _create_session_episodes
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
class TestCreateSessionEpisodes:
    async def test_no_thread_id_no_episodes(self):
        ns = uuid4()
        storage = _make_storage()
        doc = Document(namespace_id=ns, content="x")
        result = {"document_id": str(doc.id), "entity_ids": [uuid4()], "chunk_ids": [uuid4()]}
        created = await _create_session_episodes(
            namespace_id=ns,
            documents=[{"metadata": {}}],
            staged_docs=[doc],
            successful_results=[result],
            storage=storage,
        )
        assert created == 0
        storage.create_episode.assert_not_called()

    async def test_groups_by_thread_id(self):
        ns = uuid4()
        storage = _make_storage()
        thread = "thr-1"
        # Two docs share a thread_id; one third doc has no thread → ignored.
        doc_a = Document(namespace_id=ns, content="a")
        doc_a.metadata = {"thread_id": thread}
        doc_a.source_timestamp = datetime(2026, 5, 13, 10, 0, tzinfo=UTC)

        doc_b = Document(namespace_id=ns, content="b")
        doc_b.metadata = {"thread_id": thread}
        doc_b.source_timestamp = datetime(2026, 5, 13, 12, 0, tzinfo=UTC)

        doc_c = Document(namespace_id=ns, content="c")
        doc_c.metadata = {}

        eid1, eid2, cid = uuid4(), uuid4(), uuid4()
        results = [
            {"document_id": str(doc_a.id), "entity_ids": [eid1], "chunk_ids": [cid]},
            {"document_id": str(doc_b.id), "entity_ids": [eid1, eid2], "chunk_ids": [cid]},
            {"document_id": str(doc_c.id), "entity_ids": [], "chunk_ids": []},
        ]
        created = await _create_session_episodes(
            namespace_id=ns,
            documents=[{}] * 3,
            staged_docs=[doc_a, doc_b, doc_c],
            successful_results=results,
            storage=storage,
        )
        assert created == 1
        storage.create_episode.assert_awaited_once()
        episode = storage.create_episode.await_args.args[0]
        # Both message timestamps drive occurred_at / duration_seconds.
        assert episode.occurred_at == datetime(2026, 5, 13, 10, 0, tzinfo=UTC)
        assert episode.duration_seconds == 2 * 3600
        # Entity IDs deduped across messages.
        assert set(episode.entity_ids) == {eid1, eid2}
        assert episode.metadata == {"thread_id": thread, "message_count": 2}

    async def test_create_episode_exception_is_swallowed(self):
        ns = uuid4()
        storage = _make_storage(create_episode=AsyncMock(side_effect=RuntimeError("boom")))
        doc = Document(namespace_id=ns, content="x")
        doc.metadata = {"thread_id": "tx"}
        doc.source_timestamp = datetime.now(UTC)
        result = {"document_id": str(doc.id), "entity_ids": [uuid4()], "chunk_ids": []}
        # Must not raise.
        created = await _create_session_episodes(
            namespace_id=ns,
            documents=[{}],
            staged_docs=[doc],
            successful_results=[result],
            storage=storage,
        )
        assert created == 0

    async def test_no_timestamps_skips_session(self):
        ns = uuid4()
        storage = _make_storage()
        doc = Document(namespace_id=ns, content="x")
        doc.metadata = {"thread_id": "tx"}
        # Force both timestamp fields to None
        doc.source_timestamp = None
        doc.created_at = None
        result = {"document_id": str(doc.id), "entity_ids": [uuid4()], "chunk_ids": []}
        created = await _create_session_episodes(
            namespace_id=ns,
            documents=[{}],
            staged_docs=[doc],
            successful_results=[result],
            storage=storage,
        )
        assert created == 0
        storage.create_episode.assert_not_called()

    async def test_failed_doc_does_not_misattribute_or_drop_tail(self):
        """Regression for #929.

        When an earlier document fails, ``successful_results`` is shorter
        than ``staged_docs``. Pairing by list position would (a) attach the
        wrong document's entities/chunks to each surviving episode and
        (b) drop the final surviving document entirely. Pairing by
        ``document_id`` must keep each episode attached to its own doc and
        retain the tail.
        """
        ns = uuid4()
        thread = "thr-1"

        # Three docs in one session; the FIRST one fails (no result).
        doc_a = Document(namespace_id=ns, content="a")
        doc_a.metadata = {"thread_id": thread}
        doc_a.source_timestamp = datetime(2026, 5, 13, 10, 0, tzinfo=UTC)

        doc_b = Document(namespace_id=ns, content="b")
        doc_b.metadata = {"thread_id": thread}
        doc_b.source_timestamp = datetime(2026, 5, 13, 11, 0, tzinfo=UTC)

        doc_c = Document(namespace_id=ns, content="c")
        doc_c.metadata = {"thread_id": thread}
        doc_c.source_timestamp = datetime(2026, 5, 13, 12, 0, tzinfo=UTC)

        eid_b, cid_b = uuid4(), uuid4()
        eid_c, cid_c = uuid4(), uuid4()

        # doc_a raised, so it is absent from successful_results. The surviving
        # results carry their own document_id (set in process_document).
        results = [
            {"document_id": str(doc_b.id), "entity_ids": [eid_b], "chunk_ids": [cid_b]},
            {"document_id": str(doc_c.id), "entity_ids": [eid_c], "chunk_ids": [cid_c]},
        ]

        captured = {}

        async def _capture(ep):
            captured["ep"] = ep
            return ep

        storage = _make_storage(create_episode=AsyncMock(side_effect=_capture))

        created = await _create_session_episodes(
            namespace_id=ns,
            documents=[{}] * 3,
            staged_docs=[doc_a, doc_b, doc_c],
            successful_results=results,
            storage=storage,
        )

        assert created == 1
        ep = captured["ep"]
        # Only the two surviving docs are in the episode (doc_a failed).
        assert set(ep.source_document_ids) == {doc_b.id, doc_c.id}
        # doc_c (the tail) is NOT dropped: its entity/chunk are present.
        assert eid_c in ep.entity_ids
        assert cid_c in ep.source_chunk_ids
        # doc_b's entity/chunk are present and not swapped with a neighbor's.
        assert eid_b in ep.entity_ids
        assert cid_b in ep.source_chunk_ids
        assert ep.metadata["message_count"] == 2


# ---------------------------------------------------------------------------
# run_batch_inference
# ---------------------------------------------------------------------------


def _make_expertise():
    from khora.extraction.skills import ExpertiseConfig
    from khora.extraction.skills.base import ConfidenceConfig, ExpansionConfig

    return ExpertiseConfig(
        name="test-expert",
        confidence=ConfidenceConfig(min_entity=0.5, min_relationship=0.5, min_inferred=0.3),
        expansion=ExpansionConfig(enabled=True, inference_mode="batch", depth=1, batch_storage_size=10),
    )


@pytest.mark.unit
@pytest.mark.asyncio
class TestRunBatchInference:
    async def test_no_entities_returns_zeros(self):
        ns = uuid4()
        storage = _make_storage()
        expertise = _make_expertise()
        result = await run_batch_inference(ns, storage, expertise)
        assert result == {"entities": 0, "relationships": 0, "inferred_relationships": 0}

    async def test_runs_expander_and_returns_inferred_count(self):
        ns = uuid4()
        ent_a = Entity(namespace_id=ns, name="alice", entity_type="PERSON")
        ent_b = Entity(namespace_id=ns, name="acme", entity_type="ORG")
        storage = _make_storage(
            list_entities=AsyncMock(return_value=[ent_a, ent_b]),
            list_relationships=AsyncMock(return_value=[]),
            # #1320: returns (relationship, is_new) per edge; the flow counts via len().
            create_relationships_batch=AsyncMock(side_effect=lambda rels, **kw: [(r, True) for r in rels]),
        )
        expertise = _make_expertise()

        # Fake expander.expand → ExpansionResult-shaped object
        inferred_rel = Relationship(
            namespace_id=ns,
            source_entity_id=ent_a.id,
            target_entity_id=ent_b.id,
            relationship_type="WORKS_AT",
        )
        expansion_result = MagicMock()
        expansion_result.entities = [ent_a, ent_b]
        expansion_result.relationships = []
        expansion_result.inferred_relationships = [inferred_rel, inferred_rel]
        expansion_result.inferred_relationship_count = 2

        fake_expander = MagicMock(expand=AsyncMock(return_value=expansion_result))
        with patch("khora.extraction.expansion.SemanticExpander", return_value=fake_expander):
            result = await run_batch_inference(ns, storage, expertise)
        assert result["entities"] == 2
        assert result["inferred_relationships"] == 2
        storage.create_relationships_batch.assert_awaited_once()

    async def test_batch_storage_failure_logged_but_returns_zero(self):
        ns = uuid4()
        ent_a = Entity(namespace_id=ns, name="alice", entity_type="PERSON")
        storage = _make_storage(
            list_entities=AsyncMock(return_value=[ent_a]),
            create_relationships_batch=AsyncMock(side_effect=RuntimeError("db down")),
        )
        expertise = _make_expertise()

        rel = Relationship(namespace_id=ns, source_entity_id=ent_a.id, target_entity_id=ent_a.id)
        expansion_result = MagicMock()
        expansion_result.entities = [ent_a]
        expansion_result.relationships = []
        expansion_result.inferred_relationships = [rel]
        expansion_result.inferred_relationship_count = 1

        fake_expander = MagicMock(expand=AsyncMock(return_value=expansion_result))
        with patch("khora.extraction.expansion.SemanticExpander", return_value=fake_expander):
            result = await run_batch_inference(ns, storage, expertise)
        # On failure the function logs and returns zero inferred (does not raise).
        assert result["inferred_relationships"] == 0


# ---------------------------------------------------------------------------
# backfill_entity_embeddings
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
class TestBackfillEntityEmbeddings:
    async def test_no_entities_returns_zero(self):
        storage = _make_storage()
        result = await backfill_entity_embeddings(uuid4(), storage)
        assert result == {"total_entities": 0, "entities_updated": 0}

    async def test_all_entities_already_have_embeddings(self):
        ns = uuid4()
        ent = Entity(namespace_id=ns, name="a", embedding=[0.1, 0.2])
        storage = _make_storage(list_entities=AsyncMock(return_value=[ent]))
        result = await backfill_entity_embeddings(ns, storage)
        assert result == {"total_entities": 1, "entities_updated": 0}

    async def test_generates_embeddings_for_missing(self):
        ns = uuid4()
        ent_a = Entity(namespace_id=ns, name="a")
        ent_b = Entity(namespace_id=ns, name="b", description="desc")
        storage = _make_storage(
            list_entities=AsyncMock(return_value=[ent_a, ent_b]),
            update_entity_embeddings_batch=AsyncMock(return_value=2),
        )
        storage.vector.entity_exists = AsyncMock(return_value=True)

        embedder = MagicMock(embed_batch=AsyncMock(return_value=[[0.1] * 3, [0.2] * 3]))
        with patch("khora.extraction.embedders.LiteLLMEmbedder", return_value=embedder):
            result = await backfill_entity_embeddings(ns, storage, batch_size=10)
        assert result["total_entities"] == 2
        assert result["entities_updated"] == 2
        storage.update_entity_embeddings_batch.assert_awaited_once()

    async def test_missing_in_postgres_triggers_create(self):
        ns = uuid4()
        ent = Entity(namespace_id=ns, name="a")
        storage = _make_storage(
            list_entities=AsyncMock(return_value=[ent]),
            update_entity_embeddings_batch=AsyncMock(return_value=1),
        )
        storage.vector.entity_exists = AsyncMock(return_value=False)
        storage.vector.create_entity = AsyncMock()

        embedder = MagicMock(embed_batch=AsyncMock(return_value=[[0.1] * 3]))
        with patch("khora.extraction.embedders.LiteLLMEmbedder", return_value=embedder):
            result = await backfill_entity_embeddings(ns, storage, batch_size=10)
        storage.vector.create_entity.assert_awaited_once()
        assert result["entities_updated"] == 1


# ---------------------------------------------------------------------------
# ingest_documents (orchestrator) — early-exit branches
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
class TestIngestDocumentsEarlyExit:
    async def test_missing_storage_raises_value_error(self):
        with pytest.raises(ValueError, match="storage is required"):
            await ingest_documents(
                uuid4(),
                [{"content": "x"}],
                storage=None,
                entity_types=["PERSON"],
                relationship_types=[],
            )

    async def test_empty_docs_returns_skipped_summary(self):
        storage = _make_storage()
        result = await ingest_documents(
            uuid4(),
            [],
            storage=storage,
            entity_types=["PERSON"],
            relationship_types=[],
        )
        assert result["total_documents"] == 0
        assert result["processed_documents"] == 0
        assert result["skipped_documents"] == 0

    async def test_all_docs_unchanged_returns_skipped(self):
        ns = uuid4()
        existing = MagicMock(status="completed")
        # Both checksums already in storage — nothing staged.
        storage = _make_storage(
            get_documents_by_checksums=AsyncMock(side_effect=lambda ns_, checksums: {c: existing for c in checksums})
        )
        result = await ingest_documents(
            ns,
            [{"content": "a"}, {"content": "b"}],
            storage=storage,
            entity_types=["PERSON"],
            relationship_types=[],
        )
        assert result["processed_documents"] == 0
        assert result["skipped_documents"] == 2
        assert result["total_chunks"] == 0

    async def test_invalid_expertise_name_falls_back_to_default(self):
        """Bad expertise name logs a warning but does not raise."""
        ns = uuid4()
        storage = _make_storage()
        # Make load_expertise raise so we hit the warning branch + None fallback.
        with patch("khora.extraction.skills.load_expertise", side_effect=FileNotFoundError("nope")):
            result = await ingest_documents(
                ns,
                [],
                storage=storage,
                expertise="does-not-exist",
                entity_types=["PERSON"],
                relationship_types=[],
            )
        assert result["processed_documents"] == 0


# ---------------------------------------------------------------------------
# _extract_cross_chunk_relationships
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
class TestExtractCrossChunkRelationships:
    async def test_disabled_when_flag_absent(self):
        # No cross_chunk_extraction flag → immediately bails.
        result = await _extract_cross_chunk_relationships(
            chunks=[MagicMock(), MagicMock()],
            entities_by_chunk={},
            extractor=MagicMock(),
            extraction_context={},
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
        )
        assert result == []

    async def test_fewer_than_two_chunks_returns_empty(self):
        result = await _extract_cross_chunk_relationships(
            chunks=[MagicMock(content="solo", id=uuid4())],
            entities_by_chunk={},
            extractor=MagicMock(),
            extraction_context={"cross_chunk_extraction": True},
            entity_types=[],
            relationship_types=[],
        )
        assert result == []

    async def test_finds_and_dedupes_across_windows(self):
        cid_a, cid_b, cid_c = uuid4(), uuid4(), uuid4()
        # Each chunk needs .content, .id, optional chunk_index (flat field)
        chunk_a = MagicMock(content="Alice met Bob.", id=cid_a, chunk_index=0)
        chunk_b = MagicMock(content="Bob works for Acme.", id=cid_b, chunk_index=1)
        chunk_c = MagicMock(content="Acme is in Boston.", id=cid_c, chunk_index=2)

        entities_by_chunk = {
            cid_a: ["Alice", "Bob"],
            cid_b: ["Bob", "Acme"],
            cid_c: ["Acme", "Boston"],
        }

        # Return the same Bob-Acme triple from both windows; should dedupe.
        bob_acme = ExtractedRelationship(
            source_entity="Bob",
            target_entity="Acme",
            relationship_type="WORKS_FOR",
            confidence=0.8,
        )
        acme_boston = ExtractedRelationship(
            source_entity="Acme",
            target_entity="Boston",
            relationship_type="LOCATED_IN",
            confidence=0.8,
        )
        results_window_0 = [ExtractionResult(relationships=[bob_acme])]
        results_window_1 = [ExtractionResult(relationships=[bob_acme, acme_boston])]

        extractor = MagicMock()
        extractor.extract_multi = AsyncMock(side_effect=[results_window_0, results_window_1])

        result = await _extract_cross_chunk_relationships(
            chunks=[chunk_a, chunk_b, chunk_c],
            entities_by_chunk=entities_by_chunk,
            extractor=extractor,
            extraction_context={"cross_chunk_extraction": True},
            entity_types=["PERSON", "ORG"],
            relationship_types=["WORKS_FOR", "LOCATED_IN"],
        )
        # bob-acme appears in both windows → dedup keeps one.
        triples = {(r.source_entity.lower(), r.relationship_type, r.target_entity.lower()) for r in result}
        assert ("bob", "WORKS_FOR", "acme") in triples
        assert ("acme", "LOCATED_IN", "boston") in triples
        assert len(result) == 2  # deduplication held

    async def test_window_failure_is_swallowed(self):
        cid_a, cid_b = uuid4(), uuid4()
        chunk_a = MagicMock(content="x", id=cid_a, chunk_index=0)
        chunk_b = MagicMock(content="y", id=cid_b, chunk_index=1)

        extractor = MagicMock()
        extractor.extract_multi = AsyncMock(side_effect=RuntimeError("LLM blew up"))

        result = await _extract_cross_chunk_relationships(
            chunks=[chunk_a, chunk_b],
            entities_by_chunk={cid_a: ["X"], cid_b: ["Y"]},
            extractor=extractor,
            extraction_context={"cross_chunk_extraction": True},
            entity_types=[],
            relationship_types=[],
        )
        # Failed windows yield no relationships but never raise.
        assert result == []

    async def test_skips_window_with_no_entities(self):
        cid_a, cid_b = uuid4(), uuid4()
        chunk_a = MagicMock(content="x", id=cid_a, chunk_index=0)
        chunk_b = MagicMock(content="y", id=cid_b, chunk_index=1)

        extractor = MagicMock()
        extractor.extract_multi = AsyncMock(return_value=[ExtractionResult()])

        result = await _extract_cross_chunk_relationships(
            chunks=[chunk_a, chunk_b],
            entities_by_chunk={cid_a: [], cid_b: []},
            extractor=extractor,
            extraction_context={"cross_chunk_extraction": True},
            entity_types=[],
            relationship_types=[],
        )
        assert result == []
        extractor.extract_multi.assert_not_called()
