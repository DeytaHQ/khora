"""Coverage: ``temporal_chunk_to_chunk`` adapter (GitHub issue #813).

The adapter sits between the skeleton temporal store
(``TemporalChunk`` from ``khora_chunks``) and the public ``Chunk``
surface the retriever expects. Three fields must survive the
adaptation:

* ``chunker_info`` (GH #800) — drives downstream chunker telemetry
* ``created_at`` (GH #810) — drives temporal-decay reranking
* ``session_id`` (GH #620) — pulled from ``TemporalChunk.metadata``

``occurred_at`` maps to ``Chunk.source_timestamp`` so temporal boosts
key on event time rather than ingest time.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from khora.engines.skeleton.backends import (
    TemporalChunk,
    temporal_chunk_to_chunk,
)


def _make_tc(**overrides: object) -> TemporalChunk:
    base = {
        "id": uuid4(),
        "namespace_id": uuid4(),
        "document_id": uuid4(),
        "content": "hello world",
        "embedding": None,
        "occurred_at": datetime(2025, 4, 1, 12, 0, tzinfo=UTC),
        "created_at": datetime(2025, 4, 2, 9, 30, tzinfo=UTC),
        "source_system": "test",
        "metadata": {},
        "chunker_info": {"chunker": "fixed", "size": 1024},
    }
    base.update(overrides)
    return TemporalChunk(**base)  # type: ignore[arg-type]


def test_preserves_chunker_info() -> None:
    tc = _make_tc(chunker_info={"chunker": "semantic", "model": "v2"})
    c = temporal_chunk_to_chunk(tc)
    assert c.chunker_info == {"chunker": "semantic", "model": "v2"}


def test_preserves_created_at() -> None:
    ts = datetime(2024, 12, 15, 10, 0, tzinfo=UTC)
    c = temporal_chunk_to_chunk(_make_tc(created_at=ts))
    assert c.created_at == ts


def test_maps_occurred_at_to_source_timestamp() -> None:
    ts = datetime(2024, 11, 20, 8, 0, tzinfo=UTC)
    c = temporal_chunk_to_chunk(_make_tc(occurred_at=ts))
    assert c.source_timestamp == ts


def test_pulls_session_id_from_metadata_uuid_str() -> None:
    sid = uuid4()
    c = temporal_chunk_to_chunk(_make_tc(metadata={"session_id": str(sid)}))
    assert c.session_id == sid


def test_pulls_session_id_from_metadata_uuid_obj() -> None:
    sid = uuid4()
    c = temporal_chunk_to_chunk(_make_tc(metadata={"session_id": sid}))
    assert c.session_id == sid


def test_session_id_absent_when_missing_or_malformed() -> None:
    assert temporal_chunk_to_chunk(_make_tc(metadata={})).session_id is None
    assert temporal_chunk_to_chunk(_make_tc(metadata={"session_id": None})).session_id is None
    assert temporal_chunk_to_chunk(_make_tc(metadata={"session_id": "not-a-uuid"})).session_id is None
    assert temporal_chunk_to_chunk(_make_tc(metadata={"session_id": ""})).session_id is None


def test_carries_position_metadata() -> None:
    md = {"chunk_index": 3, "start_char": 100, "end_char": 200, "token_count": 25}
    c = temporal_chunk_to_chunk(_make_tc(metadata=md))
    assert c.chunk_index == 3
    assert c.start_char == 100
    assert c.end_char == 200
    assert c.token_count == 25


def test_identity_fields_pass_through() -> None:
    tc = _make_tc()
    c = temporal_chunk_to_chunk(tc)
    assert c.id == tc.id
    assert c.namespace_id == tc.namespace_id
    assert c.document_id == tc.document_id
    assert c.content == tc.content
