"""Coverage: ``temporal_chunk_to_chunk`` adapter (GitHub issue #813).

The adapter sits between the skeleton temporal store
(``TemporalChunk`` from ``khora_chunks``) and the public ``Chunk``
surface the retriever expects. Three fields must survive the
adaptation:

* ``chunker_info`` (GH #800) — drives downstream chunker telemetry
* ``created_at`` (GH #810) — drives temporal-decay reranking
* ``session_id`` (GH #620) — pulled from ``TemporalChunk.metadata``

``occurred_at`` (chunk event-time) and ``source_timestamp`` (producer
verbatim time) are surfaced as distinct fields; the recall projection
applies the event-time-then-producer-time fallback downstream.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from khora.storage.temporal import (
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


def test_surfaces_occurred_at_distinctly() -> None:
    """The chunk event-time is surfaced on its own field, not collapsed."""
    ts = datetime(2024, 11, 20, 8, 0, tzinfo=UTC)
    c = temporal_chunk_to_chunk(_make_tc(occurred_at=ts))
    assert c.occurred_at == ts
    # No producer value supplied → source_timestamp stays unset; the
    # event-time-then-producer-time fallback lives in the recall projection.
    assert c.source_timestamp is None


def test_source_timestamp_is_distinct_from_occurred_at() -> None:
    """The producer ``source_timestamp`` and chunk ``occurred_at`` both survive.

    Tripwire for date-collapse: when a chunk carries BOTH a chunk event-time
    (``occurred_at``) and a distinct producer time (``source_timestamp``), the
    adapter must surface each as its own value — never collapse one onto the
    other.
    """
    occurred = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    produced = datetime(2024, 6, 15, 9, 30, tzinfo=UTC)
    assert occurred != produced

    c = temporal_chunk_to_chunk(_make_tc(occurred_at=occurred, source_timestamp=produced))

    assert c.occurred_at == occurred
    assert c.source_timestamp == produced
    assert c.source_timestamp != c.occurred_at


def test_source_timestamp_not_derived_from_occurred_at() -> None:
    """No producer value → source_timestamp stays None (no adapter fallback).

    The event-time-then-producer-time fallback now lives in the recall
    projection, so the adapter must not silently derive source_timestamp
    from the chunk event-time.
    """
    occurred = datetime(2024, 3, 10, 7, 0, tzinfo=UTC)
    c = temporal_chunk_to_chunk(_make_tc(occurred_at=occurred, source_timestamp=None))
    assert c.occurred_at == occurred
    assert c.source_timestamp is None


def test_source_timestamp_none_when_both_absent() -> None:
    """Neither producer time nor chunk event-time → ``None``."""
    c = temporal_chunk_to_chunk(_make_tc(occurred_at=None, source_timestamp=None))
    assert c.source_timestamp is None


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


def test_carries_position_bookkeeping_from_chunker_info() -> None:
    """The four bookkeeping fields are read from ``chunker_info``.

    Chunk position bookkeeping (chunk_index / start_char / end_char /
    token_count) lives in ``chunker_info`` — ``metadata`` is reserved for
    user/document metadata only. The adapter must populate the public
    ``Chunk`` position fields from ``chunker_info``.
    """
    ci = {
        "chunker": "fixed",
        "chunk_index": 3,
        "start_char": 100,
        "end_char": 200,
        "token_count": 25,
    }
    c = temporal_chunk_to_chunk(_make_tc(chunker_info=ci))
    assert c.chunk_index == 3
    assert c.start_char == 100
    assert c.end_char == 200
    assert c.token_count == 25


def test_position_bookkeeping_not_read_from_metadata_no_fallback() -> None:
    """Legacy-shaped chunk (bookkeeping in ``metadata``, empty
    ``chunker_info``) yields ZEROS — there is NO metadata fallback.

    Pins the no-fallback decision explicitly: bookkeeping now flows through
    ``chunker_info`` exclusively. A ``TemporalChunk`` that carries the four
    keys in ``metadata`` (the pre-refactor shape) with an empty
    ``chunker_info`` must NOT have those values surfaced on the public
    ``Chunk`` — every position field falls back to 0. The ``metadata`` dict
    itself still passes through untouched.
    """
    legacy_md = {"chunk_index": 3, "start_char": 100, "end_char": 200, "token_count": 25}
    c = temporal_chunk_to_chunk(_make_tc(metadata=legacy_md, chunker_info={}))

    assert c.chunk_index == 0
    assert c.start_char == 0
    assert c.end_char == 0
    assert c.token_count == 0
    # metadata is passed through verbatim — the values are present there,
    # they are simply not read for the position fields anymore.
    assert c.metadata == legacy_md


def test_identity_fields_pass_through() -> None:
    tc = _make_tc()
    c = temporal_chunk_to_chunk(tc)
    assert c.id == tc.id
    assert c.namespace_id == tc.namespace_id
    assert c.document_id == tc.document_id
    assert c.content == tc.content
