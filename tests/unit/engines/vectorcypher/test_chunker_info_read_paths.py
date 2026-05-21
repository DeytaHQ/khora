"""Unit tests for the ``chunker_info`` read paths.

The propagation contract has two boundaries that the engine-level
:mod:`test_chunker_info_propagation` cannot reach without real Neo4j /
SurrealDB infrastructure:

* The VectorCypher retriever rebuilds :class:`Chunk` objects from Neo4j
  record dicts at two sites (``c.chunker_info`` projection in the typed
  entity fast path and the graph-search path). Neo4j stores
  ``chunker_info`` as a JSON-encoded string at write time (see
  :func:`dual_nodes.create_chunk_nodes_batch`); the read sites must
  deserialize that string, while staying robust to native dicts and to
  corrupted values that would otherwise crash ``recall()``.
* The SurrealDB temporal store maps result rows to ``TemporalChunk``
  via :meth:`_row_to_chunk`. ``chunker_info`` arrives there as a dict
  (object-typed field) and must survive missing keys / wrong types.

The two tests below pin the deserialization contract at each boundary
without any cross-network dependencies.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from khora.engines.skeleton.backends.surrealdb import SurrealDBTemporalStore
from khora.engines.vectorcypher.retriever import _decode_chunker_info

# ---------------------------------------------------------------------------
# Neo4j retriever boundary — ``_decode_chunker_info`` helper
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDecodeChunkerInfo:
    """Pin the helper used at both Neo4j retriever sites."""

    def test_native_dict_passes_through(self) -> None:
        value = {"chunker": "fixed", "size": 100}
        assert _decode_chunker_info(value) == value

    def test_json_string_is_parsed(self) -> None:
        # Neo4j stores chunker_info as a JSON string at write time.
        result = _decode_chunker_info('{"chunker": "semantic"}')
        assert result == {"chunker": "semantic"}

    def test_none_yields_empty_dict(self) -> None:
        assert _decode_chunker_info(None) == {}

    def test_missing_key_default_is_empty_dict(self) -> None:
        # The retriever calls _decode_chunker_info(record.get("chunker_info"));
        # when the projection is missing entirely, .get returns None.
        record: dict[str, object] = {}
        assert _decode_chunker_info(record.get("chunker_info")) == {}

    def test_malformed_json_does_not_raise(self) -> None:
        # A corrupted persisted JSON string (direct DB tampering, a partial
        # write) must not crash recall(). Fall back to {}.
        assert _decode_chunker_info("{not valid json") == {}

    def test_json_null_yields_empty_dict(self) -> None:
        # 'null' parses to Python None, which is not a dict.
        assert _decode_chunker_info("null") == {}

    def test_json_array_yields_empty_dict(self) -> None:
        # A JSON-array string parses to a list — also not a dict.
        assert _decode_chunker_info('["chunker"]') == {}

    def test_non_string_non_dict_yields_empty_dict(self) -> None:
        assert _decode_chunker_info(42) == {}
        assert _decode_chunker_info(0.5) == {}
        assert _decode_chunker_info(True) == {}


# ---------------------------------------------------------------------------
# SurrealDB read boundary — ``_row_to_chunk``
# ---------------------------------------------------------------------------


def _surreal_row(**overrides: object) -> dict[str, object]:
    """Build a minimal SurrealDB result row.

    ``_row_to_chunk`` reads `id`, `namespace`, `document`, plus the
    runtime fields. The UUID helpers tolerate non-record-id strings,
    so a bare hex string suffices.
    """
    chunk_id = uuid4()
    namespace_id = uuid4()
    document_id = uuid4()
    base: dict[str, object] = {
        "id": str(chunk_id),
        "namespace": str(namespace_id),
        "document": str(document_id),
        "content": "hello",
        "embedding": None,
        "occurred_at": None,
        "created_at": None,
        "source_system": None,
        "author": None,
        "channel": None,
        "tags": None,
        "confidence": 1.0,
        "metadata_": {},
        "chunker_info": {},
    }
    base.update(overrides)
    return base


@pytest.mark.unit
class TestSurrealDBRowToChunk:
    """Pin the SurrealDB read-side mapping for ``chunker_info``."""

    def test_dict_chunker_info_is_preserved(self) -> None:
        row = _surreal_row(chunker_info={"chunker": "recursive", "max_chars": 500})
        chunk = SurrealDBTemporalStore._row_to_chunk(row)
        assert chunk.chunker_info == {"chunker": "recursive", "max_chars": 500}

    def test_missing_chunker_info_defaults_to_empty_dict(self) -> None:
        row = _surreal_row()
        del row["chunker_info"]
        chunk = SurrealDBTemporalStore._row_to_chunk(row)
        assert chunk.chunker_info == {}

    def test_none_chunker_info_defaults_to_empty_dict(self) -> None:
        row = _surreal_row(chunker_info=None)
        chunk = SurrealDBTemporalStore._row_to_chunk(row)
        assert chunk.chunker_info == {}

    def test_non_dict_chunker_info_defaults_to_empty_dict(self) -> None:
        # SurrealDB declares the field as object-typed, but a corrupted
        # write could land a string. The mapper must not crash.
        row = _surreal_row(chunker_info="not-a-dict")
        chunk = SurrealDBTemporalStore._row_to_chunk(row)
        assert chunk.chunker_info == {}
