"""Unit tests for ``TurbopufferTemporalStore`` (#824).

The actual turbopuffer SDK isn't a hard dependency at test time - we
inject a fake ``turbopuffer`` module into ``sys.modules`` so the
``TurbopufferTemporalStore.connect()`` lazy import resolves to our
stubs. Mirrors the openai_agents / hermes test patterns.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from pydantic import SecretStr

from khora.engines.skeleton.backends import TemporalChunk, TemporalFilter
from khora.engines.skeleton.backends.turbopuffer import (
    TurbopufferBackendConfig,
    TurbopufferTemporalStore,
    _build_turbopuffer_filter,
    _chunk_to_row,
    _coerce_datetime,
    _row_to_chunk,
    _rrf_fuse,
)
from khora.filter import (
    FilterClause,
    FilterNode,
    FilterOp,
    RecallFilter,
    RecallFilterUnsupportedError,
    parse_to_ast,
)

# ---------------------------------------------------------------------------
# TurbopufferBackendConfig validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTurbopufferBackendConfig:
    def test_minimal_valid_config(self) -> None:
        cfg = TurbopufferBackendConfig(api_key="tpuf_test")
        assert cfg.secret_api_key() == "tpuf_test"
        assert cfg.region == "gcp-us-central1"  # default
        assert cfg.namespace_prefix == "khora_"  # default

    def test_secret_str_api_key_unwrapped(self) -> None:
        cfg = TurbopufferBackendConfig(api_key=SecretStr("tpuf_secret"))
        assert cfg.secret_api_key() == "tpuf_secret"

    def test_empty_api_key_raises(self) -> None:
        with pytest.raises(ValueError, match="requires an `api_key`"):
            TurbopufferBackendConfig(api_key="")

    def test_custom_region_and_prefix(self) -> None:
        cfg = TurbopufferBackendConfig(
            api_key="k",
            region="aws-us-east-1",
            namespace_prefix="prod_",
        )
        assert cfg.region == "aws-us-east-1"
        assert cfg.namespace_prefix == "prod_"


# ---------------------------------------------------------------------------
# Helpers (pure functions, no SDK needed)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestChunkRowRoundtrip:
    def test_chunk_to_row_minimal(self) -> None:
        ns_id = uuid4()
        doc_id = uuid4()
        chunk_id = uuid4()
        chunk = TemporalChunk(
            id=chunk_id,
            namespace_id=ns_id,
            document_id=doc_id,
            content="hello",
            embedding=[0.1] * 4,
        )
        row = _chunk_to_row(chunk)
        assert row["id"] == str(chunk_id)
        assert row["document_id"] == str(doc_id)
        assert row["namespace_id"] == str(ns_id)
        assert row["content"] == "hello"
        assert row["vector"] == [0.1] * 4
        assert row["tags"] == []
        assert row["metadata_json"] == "{}"

    def test_chunk_to_row_full_fields(self) -> None:
        chunk = TemporalChunk(
            id=uuid4(),
            namespace_id=uuid4(),
            document_id=uuid4(),
            content="body",
            embedding=[0.5, 0.5],
            occurred_at=datetime(2026, 5, 1, tzinfo=UTC),
            created_at=datetime(2026, 5, 23, tzinfo=UTC),
            source_system="git",
            author="alice",
            channel="#dev",
            tags=["t1", "t2"],
            confidence=0.85,
            metadata={"k": "v"},
        )
        row = _chunk_to_row(chunk)
        assert row["occurred_at"] == "2026-05-01T00:00:00+00:00"
        assert row["created_at"] == "2026-05-23T00:00:00+00:00"
        assert row["source_system"] == "git"
        assert row["author"] == "alice"
        assert row["channel"] == "#dev"
        assert row["tags"] == ["t1", "t2"]
        assert row["confidence"] == 0.85
        assert row["metadata_json"] == '{"k": "v"}'

    def test_row_to_chunk_dict_form(self) -> None:
        ns_id = uuid4()
        doc_id = uuid4()
        chunk_id = uuid4()
        row = {
            "id": str(chunk_id),
            "document_id": str(doc_id),
            "content": "body",
            "vector": [0.2] * 4,
            "occurred_at": "2026-05-01T00:00:00+00:00",
            "created_at": "2026-05-23T00:00:00+00:00",
            "source_system": "git",
            "author": "alice",
            "channel": "#dev",
            "tags": ["t1"],
            "confidence": 0.9,
            "metadata_json": '{"k": "v"}',
        }
        chunk = _row_to_chunk(row, ns_id)
        assert chunk.id == chunk_id
        assert chunk.namespace_id == ns_id
        assert chunk.document_id == doc_id
        assert chunk.content == "body"
        assert chunk.embedding == [0.2] * 4
        assert chunk.occurred_at == datetime(2026, 5, 1, tzinfo=UTC)
        assert chunk.tags == ["t1"]
        assert chunk.metadata == {"k": "v"}

    def test_row_to_chunk_sdk_object_form(self) -> None:
        """SDK returns Pydantic-style Row objects in 2.x; tolerate that."""
        ns_id = uuid4()
        doc_id = uuid4()
        chunk_id = uuid4()
        row = SimpleNamespace(
            id=str(chunk_id),
            document_id=str(doc_id),
            content="body",
            vector=None,
            occurred_at=None,
            created_at=None,
            source_system=None,
            author=None,
            channel=None,
            tags=None,
            confidence=None,
            metadata_json=None,
        )
        chunk = _row_to_chunk(row, ns_id)
        assert chunk.id == chunk_id
        assert chunk.embedding is None
        assert chunk.confidence == 1.0  # default fallback
        assert chunk.tags == []
        assert chunk.metadata == {}


@pytest.mark.unit
class TestCoerceDatetime:
    def test_none(self) -> None:
        assert _coerce_datetime(None) is None

    def test_datetime_passthrough(self) -> None:
        dt = datetime(2026, 5, 23, tzinfo=UTC)
        assert _coerce_datetime(dt) is dt

    def test_iso_string(self) -> None:
        out = _coerce_datetime("2026-05-23T00:00:00Z")
        assert out == datetime(2026, 5, 23, tzinfo=UTC)

    def test_invalid_returns_none(self) -> None:
        assert _coerce_datetime("not a date") is None
        assert _coerce_datetime(42) is None


# ---------------------------------------------------------------------------
# Filter compiler
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildTurbopufferFilter:
    def test_none_filter_returns_none(self) -> None:
        assert _build_turbopuffer_filter(None) is None

    def test_empty_filter_returns_none(self) -> None:
        assert _build_turbopuffer_filter(TemporalFilter()) is None

    def test_single_predicate_returns_bare_clause(self) -> None:
        f = TemporalFilter(author="alice")
        assert _build_turbopuffer_filter(f) == ("author", "Eq", "alice")

    def test_temporal_range(self) -> None:
        f = TemporalFilter(
            occurred_after=datetime(2026, 1, 1, tzinfo=UTC),
            occurred_before=datetime(2026, 6, 1, tzinfo=UTC),
        )
        out = _build_turbopuffer_filter(f)
        assert out == (
            "And",
            (
                ("occurred_at", "Gte", "2026-01-01T00:00:00+00:00"),
                ("occurred_at", "Lt", "2026-06-01T00:00:00+00:00"),
            ),
        )

    def test_tags_all_semantics_fanout(self) -> None:
        """ALL-tags must fold into N Contains clauses (no native ContainsAll)."""
        f = TemporalFilter(tags=["urgent", "review"])
        out = _build_turbopuffer_filter(f)
        assert out == (
            "And",
            (
                ("tags", "Contains", "urgent"),
                ("tags", "Contains", "review"),
            ),
        )

    def test_additional_dict_operators(self) -> None:
        f = TemporalFilter(
            additional={
                "priority": {"gte": 3},
                "label": {"in": ["x", "y"]},
                "owner": "bob",  # bare value -> Eq
            }
        )
        out = _build_turbopuffer_filter(f)
        assert isinstance(out, tuple) and out[0] == "And"
        clauses = out[1]
        # Order matters in our compiler (dict iteration order is insertion order in 3.7+).
        assert ("priority", "Gte", 3) in clauses
        assert ("label", "In", ["x", "y"]) in clauses
        assert ("owner", "Eq", "bob") in clauses

    def test_combined_complex_filter(self) -> None:
        f = TemporalFilter(
            source_system="git",
            channel="#dev",
            tags=["pr"],
        )
        out = _build_turbopuffer_filter(f)
        assert out[0] == "And"
        assert ("source_system", "Eq", "git") in out[1]
        assert ("channel", "Eq", "#dev") in out[1]
        assert ("tags", "Contains", "pr") in out[1]


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRrfFuse:
    def test_disjoint_lists_combine_with_rrf_scores(self) -> None:
        vector_rows = [
            {"id": "a", "$dist": 0.1},
            {"id": "b", "$dist": 0.2},
        ]
        text_rows = [
            {"id": "c"},
            {"id": "d"},
        ]
        fused = _rrf_fuse(vector_rows=vector_rows, text_rows=text_rows, k=60, limit=10)
        ids = [str(row.get("id") if isinstance(row, dict) else row.id) for row, _, _ in fused]
        # Top-1 from each channel ties; order is stable on dict ordering but
        # both top-1s should appear before either's bottom entry.
        assert "a" in ids[:2]
        assert "c" in ids[:2]
        assert "b" in ids[2:]
        assert "d" in ids[2:]

    def test_shared_id_gets_summed_score(self) -> None:
        """An id appearing in both channels gets both contributions."""
        vector_rows = [{"id": "shared", "$dist": 0.05}, {"id": "vec_only"}]
        text_rows = [{"id": "shared"}, {"id": "text_only"}]
        fused = _rrf_fuse(vector_rows=vector_rows, text_rows=text_rows, k=60, limit=10)
        # ``shared`` is rank 1 in both → score 2 * 1/61
        shared_score = next(
            score for row, score, _ in fused if (row["id"] if isinstance(row, dict) else row.id) == "shared"
        )
        vec_only_score = next(
            score for row, score, _ in fused if (row["id"] if isinstance(row, dict) else row.id) == "vec_only"
        )
        assert shared_score > vec_only_score

    def test_limit_truncates(self) -> None:
        vector_rows = [{"id": f"v{i}"} for i in range(20)]
        text_rows = [{"id": f"t{i}"} for i in range(20)]
        fused = _rrf_fuse(vector_rows=vector_rows, text_rows=text_rows, k=60, limit=5)
        assert len(fused) == 5

    def test_vector_distance_preserved_when_vector_channel_has_id(self) -> None:
        vector_rows = [{"id": "x", "$dist": 0.3}]
        text_rows: list[Any] = []
        fused = _rrf_fuse(vector_rows=vector_rows, text_rows=text_rows, k=60, limit=10)
        assert fused[0][2] == 0.3  # vec_dist preserved

    def test_vector_distance_none_for_text_only_hits(self) -> None:
        vector_rows: list[Any] = []
        text_rows = [{"id": "text_only"}]
        fused = _rrf_fuse(vector_rows=vector_rows, text_rows=text_rows, k=60, limit=10)
        assert fused[0][2] is None


# ---------------------------------------------------------------------------
# Connect + namespace mapping + I/O (with fake SDK injection)
# ---------------------------------------------------------------------------


def _install_fake_turbopuffer(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, MagicMock]:
    """Inject a fake ``turbopuffer`` module into ``sys.modules``.

    Returns ``(turbopuffer_module, client_instance)``.
    """
    import sys

    fake_tp = ModuleType("turbopuffer")

    client = MagicMock(name="AsyncTurbopuffer")
    client.close = AsyncMock()
    client.namespaces = MagicMock()
    client.namespaces.write = AsyncMock(return_value=SimpleNamespace(rows_affected=1, upserted_rows=1))

    ns_handle = MagicMock(name="NamespaceHandle")
    ns_handle.query = AsyncMock(return_value=SimpleNamespace(rows=[]))
    client.namespace = MagicMock(return_value=ns_handle)

    fake_tp.AsyncTurbopuffer = MagicMock(return_value=client)  # type: ignore[attr-defined]
    fake_tp.Turbopuffer = MagicMock()  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "turbopuffer", fake_tp)
    return fake_tp, client


def _build_store(turbopuffer_cfg: str | TurbopufferBackendConfig) -> TurbopufferTemporalStore:
    config = MagicMock(name="KhoraConfig")
    config.llm.embedding_dimension = 1536
    return TurbopufferTemporalStore(config, turbopuffer_cfg)


@pytest.mark.unit
class TestConnect:
    @pytest.mark.asyncio
    async def test_connect_passes_api_key_and_region(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tp, _client = _install_fake_turbopuffer(monkeypatch)
        store = _build_store(TurbopufferBackendConfig(api_key="tpuf_k", region="aws-us-east-1"))
        await store.connect()
        tp.AsyncTurbopuffer.assert_called_once()
        kwargs = tp.AsyncTurbopuffer.call_args.kwargs
        assert kwargs["api_key"] == "tpuf_k"
        assert kwargs["region"] == "aws-us-east-1"
        assert "base_url" not in kwargs  # not set

    @pytest.mark.asyncio
    async def test_connect_with_base_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tp, _client = _install_fake_turbopuffer(monkeypatch)
        store = _build_store(
            TurbopufferBackendConfig(
                api_key="k",
                base_url="https://proxy.internal:8443",
            )
        )
        await store.connect()
        assert tp.AsyncTurbopuffer.call_args.kwargs["base_url"] == "https://proxy.internal:8443"

    @pytest.mark.asyncio
    async def test_connect_string_api_key_back_compat(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tp, _client = _install_fake_turbopuffer(monkeypatch)
        store = _build_store("tpuf_bare_string")
        await store.connect()
        assert tp.AsyncTurbopuffer.call_args.kwargs["api_key"] == "tpuf_bare_string"

    @pytest.mark.asyncio
    async def test_disconnect_awaits_close(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _tp, client = _install_fake_turbopuffer(monkeypatch)
        store = _build_store("k")
        await store.connect()
        await store.disconnect()
        client.close.assert_awaited()
        assert store._client is None


@pytest.mark.unit
class TestNamespaceMapping:
    @pytest.mark.asyncio
    async def test_namespace_name_uses_prefix_and_uuid_hex(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_turbopuffer(monkeypatch)
        store = _build_store(TurbopufferBackendConfig(api_key="k", namespace_prefix="khora_"))
        await store.connect()
        ns_id = UUID("12345678-1234-5678-1234-567812345678")
        assert store._namespace_name(ns_id) == "khora_12345678123456781234567812345678"

    @pytest.mark.asyncio
    async def test_custom_namespace_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_turbopuffer(monkeypatch)
        store = _build_store(TurbopufferBackendConfig(api_key="k", namespace_prefix="prod_"))
        await store.connect()
        ns_id = uuid4()
        assert store._namespace_name(ns_id).startswith("prod_")


@pytest.mark.unit
class TestCRUD:
    @pytest.mark.asyncio
    async def test_create_chunk_calls_namespaces_write(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _tp, client = _install_fake_turbopuffer(monkeypatch)
        store = _build_store("k")
        await store.connect()

        ns_id = uuid4()
        chunk = TemporalChunk(
            id=uuid4(),
            namespace_id=ns_id,
            document_id=uuid4(),
            content="x",
            embedding=[0.1] * 4,
        )
        await store.create_chunk(chunk)

        client.namespaces.write.assert_awaited_once()
        kwargs = client.namespaces.write.call_args.kwargs
        assert kwargs["distance_metric"] == "cosine_distance"
        assert len(kwargs["upsert_rows"]) == 1
        assert kwargs["upsert_rows"][0]["content"] == "x"
        assert kwargs["namespace"].startswith("khora_")

    @pytest.mark.asyncio
    async def test_create_chunks_batch_groups_by_namespace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _tp, client = _install_fake_turbopuffer(monkeypatch)
        store = _build_store("k")
        await store.connect()

        ns_a = uuid4()
        ns_b = uuid4()
        chunks = [
            TemporalChunk(id=uuid4(), namespace_id=ns_a, document_id=uuid4(), content=f"a{i}", embedding=[0.0] * 4)
            for i in range(3)
        ] + [
            TemporalChunk(id=uuid4(), namespace_id=ns_b, document_id=uuid4(), content=f"b{i}", embedding=[0.0] * 4)
            for i in range(2)
        ]
        await store.create_chunks_batch(chunks)

        # Two HTTP writes - one per namespace, not one per chunk.
        assert client.namespaces.write.await_count == 2
        # Each write carries the right batch sizes.
        seen_sizes = sorted(len(call.kwargs["upsert_rows"]) for call in client.namespaces.write.call_args_list)
        assert seen_sizes == [2, 3]

    @pytest.mark.asyncio
    async def test_delete_chunk_uses_filter_eq_on_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _tp, client = _install_fake_turbopuffer(monkeypatch)
        store = _build_store("k")
        await store.connect()

        chunk_id = uuid4()
        ok = await store.delete_chunk(chunk_id, uuid4())
        assert ok is True

        client.namespaces.write.assert_awaited_once()
        kwargs = client.namespaces.write.call_args.kwargs
        assert kwargs["delete_by_filter"] == ("id", "Eq", str(chunk_id))

    @pytest.mark.asyncio
    async def test_delete_chunks_by_document_returns_rows_affected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _tp, client = _install_fake_turbopuffer(monkeypatch)
        client.namespaces.write = AsyncMock(return_value=SimpleNamespace(rows_affected=7))
        store = _build_store("k")
        await store.connect()
        count = await store.delete_chunks_by_document(uuid4(), uuid4())
        assert count == 7

    @pytest.mark.asyncio
    async def test_get_chunk_returns_none_when_no_rows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _tp, client = _install_fake_turbopuffer(monkeypatch)
        store = _build_store("k")
        await store.connect()
        # The ns_handle.query default returns rows=[] - so get_chunk -> None.
        chunk = await store.get_chunk(uuid4(), uuid4())
        assert chunk is None


@pytest.mark.unit
class TestSearch:
    @pytest.mark.asyncio
    async def test_vector_only_search(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _tp, client = _install_fake_turbopuffer(monkeypatch)

        ns_id = uuid4()
        doc_id = uuid4()
        chunk_id = uuid4()

        row = {
            "id": str(chunk_id),
            "document_id": str(doc_id),
            "content": "match",
            "$dist": 0.2,
            "tags": [],
            "metadata_json": "{}",
        }
        ns_handle = MagicMock()
        ns_handle.query = AsyncMock(return_value=SimpleNamespace(rows=[row]))
        client.namespace = MagicMock(return_value=ns_handle)

        store = _build_store("k")
        await store.connect()

        results = await store.search(
            namespace_id=ns_id,
            query_embedding=[0.1] * 4,
            limit=10,
        )
        assert len(results) == 1
        assert results[0].chunk.id == chunk_id
        assert results[0].similarity == pytest.approx(0.8)  # 1 - 0.2
        # Vector-only path passes a single rank_by + no rank_by="text".
        ns_handle.query.assert_awaited_once()
        kwargs = ns_handle.query.call_args.kwargs
        assert kwargs["rank_by"][0] == "vector"

    @pytest.mark.asyncio
    async def test_hybrid_search_fans_two_queries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _tp, client = _install_fake_turbopuffer(monkeypatch)

        ns_id = uuid4()
        vec_row = {
            "id": str(uuid4()),
            "document_id": str(uuid4()),
            "content": "vec",
            "$dist": 0.1,
            "metadata_json": "{}",
        }
        text_row = {
            "id": str(uuid4()),
            "document_id": str(uuid4()),
            "content": "text",
            "metadata_json": "{}",
        }
        ns_handle = MagicMock()
        # ns.query is called twice in parallel; AsyncMock side_effect cycles.
        ns_handle.query = AsyncMock(
            side_effect=[
                SimpleNamespace(rows=[vec_row]),
                SimpleNamespace(rows=[text_row]),
            ]
        )
        client.namespace = MagicMock(return_value=ns_handle)

        store = _build_store("k")
        await store.connect()
        results = await store.search(
            namespace_id=ns_id,
            query_embedding=[0.1] * 4,
            limit=10,
            hybrid_alpha=0.5,  # any non-None + query_text triggers hybrid
            query_text="search query",
        )
        # Two channels, two ids.
        assert len(results) == 2
        assert ns_handle.query.await_count == 2
        # The two calls use different rank_by channels.
        rank_bys = {call.kwargs["rank_by"][0] for call in ns_handle.query.call_args_list}
        assert rank_bys == {"vector", "content"}

    @pytest.mark.asyncio
    async def test_min_similarity_filters_out_low_scores(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _tp, client = _install_fake_turbopuffer(monkeypatch)
        # Two rows: one above threshold, one below.
        rows = [
            {
                "id": str(uuid4()),
                "document_id": str(uuid4()),
                "content": "high",
                "$dist": 0.1,
                "metadata_json": "{}",
            },  # sim=0.9
            {
                "id": str(uuid4()),
                "document_id": str(uuid4()),
                "content": "low",
                "$dist": 0.7,
                "metadata_json": "{}",
            },  # sim=0.3
        ]
        ns_handle = MagicMock()
        ns_handle.query = AsyncMock(return_value=SimpleNamespace(rows=rows))
        client.namespace = MagicMock(return_value=ns_handle)

        store = _build_store("k")
        await store.connect()
        results = await store.search(
            namespace_id=uuid4(),
            query_embedding=[0.1] * 4,
            limit=10,
            min_similarity=0.5,
        )
        assert len(results) == 1
        assert results[0].chunk.content == "high"

    @pytest.mark.asyncio
    async def test_filter_threading(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _tp, client = _install_fake_turbopuffer(monkeypatch)
        ns_handle = MagicMock()
        ns_handle.query = AsyncMock(return_value=SimpleNamespace(rows=[]))
        client.namespace = MagicMock(return_value=ns_handle)

        store = _build_store("k")
        await store.connect()
        await store.search(
            namespace_id=uuid4(),
            query_embedding=[0.1] * 4,
            limit=10,
            temporal_filter=TemporalFilter(author="alice"),
        )
        # The compiled filter reaches the SDK call.
        assert ns_handle.query.call_args.kwargs["filters"] == ("author", "Eq", "alice")


@pytest.mark.unit
class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_disconnected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_turbopuffer(monkeypatch)
        store = _build_store("k")
        out = await store.health_check()
        assert out["status"] == "disconnected"

    @pytest.mark.asyncio
    async def test_connected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_turbopuffer(monkeypatch)
        store = _build_store("k")
        await store.connect()
        out = await store.health_check()
        assert out["status"] == "healthy"


@pytest.mark.unit
class TestImportErrorWhenSdkMissing:
    @pytest.mark.asyncio
    async def test_connect_without_sdk_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If turbopuffer isn't importable, connect() must raise a clear
        ImportError pointing at the optional extra."""
        import sys

        # Ensure no fake module is in sys.modules either - this matches
        # the real-world case where the user didn't install the extra.
        monkeypatch.setitem(sys.modules, "turbopuffer", None)
        store = _build_store("k")
        with pytest.raises(ImportError, match="turbopuffer is required"):
            await store.connect()


# ---------------------------------------------------------------------------
# Recall filter_ast fail-loud contract
# ---------------------------------------------------------------------------
#
# turbopuffer does not implement deterministic recall filters, so a
# constraint-bearing ``filter_ast`` (non-empty ``children``) must fail loud
# (raise) rather than silently return unfiltered rows. ``None`` keeps the
# existing behavior unchanged; a constraint-free filter (match-everything AND
# with no children, what ``filter={}`` normalizes to) passes through as a no-op.


def _filter_ast_node() -> FilterNode:
    """A small real ``FilterNode`` — ``AND([author $eq "alice"])``."""
    return FilterNode(
        op=FilterOp.AND,
        children=(FilterClause(path=("author",), op=FilterOp.EQ, operand="alice"),),
    )


@pytest.mark.unit
class TestFilterAstFailLoud:
    @pytest.mark.asyncio
    async def test_filter_ast_none_keeps_existing_behavior(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``filter_ast=None`` still searches and returns rows from the SDK."""
        _tp, client = _install_fake_turbopuffer(monkeypatch)

        ns_id = uuid4()
        row = {
            "id": str(uuid4()),
            "document_id": str(uuid4()),
            "content": "match",
            "$dist": 0.2,
            "tags": [],
            "metadata_json": "{}",
        }
        ns_handle = MagicMock()
        ns_handle.query = AsyncMock(return_value=SimpleNamespace(rows=[row]))
        client.namespace = MagicMock(return_value=ns_handle)

        store = _build_store("k")
        await store.connect()

        results = await store.search(ns_id, [0.1] * 4, filter_ast=None)
        assert len(results) == 1
        assert results[0].chunk.content == "match"

    @pytest.mark.asyncio
    async def test_empty_filter_ast_passes_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A constraint-free ``filter_ast`` is a no-op and passes through.

        ``filter={}`` / ``RecallFilter()`` normalizes to a match-everything
        AST: an AND root with no children. It's not ``None``, but it carries
        no predicates, so the backend can honor it trivially by searching
        without any filter — it must NOT fail loud. This is the interesting
        case: not-None yet constraint-free.
        """
        empty = parse_to_ast(RecallFilter())
        # Precondition documenting WHY this case matters: not None, no children.
        assert empty is not None and not empty.children

        _tp, client = _install_fake_turbopuffer(monkeypatch)

        ns_id = uuid4()
        row = {
            "id": str(uuid4()),
            "document_id": str(uuid4()),
            "content": "match",
            "$dist": 0.2,
            "tags": [],
            "metadata_json": "{}",
        }
        ns_handle = MagicMock()
        ns_handle.query = AsyncMock(return_value=SimpleNamespace(rows=[row]))
        client.namespace = MagicMock(return_value=ns_handle)

        store = _build_store("k")
        await store.connect()

        results = await store.search(ns_id, [0.1] * 4, filter_ast=empty)
        assert len(results) == 1
        assert results[0].chunk.content == "match"

    @pytest.mark.asyncio
    async def test_filter_ast_node_raises_unsupported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A constraint-bearing ``filter_ast`` raises ``RecallFilterUnsupportedError``.

        The node here carries a real predicate (``author $eq "alice"``), so its
        ``children`` is non-empty and the guard fires. The guard short-circuits
        before any namespace I/O, but we still connect a fake SDK to match the
        file's fixture approach.
        """
        _install_fake_turbopuffer(monkeypatch)
        store = _build_store("k")
        await store.connect()

        node = _filter_ast_node()
        assert node.children  # precondition: this filter carries a constraint
        with pytest.raises(RecallFilterUnsupportedError):
            await store.search(uuid4(), [0.1] * 4, filter_ast=node)
