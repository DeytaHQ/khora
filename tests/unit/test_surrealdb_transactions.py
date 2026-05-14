"""Unit tests for SurrealDB transaction + batch primitives (Issue #541)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from khora.storage.backends.surrealdb.connection import SurrealDBConnection


def _make_conn(mode: str = "remote") -> SurrealDBConnection:
    """Build a connection in the given mode with a mocked client."""
    if mode == "remote":
        conn = SurrealDBConnection(mode="remote", url="ws://example/rpc")
    elif mode == "embedded":
        conn = SurrealDBConnection(mode="embedded", path="/tmp/x.db")
    else:
        conn = SurrealDBConnection(mode="memory")
    conn._connected = True
    conn._client = AsyncMock()
    conn._client.query = AsyncMock(return_value=[])
    # Embedded/memory have a write semaphore; remote doesn't.
    # Either way _execute_raw uses _client.query.
    return conn


# ---------------------------------------------------------------------------
# supports_transactions property
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSupportsTransactions:
    def test_remote_supports_transactions(self) -> None:
        conn = _make_conn("remote")
        assert conn.supports_transactions is True

    def test_embedded_does_not_support_transactions(self) -> None:
        conn = _make_conn("embedded")
        assert conn.supports_transactions is False

    def test_memory_does_not_support_transactions(self) -> None:
        conn = _make_conn("memory")
        assert conn.supports_transactions is False


# ---------------------------------------------------------------------------
# transaction() context manager
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTransactionContextManager:
    @pytest.mark.asyncio
    async def test_remote_success_emits_begin_then_commit(self) -> None:
        conn = _make_conn("remote")

        async with conn.transaction():
            await conn.execute("CREATE thing SET x = 1")

        calls = [args[0] for args, _ in conn._client.query.call_args_list]
        assert calls == [
            "BEGIN TRANSACTION;",
            "CREATE thing SET x = 1",
            "COMMIT TRANSACTION;",
        ]

    @pytest.mark.asyncio
    async def test_remote_exception_emits_cancel_and_reraises(self) -> None:
        conn = _make_conn("remote")

        with pytest.raises(RuntimeError, match="boom"):
            async with conn.transaction():
                await conn.execute("CREATE thing SET x = 1")
                raise RuntimeError("boom")

        calls = [args[0] for args, _ in conn._client.query.call_args_list]
        assert calls == [
            "BEGIN TRANSACTION;",
            "CREATE thing SET x = 1",
            "CANCEL TRANSACTION;",
        ]

    @pytest.mark.asyncio
    async def test_embedded_is_noop(self) -> None:
        """Embedded mode must NOT issue BEGIN — surrealkv raises on it."""
        conn = _make_conn("embedded")

        async with conn.transaction():
            await conn.execute("CREATE thing SET x = 1")

        calls = [args[0] for args, _ in conn._client.query.call_args_list]
        assert "BEGIN TRANSACTION;" not in calls
        assert "COMMIT TRANSACTION;" not in calls
        assert calls == ["CREATE thing SET x = 1"]

    @pytest.mark.asyncio
    async def test_memory_is_noop(self) -> None:
        conn = _make_conn("memory")

        async with conn.transaction():
            await conn.execute("RETURN 1")

        calls = [args[0] for args, _ in conn._client.query.call_args_list]
        assert "BEGIN TRANSACTION;" not in calls
        assert calls == ["RETURN 1"]

    @pytest.mark.asyncio
    async def test_not_connected_raises_runtime_error(self) -> None:
        conn = _make_conn("remote")
        conn._connected = False

        with pytest.raises(RuntimeError, match="not connected"):
            async with conn.transaction():
                pass

    @pytest.mark.asyncio
    async def test_cancel_failure_does_not_mask_original_exception(self) -> None:
        """If CANCEL itself fails (e.g. broken WebSocket), surface the
        ORIGINAL exception, not the cancel error — that's the actual fault."""
        conn = _make_conn("remote")

        # First call: BEGIN succeeds.
        # Second call: user statement fails (we don't reach it via execute,
        # we raise directly).
        # Third call: CANCEL also raises.
        calls = []

        async def fake_query(sql, bindings=None):
            calls.append(sql)
            if sql == "CANCEL TRANSACTION;":
                raise RuntimeError("cancel-also-broken")
            return []

        conn._client.query = AsyncMock(side_effect=fake_query)

        with pytest.raises(ValueError, match="user-error"):
            async with conn.transaction():
                raise ValueError("user-error")

        assert "BEGIN TRANSACTION;" in calls
        assert "CANCEL TRANSACTION;" in calls


# ---------------------------------------------------------------------------
# execute_batch
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExecuteBatch:
    @pytest.mark.asyncio
    async def test_empty_returns_empty_no_call(self) -> None:
        conn = _make_conn("embedded")
        result = await conn.execute_batch([])
        assert result == []
        conn._client.query.assert_not_called()

    @pytest.mark.asyncio
    async def test_joins_statements_with_semicolons(self) -> None:
        conn = _make_conn("embedded")
        conn._client.query = AsyncMock(return_value=[{"r": 1}, {"r": 2}])

        result = await conn.execute_batch(
            [
                ("CREATE thing SET x = $a", {"a": 1}),
                ("CREATE thing SET x = $b", {"b": 2}),
            ]
        )

        # Result passed through unchanged.
        assert result == [{"r": 1}, {"r": 2}]

        sql, bindings = conn._client.query.call_args[0]
        assert sql == "CREATE thing SET x = $a;\nCREATE thing SET x = $b;"
        assert bindings == {"a": 1, "b": 2}

    @pytest.mark.asyncio
    async def test_parameter_collision_raises(self) -> None:
        conn = _make_conn("embedded")
        with pytest.raises(ValueError, match="parameter 'x' conflicts"):
            await conn.execute_batch(
                [
                    ("CREATE thing SET v = $x", {"x": 1}),
                    ("CREATE thing SET v = $x", {"x": 2}),
                ]
            )

    @pytest.mark.asyncio
    async def test_duplicate_binding_value_is_fine(self) -> None:
        """Same parameter name + same value across statements: merge silently."""
        conn = _make_conn("embedded")
        await conn.execute_batch(
            [
                ("CREATE a SET v = $x", {"x": 1}),
                ("CREATE b SET v = $x", {"x": 1}),
            ]
        )
        _, bindings = conn._client.query.call_args[0]
        assert bindings == {"x": 1}

    @pytest.mark.asyncio
    async def test_not_connected_raises(self) -> None:
        conn = _make_conn("embedded")
        conn._connected = False
        with pytest.raises(RuntimeError, match="not connected"):
            await conn.execute_batch([("RETURN 1", None)])

    @pytest.mark.asyncio
    async def test_trims_trailing_semicolons_to_avoid_double(self) -> None:
        conn = _make_conn("embedded")
        await conn.execute_batch([("CREATE a SET v = 1;", None), ("CREATE b SET v = 2;", None)])
        sql, _ = conn._client.query.call_args[0]
        # Each statement's trailing ';' is normalized; we join with one ';\n'.
        assert sql == "CREATE a SET v = 1;\nCREATE b SET v = 2;"
        assert ";;" not in sql
