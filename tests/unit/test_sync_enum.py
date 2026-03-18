"""Tests for khora.db.schema – enum synchronisation helpers."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest

from khora.db.schema import sync_enum_values


def _make_engine(mock_conn: AsyncMock) -> AsyncMock:
    """Create a mock AsyncEngine whose .connect() yields *mock_conn*."""
    engine = AsyncMock()

    @asynccontextmanager
    async def _connect():
        yield mock_conn

    engine.connect = _connect
    return engine


@pytest.mark.unit
async def test_sync_enum_values_executes_alter_type_for_each_value():
    """sync_enum_values issues ALTER TYPE … ADD VALUE IF NOT EXISTS for every
    value defined in the _ENUM_SYNC mapping."""
    mock_conn = AsyncMock()
    engine = _make_engine(mock_conn)

    await sync_enum_values(engine)

    # Should have called execution_options for autocommit
    mock_conn.execution_options.assert_called_once_with(isolation_level="AUTOCOMMIT")

    # Should have issued ALTER TYPE for each DocumentStatus value
    execute_calls = mock_conn.execute.call_args_list
    executed_values = [c.args[1]["val"] for c in execute_calls]
    assert "pending" in executed_values
    assert "processing" in executed_values
    assert "completed" in executed_values
    assert "failed" in executed_values
    assert "archived" in executed_values


@pytest.mark.unit
async def test_sync_enum_values_handles_missing_type_gracefully():
    """If the enum type does not yet exist (first run), sync_enum_values
    should catch the exception and not raise."""
    mock_conn = AsyncMock()
    mock_conn.execute.side_effect = Exception("type does not exist")
    engine = _make_engine(mock_conn)

    # Should not raise
    await sync_enum_values(engine)
