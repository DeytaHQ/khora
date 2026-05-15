"""Verify the sync bridge raises (rather than deadlocks) on reentrancy.

Per #619, ``khora.integrations._sync.run_sync`` refuses to be called
from inside a running asyncio loop. The CrewAI adapter inherits that
contract: invoking any ``KhoraStorageBackend`` method from inside an
``async def`` must raise ``RuntimeError``, not block.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.integrations.crewai.storage import KhoraStorageBackend
from khora.khora import Khora


@dataclass
class _FakeRecord:
    id: str
    content: str
    scope: str = "/"
    categories: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5
    source: str | None = None
    private: bool = False
    created_at: Any = None
    last_accessed: Any = None
    embedding: list[float] | None = None


def _make_backend() -> KhoraStorageBackend:
    kb = AsyncMock(spec=Khora)
    kb.storage = MagicMock()
    return KhoraStorageBackend(
        kb=kb,
        namespace_id=uuid4(),
        user_id="user-12345678",
        app_id="crewai",
        memory_record_cls=_FakeRecord,
    )


def test_save_raises_rather_than_deadlocks_inside_running_loop() -> None:
    """Calling ``backend.save`` from inside ``asyncio.run`` must raise."""
    backend = _make_backend()
    record = _FakeRecord(id="r-1", content="x")

    async def driver() -> None:
        with pytest.raises(RuntimeError, match="running event loop"):
            backend.save([record])

    asyncio.run(driver())


def test_search_raises_rather_than_deadlocks_inside_running_loop() -> None:
    backend = _make_backend()

    async def driver() -> None:
        with pytest.raises(RuntimeError, match="running event loop"):
            backend.search([0.0])

    asyncio.run(driver())


def test_get_record_raises_rather_than_deadlocks_inside_running_loop() -> None:
    backend = _make_backend()

    async def driver() -> None:
        with pytest.raises(RuntimeError, match="running event loop"):
            backend.get_record("anything")

    asyncio.run(driver())
