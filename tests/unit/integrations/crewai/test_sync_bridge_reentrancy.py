"""Verify the sync bridge dispatches correctly when called from a running loop.

Originally this asserted ``run_sync`` raised on reentrancy. CrewAI's
flow runtime calls our sync ``StorageBackend`` methods from inside its
own asyncio loop, so refusing was the wrong default — the bridge now
dispatches to a separate daemon-thread loop in all cases (see
``khora.integrations._sync``). These tests pin the new contract: sync
methods invoked from inside an async context complete normally.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

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
    # Provide a remember stub that returns a result with a document_id.
    remember_result = MagicMock()
    remember_result.document_id = uuid4()
    kb.remember = AsyncMock(return_value=remember_result)
    kb.recall = AsyncMock(return_value=MagicMock(chunks=[]))
    kb.forget = AsyncMock(return_value=True)
    # ``storage`` needs to be a regular MagicMock with AsyncMock attributes
    # for the methods the adapter awaits.
    storage = MagicMock()
    storage.get_chunks_by_document = AsyncMock(return_value=[])
    storage.list_documents = AsyncMock(return_value=[])
    storage.get_document_by_external_id = AsyncMock(return_value=None)
    kb.storage = storage
    return KhoraStorageBackend(
        kb=kb,
        namespace_id=uuid4(),
        user_id="user-12345678",
        app_id="crewai",
        memory_record_cls=_FakeRecord,
    )


def test_save_works_from_inside_running_loop() -> None:
    """Calling ``backend.save`` from inside ``asyncio.run`` must complete.

    CrewAI's flow listeners are sync functions executed from within an
    async runtime — the sync bridge must dispatch to its daemon-thread
    loop and block the calling thread until the coroutine finishes.
    """
    backend = _make_backend()
    record = _FakeRecord(id="r-1", content="x")

    async def driver() -> None:
        backend.save([record])

    asyncio.run(driver())
    # remember was awaited on the daemon thread's loop
    assert backend.kb.remember.await_count == 1


def test_search_works_from_inside_running_loop() -> None:
    backend = _make_backend()

    async def driver() -> list:
        return backend.search([0.0])

    out = asyncio.run(driver())
    assert out == []
    assert backend.kb.recall.await_count == 1


def test_get_record_works_from_inside_running_loop() -> None:
    backend = _make_backend()

    async def driver() -> Any:
        return backend.get_record("anything")

    # get_record returns None for unknown ids — no exception.
    assert asyncio.run(driver()) is None
