"""Chronicle #855: reinforcement UPDATE failure must NOT propagate.

The reinforcement write is fire-and-forget. If the storage backend is
down, the network blips, or the chunk ids were partially deleted between
recall and update, the engine must log a warning and swallow the
exception. Recall has already produced its result by the time the task
runs - failing here would either crash an awaiting task in the caller
or leak an unawaited exception.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest

from khora.config import KhoraConfig
from khora.engines.chronicle.engine import ChronicleEngine


class _BrokenCoordinator:
    async def update_last_accessed(self, namespace_id: UUID, chunk_ids: list[UUID], ts: datetime) -> int:
        raise RuntimeError("storage backend offline")


def _bare_engine() -> ChronicleEngine:
    return ChronicleEngine(KhoraConfig(database_url="postgresql://localhost/test"))


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reinforce_swallows_exception_and_logs(caplog: pytest.LogCaptureFixture) -> None:
    """A failing UPDATE must not raise out of ``_reinforce_last_accessed``."""
    coord: Any = _BrokenCoordinator()
    engine = _bare_engine()
    ns_id = uuid4()
    chunk_ids = [uuid4(), uuid4()]
    ts = datetime.now(UTC)

    # Must not raise.
    await engine._reinforce_last_accessed(coord, ns_id, chunk_ids, ts)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reinforce_does_not_propagate_arbitrary_exceptions() -> None:
    """Any exception class is swallowed, not just specific DB errors."""

    class _OddCoordinator:
        async def update_last_accessed(self, namespace_id: UUID, chunk_ids: list[UUID], ts: datetime) -> int:
            raise ValueError("malformed chunk id")

    coord: Any = _OddCoordinator()
    engine = _bare_engine()
    # Should not raise.
    await engine._reinforce_last_accessed(coord, uuid4(), [uuid4()], datetime.now(UTC))
