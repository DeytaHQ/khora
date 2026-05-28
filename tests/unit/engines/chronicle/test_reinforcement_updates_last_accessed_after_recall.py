"""Chronicle #855: reinforcement-on-recall stamps last_accessed_at on returned chunks.

We exercise the contract directly via the engine's
``_reinforce_last_accessed`` helper - the same call that's spawned by
``recall()`` as an ``asyncio.create_task``. This avoids needing to mock
the whole 4-channel recall pipeline while still pinning the API the
recall path depends on.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest

from khora.config import KhoraConfig
from khora.engines.chronicle.engine import ChronicleEngine


class _RecordingCoordinator:
    """Records arguments passed to ``update_last_accessed``."""

    def __init__(self) -> None:
        self.calls: list[tuple[UUID, list[UUID], datetime]] = []

    async def update_last_accessed(self, namespace_id: UUID, chunk_ids: list[UUID], ts: datetime) -> int:
        self.calls.append((namespace_id, list(chunk_ids), ts))
        return len(chunk_ids)


def _bare_engine() -> ChronicleEngine:
    return ChronicleEngine(KhoraConfig(database_url="postgresql://localhost/test"))


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reinforce_last_accessed_calls_storage_with_chunk_ids() -> None:
    """``_reinforce_last_accessed`` forwards the chunk ids + ts unchanged."""
    coord: Any = _RecordingCoordinator()
    engine = _bare_engine()
    ns_id = uuid4()
    chunk_ids = [uuid4(), uuid4()]
    ts = datetime.now(UTC)

    await engine._reinforce_last_accessed(coord, ns_id, chunk_ids, ts)

    assert len(coord.calls) == 1
    (got_ns, got_ids, got_ts) = coord.calls[0]
    assert got_ns == ns_id
    assert got_ids == chunk_ids
    # Timestamp must be within 5s of "now" - covers the "approximately now"
    # invariant the recall path promises.
    assert abs((datetime.now(UTC) - got_ts).total_seconds()) < 5.0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reinforce_last_accessed_ts_is_within_5s_of_now() -> None:
    """When the engine spawns the task with ``datetime.now(UTC)``, the
    receiver sees a timestamp within 5s of the present moment."""
    coord: Any = _RecordingCoordinator()
    engine = _bare_engine()
    ns_id = uuid4()
    chunk_ids = [uuid4()]

    fired_at = datetime.now(UTC)
    await engine._reinforce_last_accessed(coord, ns_id, chunk_ids, fired_at)

    (_ns, _ids, ts) = coord.calls[0]
    assert abs((ts - fired_at).total_seconds()) < 1e-3
    assert abs((datetime.now(UTC) - ts).total_seconds()) < 5.0
