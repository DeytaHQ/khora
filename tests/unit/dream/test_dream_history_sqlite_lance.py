"""sqlite_lance parity: ``dream_history`` persists run rows (#896).

Both the dream run-row write (``_init_run_row``) and the reads
(``history`` / ``status``) used to short-circuit when the session was
not Postgres, so ``Khora.dream()`` succeeded on sqlite_lance but never
persisted a row and ``dream_history()`` always returned ``[]``.

Migration 032 now creates ``khora_dream_runs`` on both dialects, and the
orchestrator's INSERT / SELECT bind UUIDs as text on SQLite (per #875).
These tests run a real dry-run dream on the embedded stack and assert
the run is returned by ``dream_history`` / ``dream_status``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from uuid import UUID

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples._helpers import embedded_khora, install_mock_llm  # noqa: E402
from khora.dream.config import DreamConfig  # noqa: E402

pytestmark = pytest.mark.embedded


async def _remember(kb, namespace_id):
    return await kb.remember(
        "PostgreSQL was chosen for the user database.",
        namespace=namespace_id,
        entity_types=[],
        relationship_types=[],
    )


async def test_dream_history_returns_run_on_sqlite_lance() -> None:
    install_mock_llm(dim=8)
    async with embedded_khora(embedding_dimension=8, engine="vectorcypher") as kb:
        ns = await kb.create_namespace()
        await _remember(kb, ns.namespace_id)

        result = await kb.dream(ns.namespace_id, mode="dry-run", config=DreamConfig(enabled=True))
        run_id = result.run.run_id

        history = await kb.dream_history(ns.namespace_id)
        assert len(history) >= 1, "dream_history returned [] on sqlite_lance"

        # The persisted row round-trips as proper UUID / datetime types.
        info = history[0]
        assert isinstance(info.run_id, UUID)
        assert info.run_id == run_id
        assert info.namespace_id == result.run.namespace_id
        assert info.started_at is not None


async def test_dream_status_returns_run_on_sqlite_lance() -> None:
    install_mock_llm(dim=8)
    async with embedded_khora(embedding_dimension=8, engine="vectorcypher") as kb:
        ns = await kb.create_namespace()
        await _remember(kb, ns.namespace_id)

        result = await kb.dream(ns.namespace_id, mode="dry-run", config=DreamConfig(enabled=True))
        run_id = result.run.run_id

        status = await kb.dream_status(run_id)
        assert status, "dream_status returned {} on sqlite_lance"
        assert status["run_id"] == str(run_id)
        assert status["mode"] == "dry-run"
