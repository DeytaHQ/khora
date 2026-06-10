"""Dispatch-capability tests: SurrealDB satisfies the reinforcement-on-recall gate.

Chronicle's reinforcement-on-recall path stamps ``chunk.last_accessed_at`` on every
recall. The dispatch to the vector backend is ``hasattr``-gated in two places —
``StorageCoordinator.update_last_accessed`` (delegates only when the backend implements
the method) and ``ChronicleEngine.connect`` (fails loudly at connect time when the
configured backend does NOT, so the operator isn't given a config flag with no effect).

These tests prove the SurrealDB vector adapter now clears that gate:

* the method is present on the class (the exact predicate both gate sites check), and
* ``ChronicleEngine.connect`` no longer raises ``ConfigurationError`` for a SurrealDB
  vector backend when ``chronicle_enable_recall_reinforcement=True``.

A negative control with a capability-less backend confirms the gate is real (it DOES
raise), so the SurrealDB pass is meaningful rather than the check being dead.

The ``surrealdb`` SDK is an optional dependency, so this module self-skips when absent.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("surrealdb")

from khora.config import KhoraConfig  # noqa: E402
from khora.config.schema import QuerySettings  # noqa: E402
from khora.engines.chronicle import engine as engine_mod  # noqa: E402
from khora.engines.chronicle.engine import ChronicleEngine  # noqa: E402
from khora.exceptions import ConfigurationError  # noqa: E402
from khora.storage.backends.surrealdb.vector import SurrealDBVectorAdapter  # noqa: E402
from khora.storage.coordinator import StorageCoordinator  # noqa: E402

pytestmark = pytest.mark.unit


def test_surrealdb_vector_adapter_exposes_update_last_accessed() -> None:
    # The exact predicate both dispatch gates check: StorageCoordinator.update_last_accessed
    # delegates only when getattr(self._vector, "update_last_accessed", None) is not None,
    # and ChronicleEngine.connect raises unless hasattr(vector, "update_last_accessed").
    assert hasattr(SurrealDBVectorAdapter, "update_last_accessed"), (
        "SurrealDBVectorAdapter must implement update_last_accessed so the Chronicle "
        "reinforcement-on-recall dispatch routes to it instead of silently no-opping"
    )


def _coordinator_with_vector(vector: object) -> StorageCoordinator:
    """A coordinator whose vector backend is ``vector`` and whose connect is a no-op.

    The capability gate inspects only ``self._storage._vector`` (via ``hasattr``); it
    never touches the connection, so a real adapter over a ``MagicMock`` connection is
    enough to exercise the real gate without any SurrealDB I/O.
    """
    coordinator = StorageCoordinator(relational=None, vector=vector)
    coordinator.connect = AsyncMock(return_value=None)  # type: ignore[method-assign]
    return coordinator


def _patch_connect_io(monkeypatch: pytest.MonkeyPatch, coordinator: StorageCoordinator) -> None:
    """Stub the I/O steps of ``ChronicleEngine.connect`` around the capability gate."""
    monkeypatch.setattr(engine_mod, "create_storage_coordinator", lambda _cfg: coordinator)
    monkeypatch.setattr(engine_mod.LiteLLMEmbedder, "from_config", classmethod(lambda _cls, _cfg: MagicMock()))
    monkeypatch.setattr("khora.telemetry.init_telemetry", AsyncMock(return_value=None))


@pytest.mark.asyncio
async def test_connect_does_not_raise_for_surrealdb_vector_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    # With reinforcement enabled, connect must NOT raise for a SurrealDB vector backend —
    # the adapter implements update_last_accessed, so the gate at engine.connect passes.
    config = KhoraConfig(query=QuerySettings(chronicle_enable_recall_reinforcement=True))
    coordinator = _coordinator_with_vector(SurrealDBVectorAdapter(MagicMock()))
    _patch_connect_io(monkeypatch, coordinator)

    engine = ChronicleEngine(config)
    await engine.connect()  # must not raise ConfigurationError

    assert engine._connected is True


@pytest.mark.asyncio
async def test_connect_raises_for_capability_less_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    # Negative control: the gate is real. A vector backend WITHOUT update_last_accessed
    # makes connect raise ConfigurationError when reinforcement is enabled — proving the
    # SurrealDB pass above is meaningful, not a dead check.
    config = KhoraConfig(query=QuerySettings(chronicle_enable_recall_reinforcement=True))

    class _NoReinforcement:
        """A vector backend that does not implement update_last_accessed."""

    coordinator = _coordinator_with_vector(_NoReinforcement())
    _patch_connect_io(monkeypatch, coordinator)

    engine = ChronicleEngine(config)
    with pytest.raises(ConfigurationError, match="update_last_accessed"):
        await engine.connect()
