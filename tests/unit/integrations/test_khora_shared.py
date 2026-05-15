"""Tests for the ``Khora.shared()`` process-wide singleton (#619).

The singleton is meant for ephemeral adapter contexts where allocating
a fresh `Khora()` per call would churn the asyncpg pool. We test:

* First call creates + connects.
* Second call returns the SAME instance (no reconnect).
* Calls with different config get different instances.
* ``Khora.shared.clear()`` disconnects every cached instance and resets
  the cache so the next call rebuilds.
* The init path is lock-protected — concurrent first-callers don't
  double-instantiate.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from khora import Khora, KhoraConfig
from khora.khora import _SHARED_INSTANCES, _config_hash


@pytest.fixture(autouse=True)
async def clear_shared_between_tests():
    """Reset the process-wide cache around every test."""
    await Khora.shared.clear()
    yield
    await Khora.shared.clear()


def _stub_lifecycle() -> tuple[AsyncMock, AsyncMock]:
    """Patch Khora.connect / Khora.disconnect with AsyncMocks."""
    connect = AsyncMock()

    async def fake_connect(self) -> None:
        await connect(self)
        self._connected = True

    async def fake_disconnect(self) -> None:
        self._connected = False

    return connect, fake_connect, fake_disconnect  # type: ignore[return-value]


@pytest.fixture
def patched_lifecycle():
    """Patch connect/disconnect so tests don't touch a real DB."""
    connect_spy, fake_connect, fake_disconnect = _stub_lifecycle()
    with (
        patch.object(Khora, "connect", fake_connect),
        patch.object(Khora, "disconnect", fake_disconnect),
    ):
        yield connect_spy


async def test_first_call_creates_and_connects(patched_lifecycle):
    kb = await Khora.shared(KhoraConfig())
    assert isinstance(kb, Khora)
    assert kb._connected is True
    assert patched_lifecycle.call_count == 1


async def test_second_call_returns_same_instance(patched_lifecycle):
    cfg = KhoraConfig()
    kb_a = await Khora.shared(cfg)
    kb_b = await Khora.shared(cfg)
    assert kb_a is kb_b
    # connect() only fired once.
    assert patched_lifecycle.call_count == 1


async def test_different_config_returns_different_instance(patched_lifecycle):
    cfg_a = KhoraConfig()
    cfg_b = KhoraConfig()
    # Force the two configs to hash differently by mutating a field.
    cfg_b.llm.embedding_model = "text-embedding-3-large"
    assert _config_hash(cfg_a) != _config_hash(cfg_b)
    kb_a = await Khora.shared(cfg_a)
    kb_b = await Khora.shared(cfg_b)
    assert kb_a is not kb_b
    assert patched_lifecycle.call_count == 2


async def test_clear_disconnects_and_resets_cache(patched_lifecycle):
    cfg = KhoraConfig()
    kb_first = await Khora.shared(cfg)
    assert kb_first._connected is True
    assert len(_SHARED_INSTANCES) == 1

    await Khora.shared.clear()

    assert _SHARED_INSTANCES == {}
    assert kb_first._connected is False

    # Next call rebuilds.
    kb_second = await Khora.shared(cfg)
    assert kb_second is not kb_first
    assert kb_second._connected is True


async def test_concurrent_first_callers_share_one_instance(patched_lifecycle):
    cfg = KhoraConfig()
    instances = await asyncio.gather(*(Khora.shared(cfg) for _ in range(8)))
    assert all(inst is instances[0] for inst in instances)
    # Lock-protected: connect() ran exactly once even under contention.
    assert patched_lifecycle.call_count == 1


async def test_shared_without_config_uses_load_config(patched_lifecycle):
    # If no config is passed, load_config() is called once per cache
    # miss. We patch it so the test doesn't read real env.
    with patch("khora.khora.load_config", return_value=KhoraConfig()) as load:
        kb_a = await Khora.shared()
        kb_b = await Khora.shared()
    assert kb_a is kb_b
    # load_config is called per shared() invocation that needs a config
    # — that's fine because once the cache is populated the same key
    # short-circuits the second time. Important: at least one call.
    assert load.call_count >= 1


async def test_clear_is_idempotent(patched_lifecycle):
    await Khora.shared(KhoraConfig())
    await Khora.shared.clear()
    # Second clear on an empty cache is a no-op.
    await Khora.shared.clear()
    assert _SHARED_INSTANCES == {}


def test_config_hash_is_stable():
    cfg = KhoraConfig()
    assert _config_hash(cfg) == _config_hash(cfg)


def test_config_hash_differs_for_different_configs():
    cfg_a = KhoraConfig()
    cfg_b = KhoraConfig()
    cfg_b.llm.model = "gpt-4"
    assert _config_hash(cfg_a) != _config_hash(cfg_b)
