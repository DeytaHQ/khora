"""Shared fixtures for ``khora.integrations.hermes`` provider tests.

The provider tests need a ``hermes_agent.agent.memory_provider.MemoryProvider``
ABC to subclass from — without it, the factory raises ``ImportError``. We
install a minimal in-memory fake into ``sys.modules`` so the import works
without requiring the optional ``hermes-agent`` distribution (which is
intentionally NOT declared as a khora extra; see CHANGELOG #628).

Mirrors the ``sys.modules`` injection pattern used for openai_agents tests
but with autouse scope so every provider test starts with the fake in
place. Tests that need to verify the *missing-extra* error path (i.e.
``test_factory_raises_clear_error_without_hermes_agent``) explicitly remove
the modules and reset the cached ABC pointer.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from types import ModuleType
from typing import Any

import pytest


def _build_fake_hermes_agent_modules() -> tuple[ModuleType, ModuleType, ModuleType, type]:
    """Return (hermes_agent, hermes_agent.agent, hermes_agent.agent.memory_provider, MemoryProvider).

    The ABC matches the surface ``KhoraMemoryProvider`` subclasses against:
    abstract ``name``, ``is_available``, ``initialize``, ``get_tool_schemas``
    plus concrete default no-op hooks for the rest. Provider's
    ``_KhoraMemoryProviderImpl`` overrides ``name`` as a property and the
    other abstract methods as concrete implementations, so this ABC will
    accept the subclass instantiation.
    """
    fake = ModuleType("hermes_agent")
    fake_agent = ModuleType("hermes_agent.agent")
    fake_mp = ModuleType("hermes_agent.agent.memory_provider")

    class MemoryProvider(ABC):
        @property
        @abstractmethod
        def name(self) -> str: ...

        @abstractmethod
        def is_available(self) -> bool: ...

        @abstractmethod
        def initialize(self, session_id: str, **kwargs: Any) -> None: ...

        @abstractmethod
        def get_tool_schemas(self) -> list[dict[str, Any]]: ...

        def system_prompt_block(self) -> str:
            return ""

        def prefetch(self, query: str, **kw: Any) -> str:
            return ""

        def queue_prefetch(self, query: str, **kw: Any) -> None:
            return None

        def sync_turn(self, user: str, assistant: str, **kw: Any) -> None:
            return None

        def handle_tool_call(self, name: str, args: dict[str, Any]) -> str:
            return ""

        def shutdown(self) -> None:
            return None

    fake_mp.MemoryProvider = MemoryProvider
    fake_agent.memory_provider = fake_mp
    fake.agent = fake_agent
    return fake, fake_agent, fake_mp, MemoryProvider


@pytest.fixture(autouse=True)
def _fake_hermes_agent(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Inject a minimal hermes_agent into sys.modules.

    Also resets the provider module's cached ``_MemoryProviderBase`` so
    each test sees a freshly-resolved ABC. ``monkeypatch.setitem`` auto-
    rolls back via its finalizer when the test ends.
    """
    fake, fake_agent, fake_mp, MemoryProvider = _build_fake_hermes_agent_modules()
    monkeypatch.setitem(sys.modules, "hermes_agent", fake)
    monkeypatch.setitem(sys.modules, "hermes_agent.agent", fake_agent)
    monkeypatch.setitem(sys.modules, "hermes_agent.agent.memory_provider", fake_mp)

    # Reset the provider's cached ABC pointer so the lazy resolver picks
    # up our fake on first call.
    try:
        from khora.integrations.hermes import provider as _provider_mod

        monkeypatch.setattr(_provider_mod, "_MemoryProviderBase", None, raising=False)
    except ImportError:  # pragma: no cover - shouldn't happen
        pass
    return MemoryProvider
