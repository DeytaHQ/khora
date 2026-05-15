"""Khora integrations — adapters for agentic frameworks.

This package hosts adapter shims that wrap khora's async ops
(``remember``, ``recall``, ``forget``, ...) into the storage / retrieval
interfaces expected by third-party frameworks (CrewAI, LangGraph,
OpenAI Agents SDK, Google ADK, ...).

The package itself is part of khora's stable public API. Individual
``khora.integrations.<framework>`` submodules are tagged
``stability: experimental`` until they ship one full khora minor without
a breaking change.

Design pillars (see GitHub issue #619):

1. Narrow Protocols (`MemoryAdapter`, `RetrieverAdapter`) — adapters
   declare exactly what they support; ``isinstance`` checks work at
   runtime.
2. Plugin discovery via entry points (group ``khora.integrations``) with
   an explicit ``register()`` escape hatch for tests and notebooks.
3. Optional-install discipline — adapter submodules must NEVER import
   their framework at module top level. Enforced by
   ``tools/check_optional_imports.py`` in CI plus a runtime subprocess
   probe test.
4. Single sync-bridge helper (`khora.integrations._sync.run_sync`)
   handles the sync-framework-into-async-khora pattern in one place.
5. Process-wide ``Khora.shared()`` singleton for ephemeral CLI / agent
   runs that don't own a long-lived connection pool.
"""

from __future__ import annotations

from khora.integrations.protocol import (
    KhoraIntegration,
    MemoryAdapter,
    RetrieverAdapter,
)
from khora.integrations.registry import discover, register
from khora.integrations.types import RetrievedNode

__all__ = [
    "KhoraIntegration",
    "MemoryAdapter",
    "RetrieverAdapter",
    "RetrievedNode",
    "discover",
    "register",
]
