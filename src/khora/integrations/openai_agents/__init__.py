"""``khora.integrations.openai_agents`` — OpenAI Agents SDK adapter.

Three independent primitives a caller mixes and matches:

* :class:`KhoraSession` — implements ``agents.memory.session.SessionABC``
  so a ``Runner.run(..., session=...)`` call rides on khora-backed
  conversation memory.
* :func:`khora_recall_tool` — factory returning a ``FunctionTool`` an
  ``Agent`` can call to recall memories from khora at run-time.
* :class:`KhoraMemoryHooks` — ``RunHooks``-shaped class that
  automatically writes tool results to khora and optionally recalls
  context on agent start.

The SDK's published package is ``openai-agents`` (PyPI) but installs as
the Python module ``agents`` (note: no ``openai_`` prefix). The
``[openai-agents]`` extra pulls it in.

Stability: experimental. The SDK is pre-1.0 (17 releases in 7 months as
of v0.17). The adapter pins ``openai-agents>=0.17,<0.18`` and bumps in a
deliberate PR per upstream minor — see ``docs/integrations/openai_agents.md``.

Module-load discipline: nothing from ``agents`` is imported at module
top level. Verified by ``tests/unit/integrations/test_no_eager_imports.py``
(subprocess probe). The AST lint ``tools/check_optional_imports.py``
catches eager imports of ``openai_agents`` (the directory name) but the
SDK's package is ``agents`` — the subprocess test is the gate of record.
"""

from __future__ import annotations

from khora.integrations.openai_agents.hooks import KhoraMemoryHooks
from khora.integrations.openai_agents.session import KhoraSession
from khora.integrations.openai_agents.tool import khora_recall_tool

__all__ = [
    "KhoraMemoryHooks",
    "KhoraSession",
    "khora_recall_tool",
]
