"""``khora.integrations.google_adk`` — Google ADK ``BaseMemoryService`` adapter.

End-user surface: :class:`KhoraMemoryService`. Drop into a
``google.adk.runners.Runner`` in place of ``InMemoryMemoryService`` /
``VertexAiMemoryBankService``::

    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from khora.integrations.google_adk import KhoraMemoryService

    runner = Runner(
        app_name="my_app",
        agent=my_agent,
        session_service=InMemorySessionService(),
        memory_service=KhoraMemoryService(kb=kb),
    )

Scope (per issue #626): this submodule ships ``KhoraMemoryService`` only.
A ``KhoraSessionService`` is intentionally **not** part of v1 — ADK's
``DatabaseSessionService`` already covers short-term turn state and khora
offers no differentiator there.

Module-load discipline: nothing from ``google.adk`` is imported at module
top level. The framework classes are resolved lazily on first
:class:`KhoraMemoryService` instantiation. Verified by
``tools/check_optional_imports.py`` (AST lint) plus the subprocess probe
in ``tests/unit/integrations/test_no_eager_imports.py``.

Stability: experimental until v0.14 ships one full minor without a
breaking change to this adapter.
"""

from __future__ import annotations

from khora.integrations.google_adk.memory_service import KhoraMemoryService

__all__ = ["KhoraMemoryService"]
