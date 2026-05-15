"""LangGraph adapter — ``KhoraStore`` for semantic long-term memory (#624).

LangGraph's ``BaseStore`` is the long-term-memory surface a graph holds
in addition to its short-term ``Checkpointer``. This adapter wires khora
(vector search + entity graph) behind that interface so a graph can
``graph.compile(store=KhoraStore(kb, user_id=...))``.

Scope decision (see #624): this submodule ships ``KhoraStore`` only.
A LangGraph ``Checkpointer`` adapter is intentionally NOT shipped —
``langgraph-postgres``'s ``PostgresSaver`` already covers that surface
and khora offers no differentiator there. Revisit only if a real user
needs the one-DB story.

Stability: experimental until v0.13 ships one full minor without a
breaking change (see ``khora.integrations`` docstring).

Module-load discipline: this file imports nothing from ``langgraph``;
the framework import lives inside :class:`KhoraStore` methods. Verified
by ``tools/check_optional_imports.py`` (AST lint) plus the subprocess
probe in ``tests/unit/integrations/test_no_eager_imports.py``.
"""

from __future__ import annotations

from khora.integrations.langgraph.store import KhoraStore

__all__ = ["KhoraStore"]
