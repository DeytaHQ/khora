"""Adapter Protocols for ``khora.integrations``.

Three narrow, runtime-checkable Protocols. Each adapter implements one or
more of them; a god ABC with ``NotImplementedError`` defaults would hide
capability gaps until runtime, so we keep them separate.

- :class:`KhoraIntegration` — marker Protocol every adapter satisfies.
- :class:`MemoryAdapter` — CrewAI / AutoGen / generic chat-memory shape.
- :class:`RetrieverAdapter` — LangGraph / LlamaIndex retriever shape.

A fourth ``ToolkitAdapter`` is intentionally **not** shipped in v0.13.
It will be added when an actual adapter (OpenAI Agents SDK, Google ADK)
needs it — YAGNI until then.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from uuid import UUID

if TYPE_CHECKING:
    from khora.integrations.types import RetrievedNode
    from khora.khora import Khora


@runtime_checkable
class KhoraIntegration(Protocol):
    """Marker Protocol — every adapter exposes the same minimum surface.

    Attributes:
        name: Short identifier for the framework being adapted (e.g.
            ``"crewai"``, ``"langgraph"``). Used for logging, telemetry,
            and entry-point registration.
        kb: The bound :class:`khora.Khora` instance. Adapters MUST NOT
            instantiate their own ``Khora()`` — they accept one from the
            caller, who is responsible for its lifecycle.
        namespace_id: The memory namespace this adapter writes into.
    """

    name: str
    kb: Khora
    namespace_id: UUID


@runtime_checkable
class MemoryAdapter(Protocol):
    """Adapter shape for frameworks that expose a chat-memory store.

    Examples: CrewAI's ``StorageBackend``, AutoGen's memory hooks,
    LangChain's chat-message-history surface.

    Implementations wrap :meth:`khora.Khora.remember` and the search side
    of :meth:`khora.Khora.recall`. They keep narrow async signatures so
    static type checkers can ``isinstance(x, MemoryAdapter)`` pass.
    """

    async def asave(
        self,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> UUID:
        """Persist ``content`` to khora; return the document ID."""
        ...

    async def asearch(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Return up to ``limit`` matching items as plain dicts.

        Plain dicts (not :class:`RecallResult`) because the consuming
        framework is the one that decides the on-the-wire shape.
        """
        ...


@runtime_checkable
class RetrieverAdapter(Protocol):
    """Adapter shape for frameworks that expose a retriever / store.

    Examples: LangGraph's ``BaseStore``, LlamaIndex's
    ``BaseRetriever`` / vector-store interface.

    Implementations wrap :meth:`khora.Khora.recall` and surface khora
    chunks/entities as :class:`RetrievedNode` instances so the framework
    can render them in its own context-window format.
    """

    async def aretrieve(
        self,
        query: str,
        *,
        limit: int = 10,
    ) -> list[RetrievedNode]:
        """Retrieve up to ``limit`` nodes for ``query``."""
        ...
