"""``KhoraMemoryHooks`` — ``RunHooks`` that mirror agent state into khora.

Wire one of these into a ``Runner.run(..., hooks=...)`` call to have the
SDK automatically store tool results and (optionally) recall relevant
context before the agent runs::

    from agents import Agent, Runner
    from khora.integrations.openai_agents import KhoraMemoryHooks

    hooks = KhoraMemoryHooks(kb=kb, namespace=ns_id, app_id="my_app")
    result = await Runner.run(agent, "Look up X", hooks=hooks)

Two hook points are implemented:

* ``on_tool_end`` — persists the returned tool ``result`` string to khora
  via ``Khora.remember``, stamped with the tool name and timestamp. Lets
  later sessions vector-search across what the agent learned.
* ``on_agent_start`` — when ``recall_on_start=True``, runs
  ``Khora.recall`` against the agent's most recent input items and
  surfaces matches via :func:`logger.info`. This is a starter — most
  callers will want to subclass and feed the hits into ``agent.instructions``
  or a system message instead.

Module-load discipline: ``agents.RunHooks`` is imported inside method
bodies. The class itself is plain (not a subclass of ``RunHooks``);
``Runner`` duck-types hook callbacks, so a class that exposes the right
async methods is accepted without ``isinstance`` checks. We document
this and provide ``KhoraMemoryHooks.as_runhooks()`` for callers that
want a real ``RunHooks`` subclass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from loguru import logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from agents.agent import Agent
    from agents.lifecycle import RunHooks
    from agents.run_context import RunContextWrapper
    from agents.tool import Tool

    from khora.khora import Khora


# Cached RunHooks subclass with KhoraMemoryHooks' methods grafted on.
# Built lazily by :meth:`KhoraMemoryHooks.as_runhooks`.
_RunHooksAdapter: type | None = None


class KhoraMemoryHooks:
    """Hooks that persist agent activity to khora as it happens.

    Args:
        kb: A connected :class:`khora.Khora` instance.
        namespace: khora namespace UUID. All writes / reads use this.
        app_id: Free-form app identifier stamped into stored metadata.
            Default ``"openai_agents"``.
        record_tool_results: When True (default), persist every
            successful tool result via ``Khora.remember``.
        recall_on_start: When True, log top khora hits on ``on_agent_start``
            so subclasses can see the signal without overriding the hook.
            Default False — silent by default.
        recall_top_k: ``limit`` passed to ``Khora.recall`` on
            ``on_agent_start``. Default 3.

    Concurrency: the hooks fire from inside the SDK's own asyncio loop,
    which is the same loop ``Khora.remember`` / ``Khora.recall`` expect.
    No sync-bridge needed — all access stays async.

    Subclassing: override :meth:`on_tool_end` / :meth:`on_agent_start`
    in your own subclass to layer custom behaviour. Call ``super()``
    first to keep the default write / recall behaviour.
    """

    name: str = "openai_agents"

    def __init__(
        self,
        *,
        kb: Khora,
        namespace: UUID,
        app_id: str = "openai_agents",
        record_tool_results: bool = True,
        recall_on_start: bool = False,
        recall_top_k: int = 3,
    ) -> None:
        if not isinstance(namespace, UUID):
            raise TypeError(f"namespace must be a UUID, got {type(namespace).__name__}")
        if not isinstance(app_id, str) or not app_id.strip():
            raise ValueError(f"app_id must be a non-empty string, got {app_id!r}")
        if recall_top_k < 1:
            raise ValueError(f"recall_top_k must be >= 1, got {recall_top_k}")

        self.kb = kb
        self.namespace_id = namespace
        self.app_id = app_id
        self.record_tool_results = record_tool_results
        self.recall_on_start = recall_on_start
        self.recall_top_k = recall_top_k

    # ------------------------------------------------------------------
    # Stub hook implementations the SDK Runner duck-calls on us. Methods
    # we don't implement are inherited from RunHooks via ``as_runhooks``.
    # ------------------------------------------------------------------

    async def on_agent_start(
        self,
        context: RunContextWrapper,
        agent: Agent,
    ) -> None:
        """Optionally surface relevant past memories before the agent runs.

        Default behaviour: when ``recall_on_start=False`` (the default),
        does nothing. When True, recalls top-K matches against the
        current input text and logs them at info level.
        """
        if not self.recall_on_start:
            return
        query = _extract_recent_text(context)
        if not query:
            return
        try:
            result = await self.kb.recall(
                query,
                namespace=self.namespace_id,
                limit=self.recall_top_k,
            )
        except Exception as exc:  # noqa: BLE001 — observability hook, never fatal
            logger.warning("KhoraMemoryHooks.on_agent_start recall failed: {}", exc)
            return
        if not result.chunks:
            return
        logger.info(
            "KhoraMemoryHooks: {} memory hit(s) for agent {}",
            len(result.chunks),
            getattr(agent, "name", "?"),
        )

    async def on_tool_end(
        self,
        context: RunContextWrapper,
        agent: Agent,
        tool: Tool,
        result: str,
    ) -> None:
        """Persist a successful tool result to khora.

        The result string is stored as a khora document with metadata
        identifying the tool name, agent name, and run id (when the
        ``ToolContext`` carries one). Extraction is disabled — tool
        outputs are usually JSON / structured text where entity
        extraction would be noise.
        """
        if not self.record_tool_results:
            return
        if not isinstance(result, str) or not result:
            return
        tool_name = getattr(tool, "name", "tool")
        agent_name = getattr(agent, "name", "agent")
        tool_call_id = getattr(context, "tool_call_id", None) or getattr(context, "tool_name", None)

        metadata: dict[str, Any] = {
            "oai_app_id": self.app_id,
            "oai_tool_name": tool_name,
            "oai_agent_name": agent_name,
            "oai_tool_call_id": tool_call_id,
        }
        try:
            await self.kb.remember(
                result,
                namespace=self.namespace_id,
                title=f"oai_tool:{tool_name}",
                source=f"openai_agents:{self.app_id}",
                metadata=metadata,
                entity_types=[],
                relationship_types=[],
            )
        except Exception as exc:  # noqa: BLE001 — observability hook, never fatal
            logger.warning("KhoraMemoryHooks.on_tool_end remember failed: {}", exc)

    # ------------------------------------------------------------------
    # SDK glue: produce a real RunHooks subclass instance for callers
    # that need `isinstance(hooks, RunHooks)` to pass.
    # ------------------------------------------------------------------

    def as_runhooks(self) -> RunHooks:
        """Return a ``RunHooks`` subclass instance forwarding to this object.

        The SDK's ``Runner`` duck-types the hook surface (it only invokes
        methods, never ``isinstance``), so passing a bare
        :class:`KhoraMemoryHooks` works in practice. Use this helper when
        a strict static checker or a downstream wrapper insists on a real
        ``RunHooks`` subclass.
        """
        try:
            from agents.lifecycle import RunHooks  # noqa: PLC0415 — lazy
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "KhoraMemoryHooks.as_runhooks() requires the optional `openai-agents` extra. "
                "Install with: pip install 'khora[openai-agents]'"
            ) from exc

        global _RunHooksAdapter
        if _RunHooksAdapter is None:

            class _KhoraRunHooks(RunHooks):
                """RunHooks subclass that delegates to a KhoraMemoryHooks instance."""

                def __init__(self, owner: KhoraMemoryHooks) -> None:
                    self._owner = owner

                async def on_agent_start(self, context: Any, agent: Any) -> None:
                    await self._owner.on_agent_start(context, agent)

                async def on_tool_end(self, context: Any, agent: Any, tool: Any, result: str) -> None:
                    await self._owner.on_tool_end(context, agent, tool, result)

            _RunHooksAdapter = _KhoraRunHooks

        return _RunHooksAdapter(self)


def _extract_recent_text(context: Any) -> str:
    """Best-effort extraction of the most recent user input from a context.

    The SDK exposes the current run's input items via
    ``context.input`` (RunContextWrapper) but the exact shape varies
    across SDK minors. We probe a few common attribute paths and fall
    back to ``""``; the caller treats empty as "skip recall".
    """
    # Older minors expose ``context.input`` as a string or a list of
    # ``TResponseInputItem`` dicts. Newer minors thread it through
    # ``context.run.input`` instead.
    candidates: list[Any] = []
    for attr in ("input", "user_input", "messages"):
        value = getattr(context, attr, None)
        if value is not None:
            candidates.append(value)
    run = getattr(context, "run", None)
    if run is not None:
        for attr in ("input", "messages"):
            value = getattr(run, attr, None)
            if value is not None:
                candidates.append(value)

    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate
        if isinstance(candidate, list):
            for item in reversed(candidate):
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if isinstance(content, str) and content.strip():
                    return content
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict):
                            text = part.get("text")
                            if isinstance(text, str) and text.strip():
                                return text
    return ""


__all__ = ["KhoraMemoryHooks"]
