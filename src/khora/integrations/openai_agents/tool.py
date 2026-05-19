"""``khora_recall_tool`` — factory for a function-tool wired into khora.

Returns a ``FunctionTool`` an :class:`agents.Agent` can call to recall
memories from khora at run-time::

    from agents import Agent
    from khora.integrations.openai_agents import khora_recall_tool

    tool = khora_recall_tool(kb=kb, namespace=ns_id, top_k=5)
    agent = Agent(name="researcher", tools=[tool])

The factory closes over ``(kb, namespace, top_k, min_similarity)`` so
the LLM only ever sees the ``query`` parameter. khora's namespace and
recall thresholds stay server-side; one fewer hallucination surface.

Module-load discipline: ``agents.function_tool`` is imported inside the
factory body. The factory is the only entry point — there is no
top-level decorator usage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:  # pragma: no cover - typing only
    from agents.tool import FunctionTool

    from khora.khora import Khora


def khora_recall_tool(
    *,
    kb: Khora,
    namespace: UUID,
    top_k: int = 5,
    min_similarity: float = 0.0,
    name: str = "recall_memory",
    description: str | None = None,
) -> FunctionTool:
    """Build a ``FunctionTool`` that recalls khora memories on demand.

    The returned tool exposes one LLM-visible argument (``query: str``).
    The bound khora instance, namespace, ``top_k``, and similarity
    threshold are captured by closure — the LLM cannot rewrite them.

    Args:
        kb: A connected :class:`khora.Khora` instance.
        namespace: khora namespace UUID this tool reads from.
        top_k: Maximum number of chunks to return. Default 5.
        min_similarity: Cosine similarity floor (0.0 disables). Default 0.
        name: Function name exposed to the LLM. Default ``"recall_memory"``.
        description: Optional override for the tool's description shown to
            the LLM. When ``None``, a generic description is used.

    Returns:
        A ``FunctionTool`` ready to drop into ``Agent(tools=[...])``.

    Raises:
        ImportError: When the ``[openai-agents]`` extra is not installed.
        ValueError: For invalid ``top_k`` / ``min_similarity`` values.
    """
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    if not 0.0 <= min_similarity <= 1.0:
        raise ValueError(f"min_similarity must be in [0.0, 1.0], got {min_similarity}")

    try:
        from agents import function_tool  # noqa: PLC0415 — lazy
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ImportError(
            "khora_recall_tool requires the optional `openai-agents` extra. "
            "Install with: pip install 'khora[openai-agents]'"
        ) from exc

    effective_description = description or (
        "Search the agent's long-term memory for context relevant to a query. "
        "Returns the top matching memory chunks with their similarity scores."
    )

    # ``function_tool`` derives the tool name from the wrapped function's
    # ``__name__``. We define the closure inline so each factory call
    # produces an independent tool with the user's chosen name.
    async def _recall_impl(query: str) -> str:
        result = await kb.recall(
            query,
            namespace=namespace,
            limit=top_k,
            min_similarity=min_similarity,
        )
        if not result.chunks:
            return "no relevant memories found"
        lines: list[str] = []
        for idx, chunk in enumerate(result.chunks, start=1):
            lines.append(f"[{idx}] score={chunk.score:.3f} :: {chunk.content}")
        return "\n".join(lines)

    _recall_impl.__name__ = name
    _recall_impl.__doc__ = effective_description

    return function_tool(_recall_impl, name_override=name, description_override=effective_description)


__all__ = ["khora_recall_tool"]
