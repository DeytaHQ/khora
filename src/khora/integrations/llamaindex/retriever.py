"""``KhoraRetriever`` — LlamaIndex ``BaseRetriever`` over khora recall.

End-user surface: a single class that plugs khora into any LlamaIndex
``QueryEngine`` / agent that takes a retriever. The implementation is
async-only (``_aretrieve``); the sync ``_retrieve`` raises
``NotImplementedError`` with a clear message pointing the caller at
``aretrieve``. That is a deliberate design choice from issue #627:
khora's recall is async and bridging it through a thread-pool inside a
running event loop is the dominant deadlock surface for retriever
adapters. We refuse to ship the foot-gun.

Module-load discipline: no top-level ``import llama_index``. The
framework base class is resolved at first instantiation via a dynamic
``type(...)`` subclass so ``isinstance(r, BaseRetriever)`` still passes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from khora.integrations.llamaindex._mapping import (
    chunk_to_node_with_score,
    entity_to_node_with_score,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from llama_index.core.base.base_retriever import BaseRetriever
    from llama_index.core.schema import NodeWithScore, QueryBundle

    from khora.khora import Khora


# Cached dynamic subclass — resolved on first instantiation so the
# framework import stays out of module load.
_KhoraRetrieverBase: type | None = None


def _import_base_retriever() -> type[BaseRetriever]:
    """Lazy import of ``llama_index.core.base.base_retriever.BaseRetriever``."""
    try:
        from llama_index.core.base.base_retriever import (  # noqa: PLC0415
            BaseRetriever as _Base,
        )
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ImportError(
            "KhoraRetriever requires the optional `llamaindex` extra. Install with: pip install 'khora[llamaindex]'"
        ) from exc
    return _Base


def _get_base_class() -> type:
    """Return (and cache) the dynamically-resolved ``BaseRetriever`` class."""
    global _KhoraRetrieverBase
    if _KhoraRetrieverBase is None:
        _KhoraRetrieverBase = _import_base_retriever()
    return _KhoraRetrieverBase


class KhoraRetriever:
    """LlamaIndex retriever backed by ``Khora.recall``.

    Args:
        kb: A connected :class:`khora.Khora` instance. The retriever does
            NOT own the connection lifecycle — the caller does.
        namespace_id: Stable khora namespace UUID this retriever reads
            from.
        similarity_top_k: Number of chunks (+ optionally entities) to
            request from khora per ``aretrieve`` call. Default ``10``.
        include_entities: When ``True``, entity hits from the same recall
            are returned alongside chunk hits as additional
            ``NodeWithScore`` results. Default ``False`` per the issue
            acceptance criteria (entities-default-off).
        recall_kwargs: Optional dict of extra kwargs forwarded to
            ``Khora.recall`` (e.g. ``{"mode": SearchMode.HYBRID,
            "min_similarity": 0.2}``). The adapter does not vet keys —
            khora itself raises on unknowns, which is the right error
            surface.

    Stability: experimental. The class name and ``_aretrieve`` /
    ``aretrieve`` contract are public; metadata field names on returned
    ``NodeWithScore``s may evolve until v0.14 ships a stable adapter.

    Reentrancy note:
        The sync surface (``_retrieve`` / ``retrieve``) intentionally
        raises ``NotImplementedError``. LlamaIndex's own ``aretrieve``
        path on ``BaseRetriever`` is fully async-native — use it instead.
    """

    name: str = "llamaindex"
    """Identifier for ``khora.integrations`` registry / telemetry."""

    def __init__(
        self,
        kb: Khora,
        *,
        namespace_id: UUID,
        similarity_top_k: int = 10,
        include_entities: bool = False,
        recall_kwargs: dict[str, Any] | None = None,
    ) -> None:
        # Resolve the LlamaIndex base class once and rebind self.__class__
        # to a subclass that has it in the MRO. Same trick used by
        # KhoraStore in the langgraph adapter — keeps the framework
        # import out of module load while still passing
        # ``isinstance(r, BaseRetriever)``.
        base_cls = _get_base_class()
        if base_cls not in type(self).__mro__:
            new_cls = type(
                "KhoraRetriever",
                (KhoraRetriever, base_cls),
                {},
            )
            self.__class__ = new_cls

        if similarity_top_k <= 0:
            raise ValueError(f"similarity_top_k must be > 0, got {similarity_top_k}")

        self.kb = kb
        self._namespace_id = namespace_id
        self._similarity_top_k = int(similarity_top_k)
        self._include_entities = bool(include_entities)
        self._recall_kwargs = dict(recall_kwargs or {})

        # BaseRetriever.__init__ needs to run so callback_manager /
        # object_map / verbose are populated — the BaseRetriever
        # decorators reach for those attributes on ``retrieve`` /
        # ``aretrieve``. Call it after we've bound our own state.
        base_cls.__init__(self)

    # ------------------------------------------------------------------
    # KhoraIntegration marker Protocol
    # ------------------------------------------------------------------

    @property
    def namespace_id(self) -> UUID:
        """Stable khora namespace UUID this retriever reads from."""
        return self._namespace_id

    # ------------------------------------------------------------------
    # BaseRetriever surface
    # ------------------------------------------------------------------

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Sync retrieve — intentionally not implemented.

        khora.recall is async-native. Bridging it through a thread inside
        a LlamaIndex sync call that itself runs inside an event loop is
        the dominant deadlock surface for this kind of adapter; we
        refuse to ship that foot-gun. Use ``aretrieve`` instead — every
        LlamaIndex ``QueryEngine`` exposes the async path.
        """
        raise NotImplementedError(
            "KhoraRetriever is async-only. Use `await retriever.aretrieve(query)` "
            "or wire the retriever into an async LlamaIndex pipeline. The sync "
            "path is intentionally not implemented to avoid deadlock when "
            "bridging async khora.recall through a running event loop."
        )

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Async retrieve — the only supported entry point.

        Delegates to ``Khora.recall`` and converts the resulting
        ``(Chunk, score)`` / ``(Entity, score)`` tuples into LlamaIndex
        ``NodeWithScore`` objects via the mapping helpers.
        """
        # ``recall`` accepts a ``limit`` for chunks; we always pass our
        # configured similarity_top_k. Recall-level filtering (mode,
        # min_similarity, etc.) is forwarded verbatim.
        kwargs = {
            "namespace": self._namespace_id,
            "limit": self._similarity_top_k,
            **self._recall_kwargs,
        }
        result = await self.kb.recall(query_bundle.query_str, **kwargs)

        signals = None
        if result.metadata:
            signals = result.metadata.get("abstention_signals")

        nodes: list[NodeWithScore] = []
        for chunk, score in result.chunks:
            nodes.append(
                chunk_to_node_with_score(
                    chunk,
                    score,
                    abstention_signals=signals,
                )
            )
        if self._include_entities:
            for entity, score in result.entities:
                nodes.append(
                    entity_to_node_with_score(
                        entity,
                        score,
                        abstention_signals=signals,
                    )
                )
        return nodes


__all__ = ["KhoraRetriever"]
