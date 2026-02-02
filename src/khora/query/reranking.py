"""Reranking module for Khora Memory Lake.

Provides neural re-ranking of search results using:
- Cross-encoder models (sentence-transformers)
- LLM-based relevance scoring
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from loguru import logger

from khora.core.models.entity import entity_type_str

if TYPE_CHECKING:
    from khora.config.llm import LiteLLMConfig
    from khora.core.models import Chunk, Entity

T = TypeVar("T")


@dataclass
class RerankCandidate(Generic[T]):
    """A candidate for reranking."""

    item: T
    original_score: float
    content: str  # Text content for reranking
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RerankResult(Generic[T]):
    """Result of reranking."""

    item: T
    original_score: float
    rerank_score: float
    final_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class Reranker(ABC):
    """Abstract base class for rerankers."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate[T]],
        top_k: int = 10,
    ) -> list[RerankResult[T]]:
        """Rerank candidates based on relevance to query.

        Args:
            query: Query text
            candidates: Candidates to rerank
            top_k: Number of results to return

        Returns:
            List of RerankResult sorted by final_score descending
        """
        pass


class CrossEncoderReranker(Reranker):
    """Reranker using cross-encoder models.

    Uses sentence-transformers cross-encoder models for
    high-quality relevance scoring.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        """Initialize the cross-encoder reranker.

        Args:
            model_name: Cross-encoder model name from sentence-transformers
            device: Device to use (cuda, cpu, or None for auto)
            batch_size: Batch size for scoring
        """
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._model = None

    def _get_model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                raise RuntimeError("sentence-transformers not installed. " "Run: pip install sentence-transformers")
            self._model = CrossEncoder(self._model_name, device=self._device)
        return self._model

    async def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate[T]],
        top_k: int = 10,
    ) -> list[RerankResult[T]]:
        """Rerank using cross-encoder scoring.

        Args:
            query: Query text
            candidates: Candidates to rerank
            top_k: Number of results to return

        Returns:
            Reranked results
        """
        if not candidates:
            return []

        try:
            model = self._get_model()

            # Prepare pairs for cross-encoder
            pairs = [(query, c.content) for c in candidates]

            # Score in batches
            scores = model.predict(pairs, batch_size=self._batch_size)

            # Combine with original scores
            results = []
            for candidate, rerank_score in zip(candidates, scores):
                # Normalize rerank score to 0-1
                normalized_score = (
                    float(1 / (1 + (-rerank_score).exp())) if hasattr(rerank_score, "exp") else float(rerank_score)
                )

                # Combine scores (weighted average)
                final_score = 0.7 * normalized_score + 0.3 * candidate.original_score

                results.append(
                    RerankResult(
                        item=candidate.item,
                        original_score=candidate.original_score,
                        rerank_score=normalized_score,
                        final_score=final_score,
                        metadata=candidate.metadata,
                    )
                )

            # Sort by final score
            results.sort(key=lambda r: r.final_score, reverse=True)
            return results[:top_k]

        except Exception as e:
            logger.warning(f"Cross-encoder reranking failed: {e}")
            # Fall back to original ranking
            return [
                RerankResult(
                    item=c.item,
                    original_score=c.original_score,
                    rerank_score=c.original_score,
                    final_score=c.original_score,
                    metadata=c.metadata,
                )
                for c in sorted(candidates, key=lambda x: x.original_score, reverse=True)[:top_k]
            ]


LLM_RERANK_PROMPT = """You are a relevance scoring system. Given a query and a document, score the relevance of the document to the query.

Query: {query}

Document:
{document}

Score the relevance from 0 to 10 where:
- 0: Completely irrelevant
- 5: Somewhat relevant, mentions related topics
- 10: Highly relevant, directly answers or addresses the query

Respond with ONLY a single number (0-10), nothing else."""

LLM_BATCH_RERANK_PROMPT = """You are a relevance scoring system. Given a query and multiple passages, score the relevance of each passage to the query.

Query: {query}

Passages:
{passages}

Score each passage from 0 to 10 where:
- 0: Completely irrelevant
- 5: Somewhat relevant, mentions related topics
- 10: Highly relevant, directly answers or addresses the query

Respond with ONLY a JSON object: {{"scores": [score1, score2, ...]}}
The scores array must have exactly {count} numbers, one per passage in order."""


class LLMReranker(Reranker):
    """Reranker using LLM-based relevance scoring.

    Uses an LLM to score relevance, which can provide
    better understanding of complex queries but is slower
    and more expensive than cross-encoders.
    """

    def __init__(
        self,
        llm_config: LiteLLMConfig | None = None,
        model: str | None = None,
        batch_size: int = 10,
    ) -> None:
        """Initialize the LLM reranker.

        Args:
            llm_config: LiteLLM configuration
            model: Optional model override
            batch_size: Concurrent LLM calls
        """
        self._llm_config = llm_config
        self._model = model
        self._batch_size = batch_size

    async def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate[T]],
        top_k: int = 10,
    ) -> list[RerankResult[T]]:
        """Rerank using batched LLM scoring.

        Sends multiple candidates per LLM call to reduce API round-trips.

        Args:
            query: Query text
            candidates: Candidates to rerank
            top_k: Number of results to return

        Returns:
            Reranked results
        """
        import asyncio
        import json

        from khora.config.llm import LiteLLMConfig, acompletion

        if not candidates:
            return []

        config = self._llm_config or LiteLLMConfig()
        config = LiteLLMConfig(
            model=self._model or config.model,
            temperature=0.0,
            max_tokens=200,  # Enough for JSON scores array
        )

        async def score_batch(batch: list[RerankCandidate[T]]) -> list[float]:
            """Score a batch of candidates in a single LLM call."""
            passages = "\n".join(f"[{i + 1}] {c.content[:500]}" for i, c in enumerate(batch))
            prompt = LLM_BATCH_RERANK_PROMPT.format(
                query=query,
                passages=passages,
                count=len(batch),
            )

            try:
                response = await acompletion(prompt, config)
                data = json.loads(response.strip())
                scores = [float(s) for s in data["scores"]]
                # Clamp and pad/truncate to match batch size
                scores = [max(0.0, min(10.0, s)) for s in scores]
                while len(scores) < len(batch):
                    scores.append(5.0)
                return scores[: len(batch)]
            except Exception as e:
                logger.debug(f"Batch LLM scoring failed: {e}")
                return [5.0] * len(batch)

        # Split candidates into batches
        batches = [candidates[i : i + self._batch_size] for i in range(0, len(candidates), self._batch_size)]

        try:
            batch_scores = await asyncio.gather(*[score_batch(b) for b in batches])

            # Flatten and build results
            results = []
            for batch, scores in zip(batches, batch_scores):
                for candidate, raw_score in zip(batch, scores):
                    normalized_score = raw_score / 10.0
                    final_score = 0.7 * normalized_score + 0.3 * candidate.original_score
                    results.append(
                        RerankResult(
                            item=candidate.item,
                            original_score=candidate.original_score,
                            rerank_score=normalized_score,
                            final_score=final_score,
                            metadata=candidate.metadata,
                        )
                    )

            results.sort(key=lambda r: r.final_score, reverse=True)
            return results[:top_k]

        except Exception as e:
            logger.warning(f"LLM reranking failed: {e}")
            return [
                RerankResult(
                    item=c.item,
                    original_score=c.original_score,
                    rerank_score=c.original_score,
                    final_score=c.original_score,
                    metadata=c.metadata,
                )
                for c in sorted(candidates, key=lambda x: x.original_score, reverse=True)[:top_k]
            ]


def create_reranker(
    method: str = "cross_encoder",
    model: str | None = None,
    llm_config: LiteLLMConfig | None = None,
) -> Reranker:
    """Create a reranker based on method.

    Args:
        method: Reranking method (cross_encoder, llm)
        model: Model name/path
        llm_config: LLM configuration for LLM reranker

    Returns:
        Reranker instance
    """
    if method == "cross_encoder":
        model_name = model or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        return CrossEncoderReranker(model_name=model_name)
    elif method == "llm":
        return LLMReranker(llm_config=llm_config, model=model)
    else:
        raise ValueError(f"Unknown reranking method: {method}")


async def rerank_chunks(
    query: str,
    chunks: list[tuple[Chunk, float]],
    method: str = "cross_encoder",
    top_k: int = 10,
    model: str | None = None,
    llm_config: LiteLLMConfig | None = None,
) -> list[tuple[Chunk, float]]:
    """Convenience function to rerank chunks.

    Args:
        query: Query text
        chunks: List of (chunk, score) tuples
        method: Reranking method
        top_k: Number of results
        model: Optional model override
        llm_config: LLM config for LLM reranker

    Returns:
        Reranked list of (chunk, score) tuples
    """
    if not chunks:
        return []

    reranker = create_reranker(method, model, llm_config)

    candidates = [
        RerankCandidate(
            item=chunk,
            original_score=score,
            content=chunk.content,
            metadata=chunk.metadata,
        )
        for chunk, score in chunks
    ]

    results = await reranker.rerank(query, candidates, top_k)
    return [(r.item, r.final_score) for r in results]


async def rerank_entities(
    query: str,
    entities: list[tuple[Entity, float]],
    method: str = "cross_encoder",
    top_k: int = 10,
    model: str | None = None,
    llm_config: LiteLLMConfig | None = None,
) -> list[tuple[Entity, float]]:
    """Convenience function to rerank entities.

    Args:
        query: Query text
        entities: List of (entity, score) tuples
        method: Reranking method
        top_k: Number of results
        model: Optional model override
        llm_config: LLM config for LLM reranker

    Returns:
        Reranked list of (entity, score) tuples
    """
    if not entities:
        return []

    reranker = create_reranker(method, model, llm_config)

    candidates = [
        RerankCandidate(
            item=entity,
            original_score=score,
            content=f"{entity.name}: {entity.description or ''} ({entity_type_str(entity.entity_type)})",
            metadata=entity.metadata,
        )
        for entity, score in entities
    ]

    results = await reranker.rerank(query, candidates, top_k)
    return [(r.item, r.final_score) for r in results]
