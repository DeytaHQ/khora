"""Reranking module for Khora.

Provides neural re-ranking of search results using:
- Cross-encoder models (sentence-transformers)
- LLM-based relevance scoring
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from loguru import logger

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


def _date_prefix_for(metadata: Any, custom: Any) -> str:
    """Return an ``YYYY-MM-DD`` date prefix for the reranker or ``""``.

    Source priority (matches the canonical ConnectorMetadata mapping from
    #568): ``custom.occurred_at`` → ``custom.sent_at`` → ``metadata.created_at``.
    Accepts datetimes, date objects, or ISO-8601 strings; anything else
    returns ``""`` so the reranker falls back to the un-prefixed content.

    Used by :class:`CrossEncoderReranker` when
    ``include_date_prefix=True`` (Issue #594, Phase D5).
    """
    from datetime import date, datetime

    def _stringify(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, datetime):
            return value.strftime("%Y-%m-%d")
        if isinstance(value, date):
            return value.isoformat()
        if isinstance(value, str):
            # ``YYYY-MM-DD`` prefix of ``YYYY-MM-DDTHH:MM:SS...`` is correct
            # for any ISO-8601 timestamp without parsing overhead.
            return value[:10] if len(value) >= 10 and value[4] == "-" and value[7] == "-" else ""
        return ""

    if isinstance(custom, dict):
        for key in ("occurred_at", "sent_at"):
            text = _stringify(custom.get(key))
            if text:
                return text
    created_at = None
    if hasattr(metadata, "created_at"):
        created_at = metadata.created_at
    elif isinstance(metadata, dict):
        created_at = metadata.get("created_at")
    return _stringify(created_at)


class Reranker(ABC):
    """Abstract base class for rerankers."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate[T]],
        top_k: int = 10,
        blend_weight: float = 0.7,
    ) -> list[RerankResult[T]]:
        """Rerank candidates based on relevance to query.

        Args:
            query: Query text
            candidates: Candidates to rerank
            top_k: Number of results to return
            blend_weight: Weight for rerank score vs original score (0-1).
                          Final score = blend_weight * rerank + (1 - blend_weight) * original.

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
        include_date_prefix: bool = False,
    ) -> None:
        """Initialize the cross-encoder reranker.

        Args:
            model_name: Cross-encoder model name from sentence-transformers
            device: Device to use (cuda, cpu, or None for auto)
            batch_size: Batch size for scoring
            include_date_prefix: When True, prepend ``[YYYY-MM-DD] `` to each
                candidate's content using the source timestamp from
                ``metadata.custom`` (priority: ``occurred_at`` →
                ``sent_at`` → ``created_at``). Off-the-shelf cross-encoders
                tokenize ISO dates fine — the dozen extra tokens per
                candidate cost is negligible relative to the model's
                forward pass. Default OFF; flip after a positive A/B on
                the corporate-shape benchmark. Issue #594 (Phase D5).
        """
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._include_date_prefix = include_date_prefix
        self._model = None

    def _get_model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                raise RuntimeError("sentence-transformers not installed. Run: pip install sentence-transformers")
            self._model = CrossEncoder(self._model_name, device=self._device)
        return self._model

    async def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate[T]],
        top_k: int = 10,
        blend_weight: float = 0.7,
    ) -> list[RerankResult[T]]:
        """Rerank using cross-encoder scoring.

        Args:
            query: Query text
            candidates: Candidates to rerank
            top_k: Number of results to return
            blend_weight: Weight for rerank score vs original score (0-1)

        Returns:
            Reranked results
        """
        if not candidates:
            return []

        try:
            model = self._get_model()

            # Prepare pairs for cross-encoder
            pairs = []
            for c in candidates:
                doc_title = ""
                custom = getattr(c.metadata, "custom", None) if hasattr(c.metadata, "custom") else None
                if custom is None and isinstance(c.metadata, dict):
                    custom = c.metadata.get("custom")
                if isinstance(custom, dict):
                    doc_title = custom.get("title", "")
                content_with_meta = f"[{doc_title}] {c.content}" if doc_title else c.content
                if self._include_date_prefix:
                    date_prefix = _date_prefix_for(c.metadata, custom)
                    if date_prefix:
                        content_with_meta = f"[{date_prefix}] {content_with_meta}"
                pairs.append((query, content_with_meta))

            # Score in batches
            # Offload synchronous PyTorch inference to a thread to avoid
            # blocking the event loop during cross-encoder scoring.
            scores = await asyncio.to_thread(model.predict, pairs, batch_size=self._batch_size)

            # Convert all scores to floats for normalization
            # Cross-encoders output logits (roughly -3 to 3), we need to normalize
            float_scores = [float(1 / (1 + (-s).exp())) if hasattr(s, "exp") else float(s) for s in scores]

            # QUALITY FIX: Normalize cross-encoder scores to [0,1] using min-max
            # across the batch. This ensures proper combination with original_score
            # (which is already in [0,1] from RRF or similarity scoring).
            if float_scores:
                min_score = min(float_scores)
                max_score = max(float_scores)
                score_range = max_score - min_score
                if score_range > 0:
                    normalized_scores = [(s - min_score) / score_range for s in float_scores]
                else:
                    # All scores are the same - use 0.5 as neutral
                    normalized_scores = [0.5] * len(float_scores)
            else:
                normalized_scores = []

            # Combine with original scores (both now in [0,1])
            results = []
            for candidate, normalized_rerank in zip(candidates, normalized_scores):
                final_score = blend_weight * normalized_rerank + (1 - blend_weight) * candidate.original_score

                results.append(
                    RerankResult(
                        item=candidate.item,
                        original_score=candidate.original_score,
                        rerank_score=normalized_rerank,
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
- 1-3: Topically related but answers a DIFFERENT question or discusses a different entity/event
- 5: Somewhat relevant, mentions related topics but lacks specificity
- 7-8: Relevant, addresses the query with useful information
- 10: Highly relevant, directly answers or addresses the query

IMPORTANT: Be strict about confounders. A passage that shares keywords with the query but discusses a different person, project, event, or time period should score 1-3, NOT 5+. Only score 5+ if the passage genuinely helps answer the specific question asked.

Respond with ONLY a single number (0-10), nothing else."""

LLM_BATCH_RERANK_PROMPT = """You are a relevance scoring system. Given a query and multiple passages, score the relevance of each passage to the query.

Query: {query}

Passages:
{passages}

Score each passage from 0 to 10 where:
- 0: Completely irrelevant
- 1-3: CONFOUNDER — shares keywords or topics with the query but answers a different question, discusses a different entity/person, or refers to a different time period or context
- 5: Somewhat relevant, mentions related topics but lacks specificity to the query
- 7-8: Relevant, addresses the query with useful information from the right context
- 10: Highly relevant, directly answers or addresses the specific query

IMPORTANT confounder detection rules:
- If the query asks about person A but the passage discusses person B (even in the same organization), score 1-3
- If the query asks about a specific project/event but the passage discusses a different one, score 1-3
- If the passage shares keywords but the core subject or context differs from the query, score 1-3
- Only score 5+ when the passage provides information that genuinely helps answer the SPECIFIC question

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
        blend_weight: float = 0.7,
    ) -> list[RerankResult[T]]:
        """Rerank using batched LLM scoring.

        Sends multiple candidates per LLM call to reduce API round-trips.

        Args:
            query: Query text
            candidates: Candidates to rerank
            top_k: Number of results to return
            blend_weight: Weight for rerank score vs original score (0-1)

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
            passage_lines = []
            for i, c in enumerate(batch):
                doc_title = ""
                custom = getattr(c.metadata, "custom", None) if hasattr(c.metadata, "custom") else None
                if custom is None and isinstance(c.metadata, dict):
                    custom = c.metadata.get("custom")
                if isinstance(custom, dict):
                    doc_title = custom.get("title", "")
                prefix = f"[{doc_title}] " if doc_title else ""
                passage_lines.append(f"[{i + 1}] {prefix}{c.content[:500]}")
            passages = "\n".join(passage_lines)
            prompt = LLM_BATCH_RERANK_PROMPT.format(
                query=query,
                passages=passages,
                count=len(batch),
            )

            try:
                response = await acompletion(prompt, config, _telemetry_op="llm_rerank")
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
                    final_score = blend_weight * normalized_score + (1 - blend_weight) * candidate.original_score
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


# Module-level cache for reranker instances.  Cross-encoder models are
# expensive to load (~500ms + GPU memory), so we reuse them across calls.
_reranker_cache: dict[str, Reranker] = {}


def create_reranker(
    method: str = "cross_encoder",
    model: str | None = None,
    llm_config: LiteLLMConfig | None = None,
    *,
    include_date_prefix: bool = False,
) -> Reranker:
    """Create or return a cached reranker based on method.

    Cross-encoder rerankers are cached by ``(model, include_date_prefix)`` so
    the two variants coexist without a model reload (~500ms per load). LLM
    rerankers are lightweight and created fresh each time.

    Args:
        method: Reranking method (cross_encoder, llm)
        model: Model name/path
        llm_config: LLM configuration for LLM reranker
        include_date_prefix: When True (cross-encoder only), prepend an
            ``[YYYY-MM-DD]`` token to each candidate's content. Default
            OFF; see :class:`CrossEncoderReranker` and Issue #594.

    Returns:
        Reranker instance
    """
    if method == "cross_encoder":
        model_name = model or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        cache_key = f"cross_encoder:{model_name}:date_prefix={include_date_prefix}"
        if cache_key not in _reranker_cache:
            _reranker_cache[cache_key] = CrossEncoderReranker(
                model_name=model_name,
                include_date_prefix=include_date_prefix,
            )
        return _reranker_cache[cache_key]
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
            content=f"{entity.name}: {entity.description or ''} ({entity.entity_type})",
            metadata=entity.metadata,
        )
        for entity, score in entities
    ]

    results = await reranker.rerank(query, candidates, top_k)
    return [(r.item, r.final_score) for r in results]


# ---------------------------------------------------------------------------
# LLM Listwise Reranker (cost-aware, cache-backed)
# ---------------------------------------------------------------------------

_LLM_LISTWISE_PROMPT = """Given this memory retrieval query, rank the passages below by relevance.
Consider temporal recency, entity mentions, and conversational context.
For temporal queries ("what does X do now?", "latest", "current"), strongly prefer the most recent information.

Query: {query}

Passages:
{passages}

Return ONLY a JSON array of passage numbers in order of relevance, most relevant first.
Example: [3, 1, 5, 2, 4]"""


async def llm_listwise_rerank(
    query: str,
    chunks: list[tuple[Chunk, float]],
    *,
    model: str = "gpt-4o-mini",
    top_n: int = 10,
    confidence_threshold: float = 0.1,
) -> list[tuple[Chunk, float]]:
    """LLM-based listwise reranking with cost-aware triggering.

    Only activates when the score gap between rank 1 and rank 2 is below
    ``confidence_threshold`` (uncertain ranking). Falls back to original
    ordering on failure.
    """
    import hashlib
    import json
    import pathlib

    from khora.config.llm import LiteLLMConfig, acompletion

    if len(chunks) < 2:
        return chunks

    # Cost guard: only rerank when ranking is uncertain
    gap = chunks[0][1] - chunks[1][1] if len(chunks) >= 2 else 1.0
    if gap >= confidence_threshold:
        return chunks

    to_rerank = chunks[:top_n]
    remainder = chunks[top_n:]

    # Check disk cache
    cache_dir = pathlib.Path.home() / ".cache" / "khora" / "llm_reranker"
    cache_dir.mkdir(parents=True, exist_ok=True)
    content_hash = hashlib.sha256(
        (query + "||" + "||".join(c.content[:200] for c, _ in to_rerank)).encode()
    ).hexdigest()[:16]
    cache_file = cache_dir / f"{content_hash}.json"

    if cache_file.exists():
        try:
            order = json.loads(cache_file.read_text())
            reordered: list[tuple[Chunk, float]] = []
            for idx in order:
                if 0 <= idx < len(to_rerank):
                    chunk, score = to_rerank[idx]
                    reordered.append((chunk, score + (len(to_rerank) - len(reordered)) * 0.001))
            for _i, (chunk, score) in enumerate(to_rerank):
                if not any(c.id == chunk.id for c, _ in reordered):
                    reordered.append((chunk, score))
            return reordered + remainder
        except Exception:  # noqa: S110 — cache miss is non-fatal
            pass

    # Build prompt
    passage_lines = []
    for i, (chunk, _score) in enumerate(to_rerank):
        passage_lines.append(f"[{i + 1}] {chunk.content[:400]}")
    passages = "\n".join(passage_lines)
    prompt = _LLM_LISTWISE_PROMPT.format(query=query, passages=passages)

    try:
        config = LiteLLMConfig(model=model, temperature=0.0, max_tokens=100)
        response = await acompletion(prompt, config, _telemetry_op="listwise_rerank")
        order = json.loads(response.strip())
        order = [int(x) - 1 for x in order if isinstance(x, (int, float))]

        try:
            cache_file.write_text(json.dumps(order))
        except Exception:  # noqa: S110 — cache write failure is non-fatal
            pass

        reordered = []
        for idx in order:
            if 0 <= idx < len(to_rerank):
                chunk, score = to_rerank[idx]
                reordered.append((chunk, score + (len(to_rerank) - len(reordered)) * 0.001))
        for _i, (chunk, score) in enumerate(to_rerank):
            if not any(c.id == chunk.id for c, _ in reordered):
                reordered.append((chunk, score))

        return reordered + remainder

    except Exception as e:
        logger.debug(f"LLM listwise reranking failed: {e}")
        return chunks
