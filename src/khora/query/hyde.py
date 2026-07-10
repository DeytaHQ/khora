"""Hypothetical Document Embeddings (HyDE) for query expansion.

Generates a hypothetical answer document using an LLM, embeds it,
and averages with the original query embedding to improve recall
for queries that use different vocabulary than stored documents.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from khora.telemetry import metric_counter

if TYPE_CHECKING:
    from khora.config.llm import LiteLLMConfig
    from khora.core.diagnostics import Degradation
    from khora.extraction.embedders import Embedder
    from khora.query.temporal_detection import TemporalCategory


# Temporal-anchored prompt is selected for these categories. The hypothetical
# is written as if authored "today" so its surface tokens — ISO dates,
# weekdays, relative markers — dominate cosine similarity to chunks that
# carry the same tokens (Slack/email headers, calendar invites). See #592.
_TEMPORAL_HYDE_CATEGORIES = {"recency", "state_query", "change"}

# ADR-001 (issue #1324). When ``expand_query_embedding`` catches an exception
# and falls back to the original embedding, a ``Degradation`` is appended to
# the caller-supplied ``out_diagnostics`` list and this counter is incremented.
# NO namespace_id label - cardinality rule.
_HYDE_DEGRADED_COUNTER = metric_counter(
    "khora.query.hyde.degraded_total",
    unit="1",
    description=(
        "Issue #1324 (ADR-001). HyDE query-embedding expansion silent fallback. "
        "Incremented when ``expand_query_embedding`` catches any exception (LLM "
        "call failure, embed failure, etc.) and returns the original embedding "
        "unchanged. The same event is also appended to the caller-supplied "
        "out_diagnostics list, which surfaces on RecallResult.engine_info"
        "['degradations'] or QueryResult.metadata['degradations']. "
        "NO namespace_id label - cardinality rule."
    ),
)


def _detect_category(
    query: str,
    *,
    out_diagnostics: list[Degradation] | None = None,
) -> TemporalCategory | None:
    """Best-effort temporal-category detection for a HyDE query.

    Wraps ``khora.query.temporal_detection.detect_temporal_category`` so
    the import is local (avoids a hard dependency at module import time)
    and any unexpected failure degrades to ``None`` (generic prompt) —
    HyDE should never crash a query because of a detector error. On
    failure, when ``out_diagnostics`` is supplied, a :class:`Degradation`
    is appended so the fallback is observable per ADR-001 (issue #1324).
    """
    try:
        from khora._accel import detect_temporal_category
        from khora.query.temporal_detection import CATEGORY_MAP

        cat_id = detect_temporal_category(query)
        return CATEGORY_MAP.get(cat_id)
    except Exception as exc:  # noqa: BLE001 — degrade to generic prompt on any failure
        logger.debug(
            "HyDE temporal-category detection failed, using generic prompt: {} {}",
            type(exc).__name__,
            exc,
        )
        if out_diagnostics is not None:
            from khora.core.diagnostics import Degradation

            out_diagnostics.append(
                Degradation(
                    component="query.hyde",
                    reason="temporal_category_detection_failed",
                    detail="HyDE temporal-category detection failed; using generic prompt.",
                    exception=type(exc).__name__,
                )
            )
        return None


def _select_system_prompt(category: str | None, today: str) -> str:
    """Pick the HyDE system prompt for a temporal category.

    ``category`` is the string value of a :class:`TemporalCategory`. When
    ``None`` (no detector available) or one of NONE/EXPLICIT/ORDINAL/
    AGGREGATE, returns the generic time-blind prompt. For RECENCY,
    STATE_QUERY, and CHANGE, returns a prompt that anchors the
    hypothetical to ``today``.
    """
    if category in _TEMPORAL_HYDE_CATEGORIES:
        return (
            "You are a knowledgeable assistant. Given a question that asks about "
            "recent, current, or recently-changed information, write a short, "
            "factual passage (2-3 sentences) that directly answers it as if it "
            f"were authored today, {today}. Use specific dates, weekdays, or "
            "relative time markers (e.g. 'yesterday', 'this week', 'on "
            f"{today}'). Do not include any preamble or meta-commentary, just "
            "the answer content."
        )
    return (
        "You are a knowledgeable assistant. Given a question, write a short, "
        "factual passage (2-3 sentences) that directly answers it. "
        "Do not include any preamble or meta-commentary, just the answer content."
    )


class HyDEExpander:
    """Expands query embeddings using Hypothetical Document Embeddings."""

    def __init__(
        self,
        embedder: Embedder,
        llm_config: LiteLLMConfig | None = None,
        *,
        num_hypotheticals: int = 1,
    ) -> None:
        self._embedder = embedder
        self._llm_config = llm_config
        self._num_hypotheticals = num_hypotheticals

    async def generate_hypothetical(
        self,
        query: str,
        *,
        temporal_category: TemporalCategory | None = None,
        today: str | None = None,
    ) -> str:
        """Generate a hypothetical document that answers the query.

        When ``temporal_category`` is RECENCY / STATE_QUERY / CHANGE, the
        prompt is anchored to ``today`` (defaults to the current UTC date)
        so the hypothetical's surface tokens align with recency-tagged
        chunks. Other categories use the time-blind prompt.
        """
        from khora.config.llm import acompletion

        if today is None:
            today = datetime.now(UTC).strftime("%Y-%m-%d")
        category_value = temporal_category.value if temporal_category is not None else None
        system_prompt = _select_system_prompt(category_value, today)

        return await acompletion(
            query,
            self._llm_config,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=200,
            _telemetry_op="hyde",
        )

    async def generate_hypothetical_embeddings(
        self,
        query: str,
        *,
        temporal_category: TemporalCategory | None = None,
        out_diagnostics: list[Degradation] | None = None,
    ) -> list[list[float]] | None:
        """Generate + embed the hypothetical documents for ``query`` (#1469).

        This is the LLM-and-embed half of :meth:`expand_query_embedding`. It
        depends only on the query text and temporal category - NOT on the base
        query embedding - so a caller can launch it as a task at t0 and overlap
        the (expensive) LLM round-trip with the base query embed and the rest of
        the front-matter, then average the results in with
        :meth:`combine_with_query_embedding` once both are ready.

        Returns the list of hypothetical-document embeddings, or ``None`` on any
        failure (recording a :class:`Degradation` on ``out_diagnostics`` and
        incrementing the degraded counter, ADR-001) so the caller can fall back
        to the un-expanded embedding.
        """
        from khora.core.diagnostics import Degradation
        from khora.telemetry.instrument import pipeline_stage

        if temporal_category is None:
            temporal_category = _detect_category(query, out_diagnostics=out_diagnostics)

        try:
            async with pipeline_stage("query", "hyde", extra_metadata={"num_hypotheticals": self._num_hypotheticals}):
                hyde_embeddings: list[list[float]] = []
                for _ in range(self._num_hypotheticals):
                    doc = await self.generate_hypothetical(query, temporal_category=temporal_category)
                    hyde_embeddings.append(await self._embedder.embed(doc))
                return hyde_embeddings
        except Exception as e:
            logger.warning("HyDE expansion failed, using original embedding: {}", e, exc_info=True)
            _HYDE_DEGRADED_COUNTER.add(1, attributes={"channel": "query_embedding", "reason": "hyde_embedding_failed"})
            if out_diagnostics is not None:
                out_diagnostics.append(
                    Degradation(
                        component="query.hyde",
                        reason="hyde_embedding_failed",
                        detail=str(e)[:200] or None,
                        exception=type(e).__name__,
                    )
                )
            return None

    @staticmethod
    def combine_with_query_embedding(
        query_embedding: list[float],
        hyde_embeddings: list[list[float]],
    ) -> list[float]:
        """Average the base query embedding with the hypothetical embeddings."""
        avg = np.mean([query_embedding, *hyde_embeddings], axis=0)
        return avg.tolist()

    async def expand_query_embedding(
        self,
        query: str,
        query_embedding: list[float],
        *,
        temporal_category: TemporalCategory | None = None,
        out_diagnostics: list[Degradation] | None = None,
    ) -> list[float]:
        """Expand a query embedding by averaging with hypothetical doc embeddings.

        When ``temporal_category`` is set, it is forwarded to the prompt
        selector so RECENCY / STATE_QUERY / CHANGE queries get a
        time-anchored hypothetical (#592). When ``None``, the category is
        derived internally from the query via the Rust Aho-Corasick
        ``detect_temporal_category`` (sub-millisecond, no LLM cost).

        On any failure, returns the original embedding unchanged and - when
        ``out_diagnostics`` is supplied - appends a :class:`Degradation` so
        the silent fallback is observable on
        ``RecallResult.engine_info['degradations']`` (ADR-001, issue #1324).
        """
        hyde_embeddings = await self.generate_hypothetical_embeddings(
            query,
            temporal_category=temporal_category,
            out_diagnostics=out_diagnostics,
        )
        if not hyde_embeddings:
            return query_embedding
        return self.combine_with_query_embedding(query_embedding, hyde_embeddings)
