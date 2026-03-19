"""Hypothetical Document Embeddings (HyDE) for query expansion.

Generates a hypothetical answer document using an LLM, embeds it,
and averages with the original query embedding to improve recall
for queries that use different vocabulary than stored documents.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger
import numpy as np

if TYPE_CHECKING:
    from khora.config.llm import LiteLLMConfig
    from khora.extraction.embedders import Embedder


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

    async def generate_hypothetical(self, query: str) -> str:
        """Generate a hypothetical document that answers the query."""
        from khora.config.llm import acompletion

        system_prompt = (
            "You are a knowledgeable assistant. Given a question, write a short, "
            "factual passage (2-3 sentences) that directly answers it. "
            "Do not include any preamble or meta-commentary, just the answer content."
        )

        return await acompletion(
            query,
            self._llm_config,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=200,
        )

    async def expand_query_embedding(
        self,
        query: str,
        query_embedding: list[float],
    ) -> list[float]:
        """Expand a query embedding by averaging with hypothetical doc embeddings.

        On any failure, returns the original embedding unchanged.
        """
        from khora.telemetry.instrument import pipeline_stage

        try:
            async with pipeline_stage("query", "hyde", extra_metadata={"num_hypotheticals": self._num_hypotheticals}):
                hypotheticals: list[str] = []
                for _ in range(self._num_hypotheticals):
                    doc = await self.generate_hypothetical(query)
                    hypotheticals.append(doc)

                # Embed hypothetical documents
                hyde_embeddings: list[list[float]] = []
                for doc in hypotheticals:
                    emb = await self._embedder.embed(doc)
                    hyde_embeddings.append(emb)

                # Average: original query embedding + hypothetical embeddings
                all_embeddings = [query_embedding, *hyde_embeddings]
                avg = np.mean(all_embeddings, axis=0)
                return avg.tolist()

        except Exception as e:
            logger.warning(f"HyDE expansion failed, using original embedding: {e}")
            return query_embedding
