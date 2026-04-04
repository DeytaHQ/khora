"""Perplexity API client for datasource discovery.

Uses the Perplexity Sonar API (OpenAI-compatible chat completions) with
``return_citations=True`` to discover relevant datasources.  Direct httpx
calls are used instead of LiteLLM because LiteLLM strips the citations
array from the response.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import httpx
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"

# System prompt steers Perplexity toward structured, downloadable sources
_DISCOVERY_SYSTEM_PROMPT = (
    "You are a data source discovery assistant. The user needs "
    "machine-readable datasets, APIs, or structured data files for "
    "building a knowledge graph.\n"
    "For each source you mention, explain: what format the data is in, "
    "how to access it (direct download URL, API endpoint, or scraping "
    "required), and roughly how much data it contains.\n"
    "Prioritize: 1) official APIs with free tiers, 2) downloadable "
    "CSV/JSON/Parquet files, 3) structured web pages that can be scraped, "
    "4) PDF reports as last resort.\n"
    "Always include specific URLs, not just service names."
)


@dataclass(slots=True)
class PerplexitySearchResponse:
    """Response from a Perplexity search query."""

    answer: str = ""
    citations: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": self.citations,
            "usage": self.usage,
        }


class PerplexityClient:
    """Async client for the Perplexity Sonar search API.

    Discovers potential datasources by querying Perplexity with
    ``return_citations=True`` to get grounded, referenced answers.

    Usage::

        async with PerplexityClient() as client:
            response = await client.search("European wine datasets CSV download")
            for url in response.citations:
                print(url)
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "sonar-pro",
        timeout: float = 45.0,
    ) -> None:
        self._api_key = api_key or os.environ.get("PERPLEXITY_API_KEY", "")
        self._model = model
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    @property
    def available(self) -> bool:
        """Whether the Perplexity API key is configured."""
        return bool(self._api_key)

    async def __aenter__(self) -> PerplexityClient:
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(self._timeout))
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    @retry(
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError, httpx.ReadTimeout)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def search(
        self,
        query: str,
        *,
        domain_hint: str = "",
        system_prompt: str | None = None,
    ) -> PerplexitySearchResponse:
        """Search Perplexity for datasource information.

        Args:
            query: Natural language query about data sources.
            domain_hint: Optional domain context for better results.
            system_prompt: Override the default system prompt.

        Returns:
            PerplexitySearchResponse with answer text and citation URLs.

        Raises:
            httpx.HTTPStatusError: On API error after retries.
            RuntimeError: If the client is not connected.
        """
        if self._client is None:
            raise RuntimeError("Client not connected. Use `async with` context manager.")

        system = system_prompt or _DISCOVERY_SYSTEM_PROMPT
        if domain_hint:
            system += f"\nThe user is building a knowledge graph for the '{domain_hint}' domain."

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": query},
            ],
            "max_tokens": 2048,
            "temperature": 0.1,
            "return_citations": True,
            "search_recency_filter": "year",
        }

        logger.debug(f"Perplexity search: {query[:80]}...")
        response = await self._client.post(
            PERPLEXITY_API_URL,
            json=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()
        data = response.json()

        answer = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        citations = data.get("citations", [])
        usage = data.get("usage", {})

        logger.debug(f"Perplexity returned {len(citations)} citation(s), {usage.get('total_tokens', '?')} tokens")

        return PerplexitySearchResponse(
            answer=answer,
            citations=citations,
            usage={
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            },
        )
