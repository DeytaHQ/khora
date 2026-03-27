"""LLM-backed planner for the discovery agent.

Uses LiteLLM (via OntologyLLM) to make planning decisions:
- Turn user intent into search queries
- Classify and rank discovered sources
- Decide fetch strategy (Firecrawl scrape vs. direct download vs. generated script)
- Generate Python fetch scripts for complex sources
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from loguru import logger

from khora.cli.ontology.llm import OntologyLLM

from .state import DiscoveredSource, SourceType

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_QUERY_FORMULATION_SYSTEM = """\
You are a data source planning assistant.
Given a user's description of what knowledge graph they want to build,
generate 1-3 targeted search queries for finding downloadable datasets,
APIs, or structured data files.

Return JSON:
{
  "domain": "short domain label",
  "description": "1-sentence summary of what data they need",
  "search_queries": [
    "specific query targeting datasets/CSVs",
    "specific query targeting APIs",
    "optional third query for structured web pages"
  ],
  "preferred_formats": ["csv", "json", "api"]
}

Each query should be specific and target different source types."""

_SOURCE_CLASSIFICATION_SYSTEM = """\
Classify each URL into a data source type and rate its usefulness
for building a knowledge graph about the given topic.

Return JSON:
{
  "sources": [
    {
      "url": "the url",
      "title": "short descriptive title",
      "source_type": "webpage|api|csv|json|parquet|pdf|repo|rss|dataset|other",
      "access_method": "direct_download|api_call|scrape|git_clone",
      "relevance": 0.0-1.0,
      "requires_auth": false,
      "description": "1-sentence description of what data this provides"
    }
  ]
}

Rank by: direct downloadable data > free API > scrapable page > PDF.
Give higher relevance to sources with structured, machine-readable data."""

_FETCH_SCRIPT_SYSTEM = """\
Generate a Python script that downloads data from the given source.

STRICT RULES:
1. Only use: httpx, csv, json, pathlib, zipfile, gzip, io, re, datetime
2. Never use: subprocess, os.system, eval, exec, __import__, shutil.rmtree
3. Write all output to the provided output directory
4. Use httpx for HTTP requests with a User-Agent header
5. Handle pagination up to {max_pages} pages with 1-second delays
6. The script must be a complete, runnable Python file
7. Print a JSON summary to stdout: {{"files": [...], "total_records": N}}
8. Include error handling for HTTP errors
9. Never hardcode API keys — read from environment if needed
10. Maximum runtime: script will be killed after {timeout}s

Output ONLY the Python code, no explanation."""


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class QueryPlan:
    """Result of formulating search queries from user intent."""

    domain: str = ""
    description: str = ""
    search_queries: list[str] = field(default_factory=list)
    preferred_formats: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FetchStrategy:
    """How to fetch data from a particular source."""

    method: Literal["firecrawl_scrape", "firecrawl_crawl", "direct_download", "generated_script"]
    params: dict[str, Any] = field(default_factory=dict)
    script: str | None = None


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


class DiscoveryPlanner:
    """LLM-backed planner for the discovery agent.

    Uses gpt-4o-mini (or configured model) via OntologyLLM for all
    planning decisions. Budget is shared with the discovery session.

    Usage::

        planner = DiscoveryPlanner()
        plan = await planner.formulate_queries("European wine datasets")
        ranked = await planner.classify_sources(plan.domain, citations)
    """

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        budget_usd: float = 0.50,
    ) -> None:
        self._llm = OntologyLLM(model=model, budget_usd=budget_usd, interactive=False)

    @property
    def usage_summary(self) -> dict[str, Any]:
        return self._llm.usage_summary

    @property
    def cost_usd(self) -> float:
        return self._llm._total_cost_usd

    async def formulate_queries(
        self,
        user_intent: str,
        *,
        previous_queries: list[str] | None = None,
    ) -> QueryPlan:
        """Turn user intent into 1-3 targeted search queries.

        Args:
            user_intent: Natural language description of needed data.
            previous_queries: Queries already tried (to avoid repetition).

        Returns:
            QueryPlan with domain, description, and search queries.
        """
        user_msg = f"User wants: {user_intent}"
        if previous_queries:
            user_msg += f"\n\nAlready tried these queries (generate different ones): {previous_queries}"

        try:
            result = await self._llm.complete(
                system=_QUERY_FORMULATION_SYSTEM,
                user=user_msg,
                temperature=0.3,
            )
            return QueryPlan(
                domain=result.get("domain", ""),
                description=result.get("description", ""),
                search_queries=result.get("search_queries", [user_intent]),
                preferred_formats=result.get("preferred_formats", []),
            )
        except Exception as e:
            logger.warning(f"Query formulation failed, using intent as-is: {e}")
            return QueryPlan(
                domain="",
                description=user_intent,
                search_queries=[user_intent],
            )

    async def classify_sources(
        self,
        domain: str,
        citations: list[str],
    ) -> list[DiscoveredSource]:
        """Classify and rank a list of URLs by relevance and source type.

        Args:
            domain: The domain/topic context.
            citations: URLs discovered by Perplexity.

        Returns:
            List of DiscoveredSource objects, sorted by relevance.
        """
        if not citations:
            return []

        urls_text = "\n".join(f"- {url}" for url in citations)
        user_msg = f"Topic: {domain}\n\nURLs to classify:\n{urls_text}"

        try:
            result = await self._llm.complete(
                system=_SOURCE_CLASSIFICATION_SYSTEM,
                user=user_msg,
                temperature=0.1,
            )

            sources = []
            for item in result.get("sources", []):
                url = item.get("url", "")
                if not url:
                    continue
                source_type_str = item.get("source_type", "other")
                try:
                    source_type = SourceType(source_type_str)
                except ValueError:
                    source_type = SourceType.OTHER

                sources.append(
                    DiscoveredSource(
                        url=url,
                        title=item.get("title", url.split("/")[-1] or url),
                        description=item.get("description", ""),
                        source_type=source_type,
                        relevance_score=min(1.0, max(0.0, float(item.get("relevance", 0.5)))),
                        access_method=item.get("access_method", "scrape"),
                        requires_auth=item.get("requires_auth", False),
                        discovered_via="perplexity",
                    )
                )

            # Sort by relevance descending
            sources.sort(key=lambda s: s.relevance_score, reverse=True)
            return sources

        except Exception as e:
            logger.warning(f"Source classification failed, returning raw citations: {e}")
            return [
                DiscoveredSource(
                    url=url,
                    title=url.split("/")[-1] or url,
                    relevance_score=0.5,
                    discovered_via="perplexity",
                )
                for url in citations
            ]

    def plan_fetch_strategy(self, source: DiscoveredSource, *, has_firecrawl: bool = True) -> FetchStrategy:
        """Decide how to fetch data from a source (no LLM call).

        Simple rule-based dispatch based on source type and access method.
        """
        # Direct downloads for file-like sources
        if source.source_type in (SourceType.CSV, SourceType.JSON, SourceType.PARQUET, SourceType.PDF):
            if source.access_method in ("direct_download", ""):
                return FetchStrategy(method="direct_download")

        # APIs need generated scripts
        if source.source_type == SourceType.API:
            return FetchStrategy(method="generated_script")

        # Repos need generated scripts (git clone)
        if source.source_type == SourceType.REPO:
            return FetchStrategy(method="generated_script")

        # Web pages: prefer Firecrawl if available
        if has_firecrawl:
            return FetchStrategy(method="firecrawl_scrape")

        # Fallback: direct download
        return FetchStrategy(method="direct_download")

    async def generate_fetch_script(
        self,
        source: DiscoveredSource,
        output_dir: str,
        *,
        max_pages: int = 10,
        timeout: int = 120,
        extra_context: str = "",
    ) -> str:
        """Generate a Python script to download data from a source.

        Args:
            source: The source to fetch.
            output_dir: Directory to write fetched data to.
            max_pages: Maximum pagination pages.
            timeout: Script execution timeout.
            extra_context: Additional context (e.g., error from previous attempt).

        Returns:
            Python script as a string.
        """
        system = _FETCH_SCRIPT_SYSTEM.format(max_pages=max_pages, timeout=timeout)
        user_msg = (
            f"Generate a Python script to download data from:\n"
            f"URL: {source.url}\n"
            f"Source type: {source.source_type.value}\n"
            f"Access method: {source.access_method}\n"
            f"Description: {source.description}\n\n"
            f"Save output to: {output_dir}"
        )
        if extra_context:
            user_msg += extra_context

        result = await self._llm.complete(
            system=system,
            user=user_msg,
            temperature=0.2,
        )

        # The LLM returns JSON with a "code" key, or sometimes the code directly
        if isinstance(result, dict):
            return result.get("code", result.get("script", str(result)))
        return str(result)

    async def summarize_content(self, content: str, source_title: str) -> str:
        """Generate a 2-3 sentence summary of fetched content.

        Args:
            content: The fetched text content (truncated to save tokens).
            source_title: Title of the source for context.

        Returns:
            Short summary string, or empty on failure.
        """
        truncated = content[:3000]
        try:
            result = await self._llm.complete(
                system=(
                    "Summarize the following fetched content in 2-3 sentences. "
                    "Focus on: what data it contains, how many records/items, "
                    "and whether it seems useful for knowledge graph construction. "
                    'Return JSON: {"summary": "..."}'
                ),
                user=f"Source: {source_title}\n\nContent:\n{truncated}",
                temperature=0.1,
            )
            return result.get("summary", "")
        except Exception as e:
            logger.debug(f"Content summarization failed: {e}")
            return ""
