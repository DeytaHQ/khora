"""LLM-backed planner for the discovery agent.

Uses LiteLLM (via OntologyLLM) to make planning decisions:
- Turn user intent into search queries
- Classify and rank discovered sources
- Decide fetch strategy (Firecrawl scrape vs. direct download vs. generated script)
- Generate Python fetch scripts for complex sources
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from loguru import logger

from khora.cli.ontology.llm import OntologyLLM

from .state import DiscoveredSource, SourceType


def _extract_code(raw: str) -> str:
    """Extract Python code from an LLM response.

    Handles:
    - Markdown code blocks (```python ... ``` or ``` ... ```)
    - Raw code without fences
    - JSON wrapper with a "code" key
    """
    # Try markdown code blocks first
    match = re.search(r"```(?:python)?\s*\n(.*?)```", raw, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try JSON wrapper: {"code": "..."}
    if raw.strip().startswith("{"):
        import json

        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                code = parsed.get("code") or parsed.get("script") or ""
                if code:
                    return code
        except json.JSONDecodeError:
            pass

    # Assume the raw response is the code itself
    return raw.strip()


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_QUERY_FORMULATION_SYSTEM = """\
You are a data source planning assistant.
Given a user's description of what knowledge graph they want to build,
generate 3-5 targeted search queries for finding downloadable datasets,
APIs, or structured data files. Each query should target a DIFFERENT
source type or aspect to maximize coverage.

Return JSON:
{
  "domain": "short domain label",
  "description": "1-sentence summary of what data they need",
  "search_queries": [
    "specific query targeting datasets/CSVs",
    "specific query targeting APIs or download pages",
    "specific query targeting academic/research data",
    "optional fourth query for government/official data",
    "optional fifth query for structured web pages"
  ],
  "preferred_formats": ["csv", "json", "api"]
}

Each query should be specific and target different source types."""

_SOURCE_CLASSIFICATION_SYSTEM = """\
Classify each URL into a data source type and rate its usefulness
for building a knowledge graph about the given topic.

Classify ALL URLs provided. Return all of them ranked by relevance — do not omit any.

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

_EXPLORATION_SYSTEM = """\
You are a data completeness analyst. Given what data has already been collected
for building a knowledge graph, identify GAPS and suggest follow-up search queries.

Return JSON:
{
  "analysis": "Brief assessment of what's missing",
  "queries": [
    "specific follow-up query 1",
    "specific follow-up query 2"
  ],
  "confidence": 0.0-1.0
}

Rules:
- Only suggest queries if there are genuine gaps
- Each query should target a DIFFERENT aspect than previous queries
- Return empty queries list [] if the data is comprehensive
- Be specific: "European wine production statistics 2020-2024 CSV" not "wine data"
"""

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

    Uses separate models per task via OntologyLLM:
    - ``planning_model`` for query formulation and source classification (fast, cheap)
    - ``codegen_model`` for generating fetch scripts (stronger model for better code)
    - ``summarization_model`` for content summarization (fast, cheap)

    Budget is shared with the discovery session.

    Usage::

        planner = DiscoveryPlanner.from_config(
            litellm_config_path="./litellm.discovery.yaml",
        )
        plan = await planner.formulate_queries("European wine datasets")
        ranked = await planner.classify_sources(plan.domain, citations)
    """

    def __init__(
        self,
        *,
        planning_model: str,
        codegen_model: str,
        summarization_model: str,
        budget_usd: float = 0.50,
    ) -> None:
        self._planning_llm = OntologyLLM(model=planning_model, budget_usd=budget_usd, interactive=False)
        self._codegen_llm = OntologyLLM(model=codegen_model, budget_usd=budget_usd, interactive=False)
        self._summary_llm = OntologyLLM(model=summarization_model, budget_usd=budget_usd, interactive=False)

    @classmethod
    def from_config(
        cls,
        *,
        litellm_config_path: str | None = None,
        planning_model: str | None = None,
        codegen_model: str | None = None,
        summarization_model: str | None = None,
        budget_usd: float = 2.0,
    ) -> DiscoveryPlanner:
        """Create a planner from litellm YAML config with env var overrides.

        Resolution order for each model:
        1. Explicit parameter (from CLI/env var)
        2. LiteLLM YAML config file
        3. KhoraConfig main LLM model as fallback
        """
        resolved: dict[str, str | None] = {"planning": None, "codegen": None, "summarization": None}

        # Load from YAML if provided
        if litellm_config_path:
            from pathlib import Path

            import yaml

            path = Path(litellm_config_path)
            if path.exists():
                with path.open() as f:
                    cfg = yaml.safe_load(f) or {}
                resolved["planning"] = (cfg.get("planning") or {}).get("model")
                resolved["codegen"] = (cfg.get("codegen") or {}).get("model")
                resolved["summarization"] = (cfg.get("summarization") or {}).get("model")
                if "budget_usd" in cfg:
                    budget_usd = cfg["budget_usd"]

        # Env var / explicit param overrides
        final_planning = planning_model or resolved["planning"]
        final_codegen = codegen_model or resolved["codegen"]
        final_summarization = summarization_model or resolved["summarization"]

        # Last resort: fall back to khora's main LLM model
        if not final_planning or not final_codegen or not final_summarization:
            try:
                from khora.config.schema import KhoraConfig

                main_model = KhoraConfig().llm.model
            except Exception:
                main_model = "gpt-4o-mini"
            final_planning = final_planning or main_model
            final_codegen = final_codegen or main_model
            final_summarization = final_summarization or main_model

        return cls(
            planning_model=final_planning,
            codegen_model=final_codegen,
            summarization_model=final_summarization,
            budget_usd=budget_usd,
        )

    @property
    def usage_summary(self) -> dict[str, Any]:
        planning = self._planning_llm.usage_summary
        codegen = self._codegen_llm.usage_summary
        summary = self._summary_llm.usage_summary
        return {
            "calls": planning["calls"] + codegen["calls"] + summary["calls"],
            "input_tokens": planning["input_tokens"] + codegen["input_tokens"] + summary["input_tokens"],
            "output_tokens": planning["output_tokens"] + codegen["output_tokens"] + summary["output_tokens"],
            "total_tokens": planning["total_tokens"] + codegen["total_tokens"] + summary["total_tokens"],
            "cost_usd": planning["cost_usd"] + codegen["cost_usd"] + summary["cost_usd"],
            "budget_remaining_usd": min(
                planning["budget_remaining_usd"],
                codegen["budget_remaining_usd"],
                summary["budget_remaining_usd"],
            ),
        }

    @property
    def cost_usd(self) -> float:
        return (
            self._planning_llm._total_cost_usd + self._codegen_llm._total_cost_usd + self._summary_llm._total_cost_usd
        )

    async def formulate_queries(
        self,
        user_intent: str,
        *,
        previous_queries: list[str] | None = None,
    ) -> QueryPlan:
        """Turn user intent into 3-5 targeted search queries.

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
            result = await self._planning_llm.complete(
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
            result = await self._planning_llm.complete(
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
        error_history: list[dict] | None = None,
    ) -> str:
        """Generate a Python script to download data from a source.

        Args:
            source: The source to fetch.
            output_dir: Directory to write fetched data to.
            max_pages: Maximum pagination pages.
            timeout: Script execution timeout.
            error_history: List of dicts describing previous failed attempts
                (keys: ``attempt``, ``error_type``, ``error``).

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

        if error_history:
            user_msg += "\n\n## Previous attempts (avoid repeating these failures):\n"
            for entry in error_history:
                user_msg += f"\n### Attempt {entry.get('attempt', '?')}:\n"
                user_msg += f"Error type: {entry.get('error_type', 'unknown')}\n"
                user_msg += f"Error: {entry.get('error', 'unknown')}\n"
            user_msg += "\nTake a COMPLETELY DIFFERENT approach from the failed attempts above."

        # Use complete_raw — code generation returns Python, not JSON
        raw = await self._codegen_llm.complete_raw(
            system=system,
            user=user_msg,
            temperature=0.2,
        )

        return _extract_code(raw)

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
            result = await self._summary_llm.complete(
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

    async def suggest_exploration(
        self,
        intent: str,
        fetched_summaries: list[str],
        previous_queries: list[str],
    ) -> list[str]:
        """Analyze fetched content and suggest follow-up search queries.

        Identifies gaps in the data collected so far and proposes 1-3
        refinement queries to fill them.

        Args:
            intent: Original user intent.
            fetched_summaries: Summaries of successfully fetched content.
            previous_queries: Queries already tried (to avoid repetition).

        Returns:
            List of 1-3 suggested follow-up queries, or empty list.
        """
        if not fetched_summaries:
            return []

        summaries_text = "\n".join(f"- {s}" for s in fetched_summaries[:10])
        prev_text = "\n".join(f"- {q}" for q in previous_queries) if previous_queries else "None"

        try:
            result = await self._planning_llm.complete(
                system=_EXPLORATION_SYSTEM,
                user=(
                    f"User intent: {intent}\n\n"
                    f"Data collected so far:\n{summaries_text}\n\n"
                    f"Previous search queries (already tried):\n{prev_text}\n\n"
                    "Suggest 1-3 follow-up queries to fill gaps in the collected data. "
                    "If the data is already comprehensive, return an empty list."
                ),
                temperature=0.3,
            )
            queries = result.get("queries", [])
            return [q for q in queries if isinstance(q, str) and q.strip()][:3]
        except Exception as e:
            logger.warning(f"Exploration suggestion failed: {e}")
            return []
