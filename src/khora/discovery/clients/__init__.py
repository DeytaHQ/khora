"""API clients for external discovery services."""

from __future__ import annotations

from .firecrawl import FirecrawlClient, FirecrawlCrawlResult, FirecrawlScrapeResult
from .perplexity import PerplexityClient, PerplexitySearchResponse

__all__ = [
    "FirecrawlClient",
    "FirecrawlCrawlResult",
    "FirecrawlScrapeResult",
    "PerplexityClient",
    "PerplexitySearchResponse",
]
