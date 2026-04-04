"""Firecrawl API client for web scraping and crawling.

Provides async wrappers around Firecrawl's ``/scrape`` and ``/crawl``
endpoints for extracting structured content from discovered URLs.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any

import httpx
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

FIRECRAWL_API_URL = "https://api.firecrawl.dev/v1"


@dataclass(slots=True)
class FirecrawlScrapeResult:
    """Result from scraping a single URL."""

    markdown: str = ""
    html: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    links: list[str] = field(default_factory=list)
    success: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "markdown": self.markdown,
            "html": self.html,
            "metadata": self.metadata,
            "links": self.links,
            "success": self.success,
        }


@dataclass(slots=True)
class FirecrawlCrawlResult:
    """Result from crawling a site (multiple pages)."""

    pages: list[FirecrawlScrapeResult] = field(default_factory=list)
    total_pages: int = 0
    success: bool = True
    partial: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "pages": [p.to_dict() for p in self.pages],
            "total_pages": self.total_pages,
            "success": self.success,
            "partial": self.partial,
        }


class FirecrawlClient:
    """Async client for the Firecrawl web scraping API.

    Supports single-page scraping and multi-page crawling with
    automatic job polling for crawl operations.

    Usage::

        async with FirecrawlClient() as client:
            result = await client.scrape("https://example.com/data")
            print(result.markdown[:500])
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 60.0,
        max_crawl_pages: int = 20,
    ) -> None:
        self._api_key = api_key or os.environ.get("FIRECRAWL_API_KEY", "")
        self._timeout = timeout
        self._max_crawl_pages = max_crawl_pages
        self._client: httpx.AsyncClient | None = None

    @property
    def available(self) -> bool:
        """Whether the Firecrawl API key is configured."""
        return bool(self._api_key)

    async def __aenter__(self) -> FirecrawlClient:
        self._client = httpx.AsyncClient(
            base_url=FIRECRAWL_API_URL,
            timeout=httpx.Timeout(self._timeout),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _ensure_connected(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("Client not connected. Use `async with` context manager.")
        return self._client

    @retry(
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError, httpx.ReadTimeout)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def scrape(
        self,
        url: str,
        *,
        formats: list[str] | None = None,
    ) -> FirecrawlScrapeResult:
        """Scrape a single URL and return structured content.

        Args:
            url: URL to scrape.
            formats: Output formats. Default: ``["markdown"]``.
                Options: ``"markdown"``, ``"html"``, ``"rawHtml"``,
                ``"links"``, ``"screenshot"``.

        Returns:
            FirecrawlScrapeResult with markdown content and metadata.
        """
        client = self._ensure_connected()

        logger.debug(f"Firecrawl scrape: {url}")
        response = await client.post(
            "/scrape",
            json={
                "url": url,
                "formats": formats or ["markdown"],
            },
        )
        response.raise_for_status()
        data = response.json().get("data", {})

        return FirecrawlScrapeResult(
            markdown=data.get("markdown", ""),
            html=data.get("html", ""),
            metadata=data.get("metadata", {}),
            links=data.get("links", []),
            success=data.get("success", True),
        )

    async def crawl(
        self,
        url: str,
        *,
        max_pages: int | None = None,
        poll_interval: float = 5.0,
        max_poll_attempts: int = 60,
    ) -> FirecrawlCrawlResult:
        """Crawl a site starting from the given URL.

        Starts an async crawl job and polls until completion.
        On timeout, returns any pages accumulated so far with ``partial=True``.

        Args:
            url: Starting URL for the crawl.
            max_pages: Maximum pages to crawl (default: client setting).
            poll_interval: Seconds between status polls.
            max_poll_attempts: Maximum number of poll attempts before timeout.

        Returns:
            FirecrawlCrawlResult with all crawled pages.
        """
        client = self._ensure_connected()
        limit = max_pages or self._max_crawl_pages

        logger.debug(f"Firecrawl crawl: {url} (max {limit} pages)")

        # Start crawl job (retry on transient failures)
        job_id = await self._start_crawl_job(url, limit)

        if not job_id:
            raise ValueError("Firecrawl did not return a crawl job ID")

        logger.debug(f"Firecrawl crawl job started: {job_id}")

        # Poll for completion, accumulating partial pages
        accumulated_pages: list[FirecrawlScrapeResult] = []
        seen_urls: set[str] = set()

        for attempt in range(max_poll_attempts):
            await asyncio.sleep(poll_interval)

            status_resp = await client.get(f"/crawl/{job_id}")
            status_resp.raise_for_status()
            status_data = status_resp.json()
            status = status_data.get("status", "")

            # Accumulate pages from each poll response
            for page_data in status_data.get("data", []):
                page_url = page_data.get("metadata", {}).get("sourceURL", "")
                if page_url and page_url in seen_urls:
                    continue
                if page_url:
                    seen_urls.add(page_url)
                accumulated_pages.append(
                    FirecrawlScrapeResult(
                        markdown=page_data.get("markdown", ""),
                        html=page_data.get("html", ""),
                        metadata=page_data.get("metadata", {}),
                        links=page_data.get("links", []),
                    )
                )

            if status == "completed":
                logger.info(f"Firecrawl crawl complete: {len(accumulated_pages)} page(s) from {url}")
                return FirecrawlCrawlResult(
                    pages=accumulated_pages,
                    total_pages=len(accumulated_pages),
                    success=True,
                )

            if status == "failed":
                error = status_data.get("error", "unknown error")
                logger.error(f"Firecrawl crawl failed: {error}")
                if accumulated_pages:
                    logger.warning(f"Returning {len(accumulated_pages)} partial page(s) despite crawl failure")
                    return FirecrawlCrawlResult(
                        pages=accumulated_pages,
                        total_pages=len(accumulated_pages),
                        success=True,
                        partial=True,
                    )
                return FirecrawlCrawlResult(success=False)

            logger.debug(f"Firecrawl crawl poll {attempt + 1}/{max_poll_attempts}: {status}")

        # Timeout — return whatever we've accumulated
        logger.error(f"Firecrawl crawl timed out after {max_poll_attempts * poll_interval:.0f}s")
        if accumulated_pages:
            logger.warning(f"Returning {len(accumulated_pages)} partial page(s) on timeout")
            return FirecrawlCrawlResult(
                pages=accumulated_pages,
                total_pages=len(accumulated_pages),
                success=True,
                partial=True,
            )
        return FirecrawlCrawlResult(success=False)

    @retry(
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError, httpx.ReadTimeout)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _start_crawl_job(self, url: str, limit: int) -> str | None:
        """Start a crawl job and return the job ID (retries on transient errors)."""
        client = self._ensure_connected()
        response = await client.post(
            "/crawl",
            json={
                "url": url,
                "limit": limit,
                "scrapeOptions": {"formats": ["markdown"]},
            },
        )
        response.raise_for_status()
        job = response.json()
        return job.get("id")
