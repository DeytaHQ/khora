"""Unit tests for discovery API clients (Perplexity + Firecrawl).

All tests use mocked httpx transports — no real API calls are made.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from khora.discovery.clients.firecrawl import FirecrawlClient, FirecrawlScrapeResult
from khora.discovery.clients.perplexity import PerplexityClient, PerplexitySearchResponse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response(data: dict[str, Any], status_code: int = 200) -> httpx.Response:
    """Build a fake httpx.Response from a dict."""
    return httpx.Response(
        status_code=status_code,
        json=data,
        request=httpx.Request("POST", "https://mock"),
    )


# ---------------------------------------------------------------------------
# PerplexityClient
# ---------------------------------------------------------------------------


class TestPerplexityClient:
    """Tests for PerplexityClient."""

    def test_available_with_key(self) -> None:
        client = PerplexityClient(api_key="test-key")
        assert client.available is True

    def test_not_available_without_key(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            client = PerplexityClient(api_key="")
            assert client.available is False

    @pytest.mark.asyncio
    async def test_search_returns_citations(self) -> None:
        mock_data = {
            "choices": [
                {
                    "message": {
                        "content": "Here are some wine datasets: ...",
                    }
                }
            ],
            "citations": [
                "https://example.com/wine.csv",
                "https://example.com/wine-api",
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 200,
            },
        }

        async with PerplexityClient(api_key="test-key") as client:
            # Mock the httpx client's post method
            client._client.post = AsyncMock(return_value=_mock_response(mock_data))  # type: ignore[union-attr]

            response = await client.search("wine datasets CSV download")

            assert isinstance(response, PerplexitySearchResponse)
            assert "wine datasets" in response.answer
            assert len(response.citations) == 2
            assert response.citations[0] == "https://example.com/wine.csv"
            assert response.usage["input_tokens"] == 100
            assert response.usage["output_tokens"] == 200

    @pytest.mark.asyncio
    async def test_search_with_domain_hint(self) -> None:
        mock_data = {
            "choices": [{"message": {"content": "Found sources."}}],
            "citations": [],
            "usage": {},
        }

        async with PerplexityClient(api_key="test-key") as client:
            client._client.post = AsyncMock(return_value=_mock_response(mock_data))  # type: ignore[union-attr]

            await client.search("wine data", domain_hint="oenology")

            # Verify domain hint was included in the system prompt
            call_args = client._client.post.call_args  # type: ignore[union-attr]
            payload = call_args.kwargs.get("json") or call_args.args[1]
            system_msg = payload["messages"][0]["content"]
            assert "oenology" in system_msg

    @pytest.mark.asyncio
    async def test_search_empty_response(self) -> None:
        mock_data = {
            "choices": [{"message": {"content": ""}}],
            "citations": [],
            "usage": {},
        }

        async with PerplexityClient(api_key="test-key") as client:
            client._client.post = AsyncMock(return_value=_mock_response(mock_data))  # type: ignore[union-attr]

            response = await client.search("nonexistent topic")
            assert response.answer == ""
            assert response.citations == []

    @pytest.mark.asyncio
    async def test_not_connected_raises(self) -> None:
        client = PerplexityClient(api_key="test-key")
        with pytest.raises(RuntimeError, match="not connected"):
            await client.search("test")

    def test_response_to_dict(self) -> None:
        resp = PerplexitySearchResponse(
            answer="test",
            citations=["https://a.com"],
            usage={"input_tokens": 10},
        )
        d = resp.to_dict()
        assert d["answer"] == "test"
        assert d["citations"] == ["https://a.com"]


# ---------------------------------------------------------------------------
# FirecrawlClient
# ---------------------------------------------------------------------------


class TestFirecrawlClient:
    """Tests for FirecrawlClient."""

    def test_available_with_key(self) -> None:
        client = FirecrawlClient(api_key="test-key")
        assert client.available is True

    def test_not_available_without_key(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            client = FirecrawlClient(api_key="")
            assert client.available is False

    @pytest.mark.asyncio
    async def test_scrape_returns_markdown(self) -> None:
        mock_data = {
            "data": {
                "markdown": "# Wine Data\n\nSome content...",
                "metadata": {"title": "Wine Data", "sourceURL": "https://example.com"},
                "links": ["https://example.com/page2"],
                "success": True,
            }
        }

        async with FirecrawlClient(api_key="test-key") as client:
            client._client.post = AsyncMock(return_value=_mock_response(mock_data))  # type: ignore[union-attr]

            result = await client.scrape("https://example.com/wine")

            assert isinstance(result, FirecrawlScrapeResult)
            assert "Wine Data" in result.markdown
            assert result.metadata["title"] == "Wine Data"
            assert len(result.links) == 1
            assert result.success is True

    @pytest.mark.asyncio
    async def test_scrape_with_custom_formats(self) -> None:
        mock_data = {"data": {"markdown": "", "html": "<h1>Test</h1>", "success": True}}

        async with FirecrawlClient(api_key="test-key") as client:
            client._client.post = AsyncMock(return_value=_mock_response(mock_data))  # type: ignore[union-attr]

            await client.scrape("https://example.com", formats=["markdown", "html"])

            call_args = client._client.post.call_args  # type: ignore[union-attr]
            payload = call_args.kwargs.get("json") or call_args.args[1]
            assert payload["formats"] == ["markdown", "html"]

    @pytest.mark.asyncio
    async def test_crawl_polls_until_complete(self) -> None:
        # First response: job started
        start_response = _mock_response({"id": "job-123"})

        # Poll responses: first pending, then completed
        pending_response = _mock_response({"status": "scraping"})
        complete_response = _mock_response(
            {
                "status": "completed",
                "data": [
                    {
                        "markdown": "# Page 1",
                        "metadata": {"sourceURL": "https://example.com"},
                    },
                    {
                        "markdown": "# Page 2",
                        "metadata": {"sourceURL": "https://example.com/page2"},
                    },
                ],
            }
        )

        async with FirecrawlClient(api_key="test-key") as client:
            client._client.post = AsyncMock(return_value=start_response)  # type: ignore[union-attr]
            client._client.get = AsyncMock(side_effect=[pending_response, complete_response])  # type: ignore[union-attr]

            result = await client.crawl("https://example.com", poll_interval=0.01)

            assert result.success is True
            assert result.total_pages == 2
            assert len(result.pages) == 2
            assert "Page 1" in result.pages[0].markdown

    @pytest.mark.asyncio
    async def test_crawl_handles_failure(self) -> None:
        start_response = _mock_response({"id": "job-456"})
        failed_response = _mock_response({"status": "failed", "error": "rate limited"})

        async with FirecrawlClient(api_key="test-key") as client:
            client._client.post = AsyncMock(return_value=start_response)  # type: ignore[union-attr]
            client._client.get = AsyncMock(return_value=failed_response)  # type: ignore[union-attr]

            result = await client.crawl("https://example.com", poll_interval=0.01)
            assert result.success is False

    @pytest.mark.asyncio
    async def test_crawl_timeout(self) -> None:
        start_response = _mock_response({"id": "job-789"})
        pending_response = _mock_response({"status": "scraping"})

        async with FirecrawlClient(api_key="test-key") as client:
            client._client.post = AsyncMock(return_value=start_response)  # type: ignore[union-attr]
            client._client.get = AsyncMock(return_value=pending_response)  # type: ignore[union-attr]

            result = await client.crawl(
                "https://example.com",
                poll_interval=0.01,
                max_poll_attempts=3,
            )
            assert result.success is False

    @pytest.mark.asyncio
    async def test_not_connected_raises(self) -> None:
        client = FirecrawlClient(api_key="test-key")
        with pytest.raises(RuntimeError, match="not connected"):
            await client.scrape("https://example.com")

    def test_scrape_result_to_dict(self) -> None:
        result = FirecrawlScrapeResult(
            markdown="# Test",
            metadata={"title": "Test"},
            links=["https://a.com"],
            success=True,
        )
        d = result.to_dict()
        assert d["markdown"] == "# Test"
        assert d["success"] is True
