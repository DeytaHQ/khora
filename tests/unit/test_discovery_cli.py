"""Unit tests for discovery CLI command and integration."""

from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from khora.cli.ontology.commands import ontology_group
from khora.cli.ontology.discover import _has_discovery_keys, _render_discovered_sources
from khora.discovery.state import DiscoveredSource, SourceType


class TestHasDiscoveryKeys:
    def test_both_keys_present(self) -> None:
        with patch.dict("os.environ", {"PERPLEXITY_API_KEY": "pk", "FIRECRAWL_API_KEY": "fk"}):
            keys = _has_discovery_keys()
            assert keys["perplexity"] is True
            assert keys["firecrawl"] is True

    def test_perplexity_only(self) -> None:
        with patch.dict("os.environ", {"PERPLEXITY_API_KEY": "pk"}, clear=True):
            keys = _has_discovery_keys()
            assert keys["perplexity"] is True
            assert keys["firecrawl"] is False

    def test_no_keys(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            keys = _has_discovery_keys()
            assert keys["perplexity"] is False
            assert keys["firecrawl"] is False


class TestRenderDiscoveredSources:
    def test_renders_table(self) -> None:
        sources = [
            DiscoveredSource(
                url="https://example.com/data.csv",
                title="Test Dataset",
                source_type=SourceType.CSV,
                relevance_score=0.85,
            ),
            DiscoveredSource(
                url="https://api.example.com/v1",
                title="Test API",
                source_type=SourceType.API,
                relevance_score=0.3,
            ),
        ]
        table = _render_discovered_sources(sources)
        assert table.title == "Discovered Sources"
        assert table.row_count == 2


class TestDiscoverCommand:
    def test_discover_registered(self) -> None:
        """The discover command should be registered under the ontology group."""
        runner = CliRunner()
        result = runner.invoke(ontology_group, ["discover", "--help"])
        assert result.exit_code == 0
        assert "discover" in result.output.lower() or "datasources" in result.output.lower()

    def test_discover_no_keys_exits(self) -> None:
        """Without API keys, discover should show a helpful message."""
        runner = CliRunner()
        with patch.dict("os.environ", {}, clear=True):
            result = runner.invoke(ontology_group, ["discover"], input="quit\n")
            # Should mention API keys
            assert "API" in result.output or "key" in result.output.lower() or result.exit_code == 0
