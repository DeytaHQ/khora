"""Tests for khora search CLI command."""

from __future__ import annotations

import json
from uuid import uuid4

import pytest
from click.testing import CliRunner

from khora.cli.search.commands import search


@pytest.mark.unit
class TestSearchCommand:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(search, ["--help"])
        assert result.exit_code == 0
        assert "Search" in result.output

    def test_no_query_no_stdin_error(self):
        runner = CliRunner()
        result = runner.invoke(search, ["-n", str(uuid4()), "--format", "json"])
        # No query and no piped stdin -> error
        assert result.exit_code != 0

    def test_invalid_namespace_uuid(self):
        runner = CliRunner()
        result = runner.invoke(search, ["test query", "-n", "not-a-uuid", "--format", "json"])
        assert result.exit_code != 0
        data = json.loads(result.output)
        assert "Invalid namespace UUID" in data["error"]

    def test_query_from_argument(self):
        runner = CliRunner()
        ns = str(uuid4())
        result = runner.invoke(search, ["test query", "-n", ns, "--format", "json"])
        # Will fail at connection (no DB), not config error
        if result.exit_code != 0:
            data = json.loads(result.output)
            assert "error" in data

    def test_namespace_required(self):
        runner = CliRunner()
        result = runner.invoke(search, ["test query", "--format", "json"])
        assert result.exit_code != 0

    def test_mode_choices(self):
        runner = CliRunner()
        ns = str(uuid4())
        # Invalid mode should fail
        result = runner.invoke(search, ["query", "-n", ns, "--mode", "invalid"])
        assert result.exit_code != 0

    def test_valid_mode_options(self):
        """Verify all documented mode options are accepted by the CLI."""
        runner = CliRunner()
        ns = str(uuid4())
        for mode in ("vector", "graph", "hybrid", "all"):
            result = runner.invoke(search, ["query", "-n", ns, "--mode", mode, "--format", "json"])
            # Should not fail with "invalid choice" — connection error is expected
            assert "Invalid value for '--mode'" not in (result.output or "")

    def test_limit_option(self):
        runner = CliRunner()
        ns = str(uuid4())
        result = runner.invoke(search, ["query", "-n", ns, "--limit", "5", "--format", "json"])
        # Should parse limit without error (connection error is fine)
        assert "Invalid value for '--limit'" not in (result.output or "")

    def test_format_json_option(self):
        runner = CliRunner()
        ns = str(uuid4())
        result = runner.invoke(search, ["query", "-n", ns, "--format", "json"])
        # Output should be valid JSON (either success or error)
        if result.output.strip():
            json.loads(result.output)
