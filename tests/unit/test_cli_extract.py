"""Tests for khora extract CLI command."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from khora.cli.extract.commands import (
    _collect_sources,
    _read_file_content,
    extract,
)


@pytest.mark.unit
class TestExtractCommand:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(extract, ["--help"])
        assert result.exit_code == 0
        assert "Ingest files" in result.output

    def test_no_sources_error(self):
        runner = CliRunner()
        result = runner.invoke(extract, ["--format", "json"], input="")
        assert result.exit_code != 0

    def test_dry_run(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello world")
        runner = CliRunner()
        result = runner.invoke(extract, [str(f), "--dry-run", "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["status"] == "dry_run"
        assert data["files"] == 1

    def test_dry_run_directory(self, tmp_path):
        (tmp_path / "a.txt").write_text("File A")
        (tmp_path / "b.txt").write_text("File B")
        runner = CliRunner()
        result = runner.invoke(extract, [str(tmp_path), "--dry-run", "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["files"] == 2

    def test_stdin_dry_run(self):
        runner = CliRunner()
        result = runner.invoke(extract, ["-", "--dry-run", "--format", "json"], input="Hello from stdin")
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["stdin"] is True

    def test_format_json_flag(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Test content")
        runner = CliRunner()
        result = runner.invoke(extract, [str(f), "--dry-run", "--format", "json"])
        # Should be valid JSON
        json.loads(result.output)

    def test_format_text_flag(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Test content")
        runner = CliRunner()
        result = runner.invoke(extract, [str(f), "--dry-run", "--format", "text"])
        assert "Dry run" in result.output

    def test_dry_run_includes_engine(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("content")
        runner = CliRunner()
        result = runner.invoke(extract, [str(f), "--dry-run", "--format", "json", "-e", "skeleton"])
        data = json.loads(result.output)
        assert data["engine"] == "skeleton"

    def test_dry_run_includes_entity_types(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("content")
        runner = CliRunner()
        result = runner.invoke(
            extract,
            [str(f), "--dry-run", "--format", "json", "--entity-types", "PERSON,LOCATION"],
        )
        data = json.loads(result.output)
        assert data["entity_types"] == ["PERSON", "LOCATION"]

    def test_dry_run_includes_chunk_strategy(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("content")
        runner = CliRunner()
        result = runner.invoke(
            extract,
            [str(f), "--dry-run", "--format", "json", "--chunk-strategy", "fixed"],
        )
        data = json.loads(result.output)
        assert data["chunk_strategy"] == "fixed"


@pytest.mark.unit
class TestCollectSources:
    def test_collect_single_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        paths = _collect_sources((str(f),))
        assert len(paths) == 1
        assert paths[0] == f

    def test_collect_directory(self, tmp_path):
        (tmp_path / "a.txt").write_text("A")
        (tmp_path / "b.csv").write_text("B")
        (tmp_path / ".hidden").write_text("H")
        paths = _collect_sources((str(tmp_path),))
        assert len(paths) == 2  # .hidden excluded
        names = {p.name for p in paths}
        assert "a.txt" in names
        assert "b.csv" in names
        assert ".hidden" not in names

    def test_collect_nonexistent(self):
        paths = _collect_sources(("/nonexistent/path",))
        assert len(paths) == 0

    def test_collect_stdin_marker_skipped(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        paths = _collect_sources(("-", str(f)))
        assert len(paths) == 1
        assert paths[0] == f

    def test_collect_nested_directory(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.txt").write_text("nested")
        (tmp_path / "top.txt").write_text("top")
        paths = _collect_sources((str(tmp_path),))
        assert len(paths) == 2
        names = {p.name for p in paths}
        assert "nested.txt" in names
        assert "top.txt" in names


@pytest.mark.unit
class TestReadFileContent:
    def test_read_text_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello world")
        content, title, meta = _read_file_content(f)
        assert content == "Hello world"
        assert title == "test"
        assert meta["format"] == "txt"

    def test_read_json_file(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text('{"key": "value"}')
        content, title, meta = _read_file_content(f)
        assert "key" in content
        assert meta["format"] == "json"

    def test_read_csv_file(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("name,age\nAlice,30\nBob,25")
        content, title, meta = _read_file_content(f)
        assert "Alice" in content
        assert meta["format"] == "csv"

    def test_read_markdown_file(self, tmp_path):
        f = tmp_path / "notes.md"
        f.write_text("# Heading\n\nSome content")
        content, title, meta = _read_file_content(f)
        assert "Heading" in content
        assert title == "notes"
        assert meta["format"] == "md"

    def test_metadata_includes_source_path(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("content")
        _, _, meta = _read_file_content(f)
        assert meta["source_path"] == str(f)
