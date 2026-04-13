"""Tests for CLI shared utilities."""

from __future__ import annotations

import json
from unittest.mock import patch
from uuid import uuid4

import pytest

from khora.cli._common import (
    EXIT_CONFIG_ERROR,
    EXIT_CONNECTION_ERROR,
    EXIT_PARTIAL_FAILURE,
    EXIT_SUCCESS,
    detect_output_format,
    write_json,
    write_text,
)


@pytest.mark.unit
class TestExitCodes:
    def test_values_are_distinct(self):
        codes = {EXIT_SUCCESS, EXIT_PARTIAL_FAILURE, EXIT_CONFIG_ERROR, EXIT_CONNECTION_ERROR}
        assert len(codes) == 4

    def test_success_is_zero(self):
        assert EXIT_SUCCESS == 0


@pytest.mark.unit
class TestDetectOutputFormat:
    def test_explicit_json(self):
        assert detect_output_format("json") == "json"

    def test_explicit_text(self):
        assert detect_output_format("text") == "text"

    def test_auto_tty(self):
        with patch("sys.stdout") as mock_stdout:
            mock_stdout.isatty.return_value = True
            assert detect_output_format(None) == "text"

    def test_auto_pipe(self):
        with patch("sys.stdout") as mock_stdout:
            mock_stdout.isatty.return_value = False
            assert detect_output_format(None) == "json"


@pytest.mark.unit
class TestWriteJson:
    def test_writes_valid_json(self, capsys):
        write_json({"status": "success", "count": 42})
        output = capsys.readouterr().out
        parsed = json.loads(output)
        assert parsed["status"] == "success"
        assert parsed["count"] == 42

    def test_handles_uuid(self, capsys):
        uid = uuid4()
        write_json({"id": uid})
        output = capsys.readouterr().out
        parsed = json.loads(output)
        assert parsed["id"] == str(uid)

    def test_output_ends_with_newline(self, capsys):
        write_json({"a": 1})
        output = capsys.readouterr().out
        assert output.endswith("\n")


@pytest.mark.unit
class TestWriteText:
    def test_writes_lines(self, capsys):
        write_text(["line one", "line two"])
        output = capsys.readouterr().out
        assert "line one\n" in output
        assert "line two\n" in output

    def test_empty_list(self, capsys):
        write_text([])
        output = capsys.readouterr().out
        assert output == ""
