"""Tests for tools/check_examples_drift.py.

The drift detector compares the ``python title="example.py"`` block in
each ``docs/integrations/*.md`` against the on-disk
``examples/integrations/<framework>/example.py`` byte-for-byte. These
tests assert:

- foundation state (no docs dir at all) → exit 0
- empty docs dir → exit 0
- matching doc + example → exit 0
- mismatched doc + example → exit 1 with diff on stdout
- doc references an example.py that doesn't exist → exit 1
- doc with no tagged block is silently skipped
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Load the module directly from the repo's tools/ dir (it is not a package).
_TOOL_PATH = Path(__file__).resolve().parents[3] / "tools" / "check_examples_drift.py"
_spec = importlib.util.spec_from_file_location("check_examples_drift", _TOOL_PATH)
assert _spec is not None and _spec.loader is not None
check_examples_drift = importlib.util.module_from_spec(_spec)
sys.modules["check_examples_drift"] = check_examples_drift
_spec.loader.exec_module(check_examples_drift)


def _write_doc(repo: Path, framework: str, body: str | None) -> Path:
    docs = repo / "docs" / "integrations"
    docs.mkdir(parents=True, exist_ok=True)
    doc = docs / f"{framework}.md"
    if body is None:
        doc.write_text("# Adapter docs\n\nNo example yet.\n", encoding="utf-8")
    else:
        doc.write_text(
            f'# {framework} adapter\n\n```python title="example.py"\n{body}```\n',
            encoding="utf-8",
        )
    return doc


def _write_example(repo: Path, framework: str, body: str) -> Path:
    ex_dir = repo / "examples" / "integrations" / framework
    ex_dir.mkdir(parents=True, exist_ok=True)
    ex = ex_dir / "example.py"
    ex.write_text(body, encoding="utf-8")
    return ex


def test_foundation_state_no_docs_dir(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = check_examples_drift.check(tmp_path)
    assert rc == 0
    captured = capsys.readouterr()
    assert "nothing to check" in captured.out.lower()


def test_empty_docs_integrations_dir(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "docs" / "integrations").mkdir(parents=True)
    rc = check_examples_drift.check(tmp_path)
    assert rc == 0
    captured = capsys.readouterr()
    assert "nothing to check" in captured.out.lower()


def test_matching_snippet_and_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    body = "import khora\n\nprint('hi')\n"
    _write_doc(tmp_path, "crewai", body)
    _write_example(tmp_path, "crewai", body)
    rc = check_examples_drift.check(tmp_path)
    assert rc == 0
    captured = capsys.readouterr()
    assert "1 snippet(s) matched" in captured.out


def test_drift_detected(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_doc(tmp_path, "crewai", "import khora\nprint('doc version')\n")
    _write_example(tmp_path, "crewai", "import khora\nprint('disk version')\n")
    rc = check_examples_drift.check(tmp_path)
    assert rc == 1
    captured = capsys.readouterr()
    assert "DRIFT" in captured.out
    assert "doc version" in captured.out
    assert "disk version" in captured.out


def test_missing_example_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_doc(tmp_path, "langgraph", "import khora\n")
    # Note: no example.py written.
    rc = check_examples_drift.check(tmp_path)
    assert rc == 1
    captured = capsys.readouterr()
    assert "does not exist" in captured.out


def test_doc_without_tagged_block_is_skipped(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # Doc exists but has no `python title="example.py"` block.
    _write_doc(tmp_path, "future_adapter", body=None)
    rc = check_examples_drift.check(tmp_path)
    assert rc == 0
    captured = capsys.readouterr()
    assert "0 snippet(s) matched" in captured.out


def test_multiple_adapters_mixed(tmp_path: Path) -> None:
    body = "import khora\n"
    # Two passing, one drifting.
    _write_doc(tmp_path, "crewai", body)
    _write_example(tmp_path, "crewai", body)
    _write_doc(tmp_path, "langgraph", body)
    _write_example(tmp_path, "langgraph", body)
    _write_doc(tmp_path, "autogen", "import khora\nprint('drift')\n")
    _write_example(tmp_path, "autogen", body)
    rc = check_examples_drift.check(tmp_path)
    assert rc == 1


def test_extract_snippet_returns_none_when_no_match() -> None:
    assert check_examples_drift.extract_snippet("# heading\n\ntext\n") is None


def test_extract_snippet_returns_body() -> None:
    md = '```python title="example.py"\nbody line\n```\n'
    assert check_examples_drift.extract_snippet(md) == "body line\n"
