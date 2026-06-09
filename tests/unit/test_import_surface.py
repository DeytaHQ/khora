"""Subprocess harness guarding ADR-118 V1 import-boundary invariants.

These tests assert that heavy submodules are NOT loaded by ``import khora``
and that public symbols remain accessible after lazification.
"""

from __future__ import annotations

import subprocess
import sys


def _run(script: str) -> None:
    subprocess.run(  # noqa: S603 — test harness, sys.executable is trusted
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_import_khora_does_not_load_query_engine() -> None:
    _run("import sys, khora; assert 'khora.query.engine' not in sys.modules")


def test_import_khora_does_not_load_integrations() -> None:
    _run("import sys, khora; assert 'khora.integrations' not in sys.modules")


def test_searchmode_resolves_from_engine() -> None:
    _run("from khora.query.engine import SearchMode; assert SearchMode is not None")


def test_searchmode_identity_preserved_across_reexports() -> None:
    _run("import khora; from khora.query import SearchMode; assert khora.SearchMode is SearchMode")
