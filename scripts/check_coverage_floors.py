#!/usr/bin/env python3
"""Per-path coverage floor check.

Reads ``coverage.json`` (produced by ``uv run coverage json -o coverage.json``
after the existing pytest run) and asserts that each path in :data:`FLOORS`
meets or exceeds its minimum line-coverage percentage.

Why this exists
---------------

The repo already has a global ``--cov-fail-under=53`` gate in ``pyproject.toml``
which protects the aggregate. But aggregate coverage can stay flat while
critical code paths erode — e.g. someone could remove tests for
``sqlite_lance/vector.py`` and the global number wouldn't move enough to
fail. PR-A / PR-B / PR-C invested specifically in the embedded SQLite+LanceDB
stack and the supporting hot paths (chronicle, FTS5 escaping, _accel). This
gate stops those investments from being undone silently.

Floors are conservative — current measurement minus a small buffer — so the
gate only trips on real regressions, not on noise.

Usage
-----

::

    uv run coverage json -o coverage.json
    uv run python scripts/check_coverage_floors.py

Exit codes
~~~~~~~~~~

* ``0`` — every path meets/exceeds its floor.
* ``1`` — at least one floor violation, or a path listed in :data:`FLOORS`
  is absent from ``coverage.json`` (means the test run didn't touch that
  file at all — surface it loudly rather than silently passing).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Per-path line coverage floors (percent). Values are conservative — chosen as
# (current measurement - small buffer) so the gate has headroom but still
# catches real regressions. Bump these up over time, never down without a
# matching commit message explaining why.
FLOORS: dict[str, float] = {
    "src/khora/storage/backends/sqlite_lance/vector.py": 83,
    "src/khora/storage/backends/sqlite_lance/relational.py": 63,
    "src/khora/storage/backends/sqlite_lance/graph.py": 87,
    "src/khora/storage/backends/_fts5.py": 95,
    "src/khora/engines/chronicle/engine.py": 71,
    "src/khora/_accel.py": 51,
}

COVERAGE_JSON = Path("coverage.json")


def main() -> int:
    if not COVERAGE_JSON.exists():
        print(
            f"error: {COVERAGE_JSON} not found. Run `uv run coverage json -o coverage.json` first.",
            file=sys.stderr,
        )
        return 1

    data = json.loads(COVERAGE_JSON.read_text())
    files = data.get("files", {})

    violations: list[str] = []
    missing: list[str] = []

    for path, floor in FLOORS.items():
        entry = files.get(path)
        if entry is None:
            missing.append(path)
            continue
        actual = entry["summary"]["percent_covered"]
        if actual < floor:
            violations.append(f"{path}: {actual:.1f}% < {floor:.0f}% floor")

    if missing:
        print("error: paths listed in FLOORS are missing from coverage.json:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        print(
            "  (this means the test step did not exercise these files at all)",
            file=sys.stderr,
        )

    if violations:
        print("error: per-path coverage floors not met:", file=sys.stderr)
        for v in violations:
            print(f"  {v}", file=sys.stderr)

    if missing or violations:
        return 1

    print("ok: all per-path coverage floors satisfied")
    for path, floor in FLOORS.items():
        actual = files[path]["summary"]["percent_covered"]
        print(f"  {path}: {actual:.1f}% (floor {floor:.0f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
