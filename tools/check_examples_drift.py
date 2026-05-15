"""Fail CI if a doc-snippet diverges from its on-disk example.

Walks ``docs/integrations/*.md``, finds every fenced block opened with::

    ```python title="example.py"

…and asserts the body is byte-identical to
``examples/integrations/<framework>/example.py``. The ``<framework>``
slug is derived from the Markdown file stem.

Foundation-state behaviour: if ``docs/integrations/`` does not exist
yet (no adapters have merged), the script exits 0 with a note. Same
when a doc has no tagged block. Drift on any matched pair → exit 1
with a unified diff per file.

Usage::

    python tools/check_examples_drift.py [--repo-root PATH]
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from collections.abc import Iterator
from pathlib import Path

# Match a fenced block opened with ```python title="example.py" (optional
# leading whitespace, additional attrs tolerated). Captures the body.
_BLOCK_RE = re.compile(
    r"^[ \t]*```python[^\n]*title=\"example\.py\"[^\n]*\n(?P<body>.*?)(?<=\n)[ \t]*```",
    re.DOTALL | re.MULTILINE,
)


def extract_snippet(markdown: str) -> str | None:
    """Return the first ``python title=\"example.py\"`` block body, or None."""
    match = _BLOCK_RE.search(markdown)
    if match is None:
        return None
    return match.group("body")


def iter_doc_pairs(repo_root: Path) -> Iterator[tuple[Path, Path]]:
    """Yield (doc_path, example_path) for every adapter under docs/integrations/."""
    docs_dir = repo_root / "docs" / "integrations"
    if not docs_dir.is_dir():
        return
    for doc_path in sorted(docs_dir.glob("*.md")):
        framework = doc_path.stem
        example_path = repo_root / "examples" / "integrations" / framework / "example.py"
        yield doc_path, example_path


def check(repo_root: Path) -> int:
    """Return 0 on success, 1 on drift."""
    pairs = list(iter_doc_pairs(repo_root))
    if not pairs:
        print("[examples-drift] No docs/integrations/*.md files — nothing to check.")
        return 0

    drift = 0
    checked = 0
    for doc_path, example_path in pairs:
        snippet = extract_snippet(doc_path.read_text(encoding="utf-8"))
        if snippet is None:
            # No tagged block — adapter hasn't shipped a quickstart yet. Skip.
            continue
        if not example_path.is_file():
            print(
                f"[examples-drift] {doc_path.relative_to(repo_root)} has an "
                f"example.py snippet but {example_path.relative_to(repo_root)} "
                f"does not exist."
            )
            drift += 1
            continue

        on_disk = example_path.read_text(encoding="utf-8")
        checked += 1
        if snippet != on_disk:
            drift += 1
            diff = difflib.unified_diff(
                on_disk.splitlines(keepends=True),
                snippet.splitlines(keepends=True),
                fromfile=str(example_path.relative_to(repo_root)),
                tofile=f"{doc_path.relative_to(repo_root)} (snippet)",
            )
            print(f"[examples-drift] DRIFT: {doc_path.relative_to(repo_root)} vs {example_path.relative_to(repo_root)}")
            sys.stdout.writelines(diff)
            print()

    if drift:
        print(f"[examples-drift] {drift} mismatch(es) across {len(pairs)} adapter doc(s).")
        return 1

    print(f"[examples-drift] OK — {checked} snippet(s) matched byte-for-byte.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repo root (default: parent of this script's directory).",
    )
    args = parser.parse_args(argv)
    return check(args.repo_root)


if __name__ == "__main__":
    sys.exit(main())
