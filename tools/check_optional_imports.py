#!/usr/bin/env python3
"""AST lint — ban top-level framework imports in adapter submodules.

Each adapter under ``src/khora/integrations/<framework>/`` MUST NOT import
its framework at module top level. If it does, then merely importing
``khora.integrations.<framework>`` (e.g. from a registry walk) raises
``ImportError`` whenever the optional extra isn't installed — which
breaks the whole point of the optional-install design.

This script walks every file under ``src/khora/integrations/<framework>/``
and rejects:

* ``import <framework>`` at module level
* ``from <framework>[.subpkg] import ...`` at module level
* ``from <framework>[.subpkg] import ...`` inside ``if TYPE_CHECKING`` is
  allowed (no runtime cost)
* Imports inside function / method bodies are allowed (lazy)

Run::

    python tools/check_optional_imports.py

Exit codes:
* 0 — no violations
* 1 — at least one violation
* 2 — script error (no integrations tree, etc.)
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

# Adapter directories live under this root. The root itself
# (protocol.py, registry.py, _sync.py, ...) is infrastructure, not an
# adapter, and is skipped.
INTEGRATIONS_ROOT = Path(__file__).resolve().parent.parent / "src" / "khora" / "integrations"


def _is_type_checking_guard(node: ast.AST) -> bool:
    """Return True if ``node`` is an ``if TYPE_CHECKING:`` block."""
    if not isinstance(node, ast.If):
        return False
    test = node.test
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    if (
        isinstance(test, ast.Attribute)
        and isinstance(test.value, ast.Name)
        and test.value.id == "typing"
        and test.attr == "TYPE_CHECKING"
    ):
        return True
    return False


def _module_root(module: str | None) -> str | None:
    """Return the top-level package name from a dotted module path."""
    if not module:
        return None
    return module.split(".", 1)[0]


def _check_top_level(tree: ast.Module, framework: str) -> list[tuple[int, str]]:
    """Return ``(lineno, message)`` for each top-level violation in tree."""
    violations: list[tuple[int, str]] = []
    for node in tree.body:
        if _is_type_checking_guard(node):
            continue  # allowed
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _module_root(alias.name) == framework:
                    violations.append((node.lineno, f"top-level `import {alias.name}` of optional framework"))
        elif isinstance(node, ast.ImportFrom):
            if _module_root(node.module) == framework:
                violations.append(
                    (
                        node.lineno,
                        f"top-level `from {node.module} import ...` of optional framework",
                    )
                )
    return violations


def check_file(path: Path, framework: str) -> list[str]:
    """Return human-readable violation strings for one adapter file."""
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"{path}: cannot read file ({exc})"]
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [f"{path}: syntax error at line {exc.lineno}: {exc.msg}"]
    return [f"{path}:{lineno}: {msg}" for lineno, msg in _check_top_level(tree, framework)]


def main() -> int:
    if not INTEGRATIONS_ROOT.is_dir():
        # The integrations tree may not exist yet on branches that
        # haven't merged #619. Treat as a no-op rather than a failure so
        # the script can ship before the first adapter does.
        print(f"ok: no {INTEGRATIONS_ROOT} tree to lint")
        return 0

    framework_dirs = [p for p in INTEGRATIONS_ROOT.iterdir() if p.is_dir() and not p.name.startswith("_")]
    if not framework_dirs:
        print("ok: no adapter subpackages to lint")
        return 0

    all_violations: list[str] = []
    for framework_dir in framework_dirs:
        framework = framework_dir.name
        for py_file in framework_dir.rglob("*.py"):
            all_violations.extend(check_file(py_file, framework))

    if all_violations:
        print("error: optional-framework imports leaked to module scope:", file=sys.stderr)
        for v in all_violations:
            print(f"  {v}", file=sys.stderr)
        print(
            "\nFix: move the import inside the function body that uses it, or guard with "
            "`if TYPE_CHECKING:`. Top-level imports break `pip install khora` without the "
            "framework extra.",
            file=sys.stderr,
        )
        return 1

    print(f"ok: {len(framework_dirs)} adapter subpackage(s) clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
