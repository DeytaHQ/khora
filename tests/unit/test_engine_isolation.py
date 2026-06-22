"""Regression guard for vectorcypher → skeleton engine isolation.

Asserts that ``khora.engines.vectorcypher`` and all of its submodules import
nothing from ``khora.engines.skeleton``. Two independent checks back this up:
a runtime ``sys.modules`` probe in a fresh subprocess, and a static AST scan of
every source file under the vectorcypher package.
"""

# AMENDABLE GUIDELINE TRIPWIRE — not a hard import-linter contract.
#
# The engine-isolation rule says the vectorcypher engine must stand on its own
# and must not reach into the skeleton engine's internals: the two engines are
# independent implementations of the memory-engine protocol, and coupling one
# to the other erodes that boundary. This test locks that property in so an
# *accidental* re-coupling (a stray ``from khora.engines.skeleton import ...``
# added during unrelated work) trips loudly in CI rather than landing silently.
#
# A developer MAY consciously relax this guard if the isolation rule itself
# changes — but only deliberately, with a justification in the same diff that
# edits this file. It is a guideline, not a wall: the point is that the change
# is *seen*, not that it is forbidden.

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_FORBIDDEN_PREFIX = "khora.engines.skeleton"
_VECTORCYPHER_PACKAGE = "khora.engines.vectorcypher"

# Submodules imported individually by the runtime probe so that a skeleton
# import hiding in any one of them (not just the package __init__) is caught.
# Keep in sync with the package's top-level submodules; the static AST scan
# (rglob over every .py) is the backstop if this drifts.
_SUBMODULES = (
    "engine",
    "retriever",
    "dual_nodes",
    "fusion",
    "ppr_retrieval",
    "router",
    "temporal_detection",
)


def _vectorcypher_dir() -> Path:
    """Locate the vectorcypher package source directory via its spec."""
    spec = importlib.util.find_spec(_VECTORCYPHER_PACKAGE)
    assert spec is not None, f"cannot locate {_VECTORCYPHER_PACKAGE}"
    locations = spec.submodule_search_locations
    assert locations, f"{_VECTORCYPHER_PACKAGE} is not a package"
    return Path(next(iter(locations)))


def test_runtime_import_does_not_load_skeleton() -> None:
    """A fresh process importing vectorcypher must not pull in skeleton modules."""
    imports = "; ".join(f"import {_VECTORCYPHER_PACKAGE}.{name}" for name in _SUBMODULES)
    script = (
        "import sys\n"
        f"import {_VECTORCYPHER_PACKAGE}\n"
        f"{imports}\n"
        f"prefix = {_FORBIDDEN_PREFIX!r}\n"
        "offenders = sorted(\n"
        "    m for m in sys.modules\n"
        "    if m == prefix or m.startswith(prefix + '.')\n"
        ")\n"
        "if offenders:\n"
        "    print('\\n'.join(offenders))\n"
        "    sys.exit(1)\n"
    )
    result = subprocess.run(  # noqa: S603 — test harness, sys.executable is trusted
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"importing khora.engines.vectorcypher loaded forbidden skeleton modules:\n{result.stdout}{result.stderr}"
    )


def _resolved_targets(tree: ast.Module, module_name: str) -> list[tuple[int, str]]:
    """Yield (lineno, resolved-module) for every import in ``tree``.

    Relative imports are resolved against ``module_name`` so that a
    ``from ..skeleton import x`` inside the vectorcypher package resolves to
    its absolute ``khora.engines.skeleton`` form. ``from khora.engines import
    skeleton`` is expanded to the imported submodule too, so the bare-package
    form is caught. TYPE_CHECKING-guarded imports are visible to the AST and
    therefore included.
    """
    package_parts = module_name.split(".")
    results: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                results.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                base = node.module or ""
            else:
                # Resolve the relative import against the current module's
                # package. level 1 == current package, level 2 == parent, etc.
                anchor = package_parts[: len(package_parts) - node.level]
                base = ".".join([*anchor, node.module] if node.module else anchor)
            results.append((node.lineno, base))
            # ``from <base> import <name>`` may itself name a submodule, so
            # record each imported name as a candidate dotted module too.
            for alias in node.names:
                results.append((node.lineno, f"{base}.{alias.name}" if base else alias.name))
    return results


def _module_name_for(path: Path, pkg_dir: Path) -> str:
    """Compute the dotted module name for ``path`` within the package."""
    rel = path.relative_to(pkg_dir).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join([_VECTORCYPHER_PACKAGE, *parts]) if parts else _VECTORCYPHER_PACKAGE


def test_static_scan_finds_no_skeleton_import() -> None:
    """No .py file under the vectorcypher package imports skeleton (incl. relative)."""
    pkg_dir = _vectorcypher_dir()
    offenders: list[str] = []
    for path in sorted(pkg_dir.rglob("*.py")):
        module_name = _module_name_for(path, pkg_dir)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for lineno, target in _resolved_targets(tree, module_name):
            if target == _FORBIDDEN_PREFIX or target.startswith(_FORBIDDEN_PREFIX + "."):
                offenders.append(f"{path}:{lineno}: imports {target}")
    assert not offenders, "vectorcypher imports forbidden skeleton modules:\n" + "\n".join(offenders)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
