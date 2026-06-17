"""Acceptance tests for the strict-leaf ``khora.core.temporal`` module.

Locks in the guarantees of the temporal-types relocation:

* the public surface lives in ``khora.core.temporal``,
* the module's static import closure is leaf-pure — it reaches only stdlib
  and ``khora.core.models`` (no engine / DB-driver / filter machinery),
* the old ``khora.engines.skeleton.backends`` import path still works but
  emits a ``DeprecationWarning`` and preserves object identity,
* behaviour (chunk adaptation, denorm mapping) round-trips unchanged.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
import warnings
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import khora
import khora.core.temporal as temporal_mod
from khora.core.models.document import Chunk, Document
from khora.core.temporal import (
    ChunkTemporalFilter,
    TemporalChunk,
    TemporalSearchResult,
    document_denorm_fields,
    temporal_chunk_to_chunk,
)


def test_public_surface_present_and_exported() -> None:
    """All five neutral temporal names are importable and in ``__all__``."""
    expected = {
        "TemporalChunk",
        "ChunkTemporalFilter",
        "TemporalSearchResult",
        "document_denorm_fields",
        "temporal_chunk_to_chunk",
    }
    for name in expected:
        assert hasattr(temporal_mod, name), f"{name} missing from khora.core.temporal"
        assert name in temporal_mod.__all__, f"{name} missing from __all__"


# NOTE on scope: this test verifies the *static import closure* of
# ``khora.core.temporal`` — the guarantee this additive slice actually
# delivers ("the module imports only stdlib + ``khora.core.models``"). It
# deliberately does NOT assert runtime ``sys.modules`` isolation. A bare
# ``import khora.core.temporal`` still forces Python to execute the
# top-level ``khora/__init__.py``, which eagerly imports the engine and
# recall-filter subpackages (dragging sqlalchemy etc. into ``sys.modules``).
# Making that import lazy is a behavior-changing, dependency-inversion
# refactor that is out of scope here; the full runtime-isolation check is
# left to a later dependency-inversion slice.


def _resolve_khora_module_to_path(module: str) -> Path | None:
    """Map a ``khora.*`` dotted module name to its source file under ``src``.

    Returns the ``__init__.py`` for a package, the ``.py`` for a module, or
    ``None`` if no source file exists (e.g. a re-exported attribute name that
    is not itself a module).
    """
    khora_root = Path(khora.__file__).resolve().parent  # .../src/khora
    rel = module.split(".")[1:]  # drop leading "khora"
    base = khora_root.joinpath(*rel)
    candidate = base.with_suffix(".py")
    if candidate.is_file():
        return candidate
    pkg_init = base / "__init__.py"
    if pkg_init.is_file():
        return pkg_init
    return None


def _imports_of(path: Path) -> set[str]:
    """Return the set of top-level dotted module names imported by ``path``.

    Resolves ``from . import x`` / ``from .sub import y`` relative imports
    against the file's own package so the closure walk follows them.
    """
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    khora_root = Path(khora.__file__).resolve().parent.parent  # .../src
    dotted = ".".join(path.resolve().relative_to(khora_root).with_suffix("").parts)
    # The containing package is the module's parent for both a plain ``.py``
    # module and an ``__init__.py`` (whose dotted form already ends in
    # ``.__init__``).
    package = dotted.rsplit(".", 1)[0]

    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                # Relative import: resolve against this file's package.
                base_parts = package.split(".")
                # level 1 == current package, level 2 == parent, ...
                anchor = base_parts[: len(base_parts) - (node.level - 1)]
                target = ".".join([*anchor, node.module]) if node.module else ".".join(anchor)
                found.add(target)
            elif node.module:
                found.add(node.module)
    return found


def test_leaf_purity_static_import_closure() -> None:
    """``khora.core.temporal``'s static import closure is leaf-pure.

    Parses the module with :mod:`ast`, recursively follows only ``khora.*``
    imports to their source files, and asserts the transitive closure never
    references ``khora.filter`` / ``khora.engines`` / any DB driver — every
    reachable ``khora.*`` module must live under ``khora.core.models`` or be
    ``khora.core.temporal`` itself. Non-khora imports must be standard
    library only. Immune to the unrelated top-level package's eager imports
    (see the module-level scope note above).
    """
    start = "khora.core.temporal"
    allowed_prefixes = ("khora.core.models", "khora.core.temporal")

    seen: set[str] = set()
    queue = [start]
    non_khora_imports: set[str] = set()

    while queue:
        module = queue.pop()
        if module in seen:
            continue
        seen.add(module)

        if not module.startswith("khora.") and module != "khora":
            non_khora_imports.add(module)
            continue

        # Every reachable khora module in the closure must be on the allowlist.
        assert module == "khora" or module.startswith(allowed_prefixes), (
            f"khora.core.temporal's import closure reaches a non-leaf module: "
            f"{module!r} (only stdlib + khora.core.models allowed)"
        )

        path = _resolve_khora_module_to_path(module)
        if path is None:
            # Re-exported attribute (e.g. a class name), not a module.
            continue
        for imported in _imports_of(path):
            if imported not in seen:
                queue.append(imported)

    # Non-khora imports reached through the closure must all be stdlib.
    stdlib = sys.stdlib_module_names
    non_stdlib = {m for m in non_khora_imports if m.split(".")[0] not in stdlib}
    assert not non_stdlib, (
        f"khora.core.temporal's import closure reaches non-stdlib third-party modules: {sorted(non_stdlib)}"
    )


def test_deprecated_import_emits_warning() -> None:
    """Importing the relocated names from the old path raises under -W error.

    A fresh subprocess guarantees the backends module isn't already cached,
    so the import-time ``warnings.warn`` actually fires.
    """
    script = textwrap.dedent(
        """
        import warnings
        warnings.simplefilter("error", DeprecationWarning)
        import khora.engines.skeleton.backends  # noqa: F401
        """
    )
    result = subprocess.run(  # noqa: S603 — test harness, sys.executable is trusted
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0, "expected DeprecationWarning to be raised as error"
    assert "DeprecationWarning" in result.stderr
    assert "khora.core.temporal" in result.stderr


def test_legacy_alias_preserves_identity() -> None:
    """``TemporalFilter`` is the very same object as ``ChunkTemporalFilter``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from khora.engines.skeleton.backends import TemporalFilter

    assert TemporalFilter is ChunkTemporalFilter
    assert isinstance(ChunkTemporalFilter(), TemporalFilter)


def test_temporal_chunk_round_trips_to_chunk() -> None:
    """``temporal_chunk_to_chunk`` yields a ``Chunk`` with preserved identity."""
    cid, nsid, did = uuid4(), uuid4(), uuid4()
    tc = TemporalChunk(
        id=cid,
        namespace_id=nsid,
        document_id=did,
        content="hello temporal",
    )
    result = TemporalSearchResult(chunk=tc, similarity=0.9)

    chunk = temporal_chunk_to_chunk(result.chunk)

    assert isinstance(chunk, Chunk)
    assert chunk.id == cid
    assert chunk.namespace_id == nsid
    assert chunk.content == "hello temporal"


def test_document_denorm_fields_returns_eight_keys() -> None:
    """The denorm mapping surfaces exactly the eight provenance fields."""
    doc = Document(content="doc body", title="A Title", source_url="https://x")
    fields = document_denorm_fields(doc)

    assert set(fields) == {
        "source_type",
        "source_name",
        "source_url",
        "source_timestamp",
        "external_id",
        "content_type",
        "source",
        "title",
    }
    assert fields["title"] == "A Title"
    assert fields["source_url"] == "https://x"


def test_temporal_chunk_to_chunk_carries_event_and_producer_time() -> None:
    """Distinct ``occurred_at`` and ``source_timestamp`` survive adaptation."""
    occurred = datetime(2026, 1, 2, tzinfo=UTC)
    produced = datetime(2026, 1, 1, tzinfo=UTC)
    tc = TemporalChunk(
        id=uuid4(),
        namespace_id=uuid4(),
        document_id=uuid4(),
        content="body",
        occurred_at=occurred,
        source_timestamp=produced,
    )

    chunk = temporal_chunk_to_chunk(tc)

    assert chunk.occurred_at == occurred
    assert chunk.source_timestamp == produced
