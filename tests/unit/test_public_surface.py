"""Public-surface contract for khora.

This file pins the *shape* of the public API so a refactor cannot silently
drop, rename, or re-couple a symbol a downstream consumer depends on. It
complements ``tests/unit/test_import_surface.py`` (which guards the lazy
import boundary); the headline value here is the ``khora.__all__`` snapshot
and the pinned deep-import paths.

Two flavors of assertion:
- In-process checks pin ``__all__`` and resolve the deep paths so coverage
  records the re-export wiring.
- A subprocess harness (fresh interpreter) makes the cold-import / lazy-load
  contract deterministic regardless of what prior tests imported in-process.
"""

from __future__ import annotations

import subprocess
import sys

import khora

# Snapshot of ``khora.__all__`` as of this contract. Keep this sorted literal
# in lockstep with ``src/khora/__init__.py`` — a deliberate change to the
# public surface must update this list in the same change.
EXPECTED_PUBLIC_SURFACE = frozenset(
    {
        "BatchHandle",
        "BatchResult",
        "DateOps",
        "DocumentProjection",
        "DocumentResult",
        "DocumentSource",
        "DreamConfig",
        "DreamMode",
        "DreamResult",
        "DreamRunInfo",
        "DreamScope",
        "EngineCapabilityError",
        "EntityTypeConfig",
        "EventType",
        "ExpertiseConfig",
        "FilterChannelReport",
        "FilterPushdownReport",
        "Khora",
        "KhoraConfig",
        "KhoraError",
        "LLMUsage",
        "Op",
        "OpKind",
        "RecallChunk",
        "RecallEntity",
        "RecallFilter",
        "RecallFilterUnsupportedError",
        "RecallFilterValidationError",
        "RecallRelationship",
        "RecallResult",
        "RelationshipTypeConfig",
        "RememberResult",
        "SYSTEM_KEYS",
        "SearchMode",
        "SemanticFilter",
        "Stats",
        "UsageSummary",
        "StringOps",
        "context_text",
        "create_engine",
        "integrations",
        "list_engines",
        "register_engine",
    }
)


def _run(script: str) -> None:
    subprocess.run(  # noqa: S603 — test harness, sys.executable is trusted
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_public_surface_matches_snapshot() -> None:
    """``khora.__all__`` must equal the pinned snapshot, naming any drift."""
    live = frozenset(khora.__all__)
    added = sorted(live - EXPECTED_PUBLIC_SURFACE)
    removed = sorted(EXPECTED_PUBLIC_SURFACE - live)
    assert live == EXPECTED_PUBLIC_SURFACE, (
        "khora.__all__ drifted from the pinned public surface. "
        f"ADDED (in live, not pinned): {added or 'none'}. "
        f"REMOVED (in pinned, not live): {removed or 'none'}. "
        "Update EXPECTED_PUBLIC_SURFACE in the same change that alters the "
        "public API — this guard exists so a drop/rename fails loudly."
    )


def test_public_surface_symbols_are_importable() -> None:
    """Every name in ``__all__`` must resolve as a real attribute."""
    missing = [name for name in khora.__all__ if not hasattr(khora, name)]
    assert not missing, (
        f"Names declared in khora.__all__ but not resolvable on the package: "
        f"{missing}. The public surface promises these import cleanly."
    )


def test_pinned_deep_import_paths_resolve() -> None:
    """Deep paths a consumer imports directly must not move silently."""
    # If any of these fails to import, a consumer's `from ... import ...`
    # breaks — pin them here so the move is caught at our boundary, not theirs.
    from khora import Khora, SearchMode  # noqa: F401
    from khora.core.models import DocumentSource, EventType  # noqa: F401
    from khora.storage import StorageCoordinator  # noqa: F401

    assert callable(khora.context_text), "khora.context_text must remain a top-level callable on the package"


def test_import_khora_does_not_load_query_engine() -> None:
    """Cold import must not eagerly pull in the heavy query engine module."""
    _run(
        "import sys, khora; "
        "assert 'khora.query.engine' not in sys.modules, "
        "'khora.query.engine re-coupled to import khora (must stay lazy)'"
    )


def test_import_khora_does_not_load_integration_adapters() -> None:
    """No adapter submodule may be eagerly loaded by ``import khora``.

    The bare ``khora.integrations`` package may or may not be present; the
    contract is specifically that no *adapter submodule* (e.g.
    ``khora.integrations.langgraph``) is pulled in on cold import.
    """
    _run(
        "import sys, khora; "
        "loaded = [m for m in sys.modules if m.startswith('khora.integrations.')]; "
        "assert not loaded, "
        "f'integration adapter modules eagerly imported by import khora: {loaded}'"
    )


def test_searchmode_resolves_through_every_path() -> None:
    """All three SearchMode references resolve to the same object."""
    _run(
        "from khora import SearchMode as a; "
        "from khora.query import SearchMode as b; "
        "from khora.query.engine import SearchMode as c; "
        "assert a is b is c, 'SearchMode re-exports diverged across paths'; "
        "assert a.__module__ == 'khora.search_mode', "
        "f'SearchMode.__module__ moved: {a.__module__!r} (expected khora.search_mode)'"
    )


def test_langgraph_adapter_loads_on_demand() -> None:
    """The langgraph adapter resolves its real export when imported explicitly."""
    _run(
        "from khora.integrations.langgraph import KhoraStore; "
        "assert KhoraStore is not None, "
        "'khora.integrations.langgraph.KhoraStore failed to resolve on demand'"
    )
