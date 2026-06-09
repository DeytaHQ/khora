"""Subprocess harness guarding ADR-118 V1 import-boundary invariants.

These tests assert that heavy submodules are NOT loaded by ``import khora``
and that public symbols remain accessible after lazification.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


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


# In-process exercises of the PEP 562 lazy shims added by ADR-118. These run in
# the pytest process (not the _run subprocess harness) so coverage records the
# __getattr__/__dir__ bodies — subprocess coverage is not configured for this repo.


def test_lazy_integrations_attribute_loads_and_caches() -> None:
    """Accessing ``khora.integrations`` resolves via the package ``__getattr__``."""
    import khora

    # Force the lazy path even if a prior test already cached the submodule.
    khora.__dict__.pop("integrations", None)
    assert khora.integrations is not None
    # Second access returns the now-cached module object.
    assert khora.integrations is khora.integrations


def test_khora_dir_includes_lazy_integrations() -> None:
    """``__dir__`` advertises the lazily-exposed ``integrations`` attribute."""
    import khora

    assert "integrations" in dir(khora)


def test_khora_unknown_attribute_raises() -> None:
    """The package ``__getattr__`` raises ``AttributeError`` for unknown names."""
    import khora

    with pytest.raises(AttributeError):
        _ = khora.does_not_exist


def test_query_lazy_symbol_loads_and_caches() -> None:
    """A lazy ``khora.query`` symbol resolves and caches via ``__getattr__``."""
    from khora import query

    query._lazy_cache.clear()
    engine_cls = query.HybridQueryEngine
    assert engine_cls is not None
    # Second access is served from the cache.
    assert query.HybridQueryEngine is engine_cls


def test_query_unknown_attribute_raises() -> None:
    """``khora.query.__getattr__`` raises ``AttributeError`` for unknown names."""
    from khora import query

    with pytest.raises(AttributeError):
        _ = query.does_not_exist


def test_query_dir_lists_lazy_symbols() -> None:
    """``khora.query.__dir__`` advertises the lazily-exposed symbols."""
    from khora import query

    assert "HybridQueryEngine" in dir(query)
