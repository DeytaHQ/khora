"""Unit tests for khora.query.__getattr__ lazy-import paths."""

from __future__ import annotations

import pytest


def test_hybrid_query_engine_resolves_to_correct_class() -> None:
    from khora.query import HybridQueryEngine
    from khora.query.engine import HybridQueryEngine as _HQE

    assert HybridQueryEngine is _HQE


def test_unknown_attribute_raises_attribute_error() -> None:
    import khora.query

    with pytest.raises(AttributeError, match="khora.query.*nonexistent_name"):
        getattr(khora.query, "nonexistent_name")


def test_lazy_cache_returns_same_object() -> None:
    import khora.query

    obj1 = getattr(khora.query, "HybridQueryEngine")
    obj2 = getattr(khora.query, "HybridQueryEngine")
    assert obj1 is obj2
