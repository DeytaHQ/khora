"""Unit tests for :func:`khora.filter.metadata_leaf_count` — ``@internal``.

``metadata_leaf_count`` counts the metadata-rooted leaf predicates in a filter
AST (the leaves that compile to an unindexed JSONB / object-column access). A
system-key leaf — a single-segment path naming a denormalized column — does not
count. The count is surfaced as the ``filter.metadata_leaf_count`` attribute on
the ``khora.recall`` span.

Pure AST counting — no Docker / DB. ASTs are built the way the facade does, by
validating a wire-form filter and lowering it (mirrors
``tests/recall/test_canonical_hash_stability.py``).
"""

from __future__ import annotations

import pytest

from khora.filter import RecallFilter, metadata_leaf_count, parse_to_ast

pytestmark = pytest.mark.unit


def _count(doc: dict) -> int:
    """Count metadata leaves of a public filter document the way the facade does."""
    return metadata_leaf_count(parse_to_ast(RecallFilter.model_validate(doc)))


def test_system_key_only_counts_zero() -> None:
    """A system-key-only filter has no metadata leaves."""
    assert _count({"source_type": "x"}) == 0


def test_empty_filter_counts_zero() -> None:
    """An empty (match-everything) filter has no metadata leaves."""
    assert _count({}) == 0


def test_dot_path_predicates_each_count() -> None:
    """Each folded ``metadata.<path>`` predicate is one leaf."""
    assert _count({"metadata.a": 1, "metadata.b": 2}) == 2


def test_bare_metadata_blob_counts_one() -> None:
    """A bare ``metadata`` dict is a single whole-blob leaf."""
    assert _count({"metadata": {"a": 1, "b": 2}}) == 1


def test_system_and_metadata_mix_counts_only_metadata() -> None:
    """A sibling system key does not inflate the metadata-leaf count."""
    assert _count({"source_name": "linear", "metadata.tier": "gold"}) == 1


def test_or_branches_count_their_metadata_leaves() -> None:
    """Logical ``$or`` nesting counts the metadata leaves in each branch."""
    assert _count({"$or": [{"metadata.a": 1}, {"metadata.b": 2}]}) == 2


def test_not_counts_its_inner_metadata_leaf() -> None:
    """A ``$not`` over a metadata predicate counts its inner leaf."""
    assert _count({"$not": {"metadata.a": 1}}) == 1


def test_in_metadata_leaf_counts_one() -> None:
    """An ``$in`` metadata predicate is a single leaf."""
    assert _count({"metadata.tags": {"$in": ["a", "b"]}}) == 1
