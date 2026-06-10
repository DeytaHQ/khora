"""Stability contract for the recall-filter canonical hash.

The filter-pushdown spies prove a path threaded the filter by comparing
``canonical_hash`` of the captured ``filter_ast`` against the hash of an
expected AST rebuilt from the public filter document. That comparison is
only sound if the hash is STABLE under semantically-irrelevant variation
and SENSITIVE to semantically-relevant variation. This module pins both
directions so a regression in ``canonical_hash`` (which would silently
weaken or break every spy) is caught here first.

No database, no I/O — pure AST hashing. Mirrors the equality/inequality
families ``canonical_hash`` documents (``src/khora/filter/ast.py``):

* commutative ``$and`` / ``$or`` siblings sort -> reorder is invisible,
* dict-operand keys sort -> object key-order is invisible,
* ``$in`` / ``$nin`` list order is significant -> reorder is visible,
* a different op / path / operand hashes differently.

It also covers the shared helper's :func:`expected_hash` oracle, since
that is the exact function the spies trust.
"""

from __future__ import annotations

import pytest

from khora.filter import RecallFilter, canonical_hash, parse_to_ast
from tests.test_helpers.filter_spy import expected_hash

pytestmark = [pytest.mark.unit, pytest.mark.filter_enforcement]


def _h(doc: dict) -> str:
    """Hash a public filter document the way the facade does."""
    return canonical_hash(parse_to_ast(RecallFilter.model_validate(doc)))


# --------------------------------------------------------------------------- #
# Stable: semantically-equal constructions hash equally.
# --------------------------------------------------------------------------- #


def test_and_sibling_reorder_same_hash() -> None:
    """Reordering top-level ($and) predicates does not change the hash."""
    a = _h({"source_name": "linear", "occurred_at": {"$gte": "2026-04-05"}})
    b = _h({"occurred_at": {"$gte": "2026-04-05"}, "source_name": "linear"})
    assert a == b


def test_or_sibling_reorder_same_hash() -> None:
    """Reordering $or branches does not change the hash (commutative)."""
    a = _h({"$or": [{"source_name": "linear"}, {"source_name": "slack"}]})
    b = _h({"$or": [{"source_name": "slack"}, {"source_name": "linear"}]})
    assert a == b


def test_dict_operand_key_reorder_same_hash() -> None:
    """Whole-object equality operands sort keys: key-order is invisible."""
    a = _h({"metadata": {"$eq": {"a": 1, "b": 2}}})
    b = _h({"metadata": {"$eq": {"b": 2, "a": 1}}})
    assert a == b


def test_recallfilter_instance_and_dict_agree() -> None:
    """A RecallFilter instance and the equivalent dict hash identically.

    The facade uses the instance as-is but validates a dict via
    ``model_validate``; both must land on the same AST and hash.
    """
    doc = {"source_name": "linear", "occurred_at": {"$gte": "2026-04-05"}}
    from_dict = expected_hash(doc)
    from_instance = expected_hash(RecallFilter.model_validate(doc))
    assert from_dict == from_instance


def test_expected_hash_matches_facade_construction() -> None:
    """The helper oracle equals the literal facade build sequence."""
    doc = {"source_name": {"$in": ["linear", "slack"]}}
    assert expected_hash(doc) == _h(doc)


# --------------------------------------------------------------------------- #
# Sensitive: different semantics hash differently.
# --------------------------------------------------------------------------- #


def test_different_value_differs() -> None:
    assert _h({"source_name": "linear"}) != _h({"source_name": "slack"})


def test_different_operator_differs() -> None:
    assert _h({"occurred_at": {"$gte": "2026-04-05"}}) != _h({"occurred_at": {"$gt": "2026-04-05"}})


def test_different_path_differs() -> None:
    assert _h({"source_name": "linear"}) != _h({"source_type": "linear"})


def test_in_list_order_is_significant() -> None:
    """$in membership lists are order-significant: reorder MUST differ.

    This is the one case that is intentionally NOT commutative — guards
    against a future "sort everything" change that would over-normalise.
    """
    assert _h({"source_name": {"$in": ["a", "b"]}}) != _h({"source_name": {"$in": ["b", "a"]}})


def test_and_vs_or_differs() -> None:
    """Same leaves under $and vs $or are different semantics, different hash."""
    leaves = [{"source_name": "linear"}, {"source_type": "ticket"}]
    assert _h({"$and": leaves}) != _h({"$or": leaves})
