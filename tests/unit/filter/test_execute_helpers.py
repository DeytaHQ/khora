"""Unit coverage for the AST leaf-inspection helpers in ``khora.filter.execute``.

These five helpers are the thin detectors an engine uses to compose a caller
filter across adaptive sub-searches:

* :func:`iter_leaf_clauses` — the one canonical leaf walk (AND/OR/NOT recurse).
* :func:`filter_leaf_keys` — dotted leaf keys, matching ``consumed_keys`` exactly.
* :func:`has_residual_metadata` — any leaf not in a compiler's consumed set.
* :func:`filter_constrains_date_key` — touches ``occurred_at`` / ``created_at``.
* :func:`caller_channel_constraint` — top-level ``metadata.channel`` pin, or None.

No DB, no infra — runs in the fast unit suite.
"""

from __future__ import annotations

import pytest

from khora.filter import RecallFilter, parse_to_ast
from khora.filter.compilers.cypher import compile_cypher
from khora.filter.execute import (
    build_compile_context,
    caller_channel_constraint,
    filter_constrains_date_key,
    filter_leaf_keys,
    has_residual_metadata,
    iter_leaf_clauses,
)

pytestmark = pytest.mark.unit


def _ast(doc: dict) -> object:
    """Lower a wire-form filter document to its canonical AST."""
    return parse_to_ast(RecallFilter.model_validate(doc))


# --------------------------------------------------------------------------- #
# H1 iter_leaf_clauses
# --------------------------------------------------------------------------- #


def test_iter_leaf_clauses_recurses_and_or_not() -> None:
    """Every leaf under nested AND/OR/NOT is yielded exactly once."""
    ast = _ast(
        {
            "source_name": "linear",
            "$or": [
                {"source_type": "issue"},
                {"$not": {"content_type": "comment"}},
            ],
        }
    )
    leaves = list(iter_leaf_clauses(ast))
    paths = sorted(".".join(leaf.path) for leaf in leaves)
    assert paths == ["content_type", "source_name", "source_type"]


def test_iter_leaf_clauses_single_predicate() -> None:
    """A single-predicate filter yields its one leaf (root is AND([clause]))."""
    leaves = list(iter_leaf_clauses(_ast({"source_name": "linear"})))
    assert len(leaves) == 1
    assert leaves[0].path == ("source_name",)


# --------------------------------------------------------------------------- #
# H2 filter_leaf_keys — must match CompiledFilter.consumed_keys exactly
# --------------------------------------------------------------------------- #


def test_filter_leaf_keys_dotted_form() -> None:
    """Leaf keys are the dotted ``".".join(path)`` of each leaf."""
    keys = filter_leaf_keys(
        _ast(
            {
                "source_name": "linear",
                "occurred_at": {"$gte": "2026-04-05"},
                "metadata.tag": {"$in": ["urgent"]},
            }
        )
    )
    assert keys == frozenset({"source_name", "occurred_at", "metadata.tag"})


def test_filter_leaf_keys_match_cypher_consumed_keys_for_system_slice() -> None:
    """For a system-key-only filter the leaf keys equal the Cypher consumed set."""
    ast = _ast({"source_name": "linear", "occurred_at": {"$gte": "2026-04-05"}})
    compiled = compile_cypher(ast, build_compile_context("Chunk", table_alias="c", on_unsupported="split"))
    assert filter_leaf_keys(ast) == compiled.consumed_keys


# --------------------------------------------------------------------------- #
# H3 has_residual_metadata
# --------------------------------------------------------------------------- #


def test_has_residual_metadata_true_for_metadata_leaf() -> None:
    """A metadata leaf is not Cypher-pushable, so it is residual."""
    ast = _ast({"source_name": "linear", "metadata.tag": "urgent"})
    consumed = compile_cypher(
        ast, build_compile_context("Chunk", table_alias="c", on_unsupported="split")
    ).consumed_keys
    assert has_residual_metadata(ast, consumed) is True


def test_has_residual_metadata_false_for_system_only() -> None:
    """A system-key-only filter pushes down entirely, so nothing is residual."""
    ast = _ast({"source_name": "linear", "occurred_at": {"$gte": "2026-04-05"}})
    consumed = compile_cypher(
        ast, build_compile_context("Chunk", table_alias="c", on_unsupported="split")
    ).consumed_keys
    assert has_residual_metadata(ast, consumed) is False


# --------------------------------------------------------------------------- #
# H4 filter_constrains_date_key
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "doc",
    [
        {"occurred_at": {"$gte": "2026-04-05"}},
        {"occurred_at": {"$lte": "2026-04-05"}},
        {"occurred_at": "2026-04-05"},
        {"created_at": {"$gte": "2026-04-05"}},
        {"occurred_at": {"$in": ["2026-04-05", "2026-04-06"]}},
    ],
    ids=["gte", "lte", "eq", "created_at", "in"],
)
def test_filter_constrains_date_key_true(doc: dict) -> None:
    """Any operator on occurred_at / created_at is detected."""
    assert filter_constrains_date_key(_ast(doc)) is True


@pytest.mark.parametrize(
    "doc",
    [
        {"metadata.channel": "alpha"},
        {"source_name": "linear"},
        {"source_timestamp": "2026-04-05"},
    ],
    ids=["metadata_only", "system_string_key", "source_timestamp_excluded"],
)
def test_filter_constrains_date_key_false(doc: dict) -> None:
    """Non-date keys (including source_timestamp, which is excluded) are not flagged."""
    assert filter_constrains_date_key(_ast(doc)) is False


# --------------------------------------------------------------------------- #
# H5 caller_channel_constraint
# --------------------------------------------------------------------------- #


def test_caller_channel_constraint_eq() -> None:
    """A top-level ``metadata.channel`` equality pins that one channel."""
    assert caller_channel_constraint(_ast({"metadata.channel": "alpha"})) == frozenset({"alpha"})


def test_caller_channel_constraint_in() -> None:
    """A top-level ``metadata.channel`` ``$in`` pins its string members."""
    assert caller_channel_constraint(_ast({"metadata.channel": {"$in": ["alpha", "beta"]}})) == frozenset(
        {"alpha", "beta"}
    )


def test_caller_channel_constraint_with_other_top_level_keys() -> None:
    """The channel pin is read alongside other top-level conjuncts."""
    assert caller_channel_constraint(_ast({"metadata.channel": "alpha", "source_name": "linear"})) == frozenset(
        {"alpha"}
    )


def test_caller_channel_constraint_buried_in_or_is_none() -> None:
    """A channel constraint inside an ``$or`` is not a hard AND-constraint."""
    ast = _ast({"$or": [{"metadata.channel": "alpha"}, {"source_name": "linear"}]})
    assert caller_channel_constraint(ast) is None


def test_caller_channel_constraint_buried_in_not_is_none() -> None:
    """A channel constraint inside a ``$not`` is not a hard AND-constraint."""
    assert caller_channel_constraint(_ast({"$not": {"metadata.channel": "alpha"}})) is None


def test_caller_channel_constraint_non_eq_in_op_is_none() -> None:
    """A non-equality / non-membership channel op yields None (conservative)."""
    assert caller_channel_constraint(_ast({"metadata.channel": {"$gt": "alpha"}})) is None


def test_caller_channel_constraint_no_channel_is_none() -> None:
    """A filter with no channel predicate yields None (no narrowing)."""
    assert caller_channel_constraint(_ast({"source_name": "linear"})) is None
