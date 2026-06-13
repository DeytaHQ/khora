"""Regression tests for multi-operator ``$not`` lowering (#1127) - ``@internal``.

A field-position ``$not`` over an operator-expression with **two or more**
operators (e.g. ``{"metadata.score": {"$not": {"$gte": 1, "$lte": 5}}}``) must
lower to a ``NOT`` node wrapping a SINGLE child - an ``AND`` of the inner
operator clauses - because ``$not({$gte:1, $lte:5})`` means
``NOT(field >= 1 AND field <= 5)`` (Mongo semantics). Before the fix the lowering
emitted ``FilterNode(op=NOT, children=(gte, lte))`` - a NOT with multiple children,
violating the AST's documented one-child-per-NOT invariant; every backend compiler
negates only ``children[0]``, silently dropping the rest.

These guard the lowering in ``khora.filter.ast`` across three executable surfaces:

* the AST shape itself (NOT carries exactly one child; the child is the AND);
* :func:`compile_python` behavior (the oracle every backend is checked against);
* the cross-backend conformance harness (python oracle + the Chronicle
  plan/run seam), so a regression that drops a predicate fails an executable
  survivor assertion, not only a structural one.
"""

from __future__ import annotations

import pytest

from khora.filter import RecallFilter, parse_to_ast
from khora.filter.ast import FilterClause, FilterNode, Op
from khora.filter.compilers.python import compile_python
from khora.filter.conformance import (
    ChronicleExecutor,
    ConformanceCase,
    PythonExecutor,
    SeedRecord,
    assert_case,
)
from khora.filter.execute import build_compile_context

pytestmark = pytest.mark.unit


def _ast(doc: dict) -> FilterNode:
    return parse_to_ast(RecallFilter.model_validate(doc))


def _predicate(doc: dict):
    ctx = build_compile_context("chunks", on_unsupported="raise")
    return compile_python(_ast(doc), ctx).predicate


def _find_not(node: FilterNode | FilterClause) -> FilterNode:
    """Return the single NOT node in a lowered AST (depth-first)."""
    if isinstance(node, FilterNode):
        if node.op == Op.NOT:
            return node
        for child in node.children:
            found = _find_not(child)
            if found is not None:
                return found
    return None  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# AST shape: NOT over 2+ operators wraps a single AND child.
# --------------------------------------------------------------------------- #


def test_metadata_not_two_operators_wraps_single_and_child() -> None:
    not_node = _find_not(_ast({"metadata.score": {"$not": {"$gte": 1, "$lte": 5}}}))
    assert not_node.op == Op.NOT
    assert len(not_node.children) == 1, "NOT must carry exactly one child"

    (child,) = not_node.children
    assert isinstance(child, FilterNode)
    assert child.op == Op.AND
    assert len(child.children) == 2
    assert {c.op for c in child.children} == {Op.GTE, Op.LTE}


def test_metadata_not_single_operator_keeps_clause_child() -> None:
    # Single inner operator stays NOT(clause) - no spurious AND wrapper.
    not_node = _find_not(_ast({"metadata.score": {"$not": {"$gte": 1}}}))
    assert len(not_node.children) == 1
    (child,) = not_node.children
    assert isinstance(child, FilterClause)
    assert child.op == Op.GTE


def test_system_key_dateops_not_two_operators_wraps_single_and_child() -> None:
    # The _lower_ops_submodel path (DateOps system key) must also produce one child.
    doc = {"source_timestamp": {"$not": {"$gte": "2026-01-01T00:00:00Z", "$lte": "2026-12-31T00:00:00Z"}}}
    not_node = _find_not(_ast(doc))
    assert len(not_node.children) == 1
    (child,) = not_node.children
    assert isinstance(child, FilterNode)
    assert child.op == Op.AND
    assert len(child.children) == 2


def test_nested_not_in_negated_expression_wraps_single_and_child() -> None:
    # The _lower_metadata_operator_expr path: a $not nested inside a negated
    # operator-expression whose inner $not has 2+ operators.
    doc = {"metadata.score": {"$not": {"$not": {"$gte": 1, "$lte": 5}}}}
    outer = _find_not(_ast(doc))
    assert len(outer.children) == 1
    (inner,) = outer.children
    assert isinstance(inner, FilterNode) and inner.op == Op.NOT
    assert len(inner.children) == 1, "the nested NOT must also carry exactly one child"
    (and_node,) = inner.children
    assert isinstance(and_node, FilterNode) and and_node.op == Op.AND
    assert len(and_node.children) == 2


# --------------------------------------------------------------------------- #
# compile_python behavior: NOT(gte AND lte) keeps both predicates.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (10, True),  # NOT(10>=1 AND 10<=5) == NOT(True AND False) == True
        (0, True),  # NOT(0>=1 AND 0<=5)   == NOT(False AND True) == True
        (3, False),  # NOT(3>=1 AND 3<=5)  == NOT(True AND True)  == False
    ],
)
def test_compile_python_not_two_operators_negates_full_and(score: int, expected: bool) -> None:
    predicate = _predicate({"metadata.score": {"$not": {"$gte": 1, "$lte": 5}}})
    assert predicate({"metadata": {"score": score}}) is expected


# --------------------------------------------------------------------------- #
# Cross-backend conformance: python oracle + Chronicle plan/run seam.
# --------------------------------------------------------------------------- #


_NOT_RANGE_CASE = ConformanceCase(
    id="NOT-multi-operator-range",
    filter={"metadata.score": {"$not": {"$gte": 1, "$lte": 5}}},
    seed_records=(
        SeedRecord(id="above", metadata={"score": 10}),  # outside [1,5] -> kept
        SeedRecord(id="below", metadata={"score": 0}),  # outside [1,5] -> kept
        SeedRecord(id="inside", metadata={"score": 3}),  # inside [1,5]  -> dropped
    ),
    expected_ids=frozenset({"above", "below"}),
    backends=frozenset({"python", "chronicle"}),
    exercises=("NOT", "metadata.score", "$not", "$gte", "$lte"),
)


def test_conformance_python_oracle_not_multi_operator() -> None:
    assert_case(_NOT_RANGE_CASE, "python", PythonExecutor())


def test_conformance_chronicle_not_multi_operator() -> None:
    assert_case(_NOT_RANGE_CASE, "chronicle", ChronicleExecutor())
