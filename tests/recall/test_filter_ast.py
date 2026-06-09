"""Unit tests for the canonical filter AST (Layer 3) — ``@internal``.

Pins the two lowering invariants the rest of the subsystem depends on:
compilers never see dot-strings or bare-value/logical sugar. ``parse_to_ast``
takes a *validated* ``RecallFilter`` and emits a ``FilterNode`` whose paths are
segment tuples and whose sugar is fully desugared. ``canonical_hash`` derives a
stable cache key with documented order semantics.

A first cut — QA expands coverage after this.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from khora.filter import RecallFilter
from khora.filter.ast import (
    DateLiteral,
    FilterClause,
    FilterNode,
    FilterOp,
    canonical_hash,
    parse_to_ast,
)
from khora.filter.model import Op

pytestmark = pytest.mark.unit


def _ast(wire: dict) -> FilterNode | FilterClause:
    return parse_to_ast(RecallFilter.model_validate(wire))


def _clauses(node: FilterNode | FilterClause) -> list[FilterClause]:
    """Flatten the immediate clause children of an AST root (non-recursive).

    ``parse_to_ast`` normalizes a single bare predicate down to a lone
    :class:`FilterClause` (the single-child ``AND`` wrapper collapses), so the
    root may itself be a clause. Treat that as a one-clause list.
    """
    if isinstance(node, FilterClause):
        return [node]
    return [c for c in node.children if isinstance(c, FilterClause)]


# ---------------------------------------------------------------------------
# FilterOp is the model's Op (single source of truth, no parallel enum).
# ---------------------------------------------------------------------------


def test_filterop_is_model_op() -> None:
    assert FilterOp is Op
    assert FilterOp.EQ == "$eq"


# ---------------------------------------------------------------------------
# Paths are always segment tuples — never dot-strings.
# ---------------------------------------------------------------------------


def test_system_key_lowers_to_single_segment_path() -> None:
    ast = _ast({"source_name": "linear"})
    (clause,) = _clauses(ast)
    assert clause.path == ("source_name",)
    assert clause.op == Op.EQ
    assert clause.operand == "linear"


def test_metadata_dot_path_lowers_to_multi_segment_path() -> None:
    ast = _ast({"metadata.labels.tier": {"$gte": 5}})
    (clause,) = _clauses(ast)
    # ("metadata", "labels", "tier") — the leading "metadata" root is kept.
    assert clause.path == ("metadata", "labels", "tier")
    assert clause.op == Op.GTE
    assert clause.operand == 5


def test_no_path_anywhere_is_a_dot_string() -> None:
    ast = _ast(
        {
            "source_name": "linear",
            "metadata.a.b": "x",
            "$or": [{"metadata.c": 1}, {"title": "t"}],
        }
    )
    for clause in _walk_clauses(ast):
        assert isinstance(clause.path, tuple)
        assert all("." not in segment for segment in clause.path)


def test_bare_metadata_blob_is_single_metadata_segment() -> None:
    ast = _ast({"metadata": {"x": 1, "y": 2}})
    (clause,) = _clauses(ast)
    assert clause.path == ("metadata",)
    assert clause.op == Op.EQ
    assert clause.operand == {"x": 1, "y": 2}


# ---------------------------------------------------------------------------
# AC1 — a filter mixing metadata dot-paths + bare sugar + $nor lowers to an AST
# with SEGMENT-LIST paths, explicit $eq/$or/$not, NO surviving NOR node, and NO
# dot-string anywhere. Asserted by walking the whole tree.
# ---------------------------------------------------------------------------


def test_ac1_mixed_filter_lowers_to_fully_desugared_segment_path_ast() -> None:
    ast = _ast(
        {
            # bare-scalar sugar -> $eq, single-segment path
            "source_name": "linear",
            # metadata dot-path -> multi-segment path, explicit operator
            "metadata.labels.tier": {"$gte": 5},
            # $nor -> $not($or(...)); NO NOR node may survive
            "$nor": [{"title": "draft"}, {"source_type": "spam"}],
        }
    )

    # Several siblings on one document -> explicit AND root.
    assert isinstance(ast, FilterNode)
    assert ast.op == Op.AND

    nodes = _walk_nodes(ast)
    clauses = _walk_clauses(ast)

    # No NOR node survives; every logical node is AND / OR / NOT only.
    assert all(n.op != Op.NOR for n in nodes)
    assert {n.op for n in nodes} <= {Op.AND, Op.OR, Op.NOT}
    # The $nor lowered to a NOT wrapping an OR (both present in the tree).
    assert any(n.op == Op.NOT for n in nodes)
    assert any(n.op == Op.OR for n in nodes)

    # Every path is a SEGMENT tuple — no dot-strings, no "." inside any segment.
    for clause in clauses:
        assert isinstance(clause.path, tuple)
        assert all(isinstance(seg, str) and "." not in seg for seg in clause.path)
    # The metadata predicate produced its multi-segment path.
    assert ("metadata", "labels", "tier") in {c.path for c in clauses}
    # The bare scalar produced an EXPLICIT $eq clause (sugar fully desugared).
    assert any(c.path == ("source_name",) and c.op == Op.EQ for c in clauses)
    # Every leaf op is a concrete comparison op (never a logical op on a leaf).
    assert all(c.op in {Op.EQ, Op.NE, Op.GT, Op.GTE, Op.LT, Op.LTE, Op.IN, Op.NIN, Op.EXISTS} for c in clauses)


# ---------------------------------------------------------------------------
# Sugar is fully desugared — no bare value / logical sugar reaches the AST.
# ---------------------------------------------------------------------------


def test_bare_scalar_lowers_to_eq() -> None:
    (clause,) = _clauses(_ast({"title": "Q2 plan"}))
    assert clause.op == Op.EQ
    assert clause.operand == "Q2 plan"


def test_bare_list_is_eq_exact_array_not_in() -> None:
    # A bare list is $eq EXACT-ARRAY equality, NOT $in. The operand is an
    # order-preserving tuple, and the op is $eq (membership is the explicit $in).
    (clause,) = _clauses(_ast({"source_type": ["a", "b"]}))
    assert clause.op == Op.EQ
    assert clause.operand == ("a", "b")
    assert clause.op != Op.IN


def test_bare_metadata_list_is_eq_exact_array_not_in() -> None:
    # Same uniform bare-list rule on a folded metadata sub-path: $eq exact-array,
    # NOT $in. The operand is an order-preserving tuple.
    (clause,) = _clauses(_ast({"metadata.tags": ["x", "y", "z"]}))
    assert clause.path == ("metadata", "tags")
    assert clause.op == Op.EQ
    assert clause.operand == ("x", "y", "z")
    assert isinstance(clause.operand, tuple)


def test_explicit_in_stays_in_with_ordered_tuple() -> None:
    (clause,) = _clauses(_ast({"source_name": {"$in": ["a", "b"]}}))
    assert clause.op == Op.IN
    assert clause.operand == ("a", "b")


def test_metadata_explicit_in_stays_in_with_ordered_tuple() -> None:
    (clause,) = _clauses(_ast({"metadata.k": {"$in": ["a", "b", "c"]}}))
    assert clause.op == Op.IN
    assert clause.operand == ("a", "b", "c")
    assert isinstance(clause.operand, tuple)


def test_explicit_eq_list_operand_stays_a_plain_list_not_a_tuple() -> None:
    # An EXPLICIT {"$eq": [list]} is array-CONTAINMENT, not the bare-list exact-array
    # sugar. It lowers to a plain ``list`` operand (carried verbatim by
    # ``_lower_scalar_operand``) — distinct from the bare-list form, which lowers to
    # a ``tuple``. This container-type split is what ``canonical_hash`` records
    # differently via the structural ``operand_kind`` field on the clause record.
    (clause,) = _clauses(_ast({"metadata.tags": {"$eq": ["a", "b"]}}))
    assert clause.path == ("metadata", "tags")
    assert clause.op == Op.EQ
    assert clause.operand == ["a", "b"]
    assert isinstance(clause.operand, list)
    assert not isinstance(clause.operand, tuple)


def test_metadata_nin_lowers_to_ordered_tuple() -> None:
    (clause,) = _clauses(_ast({"metadata.k": {"$nin": [1, 2]}}))
    assert clause.op == Op.NIN
    assert clause.operand == (1, 2)
    assert isinstance(clause.operand, tuple)


def test_metadata_exists_operand_is_bool() -> None:
    (clause,) = _clauses(_ast({"metadata.k": {"$exists": True}}))
    assert clause.op == Op.EXISTS
    assert clause.operand is True


def test_nor_desugars_to_not_or() -> None:
    ast = _ast({"$nor": [{"source_name": "x"}, {"title": "y"}]})
    assert ast.op == Op.NOT
    (inner,) = ast.children
    assert isinstance(inner, FilterNode)
    assert inner.op == Op.OR
    assert len(inner.children) == 2


def test_no_nor_node_ever_survives_into_ast() -> None:
    # $nor desugars to $not($or(...)) at build time — no NOR node may reach the
    # compilers. The AST's only logical node ops are AND / OR / NOT (scope-creep
    # guard: a future refactor must never emit a NOR node).
    ast = _ast(
        {
            "$nor": [{"source_name": "a"}, {"title": "b"}],
            "$and": [{"$nor": [{"source": "c"}]}],
        }
    )
    for node in _walk_nodes(ast):
        assert node.op != Op.NOR
        assert node.op in {Op.AND, Op.OR, Op.NOT}


def test_not_of_not_is_preserved_not_collapsed() -> None:
    # A doubly-nested $not must stay two NOT nodes — never collapsed to identity
    # (no missing/extra negation).
    ast = _ast({"$not": {"$not": {"source_name": "x"}}})
    assert ast.op == Op.NOT
    (inner,) = ast.children
    assert isinstance(inner, FilterNode)
    assert inner.op == Op.NOT


def test_document_level_not_is_a_not_node() -> None:
    # Document-form $not negates the lowered inner filter. The inner single-key
    # filter normalizes to AND([clause]) (the root is always a FilterNode — a
    # single leaf is wrapped, never collapsed to a bare clause), so the NOT wraps
    # that AND node.
    ast = _ast({"$not": {"source_name": "x"}})
    assert isinstance(ast, FilterNode)
    assert ast.op == Op.NOT
    (inner,) = ast.children
    assert isinstance(inner, FilterNode)
    assert inner.op == Op.AND
    (clause,) = inner.children
    assert isinstance(clause, FilterClause)
    assert clause.path == ("source_name",)
    assert clause.op == Op.EQ
    assert clause.operand == "x"


def test_document_level_not_over_multikey_inner_wraps_and_node() -> None:
    # When the inner filter has several siblings it stays an AND node, so the NOT
    # wraps a FilterNode (the collapse only applies to a single child).
    ast = _ast({"$not": {"source_name": "x", "source_type": "y"}})
    assert isinstance(ast, FilterNode)
    assert ast.op == Op.NOT
    (inner,) = ast.children
    assert isinstance(inner, FilterNode)
    assert inner.op == Op.AND
    assert {c.path for c in _clauses(inner)} == {("source_name",), ("source_type",)}


def test_field_level_not_is_a_not_node() -> None:
    ast = _ast({"source_name": {"$not": {"$eq": "x"}}})
    assert ast.op == Op.NOT
    (clause,) = ast.children
    assert isinstance(clause, FilterClause)
    assert clause.path == ("source_name",)
    assert clause.op == Op.EQ


def test_metadata_field_level_not_is_a_not_node() -> None:
    ast = _ast({"metadata.k": {"$not": {"$gte": 5}}})
    assert ast.op == Op.NOT
    (clause,) = ast.children
    assert isinstance(clause, FilterClause)
    assert clause.op == Op.GTE
    assert clause.operand == 5


def test_field_level_not_and_document_level_not_both_produce_not_nodes() -> None:
    # A document-level $not whose inner filter carries a field-level $not yields
    # two NOT nodes (one per negation): document NOT wrapping the lowered inner
    # filter, which itself contains the field-level NOT.
    ast = _ast({"$not": {"source_name": {"$not": {"$eq": "x"}}}})
    assert ast.op == Op.NOT
    not_nodes = [n for n in _walk_nodes(ast) if n.op == Op.NOT]
    assert len(not_nodes) == 2
    # The single leaf survives with its concrete op.
    (clause,) = _walk_clauses(ast)
    assert clause.path == ("source_name",)
    assert clause.op == Op.EQ
    assert clause.operand == "x"


def test_nested_metadata_not_lowers_to_nested_not_nodes_no_clause_carries_not() -> None:
    # Regression: a $not nested inside a field-position $not on a metadata
    # path must lower to NESTED NOT *FilterNodes*, never to a malformed
    # FilterClause(op=NOT, operand=<raw dict>). Guard the invariant directly: no
    # FilterClause anywhere in the tree carries a logical op, and the two
    # negations produced exactly two NOT nodes.
    ast = _ast({"metadata.k": {"$not": {"$not": {"$gte": 5}}}})
    not_nodes = [n for n in _walk_nodes(ast) if n.op == Op.NOT]
    assert len(not_nodes) == 2
    # Every leaf is a real comparison clause — NONE carries a logical op.
    leaves = _walk_clauses(ast)
    assert leaves, "expected at least one leaf clause"
    assert all(c.op not in {Op.NOT, Op.AND, Op.OR, Op.NOR} for c in leaves)
    # The surviving comparison is the inner $gte, carried verbatim.
    (leaf,) = leaves
    assert leaf.path == ("metadata", "k")
    assert leaf.op == Op.GTE
    assert leaf.operand == 5


def test_no_filterclause_anywhere_carries_a_logical_op() -> None:
    # Broader scope-creep guard across a filter that exercises every $not form
    # plus $nor: logical operators live ONLY on FilterNodes, never on a leaf.
    ast = _ast(
        {
            "source_name": {"$not": {"$eq": "a"}},
            "metadata.k": {"$not": {"$in": [1, 2]}},
            "$nor": [{"title": "t"}, {"source_type": "x"}],
            "$not": {"content_type": "c"},
        }
    )
    for clause in _walk_clauses(ast):
        assert clause.op not in {Op.NOT, Op.AND, Op.OR, Op.NOR}
    for node in _walk_nodes(ast):
        assert node.op in {Op.AND, Op.OR, Op.NOT}


def test_implicit_sibling_and_becomes_explicit_and_node() -> None:
    # Two sibling system keys on one document = implicit AND -> explicit AND.
    ast = _ast({"source_name": "linear", "source_type": "issue"})
    assert ast.op == Op.AND
    paths = {c.path for c in _clauses(ast)}
    assert paths == {("source_name",), ("source_type",)}


def test_single_predicate_filter_wraps_in_and_node_root() -> None:
    # The root is ALWAYS a FilterNode (the engine boundary contract: a compiler
    # is Callable[[FilterNode, CompileContext], ...]). A single bare predicate
    # normalizes to AND([clause]) — wrapped, never collapsed to a bare clause.
    ast = _ast({"source_name": "linear"})
    assert isinstance(ast, FilterNode)
    assert ast.op == Op.AND
    (clause,) = ast.children
    assert isinstance(clause, FilterClause)
    assert clause.path == ("source_name",)
    assert clause.op == Op.EQ
    assert clause.operand == "linear"


def test_single_element_and_or_and_bare_converge_on_same_and_root() -> None:
    # {a}, {$and:[{a}]}, and {$or:[{a}]} all normalize to AND([a]) — identical
    # structure, identical hash. A single-element $and/$or wrapping a leaf is
    # rewritten to AND([leaf]) (op normalized to AND), not collapsed to a clause.
    bare = _ast({"source_name": "linear"})
    single_and = _ast({"$and": [{"source_name": "linear"}]})
    single_or = _ast({"$or": [{"source_name": "linear"}]})
    for ast in (bare, single_and, single_or):
        assert isinstance(ast, FilterNode)
        assert ast.op == Op.AND
        (clause,) = ast.children
        assert isinstance(clause, FilterClause)
        assert clause.path == ("source_name",)
    assert canonical_hash(bare) == canonical_hash(single_and) == canonical_hash(single_or)


def test_empty_filter_lowers_to_empty_and_match_everything() -> None:
    # A bare RecallFilter() (no predicates) lowers to an empty AND node — a
    # match-everything root. Compilers treat an empty AND as "no constraint".
    ast = parse_to_ast(RecallFilter())
    assert ast.op == Op.AND
    assert ast.children == ()


def test_lone_logical_node_is_not_double_wrapped() -> None:
    # A document carrying a single logical operator returns that node directly,
    # not wrapped in a redundant single-child AND.
    ast = _ast({"$or": [{"source_name": "a"}, {"source_type": "b"}]})
    assert ast.op == Op.OR
    assert len(ast.children) == 2


# ---------------------------------------------------------------------------
# Normalization: same-operator flattening (associativity) + structural
# convergence of equivalent shapes.
# ---------------------------------------------------------------------------


def test_nested_same_operator_and_is_flattened() -> None:
    # AND-of-AND splices grandchildren up one level (associativity). The result
    # is a single flat AND with all three leaves.
    ast = _ast({"$and": [{"$and": [{"source_name": "a"}, {"source_type": "b"}]}, {"title": "t"}]})
    assert isinstance(ast, FilterNode)
    assert ast.op == Op.AND
    # No AND child remains nested under the AND root (fully flattened).
    assert all(not (isinstance(c, FilterNode) and c.op == Op.AND) for c in ast.children)
    assert {c.path for c in _clauses(ast)} == {("source_name",), ("source_type",), ("title",)}


def test_nested_same_operator_or_is_flattened() -> None:
    # OR-of-OR splices the inner OR up one level (associativity). Each branch
    # leaf is wrapped as AND([leaf]) (single-child leaf wrap), so the flattened
    # OR root has three AND([leaf]) children and NO nested OR remains.
    ast = _ast({"$or": [{"$or": [{"source_name": "a"}, {"source_type": "b"}]}, {"title": "t"}]})
    assert isinstance(ast, FilterNode)
    assert ast.op == Op.OR
    # No OR child remains nested under the OR root (fully flattened).
    assert all(not (isinstance(c, FilterNode) and c.op == Op.OR) for c in ast.children)
    assert len(ast.children) == 3
    # All three leaves are present across the (AND-wrapped) branches.
    assert {c.path for c in _walk_clauses(ast)} == {("source_name",), ("source_type",), ("title",)}


def test_or_child_of_and_is_not_flattened() -> None:
    # Flattening is per-operator: an OR nested in an AND is left intact (never
    # merged across operators).
    ast = _ast({"source_name": "a", "$or": [{"source_type": "b"}, {"title": "t"}]})
    assert isinstance(ast, FilterNode)
    assert ast.op == Op.AND
    or_children = [c for c in ast.children if isinstance(c, FilterNode) and c.op == Op.OR]
    assert len(or_children) == 1


def test_implicit_and_and_explicit_and_converge_on_same_structure() -> None:
    # {a, b} (implicit AND) and {$and:[{a},{b}]} (explicit) normalize to the same
    # AND([a, b]) — identical structure, identical hash.
    implicit = canonical_hash(_ast({"source_name": "a", "source_type": "b"}))
    explicit = canonical_hash(_ast({"$and": [{"source_name": "a"}, {"source_type": "b"}]}))
    assert implicit == explicit


def test_not_is_an_opaque_flattening_boundary() -> None:
    # An AND inside a NOT is never spliced into an outer AND — NOT is opaque to
    # flattening. The inner AND survives as a node under the NOT.
    ast = _ast({"source_name": "a", "$not": {"source_type": "b", "title": "t"}})
    assert isinstance(ast, FilterNode)
    assert ast.op == Op.AND
    not_nodes = [c for c in ast.children if isinstance(c, FilterNode) and c.op == Op.NOT]
    assert len(not_nodes) == 1
    (inner,) = not_nodes[0].children
    assert isinstance(inner, FilterNode)
    assert inner.op == Op.AND


# ---------------------------------------------------------------------------
# Operands are opaque; $date literals lower to DateLiteral.
# ---------------------------------------------------------------------------


def test_date_literal_operand_lowers_to_dateliteral() -> None:
    (clause,) = _clauses(_ast({"metadata.when": {"$gte": {"$date": "2026-01-01T00:00:00Z"}}}))
    assert isinstance(clause.operand, DateLiteral)
    assert clause.operand.value == datetime(2026, 1, 1, tzinfo=UTC)


def test_bare_date_literal_in_value_position_lowers_to_eq_dateliteral() -> None:
    (clause,) = _clauses(_ast({"metadata.when": {"$date": "2026-01-01"}}))
    assert clause.op == Op.EQ
    assert isinstance(clause.operand, DateLiteral)
    assert clause.operand.value == datetime(2026, 1, 1, tzinfo=UTC)


def test_comparison_dict_operand_is_opaque_not_recursed() -> None:
    # A dict nested inside an $eq operand is matched as a literal object, NOT
    # recursed into clauses (mirrors model._check_literal_operand).
    operand = {"$or": [{"a": 1}]}
    (clause,) = _clauses(_ast({"metadata.k": {"$eq": operand}}))
    assert clause.op == Op.EQ
    assert clause.operand == operand
    assert isinstance(clause.operand, dict)


def test_logical_op_nested_in_eq_operand_stays_literal_data() -> None:
    # {"$eq": {"$or": [1, 2]}} — the $or is DATA inside an opaque equality
    # operand, NOT a logical node. The AST carries it verbatim as a dict, with
    # exactly one leaf clause and no OR node synthesized anywhere.
    ast = _ast({"metadata.k": {"$eq": {"$or": [1, 2]}}})
    (clause,) = _walk_clauses(ast)
    assert clause.op == Op.EQ
    assert clause.operand == {"$or": [1, 2]}
    assert isinstance(clause.operand, dict)
    assert all(n.op != Op.OR for n in _walk_nodes(ast))


def test_date_literal_is_an_operand_not_a_clause() -> None:
    # {"$date": ...} is a typed-literal OPERAND, not a logical/comparison clause:
    # the leaf op is $eq and the operand carries the DateLiteral. The $date key
    # never becomes its own node or op.
    (clause,) = _clauses(_ast({"metadata.when": {"$date": "2026-01-01"}}))
    assert isinstance(clause, FilterClause)
    assert clause.op == Op.EQ
    assert clause.op != Op.DATE
    assert isinstance(clause.operand, DateLiteral)


def test_mixed_key_date_dict_operand_stays_opaque() -> None:
    # {"$date": ..., "other": ...} is NOT a sole-key $date literal, so it stays
    # an opaque dict — never lowered to a DateLiteral.
    operand = {"$date": "2026-01-01", "other": "field"}
    (clause,) = _clauses(_ast({"metadata.nested": {"$eq": operand}}))
    assert clause.op == Op.EQ
    assert clause.operand == operand
    assert isinstance(clause.operand, dict)
    assert not isinstance(clause.operand, DateLiteral)


def test_in_list_lowers_date_elements_keeps_others_opaque() -> None:
    # Per-element: a sole-key $date item -> DateLiteral; a non-$date dict item
    # ($or here) stays opaque; a scalar stays verbatim. Order preserved.
    (clause,) = _clauses(_ast({"metadata.tags": {"$in": [{"$date": "2026-01-01"}, "plain", {"$or": [1]}]}}))
    assert clause.op == Op.IN
    a, b, c = clause.operand
    assert isinstance(a, DateLiteral)
    assert b == "plain"
    assert c == {"$or": [1]}


def test_whole_subdocument_equality_operand_is_opaque() -> None:
    (clause,) = _clauses(_ast({"metadata.obj": {"a": 1, "b": 2}}))
    assert clause.op == Op.EQ
    assert clause.operand == {"a": 1, "b": 2}


def test_empty_subdocument_lowers_to_eq_empty_dict() -> None:
    (clause,) = _clauses(_ast({"metadata.obj": {}}))
    assert clause.op == Op.EQ
    assert clause.operand == {}


def test_bare_blob_and_metadata_subpath_lower_distinctly() -> None:
    # The bare ``metadata`` blob is a single ("metadata",) whole-blob $eq; a
    # ``metadata.obj`` sub-path whole-subdoc equality is a ("metadata", "obj")
    # leaf. The two forms lower to DIFFERENT paths even with the same operand.
    (blob_clause,) = _clauses(_ast({"metadata": {"a": 1}}))
    (sub_clause,) = _clauses(_ast({"metadata.obj": {"a": 1}}))
    assert blob_clause.path == ("metadata",)
    assert sub_clause.path == ("metadata", "obj")
    assert blob_clause.op == Op.EQ == sub_clause.op
    # Same operand DATA, distinct paths -> distinct hashes.
    assert canonical_hash(_ast({"metadata": {"a": 1}})) != canonical_hash(_ast({"metadata.obj": {"a": 1}}))


# ---------------------------------------------------------------------------
# Explicit null is an active match (not dropped).
# ---------------------------------------------------------------------------


def test_explicit_null_lowers_to_eq_none() -> None:
    (clause,) = _clauses(_ast({"source_name": None}))
    assert clause.op == Op.EQ
    assert clause.operand is None


def test_unset_key_does_not_appear() -> None:
    ast = _ast({"source_type": "x"})
    paths = {c.path for c in _clauses(ast)}
    assert ("source_name",) not in paths
    assert paths == {("source_type",)}


# ---------------------------------------------------------------------------
# Date keys normalize to UTC at lowering (parity with DateOps).
# ---------------------------------------------------------------------------


def test_bare_date_scalar_normalized_to_utc() -> None:
    (clause,) = _clauses(_ast({"occurred_at": "2026-04-05T00:00:00"}))
    assert isinstance(clause.operand, datetime)
    assert clause.operand.tzinfo is not None
    assert clause.operand == datetime(2026, 4, 5, tzinfo=UTC)


def test_offset_aware_date_scalar_normalized_to_utc() -> None:
    # A non-UTC offset is converted to UTC at lowering, matching DateOps.
    (clause,) = _clauses(_ast({"occurred_at": "2026-04-05T02:00:00+02:00"}))
    assert isinstance(clause.operand, datetime)
    assert clause.operand == datetime(2026, 4, 5, 0, 0, 0, tzinfo=UTC)


def test_date_ops_range_lowers_per_operator() -> None:
    ast = _ast({"occurred_at": {"$gte": "2026-01-01T00:00:00Z", "$lt": "2026-02-01T00:00:00Z"}})
    by_op = {c.op: c for c in _clauses(ast)}
    assert set(by_op) == {Op.GTE, Op.LT}
    assert by_op[Op.GTE].operand == datetime(2026, 1, 1, tzinfo=UTC)
    assert by_op[Op.LT].operand == datetime(2026, 2, 1, tzinfo=UTC)


# ---------------------------------------------------------------------------
# canonical_hash — stable hex digest with documented order semantics.
# ---------------------------------------------------------------------------


def test_hash_is_stable_hex_string() -> None:
    h = canonical_hash(_ast({"source_name": "linear"}))
    assert isinstance(h, str)
    assert len(h) == 64  # SHA-256 hex
    int(h, 16)  # parses as hex


def test_hash_deterministic_across_calls() -> None:
    ast = _ast({"source_name": "linear", "occurred_at": {"$gte": "2026-04-05T00:00:00Z"}})
    assert canonical_hash(ast) == canonical_hash(ast)


def test_hash_commutative_and_is_order_insensitive() -> None:
    a = canonical_hash(_ast({"$and": [{"source_name": "a"}, {"source_type": "b"}]}))
    b = canonical_hash(_ast({"$and": [{"source_type": "b"}, {"source_name": "a"}]}))
    assert a == b


def test_hash_commutative_or_is_order_insensitive() -> None:
    a = canonical_hash(_ast({"$or": [{"source_name": "a"}, {"source_type": "b"}]}))
    b = canonical_hash(_ast({"$or": [{"source_type": "b"}, {"source_name": "a"}]}))
    assert a == b


def test_hash_in_list_is_order_sensitive() -> None:
    a = canonical_hash(_ast({"source_name": {"$in": ["a", "b"]}}))
    b = canonical_hash(_ast({"source_name": {"$in": ["b", "a"]}}))
    assert a != b


def test_hash_exact_array_operand_is_order_sensitive() -> None:
    a = canonical_hash(_ast({"source_type": ["a", "b"]}))
    b = canonical_hash(_ast({"source_type": ["b", "a"]}))
    assert a != b


def test_hash_dict_operand_key_order_insensitive() -> None:
    a = canonical_hash(_ast({"metadata.o": {"a": 1, "b": 2}}))
    b = canonical_hash(_ast({"metadata.o": {"b": 2, "a": 1}}))
    assert a == b


def test_hash_whole_blob_metadata_dict_key_order_insensitive() -> None:
    # The bare ``metadata`` WHOLE-BLOB $eq operand (a different lowering path than a
    # ``metadata.<sub>`` subdocument) also sorts its keys, so the wire dict-key order
    # on the blob does not change the hash.
    a = canonical_hash(_ast({"metadata": {"a": 1, "b": 2}}))
    b = canonical_hash(_ast({"metadata": {"b": 2, "a": 1}}))
    assert a == b


def test_hash_differs_on_operand_semantics() -> None:
    a = canonical_hash(_ast({"source_name": "a"}))
    b = canonical_hash(_ast({"source_name": "b"}))
    assert a != b


def test_hash_differs_on_operator_semantics() -> None:
    a = canonical_hash(_ast({"source_name": {"$eq": "a"}}))
    b = canonical_hash(_ast({"source_name": {"$ne": "a"}}))
    assert a != b


def test_hash_differs_on_path_semantics() -> None:
    a = canonical_hash(_ast({"metadata.a": 1}))
    b = canonical_hash(_ast({"metadata.b": 1}))
    assert a != b


def test_hash_distinguishes_in_from_exact_array() -> None:
    # $in ["a","b"] and a bare list ["a","b"] ($eq exact-array) are different
    # semantics and must hash differently.
    in_hash = canonical_hash(_ast({"source_name": {"$in": ["a", "b"]}}))
    eq_hash = canonical_hash(_ast({"source_type": ["a", "b"]}))
    # different paths AND different ops — definitely distinct.
    assert in_hash != eq_hash


# --- AC3: identical for sibling-order-only differences (both directions) --- #


def test_hash_nested_commutative_sibling_order_insensitive() -> None:
    # Order-insensitivity must hold at depth, not only at the root.
    a = canonical_hash(_ast({"$or": [{"$and": [{"source_name": "a"}, {"title": "t"}]}, {"source_type": "b"}]}))
    b = canonical_hash(_ast({"$or": [{"source_type": "b"}, {"$and": [{"title": "t"}, {"source_name": "a"}]}]}))
    assert a == b


def test_hash_implicit_sibling_and_order_insensitive() -> None:
    # Implicit sibling AND (multiple keys on one document) is also commutative —
    # the dict key order on the wire must not affect the hash.
    a = canonical_hash(_ast({"source_name": "a", "source_type": "b"}))
    b = canonical_hash(_ast({"source_type": "b", "source_name": "a"}))
    assert a == b


# --- AC3: different for semantically-different filters --------------------- #


def test_hash_differs_and_vs_or() -> None:
    # Same children, different logical operator -> different semantics.
    a = canonical_hash(_ast({"$and": [{"source_name": "a"}, {"source_type": "b"}]}))
    b = canonical_hash(_ast({"$or": [{"source_name": "a"}, {"source_type": "b"}]}))
    assert a != b


def test_hash_distinguishes_eq_list_operand_from_in_same_path() -> None:
    # A bare list on a key is $eq EXACT-ARRAY; $in is membership. SAME path, SAME
    # list contents/order — they must STILL hash differently (different op).
    eq_hash = canonical_hash(_ast({"source_name": ["a", "b"]}))
    in_hash = canonical_hash(_ast({"source_name": {"$in": ["a", "b"]}}))
    assert eq_hash != in_hash


def test_bare_list_and_explicit_eq_list_hash_differently() -> None:
    # Headline: a bare list ``{path: [a, b]}`` is $eq EXACT-ARRAY (tuple operand),
    # while an explicit ``{path: {"$eq": [a, b]}}`` is array-CONTAINMENT (list
    # operand). SAME path, SAME op ($eq), SAME elements/order — only the operand's
    # container kind differs, and the two carry different row-set semantics, so they
    # must hash DIFFERENTLY. The canonical clause record carries a structural
    # ``operand_kind`` field (tuple vs list) that breaks the collision.
    bare_hash = canonical_hash(_ast({"metadata.tags": ["a", "b"]}))
    explicit_eq_hash = canonical_hash(_ast({"metadata.tags": {"$eq": ["a", "b"]}}))
    assert bare_hash != explicit_eq_hash


def test_bare_list_does_not_collide_with_dict_operand_mimicking_the_tag() -> None:
    # The tuple-vs-list distinction is a STRUCTURAL clause field (operand_kind), not
    # an in-operand sentinel, so it is forge-proof. A user $eq operand is arbitrary
    # opaque JSON and is canonicalized only under the "operand" key — it can never
    # mint the clause-level discriminator. So a bare list (tuple, exact-array
    # equality) must NOT collide with an explicit $eq whose operand is a dict, even
    # one that mimics an internal marker. These are semantically different (exact
    # ARRAY equality vs exact DICT equality); an in-operand sentinel would have
    # collided them.
    tuple_hash = canonical_hash(_ast({"metadata.tags": ["a", "b"]}))
    arr_dict_hash = canonical_hash(_ast({"metadata.tags": {"$eq": {"$arr": ["a", "b"]}}}))
    forge_dict_hash = canonical_hash(_ast({"metadata.tags": {"$eq": {"operand_kind": "tuple"}}}))
    assert tuple_hash != arr_dict_hash
    assert tuple_hash != forge_dict_hash


# --- AC3: order preservation for non-commutative shapes ------------------- #


def test_hash_nin_list_is_order_sensitive() -> None:
    a = canonical_hash(_ast({"metadata.k": {"$nin": [1, 2]}}))
    b = canonical_hash(_ast({"metadata.k": {"$nin": [2, 1]}}))
    assert a != b


def test_hash_not_operand_order_is_preserved() -> None:
    # $not is non-commutative: not(or(a,b)) ($nor) must not collapse to or(a,b) —
    # they hash differently, so the NOT operand is not re-sorted away.
    nor_hash = canonical_hash(_ast({"$nor": [{"source_name": "a"}, {"source_type": "b"}]}))
    or_hash = canonical_hash(_ast({"$or": [{"source_name": "a"}, {"source_type": "b"}]}))
    assert nor_hash != or_hash


def test_hash_double_not_differs_from_single_not() -> None:
    # NOT(NOT(x)) is structurally distinct from NOT(x) — negation depth is part of
    # the hashed structure (not collapsed).
    once = canonical_hash(_ast({"$not": {"source_name": "x"}}))
    twice = canonical_hash(_ast({"$not": {"$not": {"source_name": "x"}}}))
    assert once != twice


def test_hash_nested_dict_operand_key_order_insensitive() -> None:
    # Key-order insensitivity is recursive through nested equality operands.
    a = canonical_hash(_ast({"metadata.o": {"a": {"x": 1, "y": 2}, "b": 3}}))
    b = canonical_hash(_ast({"metadata.o": {"b": 3, "a": {"y": 2, "x": 1}}}))
    assert a == b


def test_hash_distinguishes_date_literal_from_plain_string() -> None:
    # A DateLiteral operand must not collide with a plain ISO string operand.
    dt_hash = canonical_hash(_ast({"metadata.when": {"$eq": {"$date": "2026-01-01T00:00:00+00:00"}}}))
    str_hash = canonical_hash(_ast({"metadata.when": {"$eq": "2026-01-01T00:00:00+00:00"}}))
    assert dt_hash != str_hash


def test_hash_accepts_filterclause_root() -> None:
    # canonical_hash takes a FilterNode | FilterClause; a bare clause hashes too.
    h = canonical_hash(FilterClause(path=("source_name",), op=Op.EQ, operand="x"))
    assert isinstance(h, str)
    assert len(h) == 64


# ---------------------------------------------------------------------------
# Frozen / slotted dataclasses.
# ---------------------------------------------------------------------------


def test_nodes_are_frozen() -> None:
    clause = FilterClause(path=("x",), op=Op.EQ, operand=1)
    node = FilterNode(op=Op.AND, children=(clause,))
    with pytest.raises((AttributeError, TypeError)):
        clause.op = Op.NE  # type: ignore[misc]
    with pytest.raises((AttributeError, TypeError)):
        node.op = Op.OR  # type: ignore[misc]


def test_dateliteral_is_frozen() -> None:
    lit = DateLiteral(value=datetime(2026, 1, 1, tzinfo=UTC))
    with pytest.raises((AttributeError, TypeError)):
        lit.value = datetime(2027, 1, 1, tzinfo=UTC)  # type: ignore[misc]


def test_filterclause_default_operand_is_none() -> None:
    clause = FilterClause(path=("x",), op=Op.EXISTS)
    assert clause.operand is None


def test_filternode_default_children_is_empty_tuple() -> None:
    node = FilterNode(op=Op.AND)
    assert node.children == ()


def _walk_clauses(node: FilterNode | FilterClause) -> list[FilterClause]:
    """Recursively collect every leaf clause in an AST."""
    if isinstance(node, FilterClause):
        return [node]
    out: list[FilterClause] = []
    for child in node.children:
        out.extend(_walk_clauses(child))
    return out


def _walk_nodes(node: FilterNode | FilterClause) -> list[FilterNode]:
    """Recursively collect every logical (FilterNode) node in an AST."""
    if isinstance(node, FilterClause):
        return []
    out: list[FilterNode] = [node]
    for child in node.children:
        out.extend(_walk_nodes(child))
    return out
