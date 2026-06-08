"""Injection-guard unit test for the SurrealDB recall-filter compiler.

The recall-filter validator only checks that a folded metadata key
``startswith("metadata.")`` — it does NOT restrict the characters of the
sub-path segments. The SurrealDB compiler interpolates those segments into the
predicate string (SurrealQL cannot bind a field name as a parameter), so it MUST
validate each segment as a safe identifier and raise a controlled compiler fault
on anything else. This pins that guard.

(The exhaustive emitted-string assertions live in the QA-owned
``test_compile_surrealdb.py``; this module isolates the security-critical case so
it cannot regress unnoticed.)
"""

from __future__ import annotations

import pytest

from khora.filter import RecallFilter
from khora.filter.ast import parse_to_ast
from khora.filter.compilers.surrealdb import compile_surrealdb
from khora.filter.context import CompileContext, CompileError

pytestmark = pytest.mark.unit

_CTX = CompileContext(backend_target="temporal_chunk", field_mapping={"metadata": "metadata_"})


@pytest.mark.parametrize(
    "hostile_key",
    [
        "metadata.x = 1 OR true; --",  # SurrealQL break-out attempt
        "metadata.a b",  # whitespace
        'metadata.a"b',  # quote
        "metadata.a)b",  # paren
        "metadata.1abc",  # leading digit
        "metadata.a-b",  # hyphen
        "metadata.a.b c",  # unsafe nested segment
    ],
)
def test_unsafe_metadata_segment_raises_compile_error(hostile_key: str) -> None:
    """An unsafe metadata path segment is a controlled CompileError, not a query.

    The fault is raised regardless of ``on_unsupported`` mode — it is an injection
    guard, not a capability gap.
    """
    ast = parse_to_ast(RecallFilter.model_validate({hostile_key: "v"}))
    with pytest.raises(CompileError):
        compile_surrealdb(ast, _CTX)


def test_safe_nested_segment_compiles() -> None:
    """A well-formed nested path descends natively without raising."""
    ast = parse_to_ast(RecallFilter.model_validate({"metadata.labels.tier": "gold"}))
    compiled = compile_surrealdb(ast, _CTX)
    assert "metadata_.labels.tier" in compiled.predicate


# ---------------------------------------------------------------------------
# Legacy ``TemporalFilter.additional`` integration — the bare-equality path.
#
# ``TemporalFilter.additional`` keys are NOT char-restricted upstream, and the
# skeleton SurrealDB backend interpolates them into a WHERE clause. Both the
# range-op AND the bare-equality paths in ``_build_filter_clauses`` must route
# through the compiler's injection guard. These tests exercise that real call
# surface (the compiler-only tests above did not cover the legacy integration).
# ---------------------------------------------------------------------------


@pytest.fixture(name="store_module")
def _store_module():  # noqa: ANN202 - test fixture
    """The skeleton SurrealDB backend module (importing it registers the compiler)."""
    from khora.engines.skeleton.backends import surrealdb as mod

    return mod


@pytest.mark.parametrize(
    "additional",
    [
        {"a.b; DROP TABLE x; --": {"eq": 1}},  # dict-valued $eq, hostile key
        {"a.b OR true; --": "scalar"},  # scalar-valued (bare) $eq, hostile key
        {"a b": {"eq": 1}},  # whitespace in a dict-eq key
        {"1abc": "v"},  # leading digit in a scalar-eq key
    ],
)
def test_legacy_additional_unsafe_key_raises_compile_error(store_module, additional) -> None:  # noqa: ANN001
    """A hostile ``additional`` key (equality path) is a controlled CompileError.

    Closes the injection hole on the bare-equality branches of
    ``_build_filter_clauses`` — the key never reaches an interpolated WHERE clause.
    """
    from uuid import uuid4

    from khora.engines.skeleton.backends import TemporalFilter

    tf = TemporalFilter(additional=additional)
    with pytest.raises(CompileError):
        store_module.SurrealDBTemporalStore._build_filter_clauses(uuid4(), tf)


def test_legacy_additional_eq_routes_through_guard(store_module) -> None:  # noqa: ANN001
    """A safe ``additional`` filter routes every op through the compiler.

    ``eq`` is array-aware containment (matching the recall-filter path: a scalar
    field equal to the value OR an array field containing it), a range op gains a
    ``type::is::*`` gate, and a nested dotted key descends natively — all with
    binds carried out-of-band (never the raw user value interpolated).
    """
    from uuid import uuid4

    from khora.engines.skeleton.backends import TemporalFilter

    tf = TemporalFilter(
        additional={
            "tier": {"eq": "gold"},  # dict-valued $eq
            "score": {"gte": 5},  # range op (gated)
            "nested.key": {"gt": 1},  # nested dotted key
            "flat": "x",  # scalar (bare) $eq
        }
    )
    clauses, bindings = store_module.SurrealDBTemporalStore._build_filter_clauses(uuid4(), tf)
    joined = " ".join(clauses)

    # Equality paths: array-aware containment, key descended natively, value bound.
    assert (
        "(metadata_.tier = $af_0_eq_0 OR (type::is::array(metadata_.tier) AND metadata_.tier CONTAINS $af_0_eq_0))"
        in joined
    )
    assert (
        "(metadata_.flat = $af_3_eq_0 OR (type::is::array(metadata_.flat) AND metadata_.flat CONTAINS $af_3_eq_0))"
        in joined
    )
    # Range path: type-gated.
    assert "(type::is::number(metadata_.score) AND metadata_.score >= $af_1_gte_0)" in joined
    # Nested dotted key descends natively.
    assert "metadata_.nested.key" in joined
    # User values bind out-of-band, never interpolated.
    assert bindings["af_0_eq_0"] == "gold"
    assert bindings["af_3_eq_0"] == "x"
