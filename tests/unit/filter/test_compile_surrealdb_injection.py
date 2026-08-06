"""Injection-guard unit test for the SurrealDB recall-filter compiler.

The recall-filter validator only checks that a folded metadata key
``startswith("metadata.")`` — it does NOT restrict the characters of the
sub-path segments. The SurrealDB compiler interpolates those segments into the
predicate string (SurrealQL cannot bind a field name as a parameter), so it MUST
validate each segment as a safe identifier and never interpolate anything else.
This pins that guard, in both of the shapes it takes: a controlled ``CompileError``
under ``on_unsupported="raise"``, and a deferral to the caller's residual under
``"split"`` (the §8 ruling on this ticket — an unrenderable field name is caller
input, not a
compiler fault, wherever there is a post-filter to hand it to).

(The exhaustive emitted-string assertions live in the QA-owned
``test_compile_surrealdb.py``; this module isolates the security-critical case so
it cannot regress unnoticed.)
"""

from __future__ import annotations

import dataclasses

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

    ``_CTX`` is an ``on_unsupported="raise"`` context — the mode both temporal
    SurrealDB contexts use and the mode any direct compile gets by default. That
    is the only mode the raise belongs to now; see
    :func:`test_the_guard_is_mode_aware_raise_interpolates_nothing_split_defers`
    for the other half and for why the distinction is deliberate.
    """
    ast = parse_to_ast(RecallFilter.model_validate({hostile_key: "v"}))
    with pytest.raises(CompileError):
        compile_surrealdb(ast, _CTX)


def test_the_guard_is_mode_aware_raise_interpolates_nothing_split_defers() -> None:
    """The guard survives ``"raise"``; under ``"split"`` the leaf defers instead.

    This ticket's §8 ruling made a non-identifier metadata segment **caller input,
    not a compiler fault**: a hyphenated key like ``metadata.a.due-date`` is legal,
    common JSON, and SurrealQL has no bind form for a field *name*, so the leaf is
    simply unpushable on this backend. Under ``"split"`` it therefore takes the
    same route as any other unpushable leaf — a match-all placeholder, left out of
    ``consumed_keys``, evaluated by the caller's ``compile_python`` residual, which
    handles hyphenated keys correctly. That is what makes all four document stores
    return the same rows for the same filter
    (``tests/integration/storage/backends/surrealdb/test_relational_scan_documents.py::test_a_hyphenated_metadata_key_matches_the_oracle_and_raw_sqlite``).

    **Both halves are asserted here because the two conditions must stay in sync.**
    The compiler gates the leaf in ``_clause_consumable`` *and* diverts it in
    ``compile_clause``, on deliberately identical conditions; changing one alone
    resurrects the pre-§8 position-dependence, where the same key raised in
    conjunctive position and deferred inside an ``$or``. Pinning only the
    ``"split"`` half would let the raise be deleted as dead code — it is not dead,
    it is the injection protection on every path that has no residual to defer to.

    Measured in this tree on this branch, for all seven hostile keys the
    parametrization above carries: ``"raise"`` raises ``CompileError`` naming the
    offending segment, ``"split"`` compiles to exactly ``(true)`` with empty
    ``consumed_keys``. Neither mode interpolates the segment.
    """
    ast = parse_to_ast(RecallFilter.model_validate({"metadata.a.due-date": "v"}))

    with pytest.raises(CompileError) as excinfo:
        compile_surrealdb(ast, dataclasses.replace(_CTX, on_unsupported="raise"))
    # The segment, not the physical field: this context remaps ``metadata`` to
    # ``metadata_``, and a message echoing that would name a column the caller
    # never wrote.
    assert "due-date" in str(excinfo.value)
    assert "metadata_" not in str(excinfo.value)

    compiled = compile_surrealdb(ast, dataclasses.replace(_CTX, on_unsupported="split"))
    assert compiled.consumed_keys == frozenset()
    # The match-all placeholder, so nothing narrows and the residual decides —
    # and, critically, the unrenderable segment appears nowhere in the fragment.
    assert compiled.predicate == "(true)"
    assert "due-date" not in compiled.predicate


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
    from khora.storage.temporal import surrealdb as mod

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

    from khora.storage.temporal import TemporalFilter

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

    from khora.storage.temporal import TemporalFilter

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
