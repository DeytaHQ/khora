"""Behavioral coverage for the legacy ``TemporalFilter.additional`` path.

``SurrealDBTemporalStore._build_filter_clauses`` lowers the legacy
``additional`` dict (the pre-deterministic-filter metadata-predicate channel)
into SurrealQL ``WHERE`` fragments. Every ``additional`` key is now routed
through the recall-filter SurrealDB compiler's guarded builder
(``_legacy_metadata_predicate``) so the path segment is validated against the
injection guard before any interpolation — the ``eq`` branch in particular used
to interpolate the key verbatim, which is the regression that slipped through for
lack of a test.

These tests pin the *emitted SurrealQL* (the behavioral/semantic contract):

* ``eq`` (scalar value form AND explicit ``{"eq": ...}`` form) → array-aware
  containment ``(metadata_.<path> = $bind OR (type::is::array(...) AND ...
  CONTAINS $bind))``, matching the recall-filter path so a scalar value also
  matches an array field (the canonical ``tags: list[str]`` shape);
* each range op (``gt`` / ``gte`` / ``lt`` / ``lte``) → a type-gated
  ``(type::is::<t>(metadata_.<path>) AND metadata_.<path> <op> $bind)`` so a
  numeric value orders numerically and a wrong-typed value is excluded;
* a dotted ``additional`` key descends natively (``labels.priority`` →
  ``metadata_.labels.priority``), not collapsed/mangled into one token;
* the type-gate function tracks the operand's Python type (number / string / bool).

The injection-*rejection* unit case (an unsafe key raising ``CompileError``)
lives with the compiler's injection guard; this module owns the
happy-path/semantic side. The embedded-SurrealDB row-set proof that a range op
actually *excludes* a wrong-typed legacy value lives in
``tests/integration/filter/test_compile_surrealdb_embedded.py``.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from khora.storage.temporal import TemporalFilter
from khora.storage.temporal.surrealdb import SurrealDBTemporalStore

# Hard import (NOT importorskip): ``_build_filter_clauses`` is a pure static
# string-builder — it imports the recall-filter compiler (pure Python, no SDK)
# and only inspects the emitted predicate, so an import failure must be a LOUD
# test error, never a silent skip.

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _legacy_clauses(additional: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    """Build clauses+binds for an ``additional`` dict, dropping the namespace scope.

    ``_build_filter_clauses`` always prepends the namespace-scoping clause and its
    two binds (``ns_rid`` / ``ns_str``); strip those so assertions see only the
    legacy ``additional`` predicates and their ``af_*`` binds.
    """
    clauses, binds = SurrealDBTemporalStore._build_filter_clauses(
        uuid4(),
        TemporalFilter(additional=additional),
    )
    legacy_clauses = [c for c in clauses if "namespace" not in c]
    legacy_binds = {k: v for k, v in binds.items() if k.startswith("af_")}
    return legacy_clauses, legacy_binds


def _norm(surql: str) -> str:
    """Collapse whitespace and lowercase for resilient substring matching."""
    return " ".join(surql.split()).lower()


# ===========================================================================
# eq — plain equality, no type gate.
# ===========================================================================


def test_legacy_eq_scalar_form_is_array_aware() -> None:
    # A bare scalar value (no op dict) is the ``eq`` sugar: array-aware containment
    # on the metadata path (a scalar field equal to the value OR an array field
    # containing it), matching the recall-filter path. The operand is bound once
    # and reused across both arms, never interpolated.
    clauses, binds = _legacy_clauses({"tier": "gold"})
    assert len(clauses) == 1
    sql = _norm(clauses[0])
    assert "metadata_.tier = $" in sql
    assert "type::is::array(metadata_.tier) and metadata_.tier contains $" in sql
    assert list(binds.values()) == ["gold"]


def test_legacy_eq_dict_form_is_array_aware() -> None:
    # The explicit ``{"eq": ...}`` dict form lowers identically to the scalar form.
    clauses, binds = _legacy_clauses({"tier": {"eq": "gold"}})
    assert len(clauses) == 1
    sql = _norm(clauses[0])
    assert "metadata_.tier = $" in sql
    assert "type::is::array(metadata_.tier) and metadata_.tier contains $" in sql
    assert list(binds.values()) == ["gold"]


def test_legacy_eq_value_binds_not_interpolated() -> None:
    # The value must be carried as a bind, never spliced into the predicate string
    # — the property the guarded builder restored for the eq path.
    clauses, binds = _legacy_clauses({"tier": "gold"})
    assert "gold" not in clauses[0]
    assert "gold" in binds.values()


# ===========================================================================
# Range ops — type-gated, numeric ordering.
# ===========================================================================


@pytest.mark.parametrize(
    ("legacy_op", "surql_op"),
    [
        ("gt", ">"),
        ("gte", ">="),
        ("lt", "<"),
        ("lte", "<="),
    ],
)
def test_legacy_range_op_is_number_gated(legacy_op: str, surql_op: str) -> None:
    # Each legacy range op emits the type-gated pair: a ``type::is::number`` gate
    # AND-ed (short-circuit) ahead of the compare, so a wrong-typed / absent value
    # never reaches the compare and a numeric value orders numerically.
    clauses, binds = _legacy_clauses({"score": {legacy_op: 5}})
    assert len(clauses) == 1
    sql = _norm(clauses[0])
    assert f"(type::is::number(metadata_.score) and metadata_.score {surql_op} $" in sql
    assert list(binds.values()) == [5]


def test_legacy_range_op_gate_prefix_present() -> None:
    # The type-gate prefix is the load-bearing guarantee (the bug was a path that
    # bypassed it). Assert the ``type::is::`` prefix leads every range predicate.
    for op in ("gt", "gte", "lt", "lte"):
        clauses, _ = _legacy_clauses({"score": {op: 1}})
        assert _norm(clauses[0]).startswith("(type::is::")


def test_legacy_range_string_operand_picks_string_gate() -> None:
    # A string operand gates on ``type::is::string`` (lexicographic text compare).
    clauses, binds = _legacy_clauses({"name": {"gt": "m"}})
    sql = _norm(clauses[0])
    assert "(type::is::string(metadata_.name) and metadata_.name > $" in sql
    assert list(binds.values()) == ["m"]


def test_legacy_range_bool_operand_picks_bool_gate() -> None:
    # A bool operand gates on ``type::is::bool`` — NOT number (a bool is an int
    # subclass in Python, and SurrealDB agrees a bool is not a number).
    clauses, _ = _legacy_clauses({"flag": {"gt": True}})
    sql = _norm(clauses[0])
    assert "type::is::bool(metadata_.flag)" in sql
    assert "type::is::number" not in sql


# ===========================================================================
# Nested dotted keys descend natively.
# ===========================================================================


def test_legacy_dotted_key_descends_natively() -> None:
    # A dotted ``additional`` key (``labels.priority``) descends to
    # ``metadata_.labels.priority`` — NOT collapsed/mangled into a single token
    # (the bad sanitizer would have produced ``metadata_labels`` or similar).
    clauses, _ = _legacy_clauses({"labels.priority": {"gte": 3}})
    sql = _norm(clauses[0])
    assert "metadata_.labels.priority" in sql
    assert "metadata_labels" not in sql


def test_legacy_dotted_key_eq_descends_natively() -> None:
    # The eq path descends a dotted key the same way (the eq branch is the one the
    # regression touched).
    clauses, _ = _legacy_clauses({"labels.tier": "gold"})
    sql = _norm(clauses[0])
    assert "(metadata_.labels.tier = $" in sql


# ===========================================================================
# Multiple additional keys never collide on bind names.
# ===========================================================================


def test_legacy_multiple_keys_distinct_bind_names() -> None:
    # Two additional keys allocate distinct ``af_<i>_<op>_<n>`` bind names so
    # concurrent legacy predicates never overwrite each other.
    clauses, binds = _legacy_clauses({"a": "x", "b": {"gt": 1}})
    assert len(clauses) == 2
    assert len(binds) == 2
    assert len(set(binds)) == 2  # no bind-name collision
    assert set(binds.values()) == {"x", 1}
