"""Unit tests for the public ``RecallFilter`` model + validator.

These tests pin the Layer-2 contract: the typed pydantic model that
validates a recall filter document (kwarg form or wire dict form) and
raises a single structured error type on any grammar violation. The
model *validates*; it does not lower to an AST or compile.

Two validation regimes, one error type:
  (a) pydantic-structural — closed top-level key set (``extra="forbid"``),
      typed ``*Ops`` submodels, ``model_fields_set`` unset-vs-null.
  (b) recursive ``metadata`` walk — the sub-grammar pydantic can't express
      on ``dict[str, Any]`` (operator whitelist, ``$in``/``$nin`` lists,
      ``$exists`` bool, nonempty logical arrays, the operator-position
      closure / no-mixing rule, and operand opacity).

Both raise ``RecallFilterValidationError`` carrying a populated
``errors: list[FieldError]``.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from khora.filter import (
    SYSTEM_KEYS,
    DateOps,
    Op,
    RecallFilter,
    RecallFilterUnsupportedError,
    RecallFilterValidationError,
    StringOps,
)
from khora.filter.model import FieldError

# Every test here is a fast, in-process unit test.
pytestmark = pytest.mark.unit


# Round-trip helper: model_dump(by_alias=True, exclude_none=True) is the
# canonical wire serialization. A byte-stable round-trip means dumping a
# validated model reproduces the input dict exactly.
def _roundtrip(wire: dict) -> dict:
    return RecallFilter.model_validate(wire).model_dump(by_alias=True, exclude_none=True)


# ---------------------------------------------------------------------------
# Public surface — the 7 symbols import from both ``khora.filter`` and ``khora``
# ---------------------------------------------------------------------------


def test_seven_symbols_import_from_khora_filter() -> None:
    import khora.filter as f

    expected = {
        "RecallFilter",
        "StringOps",
        "DateOps",
        "RecallFilterValidationError",
        "RecallFilterUnsupportedError",
        "Op",
        "SYSTEM_KEYS",
    }
    assert expected.issubset(set(dir(f)))
    assert set(f.__all__) == expected


def test_seven_symbols_import_from_khora_top_level() -> None:
    import khora
    from khora import (  # noqa: F401
        SYSTEM_KEYS,
        DateOps,
        Op,
        RecallFilter,
        RecallFilterUnsupportedError,
        RecallFilterValidationError,
        StringOps,
    )

    for name in (
        "RecallFilter",
        "StringOps",
        "DateOps",
        "RecallFilterValidationError",
        "RecallFilterUnsupportedError",
        "Op",
        "SYSTEM_KEYS",
    ):
        assert name in khora.__all__, f"{name} missing from khora.__all__"


def test_field_error_importable_from_khora_filter() -> None:
    from khora.filter import FieldError as ExportedFieldError

    assert ExportedFieldError is FieldError


# ---------------------------------------------------------------------------
# SYSTEM_KEYS / Op
# ---------------------------------------------------------------------------


def test_system_keys_are_the_ten_documented_keys() -> None:
    assert SYSTEM_KEYS == frozenset(
        {
            "occurred_at",
            "created_at",
            "source_timestamp",
            "source_type",
            "source_name",
            "source_url",
            "external_id",
            "content_type",
            "source",
            "title",
        }
    )
    # metadata and the logical ops are NOT system keys.
    assert "metadata" not in SYSTEM_KEYS
    assert "$and" not in SYSTEM_KEYS
    # Non-projection fields are not system keys.
    assert "author" not in SYSTEM_KEYS
    assert "language" not in SYSTEM_KEYS
    assert "source_system" not in SYSTEM_KEYS
    assert "channel" not in SYSTEM_KEYS


def test_system_keys_is_a_frozenset() -> None:
    assert isinstance(SYSTEM_KEYS, frozenset)


def test_op_enum_mirrors_operator_literals() -> None:
    assert Op.EQ.value == "$eq"
    assert Op.NE.value == "$ne"
    assert Op.GT.value == "$gt"
    assert Op.GTE.value == "$gte"
    assert Op.LT.value == "$lt"
    assert Op.LTE.value == "$lte"
    assert Op.IN.value == "$in"
    assert Op.NIN.value == "$nin"
    assert Op.EXISTS.value == "$exists"
    assert Op.AND.value == "$and"
    assert Op.OR.value == "$or"
    assert Op.NOR.value == "$nor"
    assert Op.NOT.value == "$not"
    assert Op.DATE.value == "$date"


def test_op_is_str_enum() -> None:
    # str-enum: members compare equal to their literal.
    assert Op.EQ == "$eq"


# ---------------------------------------------------------------------------
# All 11 whitelisted top-level keys validate
# ---------------------------------------------------------------------------


def test_all_eleven_whitelisted_top_level_keys_validate() -> None:
    # 10 system keys + bare metadata = 11 whitelisted top-level data keys.
    wire = {
        "occurred_at": {"$gte": "2026-04-05T00:00:00Z"},
        "created_at": {"$lt": "2026-05-01T00:00:00Z"},
        "source_timestamp": {"$gt": "2026-01-01T00:00:00Z"},
        "source_type": "connection",
        "source_name": {"$in": ["linear", "slack"]},
        "source_url": "https://example.com",
        "external_id": "ext-1",
        "content_type": "text/markdown",
        "source": "origin",
        "title": {"$exists": True},
        "metadata": {"team": "ingest"},
    }
    model = RecallFilter.model_validate(wire)
    assert isinstance(model, RecallFilter)


@pytest.mark.parametrize(
    "key",
    [
        "occurred_at",
        "created_at",
        "source_timestamp",
        "source_type",
        "source_name",
        "source_url",
        "external_id",
        "content_type",
        "source",
        "title",
        "metadata",
    ],
)
def test_each_whitelisted_key_validates_alone(key: str) -> None:
    if key in {"occurred_at", "created_at", "source_timestamp"}:
        value: object = "2026-04-05T00:00:00Z"
    elif key == "metadata":
        value = {"team": "ingest"}
    else:
        value = "x"
    model = RecallFilter.model_validate({key: value})
    assert key in model.model_fields_set


def test_logical_op_keys_validate_at_top_level() -> None:
    for op in ("$and", "$or", "$nor"):
        model = RecallFilter.model_validate({op: [{"source_type": "connection"}]})
        assert isinstance(model, RecallFilter)
    model = RecallFilter.model_validate({"$not": {"source_type": "connection"}})
    assert isinstance(model, RecallFilter)


# ---------------------------------------------------------------------------
# Unknown top-level key raises with a populated .errors list[FieldError]
# ---------------------------------------------------------------------------


def test_unknown_top_level_key_raises() -> None:
    with pytest.raises(RecallFilterValidationError) as exc:
        RecallFilter.model_validate({"priority": 5})
    err = exc.value
    assert isinstance(err.errors, list)
    assert len(err.errors) >= 1
    assert all(isinstance(fe, FieldError) for fe in err.errors)
    fe = err.errors[0]
    assert fe.path
    assert fe.code
    assert fe.message


def test_unknown_top_level_dollar_key_raises() -> None:
    # A top-level $-key that isn't a known logical op still raises.
    with pytest.raises(RecallFilterValidationError):
        RecallFilter.model_validate({"$comment": "hello"})


def test_field_error_has_expected_shape() -> None:
    with pytest.raises(RecallFilterValidationError) as exc:
        RecallFilter.model_validate({"nope": 1})
    fe = exc.value.errors[0]
    # JSON-Pointer-ish path, a code, a message; allowed is optional.
    assert isinstance(fe.path, str)
    assert isinstance(fe.code, str)
    assert isinstance(fe.message, str)
    assert hasattr(fe, "allowed")


def test_validation_error_is_khora_error() -> None:
    from khora.exceptions import KhoraError

    assert issubclass(RecallFilterValidationError, KhoraError)
    assert issubclass(RecallFilterUnsupportedError, KhoraError)


# ---------------------------------------------------------------------------
# Per-key operator subsets (§2)
# ---------------------------------------------------------------------------


def test_exists_on_date_key_raises() -> None:
    # DateOps has NO $exists.
    with pytest.raises(RecallFilterValidationError):
        RecallFilter.model_validate({"occurred_at": {"$exists": True}})


def test_range_op_on_string_key_raises() -> None:
    # StringOps has no range ops.
    with pytest.raises(RecallFilterValidationError):
        RecallFilter.model_validate({"source_type": {"$gt": "a"}})


@pytest.mark.parametrize("op", ["$gt", "$gte", "$lt", "$lte"])
def test_each_range_op_on_string_key_raises(op: str) -> None:
    with pytest.raises(RecallFilterValidationError):
        RecallFilter.model_validate({"source_name": {op: "a"}})


def test_date_key_accepts_full_range_op_set() -> None:
    wire = {
        "occurred_at": {
            "$gte": "2026-01-01T00:00:00Z",
            "$lte": "2026-12-31T00:00:00Z",
        }
    }
    model = RecallFilter.model_validate(wire)
    assert isinstance(model.occurred_at, DateOps)


@pytest.mark.parametrize("op", ["$eq", "$ne"])
def test_date_key_accepts_equality_ops(op: str) -> None:
    # Date keys accept equality ($eq/$ne) — not just range. The DateOps arm
    # has these fields, so a date key with $eq/$ne lands on DateOps (the operand
    # is parsed + UTC-normalized like every other date operand).
    model = RecallFilter.model_validate({"occurred_at": {op: "2026-04-05T00:00:00Z"}})
    assert isinstance(model.occurred_at, DateOps)


@pytest.mark.parametrize("op", ["$in", "$nin"])
def test_date_key_accepts_set_ops(op: str) -> None:
    # Date keys accept set membership ($in/$nin) over a list of date operands.
    model = RecallFilter.model_validate({"occurred_at": {op: ["2026-01-01T00:00:00Z", "2026-02-01T00:00:00Z"]}})
    assert isinstance(model.occurred_at, DateOps)


def test_date_key_set_op_non_list_operand_raises() -> None:
    # The DateOps set arm is typed list[datetime]; a scalar operand fails it.
    with pytest.raises(RecallFilterValidationError):
        RecallFilter.model_validate({"occurred_at": {"$in": "2026-01-01T00:00:00Z"}})


def test_string_key_accepts_eq_ne_in_nin_exists() -> None:
    model = RecallFilter.model_validate({"source_type": {"$in": ["a", "b"], "$ne": "c", "$exists": True}})
    assert isinstance(model.source_type, StringOps)


def test_exists_on_string_key_validates() -> None:
    model = RecallFilter.model_validate({"title": {"$exists": False}})
    assert isinstance(model.title, StringOps)


# ---------------------------------------------------------------------------
# Bare-value sugar (§2/§3) — scalar/list/dict ⇒ $eq, bare-list exact-array
# ---------------------------------------------------------------------------


def test_bare_scalar_string_is_eq_sugar() -> None:
    model = RecallFilter.model_validate({"source_name": "linear"})
    assert model.source_name == "linear"


def test_bare_scalar_date_is_eq_sugar() -> None:
    model = RecallFilter.model_validate({"occurred_at": "2026-04-05T00:00:00Z"})
    # Scalar matches the datetime arm (= $eq sugar), not DateOps.
    assert isinstance(model.occurred_at, datetime)


def test_bare_list_is_exact_array_not_in() -> None:
    # List ⇒ $eq exact-array equality, NOT $in. The list arm must match
    # list[str], and it must NOT be coerced into a StringOps($in=...).
    model = RecallFilter.model_validate({"source_type": ["a", "b"]})
    assert model.source_type == ["a", "b"]
    assert not isinstance(model.source_type, StringOps)


def test_bare_list_roundtrips_as_list_not_in() -> None:
    wire = {"source_type": ["a", "b"]}
    assert _roundtrip(wire) == wire


def test_explicit_in_is_membership() -> None:
    model = RecallFilter.model_validate({"source_type": {"$in": ["a", "b"]}})
    assert isinstance(model.source_type, StringOps)
    assert model.source_type.in_ == ["a", "b"]


# ---------------------------------------------------------------------------
# Dot-path fold + flat re-serialize round-trips BYTE-STABLE
# ---------------------------------------------------------------------------


def test_dot_path_fold_roundtrips_byte_stable() -> None:
    wire = {"metadata.tag": {"$in": ["roadmap", "okrs"]}}
    assert _roundtrip(wire) == wire


def test_dot_path_not_rejected_as_unknown_top_level_key() -> None:
    # The mode="before" fold must pull metadata.* OUT before extra="forbid".
    model = RecallFilter.model_validate({"metadata.tag": "x"})
    assert isinstance(model, RecallFilter)


def test_bare_metadata_whole_blob_roundtrips_byte_stable() -> None:
    wire = {"metadata": {"team": "ingest", "level": 3}}
    assert _roundtrip(wire) == wire


def test_bare_metadata_populates_field_literally() -> None:
    model = RecallFilter.model_validate({"metadata": {"team": "ingest"}})
    assert model.metadata == {"team": "ingest"}


def test_dot_path_not_merged_into_metadata_field() -> None:
    # Folded dot-path predicates must NOT land on the metadata field
    # (reserved for the whole-blob case).
    model = RecallFilter.model_validate({"metadata.tag": {"$in": ["a"]}})
    assert model.metadata is None


def test_mixed_whole_blob_and_dot_path_roundtrips() -> None:
    wire = {
        "metadata": {"team": "ingest"},
        "metadata.tag": {"$eq": "x"},
    }
    assert _roundtrip(wire) == wire


def test_whole_filter_with_system_keys_and_dot_path_roundtrips() -> None:
    # Byte-stable round-trip is the contract for the dot-path fold + flat
    # re-serialize over string system keys, whole-blob, and dot-paths. Date
    # keys are intentionally excluded here: an ISO-8601 string is parsed and
    # UTC-normalized to a ``datetime`` (a required behavior), so the date arm
    # is not byte-identical on dump — see the dedicated date-normalization
    # tests below.
    wire = {
        "source_name": {"$in": ["linear"]},
        "source_type": "connection",
        "metadata.tag": {"$in": ["tag1", "tag2", "tag3"]},
    }
    assert _roundtrip(wire) == wire


def test_logical_filter_roundtrips_byte_stable() -> None:
    wire = {
        "$or": [
            {"source_name": "linear", "metadata.tag": "urgent"},
            {"source_name": "slack", "metadata.tag": "urgent"},
        ]
    }
    assert _roundtrip(wire) == wire


def test_nested_subdocument_equality_roundtrips() -> None:
    # A nested dict on a metadata SUB-path is whole-subdocument equality.
    wire = {"metadata.labels": {"team": "ingest"}}
    assert _roundtrip(wire) == wire


def test_nested_dot_path_folds_per_branch_roundtrips() -> None:
    # Each $and/$or branch carries its OWN dot-path folds — no
    # cross-contamination between branches. A branch with no metadata.*
    # fold round-trips with no spurious metadata key, and a folding branch
    # re-emits its own flat key.
    wire = {
        "source_name": "x",
        "metadata.outer": 1,
        "$and": [
            {"metadata.inner": 2},
            {"source_name": "y"},
        ],
    }
    assert _roundtrip(wire) == wire


def test_no_fold_branch_has_no_spurious_metadata_key() -> None:
    # A nested branch that defines no metadata.* predicate must not pick up
    # a metadata key from its sibling on round-trip.
    wire = {"$or": [{"metadata.tag": "a"}, {"source_name": "b"}]}
    dumped = RecallFilter.model_validate(wire).model_dump(by_alias=True, exclude_none=True)
    assert dumped == wire
    # The no-fold branch dumps to exactly its single system key.
    assert dumped["$or"][1] == {"source_name": "b"}


def test_metadata_prefix_without_trailing_dot_is_not_folded() -> None:
    # Only keys starting with "metadata." (trailing dot) fold. A bare
    # "metadata" key is the whole-blob field, not a dot-path.
    model = RecallFilter.model_validate({"metadata": {"x": 1}})
    assert model.metadata == {"x": 1}
    assert _roundtrip({"metadata": {"x": 1}}) == {"metadata": {"x": 1}}


# ---------------------------------------------------------------------------
# Operator-position closure — mixed/nested-$ raises
# ---------------------------------------------------------------------------


def test_mixed_dollar_and_non_dollar_keys_raises() -> None:
    # A value-dict that mixes $-op keys with non-$ keys raises.
    with pytest.raises(RecallFilterValidationError):
        RecallFilter.model_validate({"metadata.x": {"$gt": 1, "team": "ingest"}})


def test_nested_dollar_inside_equality_operand_raises() -> None:
    # A $-op nested inside an equality operand (whole-subdoc) raises.
    with pytest.raises(RecallFilterValidationError):
        RecallFilter.model_validate({"metadata.labels": {"priority": {"$gte": 5}}})


def test_nested_dollar_inside_bare_metadata_blob_raises() -> None:
    # The ADR's verbatim canonical example, in bare-metadata-blob form.
    with pytest.raises(RecallFilterValidationError):
        RecallFilter.model_validate({"metadata": {"labels": {"priority": {"$gte": 5}}}})


def test_deeply_nested_dollar_inside_equality_operand_raises() -> None:
    # The equality-operand $-scan must descend through nested subdocuments to
    # ANY depth — a $-op two or more levels inside a whole-subdocument equality
    # operand still raises (use dot-notation to address it). Without deep
    # descent the operator is silently treated as data, the footgun the
    # operator-position-closure rule exists to surface.
    with pytest.raises(RecallFilterValidationError):
        RecallFilter.model_validate({"metadata.x": {"b": {"c": {"$gt": 1}}}})


def test_deep_nested_dollar_in_equality_operand_via_dict_raises() -> None:
    # The whole-subdocument-equality $-scan descends through nested DICTS to any
    # depth; a $-op found anywhere inside a nested dict raises.
    with pytest.raises(RecallFilterValidationError):
        RecallFilter.model_validate({"metadata.labels": {"a": {"b": {"$gte": 5}}}})


@pytest.mark.parametrize(
    "wire",
    [
        # $ inside a list element nested in the equality operand — list elements
        # are opaque exact-array literals (no dot-addressable field name to bind
        # an operator to), so the $-op is data and does NOT raise.
        {"metadata.labels": {"items": [{"$gte": 5}]}},
        # $ reached via list nesting (the list element is opaque even though it
        # nests dicts inside) — no raise, byte-stable round-trip.
        {"metadata.labels": {"a": [{"b": {"c": {"$or": [1]}}}]}},
    ],
)
def test_dollar_inside_list_element_in_equality_operand_is_opaque(wire: dict) -> None:
    # The whole-subdocument-equality $-scan descends through nested DICTS only.
    # A list element is an opaque exact-array literal, so a $-op anywhere inside
    # a list is literal data — no raise, byte-stable round-trip.
    assert _roundtrip(wire) == wire


@pytest.mark.parametrize(
    "wire",
    [
        # Mixed $/non-$ keys are DATA inside a comparison-op operand (opaque);
        # only PREDICATE-position mixing raises. Boundary guard for the closure rule.
        {"metadata.x": {"$eq": {"team": "x", "$gt": 1}}},
        # Comparison operands stay fully opaque at ANY depth — the scan never
        # descends into them, so a deep nested $ is literal data.
        {"metadata.x": {"$eq": {"a": {"b": {"$or": [1]}}}}},
        # ...including a $ inside a list nested in the comparison operand.
        {"metadata.x": {"$ne": {"k": [{"$gte": 9}]}}},
    ],
)
def test_deep_dollar_inside_comparison_operand_is_opaque(wire: dict) -> None:
    # Mirror of the equality-operand scan from the opaque side: a comparison
    # operator's operand ($eq/$ne/...) is never inspected, so nested $-keys at
    # any depth are literal data — no raise, byte-stable round-trip.
    assert _roundtrip(wire) == wire


def test_metadata_bare_list_is_opaque_exact_array() -> None:
    # A bare list AS a metadata predicate value is $eq exact-array equality —
    # an opaque literal. Even a list whose elements contain $-keys is data
    # here (exact-array match), so it does not raise and round-trips stable.
    for wire in ({"metadata.x": [1, 2, 3]}, {"metadata.x": [{"a": 1}]}, {"metadata.x": [{"$gt": 1}]}):
        assert _roundtrip(wire) == wire


def test_metadata_unknown_operator_raises() -> None:
    with pytest.raises(RecallFilterValidationError):
        RecallFilter.model_validate({"metadata.x": {"$regex": "abc"}})


# ---------------------------------------------------------------------------
# Bare-metadata-blob grammar walk — each blob field is an independent
# predicate (sibling-AND); literal-dotted and $-prefixed field NAMES inside a
# blob are stored verbatim (not operators / not dot-paths). The folded
# metadata.<path> form is covered above; these pin the bare-blob arm.
# ---------------------------------------------------------------------------


def test_bare_blob_each_field_is_independently_validated() -> None:
    # Sibling-AND semantics: every field of a bare metadata blob is walked, so an
    # invalid predicate on the SECOND field still raises (the walk does not stop
    # at the first valid field). The error path names the offending blob field.
    with pytest.raises(RecallFilterValidationError) as exc:
        RecallFilter.model_validate({"metadata": {"good": {"$eq": 1}, "bad": {"$regex": "x"}}})
    fe = exc.value.errors[0]
    assert fe.code == "unknown_operator"
    assert fe.path == "/metadata/bad/$regex"


def test_bare_blob_operator_expr_field_validates() -> None:
    # A bare-blob field whose value is a valid operator-expression validates and
    # round-trips — the blob arm runs the same operator walk as the folded arm.
    wire = {"metadata": {"score": {"$gte": 5}}}
    model = RecallFilter.model_validate(wire)
    assert model.metadata == {"score": {"$gte": 5}}
    assert _roundtrip(wire) == wire


def test_bare_blob_mixed_operator_and_field_raises() -> None:
    # The operator-position-closure rule applies inside the bare blob too: a value
    # object mixing $-ops with plain keys raises (path names the blob field).
    with pytest.raises(RecallFilterValidationError) as exc:
        RecallFilter.model_validate({"metadata": {"field": {"$gt": 1, "plain": 2}}})
    fe = exc.value.errors[0]
    assert fe.code == "mixed_operator_and_field"
    assert fe.path == "/metadata/field"


def test_bare_blob_nested_dict_field_is_whole_subdocument_equality() -> None:
    # A nested-dict VALUE on a bare-blob field is whole-subdocument equality (an
    # opaque literal object), round-tripping byte-stable.
    wire = {"metadata": {"labels": {"team": "ingest", "tier": "gold"}}}
    model = RecallFilter.model_validate(wire)
    assert model.metadata == {"labels": {"team": "ingest", "tier": "gold"}}
    assert _roundtrip(wire) == wire


def test_bare_blob_literal_dotted_field_name_is_stored_verbatim() -> None:
    # A dotted key INSIDE a bare metadata blob is a literal field name, NOT a
    # dot-path: it is stored verbatim (the dot-path FOLD only fires on a flat
    # top-level "metadata.<path>" key, not on keys nested inside the blob). It
    # round-trips byte-stable and may carry an operator-expression value.
    for wire in (
        {"metadata": {"a.b": "x"}},
        {"metadata": {"a.b.c": {"$gt": 1}}},
    ):
        assert _roundtrip(wire) == wire


def test_bare_blob_dollar_prefixed_field_name_is_stored_verbatim() -> None:
    # A $-prefixed key as a bare-blob FIELD NAME is stored verbatim (it is a field
    # name, not an operator — operator position is the value object). The opaque
    # scalar value round-trips, and a valid operator-expression value still walks.
    for wire in (
        {"metadata": {"$weird": "x"}},
        {"metadata": {"$weird": {"$gt": 1}}},
    ):
        assert _roundtrip(wire) == wire


# ---------------------------------------------------------------------------
# Empty $and/$or/$nor raises; non-list $in/$nin raises; non-bool $exists raises
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", ["$and", "$or", "$nor"])
def test_empty_logical_array_raises(op: str) -> None:
    with pytest.raises(RecallFilterValidationError):
        RecallFilter.model_validate({op: []})


@pytest.mark.parametrize("op", ["$and", "$or", "$nor"])
def test_nonempty_logical_array_validates(op: str) -> None:
    model = RecallFilter.model_validate({op: [{"source_type": "connection"}]})
    assert isinstance(model, RecallFilter)


def test_in_must_be_list_on_metadata() -> None:
    with pytest.raises(RecallFilterValidationError):
        RecallFilter.model_validate({"metadata.x": {"$in": 5}})


def test_nin_must_be_list_on_metadata() -> None:
    with pytest.raises(RecallFilterValidationError):
        RecallFilter.model_validate({"metadata.x": {"$nin": "a"}})


def test_in_must_be_list_on_string_key() -> None:
    with pytest.raises(RecallFilterValidationError):
        RecallFilter.model_validate({"source_type": {"$in": 5}})


def test_exists_must_be_bool_on_metadata() -> None:
    with pytest.raises(RecallFilterValidationError):
        RecallFilter.model_validate({"metadata.x": {"$exists": "yes"}})


def test_exists_must_be_bool_on_string_key() -> None:
    # A non-bool, non-coercible value must raise on the typed string-key arm.
    # (Note: the typed StringOps arm uses pydantic's bool coercion, which
    # accepts "yes"/"true"/1; an integer like 5 is a clean non-bool that is
    # rejected by both the typed-key and metadata regimes.)
    with pytest.raises(RecallFilterValidationError):
        RecallFilter.model_validate({"source_type": {"$exists": 5}})


# ---------------------------------------------------------------------------
# model_fields_set — unset vs explicit null
# ---------------------------------------------------------------------------


def test_unset_key_not_in_fields_set() -> None:
    model = RecallFilter.model_validate({"source_type": "x"})
    assert "source_type" in model.model_fields_set
    assert "source_name" not in model.model_fields_set


def test_explicit_null_is_in_fields_set() -> None:
    # An explicit null is an active null-or-missing match, distinguishable
    # from an unset key via model_fields_set.
    model = RecallFilter.model_validate({"source_name": None})
    assert "source_name" in model.model_fields_set
    assert model.source_name is None


def test_unset_vs_null_distinguishable() -> None:
    unset = RecallFilter.model_validate({"source_type": "x"})
    explicit = RecallFilter.model_validate({"source_type": "x", "source_name": None})
    assert "source_name" not in unset.model_fields_set
    assert "source_name" in explicit.model_fields_set


# ---------------------------------------------------------------------------
# $date typed literal (§4) — parse + malformed raises; tz-naive → UTC
# ---------------------------------------------------------------------------


def test_date_literal_parses_in_metadata_value_position() -> None:
    model = RecallFilter.model_validate({"metadata.ts": {"$gt": {"$date": "2026-04-05T00:00:00Z"}}})
    assert isinstance(model, RecallFilter)


def test_date_literal_as_bare_metadata_value() -> None:
    model = RecallFilter.model_validate({"metadata.ts": {"$date": "2026-04-05T00:00:00Z"}})
    assert isinstance(model, RecallFilter)


def test_malformed_date_literal_raises() -> None:
    with pytest.raises(RecallFilterValidationError):
        RecallFilter.model_validate({"metadata.ts": {"$date": "not-a-date"}})


def test_malformed_date_literal_in_operand_raises() -> None:
    with pytest.raises(RecallFilterValidationError):
        RecallFilter.model_validate({"metadata.ts": {"$gt": {"$date": "13/13/2026"}}})


# ---------------------------------------------------------------------------
# Datetime normalization (§10.4) — tz-naive → UTC
# ---------------------------------------------------------------------------


def test_tz_naive_date_key_normalized_to_utc() -> None:
    model = RecallFilter.model_validate({"occurred_at": "2026-04-05T00:00:00"})
    assert isinstance(model.occurred_at, datetime)
    assert model.occurred_at.tzinfo is not None
    # Specifically normalized to UTC: zero offset.
    assert model.occurred_at.utcoffset().total_seconds() == 0


def test_tz_naive_datetime_object_normalized_to_utc() -> None:
    model = RecallFilter.model_validate({"occurred_at": datetime(2026, 4, 5, 0, 0, 0)})
    assert model.occurred_at.tzinfo is not None
    assert model.occurred_at.utcoffset().total_seconds() == 0


def test_tz_naive_in_date_ops_normalized_to_utc() -> None:
    model = RecallFilter.model_validate({"occurred_at": {"$gte": "2026-04-05T00:00:00"}})
    assert isinstance(model.occurred_at, DateOps)
    assert model.occurred_at.gte.tzinfo is not None
    assert model.occurred_at.gte.utcoffset().total_seconds() == 0


def test_tz_aware_date_preserved_as_utc() -> None:
    # A value already in UTC stays UTC.
    aware = datetime(2026, 4, 5, 0, 0, 0, tzinfo=UTC)
    model = RecallFilter.model_validate({"occurred_at": aware})
    assert model.occurred_at.utcoffset().total_seconds() == 0


# ---------------------------------------------------------------------------
# Date-key serialization contract
#
# Byte-identity to the RAW input is NOT achievable for date keys: tz-naive →
# UTC normalization rewrites the value, and ``model_dump`` (python mode) returns
# ``datetime`` objects rather than strings. The contract is instead:
#   (a) idempotency — validate → dump → validate → dump is stable, and
#   (b) the JSON-mode dump is a canonical UTC ISO-8601 string that parses back
#       to the same instant (parse-equivalence, not exact-byte, so the contract
#       survives a "Z" vs "+00:00" rendering choice).
# Structural (non-date) cases keep exact byte-stable round-trips elsewhere.
# ---------------------------------------------------------------------------


def _parse_iso_utc(value: str) -> datetime:
    """Parse an ISO-8601 string to an aware UTC datetime (accepts ``Z``)."""
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


@pytest.mark.parametrize(
    "wire",
    [
        {"occurred_at": {"$gte": "2026-04-05T00:00:00Z"}},
        {"occurred_at": "2026-04-05T00:00:00"},  # tz-naive
        {"occurred_at": "2026-04-05T00:00:00+00:00"},
        {
            "source_name": {"$in": ["linear"]},
            "occurred_at": {"$gte": "2026-04-05T00:00:00Z"},
            "metadata.tag": {"$in": ["a"]},
        },
    ],
)
def test_date_key_dump_is_idempotent_python_mode(wire: dict) -> None:
    # validate → dump → validate → dump is stable (python mode).
    first = RecallFilter.model_validate(wire)
    dump1 = first.model_dump(by_alias=True, exclude_none=True)
    dump2 = RecallFilter.model_validate(dump1).model_dump(by_alias=True, exclude_none=True)
    assert dump1 == dump2


@pytest.mark.parametrize(
    "wire",
    [
        {"occurred_at": {"$gte": "2026-04-05T00:00:00Z"}},
        {"occurred_at": "2026-04-05T00:00:00"},
        {"occurred_at": "2026-04-05T00:00:00+00:00"},
    ],
)
def test_date_key_dump_is_idempotent_json_mode(wire: dict) -> None:
    # validate → json-dump → validate → json-dump is stable (json mode).
    first = RecallFilter.model_validate(wire)
    dump1 = first.model_dump(by_alias=True, exclude_none=True, mode="json")
    dump2 = RecallFilter.model_validate(dump1).model_dump(by_alias=True, exclude_none=True, mode="json")
    assert dump1 == dump2


def test_date_ops_json_dump_is_canonical_utc_iso() -> None:
    # The JSON-mode dump of a date operand is a canonical UTC ISO-8601 string
    # that parses back to the same instant as the input (parse-equivalence).
    wire = {"occurred_at": {"$gte": "2026-04-05T00:00:00Z"}}
    dumped = RecallFilter.model_validate(wire).model_dump(by_alias=True, exclude_none=True, mode="json")
    serialized = dumped["occurred_at"]["$gte"]
    assert isinstance(serialized, str)
    assert _parse_iso_utc(serialized) == _parse_iso_utc("2026-04-05T00:00:00Z")


def test_scalar_date_key_json_dump_is_canonical_utc_iso() -> None:
    # A tz-naive scalar date key normalizes to UTC and dumps as a canonical
    # ISO string at the same instant.
    wire = {"occurred_at": "2026-04-05T00:00:00"}
    dumped = RecallFilter.model_validate(wire).model_dump(by_alias=True, exclude_none=True, mode="json")
    serialized = dumped["occurred_at"]
    assert isinstance(serialized, str)
    assert _parse_iso_utc(serialized) == datetime(2026, 4, 5, 0, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Operand opacity
# ---------------------------------------------------------------------------


def test_operand_opacity_eq_literal_object_does_not_raise() -> None:
    # {"$eq": {"$or": [1,2]}} matches the LITERAL object — never re-parsed
    # as a logical clause, so it must NOT raise.
    model = RecallFilter.model_validate({"metadata.x": {"$eq": {"$or": [1, 2]}}})
    assert isinstance(model, RecallFilter)


def test_operand_opacity_roundtrips() -> None:
    wire = {"metadata.x": {"$eq": {"$or": [1, 2]}}}
    assert _roundtrip(wire) == wire


def test_operand_opacity_in_list_of_objects() -> None:
    # $in operand is a list of opaque literals.
    model = RecallFilter.model_validate({"metadata.x": {"$in": [{"$gt": 1}, {"nested": {"$or": []}}]}})
    assert isinstance(model, RecallFilter)


# ---------------------------------------------------------------------------
# $not — document form + field form; $nor nonempty array
# ---------------------------------------------------------------------------


def test_not_document_form_validates() -> None:
    # {$not: {<filter>}} negates a whole filter.
    model = RecallFilter.model_validate({"$not": {"source_type": "connection"}})
    assert isinstance(model, RecallFilter)
    assert model.not_ is not None


def test_not_document_form_roundtrips() -> None:
    wire = {"$not": {"source_type": "connection"}}
    assert _roundtrip(wire) == wire


def test_not_field_form_validates() -> None:
    # {<field>: {$not: {<op-expr>}}} negates the field's op-expression.
    model = RecallFilter.model_validate({"source_type": {"$not": {"$eq": "connection"}}})
    assert isinstance(model.source_type, StringOps)
    assert model.source_type.not_ is not None


def test_not_field_form_on_date_key_validates() -> None:
    model = RecallFilter.model_validate({"occurred_at": {"$not": {"$gt": "2026-01-01T00:00:00Z"}}})
    assert isinstance(model.occurred_at, DateOps)
    assert model.occurred_at.not_ is not None


def test_not_field_form_bare_scalar_raises() -> None:
    # The field-form $not must take an operator-expression, not a bare
    # scalar — {field: {"$not": 5}} raises, matching MongoDB.
    with pytest.raises(RecallFilterValidationError):
        RecallFilter.model_validate({"source_type": {"$not": 5}})


def test_nor_nonempty_array_validates() -> None:
    model = RecallFilter.model_validate({"$nor": [{"source_type": "a"}, {"source_name": "b"}]})
    assert model.nor_ is not None
    assert len(model.nor_) == 2


def test_nor_is_not_desugared_here() -> None:
    # $nor is validated as a nonempty array of filters but NOT desugared
    # to $not($or[...]) — desugaring happens in the later AST step.
    model = RecallFilter.model_validate({"$nor": [{"source_type": "a"}]})
    assert model.nor_ is not None
    # The validated model keeps the $nor form on round-trip.
    assert "$nor" in model.model_dump(by_alias=True, exclude_none=True)


# ---------------------------------------------------------------------------
# Nested logical recursion
# ---------------------------------------------------------------------------


def test_nested_logical_recursion_validates() -> None:
    wire = {
        "$and": [
            {"source_type": "connection"},
            {"$or": [{"source_name": "linear"}, {"source_name": "slack"}]},
            {"$not": {"title": {"$exists": False}}},
        ]
    }
    model = RecallFilter.model_validate(wire)
    assert isinstance(model, RecallFilter)
    assert model.and_ is not None
    assert len(model.and_) == 3


def test_nested_logical_recursion_roundtrips() -> None:
    wire = {
        "$and": [
            {"source_type": "connection"},
            {"$or": [{"source_name": "linear"}, {"source_name": "slack"}]},
        ]
    }
    assert _roundtrip(wire) == wire


def test_invalid_inside_nested_logical_raises() -> None:
    # An invalid clause nested inside $or must still raise (recursion).
    with pytest.raises(RecallFilterValidationError):
        RecallFilter.model_validate({"$or": [{"occurred_at": {"$exists": True}}]})


def test_unknown_key_inside_nested_logical_raises() -> None:
    with pytest.raises(RecallFilterValidationError):
        RecallFilter.model_validate({"$and": [{"priority": 5}]})


# ---------------------------------------------------------------------------
# Kwarg construction — same validator
# ---------------------------------------------------------------------------


def test_kwarg_construction_validates() -> None:
    model = RecallFilter(
        source_name="linear",
        occurred_at=DateOps(gte=datetime(2026, 4, 5, tzinfo=UTC)),
    )
    assert model.source_name == "linear"


def test_kwarg_construction_unknown_key_raises() -> None:
    with pytest.raises(RecallFilterValidationError):
        RecallFilter(priority=5)  # type: ignore[call-arg]


def test_kwarg_and_dict_forms_produce_same_wire() -> None:
    kwarg = RecallFilter(source_name="linear")
    wire = RecallFilter.model_validate({"source_name": "linear"})
    dump_kwarg = kwarg.model_dump(by_alias=True, exclude_none=True)
    dump_wire = wire.model_dump(by_alias=True, exclude_none=True)
    assert dump_kwarg == dump_wire == {"source_name": "linear"}


# ---------------------------------------------------------------------------
# RecallFilterUnsupportedError — class is exported now (raised by later compilers)
# ---------------------------------------------------------------------------


def test_unsupported_error_constructible() -> None:
    # The class is exported now even though compilers (which raise it) are
    # later tickets. Keyword-only signature carrying path + reason.
    err = RecallFilterUnsupportedError(path="metadata.x", reason="no backend support")
    assert isinstance(err, Exception)
    assert err.path == "metadata.x"
    assert err.reason == "no backend support"


def test_unsupported_error_is_khora_error() -> None:
    from khora.exceptions import KhoraError

    assert isinstance(RecallFilterUnsupportedError(path="p", reason="r"), KhoraError)


# ---------------------------------------------------------------------------
# Recursion-depth guard — over-deep filters fail cleanly, not as RecursionError
# ---------------------------------------------------------------------------


def test_excessively_deep_metadata_nesting_raises_clean_error() -> None:
    # A pathologically deep equality operand must raise RecallFilterValidationError
    # (code max_depth_exceeded) from the iterative depth guard — never an uncaught
    # RecursionError from the recursive metadata-grammar walk.
    deep: dict = {"$gt": 1}
    for _ in range(400):
        deep = {"a": deep}
    with pytest.raises(RecallFilterValidationError) as exc:
        RecallFilter.model_validate({"metadata.x": deep})
    assert any(e.code == "max_depth_exceeded" for e in exc.value.errors)


def test_excessively_deep_logical_nesting_raises_clean_error() -> None:
    # Deeply nested $and must fail as RecallFilterValidationError before pydantic
    # recurses the model that deep (which would otherwise hit RecursionError).
    deep: dict = {"source_name": "linear"}
    for _ in range(400):
        deep = {"$and": [deep]}
    with pytest.raises(RecallFilterValidationError):
        RecallFilter.model_validate(deep)


def test_reasonable_nesting_within_limit_passes() -> None:
    # A moderately nested filter (well under the depth bound) validates fine.
    deep: dict = {"source_name": "linear"}
    for _ in range(8):
        deep = {"$and": [deep]}
    RecallFilter.model_validate(deep)
