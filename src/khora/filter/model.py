"""Deterministic recall-filter model and validator.

This module defines the public, MongoDB-flavored filter document accepted by
recall operations. It is a *typed* pydantic model with named system-key fields
(IDE autocomplete on the closed system-key whitelist, per-value-type operator
subtypes at construction time) plus a recursive grammar walk over the free-form
``metadata`` block that pydantic cannot express on ``dict[str, Any]``.

Two construction paths, one validator:

* ``RecallFilter(source_name="linear", ...)`` — kwargs, recommended in Python.
* ``RecallFilter.model_validate({...})`` — the wire/dict form (HTTP bodies,
  agent tool calls, config files).

Both flow through the same validation chain. The model *validates*; it does not
lower to an intermediate representation or compile to a backend query — those
steps live elsewhere.

Wire shape: aliases preserve the JSON form (``$or``, ``$gte``, ...). A bare
``metadata`` value is whole-blob equality; flat ``metadata.<path>`` predicate
keys are folded out before the closed-key check runs and re-emitted flat on
dump, so ``model_dump(by_alias=True, exclude_none=True)`` round-trips byte-stable
with the input.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import pydantic
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    StrictBool,
    ValidatorFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

from khora.exceptions import KhoraError

__all__ = [
    "DateOps",
    "Op",
    "RecallFilter",
    "RecallFilterUnsupportedError",
    "RecallFilterValidationError",
    "StringOps",
    "SYSTEM_KEYS",
]


# --------------------------------------------------------------------------- #
# Operator vocabulary and the closed system-key whitelist.
# --------------------------------------------------------------------------- #


class Op(str, Enum):
    """The filter operator vocabulary (wire literals).

    ``str`` mixin so ``Op.EQ == "$eq"`` and ``Op.EQ.value == "$eq"`` both hold.
    """

    EQ = "$eq"
    NE = "$ne"
    GT = "$gt"
    GTE = "$gte"
    LT = "$lt"
    LTE = "$lte"
    IN = "$in"
    NIN = "$nin"
    EXISTS = "$exists"
    AND = "$and"
    OR = "$or"
    NOR = "$nor"
    NOT = "$not"
    DATE = "$date"


# The ten filterable system keys. Two date keys live on the recall chunk; the
# other eight are denormalized document keys. ``metadata`` and the logical
# operators are intentionally excluded.
SYSTEM_KEYS: frozenset[str] = frozenset(
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

# Operators whose operand is a list ($in/$nin), and the full set of operators
# valid as keys inside a metadata operator-expression.
_LIST_OPS: frozenset[str] = frozenset({Op.IN, Op.NIN})
_METADATA_FIELD_OPS: frozenset[str] = frozenset(
    {
        Op.EQ,
        Op.NE,
        Op.GT,
        Op.GTE,
        Op.LT,
        Op.LTE,
        Op.IN,
        Op.NIN,
        Op.EXISTS,
        Op.NOT,
    }
)

# Carrier alias for folded ``metadata.<path>`` predicates. The double-underscore
# form cannot collide with a real ``metadata.<path>`` key or a ``$``-operator.
_FOLDED_ALIAS = "__folded_predicates__"


# --------------------------------------------------------------------------- #
# Error types.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FieldError:
    """A single structured validation failure.

    ``path`` is a JSON-Pointer-ish location (e.g. ``"/$or/0/source_type"`` or
    ``"/metadata.tag/$in"``). ``code`` is a stable machine reason code, ``message``
    is human-readable, and ``allowed`` carries the permitted values when the code
    is a whitelist violation.
    """

    path: str
    code: str
    message: str
    allowed: list[str] | None = None


class RecallFilterValidationError(KhoraError):
    """A recall filter failed validation.

    Carries a structured ``errors`` list (one entry per offending field) so SDKs
    can translate it into an HTTP 400 with a structured body. Raised by both
    validation regimes: the pydantic-structural pass (converted via
    :meth:`from_pydantic`) and the recursive metadata-grammar walk.
    """

    def __init__(self, errors: list[FieldError]) -> None:
        self.errors = errors
        summary = "; ".join(f"{e.path}: {e.message}" for e in errors) or "invalid filter"
        super().__init__(summary)

    @classmethod
    def from_pydantic(cls, exc: pydantic.ValidationError) -> RecallFilterValidationError:
        """Convert a pydantic ``ValidationError`` into structured field errors."""
        errors: list[FieldError] = []
        for err in exc.errors():
            loc = err.get("loc", ())
            path = "/" + "/".join(str(part) for part in loc)
            errors.append(
                FieldError(
                    path=path,
                    code=str(err.get("type", "validation_error")),
                    message=str(err.get("msg", "")),
                )
            )
        return cls(errors)


class RecallFilterUnsupportedError(KhoraError):
    """A backend compiler cannot honor a filter predicate.

    Exported now for the public surface; raised by the compilers that lower a
    validated filter to a backend query (a later step), not by this validator.
    """

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


def _raise(path: str, code: str, message: str, *, allowed: list[str] | None = None) -> None:
    raise RecallFilterValidationError([FieldError(path=path, code=code, message=message, allowed=allowed)])


# --------------------------------------------------------------------------- #
# Operator submodels for the typed system keys.
#
# The per-key operator subsets fall out of these submodels for free via
# ``extra="forbid"``: DateOps has no ``$exists`` field, so ``$exists`` on a date
# key fails to match the DateOps arm; StringOps has no range ops, so a range op
# on a string key fails the StringOps arm. Both surface as a pydantic
# ValidationError that the wrap funnel converts.
# --------------------------------------------------------------------------- #


class StringOps(BaseModel):
    """Operator submodel for string-typed system keys."""

    eq: str | None = Field(None, alias="$eq")
    ne: str | None = Field(None, alias="$ne")
    in_: list[str] | None = Field(None, alias="$in")
    nin: list[str] | None = Field(None, alias="$nin")
    # StrictBool (not plain bool) so the typed-key path matches the metadata
    # walk's strict isinstance(bool) check — "yes"/"true"/1 must raise, not coerce.
    exists: StrictBool | None = Field(None, alias="$exists")
    not_: StringOps | None = Field(None, alias="$not")  # field-level $not

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class DateOps(BaseModel):
    """Operator submodel for datetime-typed system keys.

    Has no ``$exists`` field by design — ``$exists`` on a date key must raise.
    Timezone-naive operands are normalized to UTC immediately.
    """

    eq: datetime | None = Field(None, alias="$eq")
    ne: datetime | None = Field(None, alias="$ne")
    gt: datetime | None = Field(None, alias="$gt")
    gte: datetime | None = Field(None, alias="$gte")
    lt: datetime | None = Field(None, alias="$lt")
    lte: datetime | None = Field(None, alias="$lte")
    in_: list[datetime] | None = Field(None, alias="$in")
    nin: list[datetime] | None = Field(None, alias="$nin")
    not_: DateOps | None = Field(None, alias="$not")  # field-level $not

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    @field_validator("eq", "ne", "gt", "gte", "lt", "lte", mode="after")
    @classmethod
    def _utc_scalar(cls, value: datetime | None) -> datetime | None:
        return _normalize_utc(value)

    @field_validator("in_", "nin", mode="after")
    @classmethod
    def _utc_list(cls, value: list[datetime] | None) -> list[datetime] | None:
        if value is None:
            return None
        return [_normalize_utc(item) for item in value]


def _normalize_utc(value: datetime | None) -> datetime | None:
    """Normalize a tz-naive datetime to UTC; leave tz-aware values untouched."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


# --------------------------------------------------------------------------- #
# The filter document.
# --------------------------------------------------------------------------- #


class RecallFilter(BaseModel):
    """A deterministic recall filter (typed, MongoDB-flavored).

    The closed top-level key set is the ten system keys, the bare ``metadata``
    blob, the logical operators (``$and``/``$or``/``$nor``/``$not``), and any flat
    ``metadata.<path>`` predicate key. Unknown top-level keys raise.

    Bare-value sugar: a scalar is ``$eq``; a *list* is ``$eq`` exact-array
    equality (**not** ``$in`` — use ``$in`` for membership); a nested dict on a
    metadata sub-path is whole-subdocument equality. ``null`` is an active
    null-or-missing match — to not filter on a key, omit it.

    Object-equality on the ``metadata`` blob compares normalized JSON, not source
    bytes (key order and numeric/whitespace formatting are not significant).
    """

    # Date system keys (range + set, no $exists).
    occurred_at: datetime | DateOps | None = None
    created_at: datetime | DateOps | None = None
    source_timestamp: datetime | DateOps | None = None
    # String system keys (eq + set + exists). A bare list is $eq exact-array.
    source_type: str | list[str] | StringOps | None = None
    source_name: str | list[str] | StringOps | None = None
    source_url: str | list[str] | StringOps | None = None
    external_id: str | list[str] | StringOps | None = None
    content_type: str | list[str] | StringOps | None = None
    source: str | list[str] | StringOps | None = None
    title: str | list[str] | StringOps | None = None
    # Whole-metadata-blob $eq equality (embedded-doc equality). Flat
    # metadata.<path> predicate keys are folded onto the carrier below, not here.
    metadata: dict[str, Any] | None = None
    # Logical composition — aliases match the wire shape.
    and_: list[RecallFilter] | None = Field(None, alias="$and")
    or_: list[RecallFilter] | None = Field(None, alias="$or")
    nor_: list[RecallFilter] | None = Field(None, alias="$nor")
    not_: RecallFilter | None = Field(None, alias="$not")
    # Reserved internal carrier for folded metadata.<path> predicates. Threaded
    # through validation per-instance (recursion-safe); re-emitted flat on dump.
    folded_predicates_: dict[str, Any] = Field(default_factory=dict, alias=_FOLDED_ALIAS)

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    @field_validator("occurred_at", "created_at", "source_timestamp", mode="after")
    @classmethod
    def _utc_date_key(cls, value: datetime | DateOps | None) -> datetime | DateOps | None:
        """Normalize a bare-scalar date-key value to UTC (DateOps self-normalizes)."""
        if isinstance(value, datetime):
            return _normalize_utc(value)
        return value

    # ----- regime (a): structural funnel + dot-path fold ------------------- #

    @model_validator(mode="wrap")
    @classmethod
    def _convert_pydantic_errors(
        cls,
        data: Any,
        handler: ValidatorFunctionWrapHandler,
    ) -> RecallFilter:
        """Funnel both construction paths into a single error type.

        Catches the pydantic-structural ValidationError (unknown top-level key,
        ``$exists`` on a date key, a range op on a string key, wrong list/array
        shape) and re-raises it as ``RecallFilterValidationError``. Errors raised
        by the ``before`` fold and the ``after`` metadata walk are already the
        right type and propagate unconverted.
        """
        try:
            return handler(data)
        except pydantic.ValidationError as exc:
            raise RecallFilterValidationError.from_pydantic(exc) from exc

    @model_validator(mode="before")
    @classmethod
    def _fold_dot_paths(cls, data: Any) -> Any:
        """Fold flat ``metadata.<path>`` keys onto the carrier before forbid runs.

        A bare ``metadata`` key (no trailing dot) is left to the typed field as
        whole-blob equality. Each (possibly nested) filter carries its own folds
        on its own instance, so nested ``$and``/``$or`` recursion does not
        cross-contaminate.
        """
        if not isinstance(data, dict):
            return data
        folded: dict[str, Any] = {}
        rest: dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(key, str) and key.startswith("metadata."):
                folded[key] = value
            else:
                rest[key] = value
        if folded:
            rest[_FOLDED_ALIAS] = folded
        return rest

    # ----- regime (b): recursive metadata-grammar walk -------------------- #

    @model_validator(mode="after")
    def _validate_metadata_grammar(self) -> RecallFilter:
        """Enforce the grammar pydantic cannot express on free-form values.

        Requires ``$and``/``$or``/``$nor`` to be nonempty arrays, then walks the
        bare ``metadata`` blob and every folded ``metadata.<path>`` predicate:
        operator whitelist, ``$in``/``$nin`` lists, ``$exists`` bool, the
        operator-position closure, operand opacity, and ``$date``
        literal parsing.
        """
        # $and / $or / $nor must be nonempty when present (set, not None). $not
        # takes a single filter document and is unaffected.
        for alias, branches in (("$and", self.and_), ("$or", self.or_), ("$nor", self.nor_)):
            if branches is not None and len(branches) == 0:
                _raise(f"/{alias}", "empty_logical_array", f"{alias} must be a nonempty array")
        if self.metadata is not None:
            # Bare metadata blob: each value is a per-field predicate.
            for field_name, predicate in self.metadata.items():
                _walk_predicate(predicate, f"/metadata/{field_name}")
        for dotted_key, predicate in self.folded_predicates_.items():
            _walk_predicate(predicate, f"/{dotted_key}")
        return self

    # ----- byte-stable serialization -------------------------------------- #

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Re-emit folded ``metadata.<path>`` predicates as flat keys."""
        out = handler(self)
        out.pop(_FOLDED_ALIAS, None)
        out.update(self.folded_predicates_)
        return out


# --------------------------------------------------------------------------- #
# Recursive metadata-grammar walk (regime b helpers).
# --------------------------------------------------------------------------- #


def _walk_predicate(predicate: Any, path: str) -> None:
    """Validate one metadata field predicate (bare value, operator-expr, or dict).

    A scalar / list / non-operator dict is an equality operand (opaque — not
    recursed). A dict whose keys are all ``$``-operators is an operator
    expression. A value-dict mixing ``$``-keys with non-``$`` keys raises.
    """
    if not isinstance(predicate, dict):
        # Bare scalar / list ⇒ $eq sugar. Opaque literal.
        return
    if not predicate:
        # Empty {} is whole-subdocument equality (opaque).
        return

    dollar_keys = [k for k in predicate if isinstance(k, str) and k.startswith("$")]
    non_dollar_keys = [k for k in predicate if not (isinstance(k, str) and k.startswith("$"))]

    if dollar_keys and non_dollar_keys:
        # An operator-expression must not mix $-ops with plain keys.
        _raise(
            path,
            "mixed_operator_and_field",
            "value object mixes operator ($) keys with non-operator keys; "
            'use dot-notation to address a nested field (e.g. {"metadata.a.b": {...}})',
        )

    if not dollar_keys:
        # All non-$ keys ⇒ whole-subdocument equality. The subdoc is matched as
        # a literal object, but a $-operator nested at ANY depth inside a nested
        # dict is rejected: operators are not applied inside an equality match, so
        # the caller meant a nested-path predicate, which must use dot-notation
        # (e.g. {"metadata.labels.priority": {"$gte": 5}}). The scan descends
        # through nested dicts only — list elements are opaque exact-array
        # literals — and never enters a comparison operator's operand (those are
        # reached only via the operator branch below, which treats them as
        # opaque).
        _reject_nested_operator(predicate, path)
        return

    # A bare {"$date": "<ISO-8601>"} in field-value position is a typed literal
    # (equivalent to {"$eq": {"$date": ...}}), not an operator expression.
    if set(predicate.keys()) == {Op.DATE.value}:
        _parse_date_literal(predicate[Op.DATE.value], f"{path}/$date")
        return

    # Operator expression: validate each operator and its operand.
    for op_key, operand in predicate.items():
        _walk_operator(op_key, operand, f"{path}/{op_key}")


def _reject_nested_operator(value: Any, path: str) -> None:
    """Raise if a ``$``-prefixed key appears anywhere inside an equality operand.

    Descends through nested DICTS only. List elements are opaque exact-array
    literals and are not inspected — there is no dot-addressable field name to
    bind an operator to, so a ``$``-op inside a list element is data, consistent
    with bare-list / ``$in`` / ``$eq`` list operands. This runs only on
    equality-operand subtrees (whole-subdocument equality); it is never called on
    a comparison operator's operand, which stays opaque.
    """
    if isinstance(value, dict):
        for key, sub_value in value.items():
            if isinstance(key, str) and key.startswith("$"):
                _raise(
                    path,
                    "operator_nested_in_operand",
                    "operator ($) nested inside a whole-subdocument equality value; "
                    "operators are not applied inside an equality match — use "
                    'dot-notation for a nested-field predicate (e.g. {"metadata.a.b": {...}})',
                )
            _reject_nested_operator(sub_value, path)


def _walk_operator(op_key: str, operand: Any, path: str) -> None:
    """Validate a single ``$``-operator and its operand inside a metadata predicate."""
    if op_key not in _METADATA_FIELD_OPS:
        _raise(
            path,
            "unknown_operator",
            f"unsupported operator {op_key!r}",
            allowed=sorted(op.value for op in _METADATA_FIELD_OPS),
        )

    if op_key in _LIST_OPS:
        if not isinstance(operand, list):
            _raise(path, "operator_value_not_list", f"{op_key} value must be an array")
        for item in operand:  # type: ignore[union-attr]
            _check_literal_operand(item, path)
        return

    if op_key == Op.EXISTS:
        if not isinstance(operand, bool):
            _raise(path, "operator_value_not_bool", f"{op_key} value must be a boolean")
        return

    if op_key == Op.NOT:
        # Field-position $not negates an inner operator-expression. A bare scalar
        # operand is invalid (matches MongoDB).
        if not isinstance(operand, dict):
            _raise(path, "not_operand_not_expression", "$not value must be an operator expression")
        _walk_not_operand(operand, path)
        return

    # Remaining comparison ops ($eq/$ne/$gt/$gte/$lt/$lte): the operand is an
    # opaque literal, validated only for $date typed literals.
    _check_literal_operand(operand, path)


def _walk_not_operand(operand: dict[str, Any], path: str) -> None:
    """Validate a field-position ``$not`` operand: a pure operator expression.

    The negated operand must be all ``$``-operators (an operator-expression);
    a bare scalar or a mixed/plain object raises. Each inner operator is then
    validated like any other field operator.
    """
    dollar_keys = [k for k in operand if isinstance(k, str) and k.startswith("$")]
    non_dollar_keys = [k for k in operand if not (isinstance(k, str) and k.startswith("$"))]
    if not dollar_keys or non_dollar_keys:
        _raise(path, "not_operand_not_expression", "$not value must be an operator expression")
    for op_key, inner in operand.items():
        _walk_operator(op_key, inner, f"{path}/{op_key}")


def _check_literal_operand(operand: Any, path: str) -> None:
    """Validate a comparison operand — an opaque literal.

    Comparison operands ($eq/$ne/$gt/$gte/$lt/$lte, and each item of $in/$nin)
    are never re-parsed as clauses: a ``$or`` nested inside an ``$eq`` operand is
    matched as a literal object, not a logical clause, so it does NOT raise. The
    single recognized form is the ``{"$date": "<ISO-8601>"}`` typed literal,
    parsed/normalized here (a sole-key ``$date`` object); any other content is
    opaque data.
    """
    if isinstance(operand, dict) and set(operand.keys()) == {Op.DATE.value}:
        _parse_date_literal(operand[Op.DATE.value], f"{path}/$date")


def _parse_date_literal(value: Any, path: str) -> None:
    """Parse and normalize a ``$date`` typed literal; raise on malformed input."""
    if not isinstance(value, str):
        _raise(path, "date_literal_not_string", "$date value must be an ISO-8601 string")
    try:
        datetime.fromisoformat(value)  # type: ignore[arg-type]
    except ValueError:
        _raise(path, "date_literal_malformed", f"$date value {value!r} is not valid ISO-8601")


# Resolve the self-referential forward refs ("$not" on the submodels, the
# recursive logical operators on RecallFilter).
StringOps.model_rebuild()
DateOps.model_rebuild()
RecallFilter.model_rebuild()
