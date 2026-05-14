"""EventBridge-style filter pattern evaluator for semantic hooks.

Pure-function evaluator for `SemanticFilter.match` patterns. Modeled on
AWS EventBridge filter patterns:
https://docs.aws.amazon.com/eventbridge/latest/userguide/eb-create-pattern-operators.html

Supported operators (per-key value position):

- String equality:    ``"key": ["v1", "v2"]``         OR over values
- ``prefix``:         ``"key": [{"prefix": "Ac"}]``
- ``suffix``:         ``"key": [{"suffix": "rp"}]``
- ``equals-ignore-case``: ``"key": [{"equals-ignore-case": "AcmE"}]``
- ``wildcard``:       ``"key": [{"wildcard": "Ac*me?"}]``  (* = 0+ chars, ? = 1 char)
- ``numeric``:        ``"key": [{"numeric": [">=", 0.8]}]``
                      ``"key": [{"numeric": [">=", 0.5, "<", 0.9]}]``  (multi-op AND)
- ``anything-but``:   ``"key": [{"anything-but": ["a", "b"]}]``
                      ``"key": [{"anything-but": {"prefix": "test_"}}]``
- ``exists``:         ``"key": [{"exists": True}]`` or ``False``
- ``contains-all``:   ``"key": [{"contains-all": ["x", "y"]}]``  (value must be list/tuple)

Top-level operators:

- ``$or``:            ``match["$or"] = [<pattern1>, <pattern2>]``

Top-level keys other than ``$or`` are combined with implicit AND. A list
of values for a key is OR (any one item that matches passes the key).

This module is pure: no I/O, no async, no network. It runs on every
event-subscription pair in the dispatcher hot path. Wildcard glob
patterns are cached at module level keyed by the literal pattern string
so subsequent evaluations skip regex compilation.

Nested key access (dot notation) is intentionally NOT supported. Operators
that need to match nested data should pre-flatten the value into
``event.data`` before dispatch.
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Wildcard compilation cache
# ---------------------------------------------------------------------------

_WILDCARD_CACHE: dict[str, re.Pattern[str]] = {}

_NUMERIC_OPS = {">=", "<=", "=", "!=", ">", "<"}

_MISSING = object()


def _compile_wildcard(pattern: str) -> re.Pattern[str]:
    cached = _WILDCARD_CACHE.get(pattern)
    if cached is not None:
        return cached
    # Translate glob to regex: * → .*, ? → ., everything else escaped.
    parts: list[str] = []
    for ch in pattern:
        if ch == "*":
            parts.append(".*")
        elif ch == "?":
            parts.append(".")
        else:
            parts.append(re.escape(ch))
    compiled = re.compile("^" + "".join(parts) + "$", re.DOTALL)
    _WILDCARD_CACHE[pattern] = compiled
    return compiled


# ---------------------------------------------------------------------------
# Operator evaluators
# ---------------------------------------------------------------------------


def _numeric_compare(value: Any, ops: list[Any]) -> bool:
    """Evaluate one or more (op, threshold) pairs against value (all AND)."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    if len(ops) % 2 != 0:
        return False
    for i in range(0, len(ops), 2):
        op = ops[i]
        threshold = ops[i + 1]
        if op not in _NUMERIC_OPS:
            return False
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
            return False
        if op == ">=" and not (value >= threshold):
            return False
        if op == "<=" and not (value <= threshold):
            return False
        if op == ">" and not (value > threshold):
            return False
        if op == "<" and not (value < threshold):
            return False
        if op == "=" and not (value == threshold):
            return False
        if op == "!=" and not (value != threshold):
            return False
    return True


def _anything_but(value: Any, spec: Any) -> bool:
    """anything-but: True if value is NOT in/matching the spec."""
    if isinstance(spec, list):
        return value not in spec
    if isinstance(spec, dict):
        # Negated single-operator form: {"prefix": "test_"} etc.
        if "prefix" in spec:
            return not (isinstance(value, str) and value.startswith(spec["prefix"]))
        if "suffix" in spec:
            return not (isinstance(value, str) and value.endswith(spec["suffix"]))
        if "wildcard" in spec:
            return not (isinstance(value, str) and bool(_compile_wildcard(spec["wildcard"]).match(value)))
        # Unknown negation form — fail closed (don't accidentally match).
        return False
    # Scalar form: anything-but a single literal.
    return value != spec


def _evaluate_operator(op_dict: dict[str, Any], value: Any, present: bool) -> bool:
    """Evaluate a single operator-object against the event value.

    ``present`` is True when the key existed in event.data (even if value
    is None or ""). The ``exists`` operator needs this distinction.
    """
    # exists is evaluated against presence, not value.
    if "exists" in op_dict:
        want = bool(op_dict["exists"])
        return present == want

    # Every other operator implies the key must be present.
    if not present:
        return False

    if "prefix" in op_dict:
        if not isinstance(value, str):
            return False
        return value.startswith(op_dict["prefix"])

    if "suffix" in op_dict:
        if not isinstance(value, str):
            return False
        return value.endswith(op_dict["suffix"])

    if "equals-ignore-case" in op_dict:
        if not isinstance(value, str):
            return False
        return value.lower() == str(op_dict["equals-ignore-case"]).lower()

    if "wildcard" in op_dict:
        if not isinstance(value, str):
            return False
        return bool(_compile_wildcard(op_dict["wildcard"]).match(value))

    if "numeric" in op_dict:
        ops = op_dict["numeric"]
        if not isinstance(ops, list):
            return False
        return _numeric_compare(value, ops)

    if "anything-but" in op_dict:
        return _anything_but(value, op_dict["anything-but"])

    if "contains-all" in op_dict:
        needles = op_dict["contains-all"]
        if not isinstance(needles, (list, tuple)):
            return False
        if not isinstance(value, (list, tuple)):
            return False
        return all(item in value for item in needles)

    # Unknown operator object — fail closed.
    return False


def _match_key(value_patterns: list[Any], value: Any, present: bool) -> bool:
    """Per-key OR over the list of value patterns."""
    for entry in value_patterns:
        if isinstance(entry, dict):
            if _evaluate_operator(entry, value, present):
                return True
        else:
            # Literal equality. Only meaningful when key is present.
            if present and value == entry:
                return True
    return False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def matches(pattern: dict[str, Any] | None, event_data: dict[str, Any]) -> bool:
    """Evaluate an EventBridge-style pattern against event.data.

    Returns True when the event matches. None / empty pattern → True.
    """
    if not pattern:
        return True

    for key, value_patterns in pattern.items():
        if key == "$or":
            if not isinstance(value_patterns, list):
                return False
            if not any(matches(branch, event_data) for branch in value_patterns):
                return False
            continue

        # Per-key list of value patterns. Normalize a non-list to single-item.
        if not isinstance(value_patterns, list):
            value_patterns = [value_patterns]

        raw = event_data.get(key, _MISSING)
        present = raw is not _MISSING
        value = None if not present else raw

        if not _match_key(value_patterns, value, present):
            return False

    return True
