"""Unit tests for EventBridge-style match DSL (Issue #579 Phase 2 Item A)."""

from __future__ import annotations

from uuid import uuid4

import pytest

from khora.hooks.match_dsl import matches


@pytest.mark.unit
class TestMatchDSL:
    # 1. None / empty pattern matches anything
    def test_none_pattern_matches(self) -> None:
        assert matches(None, {"a": 1}) is True

    def test_empty_pattern_matches(self) -> None:
        assert matches({}, {"a": 1}) is True

    # 2. Simple string equality (list with one value)
    def test_string_equality_single(self) -> None:
        assert matches({"entity_type": ["ORGANIZATION"]}, {"entity_type": "ORGANIZATION"}) is True
        assert matches({"entity_type": ["ORGANIZATION"]}, {"entity_type": "PERSON"}) is False

    # 3. Multi-value OR
    def test_string_equality_multi_or(self) -> None:
        pattern = {"entity_type": ["ORGANIZATION", "PRODUCT"]}
        assert matches(pattern, {"entity_type": "ORGANIZATION"}) is True
        assert matches(pattern, {"entity_type": "PRODUCT"}) is True
        assert matches(pattern, {"entity_type": "PERSON"}) is False

    # 4. prefix positive + negative
    def test_prefix(self) -> None:
        pattern = {"name": [{"prefix": "Acme"}]}
        assert matches(pattern, {"name": "Acme Corp"}) is True
        assert matches(pattern, {"name": "Beta Corp"}) is False
        # Type mismatch: non-string never matches a string operator.
        assert matches(pattern, {"name": 42}) is False

    # 5. suffix positive + negative
    def test_suffix(self) -> None:
        pattern = {"name": [{"suffix": "Corp"}]}
        assert matches(pattern, {"name": "Acme Corp"}) is True
        assert matches(pattern, {"name": "Acme Inc"}) is False

    # 6. equals-ignore-case
    def test_equals_ignore_case(self) -> None:
        pattern = {"name": [{"equals-ignore-case": "AcmE"}]}
        assert matches(pattern, {"name": "acme"}) is True
        assert matches(pattern, {"name": "ACME"}) is True
        assert matches(pattern, {"name": "Acme"}) is True
        assert matches(pattern, {"name": "Beta"}) is False

    # 7. wildcard with * and ?
    def test_wildcard_star(self) -> None:
        pattern = {"name": [{"wildcard": "Ac*Corp"}]}
        assert matches(pattern, {"name": "Acme Corp"}) is True
        assert matches(pattern, {"name": "AcCorp"}) is True
        assert matches(pattern, {"name": "Beta Corp"}) is False

    def test_wildcard_question_mark(self) -> None:
        pattern = {"name": [{"wildcard": "Acm?"}]}
        assert matches(pattern, {"name": "Acme"}) is True
        assert matches(pattern, {"name": "AcmX"}) is True
        assert matches(pattern, {"name": "Acm"}) is False  # ? requires exactly one
        assert matches(pattern, {"name": "Acmee"}) is False

    # 8. numeric binary op
    @pytest.mark.parametrize(
        "op,threshold,value,expected",
        [
            (">=", 0.8, 0.9, True),
            (">=", 0.8, 0.8, True),
            (">=", 0.8, 0.7, False),
            ("=", 1.0, 1.0, True),
            ("=", 1.0, 1.5, False),
            ("<", 0.5, 0.4, True),
            ("<", 0.5, 0.5, False),
            (">", 0.5, 0.6, True),
            (">", 0.5, 0.5, False),
            ("<=", 0.5, 0.5, True),
            ("<=", 0.5, 0.6, False),
            ("!=", 1.0, 0.5, True),
            ("!=", 1.0, 1.0, False),
        ],
    )
    def test_numeric_binary(self, op: str, threshold: float, value: float, expected: bool) -> None:
        pattern = {"confidence": [{"numeric": [op, threshold]}]}
        assert matches(pattern, {"confidence": value}) is expected

    # 9. numeric multi-op AND
    def test_numeric_multi_op(self) -> None:
        pattern = {"confidence": [{"numeric": [">=", 0.5, "<", 0.9]}]}
        assert matches(pattern, {"confidence": 0.7}) is True
        assert matches(pattern, {"confidence": 0.5}) is True
        assert matches(pattern, {"confidence": 0.9}) is False
        assert matches(pattern, {"confidence": 0.4}) is False

    # 10. anything-but simple negation
    def test_anything_but_list(self) -> None:
        pattern = {"source_system": [{"anything-but": ["test", "staging"]}]}
        assert matches(pattern, {"source_system": "prod"}) is True
        assert matches(pattern, {"source_system": "test"}) is False
        assert matches(pattern, {"source_system": "staging"}) is False

    def test_anything_but_scalar(self) -> None:
        pattern = {"source_system": [{"anything-but": "test"}]}
        assert matches(pattern, {"source_system": "prod"}) is True
        assert matches(pattern, {"source_system": "test"}) is False

    # 11. anything-but with prefix
    def test_anything_but_prefix(self) -> None:
        pattern = {"source_system": [{"anything-but": {"prefix": "test_"}}]}
        assert matches(pattern, {"source_system": "prod_us"}) is True
        assert matches(pattern, {"source_system": "test_us"}) is False
        assert matches(pattern, {"source_system": "test_eu"}) is False

    # 12. exists: True for present key
    def test_exists_true_present(self) -> None:
        pattern = {"valid_until": [{"exists": True}]}
        assert matches(pattern, {"valid_until": "2026-01-01"}) is True
        # Present-but-None still counts as present.
        assert matches(pattern, {"valid_until": None}) is True

    def test_exists_true_missing(self) -> None:
        pattern = {"valid_until": [{"exists": True}]}
        assert matches(pattern, {"name": "Acme"}) is False

    # 13. exists: False for missing key
    def test_exists_false_missing(self) -> None:
        pattern = {"valid_until": [{"exists": False}]}
        assert matches(pattern, {"name": "Acme"}) is True

    def test_exists_false_present(self) -> None:
        pattern = {"valid_until": [{"exists": False}]}
        assert matches(pattern, {"valid_until": "2026-01-01"}) is False

    # 14. contains-all on list values
    def test_contains_all(self) -> None:
        pattern = {"tags": [{"contains-all": ["urgent", "customer"]}]}
        assert matches(pattern, {"tags": ["urgent", "customer", "vip"]}) is True
        assert matches(pattern, {"tags": ["customer", "urgent"]}) is True
        assert matches(pattern, {"tags": ["urgent"]}) is False
        # Type mismatch: contains-all on a non-list value returns False.
        assert matches(pattern, {"tags": "urgent"}) is False

    # 15. $or top-level — first branch matches
    def test_or_first_branch(self) -> None:
        pattern = {
            "$or": [
                {"is_active": [True]},
                {"valid_until": [{"exists": False}]},
            ]
        }
        assert matches(pattern, {"is_active": True, "valid_until": "2026-01-01"}) is True

    # 16. $or top-level — second branch matches
    def test_or_second_branch(self) -> None:
        pattern = {
            "$or": [
                {"is_active": [True]},
                {"valid_until": [{"exists": False}]},
            ]
        }
        assert matches(pattern, {"is_active": False}) is True  # no valid_until

    # 17. $or top-level — none match
    def test_or_none_match(self) -> None:
        pattern = {
            "$or": [
                {"is_active": [True]},
                {"valid_until": [{"exists": False}]},
            ]
        }
        assert matches(pattern, {"is_active": False, "valid_until": "2026-01-01"}) is False

    # 18. Implicit AND across top-level keys
    def test_implicit_and(self) -> None:
        pattern = {
            "entity_type": ["ORGANIZATION"],
            "name": [{"prefix": "Acme"}],
        }
        assert matches(pattern, {"entity_type": "ORGANIZATION", "name": "Acme Corp"}) is True
        # First key fails.
        assert matches(pattern, {"entity_type": "PERSON", "name": "Acme Corp"}) is False
        # Second key fails.
        assert matches(pattern, {"entity_type": "ORGANIZATION", "name": "Beta Corp"}) is False

    # 19. Dot notation is NOT silently supported. Documents the intentional
    # design choice: operators must pre-flatten nested data into event.data.
    def test_dot_notation_not_supported(self) -> None:
        # "entity.name" is treated as a literal key, not a nested-access path.
        pattern = {"entity.name": [{"prefix": "Acme"}]}
        # Nested structure: key "entity.name" is absent → no match.
        assert matches(pattern, {"entity": {"name": "Acme Corp"}}) is False
        # Literal key present → matches. Confirms the key is taken verbatim.
        assert matches(pattern, {"entity.name": "Acme Corp"}) is True

    # 20. Type mismatch for numeric op returns False (does not crash).
    def test_numeric_type_mismatch(self) -> None:
        pattern = {"confidence": [{"numeric": [">=", 0.5]}]}
        assert matches(pattern, {"confidence": "high"}) is False
        assert matches(pattern, {"confidence": None}) is False
        # Booleans are not numbers (despite Python's bool-is-int quirk).
        assert matches(pattern, {"confidence": True}) is False

    # Bonus: per-key list of operator objects is OR (EventBridge semantics).
    def test_per_key_or_across_operators(self) -> None:
        pattern = {"name": [{"prefix": "Acme"}, {"suffix": "Inc"}]}
        assert matches(pattern, {"name": "Acme Corp"}) is True
        assert matches(pattern, {"name": "Beta Inc"}) is True
        assert matches(pattern, {"name": "Beta Corp"}) is False

    # Bonus: missing key — every non-exists operator returns False.
    def test_missing_key_fails_non_exists(self) -> None:
        assert matches({"name": [{"prefix": "Acme"}]}, {}) is False
        assert matches({"confidence": [{"numeric": [">=", 0.5]}]}, {}) is False
        assert matches({"x": ["value"]}, {}) is False

    # Bonus: wildcard cache hit returns the same compiled pattern.
    def test_wildcard_cache(self) -> None:
        from khora.hooks.match_dsl import _WILDCARD_CACHE, _compile_wildcard

        before = len(_WILDCARD_CACHE)
        pat = "Ac*Corp_v2"  # unique-ish so we can assert insertion
        _compile_wildcard(pat)
        _compile_wildcard(pat)
        assert pat in _WILDCARD_CACHE
        assert len(_WILDCARD_CACHE) >= before + 1

    # Bonus: combined complex pattern from issue description.
    def test_complex_combined_pattern(self) -> None:
        pattern = {
            "entity_type": ["ORGANIZATION", "PRODUCT"],
            "confidence": [{"numeric": [">=", 0.8]}],
            "name": [{"prefix": "Acme"}],
            "source_system": [{"anything-but": ["test", "staging"]}],
            "$or": [
                {"is_active": [True]},
                {"valid_until": [{"exists": False}]},
            ],
        }
        event = {
            "entity_type": "ORGANIZATION",
            "confidence": 0.9,
            "name": "Acme Corp",
            "source_system": "prod",
            "is_active": True,
        }
        assert matches(pattern, event) is True

        # Flip one field at a time — each flip should reject.
        bad = dict(event, confidence=0.5)
        assert matches(pattern, bad) is False
        bad = dict(event, name="Beta Corp")
        assert matches(pattern, bad) is False
        bad = dict(event, source_system="test")
        assert matches(pattern, bad) is False
        bad = dict(event, is_active=False, valid_until="2026-01-01")
        assert matches(pattern, bad) is False


# ---------------------------------------------------------------------------
# Dispatcher integration test
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDispatcherMatchIntegration:
    async def test_dispatcher_filter_by_match_numeric(self) -> None:
        from unittest.mock import AsyncMock

        from khora.core.models.event import EventType, MemoryEvent
        from khora.hooks.dispatcher import HookDispatcher
        from khora.hooks.models import SemanticFilter

        d = HookDispatcher()
        cb = AsyncMock()
        f = SemanticFilter(
            name="high_confidence",
            match={"confidence": [{"numeric": [">=", 0.8]}]},
        )
        d.subscribe(EventType.ENTITY_CREATED, cb, filter=f)

        passing = MemoryEvent.entity_created(
            namespace_id=uuid4(),
            entity_id=uuid4(),
            data={"name": "Acme", "entity_type": "ORGANIZATION", "confidence": 0.9},
        )
        failing = MemoryEvent.entity_created(
            namespace_id=uuid4(),
            entity_id=uuid4(),
            data={"name": "Beta", "entity_type": "ORGANIZATION", "confidence": 0.5},
        )

        assert await d.dispatch(passing) == 1
        assert await d.dispatch(failing) == 0
        cb.assert_awaited_once()
