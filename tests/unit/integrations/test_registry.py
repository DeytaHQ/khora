"""Tests for the adapter registry (entry points + explicit register)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from khora.integrations import discover, register, registry


def _fake_entry_point(name: str, value: str) -> MagicMock:
    """Build a MagicMock that quacks like an importlib.metadata.EntryPoint."""
    ep = MagicMock()
    ep.name = name
    ep.value = value
    return ep


def test_discover_empty_when_no_entry_points():
    with patch("khora.integrations.registry.entry_points", return_value=[]):
        result = discover()
    assert result == {}


def test_discover_returns_entry_points_keyed_by_name():
    eps = [_fake_entry_point("crewai", "crewai_pkg:Mem"), _fake_entry_point("lg", "lg_pkg:Store")]
    with patch("khora.integrations.registry.entry_points", return_value=eps):
        result = discover()
    assert set(result.keys()) == {"crewai", "lg"}
    assert result["crewai"] is eps[0]


def test_discover_is_cached_after_first_call():
    eps = [_fake_entry_point("crewai", "crewai_pkg:Mem")]
    mock = MagicMock(return_value=eps)
    with patch("khora.integrations.registry.entry_points", mock):
        discover()
        discover()
        discover()
    assert mock.call_count == 1


def test_register_adds_a_factory():
    def factory():
        return "x"

    register("custom", factory)
    with patch("khora.integrations.registry.entry_points", return_value=[]):
        result = discover()
    assert result["custom"] is factory


def test_register_overrides_entry_point_of_same_name():
    ep = _fake_entry_point("crewai", "real_pkg:RealMem")

    def fake_factory():
        return "fake"

    with patch("khora.integrations.registry.entry_points", return_value=[ep]):
        # Discover first so the entry-point cache is populated.
        first = discover()
        assert first["crewai"] is ep
        # Now explicit registration overrides.
        register("crewai", fake_factory)
        second = discover()
    assert second["crewai"] is fake_factory


def test_register_rejects_empty_name():
    with pytest.raises(ValueError):
        register("", lambda: None)


def test_clear_wipes_explicit_and_resets_cache():
    eps_a = [_fake_entry_point("a", "pkg:A")]
    eps_b = [_fake_entry_point("b", "pkg:B")]

    with patch("khora.integrations.registry.entry_points", return_value=eps_a):
        discover()  # cache "a"
        register("manual", lambda: None)
    registry.clear()
    # After clear, the cache should re-walk; this time we serve a different
    # entry-points list to prove the cache really reset.
    with patch("khora.integrations.registry.entry_points", return_value=eps_b):
        result = discover()
    assert set(result.keys()) == {"b"}


def test_discover_returns_a_copy_so_callers_cant_mutate_registry():
    register("x", lambda: 1)
    with patch("khora.integrations.registry.entry_points", return_value=[]):
        result = discover()
        result.clear()
        # Second call should still return the registration.
        result2 = discover()
    assert "x" in result2
