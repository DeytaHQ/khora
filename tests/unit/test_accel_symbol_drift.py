"""CI guard + resilience tests for khora._accel's Rust-symbol import layer.

Covers issue #1465: one drifted symbol in the khora-accel wheel must NOT
silently disable all Rust acceleration. Instead:

  * A stale/partial wheel (module present, one symbol missing) degrades ONLY
    the kernels that need the missing symbol and logs a WARNING naming it.
  * A wholly-absent wheel keeps the normal INFO "using fallback" behaviour.
  * A CI guard asserts a built wheel exports every symbol _accel.py expects,
    driven off the single-source-of-truth table so it can't drift.
"""

from __future__ import annotations

import importlib
import re
import sys
import types

import pytest
from loguru import logger as loguru_logger

import khora._accel as accel

# ---------------------------------------------------------------------------
# The expected-symbol list lives in exactly one place: accel._RUST_SYMBOLS.
# Every test below derives from it so the list can't drift from the imports.
# ---------------------------------------------------------------------------


def test_symbol_table_matches_kernel_usage():
    """Every ``_rust_*`` symbol the kernels call must appear in _RUST_SYMBOLS.

    This enforces the single-source-of-truth claim WITHOUT needing a wheel:
    if someone adds a new Rust-accelerated kernel that references a fresh
    ``_rust_foo`` binding but forgets to register it in the table, this test
    fails. Symmetrically, a table entry that no kernel uses is flagged too.
    """
    source = importlib.util.find_spec("khora._accel").origin
    assert source is not None
    text = open(source).read()

    # Strip the _RUST_SYMBOLS table definition itself so its keys don't count
    # as "usage". The table is the block from "_RUST_SYMBOLS: dict" to its "}".
    table_match = re.search(r"_RUST_SYMBOLS: dict.*?\n\}\n", text, re.DOTALL)
    assert table_match is not None, "could not locate _RUST_SYMBOLS table"
    body = text[table_match.end() :]

    used = set(re.findall(r"\b_rust_[a-z0-9_]+\b", body))

    registered = set(accel._RUST_SYMBOLS)

    # Forward direction is the drift that matters: every ``_rust_*`` a kernel
    # calls must be registered, else that kernel would hit a NameError.
    missing_from_table = used - registered
    assert not missing_from_table, (
        f"kernels reference Rust symbols not registered in _RUST_SYMBOLS: {sorted(missing_from_table)}"
    )

    # RustBM25Index is a re-exported class (consumed by downstream imports, not
    # via a ``_rust_*`` kernel gate) so it won't show up in ``used`` - assert it
    # is registered explicitly.
    assert "RustBM25Index" in registered

    # Reverse direction, minus the re-exported class: no dead ``_rust_*`` entries.
    dead = {name for name in registered - used if name != "RustBM25Index"}
    assert not dead, f"_RUST_SYMBOLS registers ``_rust_*`` symbols no kernel uses (dead entries): {sorted(dead)}"


def test_all_expected_symbols_bound_as_module_attrs():
    """Every registered local name must be bound on the module (to a value or None)."""
    for name in accel._RUST_SYMBOLS:
        assert hasattr(accel, name), f"{name} not bound on khora._accel"


# ---------------------------------------------------------------------------
# CI guard: when a real wheel is importable, it must export every symbol.
# Skips cleanly when the wheel is absent (normal no-accel install / CI leg).
# ---------------------------------------------------------------------------


def test_built_wheel_exports_every_expected_symbol():
    """If khora_accel is importable, assert it exports every expected symbol.

    This is the drift guard: in the wheel-build CI leg where khora_accel IS
    installed, a stale/partial wheel missing a symbol _accel.py expects fails
    here instead of silently falling back in production.
    """
    khora_accel = pytest.importorskip(
        "khora_accel",
        reason="khora-accel wheel not installed; drift guard runs in the wheel CI leg",
    )
    missing = [attr for attr in accel._RUST_SYMBOLS.values() if not hasattr(khora_accel, attr)]
    assert not missing, (
        f"installed khora_accel wheel is missing symbols _accel.py expects: "
        f"{sorted(missing)} — the wheel is stale relative to the Python layer"
    )


# ---------------------------------------------------------------------------
# Resilience: simulate a partial wheel by reloading _accel against a fake
# khora_accel module that lacks one symbol.
# ---------------------------------------------------------------------------


def _make_fake_khora_accel(missing: set[str]) -> types.ModuleType:
    """Build a stand-in khora_accel exporting every symbol except ``missing``.

    Function symbols become identity-ish stubs (they only need to be present
    and callable); the missing ones are simply not set on the module.
    """
    mod = types.ModuleType("khora_accel")
    for attr in accel._RUST_SYMBOLS.values():
        if attr in missing:
            continue
        if attr == "RustBM25Index":
            setattr(mod, attr, type("RustBM25Index", (), {}))
        else:
            # Sentinel callable so kernels can be observed calling into "Rust".
            setattr(mod, attr, lambda *a, _attr=attr, **k: ("rust", _attr))
    return mod


@pytest.fixture()
def reload_accel_with(monkeypatch):
    """Reload khora._accel against a supplied fake khora_accel module.

    Yields a callable ``load(fake_or_None)`` that reloads the module and
    returns ``(module, warning_messages)`` where ``warning_messages`` are the
    loguru WARNING+ lines emitted during import. A loguru sink is attached
    around the reload because the stale-wheel warning fires at import time (so
    pytest's stdlib ``caplog`` never sees it). The real module is restored on
    teardown so later tests see the genuine import state.
    """

    def load(fake: types.ModuleType | None):
        if fake is None:
            # Guarantee the import fails even if a real wheel is installed:
            # a ``None`` entry in sys.modules makes ``import`` raise ImportError.
            monkeypatch.setitem(sys.modules, "khora_accel", None)
        else:
            monkeypatch.setitem(sys.modules, "khora_accel", fake)

        messages: list[str] = []
        sink_id = loguru_logger.add(lambda msg: messages.append(str(msg)), level="WARNING")
        try:
            mod = importlib.reload(accel)
        finally:
            loguru_logger.remove(sink_id)
        return mod, messages

    yield load

    # Restore the genuine module state for the rest of the suite.
    monkeypatch.undo()
    importlib.reload(accel)


def test_partial_wheel_degrades_only_missing_symbol(reload_accel_with):
    """A wheel missing one symbol keeps every OTHER symbol on the Rust path."""
    fake = _make_fake_khora_accel(missing={"block_and_score_pairs"})
    mod, _ = reload_accel_with(fake)

    # Module counts as "present"...
    assert mod._HAS_RUST is True
    # ...but the missing symbol degrades to None (its kernel falls back),
    assert mod._rust_block_and_score_pairs is None
    # while every other symbol stays bound to the (fake) Rust impl.
    assert mod._rust_cosine is not None
    assert mod._rust_pagerank is not None
    assert mod.RustBM25Index is not None


def test_partial_wheel_logs_warning_naming_missing_symbol(reload_accel_with):
    """A partial wheel logs a WARNING naming the missing symbol(s)."""
    fake = _make_fake_khora_accel(missing={"block_and_score_pairs"})
    _, messages = reload_accel_with(fake)

    assert messages, "partial wheel must emit a WARNING"
    joined = " ".join(messages)
    assert "block_and_score_pairs" in joined
    assert "stale" in joined.lower()


def test_full_wheel_emits_no_stale_warning(reload_accel_with):
    """A complete (fake) wheel emits no stale-symbol WARNING."""
    fake = _make_fake_khora_accel(missing=set())
    mod, messages = reload_accel_with(fake)

    assert mod._HAS_RUST is True
    stale = [m for m in messages if "stale" in m.lower()]
    assert not stale, f"unexpected stale-wheel warning: {stale}"


def test_absent_wheel_binds_all_to_none_without_warning(reload_accel_with):
    """No wheel at all -> all symbols None, _HAS_RUST False, no WARNING."""
    mod, messages = reload_accel_with(None)

    assert mod._HAS_RUST is False
    for name in mod._RUST_SYMBOLS:
        assert getattr(mod, name) is None, f"{name} should be None with no wheel"
    stale = [m for m in messages if "stale" in m.lower()]
    assert not stale, "a wholly-absent wheel is not a stale-wheel error"
