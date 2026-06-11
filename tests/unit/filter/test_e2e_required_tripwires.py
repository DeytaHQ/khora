"""Negative-verification: the e2e ``*_REQUIRED`` tripwires have teeth.

``tests/e2e/conftest.py::pytest_configure`` converts a silent green skip into a
hard red exit on the container-free legs: when ``KHORA_E2E_EMBEDDED_REQUIRED=1``
(the embedded ``sqlite_lance`` stack) or ``KHORA_E2E_SURREAL_REQUIRED=1`` (the
embedded SurrealDB SDK) is set but the matching ``_harness`` probe reports the
backend unavailable, it calls ``pytest.exit(..., returncode=1)`` — which raises
``_pytest.outcomes.Exit``. A tripwire that never fires is worse than none, so
this hermetic test (no DB, no network) proves each one actually aborts when its
probe says the backend is missing, and stays quiet when the probe says it is
present.

Each test monkeypatches the relevant probe on the shared ``_harness`` module
object that ``conftest`` references (``_harness._embedded_available()`` /
``_harness._surreal_embedded_available()``) and deletes the other four
``KHORA_E2E_*_REQUIRED`` flags so a real local Postgres/Neo4j/Weaviate — or a
stray sibling flag — cannot perturb the assertion. ``monkeypatch`` auto-restores
both the env vars and the patched attributes after each test.

No DB, no infra — runs in the fast unit suite.
"""

from __future__ import annotations

import pytest
from _pytest.outcomes import Exit

import tests.e2e._harness as _harness
from tests.e2e import conftest

pytestmark = [pytest.mark.unit]

# Every ``KHORA_E2E_*_REQUIRED`` flag ``pytest_configure`` consults. Each test
# sets exactly one and clears the rest so an ambient env value cannot change the
# branch that fires.
_ALL_REQUIRED_FLAGS = (
    "KHORA_E2E_PG_REQUIRED",
    "KHORA_E2E_NEO4J_REQUIRED",
    "KHORA_E2E_WEAVIATE_REQUIRED",
    "KHORA_E2E_EMBEDDED_REQUIRED",
    "KHORA_E2E_SURREAL_REQUIRED",
)


def _isolate_required_flags(monkeypatch: pytest.MonkeyPatch, keep: str) -> None:
    """Clear every ``*_REQUIRED`` flag except ``keep``, which is set to ``"1"``.

    Deleting the others (``raising=False`` — they are usually unset) stops a real
    local Postgres/etc. or a sibling flag left in the environment from tripping a
    different branch of ``pytest_configure`` and masking the assertion.
    """
    for flag in _ALL_REQUIRED_FLAGS:
        monkeypatch.delenv(flag, raising=False)
    monkeypatch.setenv(keep, "1")


def test_embedded_required_aborts_when_probe_false(
    monkeypatch: pytest.MonkeyPatch, pytestconfig: pytest.Config
) -> None:
    """``KHORA_E2E_EMBEDDED_REQUIRED=1`` + embedded probe False → ``pytest.exit``.

    The container-free embedded ``sqlite_lance`` lane must fail RED, not skip
    green, when the embedded stack is unavailable in a leg that requires it.
    """
    _isolate_required_flags(monkeypatch, keep="KHORA_E2E_EMBEDDED_REQUIRED")
    monkeypatch.setattr(_harness, "_embedded_available", lambda: False)

    with pytest.raises(Exit):
        conftest.pytest_configure(pytestconfig)


def test_surreal_required_aborts_when_probe_false(monkeypatch: pytest.MonkeyPatch, pytestconfig: pytest.Config) -> None:
    """``KHORA_E2E_SURREAL_REQUIRED=1`` + surreal probe False → ``pytest.exit``.

    Symmetric to the embedded case: the in-process SurrealDB lane must abort the
    session when the optional ``surrealdb`` SDK is missing in a leg that requires it.
    """
    _isolate_required_flags(monkeypatch, keep="KHORA_E2E_SURREAL_REQUIRED")
    monkeypatch.setattr(_harness, "_surreal_embedded_available", lambda: False)

    with pytest.raises(Exit):
        conftest.pytest_configure(pytestconfig)


def test_embedded_required_passes_when_probe_true(monkeypatch: pytest.MonkeyPatch, pytestconfig: pytest.Config) -> None:
    """``KHORA_E2E_EMBEDDED_REQUIRED=1`` + embedded probe True → no abort.

    The tripwire is conditional, not unconditional: when the required backend IS
    available, ``pytest_configure`` must return cleanly so the leg runs.
    """
    _isolate_required_flags(monkeypatch, keep="KHORA_E2E_EMBEDDED_REQUIRED")
    monkeypatch.setattr(_harness, "_embedded_available", lambda: True)

    conftest.pytest_configure(pytestconfig)


def test_surreal_required_passes_when_probe_true(monkeypatch: pytest.MonkeyPatch, pytestconfig: pytest.Config) -> None:
    """``KHORA_E2E_SURREAL_REQUIRED=1`` + surreal probe True → no abort."""
    _isolate_required_flags(monkeypatch, keep="KHORA_E2E_SURREAL_REQUIRED")
    monkeypatch.setattr(_harness, "_surreal_embedded_available", lambda: True)

    conftest.pytest_configure(pytestconfig)
