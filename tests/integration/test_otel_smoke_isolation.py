"""Regression: the OTLP smoke test must not leak a real TracerProvider.

``test_otel_smoke.test_spans_reach_otlp_endpoint`` installs a REAL
``TracerProvider`` (process-wide singleton) to prove spans ship over the
wire. Before the teardown fix, it never reset that global back to the
proxy, so the real provider survived into later tests in the same
worker. That is more than an aesthetic leak: once a real provider is in
place, OTel's span ``__exit__`` runs ``record_exception()`` /
``str(exc)`` on any propagating exception. On neo4j 6.x, errors built by
``Neo4jError._basic_hydrate(...)`` (as the dual_nodes unit tests do) have
no ``_gql_status`` attribute, so ``str(exc)`` raises ``AttributeError`` —
masking the real ``ClientError`` the dual_nodes test is asserting on and
turning an unrelated test into a flaky failure under ``pytest -n auto``.

Why production code (``dual_nodes.py``) needs NO change: in production,
neo4j errors are hydrated by the driver from real server responses that
DO carry the GQL status fields in 6.x, so ``str(exc)`` works. Only the
test's ``_basic_hydrate`` mock omits them. The bug is purely a test
isolation defect; the fix is the smoke test's teardown.

These tests are deterministic and order-independent: each one installs
its own real provider, exercises the path, and resets — it does not rely
on a particular test running before it.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from neo4j.exceptions import ClientError, Neo4jError

from khora.telemetry import bootstrap
from tests.test_helpers.otel import reset_khora_telemetry

pytestmark = pytest.mark.integration


def _install_real_tracer_provider() -> None:
    """Install a real SDK ``TracerProvider`` and rebind khora's cached tracer.

    Mirrors what ``configure_telemetry(backend="otel")`` does to the
    globals — the state the smoke test leaves behind without a teardown.
    """
    from opentelemetry import trace as _t
    from opentelemetry.sdk.trace import TracerProvider

    from khora.telemetry import _otel as _otel_module

    _t.set_tracer_provider(TracerProvider())
    _otel_module._TRACER = _t.get_tracer("khora", _otel_module._KHORA_VERSION)


def _make_neo4j_session_raising(exc: BaseException) -> tuple[MagicMock, AsyncMock]:
    """Build a mock driver whose session.execute_read raises *exc*."""
    driver = MagicMock()
    session = AsyncMock()
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    driver.session.return_value = ctx
    session.execute_read = AsyncMock(side_effect=exc)
    return driver, session


async def test_leaked_real_provider_masks_client_error() -> None:
    """A leaked real provider turns a re-raised ClientError into AttributeError.

    This documents the masking mechanism that made test_dual_nodes flaky:
    with a real TracerProvider installed, the @trace span __exit__ runs
    record_exception()/str(exc) on the re-raised neo4j error, and the
    hydrated 6.x error (no _gql_status) makes str(exc) raise
    AttributeError — swallowing the real ClientError.
    """
    from khora.engines.vectorcypher.dual_nodes import DualNodeManager

    syntax_exc = Neo4jError._basic_hydrate(
        neo4j_code="Neo.ClientError.Statement.SyntaxError",
        message="boom",
    )
    driver, _session = _make_neo4j_session_raising(syntax_exc)
    manager = DualNodeManager(driver, query_timeout=1.0)

    try:
        _install_real_tracer_provider()
        # The leaked real provider records the exception on span exit;
        # str(exc) on the hydrated error raises AttributeError, masking
        # the ClientError. This is the symptom the leak produces.
        with pytest.raises(AttributeError, match="_gql_status"):
            await manager.get_entity_neighborhoods([uuid4()], uuid4(), depth=1)
    finally:
        reset_khora_telemetry()


async def test_proxy_provider_propagates_client_error() -> None:
    """With the proxy provider restored, the real ClientError propagates.

    This is the post-teardown state the fix guarantees. The proxy
    provider yields a NonRecordingSpan, so record_exception() is a no-op
    and the genuine ClientError reaches the caller untouched — which is
    exactly what test_dual_nodes.test_reraises_non_timeout_client_error
    asserts.
    """
    from khora.engines.vectorcypher.dual_nodes import DualNodeManager

    # Guarantee we start from the proxy provider regardless of any leak
    # left by an earlier test (deterministic / order-independent).
    reset_khora_telemetry()
    assert bootstrap._tracer_provider_is_proxy(), "expected the proxy TracerProvider after reset"

    syntax_exc = Neo4jError._basic_hydrate(
        neo4j_code="Neo.ClientError.Statement.SyntaxError",
        message="boom",
    )
    driver, _session = _make_neo4j_session_raising(syntax_exc)
    manager = DualNodeManager(driver, query_timeout=1.0)

    with pytest.raises(ClientError) as excinfo:
        await manager.get_entity_neighborhoods([uuid4()], uuid4(), depth=1)
    assert excinfo.value.code == "Neo.ClientError.Statement.SyntaxError"


def test_smoke_test_resets_global_provider_in_teardown(monkeypatch: pytest.MonkeyPatch) -> None:
    """Running the smoke test must leave the proxy provider in place.

    Directly drives the smoke test body, then asserts the global
    TracerProvider is back to the proxy. This fails if the smoke test's
    teardown (the reset in its ``finally`` block) is removed or stops
    resetting the global — the exact regression that pollutes later
    tests in the same worker.
    """
    from tests.integration import test_otel_smoke

    # Start from a known-clean state so the assertion reflects the smoke
    # test's own teardown, not residue from an earlier test.
    reset_khora_telemetry()

    test_otel_smoke.test_spans_reach_otlp_endpoint(monkeypatch)

    assert bootstrap._tracer_provider_is_proxy(), (
        "smoke test leaked a real TracerProvider — its teardown did not reset the global"
    )
    assert bootstrap._meter_provider_is_proxy(), (
        "smoke test leaked a real MeterProvider — its teardown did not reset the global"
    )
    assert bootstrap._handle is None, "smoke test left a stale bootstrap._handle behind"
