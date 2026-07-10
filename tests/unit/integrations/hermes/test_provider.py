"""Unit tests for :func:`khora.integrations.hermes.KhoraMemoryProvider`.

These tests cover the public factory and the dynamically-built
``MemoryProvider`` subclass it returns. The runtime is always injected
(no daemon thread / executor spun up); ``kb`` is an ``AsyncMock`` shaped
like ``khora.Khora`` with the two surfaces the provider touches at
namespace bootstrap: ``_resolve_namespace`` (async) and
``storage.create_namespace`` (async).

The fake ``hermes_agent`` ABC is injected by the autouse fixture in
``conftest.py`` so the factory's lazy ``import hermes_agent`` succeeds
without the optional distribution. The ``[hermes]`` extra is intentionally
NOT declared in pyproject (upstream ``requests==2.33.0`` pin collides
with khora's CVE-2026-25645 constraint), so these tests must run on a
plain ``make install`` venv — no extras required.
"""

from __future__ import annotations

import logging
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from khora.integrations.hermes._mapping import (
    KEY_SESSION_ID,
    KEY_TURN_SEQ,
    derive_namespace_uuid,
    format_memory_context,
)
from khora.integrations.hermes.provider import KhoraMemoryProvider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_kb() -> MagicMock:
    """Build a Khora-shaped mock with the surfaces the provider touches."""
    kb = MagicMock(name="Khora")
    # ``_build_namespace`` calls ``run_sync(_ensure())`` which awaits
    # ``kb._resolve_namespace`` first; default to a successful resolve so
    # namespace creation short-circuits without touching ``create_namespace``.
    kb._resolve_namespace = AsyncMock(side_effect=lambda ns: ns)
    kb.storage = MagicMock()
    kb.storage.create_namespace = AsyncMock(return_value=None)
    kb._connected = True
    return kb


def _make_runtime() -> MagicMock:
    """Build a ``_KhoraRuntime``-shaped mock.

    The provider exercises ``enqueue_remember`` / ``enqueue_remember_batch``
    / ``enqueue_recall`` / ``recall_sync`` / ``dispatch_sync`` / ``drain``
    / ``failure_rate_pct`` / ``last_errors`` / ``shutdown``. Everything is
    a plain ``MagicMock`` — no async machinery, so the test body stays
    deterministic.
    """
    rt = MagicMock(name="_KhoraRuntime")
    rt.enqueue_remember = MagicMock()
    rt.enqueue_remember_batch = MagicMock()
    rt.enqueue_recall = MagicMock()
    rt.recall_sync = MagicMock(return_value=None)
    rt.dispatch_sync = MagicMock(return_value="")
    rt.drain = MagicMock(return_value=0)
    rt.failure_rate_pct = MagicMock(return_value=0.0)
    rt.last_errors = MagicMock(return_value=[])
    rt.shutdown = MagicMock()
    return rt


def _make_provider(
    *,
    kb: MagicMock | None = None,
    runtime: MagicMock | None = None,
    **kwargs: Any,
) -> Any:
    """Build a provider with sensible defaults, then initialize it once.

    Tests that need to exercise pre-initialize behaviour build the
    provider directly via :func:`KhoraMemoryProvider` instead.
    """
    kb = kb or _make_kb()
    runtime = runtime or _make_runtime()
    provider = KhoraMemoryProvider(kb=kb, runtime=runtime, **kwargs)
    provider.initialize("session-1", agent_identity="agent-A")
    return provider


# ---------------------------------------------------------------------------
# Factory: import / construction surface
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_factory_raises_clear_error_without_hermes_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without ``hermes_agent`` in ``sys.modules``, the factory raises ``ImportError``.

    The error message must mention the optional ``hermes`` extra so
    operators know what to install (even though the extra is currently
    not declared in pyproject — see CHANGELOG #628; once upstream
    requests-pin is relaxed the extra will return).
    """
    # Tear down the autouse fake so the import path fails cleanly.
    for mod in [
        "hermes_agent",
        "hermes_agent.agent",
        "hermes_agent.agent.memory_provider",
    ]:
        monkeypatch.delitem(sys.modules, mod, raising=False)

    from khora.integrations.hermes import provider as _provider_mod

    monkeypatch.setattr(_provider_mod, "_MemoryProviderBase", None, raising=False)

    # Also poison sys.modules so the lazy import doesn't fall back to a
    # cached real install.
    monkeypatch.setitem(sys.modules, "hermes_agent", None)

    with pytest.raises(ImportError) as exc_info:
        KhoraMemoryProvider(kb=_make_kb())

    assert "hermes" in str(exc_info.value).lower()


@pytest.mark.unit
def test_factory_succeeds_with_fake_hermes_agent(_fake_hermes_agent: type) -> None:
    """With the fake ABC in place, the factory returns an instance of it."""
    provider = KhoraMemoryProvider(kb=_make_kb(), runtime=_make_runtime())
    assert isinstance(provider, _fake_hermes_agent)


@pytest.mark.unit
def test_kb_kwarg_is_required() -> None:
    """``KhoraMemoryProvider()`` with no ``kb`` raises ``TypeError``.

    Devil's advocate: the example plugin is the only sanctioned site
    that wires ``Khora.shared()``; the adapter itself must NEVER fall
    back to it silently — lifecycle is the caller's problem.
    """
    with pytest.raises(TypeError):
        KhoraMemoryProvider()  # type: ignore[call-arg]


@pytest.mark.unit
def test_name_property_is_khora() -> None:
    """Hermes inspects ``.name`` to register the plugin — must be ``"khora"``."""
    provider = _make_provider()
    assert provider.name == "khora"


# ---------------------------------------------------------------------------
# initialize()
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_initialize_resolves_namespace() -> None:
    """``initialize`` stamps the namespace UUID derived from stable identity."""
    kb = _make_kb()
    runtime = _make_runtime()
    provider = KhoraMemoryProvider(kb=kb, runtime=runtime)

    provider.initialize("session-XYZ", agent_identity="agent-Beta", user_id="user-9")

    expected = derive_namespace_uuid("agent-Beta", "user-9")
    assert provider._namespace_id == expected
    assert provider._session_id == "session-XYZ"
    assert provider._agent_identity == "agent-Beta"
    assert provider._user_id == "user-9"
    # _build_namespace probed and found the row (mocked resolve succeeds),
    # so no create_namespace call.
    kb._resolve_namespace.assert_awaited()
    kb.storage.create_namespace.assert_not_awaited()


@pytest.mark.unit
def test_initialize_namespace_is_stable_across_sessions() -> None:
    """Regression (#1466): two sessions for one agent share ONE namespace.

    This is the whole point of the fix — folding session_id into the
    namespace gave every conversation a fresh tenancy and voided
    cross-session entity dedup + long-term recall. Re-binding to a second
    session (via initialize) MUST keep the same namespace so memory
    accumulates across conversations.
    """
    kb = _make_kb()
    runtime = _make_runtime()
    provider = KhoraMemoryProvider(kb=kb, runtime=runtime)

    provider.initialize("session-one", agent_identity="agent-A", user_id="user-1")
    ns_first = provider._namespace_id

    provider.initialize("session-two", agent_identity="agent-A", user_id="user-1")
    ns_second = provider._namespace_id

    assert ns_first == ns_second
    # The conversation scope changed even though the tenancy did not.
    assert provider._session_id == "session-two"


@pytest.mark.unit
def test_on_session_switch_keeps_namespace_and_preserves_identity() -> None:
    """A session switch keeps the identity-scoped namespace (#1466).

    ``on_session_switch`` carries agent_identity / user_id forward so a
    switch that omits them doesn't silently re-tenant the memory.
    """
    kb = _make_kb()
    runtime = _make_runtime()
    provider = KhoraMemoryProvider(kb=kb, runtime=runtime)

    provider.initialize("session-one", agent_identity="agent-A", user_id="user-1")
    ns_before = provider._namespace_id

    # Switch supplies only the new session id — identity must be preserved.
    provider.on_session_switch("session-two")

    assert provider._namespace_id == ns_before
    assert provider._agent_identity == "agent-A"
    assert provider._user_id == "user-1"
    assert provider._session_id == "session-two"


@pytest.mark.unit
def test_sync_turn_threads_session_id_to_khora_document() -> None:
    """#1466: the enqueued document carries the first-class session_id.

    The Hermes session string is derived to a stable UUID and set on both
    ``Document.session_id`` (the ``remember`` path) and top-level
    ``metadata['session_id']`` (the ``remember_batch`` coerce path), so
    session-scoped queries / ``forget_session`` work while recall still
    spans the whole namespace.
    """
    from khora.integrations.hermes._mapping import derive_session_uuid

    kb = _make_kb()
    runtime = _make_runtime()
    provider = KhoraMemoryProvider(kb=kb, runtime=runtime)
    provider.initialize("conversation-77", agent_identity="agent-A", user_id="user-1")

    provider.sync_turn("hi", "hello")

    runtime.enqueue_remember.assert_called_once()
    _kb, _ns, doc = runtime.enqueue_remember.call_args.args
    expected_session = derive_session_uuid("conversation-77")
    assert doc.session_id == expected_session
    assert doc.metadata["session_id"] == str(expected_session)


@pytest.mark.unit
def test_initialize_creates_namespace_when_resolve_misses() -> None:
    """When ``_resolve_namespace`` raises ``ValueError`` the provider creates the row.

    Probes ``_build_namespace``'s probe-then-create path without forcing
    a real DB. The race-retry branch is exercised by the next test.
    """
    kb = _make_kb()
    kb._resolve_namespace = AsyncMock(side_effect=ValueError("missing"))
    runtime = _make_runtime()
    provider = KhoraMemoryProvider(kb=kb, runtime=runtime)

    provider.initialize("session-1", agent_identity="agent-A")

    kb.storage.create_namespace.assert_awaited_once()


@pytest.mark.unit
def test_initialize_warns_and_uses_unknown_when_agent_identity_missing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Missing ``agent_identity`` warns loudly, then falls back to 'unknown'.

    Post-#1466 the namespace no longer includes session_id, so an
    identity-less bind collapses every such call into one shared
    ``hermes:unknown:{user_id}`` bucket. The provider must surface that
    misconfiguration (WARN) rather than merge conversations silently.
    """
    kb = _make_kb()
    runtime = _make_runtime()
    provider = KhoraMemoryProvider(kb=kb, runtime=runtime)

    from loguru import logger as _loguru_logger

    handler_id = _loguru_logger.add(
        lambda msg: logging.getLogger("khora.test.hermes.noident").warning(msg),
        level="WARNING",
    )
    try:
        with caplog.at_level(logging.WARNING, logger="khora.test.hermes.noident"):
            provider.initialize("session-1")  # no agent_identity kwarg
    finally:
        _loguru_logger.remove(handler_id)

    assert provider._agent_identity == "unknown"
    assert provider._namespace_id == derive_namespace_uuid("unknown", None)
    assert any("without agent_identity" in r.getMessage() for r in caplog.records), (
        f"expected a WARN about the missing agent_identity; got {[r.getMessage() for r in caplog.records]}"
    )


# ---------------------------------------------------------------------------
# prefetch / queue_prefetch  — cache coherency + abstention
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_prefetch_returns_abstention_when_runtime_returns_none() -> None:
    """Cache miss / timeout / empty namespace → the abstention block.

    Devil's-advocate flag: this is the gate that keeps the LLM from
    confabulating "I remember discussing X" when khora actually returned
    nothing. If the abstention string ever changes shape the LLM
    system-prompt block has to be updated in lockstep.
    """
    runtime = _make_runtime()
    runtime.recall_sync.return_value = None
    provider = _make_provider(runtime=runtime)

    out = provider.prefetch("what did we discuss?")

    assert out == format_memory_context([])
    assert "<memory-context>" in out
    assert "No prior memories found." in out


@pytest.mark.unit
def test_prefetch_returns_formatted_when_runtime_returns_recall_result() -> None:
    """A ``RecallResult`` with one chunk surfaces in the ``<memory-context>`` block."""

    class _Chunk:
        content = "we discussed the migration plan"
        score = 0.91
        created_at = None

    class _Result:
        chunks = [_Chunk()]
        entities: list[Any] = []

    runtime = _make_runtime()
    runtime.recall_sync.return_value = _Result()
    provider = _make_provider(runtime=runtime)

    out = provider.prefetch("migration plan")

    assert "<memory-context>" in out
    assert "</memory-context>" in out
    assert "we discussed the migration plan" in out


@pytest.mark.unit
def test_queue_prefetch_calls_runtime_enqueue_recall() -> None:
    """``queue_prefetch`` is fire-and-forget — must hand off to runtime."""
    runtime = _make_runtime()
    provider = _make_provider(runtime=runtime)

    provider.queue_prefetch("what did we say?", session_id="session-explicit")

    runtime.enqueue_recall.assert_called_once()
    (kb_arg, ns_arg, sess_arg, query_arg), _ = runtime.enqueue_recall.call_args
    assert kb_arg is provider.kb
    assert ns_arg == provider._namespace_id
    assert sess_arg == "session-explicit"
    assert query_arg == "what did we say?"


@pytest.mark.unit
def test_prefetch_short_circuits_before_initialize() -> None:
    """Pre-init prefetch returns the abstention payload, never touches runtime."""
    runtime = _make_runtime()
    provider = KhoraMemoryProvider(kb=_make_kb(), runtime=runtime)
    # No initialize() call.
    out = provider.prefetch("anything")
    assert out == format_memory_context([])
    runtime.recall_sync.assert_not_called()


# ---------------------------------------------------------------------------
# sync_turn — non-blocking write + monotonic turn_seq
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_sync_turn_increments_turn_seq_and_enqueues_remember() -> None:
    """Two consecutive ``sync_turn`` calls produce turn_seq=1 then 2.

    Devil's-advocate flag: the turn counter is unlocked because Hermes
    contracts one ``sync_turn`` per thread per session. If that ever
    weakens we'd see duplicate seqs; this test pins the current ordering
    invariant so a regression is loud.
    """
    runtime = _make_runtime()
    provider = _make_provider(runtime=runtime)

    provider.sync_turn("user-1", "assistant-1")
    provider.sync_turn("user-2", "assistant-2")

    assert runtime.enqueue_remember.call_count == 2
    seen_turn_seqs: list[int] = []
    for call in runtime.enqueue_remember.call_args_list:
        # signature: (kb, namespace_id, document)
        _kb, _ns, doc = call.args
        custom = doc.metadata["custom"]
        assert custom[KEY_SESSION_ID] == "session-1"
        seen_turn_seqs.append(custom[KEY_TURN_SEQ])
    assert seen_turn_seqs == [1, 2]


@pytest.mark.unit
def test_sync_turn_is_non_blocking() -> None:
    """``sync_turn`` returns immediately; the runtime owns the await.

    The mock runtime's ``enqueue_remember`` returns ``None``; the
    provider must NOT call ``run_sync`` or await anything in this path
    (that would defeat the whole point of the background executor).
    """
    runtime = _make_runtime()
    provider = _make_provider(runtime=runtime)

    result = provider.sync_turn("u", "a")

    assert result is None
    runtime.enqueue_remember.assert_called_once()
    # The runtime mock's ``dispatch_sync`` and ``recall_sync`` must NOT
    # have been touched — sync_turn is fire-and-forget only.
    runtime.dispatch_sync.assert_not_called()
    runtime.recall_sync.assert_not_called()


# ---------------------------------------------------------------------------
# handle_tool_call dispatch
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_handle_tool_call_dispatches_memory_search_to_runtime() -> None:
    """``memory_search`` routes through ``runtime.dispatch_sync(dispatch_memory_search, ...)``."""
    from khora.integrations.hermes._tools import dispatch_memory_search

    runtime = _make_runtime()
    runtime.dispatch_sync.return_value = "<memory-context>\nrecalled.\n</memory-context>"
    provider = _make_provider(runtime=runtime)

    out = provider.handle_tool_call("memory_search", {"query": "x"})

    assert "<memory-context>" in out
    runtime.dispatch_sync.assert_called_once()
    fn, kb_arg, ns_arg, args_arg = runtime.dispatch_sync.call_args.args
    assert fn is dispatch_memory_search
    assert kb_arg is provider.kb
    assert ns_arg == provider._namespace_id
    assert args_arg == {"query": "x"}


@pytest.mark.unit
def test_handle_tool_call_dispatches_memory_recall_to_runtime() -> None:
    """``memory_recall`` routes to the temporal-bounded dispatch fn."""
    from khora.integrations.hermes._tools import dispatch_memory_recall

    runtime = _make_runtime()
    runtime.dispatch_sync.return_value = "ok"
    provider = _make_provider(runtime=runtime)

    provider.handle_tool_call("memory_recall", {"query": "y", "after": "2026-01-01"})

    fn, _kb, _ns, args_arg = runtime.dispatch_sync.call_args.args
    assert fn is dispatch_memory_recall
    assert args_arg["after"] == "2026-01-01"


@pytest.mark.unit
def test_handle_tool_call_unknown_raises_value_error() -> None:
    """Unknown tool names hit the ``unknown`` counter branch and raise."""
    runtime = _make_runtime()
    provider = _make_provider(runtime=runtime)

    with pytest.raises(ValueError, match="unknown tool"):
        provider.handle_tool_call("not_a_real_tool", {})


@pytest.mark.unit
def test_handle_tool_call_short_circuits_pre_initialize() -> None:
    """Pre-init tool calls return the abstention payload, never touch runtime."""
    runtime = _make_runtime()
    provider = KhoraMemoryProvider(kb=_make_kb(), runtime=runtime)
    # No initialize() call — _namespace_id is None.
    out = provider.handle_tool_call("memory_search", {"query": "x"})
    assert out == format_memory_context([])
    runtime.dispatch_sync.assert_not_called()


# ---------------------------------------------------------------------------
# on_pre_compress — batched message-pair flush
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_on_pre_compress_batches_message_pairs() -> None:
    """4 messages (2 user/assistant pairs) → 1 batch with 2 documents."""
    runtime = _make_runtime()
    provider = _make_provider(runtime=runtime)

    messages = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    provider.on_pre_compress(messages)

    runtime.enqueue_remember_batch.assert_called_once()
    _kb, _ns, documents = runtime.enqueue_remember_batch.call_args.args
    assert len(documents) == 2
    # Turn seqs are contiguous and start above the current cursor (0 → 1, 2).
    seqs = [doc.metadata["custom"][KEY_TURN_SEQ] for doc in documents]
    assert seqs == [1, 2]


@pytest.mark.unit
def test_on_pre_compress_no_op_on_empty_messages() -> None:
    """No pairs → no batch call. Avoids a zero-document round trip."""
    runtime = _make_runtime()
    provider = _make_provider(runtime=runtime)
    provider.on_pre_compress([])
    runtime.enqueue_remember_batch.assert_not_called()


# ---------------------------------------------------------------------------
# on_session_end — drain + failure-rate WARN
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_on_session_end_drains_and_warns_above_threshold(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A failure rate of 5% > default 1% threshold → WARN log line.

    Devil's-advocate flag: this is the operator's only off-host signal
    that background ingest is degraded. If the threshold check ever
    silently fails, telemetry would still tick but the operator would
    miss the inline warning. Pinning the WARN here is the gate.
    """
    runtime = _make_runtime()
    runtime.failure_rate_pct.return_value = 5.0
    runtime.last_errors.return_value = ["[remember] simulated-failure"]
    provider = _make_provider(runtime=runtime, failure_threshold_pct=1.0)

    # loguru → caplog bridge: provider logs via loguru's ``logger``, but
    # ``setup_logging`` (if it ran) routes loguru records through stdlib
    # logging. Add a propagating handler so caplog can observe them.
    from loguru import logger as _loguru_logger

    handler_id = _loguru_logger.add(
        lambda msg: logging.getLogger("khora.test.hermes").warning(msg),
        level="WARNING",
    )
    try:
        with caplog.at_level(logging.WARNING, logger="khora.test.hermes"):
            provider.on_session_end()
    finally:
        _loguru_logger.remove(handler_id)

    runtime.drain.assert_called_once()
    # The WARN message must include the 5% rate so operators can see it
    # in the log line itself, not only via the counter.
    assert any("5.00" in record.getMessage() or "5.0" in record.getMessage() for record in caplog.records), (
        f"expected a WARN containing the rate; got {[r.getMessage() for r in caplog.records]}"
    )


@pytest.mark.unit
def test_on_session_end_no_warn_below_threshold(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """0% failure rate must NOT emit the threshold WARN."""
    runtime = _make_runtime()
    runtime.failure_rate_pct.return_value = 0.0
    runtime.last_errors.return_value = []
    provider = _make_provider(runtime=runtime, failure_threshold_pct=1.0)

    from loguru import logger as _loguru_logger

    handler_id = _loguru_logger.add(
        lambda msg: logging.getLogger("khora.test.hermes.below").warning(msg),
        level="WARNING",
    )
    try:
        with caplog.at_level(logging.WARNING, logger="khora.test.hermes.below"):
            provider.on_session_end()
    finally:
        _loguru_logger.remove(handler_id)

    runtime.drain.assert_called_once()
    # No "failure rate" WARN line.
    threshold_warns = [r for r in caplog.records if "failure rate" in r.getMessage()]
    assert threshold_warns == []


@pytest.mark.unit
def test_shutdown_is_idempotent_via_runtime() -> None:
    """``provider.shutdown()`` delegates to the runtime's idempotent shutdown."""
    runtime = _make_runtime()
    provider = _make_provider(runtime=runtime)

    provider.shutdown()
    provider.shutdown()

    assert runtime.shutdown.call_count == 2  # the runtime is the one guarding idempotency
