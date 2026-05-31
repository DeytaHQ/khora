"""Issue #892: reconcile_fact must not fail open to ADD on LLM error.

Before this fix, ``MemoryCompressor.reconcile_fact`` caught ``Exception``
broadly and returned ``ReconcileAction(op=FactOperation.ADD)`` on every
LLM failure. That conflated "model said no contradiction" with "model
never answered", silently accumulating contradictory facts and breaking
Chronicle's fact-supersession premise.

After the fix:

* The except clause is narrowed to transient LLM failures (rate limit,
  timeout, connection error, JSON parse error, asyncio timeout).
* On a transient failure the fact is SKIPped (NOT added).
* The ``khora.chronicle.reconcile.failures_total`` counter increments.
* A WARNING is logged with subject / predicate / namespace_id context.
* Non-LLM exceptions (real bugs) propagate so they get noticed.
* Happy-path ADD / UPDATE / DELETE / NOOP semantics are unchanged.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import litellm
import pytest
from loguru import logger as _loguru_logger

from khora.engines.chronicle.compression import (
    FactOperation,
    MemoryCompressor,
    MemoryFact,
    ReconcileAction,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_litellm_response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.usage = MagicMock()
    response.usage.prompt_tokens = 1
    response.usage.completion_tokens = 1
    response.usage.total_tokens = 2
    return response


def _existing(subject: str = "alice", predicate: str = "works_at", obj: str = "acme") -> list[MemoryFact]:
    return [
        MemoryFact(
            id=uuid4(),
            subject=subject,
            predicate=predicate,
            object_=obj,
            fact_text=f"{subject} {predicate} {obj}",
        )
    ]


def _new(subject: str = "alice", predicate: str = "works_at", obj: str = "globex") -> MemoryFact:
    # Different object so the SVO-triple short-circuit (NOOP without LLM)
    # never fires - we need to drive the LLM path.
    ns_id = uuid4()
    return MemoryFact(
        namespace_id=ns_id,
        subject=subject,
        predicate=predicate,
        object_=obj,
        fact_text=f"{subject} {predicate} {obj}",
    )


def _build_transient_error(exc_type: type[BaseException]) -> BaseException:
    """Instantiate a litellm exception without tripping on positional args."""
    if exc_type is litellm.exceptions.RateLimitError:
        return litellm.exceptions.RateLimitError(
            message="rate limited",
            model="gpt-4o-mini",
            llm_provider="openai",
        )
    if exc_type is litellm.exceptions.Timeout:
        return litellm.exceptions.Timeout(
            message="timeout",
            model="gpt-4o-mini",
            llm_provider="openai",
        )
    if exc_type is litellm.exceptions.APIConnectionError:
        return litellm.exceptions.APIConnectionError(
            message="conn error",
            model="gpt-4o-mini",
            llm_provider="openai",
        )
    if exc_type is litellm.exceptions.ServiceUnavailableError:
        return litellm.exceptions.ServiceUnavailableError(
            message="503",
            model="gpt-4o-mini",
            llm_provider="openai",
        )
    if exc_type is litellm.exceptions.InternalServerError:
        return litellm.exceptions.InternalServerError(
            message="500",
            model="gpt-4o-mini",
            llm_provider="openai",
        )
    if exc_type is litellm.exceptions.APIError:
        return litellm.exceptions.APIError(
            status_code=500,
            message="api error",
            llm_provider="openai",
            model="gpt-4o-mini",
        )
    if exc_type is asyncio.TimeoutError:
        return TimeoutError()
    if exc_type is json.JSONDecodeError:
        return json.JSONDecodeError("bad", "{", 0)
    raise AssertionError(f"unhandled exc_type {exc_type!r}")


# ---------------------------------------------------------------------------
# Transient LLM failure -> SKIP (NOT ADD), counter increments, WARN logged
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc_type",
    [
        litellm.exceptions.RateLimitError,
        litellm.exceptions.Timeout,
        litellm.exceptions.APIConnectionError,
        litellm.exceptions.APIError,
        litellm.exceptions.ServiceUnavailableError,
        litellm.exceptions.InternalServerError,
        asyncio.TimeoutError,
        json.JSONDecodeError,
    ],
)
async def test_transient_llm_failure_returns_skip_bumps_counter_and_logs_warning(
    exc_type: type[BaseException],
) -> None:
    """All declared transient errors map to SKIP with a counter bump + WARN."""
    compressor = MemoryCompressor(model="gpt-4o-mini")
    new_fact = _new()
    existing = _existing()

    # Loguru intercept - the warning is emitted via loguru, which does NOT
    # propagate to stdlib logging by default; pytest's ``caplog`` would miss
    # it. Use a custom sink instead.
    captured: list[str] = []
    sink_id = _loguru_logger.add(lambda msg: captured.append(str(msg)), level="WARNING")

    counter_mock = MagicMock()
    try:
        with (
            patch("khora.engines.chronicle.compression.litellm") as mock_litellm,
            patch("khora.engines.chronicle.compression._RECONCILE_FAILURES", counter_mock),
        ):
            # ``mock_litellm`` patches the *module name* used inside compression.py;
            # the exception types we raise still come from the real ``litellm``
            # package imported at the top of this test, so ``except`` matches.
            mock_litellm.acompletion = AsyncMock(side_effect=_build_transient_error(exc_type))

            action = await compressor.reconcile_fact(existing, new_fact)
    finally:
        _loguru_logger.remove(sink_id)

    # 1. SKIP, NOT ADD - the whole point of the fix.
    assert action.op is FactOperation.SKIP, f"expected SKIP, got {action.op!r} for {exc_type.__name__}"
    assert action.target is None

    # 2. Counter incremented exactly once.
    counter_mock.add.assert_called_once_with(1)

    # 3. WARNING log with subject + predicate + namespace_id context.
    msg = " ".join(captured)
    assert captured, f"no WARNING logged for {exc_type.__name__}"
    assert "alice" in msg, msg
    assert "works_at" in msg, msg
    assert str(new_fact.namespace_id) in msg, msg


# ---------------------------------------------------------------------------
# Non-transient exception MUST PROPAGATE - real bugs shouldn't be swallowed
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_non_transient_exception_propagates() -> None:
    """``AttributeError`` is a real bug - the narrowed except must not catch it."""
    compressor = MemoryCompressor(model="gpt-4o-mini")
    new_fact = _new()
    existing = _existing()

    counter_mock = MagicMock()
    with (
        patch("khora.engines.chronicle.compression.litellm") as mock_litellm,
        patch("khora.engines.chronicle.compression._RECONCILE_FAILURES", counter_mock),
    ):
        mock_litellm.acompletion = AsyncMock(side_effect=AttributeError("real bug"))

        with pytest.raises(AttributeError, match="real bug"):
            await compressor.reconcile_fact(existing, new_fact)

    # Counter should NOT have moved - this wasn't a transient LLM failure.
    counter_mock.add.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_value_error_outside_json_propagates() -> None:
    """A ``ValueError`` raised by the LLM call itself (not JSON parsing) propagates.

    The narrowed except deliberately does not include ``ValueError`` - it
    catches ``json.JSONDecodeError`` (a ValueError subclass) only because the
    JSON parsing helper might raise it from an inline path. A bare
    ``ValueError`` from the LLM call itself signals a programming error in
    our request shape and should not be silently dropped.
    """
    compressor = MemoryCompressor(model="gpt-4o-mini")
    new_fact = _new()
    existing = _existing()

    with patch("khora.engines.chronicle.compression.litellm") as mock_litellm:
        mock_litellm.acompletion = AsyncMock(side_effect=ValueError("bad config"))

        with pytest.raises(ValueError, match="bad config"):
            await compressor.reconcile_fact(existing, new_fact)


# ---------------------------------------------------------------------------
# Happy-path: ADD / UPDATE / DELETE / NOOP semantics unchanged
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_happy_path_add_unchanged() -> None:
    """Empty existing list -> ADD without any LLM call."""
    compressor = MemoryCompressor(model="gpt-4o-mini")
    new_fact = _new()

    # No LLM patch needed - empty existing list short-circuits.
    action = await compressor.reconcile_fact([], new_fact)

    assert action.op is FactOperation.ADD
    assert action.target is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_happy_path_noop_on_identical_triple() -> None:
    """Identical (subject, predicate, object) triple -> NOOP without LLM."""
    compressor = MemoryCompressor(model="gpt-4o-mini")
    existing = _existing(subject="alice", predicate="works_at", obj="acme")
    new_fact = _new(subject="alice", predicate="works_at", obj="acme")

    # No LLM patch needed - SVO short-circuit fires.
    action = await compressor.reconcile_fact(existing, new_fact)

    assert action.op is FactOperation.NOOP
    assert action.target is existing[0]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_happy_path_llm_returns_add() -> None:
    compressor = MemoryCompressor(model="gpt-4o-mini")
    existing = _existing()
    new_fact = _new(obj="globex")

    with patch("khora.engines.chronicle.compression.litellm") as mock_litellm:
        mock_litellm.acompletion = AsyncMock(
            return_value=_make_litellm_response('{"operation":"add","target_id":null,"reasoning":"new"}')
        )
        action = await compressor.reconcile_fact(existing, new_fact)

    assert action.op is FactOperation.ADD
    assert action.target is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_happy_path_llm_returns_update() -> None:
    compressor = MemoryCompressor(model="gpt-4o-mini")
    existing = _existing()
    target_id = existing[0].id
    new_fact = _new(obj="globex")

    with patch("khora.engines.chronicle.compression.litellm") as mock_litellm:
        mock_litellm.acompletion = AsyncMock(
            return_value=_make_litellm_response(
                f'{{"operation":"update","target_id":"{target_id}","reasoning":"changed"}}'
            )
        )
        action = await compressor.reconcile_fact(existing, new_fact)

    assert action.op is FactOperation.UPDATE
    assert action.target is existing[0]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_happy_path_llm_returns_delete() -> None:
    compressor = MemoryCompressor(model="gpt-4o-mini")
    existing = _existing()
    target_id = existing[0].id
    new_fact = _new(obj="globex")

    with patch("khora.engines.chronicle.compression.litellm") as mock_litellm:
        mock_litellm.acompletion = AsyncMock(
            return_value=_make_litellm_response(
                f'{{"operation":"delete","target_id":"{target_id}","reasoning":"invalid"}}'
            )
        )
        action = await compressor.reconcile_fact(existing, new_fact)

    assert action.op is FactOperation.DELETE
    assert action.target is existing[0]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_llm_returning_skip_is_remapped_to_add() -> None:
    """SKIP is reserved for the transient-error path.

    If the LLM ever returns ``"skip"`` in its JSON response, we must not
    drop the fact silently - that would be a different fail-open bug.
    """
    compressor = MemoryCompressor(model="gpt-4o-mini")
    existing = _existing()
    new_fact = _new(obj="globex")

    with patch("khora.engines.chronicle.compression.litellm") as mock_litellm:
        mock_litellm.acompletion = AsyncMock(
            return_value=_make_litellm_response('{"operation":"skip","target_id":null,"reasoning":"maybe?"}')
        )
        action = await compressor.reconcile_fact(existing, new_fact)

    assert action.op is FactOperation.ADD


# ---------------------------------------------------------------------------
# Engine wiring: SKIP must not be persisted, reconcile_errors must be exposed
# in RememberResult.metadata. This guards the contract used by callers.
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_engine_reconcile_facts_returns_skip_count_and_does_not_persist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``ChronicleEngine._reconcile_facts`` must:

    * Not write a fact whose reconcile_fact returned SKIP.
    * Return the SKIP count as the second tuple element.
    """
    from khora.engines.chronicle.engine import ChronicleEngine

    ns_id = uuid4()
    new_fact = _new()
    existing = _existing()

    # Stub storage: returns the existing fact for the subject, captures writes.
    storage = MagicMock()
    storage.query_active_facts_for_subject = AsyncMock(return_value=list(existing))
    storage.write_facts = AsyncMock()
    storage.supersede_fact = AsyncMock()

    # Stub compressor: returns SKIP.
    compressor = AsyncMock()
    compressor.reconcile_fact = AsyncMock(return_value=ReconcileAction(op=FactOperation.SKIP))

    engine = ChronicleEngine.__new__(ChronicleEngine)
    monkeypatch.setattr(engine, "_get_compressor", lambda *_, **__: compressor)
    monkeypatch.setattr(engine, "_get_storage", lambda: storage)

    persisted, errors = await engine._reconcile_facts([new_fact], ns_id, expertise=None)

    assert persisted == 0
    assert errors == 1
    # SKIP must NOT be persisted - this is the data-corruption guard.
    storage.write_facts.assert_not_awaited()
