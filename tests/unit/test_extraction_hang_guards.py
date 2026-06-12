"""Regression guards for the #1113 extraction hang.

The per-call ``asyncio.wait_for`` added in #1059 bounds only ``litellm.acompletion``;
it does NOT cover the bisection ``asyncio.gather`` fan-out or the shared-semaphore
acquire, so a wedged child or a starved permit could still hang the run forever.
These tests pin the new aggregate-deadline / bounded-acquire guards.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from khora.config.llm import llm_call_timeout
from khora.extraction.extractors.base import ExtractionResult
from khora.extraction.extractors.llm import (
    LLMEntityExtractor,
    _is_retryable_extraction_exc,
)


def test_retry_predicate_never_retries_cancellation_or_timeout() -> None:
    """tenacity must NOT retry cancellation/timeout — retrying a CancelledError
    would defeat any surrounding asyncio deadline and reintroduce the hang."""
    assert _is_retryable_extraction_exc(asyncio.CancelledError()) is False
    assert _is_retryable_extraction_exc(TimeoutError()) is False  # == asyncio.TimeoutError in 3.11+
    # A generic transient error is still retryable.
    assert _is_retryable_extraction_exc(ValueError("transient")) is True


def test_batch_deadline_is_finite_and_scales_with_depth() -> None:
    ext = LLMEntityExtractor(model="test-model", timeout=60)
    d1 = ext._batch_deadline(1)
    d4 = ext._batch_deadline(4)
    d8 = ext._batch_deadline(8)
    assert 0 < d1 < d4 < d8 < float("inf")


def test_extraction_guards_finite_even_with_nonpositive_timeout() -> None:
    """``llm_call_timeout`` returns None for a non-positive timeout (a latent
    footgun in the shared helper), but the extraction guards defend locally with
    ``or DEFAULT_LLM_TIMEOUT_S`` so the aggregate ceiling is never disabled."""
    assert llm_call_timeout(0) is None  # shared-helper footgun (out of scope here)
    ext = LLMEntityExtractor(model="test-model")
    ext._timeout = 0
    assert 0 < ext._batch_deadline(4) < float("inf")  # guard stays finite anyway


@pytest.mark.asyncio
async def test_acquire_slot_times_out_when_starved(monkeypatch: pytest.MonkeyPatch) -> None:
    """A held permit must not let a sibling park forever on ``acquire()``."""
    monkeypatch.setattr("khora.config.llm._LLM_DEADLINE_GRACE_S", 0.0)
    ext = LLMEntityExtractor(model="test-model", max_concurrent=1)
    ext._timeout = 0.1  # acquire deadline ~= llm_call_timeout(0.1) = 0.1s
    await ext._semaphore.acquire()  # exhaust the only permit
    try:
        t0 = time.monotonic()
        with pytest.raises(asyncio.TimeoutError):
            async with ext._acquire_slot():
                pass
        assert time.monotonic() - t0 < 2.0  # bounded — did not park forever
    finally:
        ext._semaphore.release()


@pytest.mark.asyncio
async def test_extract_batch_aggregate_deadline_fires(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the inner multi-batch call stalls, the per-batch aggregate deadline must
    fire, mark the batch failed, and return — never hang (#1113)."""
    monkeypatch.setattr("khora.config.llm._LLM_DEADLINE_GRACE_S", 0.0)
    ext = LLMEntityExtractor(model="test-model", timeout=1)
    ext._timeout = 0.2  # small per-batch budget

    async def _stall(*_args: object, **_kwargs: object) -> list:
        await asyncio.sleep(60)
        return []

    monkeypatch.setattr(ext, "_extract_multi_batch", _stall)

    # Must exceed the tier1 regex threshold (20 chars) so they route to the LLM path.
    texts = [
        "Alice met Bob in Paris last summer to discuss the company merger deal.",
        "The Acme Corporation acquired Globex in a landmark technology transaction.",
    ]
    t0 = time.monotonic()
    # Outer wait_for is a test safety net; the guard should fire long before it.
    results = await asyncio.wait_for(ext.extract_batch(texts), timeout=10)
    elapsed = time.monotonic() - t0

    assert elapsed < 8.0, f"extract_batch did not return promptly ({elapsed:.1f}s)"
    assert len(results) == len(texts)
    assert any(r.metadata.get("error") == "extraction_timeout" for r in results)


@pytest.mark.asyncio
async def test_extract_batch_survives_uncancellable_acompletion(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real #1113 shape: the underlying litellm call IGNORES the deadline's
    cancellation (half-open socket). `_bounded_acompletion` must still return control
    to the caller within the deadline by abandoning the task, and tenacity must not
    turn the resulting cancel into a retry — so extract_batch returns, never hangs.
    """
    import litellm

    monkeypatch.setattr("khora.config.llm._LLM_DEADLINE_GRACE_S", 0.0)
    ext = LLMEntityExtractor(model="test-model", timeout=1)
    ext._timeout = 0.2  # per-call deadline ~0.2s

    async def _swallows_deadline_cancel(*_a: object, **_k: object) -> None:
        # Models an await whose cancellation teardown does not return promptly.
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            await asyncio.sleep(0.05)  # ignore the deadline's cancel, finish shortly
            return None  # caller has already abandoned us; value is irrelevant

    monkeypatch.setattr(litellm, "acompletion", _swallows_deadline_cancel)

    texts = [
        "Alice met Bob in Paris last summer to discuss the company merger deal.",
        "The Acme Corporation acquired Globex in a landmark technology transaction.",
    ]
    t0 = time.monotonic()
    results = await asyncio.wait_for(
        ext.extract_batch(texts, entity_types=["PERSON", "ORGANIZATION"]),
        timeout=10,
    )
    elapsed = time.monotonic() - t0

    assert elapsed < 8.0, f"extract_batch hung on an uncancellable call ({elapsed:.1f}s)"
    assert len(results) == len(texts)
    # Failed extraction, not a hang — every doc is marked with a TRUTHY error so
    # the fail-loud summary, circuit breaker, and fallback all see it (not the
    # silently-empty {'error': ''} that round 2 caught). And no entities leaked.
    assert all(not r.entities for r in results)
    assert all(r.metadata.get("error") for r in results), "timed-out docs must carry a truthy error"
    await asyncio.sleep(0.1)  # let the abandoned task finish before loop teardown


@pytest.mark.asyncio
async def test_timeout_during_fallback_increments_breaker_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the aggregate batch budget fires DURING the single-doc fallback, the
    circuit breaker must move by exactly ONE for the batch — not twice (the inner
    all-failed branch + the outer timeout handler). #1113."""
    monkeypatch.setattr("khora.config.llm._LLM_DEADLINE_GRACE_S", 0.0)
    ext = LLMEntityExtractor(model="test-model", timeout=1, max_retries=1)
    ext._timeout = 0.3  # budget = per_call*(batch+1)*max_retries = 0.3*3*1 = 0.9s

    async def _all_error(batch: list[str], *_a: object, **_k: object) -> list[ExtractionResult]:
        # All-failed (truthy error) → inner branch bumps the breaker + runs fallback.
        return [ExtractionResult(metadata={"error": "boom"}) for _ in batch]

    async def _slow_extract(*_a: object, **_k: object) -> ExtractionResult:
        await asyncio.sleep(60)  # stall the fallback so the aggregate budget fires
        return ExtractionResult()

    monkeypatch.setattr(ext, "_extract_multi_batch", _all_error)
    monkeypatch.setattr(ext, "extract", _slow_extract)

    before = ext._consecutive_batch_failures
    texts = ["a" * 24, "b" * 24]  # > tier1 threshold → LLM path
    results = await asyncio.wait_for(ext.extract_batch(texts), timeout=10)

    assert len(results) == len(texts)
    assert ext._consecutive_batch_failures == before + 1, "breaker must increment exactly once per batch"
