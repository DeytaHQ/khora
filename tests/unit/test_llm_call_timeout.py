"""Regression tests for the LLM-call wall-clock timeout backstop.

See https://github.com/DeytaHQ/khora/issues/1052. A stalled embedding or
extraction request — TLS handshake completes, the server then sends no bytes —
must be cancelled by ``asyncio.wait_for(..., llm_call_timeout(...))`` instead of
hanging the build forever (litellm's shared-session transport builds a
per-request ``ClientTimeout`` with no ``total``, so the await otherwise never
unblocks).

Each test patches the litellm call to hang and relies on ``pytest-timeout`` to
*fail* (rather than hang CI) if the backstop is missing. The grace margin is
shrunk so the deadline fires quickly.
"""

from __future__ import annotations

import asyncio
import contextlib

import litellm
import pytest

from khora.config import llm as llm_config
from khora.config.llm import llm_call_timeout
from khora.exceptions import EmbeddingError
from khora.extraction.embedders.litellm import LiteLLMEmbedder
from khora.extraction.extractors.llm import LLMEntityExtractor


def test_llm_call_timeout_none_when_unset() -> None:
    """No positive timeout => no enforced deadline (preserves prior behaviour)."""
    assert llm_call_timeout(None) is None
    assert llm_call_timeout(0) is None
    assert llm_call_timeout(-5) is None


def test_llm_call_timeout_adds_grace() -> None:
    """A positive timeout becomes ``timeout`` + a grace margin."""
    assert llm_call_timeout(60) == 60 + llm_config._LLM_DEADLINE_GRACE_S
    assert llm_call_timeout(1.5) == pytest.approx(1.5 + llm_config._LLM_DEADLINE_GRACE_S)


def test_llm_call_timeout_scales_with_attempts() -> None:
    """``attempts`` spans the internal-retry budget (litellm/router num_retries sites)."""
    grace = llm_config._LLM_DEADLINE_GRACE_S
    assert llm_call_timeout(10, attempts=1) == 10 + grace
    assert llm_call_timeout(10, attempts=4) == 40 + grace
    assert llm_call_timeout(None, attempts=4) is None


async def _hang(*_args, **_kwargs):
    """Simulate a connection that completes the handshake then never responds."""
    await asyncio.Event().wait()


@pytest.mark.timeout(20)
async def test_embedder_cancels_hung_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hung embedding request is cancelled within the deadline, not hung forever."""
    monkeypatch.setattr(llm_config, "_LLM_DEADLINE_GRACE_S", 0.2)
    monkeypatch.setattr(litellm, "aembedding", _hang)

    embedder = LiteLLMEmbedder(timeout=1, max_retries=1, cache_max_size=0)
    with pytest.raises((TimeoutError, EmbeddingError)):
        await embedder.embed_batch(["hello world"])


@pytest.mark.timeout(20)
async def test_extractor_cancels_hung_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hung extraction request is cancelled within the deadline, not hung forever.

    The extractor may raise or degrade to an empty result; the property under
    test is that it *returns* within the deadline rather than hanging — enforced
    by the ``pytest.mark.timeout`` marker.
    """
    monkeypatch.setattr(llm_config, "_LLM_DEADLINE_GRACE_S", 0.2)
    monkeypatch.setattr(litellm, "acompletion", _hang)

    extractor = LLMEntityExtractor(timeout=1, max_retries=1, max_concurrent=1)
    with contextlib.suppress(Exception):
        await extractor.extract("Alice met Bob in Paris.")
