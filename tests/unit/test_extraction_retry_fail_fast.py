"""Unit tests for fail-fast on non-retryable credential/auth errors.

Pre-fix, ``LLMEntityExtractor`` retried *every* exception up to
``max_retries`` with exponential backoff — including deterministic
credential/auth failures (a missing or invalid API key) that will
never succeed no matter how many times they are retried. That burned
three attempts and ~180s of backoff for a guaranteed-empty result.

The fix adds a module-level predicate ``_is_nonretryable_auth_error``
and wires both ``AsyncRetrying`` blocks (in ``extract`` and
``_extract_multi_batch``) with
``retry=retry_if_exception(lambda e: not _is_nonretryable_auth_error(e))``
plus ``reraise=True``. A non-retryable error therefore fails fast (one
attempt, no backoff) and ``reraise=True`` propagates it to the existing
``except Exception`` handler, which returns an
``ExtractionResult(metadata={"error": ...})`` rather than raising out.

This file pins:

A. ``_is_nonretryable_auth_error`` classification (direct unit tests).
B. ``extract`` (single path) fails fast on a missing-credentials error.
C. ``_extract_multi_batch`` (multi path) fails fast on the same error.
D. Regression guard: a genuine transient 5xx is STILL retried.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import litellm
import pytest

from khora.extraction.extractors.llm import (
    LLMEntityExtractor,
    _is_nonretryable_auth_error,
)


def _auth_error(message: str = "Invalid API key") -> litellm.AuthenticationError:
    """Build a real ``litellm.AuthenticationError`` (response is optional)."""
    return litellm.AuthenticationError(message=message, llm_provider="openai", model="gpt-4o-mini")


def _permission_denied_error(message: str = "Access denied") -> litellm.PermissionDeniedError:
    """Build a real ``litellm.PermissionDeniedError``.

    Unlike its siblings, this type requires a non-optional ``response``,
    so we hand it a minimal httpx 403 response.
    """
    response = httpx.Response(
        status_code=403,
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )
    return litellm.PermissionDeniedError(message=message, llm_provider="bedrock", model="claude", response=response)


def _internal_server_error(message: str) -> litellm.InternalServerError:
    """Build a real ``litellm.InternalServerError`` (response is optional)."""
    return litellm.InternalServerError(message=message, llm_provider="openai", model="gpt-4o-mini")


def _rate_limit_error(message: str) -> litellm.RateLimitError:
    """Build a real ``litellm.RateLimitError`` (response is optional)."""
    return litellm.RateLimitError(message=message, llm_provider="openai", model="gpt-4o-mini")


# Missing-credentials wording litellm surfaces as an InternalServerError when
# the client cannot be constructed because no key is configured.
_MISSING_CREDS_MESSAGE = (
    "OpenAIException - Missing credentials. Please pass an api_key, or set the OPENAI_API_KEY environment variable."
)


@pytest.mark.unit
class TestIsNonretryableAuthError:
    """Direct unit tests for the ``_is_nonretryable_auth_error`` predicate.

    These exercise ``isinstance`` against real litellm classes plus the
    message-shape match — no LLM call mocking needed, so they are fast.
    """

    def test_authentication_error_is_nonretryable(self) -> None:
        """A genuine ``AuthenticationError`` is non-retryable regardless of message."""
        assert _is_nonretryable_auth_error(_auth_error()) is True

    def test_permission_denied_error_is_nonretryable(self) -> None:
        """A genuine ``PermissionDeniedError`` (IAM/authz denial) is non-retryable."""
        assert _is_nonretryable_auth_error(_permission_denied_error()) is True

    def test_internal_server_error_with_missing_credentials_is_nonretryable(self) -> None:
        """Missing-credentials surfaced as InternalServerError matches by message shape.

        Both a credential token ("api_key") AND a missing signal
        ("missing" / "pass an api" / "set the") are present.
        """
        assert _is_nonretryable_auth_error(_internal_server_error(_MISSING_CREDS_MESSAGE)) is True

    def test_internal_server_error_generic_5xx_is_retryable(self) -> None:
        """A plain "Internal Server Error" carries neither token nor signal -> retryable."""
        assert _is_nonretryable_auth_error(_internal_server_error("Internal Server Error")) is False

    def test_internal_server_error_overloaded_is_retryable(self) -> None:
        """An "overloaded" 5xx is transient -> retryable."""
        assert _is_nonretryable_auth_error(_internal_server_error("the model is overloaded")) is False

    def test_rate_limit_error_mentioning_api_key_is_retryable(self) -> None:
        """The key false-positive guard: a 429 that echoes the key is RETRYABLE.

        A bare ``api_key`` token with no missing/absent signal must not be
        mistaken for a credentials error — a RateLimitError is transient.
        """
        rle = _rate_limit_error("Rate limit reached for default-gpt-4o-mini on api_key sk-abc123, retry soon")
        assert _is_nonretryable_auth_error(rle) is False

    def test_plain_exception_is_retryable(self) -> None:
        """An unrelated exception (no auth type, no credential wording) -> retryable."""
        assert _is_nonretryable_auth_error(Exception("connection reset by peer")) is False


@pytest.mark.unit
class TestMroNameFallback:
    """The name-MRO branch works without relying on ``isinstance`` against litellm.

    ``_is_nonretryable_auth_error`` walks ``type(exc).__mro__`` for a class
    literally named ``AuthenticationError`` / ``PermissionDeniedError`` after
    the litellm ``isinstance`` check. That fallback keeps the predicate correct
    if the lazy ``import litellm`` inside the helper fails, and also catches
    provider-specific subclasses. These tests use locally-defined classes that
    share only the name (not litellm's MRO), so they exercise that branch in
    isolation — no message tokens are present, so a True result can only come
    from the name match.
    """

    def test_locally_named_authentication_error_is_nonretryable(self) -> None:
        """A class literally named ``AuthenticationError`` matches by MRO name."""

        class AuthenticationError(Exception):
            pass

        assert _is_nonretryable_auth_error(AuthenticationError("bad key")) is True

    def test_locally_named_permission_denied_error_is_nonretryable(self) -> None:
        """A class literally named ``PermissionDeniedError`` matches by MRO name."""

        class PermissionDeniedError(Exception):
            pass

        assert _is_nonretryable_auth_error(PermissionDeniedError("denied")) is True

    def test_subclass_of_named_auth_error_matches_via_mro_walk(self) -> None:
        """A provider subclass of a named auth error is caught by the MRO walk."""

        class AuthenticationError(Exception):
            pass

        class MyProviderAuthError(AuthenticationError):
            pass

        assert _is_nonretryable_auth_error(MyProviderAuthError("x")) is True

    def test_unrelated_named_exception_is_retryable(self) -> None:
        """A differently-named exception with no credential wording is retryable."""

        class SomethingElseError(Exception):
            pass

        assert _is_nonretryable_auth_error(SomethingElseError("whatever")) is False


@pytest.mark.unit
class TestExtractFailsFastOnCredentialError:
    """Integration through ``extract`` (single path)."""

    @pytest.mark.asyncio
    async def test_missing_credentials_fails_fast_no_retry(self) -> None:
        """A missing-credentials error is attempted once and returned as an error result.

        Asserts ``acompletion`` is awaited EXACTLY once (no retries) and
        ``extract`` returns — does not raise — an ``ExtractionResult`` whose
        metadata carries the error. ``asyncio.sleep`` is patched so that if a
        backoff were ever taken the assertion on call count would still catch
        it, but a fail-fast path takes no sleep at all.
        """
        extractor = LLMEntityExtractor(model="test-model", max_retries=3)

        acompletion = AsyncMock(side_effect=_internal_server_error(_MISSING_CREDS_MESSAGE))
        sleep = AsyncMock()
        with (
            patch("litellm.acompletion", acompletion),
            patch("asyncio.sleep", sleep),
        ):
            result = await extractor.extract(
                "Alice works at Acme Corp",
                entity_types=["PERSON", "ORGANIZATION"],
                relationship_types=["WORKS_FOR"],
            )

        # Returned, not raised.
        assert "error" in result.metadata
        assert "Missing credentials" in result.metadata["error"]
        # Exactly one attempt — no retries.
        assert acompletion.await_count == 1
        # No backoff was taken (fail-fast, zero retries).
        sleep.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_authentication_error_fails_fast_no_retry(self) -> None:
        """A genuine ``AuthenticationError`` also fails fast through ``extract``."""
        extractor = LLMEntityExtractor(model="test-model", max_retries=3)

        acompletion = AsyncMock(side_effect=_auth_error())
        with (
            patch("litellm.acompletion", acompletion),
            patch("asyncio.sleep", new_callable=AsyncMock) as sleep,
        ):
            result = await extractor.extract(
                "Alice works at Acme Corp",
                entity_types=["PERSON"],
                relationship_types=["WORKS_FOR"],
            )

        assert "error" in result.metadata
        assert acompletion.await_count == 1
        sleep.assert_not_awaited()


@pytest.mark.unit
class TestExtractMultiBatchFailsFastOnCredentialError:
    """Integration through ``_extract_multi_batch`` (multi path)."""

    @pytest.mark.asyncio
    async def test_missing_credentials_fails_fast_no_retry(self) -> None:
        """The multi-batch path also fails fast and returns one error result per text.

        ``_extract_multi_batch`` takes the ``litellm`` module as a parameter
        (``extract_multi`` imports it at function scope and threads it
        through), so we hand it a mock module whose ``acompletion`` raises a
        missing-credentials error. Asserts exactly one attempt and a list of
        ``ExtractionResult`` carrying the error — does not raise.
        """
        extractor = LLMEntityExtractor(model="test-model", max_retries=3)

        mock_litellm = MagicMock()
        mock_litellm.acompletion = AsyncMock(side_effect=_internal_server_error(_MISSING_CREDS_MESSAGE))

        texts = ["first section text", "second section text"]
        with patch("asyncio.sleep", new_callable=AsyncMock) as sleep:
            results = await extractor._extract_multi_batch(
                texts,
                ["PERSON", "ORGANIZATION"],
                mock_litellm,
                relationship_types=["WORKS_FOR"],
            )

        # One error result per input text — returned, not raised.
        assert len(results) == len(texts)
        for r in results:
            assert "error" in r.metadata
            assert "Missing credentials" in r.metadata["error"]
        # Exactly one attempt — no retries, no backoff.
        assert mock_litellm.acompletion.await_count == 1
        sleep.assert_not_awaited()


@pytest.mark.unit
class TestRetryableErrorsStillRetry:
    """Regression guard: transient errors keep their pre-fix retry behavior."""

    @pytest.mark.asyncio
    async def test_transient_5xx_is_retried_in_extract(self) -> None:
        """A genuine transient ``InternalServerError`` is retried up to ``max_retries``.

        No credential wording, so the predicate returns False and the retry
        loop runs the full budget before the ``except`` handler returns an
        error result. ``asyncio.sleep`` is patched to keep the test fast;
        ``retry_wait`` is tiny as a belt-and-braces measure.
        """
        max_retries = 3
        extractor = LLMEntityExtractor(model="test-model", max_retries=max_retries, retry_wait=0.001)

        acompletion = AsyncMock(side_effect=_internal_server_error("Internal Server Error"))
        with (
            patch("litellm.acompletion", acompletion),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await extractor.extract(
                "Alice works at Acme Corp",
                entity_types=["PERSON"],
                relationship_types=["WORKS_FOR"],
            )

        # Eventually surfaces as an error result after exhausting retries.
        assert "error" in result.metadata
        # Retried the full budget — more than one attempt.
        assert acompletion.await_count == max_retries

    @pytest.mark.asyncio
    async def test_rate_limit_error_is_retried_in_extract(self) -> None:
        """A 429 whose message mentions ``api_key`` is still retried (false-positive guard).

        This is the integration-level counterpart to
        ``test_rate_limit_error_mentioning_api_key_is_retryable``: a key
        mention with no missing signal must not be mistaken for a
        non-retryable credentials error.
        """
        max_retries = 3
        extractor = LLMEntityExtractor(model="test-model", max_retries=max_retries, retry_wait=0.001)

        rle = _rate_limit_error("Rate limit reached for default-gpt-4o-mini on api_key sk-abc123")
        acompletion = AsyncMock(side_effect=rle)
        with (
            patch("litellm.acompletion", acompletion),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await extractor.extract(
                "Alice works at Acme Corp",
                entity_types=["PERSON"],
                relationship_types=["WORKS_FOR"],
            )

        assert "error" in result.metadata
        assert acompletion.await_count == max_retries
