"""Unit tests for storage backend mixins."""

from __future__ import annotations

import pytest

from khora.storage.backends.mixins import AsyncSessionMixin, _is_deadlock_error, retry_on_deadlock


class TestAsyncSessionMixin:
    """Tests for AsyncSessionMixin._get_session."""

    def test_get_session_raises_when_not_connected(self) -> None:
        """_get_session raises RuntimeError when _session_factory is None."""

        class FakeBackend(AsyncSessionMixin):
            def __init__(self):
                self._session_factory = None

        backend = FakeBackend()
        with pytest.raises(RuntimeError, match="Backend not connected"):
            backend._get_session()

    def test_get_session_returns_session_when_connected(self) -> None:
        """_get_session calls and returns _session_factory() when connected."""
        sentinel = object()

        class FakeBackend(AsyncSessionMixin):
            def __init__(self):
                self._session_factory = lambda: sentinel

        backend = FakeBackend()
        assert backend._get_session() is sentinel


class TestIsDeadlockError:
    """Tests for _is_deadlock_error helper."""

    def test_detects_deadlock(self) -> None:
        assert _is_deadlock_error(Exception("deadlock detected"))

    def test_detects_serialization(self) -> None:
        assert _is_deadlock_error(Exception("serialization failure"))

    def test_ignores_other_errors(self) -> None:
        assert not _is_deadlock_error(Exception("connection refused"))


class TestRetryOnDeadlock:
    """Tests for retry_on_deadlock decorator."""

    @pytest.mark.asyncio
    async def test_retries_on_deadlock_then_succeeds(self) -> None:
        """Decorator retries after a deadlock error and eventually succeeds."""
        call_count = 0

        @retry_on_deadlock
        async def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("deadlock detected")
            return "ok"

        result = await flaky_operation()
        assert result == "ok"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_reraises_after_max_attempts(self) -> None:
        """Decorator reraises after exhausting retry attempts."""

        @retry_on_deadlock
        async def always_deadlock():
            raise Exception("deadlock detected")

        with pytest.raises(Exception, match="deadlock"):
            await always_deadlock()

    @pytest.mark.asyncio
    async def test_does_not_retry_non_deadlock(self) -> None:
        """Decorator does not retry non-deadlock errors."""
        call_count = 0

        @retry_on_deadlock
        async def non_deadlock():
            nonlocal call_count
            call_count += 1
            raise ValueError("something else")

        with pytest.raises(ValueError, match="something else"):
            await non_deadlock()
        assert call_count == 1
