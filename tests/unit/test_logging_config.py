"""Tests for khora.logging_config — async-safety guards and sink setup.

Added for DYT-2050 to close a coverage gap on setup_logging's enqueue=True
sink configuration and the atexit drain guard. All tests fully mock the
loguru logger, atexit.register, and logging.basicConfig so they do not
mutate global process state.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from khora import logging_config
from khora.logging_config import apply_neo4j_log_level_from_env, setup_logging
from khora.telemetry import logfire_integration
from khora.telemetry.logfire_integration import (
    _HAS_LOGFIRE,
    _NEO4J_LOGFIRE_HANDLER_MARK,
    install_neo4j_logfire_handler,
)

requires_logfire = pytest.mark.skipif(
    not _HAS_LOGFIRE, reason="logfire not installed"
)


@pytest.fixture(autouse=True)
def _reset_drain_sentinel():
    """Reset the module-level _drain_registered sentinel between tests.

    setup_logging() guards atexit.register(logger.complete) behind this
    sentinel; without the reset, only the first test in this module would
    exercise the registration path and the rest would silently pass.
    """
    logging_config._drain_registered = False
    yield
    logging_config._drain_registered = False


def test_setup_logging_registers_atexit_drain_exactly_once():
    """Multiple setup_logging() calls register atexit exactly once.

    This is the guard behaviour of the _drain_registered sentinel: in tests,
    CLI re-init, or library-consumer reconfiguration, we must not stack
    duplicate atexit handlers — otherwise logger.complete() runs N times on
    shutdown and each call blocks until the queue is empty.
    """
    with (
        patch("khora.logging_config.atexit.register") as mock_register,
        patch("khora.logging_config.logger") as mock_logger,
        patch("khora.logging_config.logging.basicConfig"),
    ):
        setup_logging(level="INFO")
        setup_logging(level="DEBUG")
        setup_logging(level="WARNING")

    assert mock_register.call_count == 1, f"expected exactly one atexit.register call, got {mock_register.call_count}"
    # The registered callable must be logger.complete — that is what drains
    # the background queue on interpreter shutdown.
    assert mock_register.call_args.args[0] is mock_logger.complete


def test_setup_logging_human_stdout_sink_uses_enqueue():
    """The default human-readable stdout sink is installed with enqueue=True."""
    with (
        patch("khora.logging_config.logger") as mock_logger,
        patch("khora.logging_config.atexit.register"),
        patch("khora.logging_config.logging.basicConfig"),
    ):
        setup_logging(level="INFO", json_logs=False, log_file=None)

    assert mock_logger.add.call_count == 1
    assert mock_logger.add.call_args_list[0].kwargs.get("enqueue") is True


def test_setup_logging_json_stdout_sink_uses_enqueue():
    """The JSON-serialised stdout sink is installed with enqueue=True."""
    with (
        patch("khora.logging_config.logger") as mock_logger,
        patch("khora.logging_config.atexit.register"),
        patch("khora.logging_config.logging.basicConfig"),
    ):
        setup_logging(level="INFO", json_logs=True, log_file=None)

    assert mock_logger.add.call_count == 1
    kwargs = mock_logger.add.call_args_list[0].kwargs
    assert kwargs.get("enqueue") is True
    assert kwargs.get("serialize") is True


def test_setup_logging_file_sink_uses_enqueue(tmp_path: Path):
    """The optional rotating-file sink is installed with enqueue=True."""
    with (
        patch("khora.logging_config.logger") as mock_logger,
        patch("khora.logging_config.atexit.register"),
        patch("khora.logging_config.logging.basicConfig"),
    ):
        setup_logging(level="INFO", json_logs=False, log_file=tmp_path / "khora.log")

    # One human stdout sink + one file sink = two logger.add calls.
    assert mock_logger.add.call_count == 2
    file_call = mock_logger.add.call_args_list[1]
    assert file_call.kwargs.get("enqueue") is True
    assert file_call.kwargs.get("rotation") == "10 MB"


@pytest.fixture
def _reset_neo4j_logger_level():
    """Restore the neo4j stdlib logger level so tests don't leak global state."""
    neo4j_logger = logging.getLogger("neo4j")
    original = neo4j_logger.level
    yield neo4j_logger
    neo4j_logger.setLevel(original)


@pytest.mark.parametrize(
    ("env_value", "expected_level"),
    [
        ("DEBUG", logging.DEBUG),
        ("info", logging.INFO),
        ("Warning", logging.WARNING),
        ("ERROR", logging.ERROR),
        ("critical", logging.CRITICAL),
    ],
)
def test_apply_neo4j_log_level_from_env_sets_level(
    monkeypatch: pytest.MonkeyPatch,
    _reset_neo4j_logger_level: logging.Logger,
    env_value: str,
    expected_level: int,
) -> None:
    """Valid KHORA_NEO4J_LOG_LEVEL values set the neo4j logger level (case-insensitive)."""
    monkeypatch.setenv("KHORA_NEO4J_LOG_LEVEL", env_value)
    apply_neo4j_log_level_from_env()
    assert _reset_neo4j_logger_level.level == expected_level


def test_apply_neo4j_log_level_from_env_noop_when_unset(
    monkeypatch: pytest.MonkeyPatch,
    _reset_neo4j_logger_level: logging.Logger,
) -> None:
    """Unset env var leaves the neo4j logger level untouched."""
    monkeypatch.delenv("KHORA_NEO4J_LOG_LEVEL", raising=False)
    _reset_neo4j_logger_level.setLevel(logging.NOTSET)
    apply_neo4j_log_level_from_env()
    assert _reset_neo4j_logger_level.level == logging.NOTSET


def test_apply_neo4j_log_level_from_env_noop_when_empty(
    monkeypatch: pytest.MonkeyPatch,
    _reset_neo4j_logger_level: logging.Logger,
) -> None:
    """Empty env var value is treated as unset."""
    monkeypatch.setenv("KHORA_NEO4J_LOG_LEVEL", "")
    _reset_neo4j_logger_level.setLevel(logging.NOTSET)
    apply_neo4j_log_level_from_env()
    assert _reset_neo4j_logger_level.level == logging.NOTSET


def test_apply_neo4j_log_level_from_env_ignores_unknown_value(
    monkeypatch: pytest.MonkeyPatch,
    _reset_neo4j_logger_level: logging.Logger,
) -> None:
    """Unknown values emit a warning via loguru and do NOT change the level or raise."""
    monkeypatch.setenv("KHORA_NEO4J_LOG_LEVEL", "TRACE")
    _reset_neo4j_logger_level.setLevel(logging.NOTSET)
    with patch("khora.logging_config.logger") as mock_logger:
        apply_neo4j_log_level_from_env()
    assert _reset_neo4j_logger_level.level == logging.NOTSET
    mock_logger.warning.assert_called_once()


def test_apply_neo4j_log_level_from_env_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    _reset_neo4j_logger_level: logging.Logger,
) -> None:
    """Calling the helper twice with the same env var is fine."""
    monkeypatch.setenv("KHORA_NEO4J_LOG_LEVEL", "DEBUG")
    apply_neo4j_log_level_from_env()
    apply_neo4j_log_level_from_env()
    assert _reset_neo4j_logger_level.level == logging.DEBUG


def test_setup_logging_picks_up_neo4j_log_level(
    monkeypatch: pytest.MonkeyPatch,
    _reset_neo4j_logger_level: logging.Logger,
) -> None:
    """setup_logging() routes the env var through apply_neo4j_log_level_from_env."""
    monkeypatch.setenv("KHORA_NEO4J_LOG_LEVEL", "DEBUG")
    with (
        patch("khora.logging_config.logger"),
        patch("khora.logging_config.atexit.register"),
        patch("khora.logging_config.logging.basicConfig"),
    ):
        setup_logging(level="INFO")
    assert _reset_neo4j_logger_level.level == logging.DEBUG


def test_setup_logging_does_not_touch_neo4j_logger_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
    _reset_neo4j_logger_level: logging.Logger,
) -> None:
    """Default unset behavior: setup_logging() leaves the neo4j logger alone."""
    monkeypatch.delenv("KHORA_NEO4J_LOG_LEVEL", raising=False)
    _reset_neo4j_logger_level.setLevel(logging.NOTSET)
    with (
        patch("khora.logging_config.logger"),
        patch("khora.logging_config.atexit.register"),
        patch("khora.logging_config.logging.basicConfig"),
    ):
        setup_logging(level="INFO")
    assert _reset_neo4j_logger_level.level == logging.NOTSET


def test_setup_logging_atexit_not_reregistered_after_first_call():
    """Once the sentinel is set, atexit.register is not called on subsequent
    setup_logging() invocations — even when kwargs change."""
    with (
        patch("khora.logging_config.atexit.register") as mock_register,
        patch("khora.logging_config.logger"),
        patch("khora.logging_config.logging.basicConfig"),
    ):
        setup_logging(level="INFO")
        assert mock_register.call_count == 1

        # Change kwargs — should still not re-register.
        setup_logging(level="DEBUG", json_logs=True)
        assert mock_register.call_count == 1

        setup_logging(level="WARNING", log_file=Path("/tmp/khora-test.log"))
        assert mock_register.call_count == 1


def _marked_neo4j_handlers() -> list[logging.Handler]:
    return [h for h in logging.getLogger("neo4j").handlers if getattr(h, _NEO4J_LOGFIRE_HANDLER_MARK, False)]


@pytest.fixture
def _reset_neo4j_handlers():
    """Strip khora-marked neo4j logfire handlers around each test.

    Runs before AND after the test body to defend against leftover handlers
    from other test modules that may have called ``setup_logging()`` or
    ``install_neo4j_logfire_handler()`` at import time.
    """
    neo4j_logger = logging.getLogger("neo4j")
    for h in list(neo4j_logger.handlers):
        if getattr(h, _NEO4J_LOGFIRE_HANDLER_MARK, False):
            neo4j_logger.removeHandler(h)
    yield
    for h in list(neo4j_logger.handlers):
        if getattr(h, _NEO4J_LOGFIRE_HANDLER_MARK, False):
            neo4j_logger.removeHandler(h)


@requires_logfire
def test_install_neo4j_logfire_handler_attaches_when_env_set_and_logfire_available(
    monkeypatch: pytest.MonkeyPatch,
    _reset_neo4j_handlers: None,
) -> None:
    """With the env var set and logfire installed, a marked handler is attached."""
    from logfire.integrations.logging import LogfireLoggingHandler

    monkeypatch.setenv("KHORA_NEO4J_LOG_LEVEL", "DEBUG")
    assert install_neo4j_logfire_handler() is True
    marked = _marked_neo4j_handlers()
    assert len(marked) == 1
    assert isinstance(marked[0], LogfireLoggingHandler)


def test_install_neo4j_logfire_handler_noop_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
    _reset_neo4j_handlers: None,
) -> None:
    """Unset env var → no handler attached, returns False."""
    monkeypatch.delenv("KHORA_NEO4J_LOG_LEVEL", raising=False)
    assert install_neo4j_logfire_handler() is False
    assert _marked_neo4j_handlers() == []


def test_install_neo4j_logfire_handler_noop_when_env_empty(
    monkeypatch: pytest.MonkeyPatch,
    _reset_neo4j_handlers: None,
) -> None:
    """Empty env var is treated as unset."""
    monkeypatch.setenv("KHORA_NEO4J_LOG_LEVEL", "")
    assert install_neo4j_logfire_handler() is False
    assert _marked_neo4j_handlers() == []


def test_install_neo4j_logfire_handler_noop_when_logfire_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    _reset_neo4j_handlers: None,
) -> None:
    """When logfire is unavailable, the helper short-circuits cleanly."""
    monkeypatch.setattr(logfire_integration, "_HAS_LOGFIRE", False)
    monkeypatch.setenv("KHORA_NEO4J_LOG_LEVEL", "DEBUG")
    assert install_neo4j_logfire_handler() is False
    assert _marked_neo4j_handlers() == []


@requires_logfire
def test_install_neo4j_logfire_handler_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    _reset_neo4j_handlers: None,
) -> None:
    """Repeated calls must not stack duplicate marked handlers."""
    monkeypatch.setenv("KHORA_NEO4J_LOG_LEVEL", "DEBUG")
    install_neo4j_logfire_handler()
    install_neo4j_logfire_handler()
    assert len(_marked_neo4j_handlers()) == 1


@requires_logfire
def test_setup_logging_installs_neo4j_logfire_handler_when_env_set(
    monkeypatch: pytest.MonkeyPatch,
    _reset_neo4j_logger_level: logging.Logger,
    _reset_neo4j_handlers: None,
) -> None:
    """setup_logging() wires the Logfire handler onto the neo4j logger."""
    monkeypatch.setenv("KHORA_NEO4J_LOG_LEVEL", "DEBUG")
    with (
        patch("khora.logging_config.logger"),
        patch("khora.logging_config.atexit.register"),
        patch("khora.logging_config.logging.basicConfig"),
    ):
        setup_logging(level="INFO")
    assert len(_marked_neo4j_handlers()) == 1


def test_setup_logging_does_not_install_neo4j_logfire_handler_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
    _reset_neo4j_logger_level: logging.Logger,
    _reset_neo4j_handlers: None,
) -> None:
    """Without the env var, setup_logging() must not attach a Logfire handler."""
    monkeypatch.delenv("KHORA_NEO4J_LOG_LEVEL", raising=False)
    with (
        patch("khora.logging_config.logger"),
        patch("khora.logging_config.atexit.register"),
        patch("khora.logging_config.logging.basicConfig"),
    ):
        setup_logging(level="INFO")
    assert _marked_neo4j_handlers() == []
