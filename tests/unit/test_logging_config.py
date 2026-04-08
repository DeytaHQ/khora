"""Tests for khora.logging_config — async-safety guards and sink setup.

Added for DYT-2050 to close a coverage gap on setup_logging's enqueue=True
sink configuration and the atexit drain guard. All tests fully mock the
loguru logger, atexit.register, and logging.basicConfig so they do not
mutate global process state.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from khora import logging_config
from khora.logging_config import setup_logging


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
