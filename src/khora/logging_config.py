"""Logging configuration for Khora using loguru."""

from __future__ import annotations

import atexit
import logging
import sys
from pathlib import Path

from loguru import logger

# Module-level sentinel: atexit.register stacks duplicate callables, so guard
# against re-registering logger.complete when setup_logging() is called more
# than once (e.g. tests reconfiguring logging between cases).
_drain_registered = False


class InterceptHandler(logging.Handler):
    """Intercept standard logging and redirect to loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record by forwarding to loguru."""
        # Get corresponding loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame is not None and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # f_back is FrameType | None but loop guard handles it
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging(
    level: str = "INFO",
    json_logs: bool = False,
    log_file: Path | None = None,
) -> None:
    """Configure logging for the application using loguru.

    All sinks are added with ``enqueue=True`` so that log writes never block
    the event loop from inside ``async def`` code paths — loguru pushes
    records onto a background thread that owns the actual I/O. To guarantee
    queued records are flushed before interpreter shutdown (normal exit,
    ``sys.exit``, or an exception propagating out of ``main``), we register
    ``logger.complete`` with ``atexit`` exactly once per process.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_logs: If True, output logs in JSON format
        log_file: Optional file path to write logs to
    """
    global _drain_registered

    # Remove default loguru handler
    logger.remove()

    # NOTE: loguru 0.7.3 does not expose maxsize; queue is unbounded.
    # Acceptable for this codebase because log volume is bounded by request
    # rate, not by a loop.

    # Console handler with custom format
    if json_logs:
        logger.add(
            sys.stdout,
            level=level.upper(),
            serialize=True,  # JSON format
            enqueue=True,  # async-safe: queue writes off the event loop
        )
    else:
        logger.add(
            sys.stdout,
            level=level.upper(),
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> | <level>{message}</level>",
            colorize=True,
            enqueue=True,  # async-safe: queue writes off the event loop
        )

    # File handler (optional)
    if log_file:
        logger.add(
            log_file,
            level=level.upper(),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} | {message}",
            rotation="10 MB",
            retention="7 days",
            serialize=json_logs,
            enqueue=True,  # async-safe: queue writes off the event loop
        )

    # Drain the loguru queue on interpreter shutdown so buffered records are
    # not dropped. Guarded by a sentinel because atexit stacks duplicate
    # registrations and setup_logging() may be re-invoked (e.g. in tests).
    if not _drain_registered:
        atexit.register(logger.complete)
        _drain_registered = True

    # Intercept standard logging and redirect to loguru
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # Suppress noisy third-party loggers
    for logger_name in ["httpx", "httpcore", "LiteLLM", "litellm"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
