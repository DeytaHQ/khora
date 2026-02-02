"""Logging configuration for Khora using loguru."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from loguru import logger


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
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging(
    level: str = "INFO",
    json_logs: bool = False,
    log_file: Path | None = None,
) -> None:
    """Configure logging for the application using loguru.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_logs: If True, output logs in JSON format
        log_file: Optional file path to write logs to
    """
    # Remove default loguru handler
    logger.remove()

    # Console handler with custom format
    if json_logs:
        logger.add(
            sys.stdout,
            level=level.upper(),
            serialize=True,  # JSON format
        )
    else:
        logger.add(
            sys.stdout,
            level=level.upper(),
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> | <level>{message}</level>",
            colorize=True,
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
        )

    # Intercept standard logging and redirect to loguru
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # Suppress noisy third-party loggers
    for logger_name in ["httpx", "httpcore", "uvicorn.error", "LiteLLM", "litellm", "prefect", "prefect.flow_runs"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    # Keep uvicorn access logs visible
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
