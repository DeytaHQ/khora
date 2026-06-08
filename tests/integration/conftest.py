"""Pytest configuration for the integration test job.

Loud-DB-down guard: when ``KHORA_PG_REQUIRED=1`` is set (the CI integration
job sets it), a missing or misconfigured Postgres must FAIL the run, not pass
by skipping the PG-gated tests. Without the env var (a local dev box without
``make dev``) the guard is a no-op, so the existing per-module ``skipif`` still
lets those tests skip cleanly.
"""

from __future__ import annotations

import os
import socket
from urllib.parse import urlparse

import pytest

# This repo's compose puts Postgres on 5434 (see compose.yaml); honor an explicit
# KHORA_DATABASE_URL override, else default to the compose port.
_DEFAULT_DATABASE_URL = "postgresql://khora:khora@localhost:5434/khora"


def _database_url() -> str:
    return os.environ.get("KHORA_DATABASE_URL", _DEFAULT_DATABASE_URL)


def _pg_reachable() -> bool:
    parsed = urlparse(_database_url().replace("+asyncpg", ""))
    host = parsed.hostname or "localhost"
    port = parsed.port or 5432
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def pytest_configure(config: pytest.Config) -> None:
    """Fail loudly at session start if Postgres is required but unreachable.

    ``pytest.exit(returncode=1)`` aborts the whole session with a red exit,
    so a DB-down integration job cannot pass by silently skipping its tests.
    """
    if os.environ.get("KHORA_PG_REQUIRED") == "1" and not _pg_reachable():
        pytest.exit(
            f"KHORA_PG_REQUIRED=1 but Postgres is unreachable at {_database_url()}. "
            "The CI integration job provisions Postgres via a services block; a "
            "skip here would hide real failures.",
            returncode=1,
        )
