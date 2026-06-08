"""Pytest configuration for the integration test job.

Loud-DB-down guards: when ``KHORA_PG_REQUIRED=1`` is set (the CI integration
job sets it), a missing or misconfigured Postgres must FAIL the run, not pass
by skipping the PG-gated tests. The same applies to Neo4j: when
``NEO4J_INTEGRATION_TEST=1`` is set (the CI integration job also sets it, and
it is the same flag that flips the per-module ``skipif`` on the real-Neo4j
tests), an unreachable Neo4j must FAIL the run rather than silently skip.
Without the env vars (a local dev box without ``make dev``) the guards are
no-ops, so the existing per-module ``skipif`` still lets those tests skip
cleanly.
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


# The real-Neo4j integration modules default to bolt://localhost:7687, and the
# CI integration job's Neo4j service maps that same port; honor an explicit
# KHORA_NEO4J_URL override (local `make dev` exposes bolt on 7688 — set
# KHORA_NEO4J_URL to point at it).
_DEFAULT_NEO4J_URL = "bolt://localhost:7687"


def _neo4j_url() -> str:
    return os.environ.get("KHORA_NEO4J_URL", _DEFAULT_NEO4J_URL)


def _neo4j_reachable() -> bool:
    parsed = urlparse(_neo4j_url())
    host = parsed.hostname or "localhost"
    port = parsed.port or 7687
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def pytest_configure(config: pytest.Config) -> None:
    """Fail loudly at session start if a required backend is unreachable.

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

    if os.environ.get("NEO4J_INTEGRATION_TEST") == "1" and not _neo4j_reachable():
        pytest.exit(
            f"NEO4J_INTEGRATION_TEST=1 but Neo4j is unreachable at {_neo4j_url()}. "
            "The CI integration job provisions Neo4j via a services block; a "
            "skip here would hide real failures.",
            returncode=1,
        )
