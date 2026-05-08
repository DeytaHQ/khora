"""Unit tests for the halfvec-index diagnostics on PgVectorBackend (DYT-3787).

The connect path used to log a single WARNING — `halfvec HNSW indexes not found —
falling back to full-precision vectors. Run migrations to create them.` — for
both the "production DB missing migrations" case (real misconfiguration) and the
"fresh ephemeral DB before any migrations have run" case (benign, common in
khora-benchmarks-service per-run DBs).

The new `_halfvec_target_tables_exist` helper distinguishes them so the two
cases are logged at different levels.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from khora.storage.backends.pgvector import PgVectorBackend


def _make_backend() -> PgVectorBackend:
    """Construct a backend without touching the network / SQLAlchemy."""
    return PgVectorBackend.__new__(PgVectorBackend)


def _patch_session(backend: PgVectorBackend, scalar_value: Any) -> AsyncMock:
    """Stub `_get_session` so that `session.execute(...).scalar_one_or_none()`
    returns *scalar_value*."""
    result = MagicMock()
    result.scalar_one_or_none = MagicMock(return_value=scalar_value)
    session = AsyncMock()
    session.execute = AsyncMock(return_value=result)

    cm = AsyncMock()
    cm.__aenter__.return_value = session
    cm.__aexit__.return_value = False
    backend._get_session = MagicMock(return_value=cm)  # type: ignore[attr-defined]
    return session


@pytest.mark.asyncio
async def test_halfvec_target_tables_exist_returns_true_when_both_tables_present() -> None:
    backend = _make_backend()
    _patch_session(backend, 2)
    assert await backend._halfvec_target_tables_exist() is True


@pytest.mark.asyncio
async def test_halfvec_target_tables_exist_returns_false_when_no_tables() -> None:
    """Fresh DB: pg_class returns 0 rows for chunks/entities."""
    backend = _make_backend()
    _patch_session(backend, 0)
    assert await backend._halfvec_target_tables_exist() is False


@pytest.mark.asyncio
async def test_halfvec_target_tables_exist_returns_false_when_only_one_present() -> None:
    """Partial schema (e.g. ``chunks`` exists but not ``entities``) is also a
    "not ready yet" state — return False so the caller doesn't fire the WARN
    that points the operator to a migration that won't fully recover the DB."""
    backend = _make_backend()
    _patch_session(backend, 1)
    assert await backend._halfvec_target_tables_exist() is False


@pytest.mark.asyncio
async def test_halfvec_target_tables_exist_returns_false_on_db_error() -> None:
    backend = _make_backend()
    cm = AsyncMock()
    cm.__aenter__.side_effect = RuntimeError("connection lost")
    cm.__aexit__.return_value = False
    backend._get_session = MagicMock(return_value=cm)  # type: ignore[attr-defined]

    assert await backend._halfvec_target_tables_exist() is False
