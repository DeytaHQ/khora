"""Unit tests for PgVectorBackend halfvec index guard and cosine similarity.

Verifies that:
- connect() correctly enables/disables halfvec based on extension + index presence
- _check_halfvec_indexes() handles present, partial, and error cases
- _cosine_similarity() uses HALFVEC cast when enabled, plain vector when not
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from khora.storage.backends.pgvector import PgVectorBackend


def _make_backend(*, use_halfvec: bool = True, embedding_dimension: int = 1536) -> PgVectorBackend:
    """Create a PgVectorBackend without connecting to a real database."""
    backend = PgVectorBackend.__new__(PgVectorBackend)
    backend._database_url = "postgresql+asyncpg://localhost/test"
    backend._embedding_dimension = embedding_dimension
    backend._echo = False
    backend._pool_size = 5
    backend._max_overflow = 10
    backend._pool_pre_ping = False
    backend._hnsw_ef_search = 100
    backend._use_halfvec = use_halfvec
    backend._halfvec_available = None
    backend._engine = None
    backend._engine_shared = False
    backend._session_factory = None
    return backend


@pytest.mark.unit
class TestConnectHalfvecGuard:
    """Tests for halfvec guard logic in connect()."""

    async def test_connect_disables_halfvec_when_indexes_missing(self) -> None:
        """When extension supports halfvec but indexes are missing, halfvec should be disabled."""
        backend = _make_backend(use_halfvec=True)

        with (
            patch.object(backend, "_detect_halfvec_support", new_callable=AsyncMock, return_value=True),
            patch.object(backend, "_check_halfvec_indexes", new_callable=AsyncMock, return_value=False),
            patch("khora.storage.backends.pgvector.create_async_engine") as mock_engine,
            patch("khora.storage.backends.pgvector.async_sessionmaker") as mock_session_maker,
        ):
            # Set up engine mock to handle the CREATE EXTENSION call
            mock_conn = AsyncMock()
            mock_engine.return_value.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_engine.return_value.begin.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_session_maker.return_value = MagicMock()
            backend._engine = None
            backend._session_factory = None

            await backend.connect()

            assert backend.halfvec_enabled is False
            assert backend._halfvec_available is False

    async def test_connect_enables_halfvec_when_indexes_present(self) -> None:
        """When extension supports halfvec and indexes exist, halfvec should be enabled."""
        backend = _make_backend(use_halfvec=True)

        with (
            patch.object(backend, "_detect_halfvec_support", new_callable=AsyncMock, return_value=True),
            patch.object(backend, "_check_halfvec_indexes", new_callable=AsyncMock, return_value=True),
            patch("khora.storage.backends.pgvector.create_async_engine") as mock_engine,
            patch("khora.storage.backends.pgvector.async_sessionmaker") as mock_session_maker,
        ):
            mock_conn = AsyncMock()
            mock_engine.return_value.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_engine.return_value.begin.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_session_maker.return_value = MagicMock()

            await backend.connect()

            assert backend.halfvec_enabled is True
            assert backend._halfvec_available is True

    async def test_connect_skips_index_check_when_extension_unsupported(self) -> None:
        """When pgvector < 0.7.0, _check_halfvec_indexes should not be called."""
        backend = _make_backend(use_halfvec=True)

        with (
            patch.object(backend, "_detect_halfvec_support", new_callable=AsyncMock, return_value=False),
            patch.object(backend, "_check_halfvec_indexes", new_callable=AsyncMock) as mock_check,
            patch("khora.storage.backends.pgvector.create_async_engine") as mock_engine,
            patch("khora.storage.backends.pgvector.async_sessionmaker") as mock_session_maker,
        ):
            mock_conn = AsyncMock()
            mock_engine.return_value.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_engine.return_value.begin.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_session_maker.return_value = MagicMock()

            await backend.connect()

            mock_check.assert_not_called()
            assert backend.halfvec_enabled is False

    async def test_connect_skips_all_halfvec_checks_when_disabled(self) -> None:
        """When use_halfvec=False, no halfvec detection or index check should occur."""
        backend = _make_backend(use_halfvec=False)

        with (
            patch.object(backend, "_detect_halfvec_support", new_callable=AsyncMock) as mock_detect,
            patch.object(backend, "_check_halfvec_indexes", new_callable=AsyncMock) as mock_check,
            patch("khora.storage.backends.pgvector.create_async_engine") as mock_engine,
            patch("khora.storage.backends.pgvector.async_sessionmaker") as mock_session_maker,
        ):
            mock_conn = AsyncMock()
            mock_engine.return_value.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_engine.return_value.begin.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_session_maker.return_value = MagicMock()

            await backend.connect()

            mock_detect.assert_not_called()
            mock_check.assert_not_called()
            assert backend.halfvec_enabled is False
            assert backend._halfvec_available is None  # Never touched


@pytest.mark.unit
class TestCheckHalfvecIndexes:
    """Tests for _check_halfvec_indexes()."""

    async def test_returns_true_when_both_exist(self) -> None:
        """Should return True when both halfvec HNSW indexes are found."""
        backend = _make_backend()
        backend._session_factory = MagicMock()

        mock_result = MagicMock()
        mock_result.all.return_value = [
            ("ix_chunks_embedding_halfvec_hnsw", True),
            ("ix_entities_embedding_halfvec_hnsw", True),
        ]

        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch.object(backend, "_get_session", return_value=mock_session):
            result = await backend._check_halfvec_indexes()

        assert result is True

    async def test_returns_false_when_partial(self) -> None:
        """Should return False when only one halfvec index exists."""
        backend = _make_backend()
        backend._session_factory = MagicMock()

        mock_result = MagicMock()
        mock_result.all.return_value = [
            ("ix_chunks_embedding_halfvec_hnsw", True),
        ]

        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch.object(backend, "_get_session", return_value=mock_session):
            result = await backend._check_halfvec_indexes()

        assert result is False

    async def test_returns_false_on_error(self) -> None:
        """Should return False (graceful fallback) when query raises an exception."""
        backend = _make_backend()
        backend._session_factory = MagicMock()

        mock_session = AsyncMock()
        mock_session.execute.side_effect = RuntimeError("connection lost")
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch.object(backend, "_get_session", return_value=mock_session):
            result = await backend._check_halfvec_indexes()

        assert result is False

    async def test_returns_false_when_index_invalid(self) -> None:
        """Should return False when an index exists but is marked invalid."""
        backend = _make_backend()
        backend._session_factory = MagicMock()

        mock_result = MagicMock()
        mock_result.all.return_value = [
            ("ix_chunks_embedding_halfvec_hnsw", True),
            ("ix_entities_embedding_halfvec_hnsw", False),  # invalid
        ]

        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch.object(backend, "_get_session", return_value=mock_session):
            result = await backend._check_halfvec_indexes()

        assert result is False


@pytest.mark.unit
class TestCosineDistanceHalfvec:
    """Tests for _cosine_distance() halfvec cast behavior."""

    def test_uses_halfvec_cast_when_enabled(self) -> None:
        """When halfvec is enabled, the SQL expression should contain HALFVEC cast."""
        backend = _make_backend(use_halfvec=True)
        backend._halfvec_available = True

        mock_col = MagicMock()
        query_embedding = [0.1] * 1536

        expr = backend._cosine_distance(mock_col, query_embedding)

        # The expression should involve func.cast calls - the column should NOT
        # have cosine_distance called directly on it (it's called on the casted version)
        compiled = str(expr)
        assert "CAST" in compiled.upper() or "halfvec" in compiled.lower() or "cast" in compiled.lower()
        # The original column's cosine_distance should NOT be called directly
        mock_col.cosine_distance.assert_not_called()

    def test_uses_vector_when_disabled(self) -> None:
        """When halfvec is disabled, cosine_distance is called directly on the column."""
        backend = _make_backend(use_halfvec=False)
        backend._halfvec_available = False

        mock_col = MagicMock()
        mock_col.cosine_distance.return_value = MagicMock()
        query_embedding = [0.1] * 1536

        backend._cosine_distance(mock_col, query_embedding)

        # The column's cosine_distance should be called directly with the query embedding
        mock_col.cosine_distance.assert_called_once_with(query_embedding)
