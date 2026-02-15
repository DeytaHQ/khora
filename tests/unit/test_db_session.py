"""Unit tests for DatabaseManager in db/session.py."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from khora.db.session import DatabaseManager, get_default_manager, get_engine, get_session_factory


class TestDatabaseManager:
    """Tests for DatabaseManager class."""

    def test_initial_state(self):
        """New manager has no engine or factory."""
        mgr = DatabaseManager()
        assert mgr._engine is None
        assert mgr._session_factory is None

    @patch("khora.db.session.get_database_url", return_value="postgresql+asyncpg://test")
    @patch("khora.db.session.create_async_engine")
    def test_get_engine_creates_once(self, mock_create, mock_url):
        """get_engine creates engine on first call, returns cached on second."""
        mgr = DatabaseManager()
        engine1 = mgr.get_engine()
        engine2 = mgr.get_engine()
        assert engine1 is engine2
        mock_create.assert_called_once()

    @patch("khora.db.session.get_database_url", return_value="postgresql+asyncpg://test")
    @patch("khora.db.session.create_async_engine")
    def test_get_session_factory_creates_once(self, mock_create, mock_url):
        """get_session_factory creates factory on first call."""
        mgr = DatabaseManager()
        factory1 = mgr.get_session_factory()
        factory2 = mgr.get_session_factory()
        assert factory1 is factory2

    def test_reset_clears_state(self):
        """reset() clears engine and factory without async disposal."""
        mgr = DatabaseManager()
        mgr._engine = MagicMock()
        mgr._session_factory = MagicMock()
        mgr.reset()
        assert mgr._engine is None
        assert mgr._session_factory is None

    @pytest.mark.asyncio
    async def test_close_db_disposes_engine(self):
        """close_db disposes engine and clears state."""
        mgr = DatabaseManager()
        mock_engine = AsyncMock()
        mgr._engine = mock_engine
        mgr._session_factory = MagicMock()

        await mgr.close_db()

        mock_engine.dispose.assert_awaited_once()
        assert mgr._engine is None
        assert mgr._session_factory is None

    @pytest.mark.asyncio
    async def test_close_db_noop_when_no_engine(self):
        """close_db is a no-op when no engine exists."""
        mgr = DatabaseManager()
        await mgr.close_db()  # Should not raise


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @patch("khora.db.session._default_manager", None)
    def test_get_default_manager_creates_singleton(self):
        """get_default_manager creates manager on first call."""
        mgr1 = get_default_manager()
        mgr2 = get_default_manager()
        assert mgr1 is mgr2
        assert isinstance(mgr1, DatabaseManager)

    @patch("khora.db.session.get_default_manager")
    def test_get_engine_delegates(self, mock_get_mgr):
        """Module-level get_engine delegates to default manager."""
        mock_mgr = MagicMock()
        mock_get_mgr.return_value = mock_mgr
        get_engine()
        mock_mgr.get_engine.assert_called_once()

    @patch("khora.db.session.get_default_manager")
    def test_get_session_factory_delegates(self, mock_get_mgr):
        """Module-level get_session_factory delegates to default manager."""
        mock_mgr = MagicMock()
        mock_get_mgr.return_value = mock_mgr
        get_session_factory()
        mock_mgr.get_session_factory.assert_called_once()
