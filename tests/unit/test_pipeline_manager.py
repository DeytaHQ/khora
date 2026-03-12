"""Unit tests for PipelineManager error context."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from khora.pipelines.manager import PipelineManager


class TestPipelineManagerErrorContext:
    """Tests for improved error context in PipelineManager."""

    @pytest.mark.asyncio
    async def test_error_includes_exception_type(self) -> None:
        """Pipeline run error includes exception class name."""
        manager = PipelineManager()
        ns_id = uuid4()

        # Mock registry to return a pipeline that raises ValueError
        mock_info = MagicMock()
        mock_info.func = AsyncMock(side_effect=ValueError("bad input"))

        with patch("khora.pipelines.registry.get_registry") as mock_registry:
            mock_registry.return_value.get.return_value = mock_info

            with pytest.raises(ValueError):
                await manager.run_pipeline("test_pipeline", ns_id)

        # Check the run was recorded with error context
        runs = manager.list_runs()
        assert len(runs) == 1
        assert runs[0].status == "failed"
        assert runs[0].error == "ValueError: bad input"

    @pytest.mark.asyncio
    async def test_error_format_with_runtime_error(self) -> None:
        """Error format works for different exception types."""
        manager = PipelineManager()
        ns_id = uuid4()

        mock_info = MagicMock()
        mock_info.func = AsyncMock(side_effect=RuntimeError("connection lost"))

        with patch("khora.pipelines.registry.get_registry") as mock_registry:
            mock_registry.return_value.get.return_value = mock_info

            with pytest.raises(RuntimeError):
                await manager.run_pipeline("test_pipeline", ns_id)

        runs = manager.list_runs()
        assert runs[0].error == "RuntimeError: connection lost"


class TestPipelineRegistryAutoPopulation:
    """Importing khora.pipelines must trigger @pipeline() decorator registration."""

    def test_registry_contains_ingest_after_import(self) -> None:
        """The 'ingest' pipeline is registered when khora.pipelines is imported."""
        from khora.pipelines.registry import get_registry

        registry = get_registry()
        assert registry.get("ingest") is not None

    def test_registry_contains_all_builtin_pipelines(self) -> None:
        """All builtin pipelines are registered after import."""
        from khora.pipelines.registry import get_registry

        registry = get_registry()
        for name in ("ingest", "sync_source", "sync_all", "expand_knowledge", "unify_entities"):
            assert registry.get(name) is not None, f"Pipeline '{name}' not registered"
