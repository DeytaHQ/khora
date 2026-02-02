"""Pipeline manager for orchestrating pipeline flows."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from loguru import logger

if TYPE_CHECKING:
    from khora.config import KhoraConfig
    from khora.storage import StorageCoordinator


@dataclass
class PipelineRun:
    """Information about a pipeline run."""

    run_id: str
    pipeline_name: str
    namespace_id: UUID
    status: str  # pending, running, completed, failed
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class PipelineManager:
    """Manager for running and tracking pipelines.

    Orchestrates pipeline flows and tracks their execution
    within the context of a memory namespace.
    """

    def __init__(
        self,
        storage: StorageCoordinator | None = None,
        config: KhoraConfig | None = None,
    ) -> None:
        """Initialize the pipeline manager.

        Args:
            storage: StorageCoordinator for data access
            config: Khora configuration
        """
        self._storage = storage
        self._config = config
        self._runs: dict[str, PipelineRun] = {}

    async def run_pipeline(
        self,
        pipeline_name: str,
        namespace_id: UUID,
        *,
        parameters: dict[str, Any] | None = None,
        wait: bool = True,
    ) -> PipelineRun:
        """Run a registered pipeline.

        Args:
            pipeline_name: Name of the pipeline to run
            namespace_id: Namespace to run in
            parameters: Pipeline parameters
            wait: Wait for completion

        Returns:
            PipelineRun with execution info
        """
        from .registry import get_registry

        registry = get_registry()
        pipeline_info = registry.get(pipeline_name)

        if not pipeline_info:
            raise ValueError(f"Unknown pipeline: {pipeline_name}")

        # Create run record
        import uuid

        run_id = str(uuid.uuid4())
        run = PipelineRun(
            run_id=run_id,
            pipeline_name=pipeline_name,
            namespace_id=namespace_id,
            status="running",
            started_at=datetime.now(),
        )
        self._runs[run_id] = run

        logger.info(f"Starting pipeline {pipeline_name} (run_id={run_id})")

        try:
            # Prepare parameters
            params = parameters or {}
            params["namespace_id"] = namespace_id
            params["storage"] = self._storage
            params["config"] = self._config

            # Run the pipeline
            func = pipeline_info.func
            result = await func(**params)

            run.status = "completed"
            run.completed_at = datetime.now()
            run.metadata["result"] = result

            logger.info(f"Pipeline {pipeline_name} completed (run_id={run_id})")

        except Exception as e:
            run.status = "failed"
            run.error = str(e)
            run.completed_at = datetime.now()
            logger.error(f"Pipeline {pipeline_name} failed (run_id={run_id}): {e}")
            raise

        return run

    async def run_ingestion(
        self,
        namespace_id: UUID,
        documents: list[dict[str, Any]],
        *,
        skill_name: str = "general_entities",
    ) -> PipelineRun:
        """Run the standard ingestion pipeline.

        Args:
            namespace_id: Namespace to ingest into
            documents: List of documents to ingest
            skill_name: Extraction skill to use

        Returns:
            PipelineRun with execution info
        """
        return await self.run_pipeline(
            "ingest",
            namespace_id,
            parameters={
                "documents": documents,
                "skill_name": skill_name,
            },
        )

    async def run_sync(
        self,
        namespace_id: UUID,
        source: str,
        *,
        connector_config: dict[str, Any] | None = None,
    ) -> PipelineRun:
        """Run a sync pipeline for an external source.

        Args:
            namespace_id: Namespace to sync into
            source: Source name (e.g., "github", "notion")
            connector_config: Connector configuration

        Returns:
            PipelineRun with execution info
        """
        return await self.run_pipeline(
            f"sync_{source}",
            namespace_id,
            parameters={
                "source": source,
                "connector_config": connector_config or {},
            },
        )

    def get_run(self, run_id: str) -> PipelineRun | None:
        """Get a pipeline run by ID.

        Args:
            run_id: Run ID

        Returns:
            PipelineRun or None if not found
        """
        return self._runs.get(run_id)

    def list_runs(
        self,
        *,
        namespace_id: UUID | None = None,
        pipeline_name: str | None = None,
        status: str | None = None,
    ) -> list[PipelineRun]:
        """List pipeline runs with optional filtering.

        Args:
            namespace_id: Filter by namespace
            pipeline_name: Filter by pipeline name
            status: Filter by status

        Returns:
            List of matching PipelineRun objects
        """
        runs = list(self._runs.values())

        if namespace_id:
            runs = [r for r in runs if r.namespace_id == namespace_id]
        if pipeline_name:
            runs = [r for r in runs if r.pipeline_name == pipeline_name]
        if status:
            runs = [r for r in runs if r.status == status]

        return runs
