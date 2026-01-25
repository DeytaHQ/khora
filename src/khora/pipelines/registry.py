"""Pipeline registry with decorators for Khora Memory Lake."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class PipelineInfo:
    """Information about a registered pipeline."""

    name: str
    description: str
    func: Callable
    tags: list[str] = field(default_factory=list)
    version: str = "1.0.0"
    metadata: dict[str, Any] = field(default_factory=dict)


class PipelineRegistry:
    """Registry for managing pipeline flows.

    Provides registration and discovery of pipeline flows
    that can be run by the PipelineManager.
    """

    _instance: PipelineRegistry | None = None
    _pipelines: dict[str, PipelineInfo]

    def __new__(cls) -> PipelineRegistry:
        """Singleton pattern for global registry."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._pipelines = {}
        return cls._instance

    def register(
        self,
        name: str,
        *,
        description: str = "",
        tags: list[str] | None = None,
        version: str = "1.0.0",
        **metadata: Any,
    ) -> Callable[[Callable], Callable]:
        """Decorator to register a pipeline flow.

        Args:
            name: Pipeline name
            description: Pipeline description
            tags: Pipeline tags for categorization
            version: Pipeline version
            **metadata: Additional metadata

        Returns:
            Decorator function
        """

        def decorator(func: Callable) -> Callable:
            info = PipelineInfo(
                name=name,
                description=description or func.__doc__ or "",
                func=func,
                tags=tags or [],
                version=version,
                metadata=metadata,
            )
            self._pipelines[name] = info
            logger.debug(f"Registered pipeline: {name}")
            return func

        return decorator

    def get(self, name: str) -> PipelineInfo | None:
        """Get a pipeline by name.

        Args:
            name: Pipeline name

        Returns:
            PipelineInfo or None if not found
        """
        return self._pipelines.get(name)

    def list_pipelines(self) -> list[str]:
        """List all registered pipeline names.

        Returns:
            List of pipeline names
        """
        return list(self._pipelines.keys())

    def get_by_tag(self, tag: str) -> list[PipelineInfo]:
        """Get pipelines by tag.

        Args:
            tag: Tag to filter by

        Returns:
            List of matching PipelineInfo objects
        """
        return [p for p in self._pipelines.values() if tag in p.tags]

    def all_pipelines(self) -> list[PipelineInfo]:
        """Get all registered pipelines.

        Returns:
            List of PipelineInfo objects
        """
        return list(self._pipelines.values())

    def unregister(self, name: str) -> bool:
        """Unregister a pipeline.

        Args:
            name: Pipeline name to unregister

        Returns:
            True if removed, False if not found
        """
        if name in self._pipelines:
            del self._pipelines[name]
            logger.debug(f"Unregistered pipeline: {name}")
            return True
        return False


# Global registry instance
_registry = PipelineRegistry()


def pipeline(
    name: str,
    *,
    description: str = "",
    tags: list[str] | None = None,
    version: str = "1.0.0",
    **metadata: Any,
) -> Callable[[Callable], Callable]:
    """Decorator to register a pipeline flow.

    Usage:
        @pipeline("my_pipeline", description="My pipeline", tags=["ingestion"])
        @flow
        async def my_pipeline(param: str):
            ...

    Args:
        name: Pipeline name
        description: Pipeline description
        tags: Pipeline tags
        version: Pipeline version
        **metadata: Additional metadata

    Returns:
        Decorator function
    """
    return _registry.register(name, description=description, tags=tags, version=version, **metadata)


def get_registry() -> PipelineRegistry:
    """Get the global pipeline registry.

    Returns:
        PipelineRegistry instance
    """
    return _registry
