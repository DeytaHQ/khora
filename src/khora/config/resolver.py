"""Hierarchical configuration resolver for Khora Memory Lake.

Resolves configuration values with inheritance from namespace -> workspace -> organization -> global.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import UUID

from loguru import logger

if TYPE_CHECKING:
    from khora.core.models import MemoryNamespace, Organization, Workspace
    from khora.storage import StorageCoordinator


@dataclass
class ResolvedConfig:
    """Resolved configuration with inheritance applied."""

    values: dict[str, Any] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)  # key -> source level

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self.values.get(key, default)

    def get_source(self, key: str) -> str | None:
        """Get the source level for a configuration key."""
        return self.sources.get(key)


class ConfigResolver:
    """Resolves configuration with hierarchical inheritance.

    Configuration inheritance order (highest to lowest priority):
    1. Namespace config_overrides
    2. Workspace metadata config
    3. Organization metadata config
    4. Global application config
    """

    def __init__(
        self,
        storage: StorageCoordinator | None = None,
        global_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the config resolver.

        Args:
            storage: StorageCoordinator for fetching hierarchy
            global_config: Global application configuration
        """
        self._storage = storage
        self._global_config = global_config or {}

    async def resolve_for_namespace(
        self,
        namespace_id: UUID,
        *,
        keys: list[str] | None = None,
    ) -> ResolvedConfig:
        """Resolve configuration for a namespace.

        Args:
            namespace_id: Namespace to resolve config for
            keys: Optional list of keys to resolve (None = all)

        Returns:
            ResolvedConfig with inherited values
        """
        if self._storage is None:
            return ResolvedConfig(values=dict(self._global_config), sources={k: "global" for k in self._global_config})

        # Fetch namespace
        namespace = await self._storage.get_namespace(namespace_id)
        if namespace is None:
            logger.warning(f"Namespace {namespace_id} not found")
            return ResolvedConfig(values=dict(self._global_config), sources={k: "global" for k in self._global_config})

        # Fetch workspace
        workspace = await self._storage.get_workspace(namespace.workspace_id)

        # Fetch organization if workspace exists
        organization = None
        if workspace:
            organization = await self._storage.get_organization(workspace.organization_id)

        return self._merge_configs(
            namespace=namespace,
            workspace=workspace,
            organization=organization,
            keys=keys,
        )

    def _merge_configs(
        self,
        namespace: MemoryNamespace | None,
        workspace: Workspace | None,
        organization: Organization | None,
        keys: list[str] | None,
    ) -> ResolvedConfig:
        """Merge configurations from all levels."""
        result = ResolvedConfig()

        # Start with global config
        for key, value in self._global_config.items():
            if keys is None or key in keys:
                result.values[key] = value
                result.sources[key] = "global"

        # Apply organization config
        if organization and organization.metadata:
            config = organization.metadata.get("config", {})
            for key, value in config.items():
                if keys is None or key in keys:
                    result.values[key] = value
                    result.sources[key] = "organization"

        # Apply workspace config
        if workspace and workspace.metadata:
            config = workspace.metadata.get("config", {})
            for key, value in config.items():
                if keys is None or key in keys:
                    result.values[key] = value
                    result.sources[key] = "workspace"

        # Apply namespace config (highest priority)
        if namespace and namespace.config_overrides:
            for key, value in namespace.config_overrides.items():
                if keys is None or key in keys:
                    result.values[key] = value
                    result.sources[key] = "namespace"

        return result

    def resolve_immediate(
        self,
        global_config: dict[str, Any] | None = None,
        organization_config: dict[str, Any] | None = None,
        workspace_config: dict[str, Any] | None = None,
        namespace_config: dict[str, Any] | None = None,
    ) -> ResolvedConfig:
        """Resolve configuration from provided values (no storage lookup).

        Useful for testing or when hierarchy data is already loaded.

        Args:
            global_config: Global configuration
            organization_config: Organization-level config
            workspace_config: Workspace-level config
            namespace_config: Namespace-level config

        Returns:
            ResolvedConfig with merged values
        """
        result = ResolvedConfig()

        # Apply in order of priority (lowest to highest)
        for config, source in [
            (global_config or self._global_config, "global"),
            (organization_config or {}, "organization"),
            (workspace_config or {}, "workspace"),
            (namespace_config or {}, "namespace"),
        ]:
            for key, value in config.items():
                result.values[key] = value
                result.sources[key] = source

        return result

    def get_pipeline_config(self, resolved: ResolvedConfig) -> dict[str, Any]:
        """Extract pipeline-specific configuration.

        Args:
            resolved: ResolvedConfig instance

        Returns:
            Pipeline configuration dict
        """
        return {
            "chunking_strategy": resolved.get("chunking_strategy", "semantic"),
            "chunk_size": resolved.get("chunk_size", 512),
            "chunk_overlap": resolved.get("chunk_overlap", 50),
            "embedding_model": resolved.get("embedding_model", "text-embedding-3-small"),
            "extraction_model": resolved.get("extraction_model", "gpt-4o-mini"),
            "extraction_skill": resolved.get("extraction_skill", "general_entities"),
        }

    def get_llm_config(self, resolved: ResolvedConfig) -> dict[str, Any]:
        """Extract LLM-specific configuration.

        Args:
            resolved: ResolvedConfig instance

        Returns:
            LLM configuration dict
        """
        return {
            "model": resolved.get("llm_model", "gpt-4o-mini"),
            "temperature": resolved.get("llm_temperature", 0.7),
            "max_tokens": resolved.get("llm_max_tokens", 2000),
            "timeout": resolved.get("llm_timeout", 30),
        }
