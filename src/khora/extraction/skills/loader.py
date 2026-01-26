"""Expertise configuration loader.

Loads expertise configurations from various sources:
- YAML/JSON files
- Built-in configurations
- Database storage
- Programmatic configuration
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

from loguru import logger

from .base import ExpertiseConfig

if TYPE_CHECKING:
    from khora.storage import StorageCoordinator


class ExpertiseLoadError(Exception):
    """Error loading expertise configuration."""

    pass


class ExpertiseLoader:
    """Load expertise configurations from various sources.

    Supports loading from:
    - File paths (YAML/JSON)
    - Built-in configurations (bundled with library)
    - Database storage (per-namespace)
    - Merged configurations from multiple sources

    Example usage:
        loader = ExpertiseLoader()

        # Load from file
        expertise = loader.load_file("./config/saas_expert.yaml")

        # Load built-in
        expertise = loader.load_builtin("general")

        # Load and merge multiple sources
        expertise = loader.load_merged([
            "builtin:general",
            "file:./config/custom.yaml",
        ])

        # Load from database
        expertise = await loader.load_from_db(namespace_id, storage)
    """

    def __init__(self, search_paths: list[Path] | None = None) -> None:
        """Initialize the expertise loader.

        Args:
            search_paths: Additional paths to search for expertise files
        """
        self._search_paths = search_paths or []
        self._cache: dict[str, ExpertiseConfig] = {}

    def load_file(self, path: str | Path, *, use_cache: bool = True) -> ExpertiseConfig:
        """Load expertise configuration from a YAML or JSON file.

        Args:
            path: Path to the configuration file
            use_cache: Whether to use cached configurations

        Returns:
            ExpertiseConfig loaded from the file

        Raises:
            ExpertiseLoadError: If file cannot be loaded or parsed
        """
        path = Path(path).expanduser().resolve()
        cache_key = f"file:{path}"

        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        if not path.exists():
            raise ExpertiseLoadError(f"Expertise file not found: {path}")

        try:
            data = self._load_file_data(path)
            config = ExpertiseConfig.from_dict(data)

            if use_cache:
                self._cache[cache_key] = config

            logger.debug(f"Loaded expertise config from file: {path}")
            return config

        except Exception as e:
            raise ExpertiseLoadError(f"Failed to load expertise from {path}: {e}") from e

    def load_builtin(self, name: str, *, use_cache: bool = True) -> ExpertiseConfig:
        """Load a built-in expertise configuration.

        Built-in configurations are bundled with the library and provide
        common expertise definitions.

        Args:
            name: Name of the built-in expertise (e.g., "general", "saas_expert")
            use_cache: Whether to use cached configurations

        Returns:
            ExpertiseConfig for the built-in expertise

        Raises:
            ExpertiseLoadError: If built-in not found
        """
        cache_key = f"builtin:{name}"

        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        # Look for built-in in the builtin directory
        builtin_dir = Path(__file__).parent / "builtin"
        if not builtin_dir.exists():
            builtin_dir.mkdir(parents=True, exist_ok=True)

        # Try various extensions
        for ext in [".yaml", ".yml", ".json"]:
            builtin_path = builtin_dir / f"{name}{ext}"
            if builtin_path.exists():
                config = self.load_file(builtin_path, use_cache=False)
                if use_cache:
                    self._cache[cache_key] = config
                return config

        # If no file found, try to generate a basic config for known names
        config = self._generate_builtin(name)
        if config:
            if use_cache:
                self._cache[cache_key] = config
            return config

        raise ExpertiseLoadError(f"Built-in expertise not found: {name}")

    def load_source(self, source: str, *, use_cache: bool = True) -> ExpertiseConfig:
        """Load expertise from a source specification.

        Source formats:
        - "builtin:<name>" - Load built-in expertise
        - "file:<path>" - Load from file path
        - "<path>" - Load from file (assumed if no prefix)

        Args:
            source: Source specification string
            use_cache: Whether to use cached configurations

        Returns:
            ExpertiseConfig from the specified source
        """
        if source.startswith("builtin:"):
            name = source[8:]
            return self.load_builtin(name, use_cache=use_cache)

        if source.startswith("file:"):
            path = source[5:]
            return self.load_file(path, use_cache=use_cache)

        # Assume it's a file path
        return self.load_file(source, use_cache=use_cache)

    def load_merged(self, sources: list[str], *, use_cache: bool = True) -> ExpertiseConfig:
        """Load and merge multiple expertise configurations.

        Configurations are merged in order, with later sources taking
        precedence. The 'extends' field is resolved during merging.

        Args:
            sources: List of source specifications to merge
            use_cache: Whether to use cached configurations

        Returns:
            Merged ExpertiseConfig
        """
        from .composer import ExpertiseComposer

        configs = []
        for source in sources:
            try:
                config = self.load_source(source, use_cache=use_cache)
                configs.append(config)
            except ExpertiseLoadError as e:
                logger.warning(f"Failed to load expertise source {source}: {e}")
                continue

        if not configs:
            raise ExpertiseLoadError(f"No expertise configurations loaded from sources: {sources}")

        composer = ExpertiseComposer(self)
        return composer.merge(configs)

    async def load_from_db(
        self,
        namespace_id: UUID,
        storage: StorageCoordinator,
    ) -> ExpertiseConfig | None:
        """Load expertise configuration from database.

        Args:
            namespace_id: Namespace to load expertise for
            storage: Storage coordinator for database access

        Returns:
            ExpertiseConfig if found, None otherwise
        """
        try:
            # Import here to avoid circular imports
            from khora.storage.expertise_store import ExpertiseStore

            store = ExpertiseStore(storage)
            return await store.get_by_namespace(namespace_id)

        except ImportError:
            logger.debug("ExpertiseStore not available, skipping database load")
            return None
        except Exception as e:
            logger.warning(f"Failed to load expertise from database: {e}")
            return None

    def resolve_extends(self, config: ExpertiseConfig) -> ExpertiseConfig:
        """Resolve 'extends' references in a configuration.

        Loads all parent configurations and merges them with the current config.

        Args:
            config: Configuration with potential 'extends' references

        Returns:
            Resolved ExpertiseConfig with all parents merged
        """
        if not config.extends:
            return config

        from .composer import ExpertiseComposer

        # Load all parent configs
        parents = []
        for parent_source in config.extends:
            try:
                parent = self.load_source(parent_source)
                # Recursively resolve parent's extends
                parent = self.resolve_extends(parent)
                parents.append(parent)
            except ExpertiseLoadError as e:
                logger.warning(f"Failed to load parent expertise {parent_source}: {e}")

        if not parents:
            return config

        # Merge parents first, then apply current config on top
        composer = ExpertiseComposer(self)
        base = composer.merge(parents) if len(parents) > 1 else parents[0]

        # Merge current config on top of base
        return composer.merge([base, config])

    def clear_cache(self) -> None:
        """Clear the expertise configuration cache."""
        self._cache.clear()

    def _load_file_data(self, path: Path) -> dict[str, Any]:
        """Load data from a file (YAML or JSON)."""
        content = path.read_text(encoding="utf-8")

        if path.suffix in (".yaml", ".yml"):
            try:
                import yaml

                return yaml.safe_load(content) or {}
            except ImportError:
                raise ExpertiseLoadError("PyYAML not installed. Run: pip install pyyaml")

        if path.suffix == ".json":
            import json

            return json.loads(content)

        # Try YAML first, then JSON
        try:
            import yaml

            return yaml.safe_load(content) or {}
        except Exception:
            import json

            return json.loads(content)

    def _generate_builtin(self, name: str) -> ExpertiseConfig | None:
        """Generate a basic built-in configuration for known names."""
        builtins: dict[str, dict[str, Any]] = {
            "general": {
                "name": "general",
                "description": "General entity extraction",
                "entity_types": [
                    {"name": "PERSON", "description": "A person or individual"},
                    {"name": "ORGANIZATION", "description": "A company, institution, or group"},
                    {"name": "LOCATION", "description": "A place or geographic location"},
                    {"name": "CONCEPT", "description": "An idea, topic, or abstract concept"},
                    {"name": "EVENT", "description": "An occurrence or happening"},
                    {"name": "TECHNOLOGY", "description": "A technology, tool, or system"},
                ],
                "relationship_types": [
                    {"name": "WORKS_FOR", "source_types": ["PERSON"], "target_types": ["ORGANIZATION"]},
                    {"name": "KNOWS", "source_types": ["PERSON"], "target_types": ["PERSON"]},
                    {"name": "LOCATED_IN", "source_types": ["*"], "target_types": ["LOCATION"]},
                    {"name": "RELATES_TO", "source_types": ["*"], "target_types": ["*"]},
                    {"name": "PART_OF", "source_types": ["*"], "target_types": ["*"]},
                ],
            },
            "general_entities": {
                "name": "general_entities",
                "description": "General entity extraction (alias for general)",
                "extends": ["builtin:general"],
            },
        }

        if name in builtins:
            return ExpertiseConfig.from_dict(builtins[name])

        return None


# Module-level loader instance
_default_loader: ExpertiseLoader | None = None


def get_default_loader() -> ExpertiseLoader:
    """Get the default expertise loader instance."""
    global _default_loader
    if _default_loader is None:
        _default_loader = ExpertiseLoader()
    return _default_loader


def load_expertise(source: str) -> ExpertiseConfig:
    """Convenience function to load expertise from a source.

    Args:
        source: Source specification (file path, "builtin:name", etc.)

    Returns:
        ExpertiseConfig from the source
    """
    return get_default_loader().load_source(source)
