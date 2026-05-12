"""Engine registry and factory for pluggable memory engines.

Memory engines implement different strategies for storing and retrieving memories.
The default engine is "vectorcypher" which uses knowledge graphs, vector embeddings,
and LLM-based entity extraction with selective (KET-RAG) skeleton indexing.

Usage:
    from khora.engines import create_engine, list_engines, register_engine

    # List available engines
    engines = list_engines()  # ["skeleton", "vectorcypher", "chronicle"]

    # Create an engine instance
    engine = create_engine("vectorcypher", config)

    # Register a custom engine
    register_engine("my_engine", "my_package.engine", "MyEngine")
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from khora.config import KhoraConfig
    from khora.storage import StorageConfig

    from .protocol import MemoryEngineProtocol

# Registry: name -> (module_path, class_name)
_ENGINE_REGISTRY: dict[str, tuple[str, str]] = {
    "skeleton": ("khora.engines.skeleton.engine", "SkeletonConstructionEngine"),
    "vectorcypher": ("khora.engines.vectorcypher.engine", "VectorCypherEngine"),
    "chronicle": ("khora.engines.chronicle.engine", "ChronicleEngine"),
}


def register_engine(name: str, module_path: str, class_name: str) -> None:
    """Register a custom engine implementation.

    Args:
        name: Engine name to register (used in Khora(engine=...))
        module_path: Full module path containing the engine class
        class_name: Name of the engine class

    Example:
        register_engine("my_engine", "my_package.engine", "MyEngine")
        async with Khora("postgresql://...", engine="my_engine") as kb:
            ...
    """
    _ENGINE_REGISTRY[name] = (module_path, class_name)


def list_engines() -> list[str]:
    """List available engine names.

    Returns:
        List of registered engine names
    """
    return list(_ENGINE_REGISTRY.keys())


def create_engine(
    name: str,
    config: KhoraConfig,
    *,
    storage_config: StorageConfig | None = None,
    **kwargs,
) -> MemoryEngineProtocol:
    """Create an engine instance by name.

    Args:
        name: Engine name (e.g., "vectorcypher")
        config: KhoraConfig instance
        storage_config: Optional StorageConfig (deprecated, for backwards compat)
        **kwargs: Additional arguments passed to the engine constructor

    Returns:
        MemoryEngineProtocol implementation

    Raises:
        ValueError: If the engine name is not registered
    """
    if name not in _ENGINE_REGISTRY:
        available = ", ".join(_ENGINE_REGISTRY.keys())
        raise ValueError(f"Unknown engine: {name}. Available: {available}")

    module_path, class_name = _ENGINE_REGISTRY[name]
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(config, storage_config=storage_config, **kwargs)


__all__ = [
    "create_engine",
    "list_engines",
    "register_engine",
]
