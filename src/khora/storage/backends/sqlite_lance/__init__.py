"""SQLite + LanceDB embedded unified backend."""

from __future__ import annotations

_HAS_LANCEDB = False
try:
    import lancedb  # noqa: F401

    _HAS_LANCEDB = True
except ImportError:
    pass

_HAS_AIOSQLITE = False
try:
    import aiosqlite  # noqa: F401

    _HAS_AIOSQLITE = True
except ImportError:
    pass

if _HAS_LANCEDB and _HAS_AIOSQLITE:
    from .connection import EmbeddedStorageHandle
    from .event_store import SQLiteLanceEventStoreAdapter
    from .graph import SQLiteLanceGraphAdapter
    from .relational import SQLiteLanceRelationalAdapter
    from .vector import SQLiteLanceVectorAdapter

    __all__ = [
        "EmbeddedStorageHandle",
        "SQLiteLanceEventStoreAdapter",
        "SQLiteLanceGraphAdapter",
        "SQLiteLanceRelationalAdapter",
        "SQLiteLanceVectorAdapter",
    ]
else:
    __all__ = []
