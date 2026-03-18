"""SurrealDB unified backend for Khora.

Provides graph, vector, and relational storage in a single database.
Supports embedded mode (memory/file) and remote mode (WebSocket).

Install: pip install khora[surrealdb]
"""

from __future__ import annotations

try:
    from surrealdb import AsyncSurreal as _AsyncSurreal

    _HAS_SURREALDB = True
except ImportError:
    _AsyncSurreal = None  # type: ignore[assignment,misc]
    _HAS_SURREALDB = False

__all__ = ["_HAS_SURREALDB"]
