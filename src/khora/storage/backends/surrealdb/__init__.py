"""SurrealDB unified backend for Khora.

Provides graph, vector, and relational storage in a single database.
Supports embedded mode (memory/file) and remote mode (WebSocket).

Install: pip install khora[surrealdb]
"""

from __future__ import annotations

from typing import Any

_AsyncSurreal: Any = None
_HAS_SURREALDB = False

try:
    from surrealdb import AsyncSurreal as _AsyncSurreal  # noqa: F401

    _HAS_SURREALDB = True
except ImportError:
    pass

__all__ = ["_HAS_SURREALDB"]
