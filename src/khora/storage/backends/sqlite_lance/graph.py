"""SQLite graph adapter — DYT-2729."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .connection import EmbeddedStorageHandle


class SQLiteLanceGraphAdapter:
    """Placeholder. Methods implemented in DYT-2729."""

    def __init__(self, handle: EmbeddedStorageHandle) -> None:
        self._handle = handle
