"""SQLite event store adapter — DYT-2731."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .connection import EmbeddedStorageHandle


class SQLiteLanceEventStoreAdapter:
    """Placeholder. Methods implemented in DYT-2731."""

    def __init__(self, handle: EmbeddedStorageHandle) -> None:
        self._handle = handle
