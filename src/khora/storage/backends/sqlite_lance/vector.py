"""SQLite + LanceDB vector adapter — DYT-2730."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .connection import EmbeddedStorageHandle


class SQLiteLanceVectorAdapter:
    """Placeholder. Methods implemented in DYT-2730."""

    def __init__(self, handle: EmbeddedStorageHandle) -> None:
        self._handle = handle
