"""SQLite relational adapter — DYT-2728."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .connection import EmbeddedStorageHandle


class SQLiteLanceRelationalAdapter:
    """Placeholder. Methods implemented in DYT-2728."""

    def __init__(self, handle: EmbeddedStorageHandle) -> None:
        self._handle = handle
