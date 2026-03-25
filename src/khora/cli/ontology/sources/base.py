"""Base protocol and data classes for ontology data sources."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(slots=True)
class SampleChunk:
    """A chunk of sampled text from a data source."""

    source_id: str
    content: str
    byte_offset: int = 0
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.content)


@dataclass(slots=True)
class SourceSummary:
    """Summary of a scanned data source."""

    source_id: str
    source_type: str  # "file", "directory"
    path: Path
    file_count: int = 1
    total_bytes: int = 0
    extensions: list[str] = field(default_factory=list)

    @property
    def size_human(self) -> str:
        """Human-readable size string."""
        b = self.total_bytes
        for unit in ("B", "KB", "MB", "GB"):
            if b < 1024:
                return f"{b:.1f} {unit}" if unit != "B" else f"{b} {unit}"
            b /= 1024
        return f"{b:.1f} TB"


@runtime_checkable
class DataSource(Protocol):
    """Protocol for ontology data sources."""

    def scan(self) -> SourceSummary:
        """Scan the source and return a summary."""
        ...

    def sample(self, budget_chars: int) -> list[SampleChunk]:
        """Sample text content up to budget_chars total characters."""
        ...
