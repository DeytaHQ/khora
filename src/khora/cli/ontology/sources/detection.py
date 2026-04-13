"""Auto-detect data source type from user-provided strings."""

from __future__ import annotations

from pathlib import Path

from .base import DataSource
from .local import LocalDirectorySource, LocalFileSource


def detect_source(raw: str) -> DataSource:
    """Detect and create the appropriate DataSource from a path string.

    Raises:
        ValueError: If the source string cannot be resolved to a known source type.
    """
    path = Path(raw).expanduser()

    if path.is_file():
        return LocalFileSource(path)

    if path.is_dir():
        return LocalDirectorySource(path)

    # Might be a path that doesn't exist yet
    if path.suffix:
        raise FileNotFoundError(f"File not found: {path}")

    raise ValueError(f"Cannot determine source type for: {raw!r}. Provide a path to a local file or directory.")
