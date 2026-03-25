"""Data source handlers for ontology construction."""

from __future__ import annotations

from .base import DataSource, SampleChunk, SourceSummary
from .detection import detect_source
from .local import LocalDirectorySource, LocalFileSource

__all__ = [
    "DataSource",
    "LocalDirectorySource",
    "LocalFileSource",
    "SampleChunk",
    "SourceSummary",
    "detect_source",
]
