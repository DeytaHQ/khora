"""Pipeline orchestration for Khora.

Provides pipelines for document ingestion,
processing, and synchronization.
"""

from __future__ import annotations

from . import flows as flows  # noqa: F811 — triggers @pipeline() decorator registration
from .connector_metadata import (
    CANONICAL_TIMESTAMP_FIELDS,
    ConnectorMetadata,
    SourceSystem,
    extract_source_timestamp,
    validate_connector_metadata,
)
from .manager import PipelineManager
from .registry import PipelineRegistry, pipeline

__all__ = [
    "CANONICAL_TIMESTAMP_FIELDS",
    "ConnectorMetadata",
    "PipelineManager",
    "PipelineRegistry",
    "SourceSystem",
    "extract_source_timestamp",
    "pipeline",
    "validate_connector_metadata",
]
