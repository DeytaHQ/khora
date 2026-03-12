"""Pipeline orchestration for Khora Memory Lake.

Provides pipelines for document ingestion,
processing, and synchronization.
"""

from __future__ import annotations

from . import flows as flows  # noqa: F811 — triggers @pipeline() decorator registration
from .manager import PipelineManager
from .registry import PipelineRegistry, pipeline

__all__ = [
    "PipelineManager",
    "PipelineRegistry",
    "pipeline",
]
