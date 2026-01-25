"""Pipeline orchestration for Khora Memory Lake.

Provides Prefect-based pipelines for document ingestion,
processing, and synchronization.
"""

from __future__ import annotations

from .manager import PipelineManager
from .registry import PipelineRegistry, pipeline

__all__ = [
    "PipelineManager",
    "PipelineRegistry",
    "pipeline",
]
