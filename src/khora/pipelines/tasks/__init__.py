"""Prefect tasks for Khora Memory Lake pipelines."""

from __future__ import annotations

from .chunk import chunk_document
from .embed import embed_chunks
from .extract import extract_entities

__all__ = [
    "chunk_document",
    "embed_chunks",
    "extract_entities",
]
