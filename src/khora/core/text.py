"""Text sanitization helpers applied at the ingestion boundary."""

from __future__ import annotations

from typing import Any


def strip_nul(value: str) -> str:
    """Return ``value`` with NUL bytes (``U+0000``) removed.

    PostgreSQL text / varchar / jsonb columns cannot store ``0x00`` and abort
    the INSERT with ``CharacterNotInRepertoireError`` (#1528). NUL bytes are
    common in real corpora (PDFs, scraped HTML, OCR output) but never carry
    meaning in text, so stripping them is lossless in practice and keeps every
    backend (PostgreSQL, Neo4j, embedded) storing identical clean data.
    """
    return value.replace("\x00", "")


def strip_nul_json(value: Any) -> Any:
    """Recursively strip NUL bytes from strings inside JSON-shaped data.

    Walks dicts and lists, stripping ``0x00`` from every string key and value;
    non-string scalars pass through untouched. Used for the ``attributes`` /
    ``properties`` / ``metadata`` JSON written to ``jsonb`` columns.
    """
    if isinstance(value, str):
        return value.replace("\x00", "")
    if isinstance(value, dict):
        return {strip_nul_json(k): strip_nul_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [strip_nul_json(item) for item in value]
    return value
