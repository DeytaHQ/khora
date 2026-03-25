"""Local file and directory data sources for ontology construction."""

from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
from typing import Any

from loguru import logger

from .base import SampleChunk, SourceSummary

# Supported text-based extensions
_TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".rst",
    ".csv",
    ".tsv",
    ".json",
    ".jsonl",
    ".ndjson",
    ".yaml",
    ".yml",
    ".xml",
    ".html",
    ".htm",
    ".log",
    ".cfg",
    ".ini",
    ".toml",
    ".py",
    ".js",
    ".ts",
    ".sql",
}


def _is_supported(path: Path) -> bool:
    return path.suffix.lower() in _TEXT_EXTENSIONS


def _read_text_safe(path: Path, limit: int = 0) -> str:
    """Read text from a file, handling encoding errors gracefully."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            if limit > 0:
                return f.read(limit)
            return f.read()
    except (OSError, ValueError):
        logger.debug(f"Could not read {path}")
        return ""


def _extract_text_from_content(path: Path, raw: str) -> str:
    """Extract readable text from file content based on extension."""
    ext = path.suffix.lower()

    if ext in (".json",):
        return _extract_json_text(raw)
    if ext in (".jsonl", ".ndjson"):
        return _extract_jsonl_text(raw)
    if ext in (".csv", ".tsv"):
        return _extract_csv_text(raw, delimiter="\t" if ext == ".tsv" else ",")
    # For everything else, return as-is (txt, md, yaml, xml, html, code, etc.)
    return raw


def _extract_json_text(raw: str) -> str:
    """Extract string values from JSON for sampling."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    return _flatten_json_values(data)


def _extract_jsonl_text(raw: str) -> str:
    """Extract string values from JSONL lines."""
    parts: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            parts.append(_flatten_json_values(obj))
        except json.JSONDecodeError:
            parts.append(line)
    return "\n".join(parts)


def _flatten_json_values(data: Any, max_depth: int = 5) -> str:
    """Recursively extract string values from a JSON structure."""
    if max_depth <= 0:
        return ""
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        parts = []
        for k, v in data.items():
            val = _flatten_json_values(v, max_depth - 1)
            if val:
                parts.append(f"{k}: {val}")
        return "\n".join(parts)
    if isinstance(data, list):
        items = [_flatten_json_values(item, max_depth - 1) for item in data[:50]]
        return "\n".join(i for i in items if i)
    if data is not None:
        return str(data)
    return ""


def _extract_csv_text(raw: str, delimiter: str = ",") -> str:
    """Convert CSV rows to key: value text for sampling."""
    reader = csv.DictReader(io.StringIO(raw), delimiter=delimiter)
    parts: list[str] = []
    for i, row in enumerate(reader):
        if i >= 50:  # Limit rows for sampling
            break
        line_parts = [f"{k}: {v}" for k, v in row.items() if v]
        if line_parts:
            parts.append(" | ".join(line_parts))
    return "\n".join(parts)


class LocalFileSource:
    """A single local file as a data source."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path).resolve()
        if not self._path.is_file():
            raise FileNotFoundError(f"Not a file: {self._path}")

    def scan(self) -> SourceSummary:
        return SourceSummary(
            source_id=str(self._path),
            source_type="file",
            path=self._path,
            file_count=1,
            total_bytes=self._path.stat().st_size,
            extensions=[self._path.suffix.lower()] if self._path.suffix else [],
        )

    def sample(self, budget_chars: int) -> list[SampleChunk]:
        size = self._path.stat().st_size
        if size == 0:
            return []

        if size <= budget_chars:
            # Small file: read all
            raw = _read_text_safe(self._path)
            content = _extract_text_from_content(self._path, raw)
            return [
                SampleChunk(
                    source_id=str(self._path),
                    content=content[:budget_chars],
                    byte_offset=0,
                    metadata={"file": self._path.name},
                )
            ]

        # Larger file: head + middle + tail
        third = budget_chars // 3
        chunks: list[SampleChunk] = []

        # Head
        raw = _read_text_safe(self._path, limit=third)
        content = _extract_text_from_content(self._path, raw)
        chunks.append(
            SampleChunk(
                source_id=str(self._path),
                content=content[:third],
                byte_offset=0,
                metadata={"file": self._path.name, "region": "head"},
            )
        )

        # Middle
        mid_offset = max(0, size // 2 - third // 2)
        try:
            with open(self._path, encoding="utf-8", errors="replace") as f:
                f.seek(mid_offset)
                # Skip partial line
                if mid_offset > 0:
                    f.readline()
                raw_mid = f.read(third)
        except (OSError, ValueError):
            raw_mid = ""
        if raw_mid:
            content_mid = _extract_text_from_content(self._path, raw_mid)
            chunks.append(
                SampleChunk(
                    source_id=str(self._path),
                    content=content_mid[:third],
                    byte_offset=mid_offset,
                    metadata={"file": self._path.name, "region": "middle"},
                )
            )

        # Tail
        tail_offset = max(0, size - third)
        try:
            with open(self._path, encoding="utf-8", errors="replace") as f:
                f.seek(tail_offset)
                if tail_offset > 0:
                    f.readline()
                raw_tail = f.read(third)
        except (OSError, ValueError):
            raw_tail = ""
        if raw_tail:
            content_tail = _extract_text_from_content(self._path, raw_tail)
            chunks.append(
                SampleChunk(
                    source_id=str(self._path),
                    content=content_tail[:third],
                    byte_offset=tail_offset,
                    metadata={"file": self._path.name, "region": "tail"},
                )
            )

        return chunks


class LocalDirectorySource:
    """A local directory as a data source (scans for supported files)."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path).resolve()
        if not self._path.is_dir():
            raise NotADirectoryError(f"Not a directory: {self._path}")
        self._files: list[Path] = []

    def scan(self) -> SourceSummary:
        self._files = sorted(f for f in self._path.rglob("*") if f.is_file() and _is_supported(f))

        total_bytes = sum(f.stat().st_size for f in self._files)
        extensions = sorted({f.suffix.lower() for f in self._files if f.suffix})

        return SourceSummary(
            source_id=str(self._path),
            source_type="directory",
            path=self._path,
            file_count=len(self._files),
            total_bytes=total_bytes,
            extensions=extensions,
        )

    def sample(self, budget_chars: int) -> list[SampleChunk]:
        if not self._files:
            # Scan if not already done
            self.scan()
        if not self._files:
            return []

        # Ensure extension diversity: at least one file per extension
        by_ext: dict[str, list[Path]] = {}
        for f in self._files:
            by_ext.setdefault(f.suffix.lower(), []).append(f)

        # Select files: one per extension first, then stride sampling
        selected: list[Path] = []
        for ext_files in by_ext.values():
            selected.append(ext_files[0])  # First file of each extension

        # Fill remaining with stride sampling
        remaining_budget = max(1, len(self._files) // 5)  # ~20% of files
        stride = max(1, len(self._files) // remaining_budget)
        for i in range(0, len(self._files), stride):
            if self._files[i] not in selected:
                selected.append(self._files[i])

        # Allocate budget per file (sqrt weighting)
        import math

        sizes = {f: max(1, f.stat().st_size) for f in selected}
        sqrt_total = sum(math.sqrt(s) for s in sizes.values())
        per_file_budget = {f: max(500, int(budget_chars * math.sqrt(s) / sqrt_total)) for f, s in sizes.items()}

        chunks: list[SampleChunk] = []
        total_chars = 0

        for f in selected:
            if total_chars >= budget_chars:
                break
            file_budget = min(per_file_budget[f], budget_chars - total_chars)

            raw = _read_text_safe(f, limit=file_budget)
            if not raw.strip():
                continue

            content = _extract_text_from_content(f, raw)[:file_budget]
            if not content.strip():
                continue

            rel = os.path.relpath(f, self._path)
            chunks.append(
                SampleChunk(
                    source_id=str(self._path),
                    content=content,
                    byte_offset=0,
                    metadata={"file": rel, "extension": f.suffix.lower()},
                )
            )
            total_chars += len(content)

        return chunks
