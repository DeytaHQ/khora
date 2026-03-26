"""Data validation pipeline for fetched discovery sources.

Validates fetched data before ingestion: format detection, content
quality metrics, relevance estimation, and deduplication.  Each check
produces a score in [0, 1] and an optional reason string.  The composite
quality score determines whether data is auto-accepted, flagged for
review, or rejected.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Quality thresholds
# ---------------------------------------------------------------------------

QUALITY_ACCEPT: float = 0.7
QUALITY_REVIEW: float = 0.4


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ValidationResult:
    """Result of validating a single fetched file."""

    path: str
    format_score: float = 0.0
    format_detected: str = "unknown"
    relevance_score: float = 0.0
    quality_score: float = 0.0
    completeness_score: float = 0.0
    duplicate: bool = False
    composite_score: float = 0.0
    decision: str = "review"  # "accept", "review", "reject"
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "format_score": self.format_score,
            "format_detected": self.format_detected,
            "relevance_score": self.relevance_score,
            "quality_score": self.quality_score,
            "completeness_score": self.completeness_score,
            "duplicate": self.duplicate,
            "composite_score": self.composite_score,
            "decision": self.decision,
            "reasons": self.reasons,
        }


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

#: File extensions and their expected format types
_EXT_FORMAT_MAP: dict[str, str] = {
    ".json": "json",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".csv": "csv",
    ".tsv": "tsv",
    ".xml": "xml",
    ".html": "html",
    ".htm": "html",
    ".md": "markdown",
    ".txt": "text",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".parquet": "parquet",
    ".pdf": "pdf",
}


def detect_format(path: Path) -> tuple[str, float]:
    """Detect file format and return (format_name, confidence).

    Uses file extension first, then content sniffing for ambiguous cases.
    """
    if path.stat().st_size == 0:
        return "empty", 0.0

    ext = path.suffix.lower()

    # Extension-based detection
    if ext in _EXT_FORMAT_MAP:
        fmt = _EXT_FORMAT_MAP[ext]
        # Verify with content sniffing for structured formats
        if fmt in ("json", "csv", "jsonl") and path.stat().st_size > 0:
            try:
                content = path.read_text(encoding="utf-8", errors="replace")[:4096]
                if fmt == "json" and _is_valid_json(content):
                    return "json", 1.0
                if fmt == "jsonl" and _is_valid_jsonl(content):
                    return "jsonl", 1.0
                if fmt == "csv" and _is_valid_csv(content):
                    return "csv", 1.0
            except Exception:
                pass
            return fmt, 0.7  # extension matches but content didn't validate
        return fmt, 0.9

    # Content sniffing for files without recognized extension
    if path.stat().st_size == 0:
        return "empty", 0.0

    try:
        content = path.read_text(encoding="utf-8", errors="replace")[:4096]
    except Exception:
        return "binary", 0.3

    if _is_valid_json(content):
        return "json", 0.8
    if _is_valid_jsonl(content):
        return "jsonl", 0.8
    if _is_valid_csv(content):
        return "csv", 0.7
    if content.strip().startswith(("<!DOCTYPE", "<html", "<?xml")):
        return "html" if "<html" in content.lower() else "xml", 0.8
    if content.startswith("#") or "\n##" in content:
        return "markdown", 0.6

    return "text", 0.5


def _is_valid_json(content: str) -> bool:
    """Check if content looks like valid JSON."""
    stripped = content.strip()
    if not (stripped.startswith(("{", "[")) and stripped.endswith(("}", "]"))):
        return False
    try:
        json.loads(stripped)
        return True
    except json.JSONDecodeError:
        # Might be truncated — check if start parses
        return False


def _is_valid_jsonl(content: str) -> bool:
    """Check if content looks like JSON Lines."""
    lines = [line for line in content.strip().splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    valid = 0
    for line in lines[:10]:
        try:
            json.loads(line)
            valid += 1
        except json.JSONDecodeError:
            pass
    return valid >= len(lines[:10]) * 0.8


def _is_valid_csv(content: str) -> bool:
    """Check if content looks like valid CSV."""
    try:
        dialect = csv.Sniffer().sniff(content[:2048])
        reader = csv.reader(io.StringIO(content[:2048]), dialect)
        rows = list(reader)
        if len(rows) < 2:
            return False
        # Check column consistency
        widths = [len(row) for row in rows[:10]]
        return len(set(widths)) <= 2 and widths[0] >= 2
    except (csv.Error, Exception):
        return False


# ---------------------------------------------------------------------------
# Text quality metrics
# ---------------------------------------------------------------------------


def compute_text_quality(content: str) -> tuple[float, list[str]]:
    """Compute text quality score from statistical metrics.

    Returns (score in [0, 1], list of reasons for low quality).
    """
    reasons: list[str] = []

    if not content or not content.strip():
        return 0.0, ["empty content"]

    words = content.split()
    word_count = len(words)

    if word_count < 10:
        return 0.1, ["too few words"]

    scores: list[float] = []

    # 1. Shannon entropy (information density)
    entropy = _shannon_entropy(words)
    # Good English text: 7-12 bits/word
    if entropy < 3.0:
        reasons.append(f"low entropy ({entropy:.1f} — repetitive content)")
        scores.append(0.2)
    elif entropy > 14.0:
        reasons.append(f"high entropy ({entropy:.1f} — possibly noisy/encoded)")
        scores.append(0.4)
    else:
        scores.append(min(1.0, entropy / 10.0))

    # 2. Type-Token Ratio (lexical diversity)
    # Use corrected TTR to account for document length
    unique_words = len({w.lower() for w in words})
    cttr = unique_words / math.sqrt(2 * word_count) if word_count > 0 else 0
    if cttr < 2.0:
        reasons.append(f"low lexical diversity (CTTR={cttr:.1f})")
        scores.append(0.3)
    else:
        scores.append(min(1.0, cttr / 8.0))

    # 3. Repetition detection
    # Check for repeated paragraphs (scraping artifact)
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    if len(paragraphs) > 3:
        para_counts = Counter(paragraphs)
        max_repeat = max(para_counts.values())
        if max_repeat > 3:
            reasons.append(f"repeated paragraph ({max_repeat}x — possible scraping artifact)")
            scores.append(0.2)
        else:
            scores.append(1.0)
    else:
        scores.append(0.8)

    # 4. Boilerplate/placeholder detection
    lower = content.lower()
    boilerplate_patterns = [
        "lorem ipsum",
        "todo",
        "example data",
        "placeholder",
        "coming soon",
        "under construction",
        "cookie",
        "privacy policy",
        "terms of service",
    ]
    boilerplate_hits = sum(1 for p in boilerplate_patterns if p in lower)
    if boilerplate_hits >= 3:
        reasons.append(f"boilerplate content detected ({boilerplate_hits} patterns)")
        scores.append(0.2)
    elif boilerplate_hits >= 1:
        scores.append(0.7)
    else:
        scores.append(1.0)

    # 5. Encoding quality
    replacement_chars = content.count("\ufffd")
    if replacement_chars > len(content) * 0.01:
        reasons.append("encoding issues (>1% replacement characters)")
        scores.append(0.3)
    else:
        scores.append(1.0)

    return sum(scores) / len(scores), reasons


def _shannon_entropy(words: list[str]) -> float:
    """Compute Shannon entropy over word frequencies."""
    if not words:
        return 0.0
    counts = Counter(w.lower() for w in words)
    total = len(words)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


# ---------------------------------------------------------------------------
# Relevance estimation (keyword-based, no embedding model required)
# ---------------------------------------------------------------------------


def estimate_relevance(content: str, query: str) -> float:
    """Estimate content relevance to the user's query.

    Uses keyword overlap (TF-based) — lightweight, no embedding model needed.
    For production use, swap with embedding similarity.
    """
    if not content or not query:
        return 0.0

    # Tokenize
    query_words = set(re.findall(r"\w{3,}", query.lower()))
    content_words = re.findall(r"\w{3,}", content.lower())

    if not query_words or not content_words:
        return 0.0

    # Count query term occurrences in content
    content_counter = Counter(content_words)
    total_content_words = len(content_words)

    hits = 0
    for qw in query_words:
        if content_counter[qw] > 0:
            hits += 1

    # Coverage: what fraction of query terms appear in content
    coverage = hits / len(query_words)

    # Density: how frequently query terms appear
    query_term_count = sum(content_counter[qw] for qw in query_words)
    density = min(1.0, query_term_count / max(1, total_content_words) * 20)

    # Weighted combination
    return 0.6 * coverage + 0.4 * density


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class DeduplicationIndex:
    """Simple hash-based deduplication index.

    Uses SHA-256 for exact dedup and a set of content fingerprints.
    """

    def __init__(self) -> None:
        self._hashes: set[str] = set()

    def is_duplicate(self, content: str) -> bool:
        """Check if content has been seen before."""
        h = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()
        if h in self._hashes:
            return True
        self._hashes.add(h)
        return False

    def add_existing(self, content: str) -> None:
        """Add content to the index without checking."""
        h = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()
        self._hashes.add(h)

    @property
    def size(self) -> int:
        return len(self._hashes)


# ---------------------------------------------------------------------------
# Validation pipeline
# ---------------------------------------------------------------------------


def validate_file(
    path: Path,
    *,
    query: str = "",
    dedup_index: DeduplicationIndex | None = None,
) -> ValidationResult:
    """Run the full validation pipeline on a single fetched file.

    Args:
        path: Path to the fetched file.
        query: The user's original query (for relevance scoring).
        dedup_index: Optional dedup index to check for duplicates.

    Returns:
        ValidationResult with scores and decision.
    """
    result = ValidationResult(path=str(path))

    if not path.exists():
        result.reasons.append("file does not exist")
        result.decision = "reject"
        return result

    if path.stat().st_size == 0:
        result.reasons.append("empty file")
        result.decision = "reject"
        return result

    # 1. Format detection
    fmt, fmt_confidence = detect_format(path)
    result.format_detected = fmt
    result.format_score = fmt_confidence

    if fmt == "empty":
        result.reasons.append("empty file")
        result.decision = "reject"
        return result

    # 2. Read content for text-based analysis
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        result.reasons.append(f"cannot read file: {e}")
        result.format_score = 0.2
        content = ""

    # 3. Text quality
    if content:
        quality, quality_reasons = compute_text_quality(content)
        result.quality_score = quality
        result.reasons.extend(quality_reasons)
    else:
        result.quality_score = 0.0

    # 4. Relevance
    if content and query:
        result.relevance_score = estimate_relevance(content, query)
    else:
        result.relevance_score = 0.5  # neutral if no query

    # 5. Completeness (simple heuristic: content length)
    if content:
        word_count = len(content.split())
        if word_count < 50:
            result.completeness_score = 0.2
            result.reasons.append("very short content")
        elif word_count < 200:
            result.completeness_score = 0.5
        elif word_count < 1000:
            result.completeness_score = 0.8
        else:
            result.completeness_score = 1.0
    else:
        result.completeness_score = 0.0

    # 6. Deduplication
    if dedup_index and content:
        result.duplicate = dedup_index.is_duplicate(content)
        if result.duplicate:
            result.reasons.append("duplicate content")

    # 7. Composite score
    weights = {
        "format": 0.15,
        "quality": 0.30,
        "relevance": 0.25,
        "completeness": 0.20,
        "dedup": 0.10,
    }
    result.composite_score = (
        weights["format"] * result.format_score
        + weights["quality"] * result.quality_score
        + weights["relevance"] * result.relevance_score
        + weights["completeness"] * result.completeness_score
        + weights["dedup"] * (0.0 if result.duplicate else 1.0)
    )

    # 8. Decision
    if result.duplicate:
        result.decision = "reject"
    elif result.composite_score >= QUALITY_ACCEPT:
        result.decision = "accept"
    elif result.composite_score >= QUALITY_REVIEW:
        result.decision = "review"
    else:
        result.decision = "reject"

    return result


def validate_batch(
    paths: list[Path],
    *,
    query: str = "",
) -> list[ValidationResult]:
    """Validate a batch of fetched files.

    Returns results sorted by composite score (best first).
    """
    dedup = DeduplicationIndex()
    results = [validate_file(p, query=query, dedup_index=dedup) for p in paths]
    results.sort(key=lambda r: r.composite_score, reverse=True)
    return results
