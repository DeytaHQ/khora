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
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

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
            except Exception as e:
                logger.debug(f"Content validation failed for {path}: {e}")
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


_BOILERPLATE_PATTERNS = [
    re.compile(r"\blorem\s+ipsum\b", re.IGNORECASE),
    re.compile(r"\bexample\s+data\b", re.IGNORECASE),
    re.compile(r"\bplaceholder\b", re.IGNORECASE),
    re.compile(r"\bcoming\s+soon\b", re.IGNORECASE),
    re.compile(r"\bunder\s+construction\b", re.IGNORECASE),
    re.compile(r"\bcookie\s+(policy|consent|notice|preferences)\b", re.IGNORECASE),
    re.compile(r"\bprivacy\s+policy\b", re.IGNORECASE),
    re.compile(r"\bterms\s+(of\s+)?(service|use)\b", re.IGNORECASE),
    re.compile(r"\bsubscribe\s+to\s+(our\s+)?newsletter\b", re.IGNORECASE),
    re.compile(r"\bwe\s+use\s+cookies\b", re.IGNORECASE),
    re.compile(r"\baccept\s+(all\s+)?cookies\b", re.IGNORECASE),
]


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
    boilerplate_hits = sum(1 for p in _BOILERPLATE_PATTERNS if p.search(content))
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


def _simple_stem(word: str) -> str:
    """Lightweight English suffix stripping (no external dependencies)."""
    for suffix in (
        "ation",
        "tion",
        "sion",
        "ment",
        "ness",
        "ence",
        "ance",
        "ible",
        "able",
        "ious",
        "eous",
        "ings",
        "ally",
        "ful",
        "less",
        "ing",
        "ity",
        "ies",
        "ous",
        "ive",
        "ent",
        "ant",
        "ist",
        "ism",
        "ers",
        "ed",
        "ly",
        "er",
        "es",
        "al",
        "ic",
        "en",
        "or",
        "ty",
        "ry",
        "ar",
        "s",
    ):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def estimate_relevance(content: str, query: str) -> float:
    """Estimate content relevance to the user's query.

    Uses keyword overlap with stemming and bigram matching —
    lightweight, no embedding model needed.
    """
    if not content or not query:
        return 0.0

    # Tokenize
    query_tokens = re.findall(r"\w{3,}", query.lower())
    content_tokens = re.findall(r"\w{3,}", content.lower())

    if not query_tokens or not content_tokens:
        return 0.0

    # Exact word matching
    query_words = set(query_tokens)
    content_counter = Counter(content_tokens)
    exact_hits = sum(1 for qw in query_words if content_counter[qw] > 0)

    # Stemmed matching (catches "datasets" vs "dataset", "production" vs "produce")
    query_stems = {_simple_stem(w) for w in query_tokens}
    content_stems = Counter(_simple_stem(w) for w in content_tokens)
    stem_hits = sum(1 for qs in query_stems if content_stems[qs] > 0)

    # Bigram matching (catches "machine learning", "wine production")
    query_bigrams: set[str] = set()
    for i in range(len(query_tokens) - 1):
        query_bigrams.add(f"{query_tokens[i]} {query_tokens[i + 1]}")

    content_bigram_set: set[str] = set()
    for i in range(len(content_tokens) - 1):
        content_bigram_set.add(f"{content_tokens[i]} {content_tokens[i + 1]}")

    bigram_hits = len(query_bigrams & content_bigram_set)

    # Coverage scores
    exact_coverage = exact_hits / len(query_words) if query_words else 0
    stem_coverage = stem_hits / len(query_stems) if query_stems else 0
    bigram_coverage = bigram_hits / len(query_bigrams) if query_bigrams else 0

    # Density: how frequently query terms appear
    total_content_words = len(content_tokens)
    query_term_count = sum(content_counter[qw] for qw in query_words)
    density = min(1.0, query_term_count / max(1, total_content_words) * 20)

    # Weighted combination: stem coverage > exact coverage > density > bigrams
    return 0.30 * exact_coverage + 0.30 * stem_coverage + 0.20 * density + 0.20 * bigram_coverage


async def estimate_relevance_semantic(
    content: str,
    query: str,
    *,
    llm: Any | None = None,
) -> float:
    """Estimate relevance using LLM (optional, budget-aware).

    Falls back to keyword-based ``estimate_relevance()`` on error or if
    no *llm* is provided.
    """
    if llm is None:
        return estimate_relevance(content, query)

    truncated = content[:2000]
    try:
        result = await llm.complete(
            system=(
                "Rate how relevant this content is to the user's query on a scale of 0 to 10. "
                "10 = perfectly relevant dataset/source, 0 = completely unrelated. "
                'Return JSON: {"score": <0-10>, "reason": "brief explanation"}'
            ),
            user=f"Query: {query}\n\nContent preview:\n{truncated}",
            temperature=0.1,
        )
        score = float(result.get("score", 5)) / 10.0
        return max(0.0, min(1.0, score))
    except Exception as e:
        logger.debug("Semantic relevance failed, falling back to keyword: %s", e)
        return estimate_relevance(content, query)


# ---------------------------------------------------------------------------
# Content classification (link-index detection)
# ---------------------------------------------------------------------------

#: File extensions that indicate downloadable documents
_DOCUMENT_EXTENSIONS = frozenset(
    {
        ".pdf",
        ".csv",
        ".xls",
        ".xlsx",
        ".doc",
        ".docx",
        ".json",
        ".jsonl",
        ".xml",
        ".zip",
        ".gz",
        ".parquet",
        ".ppt",
        ".pptx",
        ".txt",
        ".tsv",
    }
)

_LINK_RE = re.compile(r"\[([^\]]*)\]\((https?://[^)]+)\)")


class ContentClass:
    """Classification of fetched content."""

    CONTENT = "content"  # actual data/text
    INDEX = "index"  # link directory (e.g., DOJ page with PDF links)
    METADATA = "metadata"  # about page, FAQ, etc.
    ERROR = "error"  # auth wall, 404, captcha


@dataclass(slots=True)
class ContentClassification:
    """Result of classifying fetched content."""

    content_class: str = ContentClass.CONTENT
    document_links: list[tuple[str, str]] = field(default_factory=list)  # (title, url)
    subpage_links: list[tuple[str, str]] = field(default_factory=list)  # (title, url)
    link_count: int = 0
    prose_lines: int = 0
    reason: str = ""


def classify_content(content: str, source_url: str = "") -> ContentClassification:
    """Classify fetched content using heuristics (no LLM needed).

    Detects link-index pages (like DOJ pages listing 100+ PDF links),
    error pages (auth walls, 404s), and metadata pages.

    Args:
        content: The fetched text/markdown content.
        source_url: Original URL for domain matching.

    Returns:
        ContentClassification with detected type and extracted links.
    """
    result = ContentClassification()

    if not content or len(content.strip()) < 50:
        result.content_class = ContentClass.ERROR
        result.reason = "empty or very short content"
        return result

    lines = content.strip().splitlines()

    # Extract all markdown links
    all_links = _LINK_RE.findall(content)
    result.link_count = len(all_links)

    # Count prose lines (non-empty, non-link-only lines)
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("- [") and not stripped.startswith("* ["):
            if not _LINK_RE.fullmatch(stripped):
                result.prose_lines += 1

    # Classify links into document links and subpage links
    for title, url in all_links:
        url_lower = url.lower()
        ext = Path(url_lower.split("?")[0]).suffix
        if ext in _DOCUMENT_EXTENSIONS:
            result.document_links.append((title, url))
        elif source_url:
            # Same domain, no file extension = likely a subpage
            from urllib.parse import urlparse

            src_domain = urlparse(source_url).netloc
            link_domain = urlparse(url).netloc
            if src_domain and link_domain == src_domain and ext not in _DOCUMENT_EXTENSIONS:
                result.subpage_links.append((title, url))

    # -- Error detection --
    lower = content.lower()
    error_patterns = [
        "403 forbidden",
        "401 unauthorized",
        "access denied",
        "sign in",
        "login required",
        "captcha",
        "404 not found",
    ]
    if len(content) < 500 and any(p in lower for p in error_patterns):
        result.content_class = ContentClass.ERROR
        result.reason = "error/auth page detected"
        return result

    # -- Index detection --
    # Weight document links higher than generic links
    doc_link_weight = len(result.document_links) * 2  # PDFs/CSVs are strong index signals
    total_weighted_links = doc_link_weight + len(result.subpage_links)

    if total_weighted_links > 15 and len(result.document_links) > 3:
        result.content_class = ContentClass.INDEX
        result.reason = f"{len(result.document_links)} document links detected (weighted score: {total_weighted_links})"
        return result

    if result.link_count > 10 and result.prose_lines > 0:
        link_ratio = result.link_count / max(1, result.prose_lines)
        if link_ratio > 2.0:
            result.content_class = ContentClass.INDEX
            result.reason = f"high link-to-prose ratio ({link_ratio:.1f})"
            return result

    # -- Metadata detection --
    meta_patterns = ["about us", "contact us", "privacy policy", "terms of service", "frequently asked"]
    meta_hits = sum(1 for p in meta_patterns if p in lower)
    if meta_hits >= 2 and result.prose_lines < 50:
        result.content_class = ContentClass.METADATA
        result.reason = f"metadata/about page ({meta_hits} patterns)"
        return result

    result.content_class = ContentClass.CONTENT
    return result


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
