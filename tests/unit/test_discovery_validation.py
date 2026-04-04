"""Unit tests for discovery data validation pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from khora.discovery.validation import (
    DeduplicationIndex,
    _simple_stem,
    compute_text_quality,
    detect_format,
    estimate_relevance,
    estimate_relevance_semantic,
    validate_batch,
    validate_file,
)

# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


class TestDetectFormat:
    def test_json_file(self, tmp_path: Path) -> None:
        f = tmp_path / "data.json"
        f.write_text('{"key": "value"}')
        fmt, conf = detect_format(f)
        assert fmt == "json"
        assert conf >= 0.9

    def test_csv_file(self, tmp_path: Path) -> None:
        f = tmp_path / "data.csv"
        f.write_text("name,age\nAlice,30\nBob,25\n")
        fmt, conf = detect_format(f)
        assert fmt == "csv"
        assert conf >= 0.7

    def test_jsonl_file(self, tmp_path: Path) -> None:
        f = tmp_path / "data.jsonl"
        f.write_text('{"a": 1}\n{"a": 2}\n{"a": 3}\n')
        fmt, conf = detect_format(f)
        assert fmt == "jsonl"
        assert conf >= 0.9

    def test_markdown_file(self, tmp_path: Path) -> None:
        f = tmp_path / "readme.md"
        f.write_text("# Title\n\nSome content here\n")
        fmt, conf = detect_format(f)
        assert fmt == "markdown"
        assert conf >= 0.6

    def test_html_file(self, tmp_path: Path) -> None:
        f = tmp_path / "page.html"
        f.write_text("<html><body>Hello</body></html>")
        fmt, conf = detect_format(f)
        assert fmt == "html"
        assert conf >= 0.8

    def test_text_file(self, tmp_path: Path) -> None:
        f = tmp_path / "notes.txt"
        f.write_text("Just some plain text content.")
        fmt, conf = detect_format(f)
        assert fmt == "text"
        assert conf >= 0.5

    def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.txt"
        f.write_text("")
        fmt, conf = detect_format(f)
        assert fmt == "empty"
        assert conf == 0.0

    def test_unknown_extension_json_content(self, tmp_path: Path) -> None:
        f = tmp_path / "data.dat"
        f.write_text('{"key": "value"}')
        fmt, conf = detect_format(f)
        assert fmt == "json"

    def test_unknown_extension_csv_content(self, tmp_path: Path) -> None:
        f = tmp_path / "data.dat"
        f.write_text("a,b,c\n1,2,3\n4,5,6\n")
        fmt, conf = detect_format(f)
        assert fmt == "csv"


# ---------------------------------------------------------------------------
# Text quality
# ---------------------------------------------------------------------------


class TestTextQuality:
    def test_good_text(self) -> None:
        text = (
            "The European wine industry produces a wide variety of wines "
            "from different grape varieties across numerous regions. France, "
            "Italy, and Spain are among the largest producers. Each region "
            "has unique terroir characteristics that influence the flavor "
            "profile of the wines produced there. Climate, soil composition, "
            "and winemaking traditions all play important roles."
        )
        score, reasons = compute_text_quality(text)
        assert score > 0.5
        assert len(reasons) == 0

    def test_empty_text(self) -> None:
        score, reasons = compute_text_quality("")
        assert score == 0.0
        assert "empty" in reasons[0]

    def test_very_short(self) -> None:
        score, reasons = compute_text_quality("Hello world.")
        assert score < 0.3

    def test_repetitive_text(self) -> None:
        text = (
            ("Buy now! " * 200)
            + "\n\n"
            + ("Buy now! " * 200)
            + "\n\n"
            + ("Buy now! " * 200)
            + "\n\n"
            + ("Buy now! " * 200)
        )
        score, reasons = compute_text_quality(text)
        assert score < 0.6
        assert any("entropy" in r or "diversity" in r or "repeated" in r for r in reasons)

    def test_boilerplate(self) -> None:
        text = (
            "Lorem ipsum dolor sit amet. Cookie policy and privacy policy "
            "terms of service placeholder text coming soon under construction. "
            "This is just example data for testing purposes." * 5
        )
        score, reasons = compute_text_quality(text)
        assert any("boilerplate" in r for r in reasons)

    def test_boilerplate_cookie_consent(self) -> None:
        """Context-aware patterns match cookie consent but not random 'cookie' mentions."""
        text = (
            "We use cookies to improve your experience. Accept all cookies. "
            "Subscribe to our newsletter for updates. Privacy policy applies. " * 10
        )
        score, reasons = compute_text_quality(text)
        assert any("boilerplate" in r for r in reasons)

    def test_encoding_issues(self) -> None:
        text = "Normal text " + "\ufffd" * 100 + " more text " * 50
        score, reasons = compute_text_quality(text)
        assert any("encoding" in r for r in reasons)


# ---------------------------------------------------------------------------
# Relevance estimation
# ---------------------------------------------------------------------------


class TestSimpleStem:
    def test_strips_common_suffixes(self) -> None:
        assert _simple_stem("production") == "produc"  # strips "tion"
        assert _simple_stem("datasets") == "dataset"  # strips "s"
        assert _simple_stem("running") == "runn"  # strips "ing"

    def test_preserves_short_words(self) -> None:
        # Word must be longer than suffix + 3 to strip
        assert _simple_stem("the") == "the"
        assert _simple_stem("used") == "used"
        assert _simple_stem("is") == "is"

    def test_no_suffix_match(self) -> None:
        assert _simple_stem("graph") == "graph"


class TestRelevance:
    def test_high_relevance(self) -> None:
        content = (
            "This dataset contains European wine quality measurements "
            "including acidity, sugar content, and expert ratings for "
            "red and white wines from Portugal."
        )
        score = estimate_relevance(content, "European wine quality dataset")
        assert score > 0.5

    def test_low_relevance(self) -> None:
        content = (
            "This document describes the migration patterns of Arctic "
            "terns across the Pacific Ocean during winter months."
        )
        score = estimate_relevance(content, "European wine quality dataset")
        assert score < 0.3

    def test_empty_query(self) -> None:
        assert estimate_relevance("some content", "") == 0.0

    def test_empty_content(self) -> None:
        assert estimate_relevance("", "some query") == 0.0

    def test_stemmed_matching_boosts_score(self) -> None:
        """Stemming should match 'production' to 'produce', etc."""
        content = (
            "This report covers wine production statistics across " "European regions including quality assessments."
        )
        # "datasets" in query should match "dataset"-like stems in content
        score = estimate_relevance(content, "wine production quality")
        assert score > 0.4

    def test_bigram_matching(self) -> None:
        """Consecutive query words matching in content should boost score."""
        content = (
            "The field of machine learning has transformed data science. "
            "Machine learning techniques are used across many industries."
        )
        score_bigram = estimate_relevance(content, "machine learning data")
        score_no_bigram = estimate_relevance(content, "learning machine data")
        # Bigram "machine learning" appears in content, so score should be higher
        assert score_bigram >= score_no_bigram

    def test_single_word_query_no_bigrams(self) -> None:
        """A single-word query produces no bigrams and should still work."""
        content = "Wine quality measurements from various regions."
        score = estimate_relevance(content, "wine")
        assert score > 0.0


class TestRelevanceSemantic:
    @pytest.mark.asyncio
    async def test_falls_back_without_llm(self) -> None:
        """Without an LLM, should fall back to keyword-based scoring."""
        content = "European wine quality data with expert ratings."
        score = await estimate_relevance_semantic(content, "wine quality")
        keyword_score = estimate_relevance(content, "wine quality")
        assert score == keyword_score

    @pytest.mark.asyncio
    async def test_falls_back_on_llm_error(self) -> None:
        """If the LLM raises, should fall back gracefully."""

        class BrokenLLM:
            async def complete(self, **kwargs: object) -> dict:
                raise RuntimeError("LLM unavailable")

        content = "European wine quality data with expert ratings."
        score = await estimate_relevance_semantic(content, "wine quality", llm=BrokenLLM())
        keyword_score = estimate_relevance(content, "wine quality")
        assert score == keyword_score

    @pytest.mark.asyncio
    async def test_uses_llm_score(self) -> None:
        """When LLM returns a valid score, should use it."""

        class FakeLLM:
            async def complete(self, **kwargs: object) -> dict:
                return {"score": 8, "reason": "highly relevant"}

        score = await estimate_relevance_semantic("content", "query", llm=FakeLLM())
        assert score == pytest.approx(0.8)

    @pytest.mark.asyncio
    async def test_clamps_score(self) -> None:
        """Scores outside 0-10 should be clamped to [0, 1]."""

        class HighLLM:
            async def complete(self, **kwargs: object) -> dict:
                return {"score": 15, "reason": "off scale"}

        score = await estimate_relevance_semantic("content", "query", llm=HighLLM())
        assert score == 1.0


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_detects_duplicate(self) -> None:
        idx = DeduplicationIndex()
        assert idx.is_duplicate("hello world") is False
        assert idx.is_duplicate("hello world") is True

    def test_different_content(self) -> None:
        idx = DeduplicationIndex()
        assert idx.is_duplicate("hello") is False
        assert idx.is_duplicate("world") is False

    def test_add_existing(self) -> None:
        idx = DeduplicationIndex()
        idx.add_existing("preloaded content")
        assert idx.is_duplicate("preloaded content") is True
        assert idx.size == 1

    def test_size(self) -> None:
        idx = DeduplicationIndex()
        idx.is_duplicate("a")
        idx.is_duplicate("b")
        idx.is_duplicate("a")  # duplicate, not added again
        assert idx.size == 2


# ---------------------------------------------------------------------------
# Full validation pipeline
# ---------------------------------------------------------------------------


class TestValidateFile:
    def test_good_json_file(self, tmp_path: Path) -> None:
        f = tmp_path / "data.json"
        data = [{"name": "Wine A", "region": "Bordeaux", "rating": 92}] * 20
        f.write_text(json.dumps(data, indent=2))
        result = validate_file(f, query="wine data")
        assert result.format_detected == "json"
        assert result.format_score >= 0.9
        assert result.composite_score > 0.4
        assert result.decision in ("accept", "review")

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        result = validate_file(tmp_path / "missing.txt")
        assert result.decision == "reject"
        assert "does not exist" in result.reasons[0]

    def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.txt"
        f.write_text("")
        result = validate_file(f)
        assert result.decision == "reject"

    def test_duplicate_detection(self, tmp_path: Path) -> None:
        content = "Wine data from Bordeaux region with quality ratings." * 20
        f1 = tmp_path / "data1.txt"
        f2 = tmp_path / "data2.txt"
        f1.write_text(content)
        f2.write_text(content)

        idx = DeduplicationIndex()
        r1 = validate_file(f1, dedup_index=idx)
        r2 = validate_file(f2, dedup_index=idx)
        assert r1.duplicate is False
        assert r2.duplicate is True
        assert r2.decision == "reject"

    def test_result_to_dict(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("Some content here " * 50)
        result = validate_file(f)
        d = result.to_dict()
        assert "path" in d
        assert "composite_score" in d
        assert "decision" in d


# ---------------------------------------------------------------------------
# Batch validation
# ---------------------------------------------------------------------------


class TestValidateBatch:
    def test_sorts_by_score(self, tmp_path: Path) -> None:
        # Good file
        good = tmp_path / "good.json"
        good.write_text(json.dumps([{"wine": "Bordeaux", "rating": 95}] * 30))

        # Bad file
        bad = tmp_path / "bad.txt"
        bad.write_text("x")

        results = validate_batch([bad, good], query="wine ratings")
        assert len(results) == 2
        # Good file should be first (higher score)
        assert results[0].path == str(good)

    def test_dedup_across_batch(self, tmp_path: Path) -> None:
        content = "Identical content " * 100
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text(content)
        f2.write_text(content)

        results = validate_batch([f1, f2])
        duplicates = [r for r in results if r.duplicate]
        assert len(duplicates) == 1

    def test_empty_batch(self) -> None:
        results = validate_batch([])
        assert results == []


# ---------------------------------------------------------------------------
# Content classification
# ---------------------------------------------------------------------------


class TestClassifyContent:
    def test_index_page_with_pdfs(self) -> None:
        """A page listing many PDF links should be classified as INDEX."""
        from khora.discovery.validation import ContentClass, classify_content

        content = "# Data Set 1 Files\n\n"
        for i in range(50):
            content += f"- [EFTA{i:05d}.pdf](https://example.gov/files/EFTA{i:05d}.pdf)\n"
        result = classify_content(content, "https://example.gov/dataset")
        assert result.content_class == ContentClass.INDEX
        assert len(result.document_links) == 50
        assert "weighted score" in result.reason

    def test_index_page_with_few_doc_links(self) -> None:
        """4 document links with weighting should trigger index detection."""
        from khora.discovery.validation import ContentClass, classify_content

        content = "# Downloads\n\n"
        for i in range(4):
            content += f"- [report{i}.pdf](https://example.gov/report{i}.pdf)\n"
        # 4 doc links * 2 weight = 8, + 0 subpage = 8 < 15 threshold
        # So this should NOT be classified as index (too few)
        result = classify_content(content, "https://example.gov/data")
        assert result.content_class == ContentClass.CONTENT

    def test_normal_content(self) -> None:
        """Normal prose content should be classified as CONTENT."""
        from khora.discovery.validation import ContentClass, classify_content

        content = "# Wine Quality Analysis\n\n" + ("This is a detailed analysis of wine data. " * 50)
        result = classify_content(content)
        assert result.content_class == ContentClass.CONTENT

    def test_error_page(self) -> None:
        """Short error content should be classified as ERROR."""
        from khora.discovery.validation import ContentClass, classify_content

        result = classify_content("403 Forbidden. Access denied.")
        assert result.content_class == ContentClass.ERROR

    def test_empty_content(self) -> None:
        from khora.discovery.validation import ContentClass, classify_content

        result = classify_content("")
        assert result.content_class == ContentClass.ERROR

    def test_link_extraction_filters_by_extension(self) -> None:
        from khora.discovery.validation import classify_content

        content = (
            "- [data.pdf](https://example.com/data.pdf)\n"
            "- [info.csv](https://example.com/info.csv)\n"
            "- [About](https://example.com/about)\n"
            "- [Home](https://other.com/home)\n"
        )
        result = classify_content(content, "https://example.com/page")
        assert len(result.document_links) == 2
        assert result.document_links[0][1] == "https://example.com/data.pdf"
