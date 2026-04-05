"""Tests for discovery binary format extractors."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from khora.discovery.extraction import (
    extract_if_needed,
    get_extraction_warning,
)


@pytest.mark.unit
class TestExtractionWarnings:
    """Test get_extraction_warning for all supported types."""

    def test_no_warning_for_txt(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        assert get_extraction_warning(f) is None

    def test_pdf_warning_when_missing(self, tmp_path):
        f = tmp_path / "test.pdf"
        f.write_bytes(b"fake pdf")
        with patch("khora.discovery.extraction._HAS_PYMUPDF", False):
            warning = get_extraction_warning(f)
            assert warning is not None
            assert "pymupdf" in warning

    def test_xlsx_warning_when_missing(self, tmp_path):
        f = tmp_path / "test.xlsx"
        f.write_bytes(b"fake xlsx")
        with patch("khora.discovery.extraction._HAS_OPENPYXL", False):
            warning = get_extraction_warning(f)
            assert warning is not None
            assert "openpyxl" in warning

    def test_docx_warning_when_missing(self, tmp_path):
        f = tmp_path / "test.docx"
        f.write_bytes(b"fake docx")
        with patch("khora.discovery.extraction._HAS_DOCX", False):
            warning = get_extraction_warning(f)
            assert warning is not None
            assert "python-docx" in warning

    def test_parquet_warning_when_missing(self, tmp_path):
        f = tmp_path / "test.parquet"
        f.write_bytes(b"fake parquet")
        with patch("khora.discovery.extraction._HAS_PYARROW", False):
            warning = get_extraction_warning(f)
            assert warning is not None
            assert "pyarrow" in warning

    def test_no_warning_when_pdf_available(self, tmp_path):
        f = tmp_path / "test.pdf"
        f.write_bytes(b"fake pdf")
        with patch("khora.discovery.extraction._HAS_PYMUPDF", True):
            assert get_extraction_warning(f) is None

    def test_no_warning_when_docx_available(self, tmp_path):
        f = tmp_path / "test.docx"
        f.write_bytes(b"fake docx")
        with patch("khora.discovery.extraction._HAS_DOCX", True):
            assert get_extraction_warning(f) is None

    def test_no_warning_when_parquet_available(self, tmp_path):
        f = tmp_path / "test.parquet"
        f.write_bytes(b"fake parquet")
        with patch("khora.discovery.extraction._HAS_PYARROW", True):
            assert get_extraction_warning(f) is None


@pytest.mark.unit
class TestExtractIfNeeded:
    """Test the extract_if_needed dispatcher."""

    def test_returns_none_for_unknown_extension(self, tmp_path):
        f = tmp_path / "test.zip"
        f.write_bytes(b"fake zip")
        assert extract_if_needed(f) is None

    def test_returns_none_for_txt(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        assert extract_if_needed(f) is None

    def test_returns_none_when_extractor_returns_empty(self, tmp_path):
        f = tmp_path / "test.pdf"
        f.write_bytes(b"fake pdf")
        with patch("khora.discovery.extraction._HAS_PYMUPDF", False):
            assert extract_if_needed(f) is None

    def test_returns_none_when_docx_extractor_returns_empty(self, tmp_path):
        f = tmp_path / "test.docx"
        f.write_bytes(b"fake docx")
        with patch("khora.discovery.extraction._HAS_DOCX", False):
            assert extract_if_needed(f) is None

    def test_returns_none_when_parquet_extractor_returns_empty(self, tmp_path):
        f = tmp_path / "test.parquet"
        f.write_bytes(b"fake parquet")
        with patch("khora.discovery.extraction._HAS_PYARROW", False):
            assert extract_if_needed(f) is None
