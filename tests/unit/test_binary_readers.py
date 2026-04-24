"""Tests for extraction.binary_readers.

Focus is on the PDF-removal boundary per DYT-3032: callers hitting a .pdf
path should get a clear ``NotImplementedError``, and ``get_extraction_warning``
should surface the same message for pre-flight checks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from khora.extraction.binary_readers import (
    _PDF_REMOVED_MESSAGE,
    extract_if_needed,
    get_extraction_warning,
)


def test_extract_if_needed_raises_on_pdf(tmp_path: Path) -> None:
    pdf = tmp_path / "example.pdf"
    pdf.write_bytes(b"%PDF-1.4 not a real pdf")

    with pytest.raises(NotImplementedError) as exc:
        extract_if_needed(pdf)

    assert _PDF_REMOVED_MESSAGE in str(exc.value)


def test_extract_if_needed_returns_none_for_unknown_extension(tmp_path: Path) -> None:
    """Unknown extensions are still a silent no-op — only .pdf changed."""
    unknown = tmp_path / "example.unknown"
    unknown.write_text("text")

    assert extract_if_needed(unknown) is None


def test_get_extraction_warning_for_pdf_mentions_removal(tmp_path: Path) -> None:
    pdf = tmp_path / "example.pdf"
    pdf.touch()

    warning = get_extraction_warning(pdf)
    assert warning is not None
    assert _PDF_REMOVED_MESSAGE in warning


def test_get_extraction_warning_none_for_clean_path(tmp_path: Path) -> None:
    """No warning for an extension that has no registered extractor."""
    plain = tmp_path / "example.txt"
    plain.touch()

    assert get_extraction_warning(plain) is None
