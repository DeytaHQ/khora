"""Tests for extraction.binary_readers.

Focus is on the PDF boundary: callers hitting a ``.pdf`` path get a clear
``NotImplementedError``, and ``get_extraction_warning`` surfaces the same
message for pre-flight checks. Non-PDF behavior (no-op for unknown
extensions) is preserved.

Also covers #1229 (fd leak on malformed xlsx) and #1233 (failure vs
legitimately-empty distinguishability).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from khora.exceptions import ExtractionError
from khora.extraction.binary_readers import (
    _PDF_REMOVED_MESSAGE,
    extract_docx_text,
    extract_if_needed,
    extract_parquet_text,
    extract_xlsx_text,
    get_extraction_warning,
)

# ---------------------------------------------------------------------------
# Helpers - inject fake optional deps into sys.modules so tests run without
# the real openpyxl / docx / pyarrow packages installed.
# ---------------------------------------------------------------------------


def _fake_openpyxl_module(load_workbook_func):
    """Return a fake openpyxl module whose load_workbook is controllable."""
    mod = ModuleType("openpyxl")
    mod.load_workbook = load_workbook_func
    return mod


def _fake_docx_module(document_func):
    """Return a fake docx module whose Document is controllable."""
    mod = ModuleType("docx")
    mod.Document = document_func
    return mod


def _fake_pyarrow_parquet_module(read_table_func):
    """Return a fake pyarrow.parquet module."""
    mod = ModuleType("pyarrow.parquet")
    mod.read_table = read_table_func
    return mod


def test_extract_if_needed_raises_on_pdf(tmp_path: Path) -> None:
    pdf = tmp_path / "example.pdf"
    pdf.write_bytes(b"%PDF-1.4 not a real pdf")

    with pytest.raises(NotImplementedError) as exc:
        extract_if_needed(pdf)

    assert _PDF_REMOVED_MESSAGE in str(exc.value)


def test_extract_if_needed_raises_on_uppercase_pdf(tmp_path: Path) -> None:
    """Suffix matching is case-insensitive — .PDF routes like .pdf."""
    pdf = tmp_path / "example.PDF"
    pdf.write_bytes(b"%PDF-1.4 not a real pdf")

    with pytest.raises(NotImplementedError):
        extract_if_needed(pdf)


def test_extract_if_needed_returns_none_for_unknown_extension(tmp_path: Path) -> None:
    """Unknown extensions remain a silent no-op — only .pdf changed."""
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


# ---------------------------------------------------------------------------
# #1229 - fd leak on malformed xlsx
# ---------------------------------------------------------------------------


def _xlsx_patches(mock_wb):
    """Context manager that injects a fake openpyxl module and enables _HAS_OPENPYXL."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        fake_mod = _fake_openpyxl_module(MagicMock(return_value=mock_wb))
        with (
            patch.dict(sys.modules, {"openpyxl": fake_mod}),
            patch("khora.extraction.binary_readers._HAS_OPENPYXL", True),
        ):
            yield

    return _ctx()


def test_xlsx_close_called_on_iter_rows_exception(tmp_path: Path) -> None:
    """wb.close() runs even when iter_rows() raises mid-iteration (#1229).

    Simulates a workbook that opens successfully but raises when iterating
    rows (e.g. corrupt cell data, XML parse fault). The fix wraps the
    iteration body in try/finally so close() is guaranteed to run.
    """
    fake_path = tmp_path / "bad.xlsx"
    fake_path.write_bytes(b"not real xlsx")

    mock_wb = MagicMock()
    mock_wb.sheetnames = ["Sheet1"]
    mock_ws = MagicMock()
    mock_ws.iter_rows.side_effect = ValueError("corrupt cell data")
    mock_wb.__getitem__ = MagicMock(return_value=mock_ws)

    with _xlsx_patches(mock_wb):
        with pytest.raises(ExtractionError):
            extract_xlsx_text(fake_path)

    mock_wb.close.assert_called_once()


def test_xlsx_close_called_on_load_workbook_exception(tmp_path: Path) -> None:
    """ExtractionError is raised and no close() attempted when load_workbook fails.

    When the workbook cannot even be opened (corrupt zip, bad magic bytes),
    close() should NOT be called because wb was never assigned.
    """
    fake_path = tmp_path / "bad.xlsx"
    fake_path.write_bytes(b"not real xlsx")

    fake_mod = _fake_openpyxl_module(MagicMock(side_effect=Exception("bad zip")))
    with (
        patch.dict(sys.modules, {"openpyxl": fake_mod}),
        patch("khora.extraction.binary_readers._HAS_OPENPYXL", True),
    ):
        with pytest.raises(ExtractionError, match="bad zip"):
            extract_xlsx_text(fake_path)


# ---------------------------------------------------------------------------
# #1233 - failure distinguishable from legitimately-empty
# ---------------------------------------------------------------------------


def test_xlsx_raises_extraction_error_on_corrupt_file(tmp_path: Path) -> None:
    """A genuine parse error raises ExtractionError, not returns ''.

    Before the fix, corrupt xlsx silently returned '' - same as an empty
    spreadsheet. Now callers can distinguish corruption from empty content.
    """
    fake_path = tmp_path / "corrupt.xlsx"
    fake_path.write_bytes(b"not real xlsx")

    mock_wb = MagicMock()
    mock_wb.sheetnames = ["Sheet1"]
    mock_ws = MagicMock()
    mock_ws.iter_rows.side_effect = ValueError("XML parse error")
    mock_wb.__getitem__ = MagicMock(return_value=mock_ws)

    with _xlsx_patches(mock_wb):
        with pytest.raises(ExtractionError):
            extract_xlsx_text(fake_path)


def test_xlsx_returns_empty_string_for_legitimately_empty_file(tmp_path: Path) -> None:
    """A workbook with no rows returns '' - no extraction error raised.

    Legitimately-empty content (zero rows across all sheets) still returns ''
    so extract_if_needed returns None (not an error).
    """
    fake_path = tmp_path / "empty.xlsx"
    fake_path.write_bytes(b"placeholder")

    mock_wb = MagicMock()
    mock_wb.sheetnames = ["Sheet1"]
    mock_ws = MagicMock()
    mock_ws.iter_rows.return_value = iter([])  # zero rows
    mock_wb.__getitem__ = MagicMock(return_value=mock_ws)

    with _xlsx_patches(mock_wb):
        result = extract_xlsx_text(fake_path)

    assert result == ""


def test_docx_raises_extraction_error_on_corrupt_file(tmp_path: Path) -> None:
    """Corrupt docx raises ExtractionError instead of returning '' (#1233)."""
    fake_path = tmp_path / "corrupt.docx"
    fake_path.write_bytes(b"not real docx")

    fake_mod = _fake_docx_module(MagicMock(side_effect=Exception("bad zip file")))
    with (
        patch.dict(sys.modules, {"docx": fake_mod}),
        patch("khora.extraction.binary_readers._HAS_DOCX", True),
    ):
        with pytest.raises(ExtractionError, match="bad zip file"):
            extract_docx_text(fake_path)


def test_docx_returns_empty_string_for_no_paragraphs(tmp_path: Path) -> None:
    """A docx with no non-empty paragraphs returns '' - not an error."""
    fake_path = tmp_path / "empty.docx"
    fake_path.write_bytes(b"placeholder")

    mock_doc = MagicMock()
    mock_doc.paragraphs = []  # no paragraphs

    fake_mod = _fake_docx_module(MagicMock(return_value=mock_doc))
    with (
        patch.dict(sys.modules, {"docx": fake_mod}),
        patch("khora.extraction.binary_readers._HAS_DOCX", True),
    ):
        result = extract_docx_text(fake_path)

    assert result == ""


def test_parquet_raises_extraction_error_on_corrupt_file(tmp_path: Path) -> None:
    """Corrupt parquet raises ExtractionError instead of returning '' (#1233)."""
    fake_path = tmp_path / "corrupt.parquet"
    fake_path.write_bytes(b"not real parquet")

    fake_pq_mod = _fake_pyarrow_parquet_module(MagicMock(side_effect=Exception("arrow error")))
    with (
        patch.dict(sys.modules, {"pyarrow.parquet": fake_pq_mod}),
        patch("khora.extraction.binary_readers._HAS_PYARROW", True),
    ):
        with pytest.raises(ExtractionError, match="arrow error"):
            extract_parquet_text(fake_path)


def test_extract_if_needed_propagates_extraction_error(tmp_path: Path) -> None:
    """extract_if_needed propagates ExtractionError from a reader (#1233).

    Before the fix, a corrupt xlsx would silently return None from
    extract_if_needed - indistinguishable from 'file has no text'.
    Now callers see ExtractionError.
    """
    fake_path = tmp_path / "corrupt.xlsx"
    fake_path.write_bytes(b"not real xlsx")

    mock_wb = MagicMock()
    mock_wb.sheetnames = ["Sheet1"]
    mock_ws = MagicMock()
    mock_ws.iter_rows.side_effect = ValueError("corrupt")
    mock_wb.__getitem__ = MagicMock(return_value=mock_ws)

    with _xlsx_patches(mock_wb):
        with pytest.raises(ExtractionError):
            extract_if_needed(fake_path)
