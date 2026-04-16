"""Binary format extraction for ingestion.

Converts PDF, Excel, Word, and Parquet binary formats to text/markdown so
they can be fed into the standard text-ingest pipeline.  All extractors
are optional — they degrade gracefully with a warning when dependencies
are missing.

Install: ``pip install khora[binary-readers]`` for pymupdf + openpyxl +
python-docx, and ``pip install khora[parquet]`` for pyarrow.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

from loguru import logger

# ---------------------------------------------------------------------------
# Optional dependency detection
# ---------------------------------------------------------------------------

_HAS_PYMUPDF = False
try:
    import pymupdf  # noqa: F401

    _HAS_PYMUPDF = True
except ImportError:
    pass

_HAS_OPENPYXL = False
try:
    import openpyxl  # noqa: F401

    _HAS_OPENPYXL = True
except ImportError:
    pass

_HAS_DOCX = False
try:
    import docx  # noqa: F401

    _HAS_DOCX = True
except ImportError:
    pass

_HAS_PYARROW = False
try:
    import pyarrow  # noqa: F401

    _HAS_PYARROW = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------


def extract_pdf_text(path: Path) -> str:
    """Extract text content from a PDF file.

    Uses pymupdf (PyMuPDF) for fast text extraction.
    Returns empty string if pymupdf is not installed.
    """
    if not _HAS_PYMUPDF:
        logger.warning(f"pymupdf not installed — cannot extract text from {path.name}. Install: pip install pymupdf")
        return ""

    import pymupdf

    try:
        doc = pymupdf.open(str(path))
        pages: list[str] = []
        for page in doc:
            text = page.get_text()
            if text.strip():
                pages.append(text)
        page_count = doc.page_count
        doc.close()

        full_text = "\n\n---\n\n".join(pages)

        # Check for scanned PDFs (very little text per page)
        if page_count > 0:
            chars_per_page = len(full_text) / page_count
            if chars_per_page < 50:
                logger.warning(
                    f"{path.name}: ~{chars_per_page:.0f} chars/page — "
                    "possibly a scanned PDF (OCR not supported, text may be incomplete)"
                )

        return full_text
    except Exception as e:
        logger.error(f"PDF extraction failed for {path.name}: {e}")
        return ""


def extract_xlsx_text(path: Path) -> str:
    """Extract text from an Excel (.xlsx) file as CSV-formatted text.

    Uses openpyxl for reading. Returns empty string if not installed.
    """
    if not _HAS_OPENPYXL:
        logger.warning(f"openpyxl not installed — cannot extract text from {path.name}. Install: pip install openpyxl")
        return ""

    import openpyxl

    try:
        wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
        sections: list[str] = []

        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            buf = io.StringIO()
            writer = csv.writer(buf)
            row_count = 0

            for row in ws.iter_rows(values_only=True):
                writer.writerow([str(cell) if cell is not None else "" for cell in row])
                row_count += 1

            if row_count > 0:
                sections.append(f"## Sheet: {sheet_name}\n\n```csv\n{buf.getvalue()}```\n")

        wb.close()
        return "\n".join(sections)
    except Exception as e:
        logger.error(f"Excel extraction failed for {path.name}: {e}")
        return ""


def extract_xls_text(path: Path) -> str:
    """Extract text from a legacy .xls file.

    Falls back to a warning — xlrd is not included as a dependency.
    """
    logger.warning(f"Legacy .xls format not supported: {path.name}. Convert to .xlsx first.")
    return ""


def extract_docx_text(path: Path) -> str:
    """Extract text from a Word (.docx) file.

    Uses python-docx for paragraph extraction.
    Returns empty string if python-docx is not installed.
    """
    if not _HAS_DOCX:
        logger.warning(
            f"python-docx not installed — cannot extract text from {path.name}. Install: pip install python-docx"
        )
        return ""

    import docx

    try:
        doc = docx.Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)
    except Exception as e:
        logger.error(f"DOCX extraction failed for {path.name}: {e}")
        return ""


def extract_parquet_text(path: Path) -> str:
    """Extract data from a Parquet file as markdown table.

    Uses pyarrow for reading. Returns the first 100 rows and schema.
    Returns empty string if pyarrow is not installed.
    """
    if not _HAS_PYARROW:
        logger.warning(f"pyarrow not installed — cannot extract data from {path.name}. Install: pip install pyarrow")
        return ""

    import pyarrow.parquet as pq

    try:
        table = pq.read_table(str(path))
        schema_text = "## Schema\n\n"
        for field in table.schema:
            schema_text += f"- **{field.name}**: {field.type}\n"

        # Convert first 100 rows to pandas for display
        df = table.to_pandas().head(100)
        rows_text = f"\n## Data ({len(table)} total rows, showing first {min(100, len(table))})\n\n"
        rows_text += df.to_markdown(index=False) if hasattr(df, "to_markdown") else df.to_string(index=False)

        return schema_text + rows_text
    except Exception as e:
        logger.error(f"Parquet extraction failed for {path.name}: {e}")
        return ""


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

#: Map of file extensions to extraction functions
_EXTRACTORS: dict[str, callable] = {
    ".pdf": extract_pdf_text,
    ".xlsx": extract_xlsx_text,
    ".xls": extract_xls_text,
    ".docx": extract_docx_text,
    ".parquet": extract_parquet_text,
}


def get_extraction_warning(path: Path) -> str | None:
    """Return a user-facing warning if extraction would fail for this file type.

    Checks whether the required optional dependency is installed for the
    given file extension.  Returns ``None`` when everything looks fine.
    """
    ext = path.suffix.lower()
    if ext == ".pdf" and not _HAS_PYMUPDF:
        return f"Cannot extract text from {path.name} — install pymupdf: pip install pymupdf"
    if ext in (".xlsx", ".xls") and not _HAS_OPENPYXL:
        return f"Cannot extract text from {path.name} — install openpyxl: pip install openpyxl"
    if ext == ".docx" and not _HAS_DOCX:
        return f"Cannot extract text from {path.name} — install python-docx: pip install python-docx"
    if ext == ".parquet" and not _HAS_PYARROW:
        return f"Cannot extract text from {path.name} — install pyarrow: pip install pyarrow"
    return None


def extract_if_needed(path: Path) -> Path | None:
    """Extract text from a binary file if an extractor is available.

    If extraction succeeds, writes a `{name}_extracted.md` file alongside
    the original and returns the path to it.  Returns None if no
    extraction was needed or if it failed.
    """
    ext = path.suffix.lower()
    extractor = _EXTRACTORS.get(ext)

    if extractor is None:
        return None

    text = extractor(path)
    if not text:
        return None

    # Write extracted text alongside the original
    extracted_path = path.with_name(f"{path.stem}_extracted.md")
    extracted_path.write_text(text, encoding="utf-8")
    logger.info(f"Extracted {len(text):,} chars from {path.name} → {extracted_path.name}")
    return extracted_path
