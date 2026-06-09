"""Tests for CLAUDE.md structure and content preservation."""

from __future__ import annotations

from pathlib import Path

import pytest

# Resolve CLAUDE.md relative to this test file: tests/unit/ -> project root
CLAUDE_MD = Path(__file__).parents[2] / "CLAUDE.md"


@pytest.fixture(scope="module")
def claude_md_content() -> str:
    """Read CLAUDE.md once for all tests in this module."""
    assert CLAUDE_MD.exists(), f"CLAUDE.md not found at {CLAUDE_MD}"
    return CLAUDE_MD.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Section structure
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRequiredSections:
    """CLAUDE.md must include the standard template sections."""

    @pytest.mark.parametrize(
        "heading",
        [
            "## Commands",
            "## Architecture",
            "## Conventions",
            "## Gotchas",
        ],
    )
    def test_required_top_level_sections(self, claude_md_content: str, heading: str) -> None:
        assert heading in claude_md_content, f"Missing required section: {heading}"

    def test_starts_with_project_name(self, claude_md_content: str) -> None:
        assert claude_md_content.startswith("# Khora"), "CLAUDE.md must start with '# Khora'"

    def test_gotcha_subsections(self, claude_md_content: str) -> None:
        """Gotchas section must retain its subsections."""
        for subsection in [
            "### Migrations & Schema",
            "### UUID & Type Handling",
            "### Backend Specifics",
            "### Downstream",
        ]:
            assert subsection in claude_md_content, f"Missing Gotchas subsection: {subsection}"


# ---------------------------------------------------------------------------
# Critical gotchas (must be preserved verbatim)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCriticalGotchas:
    """Four critical gotchas that must never be lost during restructuring."""

    def test_create_tables_deprecated(self, claude_md_content: str) -> None:
        assert "create_tables()" in claude_md_content, "Missing critical gotcha: create_tables() deprecation warning"
        assert "deprecated" in claude_md_content.lower(), "create_tables() must be marked as deprecated"

    def test_khora_alembic_version_table(self, claude_md_content: str) -> None:
        assert "khora_alembic_version" in claude_md_content, (
            "Missing critical gotcha: khora_alembic_version table naming"
        )

    def test_as_uuid_true(self, claude_md_content: str) -> None:
        assert "as_uuid=True" in claude_md_content, "Missing critical gotcha: as_uuid=True UUID handling"
        assert "52 UUID columns" in claude_md_content, "Must mention all 52 UUID columns"

    def test_surrealdb_knn_broken(self, claude_md_content: str) -> None:
        assert "SurrealDB KNN broken" in claude_md_content, "Missing critical gotcha: SurrealDB KNN broken"
        assert "<|K|>" in claude_md_content, "Must mention <|K|> operator as unreliable"


# ---------------------------------------------------------------------------
# Additional content preservation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestContentPreservation:
    """Important content that must survive the restructuring."""

    def test_llmusage_contract(self, claude_md_content: str) -> None:
        assert "LLMUsage contract" in claude_md_content

    def test_expertiseconfig_contract(self, claude_md_content: str) -> None:
        assert "ExpertiseConfig contract" in claude_md_content

    def test_pg_advisory_lock(self, claude_md_content: str) -> None:
        assert "pg_advisory_lock" in claude_md_content

    def test_database_ahead_error(self, claude_md_content: str) -> None:
        assert "_DatabaseAheadError" in claude_md_content

    def test_in_failed_sql_transaction_error(self, claude_md_content: str) -> None:
        assert "InFailedSQLTransactionError" in claude_md_content

    def test_surrealdb_entity_key_gate(self, claude_md_content: str) -> None:
        assert "_SurrealDBEntityKeyGate" in claude_md_content

    def test_version_bumps_checklist(self, claude_md_content: str) -> None:
        """Version bump section must reference key files and commands."""
        for item in [
            "rust/khora-accel/Cargo.toml",
            "cargo generate-lockfile",
            "git tag",
        ]:
            assert item in claude_md_content, f"Version bump checklist missing: {item}"

    @pytest.mark.parametrize(
        "command",
        [
            "make test",
            "make format",
            "make lint",
            "make dev",
        ],
    )
    def test_make_commands_present(self, claude_md_content: str, command: str) -> None:
        assert command in claude_md_content, f"Missing make command: {command}"

    def test_workflow_section(self, claude_md_content: str) -> None:
        """CLAUDE.md must describe the issue-tracking + PR workflow inline.

        The repo used to inline a shared workflow doc via
        ``@.claude/docs/workflow.md``; that mechanism was retired when khora
        moved its issue tracking to GitHub Issues for the OSS release. The
        equivalent guidance now lives directly in CLAUDE.md.
        """
        assert "GitHub Issues" in claude_md_content
        assert "github.com/DeytaHQ/khora/issues" in claude_md_content


# ---------------------------------------------------------------------------
# Removed content (README-duplicated sections)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRemovedDuplicateContent:
    """Sections that duplicate README content should NOT be in CLAUDE.md."""

    @pytest.mark.parametrize(
        "heading",
        [
            "## Public API",
            "## Engine Selection",
            "## Dependencies & Extras",
        ],
    )
    def test_readme_sections_removed(self, claude_md_content: str, heading: str) -> None:
        assert heading not in claude_md_content, f"README-duplicated section should be removed: {heading}"
