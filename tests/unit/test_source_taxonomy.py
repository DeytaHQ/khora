"""Tests for source taxonomy and normalization."""

from khora.core.models.source import (
    SOURCE_TYPE_ALIASES,
    SourceTool,
    is_known_source,
    normalize_source_type,
    register_source_alias,
    register_source_type,
)


class TestSourceTool:
    """Test SourceTool enum."""

    def test_enum_values(self):
        """SourceTool should have generic values only."""
        assert SourceTool.FILE.value == "file"
        assert SourceTool.URL.value == "url"
        assert SourceTool.API.value == "api"
        assert SourceTool.UNKNOWN.value == "unknown"


class TestNormalizeSourceType:
    """Test normalize_source_type function."""

    def test_generic_types(self):
        """Generic types normalize correctly."""
        assert normalize_source_type("file") == "file"
        assert normalize_source_type("url") == "url"
        assert normalize_source_type("api") == "api"

    def test_unknown_source(self):
        """Unrecognized source types return 'unknown'."""
        assert normalize_source_type("random_thing") == "unknown"
        assert normalize_source_type("foobar") == "unknown"

    def test_empty_string(self):
        """Empty string returns 'unknown'."""
        assert normalize_source_type("") == "unknown"

    def test_whitespace_handling(self):
        """Leading/trailing whitespace is handled."""
        assert normalize_source_type("  file  ") == "file"


class TestRegistration:
    """Test dynamic source type registration."""

    def test_register_source_type(self):
        """Registering a source type makes it known and normalizable."""
        register_source_type("slack", display_name="Slack")
        assert normalize_source_type("slack") == "slack"
        assert is_known_source("slack")

        # Clean up
        SOURCE_TYPE_ALIASES.pop("slack", None)

    def test_register_source_alias(self):
        """Registering aliases maps them to canonical types."""
        register_source_type("linear")
        register_source_alias("linear_issue", "linear")
        register_source_alias("linear_project", "linear")

        assert normalize_source_type("linear_issue") == "linear"
        assert normalize_source_type("linear_project") == "linear"

        # Clean up
        SOURCE_TYPE_ALIASES.pop("linear", None)
        SOURCE_TYPE_ALIASES.pop("linear_issue", None)
        SOURCE_TYPE_ALIASES.pop("linear_project", None)

    def test_register_case_insensitive(self):
        """Registration and lookup should be case-insensitive."""
        register_source_type("GitHub")
        assert normalize_source_type("github") == "github"
        assert normalize_source_type("GitHub") == "github"

        # Clean up
        SOURCE_TYPE_ALIASES.pop("github", None)

    def test_is_known_source_unregistered(self):
        """Unregistered source types should not be known."""
        assert not is_known_source("totally_new_tool")
