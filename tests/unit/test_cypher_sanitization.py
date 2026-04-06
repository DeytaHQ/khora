"""Tests for Cypher injection prevention across graph backends."""

import pytest

from khora.storage.backends.mixins import sanitize_cypher_label


@pytest.mark.unit
class TestSanitizeCypherLabel:
    def test_normal_label(self):
        assert sanitize_cypher_label("RELATES_TO") == "RELATES_TO"

    def test_lowercase(self):
        assert sanitize_cypher_label("relates_to") == "RELATES_TO"

    def test_special_chars_stripped(self):
        assert sanitize_cypher_label("my-relationship") == "MY_RELATIONSHIP"

    def test_injection_attempt_neutralized(self):
        malicious = "X}]-(a)-[:HACK]->(b) RETURN b//"
        result = sanitize_cypher_label(malicious)
        # All Cypher structural characters are stripped
        assert "}" not in result
        assert "]" not in result
        assert "[" not in result
        assert "(" not in result
        assert ")" not in result
        assert ">" not in result
        assert "-" not in result
        assert "/" not in result
        assert ":" not in result
        # The word RETURN survives (it's alphanumeric) but cannot form
        # a valid Cypher clause because all structural chars are gone.
        assert result == "X____A____HACK____B__RETURN_B__"

    def test_empty_returns_default(self):
        assert sanitize_cypher_label("") == "RELATES_TO"
        assert sanitize_cypher_label("   ") == "RELATES_TO"

    def test_only_special_chars(self):
        # All special chars become underscores; result is a valid (if ugly) label
        assert sanitize_cypher_label("!@#$%") == "_____"

    def test_unicode_stripped(self):
        result = sanitize_cypher_label("\u5173\u7cfb\u7c7b\u578b")
        # All non-ASCII characters become underscores; no alphanumeric survives
        assert result == "____"
        # Crucially, no unicode characters remain in the output
        assert all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_" for c in result)

    def test_spaces_converted(self):
        assert sanitize_cypher_label("WORKS AT") == "WORKS_AT"

    def test_dots_converted(self):
        assert sanitize_cypher_label("org.relates_to") == "ORG_RELATES_TO"


@pytest.mark.unit
class TestAGEEscape:
    def test_normal_string(self):
        from khora.storage.backends.age import AGEBackend

        assert AGEBackend._escape("hello world") == "hello world"

    def test_single_quotes(self):
        from khora.storage.backends.age import AGEBackend

        assert AGEBackend._escape("it's") == "it\\'s"

    def test_backslash(self):
        from khora.storage.backends.age import AGEBackend

        assert AGEBackend._escape("path\\to") == "path\\\\to"

    def test_null_bytes_stripped(self):
        from khora.storage.backends.age import AGEBackend

        assert "\x00" not in AGEBackend._escape("test\x00inject")

    def test_newlines_escaped(self):
        from khora.storage.backends.age import AGEBackend

        assert AGEBackend._escape("line1\nline2") == "line1\\nline2"

    def test_carriage_return_escaped(self):
        from khora.storage.backends.age import AGEBackend

        assert AGEBackend._escape("line1\rline2") == "line1\\rline2"

    def test_empty_string(self):
        from khora.storage.backends.age import AGEBackend

        assert AGEBackend._escape("") == ""
