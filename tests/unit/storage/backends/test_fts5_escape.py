"""Unit tests for ``khora.storage.backends._fts5.escape_fts5_query``.

Regression for https://github.com/DeytaHQ/khora/issues/526 — natural-language
``recall()`` queries crashed SQLite FTS5 with ``syntax error near "?"``.
"""

from __future__ import annotations

import pytest

from khora.storage.backends._fts5 import escape_fts5_query


class TestEscapeFTS5Query:
    def test_punctuation_in_query_is_safe(self) -> None:
        # The exact input that crashed in issue #526.
        result = escape_fts5_query("What did Curie win?")
        assert result == '"What" "did" "Curie" "win?"'

    def test_each_punctuation_class(self) -> None:
        # `?`, `!`, `.`, `,`, `@`, `:` were the symptoms in the bug report.
        result = escape_fts5_query("hi! @user, what's up: doc. ok?")
        assert result == '"hi!" "@user," "what\'s" "up:" "doc." "ok?"'

    def test_embedded_double_quote_is_doubled(self) -> None:
        # FTS5 escape rule: a literal `"` inside `"..."` is written as `""`.
        result = escape_fts5_query('say "hello" world')
        assert result == '"say" """hello""" "world"'

    def test_fts5_operator_words_become_literal_phrases(self) -> None:
        # `AND`, `OR`, `NOT`, `NEAR` are FTS5 operators when bare; quoted, they
        # are literal terms. Quoting preserves "treat user input as words" intent.
        for op in ("AND", "OR", "NOT", "NEAR"):
            result = escape_fts5_query(f"foo {op} bar")
            assert result == f'"foo" "{op}" "bar"'

    def test_fts5_metacharacters_are_neutralized(self) -> None:
        # `*` is a prefix operator, `(` `)` group, `:` is column filter,
        # leading `-` is a NOT unary in some contexts. None of them leak.
        result = escape_fts5_query("c++ rust* (kernel) col:val -bad")
        assert result == '"c++" "rust*" "(kernel)" "col:val" "-bad"'

    def test_empty_input_returns_empty_string(self) -> None:
        assert escape_fts5_query("") == ""

    def test_whitespace_only_input_returns_empty_string(self) -> None:
        assert escape_fts5_query("   \t\n  ") == ""

    def test_unicode_preserved(self) -> None:
        # FTS5's unicode61/porter tokenizer handles unicode at index/query time.
        # The escape must not mangle multi-byte chars.
        result = escape_fts5_query("café résumé 北京")
        assert result == '"café" "résumé" "北京"'

    def test_single_token(self) -> None:
        assert escape_fts5_query("hello") == '"hello"'

    def test_leading_and_trailing_whitespace_collapsed(self) -> None:
        assert escape_fts5_query("  foo  bar  ") == '"foo" "bar"'

    def test_long_query_truncated_to_max_tokens(self) -> None:
        # Cap protects against SQLITE_MAX_EXPR_DEPTH on adversarial input.
        tokens = [f"t{i}" for i in range(200)]
        result = escape_fts5_query(" ".join(tokens))
        # Must have exactly 64 phrases (the documented cap).
        # Counting balanced "..." phrases by halving the quote count.
        assert result.count('"') == 64 * 2

    @pytest.mark.parametrize(
        "given",
        [
            'foo " bar',  # lone unmatched quote
            '"',  # just a quote
            '""',  # just two quotes
            'a"b',  # quote in the middle of a token
        ],
    )
    def test_quotes_in_unexpected_positions_are_escaped(self, given: str) -> None:
        # Each whitespace-token has any `"` doubled. Result must be
        # well-formed FTS5 (no unbalanced quotes).
        result = escape_fts5_query(given)
        assert result.count('"') % 2 == 0
