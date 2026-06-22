"""Unit tests for ``khora.storage.backends._fts5.escape_fts5_query``.

Regression for https://github.com/DeytaHQ/khora/issues/526 — natural-language
``recall()`` queries crashed SQLite FTS5 with ``syntax error near "?"``.

Tokens are OR-joined (not implicit-AND space-joined) since
https://github.com/DeytaHQ/khora/issues/1330 — a natural-language query that
embeds an exact ID must still match the chunk holding that ID.
"""

from __future__ import annotations

import sqlite3

import pytest

from khora.storage.backends._fts5 import escape_fts5_query


class TestEscapeFTS5Query:
    def test_punctuation_in_query_is_safe(self) -> None:
        # The exact input that crashed in issue #526. Tokens are now OR-joined
        # (#1330) so a sentence does not require every word to co-occur.
        result = escape_fts5_query("What did Curie win?")
        assert result == '"What" OR "did" OR "Curie" OR "win?"'

    def test_each_punctuation_class(self) -> None:
        # `?`, `!`, `.`, `,`, `@`, `:` were the symptoms in the bug report.
        result = escape_fts5_query("hi! @user, what's up: doc. ok?")
        assert result == '"hi!" OR "@user," OR "what\'s" OR "up:" OR "doc." OR "ok?"'

    def test_embedded_double_quote_is_doubled(self) -> None:
        # FTS5 escape rule: a literal `"` inside `"..."` is written as `""`.
        result = escape_fts5_query('say "hello" world')
        assert result == '"say" OR """hello""" OR "world"'

    def test_fts5_operator_words_become_literal_phrases(self) -> None:
        # `AND`, `OR`, `NOT`, `NEAR` are FTS5 operators when bare; quoted, they
        # are literal terms. Quoting preserves "treat user input as words" intent.
        for op in ("AND", "OR", "NOT", "NEAR"):
            result = escape_fts5_query(f"foo {op} bar")
            assert result == f'"foo" OR "{op}" OR "bar"'

    def test_fts5_metacharacters_are_neutralized(self) -> None:
        # `*` is a prefix operator, `(` `)` group, `:` is column filter,
        # leading `-` is a NOT unary in some contexts. None of them leak.
        result = escape_fts5_query("c++ rust* (kernel) col:val -bad")
        assert result == '"c++" OR "rust*" OR "(kernel)" OR "col:val" OR "-bad"'

    def test_empty_input_returns_empty_string(self) -> None:
        assert escape_fts5_query("") == ""

    def test_whitespace_only_input_returns_empty_string(self) -> None:
        assert escape_fts5_query("   \t\n  ") == ""

    def test_unicode_preserved(self) -> None:
        # FTS5's unicode61/porter tokenizer handles unicode at index/query time.
        # The escape must not mangle multi-byte chars.
        result = escape_fts5_query("café résumé 北京")
        assert result == '"café" OR "résumé" OR "北京"'

    def test_single_token(self) -> None:
        assert escape_fts5_query("hello") == '"hello"'

    def test_leading_and_trailing_whitespace_collapsed(self) -> None:
        assert escape_fts5_query("  foo  bar  ") == '"foo" OR "bar"'

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

    def test_all_stopword_query_is_not_emptied(self) -> None:
        # #1330 dropped stopword-pruning: domain words like "status"/"plan" are
        # meaningful. A non-empty token set must NEVER reduce to "" — that would
        # silently disable the lexical channel for whole-sentence queries.
        result = escape_fts5_query("what is the status of")
        assert result != ""
        assert result == '"what" OR "is" OR "the" OR "status" OR "of"'


@pytest.fixture()
def fts5_conn():
    """In-memory FTS5 table seeded with the issue #1330 ticket corpus."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE tickets USING fts5(content, tokenize='porter unicode61')")
    conn.executemany(
        "INSERT INTO tickets (content) VALUES (?)",
        [
            ("Support ticket MER-0001 - TechWave Solutions. Status: open. Failed data ingestion.",),
            ("Support ticket MER-0002 - GreenField Agriculture. Status: pending. Billing discrepancy.",),
            ("Support ticket MER-0003 - Fintech Innovations. Status: resolved. Token expiry on mobile.",),
        ],
    )
    conn.commit()
    yield conn
    conn.close()


def _match_rowids(conn: sqlite3.Connection, match_expr: str) -> list[int]:
    """Rowids matching ``match_expr``, ordered by bm25() ascending (best first)."""
    cur = conn.execute(
        "SELECT rowid FROM tickets WHERE tickets MATCH ? ORDER BY bm25(tickets) ASC",
        (match_expr,),
    )
    return [r[0] for r in cur.fetchall()]


class TestEscapeFTS5MatchSemantics:
    """OR-join behavioral tests against a real FTS5 index (#1330)."""

    def test_natural_language_id_query_matches_under_or(self, fts5_conn) -> None:
        # The issue's repro shape: a sentence embedding an exact ID. Under the
        # old implicit-AND join this returned ZERO rows (no chunk has every
        # word). Under OR + bm25 ranking the MER-0001 chunk is found and ranks
        # #1 (it carries the rare high-signal token).
        match_expr = escape_fts5_query("What is the status of ticket MER-0001?")
        rowids = _match_rowids(fts5_conn, match_expr)
        assert rowids, "OR join must yield > 0 rows for the NL-ID query"
        assert rowids[0] == 1, "the MER-0001 chunk must rank #1"

    def test_bare_id_query_still_matches(self, fts5_conn) -> None:
        # The bare-ID lookup that already worked must stay green.
        match_expr = escape_fts5_query("MER-0001")
        rowids = _match_rowids(fts5_conn, match_expr)
        assert rowids == [1]

    def test_precision_guard_and_perfect_hit_ranks_first(self, fts5_conn) -> None:
        # A query whose terms genuinely co-occur in exactly one chunk must still
        # rank that chunk #1 under OR+bm25 — proves OR didn't demote the
        # AND-perfect hit. "billing discrepancy" co-occurs only in MER-0002.
        match_expr = escape_fts5_query("billing discrepancy")
        rowids = _match_rowids(fts5_conn, match_expr)
        assert rowids[0] == 2, f"AND-perfect co-occurrence chunk must rank #1, got {rowids}"
