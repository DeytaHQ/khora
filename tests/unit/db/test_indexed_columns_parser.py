"""Unit coverage for ``indexed_columns``, the shared index-DDL key parser.

Both migration-057 lanes compare an index's key list against an expected one,
and the SQLite lane maps this parser over EVERY index on ``documents`` rather
than a hand-picked one. A wrong parse does not raise — it returns a key list
that silently fails to match — so the parser needs its own tests rather than
being exercised only through the one index those lanes assert on.

The shapes below are grouped by whether they exist in this schema today:

* **live shapes** — spellings the chain actually produces, on either dialect;
* **guarded shapes** — the quoted-parenthesis and quoted-comma cases, which no
  index in this chain currently produces. They are tested because the failure
  mode is silent: a naive depth count treats the ``)`` in ``COALESCE(a, ')')``
  as structural, closes the key list early, and drops every key after it.

``test_no_index_in_the_chain_relies_on_quote_handling`` pins that "no index
currently quotes anything" claim against the real chain rather than leaving it
as a comment that rots the first time someone adds a partial index.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from alembic import command

from tests.test_helpers.documents_created_at_index import indexed_columns, make_config


class TestIndexedColumnsLiveShapes:
    """Spellings the chain produces today, on one dialect or the other."""

    @pytest.mark.parametrize(
        ("sql", "expected"),
        [
            # SQLite's sqlite_master.sql spelling.
            (
                "CREATE INDEX ix_documents_created_at ON documents (created_at)",
                ["created_at"],
            ),
            # Postgres' pg_indexes.indexdef spelling, with the USING clause.
            (
                "CREATE INDEX ix_documents_created_at ON public.documents USING btree (created_at)",
                ["created_at"],
            ),
            (
                "CREATE INDEX ix_d ON documents (namespace_id, created_at, id)",
                ["namespace_id", "created_at", "id"],
            ),
            # DESC key - the parser keeps the modifier, which is what makes a
            # DESC rebuild distinguishable from an ASC one.
            (
                "CREATE INDEX ix_d ON documents (created_at DESC)",
                ["created_at DESC"],
            ),
            # Expression key: the final ")" is not the end of the key list.
            # This is ix_chunks_ns_temporal's real shape (migration 017).
            (
                "CREATE INDEX ix_chunks_ns_temporal ON chunks (namespace_id, COALESCE(source_timestamp, created_at))",
                ["namespace_id", "COALESCE(source_timestamp, created_at)"],
            ),
            # Partial index: rindex(")") would reach past the key list into
            # the predicate. Both live partial predicates on documents are
            # this shape.
            (
                "CREATE UNIQUE INDEX ix_d ON documents (namespace_id, external_id) WHERE external_id IS NOT NULL",
                ["namespace_id", "external_id"],
            ),
            (
                "CREATE INDEX ix_d ON documents (namespace_id, checksum) WHERE status != 'failed'",
                ["namespace_id", "checksum"],
            ),
            # Partial predicate that itself contains parentheses.
            (
                "CREATE INDEX ix_d ON documents (a) WHERE lower(b) IS NOT NULL",
                ["a"],
            ),
        ],
    )
    def test_parses_live_shapes(self, sql: str, expected: list[str]) -> None:
        assert indexed_columns(sql) == expected


class TestIndexedColumnsQuotedRegions:
    """Quoted parentheses and commas - guarded, not currently produced.

    Each of these silently truncates under a quote-blind depth count, which is
    why they are asserted rather than reasoned about.
    """

    @pytest.mark.parametrize(
        ("sql", "expected"),
        [
            # The case that motivated the guard: a ")" inside a string literal
            # closes the key list early under a naive count, dropping
            # created_at entirely.
            (
                "CREATE INDEX ix_d ON documents (COALESCE(label, ')'), created_at)",
                ["COALESCE(label, ')')", "created_at"],
            ),
            # A comma inside a literal must not split a key.
            (
                "CREATE INDEX ix_d ON documents (COALESCE(label, 'a,b'), created_at)",
                ["COALESCE(label, 'a,b')", "created_at"],
            ),
            # An "(" inside a literal must not inflate depth - if it did, the
            # group would never close and the parser would return [].
            (
                "CREATE INDEX ix_d ON documents (COALESCE(label, '('), created_at)",
                ["COALESCE(label, '(')", "created_at"],
            ),
            # Quoted IDENTIFIER containing a parenthesis, before the key list.
            (
                'CREATE INDEX "weird (name)" ON documents (created_at)',
                ["created_at"],
            ),
            # Quoted identifier as a key.
            (
                'CREATE INDEX ix_d ON documents ("created_at")',
                ['"created_at"'],
            ),
            # Doubled quote is an escaped quote, not a terminator: the region
            # continues, so the ")" after it is still data.
            (
                "CREATE INDEX ix_d ON documents (COALESCE(label, 'it''s )'), created_at)",
                ["COALESCE(label, 'it''s )')", "created_at"],
            ),
        ],
    )
    def test_quoted_regions_are_not_structural(self, sql: str, expected: list[str]) -> None:
        assert indexed_columns(sql) == expected

    def test_naive_depth_count_would_truncate(self) -> None:
        """Pin that the quoted case is a real trap, not a hypothetical one.

        Reproduces the pre-fix algorithm inline and shows it disagreeing. If a
        future refactor reverts to quote-blind counting, the parametrized case
        above fails and this explains why.
        """
        sql = "CREATE INDEX ix_d ON documents (COALESCE(label, ')'), created_at)"

        start = sql.find("(")
        depth = 0
        end = -1
        for position in range(start, len(sql)):
            if sql[position] == "(":
                depth += 1
            elif sql[position] == ")":
                depth -= 1
                if depth == 0:
                    end = position
                    break

        naive = sql[start + 1 : end]
        assert "created_at" not in naive, "the naive parse was expected to truncate before created_at"
        assert indexed_columns(sql)[-1] == "created_at"


class TestIndexedColumnsDegenerateInput:
    @pytest.mark.parametrize(
        "sql",
        [
            None,  # sqlite_autoindex_* carries NULL sql
            "",
            "CREATE INDEX ix_d ON documents",  # no group at all
            "CREATE INDEX ix_d ON documents (created_at",  # unbalanced
            "CREATE INDEX ix_d ON documents ('unterminated",  # unterminated quote
        ],
    )
    def test_returns_empty_rather_than_raising(self, sql: str | None) -> None:
        assert indexed_columns(sql) == []


class TestIndexedColumnsAgainstTheRealChain:
    """The claim "no index in this chain quotes anything" pinned to the chain.

    The parser's docstring says quote handling guards a future index rather
    than a present one. That is only checkable against the built schema, and
    it is exactly the kind of statement that silently stops being true.
    """

    def test_no_index_in_the_chain_relies_on_quote_handling(self, tmp_path: Path) -> None:
        db_path = tmp_path / "chain.db"
        command.upgrade(make_config(f"sqlite:///{db_path}"), "head")

        connection = sqlite3.connect(db_path)
        try:
            rows = connection.execute(
                "SELECT name, sql FROM sqlite_master WHERE type = 'index' AND sql IS NOT NULL"
            ).fetchall()
        finally:
            connection.close()

        assert rows, "no index DDL was reflected - the chain did not build"

        # Every index in the chain must parse to a non-empty key list. This is
        # the assertion that matters: it is what the 057 lanes depend on.
        for name, sql in rows:
            assert indexed_columns(sql), f"{name} parsed to an empty key list: {sql}"

        # And the documentation guard: no KEY currently contains a quoted
        # region. Quoting is supported, so this failing is not a bug - it means
        # the parser docstring's "no index in this chain quotes anything" has
        # gone stale and should be reworded.
        for name, sql in rows:
            keys = "".join(indexed_columns(sql))
            assert "'" not in keys and '"' not in keys, (
                f"{name} now quotes something inside its key list: {sql}. "
                "The parser handles this correctly; update the claim in "
                "indexed_columns' docstring that no index in this chain quotes anything."
            )
