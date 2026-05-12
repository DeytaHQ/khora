"""Property-based fuzz of ``escape_fts5_query``.

The fix for issue #526 (PR #528) introduced
``khora.storage.backends._fts5.escape_fts5_query``. SQLite FTS5 has a
rich query expression syntax with magic characters (``"``, ``*``, ``-``,
``:``, ``(``, ``)``, ``AND``, ``OR``, ``NEAR``, ``^``). Any helper that
neutralises it MUST hold three invariants:

1. **Double-escape safety**: applying ``escape`` to an already-escaped
   string still produces parseable FTS5. Strict idempotence does NOT
   hold (a quoting-based escape inherently nests on second application),
   but the double-escaped output must remain a valid FTS5 expression so
   accidentally calling the helper twice can't introduce a syntax error.
2. **Parseability**: for ANY input ``x``, SQLite must accept
   ``MATCH escape(x)`` without raising a syntax error. The whole point of
   the helper.
3. **Bounded token count**: large input shouldn't blow up the FTS5
   expression depth (the helper caps tokens internally).

Hypothesis fuzzes the input space; SQLite FTS5 acts as the oracle for
parseability — we hit an in-memory FTS5 table per case to catch any
escape-helper regression that produces syntactically invalid output.
"""

from __future__ import annotations

import sqlite3

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from khora.storage.backends._fts5 import escape_fts5_query


@pytest.fixture(scope="module")
def fts5_conn():
    """One in-memory FTS5 table for the whole module — reused across cases."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(content, tokenize='porter')")
    conn.execute("INSERT INTO chunks_fts (content) VALUES ('alpha beta gamma delta')")
    conn.commit()
    yield conn
    conn.close()


# Characters that exercise every FTS5 magic-character class. Hypothesis's
# default text strategy is too narrow on punctuation, so we widen it.
_FUZZ_ALPHABET = st.characters(
    blacklist_categories=("Cs",),  # no surrogates
    blacklist_characters="\x00",  # SQLite chokes on embedded NULs in text — exclude
)


# ---------------------------------------------------------------------------
# Property 1: double-escape safety
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEscapeDoubleSafety:
    @given(text=st.text(alphabet=_FUZZ_ALPHABET, max_size=200))
    @settings(
        max_examples=300,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_double_escape_remains_parseable(self, text: str, fts5_conn) -> None:
        """Applying escape() twice must still produce a valid FTS5 MATCH expression.

        Strict idempotence (``escape(escape(x)) == escape(x)``) does NOT hold
        because the helper quotes every whitespace token and doubles embedded
        quotes — re-applying it nests another quoting layer. That's fine as
        long as the doubly-escaped output is still syntactically valid FTS5.
        This rules out the only practically dangerous regression: a defensive
        "let me escape this defensively just in case" wrapper somewhere
        causing a MATCH-time crash.
        """
        once = escape_fts5_query(text)
        twice = escape_fts5_query(once)
        if not twice:
            return
        try:
            cur = fts5_conn.execute(
                "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?",
                (twice,),
            )
            cur.fetchall()
        except sqlite3.OperationalError as exc:
            raise AssertionError(
                f"double-escape produced invalid FTS5 for {text!r}:\n"
                f"  once  = {once!r}\n"
                f"  twice = {twice!r}\n  error = {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# Property 2: parseability against a real FTS5 table
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEscapeParseability:
    @given(text=st.text(alphabet=_FUZZ_ALPHABET, max_size=200))
    @settings(
        max_examples=500,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_escape_output_parses_in_fts5(self, text: str, fts5_conn) -> None:
        """For ANY input, the escaped string MUST be a valid FTS5 MATCH expression.

        Run ``SELECT … WHERE chunks_fts MATCH ?`` against an in-memory FTS5
        table. The query may return zero rows — that's fine. The property
        is "no sqlite3.OperationalError raised".
        """
        escaped = escape_fts5_query(text)
        if not escaped:
            # Empty escape — caller is expected to skip the query entirely
            # rather than send empty MATCH. Don't probe SQLite (it would
            # raise on the empty string, which is documented and expected).
            return
        try:
            cur = fts5_conn.execute(
                "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?",
                (escaped,),
            )
            cur.fetchall()
        except sqlite3.OperationalError as exc:
            raise AssertionError(
                f"escape_fts5_query produced an unparseable expression for {text!r}:\n"
                f"  escaped = {escaped!r}\n  error  = {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# Property 3: bounded output (token cap protects SQLITE_MAX_EXPR_DEPTH)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEscapeTokenCap:
    @given(token_count=st.integers(min_value=0, max_value=500))
    @settings(max_examples=50, deadline=None)
    def test_output_token_count_is_capped(self, token_count: int) -> None:
        """The helper caps tokens internally so a 10k-word adversarial input
        can't blow up SQLITE_MAX_EXPR_DEPTH (default 1000)."""
        tokens = " ".join(f"word{i}" for i in range(token_count))
        escaped = escape_fts5_query(tokens)
        # Each token becomes "..." in the escaped output → quote count is
        # 2 × actual_token_count. Cap is 64 per the helper.
        quote_count = escaped.count('"')
        assert quote_count <= 64 * 2, (
            f"escape did not cap tokens: input had {token_count} words, output has {quote_count // 2} quoted phrases"
        )


# ---------------------------------------------------------------------------
# Property 4: known-good fixed-input examples (regression for #526 exact cases)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEscapeFixedExamples:
    @pytest.mark.parametrize(
        "query",
        [
            "What did Curie win?",  # the issue #526 repro
            "Curie: Nobel",
            "Curie (Nobel)",
            "Curie AND Physics",
            'say "hello" Curie',
            "Curie*",
            "-Curie",
            "Curie NEAR Nobel",
            "^Curie",
            "",
            "   ",
            "\t\n",
        ],
    )
    def test_known_inputs_dont_crash_fts5(self, query: str, fts5_conn) -> None:
        escaped = escape_fts5_query(query)
        if not escaped:
            return  # short-circuit path; caller skips MATCH entirely
        cur = fts5_conn.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?",
            (escaped,),
        )
        cur.fetchall()
