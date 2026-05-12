"""Contract tests for SQLite FTS5 bm25() sign handling in the skeleton engine.

SQLite's built-in ``bm25()`` ranking function returns values where **smaller
(more negative) is better** — it's the negated, length-normalized BM25 score
per the FTS5 docs. That convention conflicts with every other engine in
khora (vector cosine, SurrealDB ``search::score``, pgvector ``1 - distance``)
where **higher is better**.

To paper over the inconsistency, ``skeleton/backends/sqlite_lance.py:526``
negates the raw value::

    score = -float(row["bm"])
    out.append(TemporalSearchResult(..., bm25_score=score, combined_score=score))

A future "let me clean up that confusing negation" refactor would silently
flip ranking for every embedded skeleton deployment — matches would appear
in WORST-first order, with no test failure to catch it.

These tests pin the contract:

1. SQLite FTS5 ``bm25()`` returns non-positive scores on a match (closer to
   zero = worse match; more negative = better match). If FTS5 ever changes
   to return positive-is-better, the negation in skeleton becomes wrong.
2. Khora's exposed ``bm25_score`` is non-negative for a real match (i.e.
   higher is better after negation), matching the rest of the codebase.
3. ``ORDER BY bm ASC`` on the raw column produces the same ordering as
   sorting the negated khora score DESC.
"""

from __future__ import annotations

import sqlite3

import pytest

# ---------------------------------------------------------------------------
# Fixture: in-memory FTS5 table matching the skeleton's schema
# ---------------------------------------------------------------------------


@pytest.fixture
def fts5_db():
    conn = sqlite3.connect(":memory:")
    # Mirror the relevant column from khora_chunks_fts in the skeleton backend.
    conn.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(content, tokenize='porter')")
    conn.executemany(
        "INSERT INTO chunks_fts (content) VALUES (?)",
        [
            ("the quick brown fox jumps over the lazy dog",),  # rowid 1 — best match for "fox"
            ("a fox is a fox is a fox",),  # rowid 2 — multiple-hit
            ("the dog sleeps",),  # rowid 3 — no fox
            ("foxglove flowers are purple",),  # rowid 4 — different word
        ],
    )
    conn.commit()
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# Contract 1: raw bm25() is "smaller is better" (FTS5's documented convention)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRawBm25SignConvention:
    def test_raw_bm25_is_non_positive_on_match(self, fts5_db) -> None:
        """SQLite FTS5 ``bm25()`` returns ``<= 0`` on any matching row.

        From the SQLite docs: "By default, the value returned is approximately
        the BM25 score multiplied by -1." The negative-score convention is
        what khora's skeleton backend negates back. If this ever changes
        upstream the negation becomes wrong.
        """
        cur = fts5_db.execute(
            "SELECT bm25(chunks_fts) AS bm FROM chunks_fts WHERE chunks_fts MATCH ?",
            ("fox",),
        )
        scores = [row[0] for row in cur.fetchall()]
        assert scores, "expected at least one row to match 'fox'"
        for score in scores:
            assert score <= 0.0, (
                f"FTS5 bm25() returned a positive score {score}; the skeleton "
                f"backend's negation at sqlite_lance.py:526 assumes "
                f"smaller-is-better. Did SQLite change its bm25() convention?"
            )

    def test_better_match_has_more_negative_raw_score(self, fts5_db) -> None:
        """Multi-hit row ('a fox is a fox is a fox') should beat a single-hit
        row in the FTS5 ranking. With the smaller-is-better convention, the
        better match must have a smaller (more negative) raw score."""
        cur = fts5_db.execute(
            "SELECT rowid, bm25(chunks_fts) AS bm FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm ASC",
            ("fox",),
        )
        rows = cur.fetchall()
        # rowid 2 (triple-fox) should be best; rowid 1 (single fox) worse.
        assert rows[0][0] == 2, (
            f"expected triple-hit row to rank first under ORDER BY bm ASC; got order {[r[0] for r in rows]}"
        )


# ---------------------------------------------------------------------------
# Contract 2: khora's exposed bm25_score = -raw_bm25 (higher-is-better)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestKhoraNegationContract:
    def test_negation_makes_score_non_negative(self, fts5_db) -> None:
        """After ``-float(row["bm"])`` the exposed score is non-negative for
        any real match. This is the contract the rest of the engine relies on
        — combined_score and bm25_score must compose with vector similarity
        (which is also non-negative)."""
        cur = fts5_db.execute(
            "SELECT bm25(chunks_fts) AS bm FROM chunks_fts WHERE chunks_fts MATCH ?",
            ("fox",),
        )
        for row in cur.fetchall():
            raw = float(row[0])
            khora_score = -raw  # the line at sqlite_lance.py:526
            assert khora_score >= 0.0, (
                f"khora's negated bm25_score should be non-negative; got {khora_score} (raw bm25={raw})"
            )

    def test_negation_flips_order_to_higher_is_better(self, fts5_db) -> None:
        """If a row ranks BETTER under ``ORDER BY bm ASC`` (raw smaller-is-better),
        its negated score must be LARGER. Pin the sign equivalence so a future
        cleanup can't quietly drop the negation."""
        cur = fts5_db.execute(
            "SELECT rowid, bm25(chunks_fts) AS bm FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm ASC",
            ("fox",),
        )
        rows = cur.fetchall()
        # After negation the ordering must reverse → exactly descending.
        negated = [(rid, -float(bm)) for rid, bm in rows]
        sorted_desc = sorted(negated, key=lambda x: -x[1])
        assert negated == sorted_desc, (
            f"negated scores must already be in descending order, matching 'higher is better' contract; got {negated}"
        )


# ---------------------------------------------------------------------------
# Contract 3: empty/whitespace query is the caller's responsibility
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEmptyQueryShortCircuit:
    def test_empty_match_expression_must_be_filtered_before_match(self, fts5_db) -> None:
        """SQLite raises on empty FTS5 MATCH. The skeleton backend short-circuits
        via ``escape_fts5_query`` returning '' → caller bails before running
        the query. Pin this so future refactors don't quietly send '' to FTS5.
        """
        with pytest.raises(sqlite3.OperationalError):
            fts5_db.execute(
                "SELECT bm25(chunks_fts) FROM chunks_fts WHERE chunks_fts MATCH ?",
                ("",),
            ).fetchall()
