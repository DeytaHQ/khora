"""SQLite build-capability probes shared by the SQLite-family backends — ``@internal``.

Both SQLite relational stores (the raw-SQL backend and the SQLAlchemy-over-
aiosqlite ``sqlite_lance`` adapter) need the same answer to the same question:
does *this process's* SQLite build have the JSON1 functions? The probe lives
here rather than being duplicated per store because it interrogates the
interpreter's stdlib ``sqlite3`` library, not either store's schema — the two
stores genuinely diverge on their DDL (they disagree on the physical metadata
column name), but they cannot disagree on this.
"""

from __future__ import annotations

import sqlite3
from functools import lru_cache

__all__ = ["sqlite_has_json1"]


@lru_cache(maxsize=1)
def sqlite_has_json1() -> bool:
    """Whether this process's SQLite build has the JSON1 functions.

    ``compile_lance`` gates all metadata pushdown on this (``json_extract`` /
    ``json_type`` / ``json_each``). ``json_valid`` is probed as the proxy: it is
    a scalar function from the same JSON1 extension, so it is available exactly
    when the others are, and unlike ``json_each`` — a table-valued function — it
    can be tested with a bare ``SELECT``.

    Probed once per process (memoized) against a throwaway in-memory database.
    aiosqlite and SQLAlchemy's ``sqlite+aiosqlite`` driver both run on the same
    in-process ``sqlite3`` library, so one answer serves both stores.

    Returns ``False`` on any :class:`sqlite3.Error`, including a failure to open
    the in-memory database — the conservative direction, since ``False`` only
    routes metadata leaves to the caller's post-filter and never produces a
    wrong row-set.
    """
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(":memory:")
        return conn.execute("SELECT json_valid('{}')").fetchone()[0] == 1
    except sqlite3.Error:
        return False
    finally:
        if conn is not None:
            conn.close()
