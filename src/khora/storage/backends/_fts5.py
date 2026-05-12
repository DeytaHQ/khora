"""Shared FTS5 query helpers for SQLite-based backends.

SQLite FTS5 parses MATCH operands as a *query expression*, not as literal
text. Punctuation (``?``, ``:``, ``.``, ``(``, ``*``, leading ``-``) and
operator words (``AND``, ``OR``, ``NEAR``) all have syntactic meaning. Binding
raw user input via ``? params`` does not protect the parser — only the SQL
layer is parameterised; FTS5 still parses the bound string.

This module exposes a single helper, :func:`escape_fts5_query`, used by:

* ``khora.engines.skeleton.backends.sqlite_lance._bm25_search``
* ``khora.storage.backends.sqlite_lance.vector.search_fulltext``
* ``khora.storage.backends.sqlite.search_fulltext``

Regression for https://github.com/DeytaHQ/khora/issues/526.
"""

from __future__ import annotations

import re

from loguru import logger

_TOKEN_RE = re.compile(r"\S+")

# Cap on the number of phrase terms we splice into a MATCH expression. SQLite's
# default ``SQLITE_MAX_EXPR_DEPTH`` is 1000; a chain of N implicit-AND'd phrases
# is depth N. Most user queries are <20 tokens. 64 is comfortably above that and
# far below the parser's depth limit. Excess tokens are dropped silently — long
# queries usually contain enough informative head terms that recall is preserved.
_MAX_TOKENS = 64


def escape_fts5_query(query_text: str) -> str:
    """Convert free-form text into a safe FTS5 ``MATCH`` expression.

    Tokenises on whitespace, wraps each token in a quoted FTS5 phrase, doubles
    any embedded ``"``, and joins with spaces (implicit AND). Result preserves
    bag-of-words BM25 recall semantics while neutralising operator characters.

    The porter / unicode61 tokenisers strip non-alphanumeric characters at both
    index and query time, so a token like ``"win?"`` matches the indexed token
    ``win`` with no recall loss.

    Returns ``""`` for empty or whitespace-only input (and for input that
    reduces to only operator-punctuation after tokenisation never happens here
    because we don't strip; callers should treat an empty return as "skip the
    FTS query, fall back to vector-only" rather than send empty MATCH).

    Example: ``"What did Curie win?"`` becomes ``"What" "did" "Curie" "win?"``.
    Embedded quotes are doubled: ``'say "hello"'`` becomes
    ``"say" \\"\\"\\"hello\\"\\"\\"``. Whitespace-only input returns ``""``.
    """
    tokens = _TOKEN_RE.findall(query_text)
    if not tokens:
        logger.debug("FTS5 escape produced empty match expression (input was whitespace only)")
        return ""
    if len(tokens) > _MAX_TOKENS:
        logger.debug(
            "FTS5 query has {} tokens; truncating to first {} to stay below SQLITE_MAX_EXPR_DEPTH",
            len(tokens),
            _MAX_TOKENS,
        )
        tokens = tokens[:_MAX_TOKENS]
    return " ".join('"' + t.replace('"', '""') + '"' for t in tokens)
