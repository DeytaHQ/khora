"""Secret-redaction helpers for log / exception output."""

from __future__ import annotations

import re

# Matches the userinfo segment of a URI: ``://user:password@``. Captures
# nothing — substitution replaces the whole match with ``://[REDACTED]@``.
_DSN_USERINFO_RE = re.compile(r"://[^:@/]+:[^@/]+@")


def redact_dsn(text: str) -> str:
    """Return ``text`` with any DSN userinfo replaced by ``[REDACTED]``.

    The pattern matches ``scheme://user:password@`` segments and replaces the
    ``user:password`` part. Strings without an embedded DSN are returned
    unchanged.

    >>> redact_dsn("postgresql://alice:hunter2@db:5432/app")
    'postgresql://[REDACTED]@db:5432/app'
    >>> redact_dsn("no secret here")
    'no secret here'
    """
    if not text:
        return text
    return _DSN_USERINFO_RE.sub("://[REDACTED]@", text)


__all__ = ["redact_dsn"]
