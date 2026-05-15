"""Secret-redaction helpers and secret-field annotation types for log / exception output."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Matches the userinfo segment of a URI. Captures nothing — substitution
# replaces the whole match with ``://[REDACTED]@``.
# Two alternatives cover both credential forms:
#   1. ``[^:@]*:[^@]+`` — user:password (empty user allowed, e.g. redis://:pw@)
#   2. ``[^:@/?#]+``     — user only (no password — token/SASL DSNs where the
#                          username itself is the credential, e.g. scheme://svc@host).
#                          Excludes ``/``, ``?``, ``#`` so the alternative can only
#                          match the URI authority's userinfo segment and will not
#                          consume host/path text to reach a later ``@`` in a path
#                          or query (e.g. ``https://example.com/@alice``).
# Password class ``[^@]+`` allows ``/`` so passwords like ``pass/word`` are
# fully redacted rather than truncated at the first slash.
_DSN_USERINFO_RE = re.compile(r"://(?:[^:@]*:[^@]+|[^:@/?#]+)@")


@dataclass(frozen=True)
class AllowSecretTyping:
    """Annotation marker for fields that intentionally remain plain ``str``.

    Use as ``Annotated[str, AllowSecretTyping(reason="...")]`` on a
    secret-named field that intentionally stays as plain ``str`` (e.g.,
    env-var name pointers or legacy factory intermediaries that hold
    post-boundary unwrapped values). The semgrep rule excludes fields
    bearing this annotation from the secret-typing lint rule.
    """

    reason: str


def redact_dsn(text: str) -> str:
    """Return ``text`` with any DSN userinfo replaced by ``[REDACTED]``.

    Handles both ``scheme://user:password@`` and password-less
    ``scheme://user@`` forms (token/SASL DSNs where the username is the
    credential). Strings without an embedded DSN are returned unchanged.

    >>> redact_dsn("postgresql://alice:hunter2@db:5432/app")
    'postgresql://[REDACTED]@db:5432/app'
    >>> redact_dsn("postgresql://serviceuser@host/db")
    'postgresql://[REDACTED]@host/db'
    >>> redact_dsn("no secret here")
    'no secret here'
    """
    if not text:
        return text
    return _DSN_USERINFO_RE.sub("://[REDACTED]@", text)


__all__ = ["AllowSecretTyping", "redact_dsn"]
