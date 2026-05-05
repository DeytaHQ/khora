import pytest

from khora.storage._log_safe import _safe_url_for_log


@pytest.mark.parametrize(
    "url, expected",
    [
        # (a) URL with user + password
        (
            "postgresql+asyncpg://peras:ebesspyg7jl0kwkbglto@pgbouncer.example.net/peras",
            "postgresql+asyncpg://<redacted>@pgbouncer.example.net/peras",
        ),
        # (b) URL with user only (no password)
        (
            "bolt://admin@neo4j.example.com:7687",
            "bolt://<redacted>@neo4j.example.com:7687",
        ),
        # (c) URL with neither user nor password
        (
            "postgresql+asyncpg://pgbouncer.example.net/peras",
            "postgresql+asyncpg://pgbouncer.example.net/peras",
        ),
        # (d) URL with non-default port
        (
            "bolt://user:secret@neo4j.example.com:7688",
            "bolt://<redacted>@neo4j.example.com:7688",
        ),
        # (e) sqlite-style file: URL with no netloc — passthrough
        (
            "sqlite+aiosqlite:///path/to/khora.db",
            "sqlite+aiosqlite:///path/to/khora.db",
        ),
        # file: URL passthrough
        (
            "file:///tmp/mydb",
            "file:///tmp/mydb",
        ),
    ],
)
def test_safe_url_for_log(url: str, expected: str) -> None:
    assert _safe_url_for_log(url) == expected
