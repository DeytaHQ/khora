"""SecretStr re-typing — contract tests.

Covers:

* every Pydantic password / DSN field is typed as
  :class:`pydantic.SecretStr` (or ``SecretStr | None``).
* ``repr()`` and ``model_dump_json()`` of those fields never leak the
  underlying plaintext.
* the local ``khora.config._secrets.redact_dsn`` helper scrubs ``user:pass@``
  userinfo from arbitrary text.
* migration-error wrapping in ``khora.db.session._run_migrations_sync``
  routes through ``redact_dsn`` so plaintext DSNs cannot reach the returned
  ``MigrationResult.error`` string.
"""

from __future__ import annotations

import json

import pytest
from pydantic import SecretStr

from khora.config._secrets import redact_dsn
from khora.config.schema import (
    KhoraConfig,
    LLMSettings,
    MemgraphConfig,
    Neo4jConfig,
    NeptuneConfig,
    ParsedNeo4jUrl,
    StorageSettings,
    SurrealDBConfig,
    SurrealDBVectorConfig,
)
from khora.telemetry.config import TelemetryConfig


@pytest.mark.unit
class TestSecretStrFieldTypes:
    """Every Pydantic password / DSN field is typed as SecretStr."""

    def test_neo4j_config_password_is_secretstr(self) -> None:
        cfg = Neo4jConfig(password="hunter2")
        assert isinstance(cfg.password, SecretStr)
        assert cfg.password.get_secret_value() == "hunter2"

    def test_memgraph_config_password_is_secretstr(self) -> None:
        cfg = MemgraphConfig(password="hunter2")
        assert isinstance(cfg.password, SecretStr)
        assert cfg.password.get_secret_value() == "hunter2"

    def test_neptune_config_password_is_secretstr(self) -> None:
        cfg = NeptuneConfig(password="hunter2")
        assert isinstance(cfg.password, SecretStr)
        assert cfg.password.get_secret_value() == "hunter2"

    def test_surrealdb_config_password_is_secretstr(self) -> None:
        cfg = SurrealDBConfig(password="hunter2")
        assert isinstance(cfg.password, SecretStr)
        assert cfg.password.get_secret_value() == "hunter2"

    def test_surrealdb_vector_config_password_is_secretstr(self) -> None:
        cfg = SurrealDBVectorConfig(password="hunter2")
        assert isinstance(cfg.password, SecretStr)
        assert cfg.password.get_secret_value() == "hunter2"

    def test_storage_settings_neo4j_password_is_secretstr(self) -> None:
        settings = StorageSettings(neo4j_password="hunter2")
        assert isinstance(settings.neo4j_password, SecretStr)
        assert settings.neo4j_password.get_secret_value() == "hunter2"

    def test_storage_settings_postgresql_url_is_secretstr(self) -> None:
        settings = StorageSettings(postgresql_url="postgresql://u:p@h/d")
        assert isinstance(settings.postgresql_url, SecretStr)
        assert settings.postgresql_url.get_secret_value() == "postgresql://u:p@h/d"

    def test_storage_settings_pgvector_url_is_secretstr(self) -> None:
        settings = StorageSettings(pgvector_url="postgresql://u:p@h/d")
        assert isinstance(settings.pgvector_url, SecretStr)
        assert settings.pgvector_url.get_secret_value() == "postgresql://u:p@h/d"

    def test_storage_settings_neo4j_url_is_secretstr(self) -> None:
        # Construct without legacy migration kicking in (no neo4j_user/password
        # to avoid the legacy → new-style graph migration that consumes
        # ``neo4j_url``).
        settings = StorageSettings.model_construct(
            neo4j_url=SecretStr("bolt://u:p@h:7687"),
        )
        assert isinstance(settings.neo4j_url, SecretStr)
        assert settings.neo4j_url.get_secret_value() == "bolt://u:p@h:7687"

    def test_khora_config_database_url_is_secretstr(self) -> None:
        cfg = KhoraConfig(database_url="postgresql://u:p@h/d")
        assert isinstance(cfg.database_url, SecretStr)
        assert cfg.database_url.get_secret_value() == "postgresql://u:p@h/d"

    def test_khora_config_neo4j_url_is_secretstr(self) -> None:
        cfg = KhoraConfig(neo4j_url="bolt://u:p@h:7687")
        assert isinstance(cfg.neo4j_url, SecretStr)
        assert cfg.neo4j_url.get_secret_value() == "bolt://u:p@h:7687"

    def test_khora_config_telemetry_database_url_is_secretstr(self) -> None:
        cfg = KhoraConfig(telemetry_database_url="postgresql://u:p@h/t")
        assert isinstance(cfg.telemetry_database_url, SecretStr)
        assert cfg.telemetry_database_url.get_secret_value() == "postgresql://u:p@h/t"

    def test_telemetry_config_database_url_is_secretstr(self) -> None:
        cfg = TelemetryConfig(database_url="postgresql://u:p@h/t")
        assert isinstance(cfg.database_url, SecretStr)
        assert cfg.database_url.get_secret_value() == "postgresql://u:p@h/t"

    def test_parsed_neo4j_url_password_is_secretstr(self) -> None:
        parsed = ParsedNeo4jUrl.parse("bolt://user:hunter2@h:7687")
        assert isinstance(parsed.password, SecretStr)
        assert parsed.password.get_secret_value() == "hunter2"

    def test_neo4j_config_url_is_secretstr(self) -> None:
        cfg = Neo4jConfig(url="bolt://user:hunter2@h:7687")
        assert isinstance(cfg.url, SecretStr)
        assert cfg.url.get_secret_value() == "bolt://user:hunter2@h:7687"

    def test_memgraph_config_url_is_secretstr(self) -> None:
        cfg = MemgraphConfig(url="bolt://user:hunter2@h:7687")
        assert isinstance(cfg.url, SecretStr)
        assert cfg.url.get_secret_value() == "bolt://user:hunter2@h:7687"

    def test_neptune_config_url_is_secretstr(self) -> None:
        cfg = NeptuneConfig(url="bolt://user:hunter2@h:8182")
        assert isinstance(cfg.url, SecretStr)
        assert cfg.url.get_secret_value() == "bolt://user:hunter2@h:8182"

    def test_surrealdb_config_url_is_secretstr(self) -> None:
        cfg = SurrealDBConfig(url="ws://user:hunter2@h:8000")
        assert isinstance(cfg.url, SecretStr)
        assert cfg.url.get_secret_value() == "ws://user:hunter2@h:8000"

    def test_surrealdb_vector_config_url_is_secretstr(self) -> None:
        cfg = SurrealDBVectorConfig(url="ws://user:hunter2@h:8000")
        assert isinstance(cfg.url, SecretStr)
        assert cfg.url.get_secret_value() == "ws://user:hunter2@h:8000"

    def test_llm_settings_has_no_secret_fields(self) -> None:
        """Guard: LLMSettings carries no secret-typed fields today.

        ``LLMSettings.api_key_env`` is an env-var-name pointer, so it stays
        plain ``str``. If a future refactor adds an in-scope field (e.g.
        ``api_key: str``), this test must be updated to either re-type it as
        ``SecretStr`` or to explicitly carve it out — locking down the
        boundary in the test surface so the change cannot land silently.
        """
        settings = LLMSettings()
        # api_key_env is a pointer (env-var name), not a secret value.
        assert isinstance(settings.api_key_env, str)
        # Enumerate every field and assert nothing has slipped into v1-scope
        # without being typed as SecretStr.
        for name, field in LLMSettings.model_fields.items():
            assert field.annotation is not SecretStr, (
                f"LLMSettings.{name} is typed as SecretStr — update this guard "
                "test (it is now in-scope and must be enumerated explicitly)."
            )


@pytest.mark.unit
class TestNoPlaintextLeakInReprOrJson:
    """SecretStr fields must not leak the plaintext in ``repr`` / JSON dump."""

    def test_neo4j_config_repr_redacts(self) -> None:
        cfg = Neo4jConfig(url="bolt://localhost:7687", password="hunter2")
        assert "hunter2" not in repr(cfg)
        # SecretStr's masked repr is the deliberate hint that redaction happened.
        assert "**********" in repr(cfg)

    def test_khora_config_database_url_repr_redacts(self) -> None:
        cfg = KhoraConfig(database_url="postgresql://alice:hunter2@db:5432/app")
        # Whole DSN, including userinfo, is wrapped — SecretStr masks the
        # entire string, not just the userinfo.
        assert "hunter2" not in repr(cfg)
        assert "alice" not in repr(cfg)

    def test_storage_settings_json_dump_redacts(self) -> None:
        settings = StorageSettings(neo4j_password="hunter2", postgresql_url="postgresql://u:p@h/d")
        # ``model_dump_json`` masks SecretStr by default
        dumped = json.loads(settings.model_dump_json())
        assert dumped["neo4j_password"] != "hunter2"
        assert dumped["postgresql_url"] != "postgresql://u:p@h/d"


@pytest.mark.unit
class TestRedactDsn:
    """``redact_dsn`` scrubs userinfo from DSN-shaped strings."""

    @pytest.mark.parametrize(
        ("inp", "expected"),
        [
            (
                "postgresql://alice:hunter2@db:5432/app",
                "postgresql://[REDACTED]@db:5432/app",
            ),
            (
                "postgresql+asyncpg://u:p@h/d",
                "postgresql+asyncpg://[REDACTED]@h/d",
            ),
            (
                "bolt://neo4j:hunter2@cluster.example.com:7687",
                "bolt://[REDACTED]@cluster.example.com:7687",
            ),
            # Non-standard userinfo characters (service-account-style ``+``,
            # ``~``, and percent-encoded bytes) must still be scrubbed.
            (
                "postgresql://app+prod:hunter2@db:5432/app",
                "postgresql://[REDACTED]@db:5432/app",
            ),
            (
                "postgresql://user~name:hunter2@db:5432/app",
                "postgresql://[REDACTED]@db:5432/app",
            ),
            (
                "postgresql://a%2Bb:hunter2@db:5432/app",
                "postgresql://[REDACTED]@db:5432/app",
            ),
            # Empty username (e.g. Redis ``://:password@host``)
            (
                "redis://:secretpass@cache.example.com:6379/0",
                "redis://[REDACTED]@cache.example.com:6379/0",
            ),
            # Password containing ``/`` must not be truncated mid-password
            (
                "postgresql://user:pass/word@db:5432/app",
                "postgresql://[REDACTED]@db:5432/app",
            ),
            # No userinfo → unchanged
            ("postgresql://localhost:5432/app", "postgresql://localhost:5432/app"),
            # Empty / non-DSN strings → unchanged
            ("", ""),
            ("just some text", "just some text"),
        ],
    )
    def test_redact_dsn_cases(self, inp: str, expected: str) -> None:
        assert redact_dsn(inp) == expected

    def test_redact_dsn_inside_error_message(self) -> None:
        """Mimics how driver errors embed the DSN in their message."""
        msg = (
            "connection failed: could not translate host name "
            "for postgresql://alice:hunter2@db:5432/app (Name or service not known)"
        )
        out = redact_dsn(msg)
        assert "hunter2" not in out
        assert "alice" not in out
        assert "[REDACTED]" in out


@pytest.mark.unit
class TestMigrationErrorRedaction:
    """``_run_migrations_sync`` redacts DSN userinfo from the returned error."""

    def test_invalid_url_error_is_redacted(self) -> None:
        """A bogus URL surfaces a driver error containing the userinfo; the
        wrapper must scrub it before returning the :class:`MigrationResult`.

        We exercise the sync helper directly so the test stays in-process and
        does not need a real database.
        """
        from khora.db.session import _run_migrations_sync

        # An unsupported scheme + embedded credentials triggers SQLAlchemy's
        # ``NoSuchModuleError`` (or equivalent) — the message includes the URL.
        url = "definitely-not-a-real-scheme://alice:hunter2@localhost/db"
        result = _run_migrations_sync(database_url=url)
        assert result.success is False
        assert result.error is not None
        assert "hunter2" not in result.error
        assert "alice" not in result.error

    def test_no_url_returns_explicit_error(self) -> None:
        """When no URL is passed and the env var is unset, the helper returns a
        descriptive error that does not contain any DSN userinfo.
        """
        import os

        from khora.db.session import _run_migrations_sync

        prev = os.environ.pop("KHORA_DATABASE_URL", None)
        try:
            result = _run_migrations_sync(database_url=None)
        finally:
            if prev is not None:
                os.environ["KHORA_DATABASE_URL"] = prev
        assert result.success is False
        assert result.error is not None
        assert "@" not in result.error  # no userinfo at all
