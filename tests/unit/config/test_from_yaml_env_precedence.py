"""Config tests for #897 (env-var precedence in from_yaml) and #908/#1432 (dead field removal)."""

from __future__ import annotations

from pathlib import Path

import pydantic
import pytest

from khora.config import KhoraConfig


@pytest.mark.unit
class TestFromYamlEnvPrecedence:
    """#897: KHORA_DATABASE_URL / KHORA_NEO4J_URL must win over YAML in from_yaml."""

    def test_env_overrides_yaml_database_url(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        yaml_path = tmp_path / "khora.yaml"
        yaml_path.write_text("database_url: postgresql://yaml-host:5432/yaml\n")
        monkeypatch.setenv("KHORA_DATABASE_URL", "postgresql://env-host:5432/env")

        config = KhoraConfig.from_yaml(yaml_path)

        assert config.get_postgresql_url() == "postgresql://env-host:5432/env"

    def test_env_overrides_yaml_neo4j_url(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        yaml_path = tmp_path / "khora.yaml"
        yaml_path.write_text("neo4j_url: bolt://yaml:pw@yaml-host:7687\n")
        monkeypatch.setenv("KHORA_NEO4J_URL", "bolt://env:pw@env-host:7687")

        config = KhoraConfig.from_yaml(yaml_path)

        assert config.get_neo4j_url() == "bolt://env-host:7687"

    def test_env_fills_when_yaml_absent(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        yaml_path = tmp_path / "khora.yaml"
        yaml_path.write_text("app_name: khora\n")
        monkeypatch.setenv("KHORA_DATABASE_URL", "postgresql://env-host:5432/env")

        config = KhoraConfig.from_yaml(yaml_path)

        assert config.get_postgresql_url() == "postgresql://env-host:5432/env"

    def test_yaml_wins_when_env_absent(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        yaml_path = tmp_path / "khora.yaml"
        yaml_path.write_text("database_url: postgresql://yaml-host:5432/yaml\n")
        monkeypatch.delenv("KHORA_DATABASE_URL", raising=False)

        config = KhoraConfig.from_yaml(yaml_path)

        assert config.get_postgresql_url() == "postgresql://yaml-host:5432/yaml"


@pytest.mark.unit
class TestAuthEnabledRemoved:
    """#908: auth_enabled was a dead field enforcing nothing. It is removed."""

    def test_field_not_on_model(self) -> None:
        assert "auth_enabled" not in KhoraConfig.model_fields

    def test_constructing_with_auth_enabled_rejected(self) -> None:
        # KhoraConfig's model_config rejects extras (extra='forbid'), so the
        # removed kwarg now raises rather than being silently dropped.
        with pytest.raises(pydantic.ValidationError):
            KhoraConfig(auth_enabled=True)  # type: ignore[call-arg]


@pytest.mark.unit
class TestEnvironmentDebugRemoved:
    """#1432: environment / debug were dead fields read by nothing. Removed.

    Same removal pattern as #908's auth_enabled: field gone from the model,
    the removed constructor kwarg raises (extra='forbid'), but a stale
    KHORA_ENVIRONMENT / KHORA_DEBUG env var (e.g. an old compose file) is
    ignored rather than breaking config construction.
    """

    @pytest.mark.parametrize("field", ["environment", "debug"])
    def test_field_not_on_model(self, field: str) -> None:
        assert field not in KhoraConfig.model_fields

    def test_constructing_with_environment_rejected(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            KhoraConfig(environment="production")  # type: ignore[call-arg]

    def test_constructing_with_debug_rejected(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            KhoraConfig(debug=True)  # type: ignore[call-arg]

    def test_stale_env_vars_do_not_break_construction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KHORA_ENVIRONMENT", "production")
        monkeypatch.setenv("KHORA_DEBUG", "true")

        config = KhoraConfig()

        assert not hasattr(config, "environment")
        assert not hasattr(config, "debug")
