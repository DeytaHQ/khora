"""Unit tests for TelemetryConfig (ADR-084 secret_typing_mode, DYT-3993)."""

import pytest
from pydantic import ValidationError

from khora.telemetry.config import TelemetryConfig


@pytest.mark.unit
class TestTelemetryConfigFromEnv:
    def test_default_when_env_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default secret_typing_mode is 'warn' when env var is absent."""
        monkeypatch.delenv("KHORA_TELEMETRY_SECRET_TYPING_MODE", raising=False)
        config = TelemetryConfig.from_env()
        assert config.secret_typing_mode == "warn"

    def test_fail_mode_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """secret_typing_mode is 'fail' when env var is set to 'fail'."""
        monkeypatch.setenv("KHORA_TELEMETRY_SECRET_TYPING_MODE", "fail")
        config = TelemetryConfig.from_env()
        assert config.secret_typing_mode == "fail"

    def test_warn_mode_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """secret_typing_mode is 'warn' when env var is explicitly set to 'warn'."""
        monkeypatch.setenv("KHORA_TELEMETRY_SECRET_TYPING_MODE", "warn")
        config = TelemetryConfig.from_env()
        assert config.secret_typing_mode == "warn"

    def test_invalid_value_raises_validation_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Invalid env var value raises ValidationError from TelemetryConfig."""
        monkeypatch.setenv("KHORA_TELEMETRY_SECRET_TYPING_MODE", "invalid")
        with pytest.raises(ValidationError):
            TelemetryConfig.from_env()
