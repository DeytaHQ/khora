"""Tests for API module."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.mark.unit
class TestStatusEndpoints:
    """Tests for status check endpoints."""

    def test_status_check(self, test_client: TestClient) -> None:
        """Test basic status check endpoint."""
        response = test_client.get("/status")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "timestamp" in data
        assert data["version"] == "0.0.4"
        assert data["service"] == "khora"

    def test_health_check(self, test_client: TestClient) -> None:
        """Test health check endpoint."""
        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["version"] == "0.0.4"

    def test_readiness_check(self, test_client: TestClient) -> None:
        """Test readiness check endpoint."""
        response = test_client.get("/health/ready")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["ready", "not_ready"]
        assert "timestamp" in data
        assert "checks" in data

    def test_liveness_check(self, test_client: TestClient) -> None:
        """Test liveness check endpoint."""
        response = test_client.get("/health/live")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"
        assert "timestamp" in data


@pytest.mark.unit
class TestConfig:
    """Tests for configuration."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        from khora.config import KhoraConfig

        config = KhoraConfig()
        assert config.app_name == "khora"
        assert config.environment == "development"
        assert config.debug is False
        assert config.api_host == "127.0.0.1"
        assert config.api_port == 8000
        assert config.auth_enabled is True

    def test_config_from_env(self, monkeypatch) -> None:
        """Test configuration from environment variables."""
        from khora.config import KhoraConfig

        monkeypatch.setenv("KHORA_DEBUG", "true")
        monkeypatch.setenv("KHORA_API_PORT", "9000")
        monkeypatch.setenv("KHORA_ENVIRONMENT", "staging")

        config = KhoraConfig()
        assert config.debug is True
        assert config.api_port == 9000
        assert config.environment == "staging"
