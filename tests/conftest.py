"""Pytest configuration and fixtures for Khora tests."""

from __future__ import annotations

import asyncio
from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from khora.config import KhoraConfig


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop]:
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_config() -> KhoraConfig:
    """Create a test configuration."""
    return KhoraConfig(
        app_name="khora-test",
        environment="test",
        debug=True,
        api_host="127.0.0.1",
        api_port=8000,
        auth_enabled=False,  # Disable authentication for tests
    )


@pytest.fixture
def test_client(test_config: KhoraConfig) -> TestClient:
    """Create a test client for the FastAPI app."""
    from khora.api.app import create_app

    app = create_app(test_config)
    return TestClient(app)
