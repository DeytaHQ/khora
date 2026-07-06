"""Pytest configuration and fixtures for Khora tests."""

from __future__ import annotations

import asyncio
from collections.abc import Generator

import pytest

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
    )
