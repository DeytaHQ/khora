"""Fixtures for khora.integrations unit tests.

Most fixtures here exist to keep tests isolated from one another:

- ``clear_registry`` wipes the integrations registry between tests so
  one test's explicit ``register()`` can't leak into the next.
- ``shutdown_sync_bridge`` tears down the shared sync-bridge event loop
  between tests so reentrancy and cleanup-after-exception cases each
  start from a clean state.
"""

from __future__ import annotations

import pytest

from khora.integrations import _sync, registry


@pytest.fixture(autouse=True)
def clear_registry():
    """Reset the integrations registry around every test."""
    registry.clear()
    yield
    registry.clear()


@pytest.fixture
def shutdown_sync_bridge():
    """Tear down the sync-bridge loop after the test.

    Not autouse — most tests want the loop kept alive across calls. Use
    this in tests that explicitly verify lifecycle behaviour.
    """
    yield
    _sync._shutdown_for_tests()
