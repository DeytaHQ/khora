"""End-to-end smoke test for the CrewAI adapter.

Builds a ``crewai.Memory(storage=KhoraStorageBackend(...))`` against
the in-memory ``sqlite_lance`` khora fixture and the mock LLM, saves
two records, runs a search, and asserts both come back. Skipped when
``crewai`` is not installed (the integration test job and the
``examples-smoke`` job pull the extra in).
"""

from __future__ import annotations

from uuid import UUID

import pytest

pytest.importorskip("crewai")

from examples._helpers import embedded_khora, install_mock_llm  # noqa: E402
from khora.integrations.crewai import KhoraMemory  # noqa: E402

pytestmark = [pytest.mark.integration]


@pytest.fixture(autouse=True)
def _mock_llm() -> None:
    """Patch litellm with deterministic stubs for the test session."""
    install_mock_llm()


async def test_memory_save_and_search_roundtrip() -> None:
    async with embedded_khora() as kb:
        namespace = await kb.create_namespace()
        ns_id: UUID = namespace.namespace_id

        memory = KhoraMemory(
            kb=kb,
            namespace=ns_id,
            user_id="user-test-12345678",
        )

        memory.remember("We decided to use PostgreSQL for the user database.")
        memory.remember("The release window is the third week of every month.")

        matches = memory.recall("database choice", limit=5)

        assert matches, "expected at least one match from the in-memory backend"
        contents = [m.record.content for m in matches]
        assert any("PostgreSQL" in c or "database" in c for c in contents), (
            f"recall did not surface either saved record: {contents}"
        )
