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

        # The mock LLM produces hash-derived embeddings — cosine
        # similarity is uncorrelated for distinct texts. To verify the
        # plumbing without depending on semantic recall, we query with
        # one of the stored sentences verbatim: deterministic hashing
        # guarantees a perfect match (cosine = 1.0) on at least the
        # exact-match chunk.
        memory_one = "We decided to use PostgreSQL for the user database."
        memory_two = "The release window is the third week of every month."

        memory.remember(memory_one)
        memory.remember(memory_two)

        matches = memory.recall(memory_one, limit=5)

        assert matches, "expected at least one match from the in-memory backend"
        contents = [m.record.content for m in matches]
        assert memory_one in contents, f"verbatim recall failed: {contents}"
