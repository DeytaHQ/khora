"""End-to-end integration: ``KhoraMemoryService`` round-trip on sqlite_lance.

Runs against an in-memory ``sqlite_lance`` khora (no Postgres, no
Neo4j). The mock LLM helper patches ``litellm.acompletion`` /
``litellm.aembedding`` so no API keys are needed.

This test proves the adapter is wired up correctly end-to-end:

1. Build a real ``Khora`` on sqlite_lance.
2. Build a ``KhoraMemoryService`` over it.
3. Construct an ADK ``Session`` with a few conversational events.
4. ``add_session_to_memory`` -> khora stores documents + chunks.
5. ``search_memory`` -> chunks come back as ``MemoryEntry`` instances.
"""

from __future__ import annotations

import time

import pytest

try:
    import aiosqlite  # noqa: F401
    import lancedb  # noqa: F401

    _HAS_EMBEDDED = True
except ImportError:
    _HAS_EMBEDDED = False

try:
    from google.adk.events.event import Event
    from google.adk.sessions.session import Session
    from google.genai import types as genai_types

    _HAS_ADK = True
except ImportError:
    _HAS_ADK = False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _HAS_EMBEDDED, reason="aiosqlite/lancedb not installed"),
    pytest.mark.skipif(not _HAS_ADK, reason="google-adk not installed"),
]


@pytest.mark.asyncio
async def test_session_ingest_and_search_roundtrip(monkeypatch):
    """Ingest a Session, then recall its events back through search_memory."""
    from examples._helpers import embedded_khora, install_mock_llm
    from khora.integrations.google_adk import KhoraMemoryService

    install_mock_llm(monkeypatch=monkeypatch)

    async with embedded_khora() as kb:
        memory = KhoraMemoryService(kb=kb)

        now = time.time()
        session = Session(
            id="e2e-session-1",
            app_name="e2e_app",
            user_id="e2e-user-1234",
            events=[
                Event(
                    author="user",
                    content=genai_types.Content(
                        role="user", parts=[genai_types.Part(text="The launch is in March 2026.")]
                    ),
                    timestamp=now,
                ),
                Event(
                    author="agent",
                    content=genai_types.Content(
                        role="model",
                        parts=[genai_types.Part(text="Acknowledged — March 2026 launch.")],
                    ),
                    timestamp=now + 1,
                ),
                Event(
                    author="user",
                    content=genai_types.Content(
                        role="user",
                        parts=[genai_types.Part(text="We chose PostgreSQL for the user DB.")],
                    ),
                    timestamp=now + 2,
                ),
            ],
            last_update_time=now + 2,
        )

        await memory.add_session_to_memory(session)

        response = await memory.search_memory(
            app_name="e2e_app",
            user_id="e2e-user-1234",
            query="which database did we pick?",
        )

        # The mock embedder is deterministic but not semantic; the assertion
        # focuses on contract conformance (non-empty response, well-formed
        # entries) rather than relevance.
        assert response.memories, "expected at least one MemoryEntry"
        for entry in response.memories:
            assert entry.content is not None
            assert entry.content.parts, "entry must carry at least one Part"
            assert entry.author in {"user", "agent"}
            assert entry.timestamp is not None  # ISO 8601 string


@pytest.mark.asyncio
async def test_add_events_to_memory_incremental(monkeypatch):
    """add_events_to_memory ingests events without re-running a full Session."""
    from examples._helpers import embedded_khora, install_mock_llm
    from khora.integrations.google_adk import KhoraMemoryService

    install_mock_llm(monkeypatch=monkeypatch)

    async with embedded_khora() as kb:
        memory = KhoraMemoryService(kb=kb)

        now = time.time()
        events = [
            Event(
                author="user",
                content=genai_types.Content(role="user", parts=[genai_types.Part(text="Delta event one.")]),
                timestamp=now,
            ),
            Event(
                author="agent",
                content=genai_types.Content(role="model", parts=[genai_types.Part(text="Delta response one.")]),
                timestamp=now + 1,
            ),
        ]

        await memory.add_events_to_memory(
            app_name="delta_app",
            user_id="delta-user-1234",
            events=events,
            session_id="delta-session-1",
        )

        response = await memory.search_memory(
            app_name="delta_app",
            user_id="delta-user-1234",
            query="delta",
        )
        assert response.memories
