"""Google ADK + khora example - long-term memory via ``KhoraMemoryService``.

Runs without Postgres, Neo4j, or an API key. The mock LLM patches
``litellm.acompletion`` / ``litellm.aembedding`` so the example is
hermetic. The khora fixture spins up an in-memory ``sqlite_lance``
backend in a tmp dir.

This is a drop-in replacement for ADK's ``InMemoryMemoryService``:
swap the ``memory_service=`` constructor argument and every
``add_session_to_memory`` / ``search_memory`` call now lands in khora.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

# Add repo root to sys.path so ``examples._helpers`` is importable when
# this script is run from its own directory (CI smoke loop does that).
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from google.adk.events.event import Event  # noqa: E402
from google.adk.sessions.session import Session  # noqa: E402
from google.genai import types as genai_types  # noqa: E402

from examples._helpers import embedded_khora, install_mock_llm  # noqa: E402
from khora.integrations.google_adk import KhoraMemoryService  # noqa: E402


def _user_event(text: str, *, ts: float) -> Event:
    return Event(
        author="user",
        content=genai_types.Content(role="user", parts=[genai_types.Part(text=text)]),
        timestamp=ts,
    )


def _agent_event(text: str, *, ts: float) -> Event:
    return Event(
        author="agent",
        content=genai_types.Content(role="model", parts=[genai_types.Part(text=text)]),
        timestamp=ts,
    )


async def main() -> None:
    install_mock_llm()

    async with embedded_khora() as kb:
        memory = KhoraMemoryService(kb=kb)

        # Build a session with a couple of conversational turns. In a real
        # ADK app this Session is produced by SessionService - here we
        # synthesise it so the example stays self-contained.
        stored_text = "Remember that the launch is in March 2026."
        now = time.time()
        session = Session(
            id="example-session-1",
            app_name="example_app",
            user_id="example-user-1234",
            events=[
                _user_event(stored_text, ts=now),
                _agent_event("Acknowledged: PostgreSQL for the user DB.", ts=now + 1),
            ],
            last_update_time=now + 1,
        )

        await memory.add_session_to_memory(session)

        # Query with the exact stored text: hash-derived embeddings give a
        # cosine-1.0 match, guaranteeing at least one result.
        response = await memory.search_memory(
            app_name="example_app",
            user_id="example-user-1234",
            query=stored_text,
        )
        assert len(response.memories) > 0, "search_memory returned no entries"
        print(f"Recovered {len(response.memories)} memory entries:")
        for entry in response.memories:
            text = " ".join(part.text for part in (entry.content.parts or []) if part.text)
            print(f"  [{entry.author}] {text!r}")


if __name__ == "__main__":
    asyncio.run(main())
