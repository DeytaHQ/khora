# Google ADK integration

`khora.integrations.google_adk.KhoraMemoryService` implements Google
ADK's `BaseMemoryService` so a `Runner` can use khora as its
long-term memory in one line:

```python
runner = Runner(
    app_name="my_app",
    agent=my_agent,
    session_service=InMemorySessionService(),
    memory_service=KhoraMemoryService(kb=kb),
)
```

The adapter is a drop-in replacement for ADK's `InMemoryMemoryService`
and `VertexAiMemoryBankService`. Each ADK `(app_name, user_id)` pair
maps to a deterministic khora namespace UUID5 (see #618), so two
service instances on the same khora deployment see the same memory
without a shared registry.

> **ADK 2.0 incoming.** `google-adk` is on a weekly release cadence
> and the 2.0 line is already in beta. The adapter is pinned
> `google-adk>=1.32,<2.0` and tagged `stability: experimental` until
> ADK 2.x GAs and the adapter smoke passes against it.

## Scope (v0.14)

- `KhoraMemoryService` - long-term memory service implementing
  `add_session_to_memory`, `add_events_to_memory`, and
  `search_memory`. **Shipped.**
- `KhoraSessionService` - **NOT shipped.** ADK ships
  `InMemorySessionService` + `DatabaseSessionService` (SQLAlchemy)
  for short-term turn state. khora offers no differentiator there.
  Revisit only if a single-DB story for sessions + memory becomes a
  real user ask.

## Install

```bash
pip install 'khora[google-adk]'
```

This pulls `google-adk>=1.32,<2.0`. The adapter is registered under
the `khora.integrations` entry-point group, so `discover()` returns
it without explicit registration.

## Constructor

| arg | default | notes |
| --- | --- | --- |
| `kb` | - | A connected `Khora` instance. The service does NOT own its lifecycle. |
| `app_id` | `"google_adk"` | Free-form identifier stamped into stored metadata for audit / debugging. Distinct from the per-call `app_name` ADK passes - that one is part of the namespace key. |
| `recall_limit` | `10` | Default `limit` forwarded to `Khora.recall`. |
| `min_similarity` | `0.0` | Default similarity floor forwarded to `Khora.recall`. |

## Method semantics

All methods are async - ADK invokes them from its own event loop, so
no sync bridging is involved.

- `add_session_to_memory(session)` - ingests every event in
  `session.events` as a separate khora document. Events with no usable
  content (no text parts AND no non-text parts) are skipped, matching
  `InMemoryMemoryService`. Re-ingesting the same session is safe:
  deduplication keys off `event.id` via `Document.external_id`.
- `add_events_to_memory(*, app_name, user_id, events, session_id, custom_metadata)`
  - incremental delta of events for an existing namespace. Same
  deduplication contract as `add_session_to_memory`.
  `custom_metadata` is merged into every event's
  `Document.metadata`.
- `search_memory(*, app_name, user_id, query)` - `Khora.recall`
  against the resolved namespace. Returns `SearchMemoryResponse` with
  one `MemoryEntry` per matched event (chunks belonging to the same
  event are coalesced). Returns an empty response when the namespace
  hasn't been ingested into yet.

### Session attribution

`Session.id` (an arbitrary string) maps to a UUID5-derived
`session_id` (#620). Pure UUID strings round-trip verbatim. Use
`Khora.forget_session(namespace, session_id)` to drop a whole
conversation atomically.

### Non-text Parts

`function_call`, `function_response`, and `inline_data` parts are
JSON-encoded into `Document.metadata["adk_parts"]`. The bytes
of `inline_data` are dropped (mime type + sha1 prefix kept) - they
would bloat the document store without being useful for vector recall.
A short placeholder is rendered as the document content for events
that carry only tool calls so they remain retrievable by name.

## Quickstart

```python title="example.py"
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
        now = time.time()
        session = Session(
            id="example-session-1",
            app_name="example_app",
            user_id="example-user-1234",
            events=[
                _user_event("Remember that the launch is in March 2026.", ts=now),
                _agent_event("Acknowledged: PostgreSQL for the user DB.", ts=now + 1),
            ],
            last_update_time=now + 1,
        )

        await memory.add_session_to_memory(session)

        response = await memory.search_memory(
            app_name="example_app",
            user_id="example-user-1234",
            query="which database did we pick?",
        )
        assert response.memories, "expected search_memory to recover the session events"
        print(f"Recovered {len(response.memories)} memory entries:")
        for entry in response.memories:
            text = " ".join(part.text for part in (entry.content.parts or []) if part.text)
            print(f"  [{entry.author}] {text!r}")


if __name__ == "__main__":
    asyncio.run(main())
```

The block above is enforced byte-identical against
`examples/integrations/google_adk/example.py` by
`tools/check_examples_drift.py` (CI gate).

## Limits and future work

- Filter / metadata pushdown - `search_memory` runs khora's standard
  hybrid recall; ADK's contract doesn't surface a filter parameter
  yet. When it does, we'll forward `app_name` / `user_id` / `tags`
  to the SQL layer instead of relying on the namespace partition.
- Session service - explicit non-goal for v0.14 (see "Scope" above).
- `inline_data` bytes are dropped on ingest. Multi-modal long-term
  memory needs a dedicated blob-store hookup before the bytes can be
  preserved without ballooning the document table.
- Aligning with ADK 2.x - the adapter is pinned `<2.0` until the
  beta's `BaseMemoryService` shape stabilises. Track upstream changes
  via the nightly-skew CI job (#618).
