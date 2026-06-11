# OpenAI Agents SDK integration

> **Stability warning.** The upstream `openai-agents` SDK is pre-1.0 - 17
> releases in the 7 months leading up to `v0.17`. The adapter pins
> tightly (`openai-agents>=0.17,<0.18`) and is tagged
> **experimental**: expect monthly rework as upstream's session /
> hooks / tool surfaces shift. A nightly skew job (issue #618) opens
> auto-issues when the latest upstream breaks.

`khora.integrations.openai_agents` exposes three independent primitives
a caller mixes and matches against the OpenAI Agents SDK:

| Primitive            | Implements                      | Use it when                                                                                  |
| -------------------- | ------------------------------- | -------------------------------------------------------------------------------------------- |
| `KhoraSession`       | `agents.memory.session.SessionABC` | You want khora-backed conversation memory passed via `Runner.run(..., session=...)`.         |
| `khora_recall_tool`  | `agents.FunctionTool` factory   | You want an `Agent` to call khora at run-time to surface relevant past memories.             |
| `KhoraMemoryHooks`   | `RunHooks`-shaped class         | You want tool outputs auto-written to khora (and optionally to recall context on agent start). |

The three are designed to compose - a typical setup wires all of them
into one `Agent` / `Runner` call.

## Install

```bash
pip install 'khora[openai-agents]'
```

This pulls `openai-agents>=0.17,<0.18`. The adapter is also registered
under the `khora.integrations` entry-point group, so `discover()`
returns it without explicit registration.

## Constructors

### `KhoraSession`

| arg            | default              | notes                                                                                  |
| -------------- | -------------------- | -------------------------------------------------------------------------------------- |
| `kb`           | -                    | A connected `Khora` instance. Adapter does NOT own the lifecycle.                      |
| `namespace`    | -                    | Stable khora namespace UUID. Every read/write is scoped to it.                         |
| `session_id`   | -                    | The SDK session id string. Maps to a deterministic khora `session_id` via UUID5.       |
| `app_id`       | `"openai_agents"`    | Free-form app identifier stamped into stored metadata.                                  |

`KhoraSession` is a runtime `SessionABC` (verified via `isinstance` in
tests) - catches SDK rename drift on construction rather than at the
next `Runner.run`.

### `khora_recall_tool`

```python
khora_recall_tool(
    *,
    kb: Khora,
    namespace: UUID,
    top_k: int = 5,
    min_similarity: float = 0.0,
    name: str = "recall_memory",
    description: str | None = None,
) -> agents.FunctionTool
```

Returns a `FunctionTool` whose only LLM-visible argument is `query:
str`. The bound khora instance, namespace, and recall thresholds are
captured by closure - the LLM cannot rewrite them. Drop straight into
`Agent(tools=[tool])`.

### `KhoraMemoryHooks`

| arg                   | default              | notes                                                                                   |
| --------------------- | -------------------- | --------------------------------------------------------------------------------------- |
| `kb`                  | -                    | A connected `Khora` instance.                                                            |
| `namespace`           | -                    | khora namespace UUID.                                                                    |
| `app_id`              | `"openai_agents"`    | Free-form app identifier.                                                                |
| `record_tool_results` | `True`               | Persist every successful tool result via `Khora.remember`.                              |
| `recall_on_start`     | `False`              | When `True`, log top-K khora hits on `on_agent_start`.                                   |
| `recall_top_k`        | `3`                  | `limit` passed to `Khora.recall` on agent start.                                         |

`KhoraMemoryHooks` is plain - `Runner` duck-types hook callbacks so the
class is accepted without `isinstance`. Use `hooks.as_runhooks()` if a
static checker insists on a `RunHooks` subclass.

## Mapping

| OpenAI Agents SDK                                | khora                                                            |
| ------------------------------------------------ | ---------------------------------------------------------------- |
| `Session.session_id` (string)                    | `Document.metadata["oai_session_id"]` + UUID5 `session_id` |
| One `TResponseInputItem`                         | One `Document`. Verbatim JSON stored in `metadata["oai_item"]`. |
| Monotonic write order                            | `metadata["oai_seq"]` (0, 1, 2, ...) - ordering key on read-back. |
| `function_tool` call output                      | (Via `KhoraMemoryHooks.on_tool_end`) one `Document` tagged with `oai_tool_name`. |

Documents stamped by this adapter use the prefix `oai:` on their
`external_id` (`oai:<session>:<seq>`). Foreign documents in the same
namespace are silently skipped on read-back.

### `TResponseInputItem` serialisation

The verbatim JSON of every item is preserved - non-text items
(function calls, function responses, refusals, reasoning items, …)
round-trip exactly. The adapter never tries to "interpret" SDK union
variants beyond projecting a human-readable text body onto
`Document.content` so vector recall has something to embed.

### Why documents and not chunks?

khora's chunk storage layout varies per backend (the `sqlite_lance`
embedded stack stores chunks in LanceDB plus a Skeleton temporal
table). `storage.list_documents` is the universally reliable
iteration surface, and a `Document.metadata` dict survives the
round trip on every backend. We keep the verbatim item JSON on the
document so `get_items` never needs to peek at chunks.

## Quickstart

The block below is byte-identical to
[`examples/integrations/openai_agents/example.py`](../../examples/integrations/openai_agents/example.py)
- CI fails if they diverge.

```python title="example.py"
"""OpenAI Agents SDK + khora example - session memory via ``KhoraSession``.

Runs without Postgres, Neo4j, or an API key. The mock LLM patches
``litellm.acompletion`` / ``litellm.aembedding`` so the example is
hermetic. The khora fixture spins up an in-memory ``sqlite_lance``
backend in a tmp dir.

The example does NOT spin up a real ``agents.Runner`` - that would
require a live LLM. Instead it exercises the three khora primitives the
adapter exposes directly: ``KhoraSession`` (SessionABC contract),
``khora_recall_tool`` (FunctionTool factory), and ``KhoraMemoryHooks``
(RunHooks-shaped). Each is what an ``Agent`` would call into.

Kept deliberately small (single ``add_items`` write) so it finishes
well under the CI smoke budget - every khora write still runs the full
extraction pipeline against the mock LLM.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Add repo root to sys.path so ``examples._helpers`` is importable when
# this script is run from its own directory (CI smoke loop does that).
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples._helpers import embedded_khora, install_mock_llm  # noqa: E402
from khora.integrations.openai_agents import (  # noqa: E402
    KhoraMemoryHooks,
    KhoraSession,
    khora_recall_tool,
)


async def main() -> None:
    install_mock_llm()

    async with embedded_khora() as kb:
        namespace = await kb.create_namespace()
        ns_id = namespace.namespace_id

        # 1) Session - one turn is enough to demonstrate the SessionABC
        #    contract. Every khora write runs full extraction, so we keep
        #    the example light to fit the CI smoke budget.
        session = KhoraSession(kb=kb, namespace=ns_id, session_id="example-conv-1")
        await session.add_items([{"role": "user", "content": "We picked PostgreSQL for the user DB."}])
        items = await session.get_items()
        assert len(items) == 1, "expected the session to round-trip exactly one item"
        assert items[-1]["content"] == "We picked PostgreSQL for the user DB."
        print(f"Session has {len(items)} item(s); latest: {items[-1]['content']!r}")

        # 2) Recall tool - closes over (kb, namespace, top_k). Construction
        #    is pure Python; no LLM I/O. An Agent would invoke it later.
        tool = khora_recall_tool(kb=kb, namespace=ns_id, top_k=3)
        print(f"Built recall tool: name={tool.name!r}")

        # 3) Memory hooks - construct only. ``on_tool_end`` would normally
        #    fire from inside ``Runner.run(...)`` and persist the tool
        #    output. We skip the live call here to keep the example fast.
        hooks = KhoraMemoryHooks(kb=kb, namespace=ns_id, app_id="example")
        print(f"Built memory hooks: app_id={hooks.app_id!r}")


if __name__ == "__main__":
    asyncio.run(main())
```

The block above is enforced byte-identical against
`examples/integrations/openai_agents/example.py` by
`tools/check_examples_drift.py` (CI gate).

## Limits and future work

- **SDK skew tolerance.** Pinned to one upstream minor. Bump in a
  deliberate PR per `openai-agents` minor; the nightly skew job
  (#618) flags breakage.
- **`khora_recall_tool` argument set.** v1 exposes one arg (`query`).
  Adding filters (date range, tool-name filter) is a clean addition
  - gate them behind explicit factory kwargs so the LLM-visible
  schema stays minimal.
- **`KhoraMemoryHooks.on_agent_start` recall surfacing.** Default
  behaviour logs hits; downstream callers will usually want to feed
  them into `agent.instructions` or a system message. Subclass and
  override.
- **`run_compaction`** (`OpenAIResponsesCompactionAwareSession`) -
  intentionally NOT implemented in v1. Add when a real caller needs
  it; the SDK's protocol-on-top-of-protocol surface is still in flux.

## Filename note

The doc file is `docs/integrations/openai_agents.md` (underscore, not
hyphen). `tools/check_examples_drift.py` derives the framework slug
from the Markdown stem and matches it to
`examples/integrations/<slug>/example.py`; since the Python package
must be `openai_agents` (PEP 8 dotted-path), the doc filename has to
match.
