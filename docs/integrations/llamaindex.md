# LlamaIndex integration

`khora.integrations.llamaindex` wires khora behind three LlamaIndex
surfaces in one extra:

- `KhoraRetriever` - `BaseRetriever` for any `QueryEngine` / agent that
  takes a retriever. **Async-only** - see "Sync is not implemented"
  below.
- `KhoraMemoryBlock` - `BaseMemoryBlock[str]` factory for long-term
  semantic memory inside `llama_index.core.memory.Memory`.
- `KhoraChatStore` - **deprecated** legacy `BaseChatStore` for
  `ChatMemoryBuffer` users. New code should use `KhoraMemoryBlock`.

!!! important "Namespace contract: per-user / per-agent, NOT per-session"

    `KhoraRetriever`, `KhoraMemoryBlock`, and `KhoraChatStore` all take a
    caller-supplied `namespace_id` and do no derivation. A khora namespace
    is the **tenancy** boundary: entity dedup, canonical ids, and long-term
    recall all operate *within* one namespace. Derive it from a **stable
    per-user or per-agent identity** and reuse it across conversations. Do
    **not** mint a fresh namespace per session / conversation / thread -
    that isolates every conversation into its own memory and voids
    cross-session recall. The conversation scope belongs in khora's
    first-class `session_id` (see #620), not the namespace.

## Install

```bash
pip install 'khora[llamaindex]'
```

This pulls `llama-index-core>=0.14,<0.15`. The pin is intentionally
narrow because LlamaIndex has shipped breaking changes on minor bumps
before (`BaseMemoryBlock` reshape across 0.11 → 0.12 → 0.14,
`BaseMemory.put` → `aput`). Plan one maintenance PR per LlamaIndex
minor release; the nightly skew job in CI catches breaks against the
latest tagged minor.

The adapter is also registered under the `khora.integrations`
entry-point group (factory: `KhoraRetriever`), so
`khora.integrations.discover()` returns it without explicit
registration.

## Quickstart

```python title="example.py"
"""LlamaIndex + khora example - async retrieval via ``KhoraRetriever``.

Runs without Postgres, Neo4j, or an API key. The mock LLM patches
``litellm.acompletion`` / ``litellm.aembedding`` so the example is
hermetic. The khora fixture spins up an in-memory ``sqlite_lance``
backend in a tmp dir.

Demonstrates:

* Stash a couple of documents into khora via ``Khora.remember``.
* Wrap khora in ``KhoraRetriever`` and call ``aretrieve(...)``.
* Each returned ``NodeWithScore`` carries chunk text + khora metadata
  (chunk_id, document_id, abstention signal).

``KhoraMemoryBlock`` and ``KhoraChatStore`` are also exported by the
adapter; see ``docs/integrations/llamaindex.md`` for usage notes.
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
from khora.integrations.llamaindex import KhoraRetriever  # noqa: E402


async def main() -> None:
    install_mock_llm()

    async with embedded_khora() as kb:
        namespace = await kb.create_namespace()
        ns_id = namespace.namespace_id

        # Stash two documents through khora's normal write path.
        memory_one = "We decided to use PostgreSQL for the user database."
        memory_two = "The release window is the third week of every month."
        await kb.remember(memory_one, namespace=ns_id, entity_types=[], relationship_types=[])
        await kb.remember(memory_two, namespace=ns_id, entity_types=[], relationship_types=[])

        retriever = KhoraRetriever(kb, namespace_id=ns_id, similarity_top_k=3)

        # Verbatim recall: the mock LLM's hash-derived embeddings give
        # an exact match (cosine = 1.0) for the stored text.
        nodes = await retriever.aretrieve(memory_one)
        assert len(nodes) > 0, "recall returned no nodes"
        for node in nodes:
            text = node.node.text.replace("\n", " ")
            print(f"[{node.score:.2f}] {text}")


if __name__ == "__main__":
    asyncio.run(main())
```

The block above is enforced byte-identical against
`examples/integrations/llamaindex/example.py` by
`tools/check_examples_drift.py` (CI gate).

## `KhoraRetriever`

| arg | default | notes |
| --- | --- | --- |
| `kb` | - | A connected `Khora` instance. Adapter does NOT own the lifecycle. |
| `namespace_id` | - | Required khora namespace UUID this retriever reads from. |
| `similarity_top_k` | `10` | Max chunks (and optionally entities) returned per `aretrieve` call. |
| `include_entities` | `False` | When `True`, entity hits are returned alongside chunk hits as additional `NodeWithScore`s. Default off per issue #627 acceptance criteria. |
| `recall_kwargs` | `None` | Optional dict of extra kwargs forwarded to `Khora.recall` (e.g. `{"mode": SearchMode.HYBRID, "min_similarity": 0.2}`). |

Each returned `NodeWithScore`'s `node.metadata` carries:

| key | description |
| --- | --- |
| `khora_kind` | `"chunk"` or `"entity"`. |
| `chunk_id` / `entity_id` | Source object's khora UUID. |
| `document_id` | Parent document UUID (chunk nodes only). |
| `namespace_id` | Khora namespace this node came from. |
| `khora_should_abstain` | Boolean - `True` when khora's chronicle abstention signals say the recall is low-confidence. Present only when the recall result carries abstention signals (chronicle path); absent on vectorcypher/skeleton paths. |

### Sync is not implemented

`KhoraRetriever._retrieve` raises `NotImplementedError`. The reason is
specific to this adapter: khora's recall is async-native and the
deadlock surface for bridging it through a thread inside a running event
loop dominates the failure modes for this kind of plumbing. The fix is
straightforward - every LlamaIndex `QueryEngine` exposes
`aquery(...) / aretrieve(...)`. Use those.

If you genuinely need a sync path (e.g. a notebook outside any event
loop), wrap the call yourself:

```python
import asyncio
nodes = asyncio.run(retriever.aretrieve("query"))
```

We deliberately do not ship a `nest_asyncio` workaround - that's a
hidden reentrancy hazard under any real agent loop.

## `KhoraMemoryBlock`

Long-term memory block for `llama_index.core.memory.Memory`. The factory
returns a `BaseMemoryBlock[str]` instance:

```python
from llama_index.core.memory import Memory
from khora.integrations.llamaindex import KhoraMemoryBlock

block = KhoraMemoryBlock(
    kb=kb,
    namespace_id=ns_id,
    name="khora_long_term",
    priority=1,
    similarity_top_k=5,
    session_id=session_uuid,  # optional - enables Khora.forget_session() cleanup
)

memory = Memory.from_defaults(
    session_id="agent-1",
    memory_blocks=[block],
)
```

Semantics:

- `_aget(messages)` picks the last user-role message, calls
  `Khora.recall(query, namespace=…, limit=similarity_top_k)`, and
  returns the rendered context wrapped in `<khora_memory>…</khora_memory>`
  so the prompt template can spot it.
- `_aput(messages)` calls `Khora.remember(content, namespace=…)` once
  per message (skipping empty ones). The returned `document_id` is
  stamped onto `message.additional_kwargs["khora_event_id"]` so callers
  can round-trip a delete handle.
- `atruncate(content, tokens_to_truncate)` returns `None` - khora is
  the persistent store, so dropping the in-flight payload loses nothing.

| arg | default | notes |
| --- | --- | --- |
| `kb` | - | A connected `Khora` instance. |
| `namespace_id` | - | Required khora namespace UUID. |
| `name` | `"khora_memory"` | LlamaIndex memory block name. |
| `description` | `None` | Optional human-readable description. |
| `priority` | `1` | LlamaIndex truncation priority (lower = kept longer). |
| `similarity_top_k` | `5` | Recall limit per `_aget` call. |
| `session_id` | `None` | Optional khora session UUID stamped on every `remember`. Enables `Khora.forget_session(...)` cleanup. |
| `skill_name` | `"general_entities"` | khora extraction skill name. |
| `entity_types` | `None` (→ `[]`) | Extraction whitelist. Empty disables extraction entirely (cheap chat-history writes). |
| `relationship_types` | `None` (→ `[]`) | Same - empty disables. |

## `KhoraChatStore` (deprecated)

Legacy `BaseChatStore` for `ChatMemoryBuffer`. Instantiation emits a
`DeprecationWarning`. Provided only for compatibility with existing
code; new agents should use `KhoraMemoryBlock` instead.

```python
from khora.integrations.llamaindex import KhoraChatStore
from llama_index.core.memory import ChatMemoryBuffer

chat_store = KhoraChatStore(kb=kb, namespace_id=ns_id)  # DeprecationWarning
memory = ChatMemoryBuffer.from_defaults(
    token_limit=3000,
    chat_store=chat_store,
    chat_store_key="conversation-1",
)
```

All seven `BaseChatStore` abstract sync methods are implemented and
bridged through `khora.integrations._sync.run_sync` (which runs the
coroutine on a daemon-thread loop and blocks the caller - calling from
inside an `async def` will stall the calling event loop; see "Sync is
not implemented" above for the same reasoning).

`get_keys()` and the per-key list-by-index operations scan documents in
the bound namespace and filter on metadata client-side
(`llamaindex_chat_key`, `llamaindex_chat_index`). This is fine for
bounded chat workloads (one key per conversation, dozens of messages
each). Multi-tenant deployments with many active conversations in one
namespace should partition by `namespace_id` instead.

## Limits and future work

- Filter pushdown to SQL for `KhoraChatStore.get_messages` - the current
  O(N_docs) scan is acceptable for bounded chat workloads but not for
  hot multi-tenant deployments.
- No support for image / audio / tool-call blocks inside `ChatMessage`
  - only the rendered text is persisted (via `ChatMessage.content`).
  The original `additional_kwargs` round-trips so the consumer can
  reconstruct non-text payloads from its own side channel.
- `KhoraRetriever` returns chunks and (optionally) entities; it does
  **not** return relationships. LlamaIndex has no first-class
  relationship node type and forcing them into `TextNode` would
  pollute the response synthesizer. Use `Khora.recall(...)` directly
  if you need relationship data.
