# CrewAI adapter (`khora.integrations.crewai`)

Wire khora as the storage backend for CrewAI's unified `Memory`. One
line on top of `pip install khora[crewai]`:

```python
from khora.integrations.crewai import KhoraMemory

memory = KhoraMemory(kb=kb, namespace=ns_id, user_id="user-…")
agent = Agent(role="…", memory=memory)
```

Stability: experimental. Will be promoted to stable after one full
khora minor ships without a breaking change to the adapter surface.

## Install

```bash
pip install "khora[crewai]"
```

Pulls `crewai>=1.10,<2.0` plus a stable khora.

## Quickstart

The block below is byte-identical to
[`examples/integrations/crewai/example.py`](../../examples/integrations/crewai/example.py)
- CI fails if they diverge.

```python title="example.py"
"""Smoke example for the khora CrewAI adapter.

Runs without external services or API keys: the in-memory sqlite_lance
khora fixture plus the deterministic mock LLM cover everything the
adapter needs end-to-end.
"""

from __future__ import annotations

import asyncio

from examples._helpers import embedded_khora, install_mock_llm
from khora.integrations.crewai import KhoraMemory


async def _main() -> None:
    install_mock_llm()
    async with embedded_khora() as kb:
        namespace = await kb.create_namespace()
        memory = KhoraMemory(
            kb=kb,
            namespace=namespace.namespace_id,
            user_id="user-example-12345678",
        )

        decision = "We decided to use PostgreSQL for the user database."
        memory.remember(
            decision,
            scope="/project/decisions",
            importance=0.9,
        )
        memory.remember(
            "The release window is the third week of every month.",
            scope="/project/process",
            importance=0.6,
        )

        # Query with the exact stored text: hash-derived embeddings give a
        # cosine-1.0 match, guaranteeing at least one result.
        matches = memory.recall(decision, limit=3)
        assert len(matches) > 0, "recall returned no results"
        for match in matches:
            print(f"[{match.score:.2f}] {match.record.content}")


if __name__ == "__main__":
    asyncio.run(_main())
```

## Public surface

- `KhoraMemory(kb, namespace, *, user_id, app_id="crewai", scope_root="/", **memory_kwargs)`
  - factory returning a `crewai.Memory` wired against khora.
- `KhoraStorageBackend` - the duck-typed `crewai.memory.storage.backend.StorageBackend`
  implementation. Exposed for advanced users who want to construct the
  CrewAI `Memory` themselves.

## Scope ↔ namespace + tags mapping

CrewAI organises memories under a hierarchical `scope` path
(`/crew/research/<session>`) plus a flat `categories` list. Khora has
a single `namespace_id` per memory. The adapter resolves the two like
this:

| CrewAI                              | Khora                                                  |
|-------------------------------------|--------------------------------------------------------|
| `namespace` arg to `KhoraMemory`    | `Document.namespace_id`                                |
| `MemoryRecord.scope`                | `Document.metadata["crewai_scope"]`                    |
| trailing UUID on the scope path     | `Document.session_id` (and `Chunk.session_id`)         |
| `MemoryRecord.categories`           | `Document.metadata["crewai_categories"]`               |
| `MemoryRecord.importance`           | `Document.metadata["crewai_importance"]`               |
| `MemoryRecord.source`               | `Document.metadata["crewai_source"]`                   |
| `user_id` arg to `KhoraMemory`      | `Document.metadata["crewai_user_id"]`                  |

Filtering on `scope_prefix` / `categories` / `metadata_filter` in
`search` and `list_records` is performed **post-recall** against
`Document.metadata` - khora has no per-document scope or
category columns to push the filter down into. For typical CrewAI
working-set sizes (hundreds to low thousands of records per
namespace), the post-filter is fast enough. Deployments with deep
scope trees and millions of records should partition by
`KhoraMemory.namespace` instead of relying on scope filters.

### `user_id` validation

The factory **rejects** the following `user_id` values with
`khora.exceptions.KhoraIntegrationError`:

- empty string
- `"default"`
- any value shorter than 8 characters

Silent cross-user reads are the dominant misuse mode for any
multi-tenant memory adapter. The rule trades one upfront error for a
class of data-leak bugs that's hard to detect after the fact.

## Caveats

### Pre-computed embeddings are ignored

CrewAI's `Memory.recall` computes a query embedding via its own
embedder, then calls `StorageBackend.search(query_embedding, …)` -
passing only the vector, not the source text. The adapter
**discards** that embedding and threads the original query text into
khora's `recall()` via a stashing embedder installed at factory
construction. Two consequences:

- khora runs its own embedding step on the text. The embedder
  configured on `Khora()` (its dimension, model, normalisation) wins;
  the embedder configured on `crewai.Memory` only contributes the
  text-stashing side channel.
- CrewAI's HyDE / rerank step (in `recall_flow.py`) and khora's
  HyDE/rerank/temporal-anchor stack both execute. Operators paying
  the LLM bill should be aware: a single CrewAI `recall()` can spend
  tokens at both layers.

If your deployment can't tolerate the dual-LLM cost, configure
`crewai.Memory(...)` with `query_analysis_threshold=10_000` so CrewAI
skips its own analysis on most queries, leaving HyDE to khora alone.

### CrewAI's encoding LLM is not duplicated

CrewAI's `Memory.remember` runs its own LLM-driven scope / categories
/ importance analysis before calling `StorageBackend.save([record])`.
The adapter forwards those fields directly via
`Document.metadata` - `kb.remember` is called with
`entity_types=[]` and `relationship_types=[]` so khora does **not**
trigger a second extraction LLM call.

If you want khora's entity extraction to run on CrewAI records
anyway, construct `KhoraStorageBackend` directly and pass non-empty
`entity_types` / `relationship_types` via the `extraction_params` on
your `Khora()` config.

### Sync entry point only - no async loops above the adapter

`KhoraStorageBackend` is a sync class. Every async call into khora is
dispatched through `khora.integrations._sync.run_sync`, which refuses
to run from inside an existing asyncio loop (deadlock surface). Do not
call `KhoraMemory(...)` or any of its methods from inside an `async
def` - call it from a sync entry point or a worker thread.
