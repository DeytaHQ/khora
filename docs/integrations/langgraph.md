# LangGraph integration

`khora.integrations.langgraph.KhoraStore` implements LangGraph's
`BaseStore` interface so a `StateGraph` can use khora as its long-term
semantic memory in one line:

```python
graph = builder.compile(store=KhoraStore(kb, user_id="user-1234"))
```

The adapter wraps `Khora.remember` / `Khora.recall` / `Khora.forget` and
maps LangGraph's `(tuple[str, ...], str, dict)` item shape onto khora
documents. Each `(namespace_root, user_id)` pair gets a deterministic
khora `namespace_id` (UUID5), so a second `KhoraStore` over the same
user sees the same memory.

## Scope (v0.13)

- `KhoraStore` — semantic long-term memory store. **Shipped.**
- `KhoraCheckpointer` — **NOT shipped**. LangGraph's
  `PostgresSaver` (in `langgraph-postgres`) already covers the
  opaque-blob checkpoint surface and khora offers no differentiator
  there. Revisit only if a single-DB-dependency story matters to a
  real user.

## Install

```bash
pip install 'khora[langgraph]'
```

This pulls `langgraph>=1.0,<2.0`. The adapter is also registered under
the `khora.integrations` entry-point group, so `discover()` returns it
without any explicit registration.

## Constructor

| arg | default | notes |
| --- | --- | --- |
| `kb` | — | A connected `Khora` instance. Adapter does NOT own the lifecycle. |
| `user_id` | — | Required, ≥ 8 chars, not in `{"", "default", "anon", "anonymous", "user", "test"}`. Disaster-mode prevention per #618. |
| `namespace_root` | `"user_id"` | Bucket key under which this app's LangGraph namespaces live. |
| `app_id` | `"langgraph"` | Free-form app identifier stamped into stored metadata. |
| `namespace_sep` | `"/"` | Single-character separator used to flatten tuple namespaces. Must not appear in any tuple segment. |
| `index_config` | `None` | Optional LangGraph `IndexConfig`. Only `dims` is consulted — must match khora's embedder dim or construction raises. |
| `skill_name` | `"general_entities"` | khora extraction skill name forwarded to `remember`. |
| `entity_types` | `[]` | Extraction whitelist. Empty list disables extraction (pure KV blob mode). |
| `relationship_types` | `[]` | Same — empty disables. |

## Method semantics

All 6 async methods are first-class. Sync variants (`put`, `get`,
`search`, `delete`, `list_namespaces`, `batch`) bridge through
`khora.integrations._sync.run_sync`, which **rejects calls made from
inside a running event loop**. From inside a graph node, use the async
methods. From a notebook or sync script, use the sync ones.

- `aput` → `Khora.remember` with `external_id` derived from
  `(flat_namespace, key)`. Overwriting an existing item deletes the
  previous document first so chunks don't accumulate.
- `aget` → `Khora.storage.get_document_by_external_id` then project
  metadata back to a LangGraph `Item`. Returns `None` for foreign
  documents (no `lg_namespace` in metadata).
- `asearch(query=...)` → `Khora.recall` then map chunks to
  `SearchItem`. Without `query`, falls back to a `list_documents`
  scan. `filter` is applied client-side (exact match only in v1).
- `adelete` → `Khora.forget`. Missing keys are a silent no-op,
  matching `InMemoryStore` semantics.
- `alist_namespaces` — list documents in the bound khora namespace and
  aggregate distinct `lg_namespace` tuples. **O(N_documents)** scan;
  acceptable for bounded LangGraph workloads. Track a dedicated
  table at >= O(10⁴) docs.
- `abatch` → serial dispatch over the per-op methods.

### Ignored kwargs

- `ttl` (per-item) — khora has no per-item TTL. The adapter accepts it
  to satisfy the interface and emits one `RuntimeWarning` per
  `KhoraStore` instance. Use `Khora.forget_session` for bulk cleanup.
- `index=False` — khora always embeds. The adapter accepts the kwarg
  and emits one `RuntimeWarning` per `KhoraStore` instance; items
  remain retrievable.

## Quickstart

```python title="example.py"
"""LangGraph + khora example — long-term memory via ``KhoraStore``.

Runs without Postgres, Neo4j, or an API key. The mock LLM patches
``litellm.acompletion`` / ``litellm.aembedding`` so the example is
hermetic. The khora fixture spins up an in-memory ``sqlite_lance``
backend in a tmp dir.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import TypedDict

# Add repo root to sys.path so ``examples._helpers`` is importable when
# this script is run from its own directory (CI smoke loop does that).
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from langgraph.graph import StateGraph  # noqa: E402

from examples._helpers import embedded_khora, install_mock_llm  # noqa: E402
from khora.integrations.langgraph import KhoraStore  # noqa: E402


class State(TypedDict):
    note: str


async def main() -> None:
    install_mock_llm()

    async with embedded_khora() as kb:
        store = KhoraStore(kb, user_id="example-user-1234")

        async def write_note(state: State) -> State:
            await store.aput(("memories",), "note-1", {"text": state["note"]})
            return state

        builder = StateGraph(State)
        builder.add_node("write", write_note)
        builder.set_entry_point("write")
        builder.set_finish_point("write")
        graph = builder.compile(store=store)

        await graph.ainvoke({"note": "the sky is blue today"})

        item = await store.aget(("memories",), "note-1")
        assert item is not None
        print(f"Stored memory: {item.value['text']!r}")

        namespaces = await store.alist_namespaces()
        print(f"Namespaces in store: {namespaces}")


if __name__ == "__main__":
    asyncio.run(main())
```

The block above is enforced byte-identical against
`examples/integrations/langgraph/example.py` by
`tools/check_examples_drift.py` (CI gate).

## Limits and future work

- Filter operators (`$gt`, `$lt`, ...) — v1 supports exact match only.
  Operator support is a clean addition behind a feature flag.
- `alist_namespaces` SQL pushdown — the current O(N) scan is fine for
  typical workloads but not for hot multi-tenant deployments. A
  `SELECT DISTINCT metadata->'lg_namespace'` helper on the storage
  layer would fix it.
- Checkpointer — explicit non-goal (see "Scope" above).
