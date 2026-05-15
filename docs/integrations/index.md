# Integrations

khora ships first-party adapters for the major agentic-framework
ecosystems. Each adapter lives in its own optional extra so importing
`khora` never drags in a framework you don't use, and the adapter
module itself imports the framework lazily — the top-level package
load is free even with all five extras installed. Every adapter
satisfies one of two runtime-checkable Protocols in
`khora.integrations.protocol` (`MemoryAdapter` or `RetrieverAdapter`)
and registers via the `khora.integrations` entry-point group so
downstream tooling can discover what is installed.

## Adapters

| Framework | Install | Khora surface |
|---|---|---|
| [CrewAI](crewai.md) | `pip install khora[crewai]` | `KhoraMemory` — drop-in storage backend for CrewAI's unified `Memory`. |
| [LangGraph](langgraph.md) | `pip install khora[langgraph]` | `KhoraStore` — `BaseStore` implementation for `StateGraph` semantic long-term memory. |
| [Google ADK](google_adk.md) | `pip install khora[google-adk]` | `KhoraMemoryService` — `BaseMemoryService` drop-in for ADK `Runner`. |
| [OpenAI Agents SDK](openai_agents.md) | `pip install khora[openai-agents]` | `KhoraSession` (`SessionABC`), `khora_recall_tool`, `KhoraMemoryHooks` — compose for session memory, recall-as-tool, and auto-persist. |
| [LlamaIndex](llamaindex.md) | `pip install khora[llamaindex]` | `KhoraRetriever` (async `BaseRetriever`), `KhoraMemoryBlock`, and the deprecated `KhoraChatStore`. |

All five adapters share the same khora primitives — `Khora.remember`,
`Khora.recall`, `Khora.forget`, and `Khora.submit_batch` — so a single
khora instance can back several frameworks at once. Each adapter
documents its namespace-resolution rule (typically a UUID5 derived
from framework-native identifiers) so two instances pointed at the
same khora deployment see the same memory without a shared registry.

### `crewai` and `google-adk` cannot be installed together

The `crewai` extra transitively pins `opentelemetry-api<1.35` and the
`google-adk` extra pins `>=1.36`. They are declared as mutually
exclusive in `[tool.uv].conflicts`, so `uv sync --all-extras` is
rejected. Pick one:

```bash
# Crewai combo (CI default)
uv sync --all-extras --no-extra google-adk
# Or via Makefile: make install

# Google ADK combo
uv sync --all-extras --no-extra crewai
# Or via Makefile: make install-adk
```

The other three adapters (`langgraph`, `openai-agents`, `llamaindex`)
have no transitive conflicts and install cleanly alongside either
combo. If you need both crewai and google-adk in the same process,
use two separate virtual environments.

## Stability

The OpenAI Agents adapter is tagged **experimental** while upstream
remains pre-1.0 (17 releases in 7 months leading up to `v0.17`). The
CrewAI, LangGraph, Google ADK, and LlamaIndex adapters are tagged
experimental for now and will be promoted to stable once a full khora
minor ships without a breaking change to the adapter surface. See
each adapter's page for its specific framework version pin and
upstream compatibility notes.

## Writing your own adapter

The integration foundation is intentionally small. To add a new
framework adapter:

- Implement `khora.integrations.protocol.MemoryAdapter` (write side)
  or `RetrieverAdapter` (read side), or both. Both are
  runtime-checkable Protocols.
- Import the target framework lazily inside the adapter module — top-level
  imports of optional frameworks are linted out by
  `tools/check_optional_imports.py` in CI.
- Map framework-native identifiers to khora namespaces deterministically
  (UUID5 over an adapter-scoped salt is the established pattern).
- Register the adapter under the `khora.integrations` entry-point
  group in `pyproject.toml`, or call `khora.integrations.register()` for
  test-only registration.
- Use `khora.integrations._sync.run_sync` if you need to bridge a sync
  framework callback into khora's async API — it raises if invoked
  from inside a running event loop, surfacing the deadlock surface
  loudly rather than hanging.

See any of the five shipped adapters for a working template — they
range from ~150 LOC (CrewAI) to ~600 LOC (OpenAI Agents) and exercise
every part of the foundation.
