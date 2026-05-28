# Khora examples

Runnable tutorials for the public Khora API. Two styles:

- **Numbered tutorials (`01_*.py` through `08_*.py`)** — sequential
  learning path. Start at `01` and progress; each builds on what came
  before.
- **Scenario tutorials** (e.g. `chat_agent_langgraph.py`,
  `namespace_versioning.py`, `feedback_loop_promotion.py`) — focused on
  one pattern. Pick whichever matches what you're trying to build.

Plus per-adapter smoke tests under `integrations/<framework>/` (see
`integrations/README.md` for the strict ≤80-LOC convention).

## Default: zero infrastructure

Every example runs **without Postgres, Neo4j, or any external service**
by default. The embedded backend (`sqlite_lance` — SQLite + LanceDB
in-process) handles relational, vector, and graph storage in a single
local directory. The numbered tutorials require `OPENAI_API_KEY` for
real LLM extraction; integrations smoke tests use a deterministic mock.

```bash
uv pip install -e ".[sqlite-lance]"  # or: pip install -e ".[sqlite-lance]"
export OPENAI_API_KEY=sk-...
uv run python examples/01_hello_memory.py
# or, if you installed khora into the current venv:
python examples/01_hello_memory.py
```

That's it. No services, no setup.

## Switching to PostgreSQL + pgvector + Neo4j or SurrealDB

Every tutorial uses the same switch: pass `--config <path>`. Three
configs ship in this directory:

| Config | Backend | When to use |
|---|---|---|
| `khora.embedded.yaml` (default) | SQLite + LanceDB | Zero-infrastructure runs, evaluation, CI |
| `khora.standard.yaml` | PostgreSQL + pgvector + Neo4j | Production stack — needs `make dev` |
| `khora.surrealdb.yaml` | SurrealDB (memory mode) | Single-store alternative; **Experimental** |

```bash
# Embedded (default — no flag needed)
uv run python examples/01_hello_memory.py
python examples/01_hello_memory.py

# Production stack
make dev    # docker compose: postgres + neo4j
uv run python examples/01_hello_memory.py --config examples/khora.standard.yaml
python examples/01_hello_memory.py --config examples/khora.standard.yaml

# SurrealDB unified store
uv run python examples/01_hello_memory.py --config examples/khora.surrealdb.yaml
python examples/01_hello_memory.py --config examples/khora.surrealdb.yaml
```

For a tour of the three backends side-by-side, see
**[`storage_backend_selector.py`](storage_backend_selector.py)** below.

Any field in the YAML can be overridden via env vars (e.g.
`KHORA_LLM_MODEL=gpt-4o`), so you can pin URLs in YAML and rotate
credentials at run time.

---

## The numbered tutorials — learn the API

Start here if Khora is new to you. Each tutorial is self-contained and
under ~200 LOC.

### [`01_hello_memory.py`](01_hello_memory.py) — *VectorCypher (default)*

The smallest viable Khora program: create a namespace, remember one
fact, recall it. Mirrors the "5-line hello" every memory library
ships, translated to Khora's shape. Teaches the full default pipeline
(chunking → embeddings → KET-RAG selective extraction → graph writes →
hybrid retrieval) in 30 lines of demo code.

### [`02_per_user_preferences.py`](02_per_user_preferences.py) — *Chronicle*

The universal "user preferences" pattern with Ebbinghaus recency
decay. Two users, contradictory statements, older ones fade naturally.
Introduces Chronicle's abstention signals (`chunks_empty`,
`top_score_low`, `should_abstain`) — the cheap way to keep an LLM
from making things up about a namespace it doesn't have data on.

### [`03_document_qa_with_abstention.py`](03_document_qa_with_abstention.py) — *Chronicle*

Corpus Q&A. Ingest an HR policy corpus from JSONL; ask on-topic and
off-topic questions; observe when abstention fires. Introduces
abstention threshold tuning per corpus (the defaults are loose on
purpose). Pairs with `examples/data/hr_policies.jsonl`.

### [`04_support_ticket_graph.py`](04_support_ticket_graph.py) — *VectorCypher*

Multi-hop entity reasoning. Ingest ~100 support tickets, then run two
query shapes side-by-side: (a) "tickets about X" (vector search alone
wins), (b) "context around customer Y" (graph traversal wins). Shows
`find_related_entities` and the cross-document entity dedup that
VectorCypher's HippoRAG-style retrieval relies on.

### [`05_temporal_podcast.py`](05_temporal_podcast.py) — *Chronicle*

The Graphiti "Kendra loved Adidas then Nike" example, ported to
Khora. Timestamped utterances, semantically similar but temporally
ordered; recall picks the latest stance via Ebbinghaus decay without
any explicit `forget()`. Then shows point-in-time recall
(`start_time` / `end_time`) restricting the window before decay even
fires.

### [`06_agent_memory_tool.py`](06_agent_memory_tool.py) — *Chronicle*

Memory-as-a-tool pattern. Two bare async functions — `remember` and
`recall` — that an agent can call. Three-turn loop: cold namespace
(abstain), user supplies fact (store), user re-asks (answer from
memory). Framework-agnostic; the framework chapters (LangGraph,
CrewAI, OpenAI Agents) layer on top.

### [`07_expertise_config_resumes.py`](07_expertise_config_resumes.py) — *VectorCypher (full-extraction lane)*

Full-extraction lane — the replacement for the removed graphrag
engine. Four CV blurbs, deliberate naming variants ("Stripe" vs
"Stripe Inc."), domain-specific `ExpertiseConfig`. Shows
`engine_kwargs={"skeleton_core_ratio": 1.0}` for "extract everything"
on small dense corpora, plus how to canonicalize entities in the
system prompt rather than via post-hoc dedup.

> **Known issue.** The `unify_entities` pipeline helper that the demo
> calls hits a method (`storage.graph.get_entities_by_namespace`) that
> doesn't exist on any backend (Bug #9 in the repo audit). The demo
> catches the AttributeError and continues — cross-document dedup is
> currently degraded to whatever the LLM canonicalized via the system
> prompt. Demo still runs end-to-end and shows the entity-centric APIs
> correctly.

### [`08_slack_archive_bulk.py`](08_slack_archive_bulk.py) — *Skeleton*

Cost optimization. ~50 synthetic Slack messages (75% noise, 25%
signal). Shows Skeleton's PageRank-style importance scorer picking
~10% of chunks for full LLM extraction — and the napkin math: 5k LLM
calls (Skeleton) vs ~35k (VectorCypher KET-RAG) vs 50k (full
extraction) on 50k messages. Also demos `remember_batch` with
`on_progress`, and time-filtered recall with SQL pushdown.

---

## Scenario tutorials — one pattern each

Pick the ones that match what you're building.

### Memory + agent frameworks

#### [`chat_agent_langgraph.py`](chat_agent_langgraph.py) — *VectorCypher (default)*

LangGraph integration via the `KhoraStore` adapter. Two-node
StateGraph (recall → respond) backed by Khora memory. Drops the
graph, rebuilds it with the same store, and the memory survives —
demonstrating that memory belongs to the namespace, not the graph
lifecycle.

#### [`resume_session_openai_agents.py`](resume_session_openai_agents.py) — *Chronicle*

OpenAI Agents SDK integration via `KhoraSession`. Two simulated
sessions ("Monday", "Tuesday") on the same namespace, different
session_ids. Shows session as a per-conversation handle, namespace as
the actual isolation boundary, and `forget_session` for
GDPR-friendly per-session cleanup.

#### [`multi_agent_shared_memory_crewai.py`](multi_agent_shared_memory_crewai.py) — *VectorCypher (default)*

CrewAI multi-agent pattern via the `KhoraMemory` adapter. Two agents
(researcher, writer), three namespaces (one shared, two per-agent
private). Shows the collaboration shape: researcher writes to shared,
writer reads from shared and replies. Also verifies the isolation
guarantee — writer's private namespace doesn't see researcher's
TODOs.

#### [`tool_router_learning_chronicle.py`](tool_router_learning_chronicle.py) — *Chronicle*

Memory as a routing oracle. Two tools that look interchangeable;
agent learns which one works for which request by recording
outcomes. Recall-based route selection: find similar prior requests,
tally tools that succeeded vs failed, pick the winner. Convergence
visible across ~10 requests. The pattern transfers directly to
"which RAG corpus", "which API endpoint", "which sub-agent".

### Time and consolidation

#### [`temporal_range_query_chronicle.py`](temporal_range_query_chronicle.py) — *Chronicle*

Chronicle's killer demo. ~30 events spread over six simulated months
across five narrative threads (postgres migration, deploy pipeline,
security audit, rebrand, hiring). Five query shapes: unconstrained,
single-month, quarter, anchored-by-event, negative (empty window).
Shows SQL pushdown for the time filter — semantic ranking only runs
on already-filtered candidates.

#### [`dream_phase_rule_extraction_chronicle.py`](dream_phase_rule_extraction_chronicle.py) — *Chronicle*

Offline maintenance — Khora's "dream phase". Ten observations,
near-duplicates clustered into a representative "rule" event via
`OpKind.CLUSTER_EVENTS`. Walks the `dry-run` → `apply` workflow with
its bi-temporal soft-delete semantics. Shows the guardrails (7-day
retention floor, kill-switch env var, advisory lock,
snapshot-before-mutate) wired through a real op.

#### [`feedback_loop_promotion.py`](feedback_loop_promotion.py) — *Chronicle*

User feedback shifts memory ranking. Thumbs-up re-ingests with a
fresh `occurred_at` to outrank older siblings; thumbs-down calls
`forget(document_id, namespace=...)` to cascade-delete. Combines
implicit decay with explicit gates on user signal — the operator
feedback pattern wired end-to-end.

### Storage and lifecycle (power-user)

#### [`namespace_versioning.py`](namespace_versioning.py) — *VectorCypher (default)*

Version a namespace end-to-end. Create v1, ingest data, cut v2 via
`storage.create_namespace_version(previous_version=v1)`, ingest fresh
data into v2. Then read either version's data via the storage layer
using its row id (the "version handle"). Demonstrates the dual-UUID
model on `MemoryNamespace` (`namespace_id` is stable; `id` is
per-version) and the bridge `storage.resolve_namespace(stable_id)`.

#### [`storage_backend_selector.py`](storage_backend_selector.py) — *VectorCypher (default)*

The same Python code against three backends — `sqlite_lance` (default),
`postgresql + neo4j`, and `surrealdb`. Pick by `--config`. Walks
`create_namespace` → `remember` → `recall` → `list_entities` on each
to show the API is backend-agnostic. Includes per-backend trade-offs
and the SurrealDB v0.12.0 known-issues block.

### Operator concerns

#### [`neo4j_debug_logging.py`](neo4j_debug_logging.py) — *standalone*

Routes neo4j Python driver `DEBUG` logs through the loguru sink Khora
uses. Useful when a graph query is misbehaving and you want to see
the actual Bolt traffic without losing Khora's own logs.

---

## Engine coverage and when to pick each

| Engine | Strengths | Examples |
|---|---|---|
| **VectorCypher** (default) | Multi-hop graph traversal, entity reasoning, query complexity routing, HippoRAG-style retrieval | `01_hello_memory`, `04_support_ticket_graph`, `07_expertise_config_resumes`, `chat_agent_langgraph`, `multi_agent_shared_memory_crewai`, `namespace_versioning`, `storage_backend_selector` |
| **Chronicle** | Event streams, bi-temporal model, Ebbinghaus decay, abstention signals, time-bounded queries; no graph backend required | `02_per_user_preferences`, `03_document_qa_with_abstention`, `05_temporal_podcast`, `06_agent_memory_tool`, `temporal_range_query_chronicle`, `feedback_loop_promotion`, `resume_session_openai_agents`, `tool_router_learning_chronicle`, `dream_phase_rule_extraction_chronicle` |
| **Skeleton** | Cost-efficient hybrid search; ~10% LLM extraction; long-form / large corpora | `08_slack_archive_bulk` |

### Current distribution and coverage gap

Out of 18 tutorial files:

- VectorCypher: **8 (44%)**
- Chronicle: **9 (50%)**
- Skeleton: **1 (6%)**

The 60%-VectorCypher target isn't met yet — Chronicle is heavier than
intended because the time / decay / abstention story is uniquely
Chronicle's, and those scenarios mapped cleanly onto Chronicle-shaped
demos. To rebalance to 60% VectorCypher we'd need three more
VectorCypher-leaning tutorials. Candidates that would showcase
strengths not yet demonstrated:

- **Investigation / case-file workflow** — ingest case notes, query
  "everyone connected to subject X within N hops" — uses
  `find_related_entities`, graph depth tuning, query complexity routing.
- **Knowledge graph QA over a corpus of papers / wiki** — multi-document
  ingestion + entity-mediated cross-document retrieval. Shows the
  `DualNodeManager` (Chunk + Entity nodes linked via `MENTIONED_IN`)
  paying off for "which chunk supports which claim" answers.
- **Comparison-shaped queries** — VectorCypher's
  `QueryComplexityRouter` classifies queries as SIMPLE / MODERATE /
  COMPLEX and changes the retrieval strategy. A demo that runs the
  same corpus through all three shapes makes the router visible.

These are listed for review, not committed work. Drop any of them in
without disturbing the existing numbering.

## Backend coverage

Every tutorial above runs on both `sqlite_lance` (embedded, default)
and `postgresql + neo4j` (standard). SurrealDB compatibility is
partial — see the caveat block in `khora.surrealdb.yaml`. Known
SurrealDB v0.12.0 limitations affect:

- `find_related_entities` (demo 04) — SurrealQL parse error in
  `get_neighborhood`.
- `recall()` chunks=0 on the skeleton + vectorcypher engines (data IS
  persisted; the search path filters them out).
- `chunk.source_timestamp` returns `None` on recall — affects demos 05
  and 08.
- Chronicle event/fact persistence emits "No backend supports method
  'write_events'" — affects demos 02, 03, 05, 06, and the four
  scenario tutorials in the Chronicle column.

`storage_backend_selector.py` runs cleanly on all three (it doesn't
exercise the broken paths).

## File layout

```
examples/
├── 01_hello_memory.py                    # numbered tutorials — start here
├── 02_per_user_preferences.py
├── 03_document_qa_with_abstention.py
├── 04_support_ticket_graph.py
├── 05_temporal_podcast.py
├── 06_agent_memory_tool.py
├── 07_expertise_config_resumes.py
├── 08_slack_archive_bulk.py
│
├── chat_agent_langgraph.py               # scenarios — one pattern each
├── resume_session_openai_agents.py
├── multi_agent_shared_memory_crewai.py
├── tool_router_learning_chronicle.py
├── temporal_range_query_chronicle.py
├── dream_phase_rule_extraction_chronicle.py
├── feedback_loop_promotion.py
├── namespace_versioning.py
├── storage_backend_selector.py
├── neo4j_debug_logging.py                # operator helper
│
├── khora.embedded.yaml                   # configs — picks the backend
├── khora.standard.yaml
├── khora.surrealdb.yaml
│
├── data/                                 # JSONL corpora
│   ├── hr_policies.jsonl                 #   used by 03
│   └── support_tickets.jsonl             #   used by 04
│
├── _helpers/                             # khora_for_examples + install_mock_llm
├── config/                               # expertise + litellm sub-configs (used by 07)
└── integrations/                         # per-adapter smoke tests (see its README)
```

## Convention summary

| Question | Answer |
|---|---|
| **Where do examples default?** | `sqlite_lance` (zero infra) via `khora.embedded.yaml`. |
| **How to switch backend?** | `--config examples/khora.standard.yaml` (or `khora.surrealdb.yaml`) — every tutorial accepts the flag. |
| **How to switch engine?** | Each tutorial picks the engine that matches its scenario via `Khora(config, engine="…")`. Read the docstring for the rationale. |
| **Do examples make real LLM calls?** | Yes — numbered tutorials require `OPENAI_API_KEY`. Integration smoke tests under `integrations/<framework>/` use the deterministic mock LLM. |
| **Override individual fields without editing YAML?** | Export the matching `KHORA_*` env var (e.g. `KHORA_LLM_MODEL=gpt-4o`) — pydantic-settings overlays it onto the loaded config. |
| **What if my example needs a SurrealDB-broken path?** | Mark the YAML with the affected demo and skip on SurrealDB — the caveat block in `khora.surrealdb.yaml` is the canonical list. |
