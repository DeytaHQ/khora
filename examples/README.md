# Khora examples

Runnable tutorials for the public Khora API, organized into four numbered
tiers. Work through them in order, or jump to the tier that matches what
you're building:

- **`00_quickstart/`** — the core loop (remember, recall, abstain, forget, isolate).
- **`10_core_apis/`** — the API surface (batch ingest, recall filters, ontology config, graph reads).
- **`20_integrations/`** — framework adapters (LangGraph, OpenAI Agents, CrewAI).
- **`30_workloads/`** — end-to-end scenarios that compose the APIs into real apps.

Plus per-adapter smoke tests under `integrations/<framework>/` (see
[`integrations/README.md`](integrations/README.md) for the strict ≤80-LOC convention).

## Default: zero infrastructure

Every tutorial runs **without Postgres, Neo4j, or any external service** by
default. The embedded backend (`sqlite_lance` — SQLite + LanceDB in-process)
handles relational, vector, and graph storage in a single local directory. The
numbered tutorials require `OPENAI_API_KEY` for real LLM extraction; the
integration smoke tests use a deterministic mock.

```bash
uv pip install -e ".[sqlite-lance]"  # or: pip install -e ".[sqlite-lance]"
export OPENAI_API_KEY=sk-...
uv run python examples/00_quickstart/01_remember_recall.py
# or, if khora is installed into the current venv:
python examples/00_quickstart/01_remember_recall.py
```

That's it. No services, no setup.

## Switching to PostgreSQL + pgvector + Neo4j

Every tutorial takes the same switch: pass `--config <path>`. Two configs ship
in this directory:

| Config | Backend | When to use |
|---|---|---|
| `khora.embedded.yaml` (default) | SQLite + LanceDB | Zero-infrastructure runs, evaluation, CI |
| `khora.standard.yaml` | PostgreSQL + pgvector + Neo4j | Production stack — needs `make dev` |

```bash
# Embedded (default — no flag needed)
python examples/00_quickstart/01_remember_recall.py

# Production stack
make dev    # docker compose: postgres + neo4j
python examples/00_quickstart/01_remember_recall.py --config examples/khora.standard.yaml
```

Any field in the YAML can be overridden via env vars (e.g.
`KHORA_LLM_MODEL=gpt-4o`), so you can pin URLs in YAML and rotate credentials at
run time.

---

## `00_quickstart/` — the core loop

Start here if Khora is new to you. Each is self-contained and under ~150 LOC.

- [`01_remember_recall.py`](00_quickstart/01_remember_recall.py) — *skeleton* —
  Semantic recall vs. a naive keyword scan: create a namespace, remember a few
  facts, recall them by meaning rather than shared words.
- [`02_grounded_answers.py`](00_quickstart/02_grounded_answers.py) — *chronicle* —
  Grounded answers and abstention. Use `max_raw_vector_score` /
  `abstention_signals` to refuse when the corpus has nothing on-topic.
- [`03_forget_what_was_wrong.py`](00_quickstart/03_forget_what_was_wrong.py) — *skeleton* —
  Explicit unlearning: `forget()` a memory by `document_id`, then re-remember
  the correction.
- [`04_namespaces_for_users.py`](00_quickstart/04_namespaces_for_users.py) — *skeleton* —
  Per-user isolation: bury a needle in one namespace and prove it never
  surfaces from another.

## `10_core_apis/` — the API surface

The everyday calls, one concept at a time. Most run on VectorCypher (the default); image ingestion (06) uses skeleton.

- [`01_remember_batch.py`](10_core_apis/01_remember_batch.py) —
  Bulk ingestion with `remember_batch` and `on_progress` (uses `data/support_tickets.jsonl`).
- [`02_recall_with_filters.py`](10_core_apis/02_recall_with_filters.py) —
  `recall()` with `limit`, `min_similarity`, and `mode` (the search channels).
- [`03_ontology_config.py`](10_core_apis/03_ontology_config.py) —
  Constrain extraction with `entity_types` / `relationship_types`.
- [`04_recall_entities_and_relationships.py`](10_core_apis/04_recall_entities_and_relationships.py) —
  Inspect the graph khora builds at `remember` time.
- [`05_find_related_entities.py`](10_core_apis/05_find_related_entities.py) —
  Explore the entity / relationship graph with `find_related_entities`.
- [`06_expertise_config.py.py`](10_core_apis/06_expertise_config.py) — 
  Configure and use ontology as Khora ExpertiseConfig
- [`07_image_ingestion.py`](10_core_apis/07_image_ingestion.py) — *skeleton* —
  Image ingestion: describe a set of figures with a vision model, remember the
  descriptions, then answer one question that spans several images (uses
  `OPENAI_API_KEY`).

## `20_integrations/` — framework adapters

Full tutorials wiring Khora into an agent framework. (Minimal byte-identical
smoke tests live under `integrations/<framework>/` — see its README.)

- [`01_langgraph.py`](20_integrations/01_langgraph.py) — *chronicle* —
  Chat agent with long-term memory via the LangGraph `KhoraStore` adapter.
- [`02_openai_agents.py`](20_integrations/02_openai_agents.py) — *chronicle* —
  Resume a chat session days later with the OpenAI Agents SDK `KhoraSession`.
- [`03_crewai_multi_agent.py`](20_integrations/03_crewai_multi_agent.py) — *vectorcypher* —
  Multi-agent shared memory via the CrewAI `KhoraMemory` adapter.

## `30_workloads/` — end-to-end scenarios

Applications you'd actually ship. Each composes the core APIs around one shape.

- [`01_per_user_preferences.py`](30_workloads/01_per_user_preferences.py) — *chronicle* —
  Per-user preferences with temporal drift; recency decay surfaces the latest stance.
- [`02_document_qa_with_abstention.py`](30_workloads/02_document_qa_with_abstention.py) — *chronicle* —
  Document Q&A with multi-signal abstention (uses `data/hr_policies.jsonl`).
- [`03_support_ticket_graph.py`](30_workloads/03_support_ticket_graph.py) — *vectorcypher* —
  Support tickets → knowledge graph: when vector search wins vs. when multi-hop
  traversal wins (uses `data/support_tickets.jsonl`).
- [`04_agent_chat_with_memory.py`](30_workloads/04_agent_chat_with_memory.py) — *chronicle* —
  Memory-as-a-tool: two tools (`remember` / `recall`), branch on `should_abstain`.
- [`05_dream_phase_consolidation.py`](30_workloads/05_dream_phase_consolidation.py) — *chronicle* —
  Offline dream-phase event clustering, walked through `dry-run` → `apply`.
- [`06_namespace_versioning.py`](30_workloads/06_namespace_versioning.py) — *vectorcypher* —
  Version a namespace via the storage API (the dual-UUID model).
- [`07_temporal_range_query.py`](30_workloads/07_temporal_range_query.py) — *chronicle* —
  Time-bounded recall with `start_time` / `end_time` and SQL pushdown.
- [`08_resume_search.py`](30_workloads/08_resume_search.py) — *vectorcypher* —
  Full extraction + cross-document entity resolution with an `ExpertiseConfig`
  (uses `data/resumes.jsonl`).
- [`09_bulk_archive.py`](30_workloads/09_bulk_archive.py) — *skeleton* —
  Bulk Slack-archive ingest — the cost-story demo (~10% extraction).
- [`10_tool_router_learning.py`](30_workloads/10_tool_router_learning.py) — *chronicle* —
  Memory as a routing oracle: learn which tool resolves which request.
- [`11_multimodal_document_qa.py`](30_workloads/11_multimodal_document_qa.py) — *vectorcypher* —
  Multimodal document QA: parse a folder of NASA Mars-rover markdown (text +
  `<img>` figures), vision-describe the figures, remember everything, then answer
  cross-rover questions with retrieval-augmented generation (uses `OPENAI_API_KEY`).

## Operator helper

- [`neo4j_debug_logging.py`](neo4j_debug_logging.py) — route the neo4j driver's
  `DEBUG` logs through Khora's loguru sink to inspect Bolt traffic.

---

## Engine coverage

Each tutorial picks the engine that matches its scenario via
`Khora(config, engine="…")` — read the docstring for the rationale.

| Engine | Strengths | Examples |
|---|---|---|
| **VectorCypher** (default) | Multi-hop graph traversal, entity reasoning, query-complexity routing, hybrid retrieval | `10_core_apis/{01-05}`, `20_integrations/03`, `30_workloads/{03,06,08,11}` |
| **Chronicle** | Event streams, bi-temporal model, Ebbinghaus decay, abstention signals, time-bounded queries; no graph backend required | `00_quickstart/02`, `20_integrations/{01,02}`, `30_workloads/{01,02,04,05,07,10}` |
| **Skeleton** | Cost-efficient hybrid search; ~10% LLM extraction; long-form / large corpora | `00_quickstart/{01,03,04}`, `10_core_apis/06`, `30_workloads/09` |

Every tutorial runs on both `sqlite_lance` (embedded, default) and
`postgresql + neo4j` (standard).

## File layout

```
examples/
├── 00_quickstart/                  # the core loop — start here
│   ├── 01_remember_recall.py
│   ├── 02_grounded_answers.py
│   ├── 03_forget_what_was_wrong.py
│   └── 04_namespaces_for_users.py
├── 10_core_apis/                   # the API surface
│   ├── 01_remember_batch.py
│   ├── 02_recall_with_filters.py
│   ├── 03_ontology_config.py
│   ├── 04_recall_entities_and_relationships.py
│   ├── 05_find_related_entities.py
│   └── 06_image_ingestion.py
├── 20_integrations/                # framework adapter tutorials
│   ├── 01_langgraph.py
│   ├── 02_openai_agents.py
│   └── 03_crewai_multi_agent.py
├── 30_workloads/                   # end-to-end scenarios (01–11)
│   ├── 01_per_user_preferences.py
│   ├── …
│   └── 11_multimodal_document_qa.py
│
├── khora.embedded.yaml             # configs — picks the backend
├── khora.standard.yaml
│
├── data/                           # corpora + image figures
│   ├── hr_policies.jsonl           #   used by 30_workloads/02
│   ├── resumes.jsonl               #   used by 30_workloads/08
│   ├── support_tickets.jsonl       #   used by 10_core_apis/01 + 30_workloads/03
│   ├── images/                     #   public-domain figures used by 10_core_apis/06 (see CREDITS.md)
│   └── mars_rovers/                #   NASA markdown corpus + figures used by 30_workloads/11
│
├── config/                         # expertise + litellm sub-configs
├── _helpers/                       # shared khora fixtures + mock LLM
├── integrations/                   # per-adapter smoke tests (see its README)
└── neo4j_debug_logging.py          # operator helper
```

## Convention summary

| Question | Answer |
|---|---|
| **Where do examples default?** | `sqlite_lance` (zero infra) via `khora.embedded.yaml`. |
| **How to switch backend?** | `--config examples/khora.standard.yaml` — every tutorial accepts the flag. |
| **How to switch engine?** | Each tutorial picks the engine that matches its scenario via `Khora(config, engine="…")`. Read the docstring for the rationale. |
| **Do examples make real LLM calls?** | Yes — numbered tutorials require `OPENAI_API_KEY`. Integration smoke tests under `integrations/<framework>/` use the deterministic mock LLM. |
| **Override individual fields without editing YAML?** | Export the matching `KHORA_*` env var (e.g. `KHORA_LLM_MODEL=gpt-4o`) — pydantic-settings overlays it onto the loaded config. |
