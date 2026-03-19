# Khora

Memory Lake library combining knowledge graphs, vector database (pgvector), and PostgreSQL for unified knowledge storage and retrieval. **This is a library, not a deployable application.**

## Project Config

```
LINEAR_TEAM: {TEAM_KEY}
```

<!-- LINEAR_TEAM: the Linear team key used for branch names and issue refs (e.g., ENG) -->

## Commands

```bash
make test              # Run tests (pytest, coverage ≥30%)
make format            # Format code (black, isort, ruff)
make lint              # Lint + typecheck (ruff, ty)
make dev               # Start local databases (postgres + neo4j)
uv run alembic upgrade head  # Run migrations
```

## Architecture

```
MemoryLake (facade) → Engine (graphrag | skeleton | vectorcypher) → StorageCoordinator
                                                    ├── PostgreSQL (documents, tenancy)
                                                    ├── pgvector (embeddings)
                                                    └── Graph backend (entities, relationships)
```

- **Engines are pluggable** — implement `MemoryEngineProtocol` in `engines/protocol.py`
- **Graph backends are interchangeable** — all implement `GraphBackend` in `storage/backends/base.py`
- **Extraction skills are YAML-defined** — see `extraction/skills/builtin/`
- **Multi-tenancy:** MemoryNamespace (sole isolation boundary)
- **Config via env vars** — prefix `KHORA_`, use `__` for nesting (e.g., `KHORA_QUERY__ENABLE_HYDE=true`)

## Key Entry Points

- `memory_lake.py` — Public API: `remember()`, `recall()`, `forget()`, `remember_batch()`, `create_namespace()`, `get_namespace_by_stable_id()`
- `storage/coordinator.py` — Backend orchestration, `TransactionContext`, `transaction()`
- `storage/factory.py` — Backend creation with shared engine pools
- `db/session.py` — `DatabaseManager` class for session/engine lifecycle
- `db/models.py` — SQLAlchemy ORM (all UUID columns use `as_uuid=True`)
- `engines/` — GraphRAG (default), Skeleton Construction, VectorCypher
- `query/engine.py` — `HybridQueryEngine` search pipeline
- `_accel.py` — Rust/NumPy/Python acceleration facade (MMR, cosine, `detect_temporal_category()`, BM25, etc.)
- `engines/vectorcypher/temporal_detection.py` — `TemporalDetector`, category-specific `RetrievalParams` for VectorCypher recall
- `pipelines/flows/ingest.py` — Document ingestion pipeline with entity ID mapping

## Engine Selection

| Use Case | Engine | Key Trait |
|----------|--------|-----------|
| Knowledge bases, entity exploration | `graphrag` | Full graph extraction, requires Neo4j/Kuzu |
| Multi-hop queries, complex relationships | `vectorcypher` | Vector + Cypher hybrid, requires Neo4j |
| Chat history, event streams, cost-sensitive | `skeleton` | Temporal-first, 5-10x fewer LLM calls, Neo4j optional |

## Testing

```bash
uv run pytest tests/unit/ -v               # Unit tests only
uv run pytest -k "test_remember" -v         # By name
uv run pytest tests/unit/test_memory_lake.py  # Single file
```

Markers: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.e2e`. Async tests use `asyncio_mode = "auto"`.

## Version Bumps

IMPORTANT: When bumping the version, always update **all four files** and regenerate lockfiles:
1. `pyproject.toml` — khora version
2. `src/khora/__init__.py` — `__version__`
3. `rust/khora-accel/Cargo.toml` — khora-accel version
4. `rust/khora-accel/pyproject.toml` — khora-accel version
5. Run `uv lock` and `cargo generate-lockfile` in `rust/khora-accel/`

## Claude Code Settings

Run `uv run ttoj install -p /path/to/your-project` to deploy shared settings, commands, and skills. This installs `.claude/settings.json` with team-wide tool permissions (git, GitHub CLI, basic shell) so engineers don't get prompted for every action.

- Add project-specific commands (lint, test, build) to the `allow` list.
- Commit `.claude/settings.json` to version control.
- Personal overrides go in `.claude/settings.local.json` (auto-gitignored by Claude Code).

See the TTOJ README for a full explanation of the settings layers.

### Linear MCP

Set up the Linear MCP server (one-time):

```bash
claude mcp add --transport http linear-server https://mcp.linear.app/mcp
```

Then run `/mcp` inside Claude Code to authenticate with your Linear account. The TTOJ settings template pre-allows all Linear MCP tools so you won't be prompted for routine operations.

## Workflow Lifecycle

Every feature, bugfix, or task follows: **Linear ticket → branch → implement → PR → done**.

If your project uses Linear (has `LINEAR_TEAM` in its CLAUDE.md):

### Starting work
1. Search Linear for an existing ticket matching the task. If none exists, create one.
2. Assign yourself (or the requesting user) to the ticket.
3. Move the ticket to **In Progress**.
4. Create a branch from `staging` (default) following the branch naming convention below. For hotfixes, branch from `main`.
5. Begin implementation.

### During work
- Commit early and often with descriptive messages.
- Keep the Linear ticket updated if scope changes or blockers arise.
- If you discover sub-tasks, create child issues in Linear.

### Finishing work
1. Run any project-defined lint/format/test commands.
2. Commit all changes with a clear message.
3. Push the branch and create a PR following PR standards below. PRs target `staging` by default; only hotfixes target `main`.
4. Add the PR link as a comment on the Linear ticket.
5. The ticket stays **In Progress** until the PR is merged, then move to **Done**.

## Git Conventions

### Branch naming
Format: `{initials}/{TICKET-ID}-{kebab-description}`
Examples: `nn/DYT-42-add-auth`, `ms/DYT-108-fix-pagination`

The `initials` and `TICKET-ID` are required. Every branch must trace back to a Linear ticket.

### Commit messages
- Use imperative mood: "Add feature", not "Added feature"
- First line: concise summary under 72 characters
- Format: `{TICKET-ID}: {description}` (e.g., `DYT-42: Add auth middleware`)
- Body (optional): explain *why*, not *what*
- Reference the Linear ticket ID when relevant

### Branching strategy
- `main` = production. `staging` = staging / default PR target.
- Feature branches are created from `staging` (default). For hotfixes, branch from `main`.
- PRs **always** target `staging`, unless it's a hotfix.
- Hotfixes: branch from `main`, PR targets `main` (production deploy).

### Rules
- Never force-push to `main`, `staging`, or shared branches.
- Rebase feature branches on the PR target branch (`staging` or `main` for hotfixes) before opening a PR when possible.
- Delete branches after merge.

## Linear Integration

Use the Linear MCP tools for all ticket operations. Refer to the `linear-workflow` skill for exact tool patterns.

### Status transitions
- **Backlog** → **Todo** → **In Progress** → **Done**
- Other states: **Canceled**, **Duplicate**
- Move to In Progress when you start working.
- Move to Done only after the PR is merged.

### Linking
- Always add the PR URL as a comment on the Linear ticket.
- Include the ticket ID (e.g., `DYT-42`) in the PR title.

## PR Standards

### Title format
`TICKET-ID: Short imperative description` (e.g., `DYT-42: Add JWT authentication`)

### Body structure
```
## Summary
- Bullet points describing what changed and why

## Test plan
- How to verify the changes work
```

- PRs target `staging` by default; hotfix PRs target `main`.
- Keep it concise. The code should speak for itself.

## Multi-Agent Coordination

When working as part of an agent team:

- Claim tasks explicitly via TaskUpdate before starting work.
- Never modify files owned by another agent without coordinating first.
- Avoid editing the same file concurrently. If unavoidable, coordinate line ranges.
- If two agents disagree on approach, escalate to the team lead.

## PRD & ADR

Major features require a PRD in `docs/prds/`. Significant architectural decisions require an ADR in `docs/adrs/`. Use `/plan:create` and `/plan:adr` to scaffold these documents. See the TTOJ repo for templates.

## Agent Teams

Use `/team:feature`, `/team:bugfix`, or `/team:review` for purpose-built teams. Default to single-agent work unless the task clearly benefits from parallelism. Role definitions are in the TTOJ repo under `content/templates/team-profiles/` and are available via the `/team:*` commands.

## Slack Integration

Commands post to team-wide default channels (deployed via TTOJ):
- `/done` → `#pull-requests` — PR opened notification with detailed context in thread.
- `/automerge` → `#pull-requests` — merge success or CI failure.
- `/plan:create` → `#engineering` — optionally shares a PRD summary with goals and requirements in thread.
- `/slack:notify` — sends an ad-hoc message to any channel or user.

Default channels are defined in the `slack-workflow` skill. Commands never fail due to Slack errors.

## Orchestrated Workflow

The recommended command sequence for feature development:

1. `/plan:create` — scaffold a PRD (optionally share on Slack)
2. `/plan:adr` — document key decisions (if needed)
3. `/plan:expand` — break PRD into Linear tickets
4. `/workflow` — pick a ticket, create branch, start work
5. Implement (solo or `/team:feature`)
6. `/done` — commit, PR, update Linear, notify Slack
7. `/audit` — review the PR

## Tool Preferences

- **Documentation lookup**: Use Context7 MCP for up-to-date library docs.
- **Linear operations**: Use Linear MCP tools (list_issues, create_issue, update_issue, etc.).
- **GitHub operations**: Use `gh` CLI for PRs, issues, and repo operations.
- **File operations**: Prefer dedicated Claude Code tools (Read, Write, Edit, Grep, Glob) over shell equivalents.
- **Web research**: Use WebSearch/WebFetch for current information beyond training data.

## Team Configuration

Default team profiles are deployed to `.ttoj/templates/team-profiles/` and referenced by the `/team:*` commands. To customize team composition for this project, edit the deployed profiles or create project-specific overrides in `.claude/commands/team/`.

## AI Changelog

After every completed task, append an entry to `docs/AI_CHANGELOG.md`. Create the file if it doesn't exist.

### Format

```
- YYYY-MM-DD: TICKET-ID: Brief description of change
```

- One line per task, max 72 characters.
- Use the Linear ticket ID (e.g., `DYT-42`) when available.
- If there is no ticket, omit the ticket ID: `- YYYY-MM-DD: Brief description of change`.
- Append new entries at the bottom of the file.
- Do not edit or remove existing entries.

## Conventions

<!-- Replace with project-specific coding conventions. Examples:              -->
<!-- - All API responses use a standard envelope: { data, error, meta }      -->
<!-- - Prefer named exports over default exports                             -->
<!-- - Database migrations must be backwards-compatible (expand-and-contract)-->
<!-- - Error messages are user-facing; keep them clear and actionable        -->

## Gotchas

- **No Docker in CI** — khora is a library; CI only runs tests, linting, and type checking
- **UUID columns use `as_uuid=True`** — all 52 UUID columns in `db/models.py` map to native Python `uuid.UUID` objects. Never use `str()` wrapping when building ORM models
- **Graph backends need `str()` at boundary** — Neo4j/Kuzu/Memgraph don't support native UUIDs, so convert at the graph DB boundary only
- **Shared engine pools** — `StorageFactory` caches engines by normalized URL. Backends sharing the same URL reuse one `AsyncEngine`. Shared-engine backends must skip `dispose()` on disconnect
- **Transactions** — use `async with coordinator.transaction() as txn:` for atomic multi-backend operations. Backend write methods accept optional `session` parameter to join an existing transaction
- **spaCy is optional** — `_HAS_SPACY` flag controls sentence splitting. Uses blank model with `sentencizer` pipe (no model download needed). Falls back to regex when spaCy is not installed
- **Logfire is optional** — `_HAS_LOGFIRE` flag in `telemetry/logfire_integration.py` controls OTEL span emission. Install with `pip install khora[logfire]`. When absent, `trace_span()` yields a no-op `Span` singleton that silently discards attribute writes (zero-cost). Consumers import `trace_span` from `khora.telemetry`, not from `logfire_integration` directly. Custom telemetry (`collector.record_*`) fires regardless of logfire presence. Khora never calls `logfire.configure()` or `logfire.instrument_*()` — that's the consumer's responsibility
- **@trace decorator** — Use `from khora.telemetry import trace` for automatic span creation. Decorates sync/async functions, auto-captures arguments as span attributes (UUID→str, list/tuple/set→count, enum→value, complex objects skipped). Supports `include`/`exclude` filters and `result` extractor for return values. When logfire is absent, short-circuits to direct function call (zero overhead). Use `@trace` for simple span-per-function patterns; use `trace_span()` context manager for complex methods needing mid-function attributes. Example: `@trace("khora.search", exclude={"query"}, result=lambda r: {"count": len(r)})`
- **Namespace versioning** — `MemoryNamespace` has two IDs: `id` (row-level, changes per version) and `namespace_id` (stable across versions). Public API methods accept `namespace_id` and resolve to the active version's `id` via DB lookup. Child table FKs reference `id`, not `namespace_id`. Resolution (`resolve_namespace`) is idempotent — accepts either ID type. This adds one indexed query per public API call (sub-ms but visible in benchmarks). If namespace versioning is removed in the future, the resolution layer and dual-ID scheme can be collapsed to a single UUID
- **Downstream consumers** — `genesis` and `khora-benchmarks` depend on khora. Check compatibility when changing public APIs. `lake.storage` is a stable public API used by both
- **Entity unique constraint** — `entities(namespace_id, name, entity_type)` has a UNIQUE constraint (migration 008). Entity upserts use `ON CONFLICT` on this constraint. Dedup migration is irreversible
- **Pre-normalized embeddings** — All embeddings are L2-normalized at ingest time. Scoring uses `batch_dot_product` instead of `batch_cosine_similarity` for ~3x speedup. Dot product of unit vectors = cosine similarity
- **MMR diversity enabled by default** — `enable_diversity=True` in `QuerySettings`. The MMR stage runs in Rust via `_accel.mmr_diversity_select` with NumPy and pure-Python fallbacks
- **`include_sources` on read methods** — `recall()`, `get_entity()`, `list_entities()`, `find_related_entities()`, and `search_entities()` accept `include_sources: bool = False`. When `False` (default), no extra query runs — zero overhead. When `True`, `_populate_sources()` batch-fetches `DocumentSource` metadata (chunked at 1 000 IDs) and populates `chunk.source_document`, `entity.source_documents`, and `relationship.source_documents` in-place
- **`ty` type checker** — Pre-commit hook runs `ty check src/` which passes clean (`All checks passed!`). If ty fails on your changes, fix the diagnostics before committing
