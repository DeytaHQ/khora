# ADR-024: Memory Lake Public API

- **Status:** Accepted
- **Date:** 2026-04-16
- **Deciders:** Khora architecture team

## Context

ADR-022 stabilized the extraction-skills dataclasses
(`ExpertiseConfig` et al.) that callers pass into
`MemoryLake.remember()`. It did not cover the rest of the surface
downstream packages already depend on: the `MemoryLake` class itself,
result dataclasses returned from `remember()` / `recall()` /
`remember_batch()`, the `KhoraConfig` settings hierarchy, a few
supporting helpers, and the `khora.chat` module that `genesis` drives
its TUI from.

A new first-party consumer (`khora-cli`, tracked as DYT-2655) is about
to take a hard dependency on a subset of this surface. Before adding a
fourth consumer — on top of `genesis`, `khora-benchmarks`, and
`khora-explorer` — we should ratify what is actually stable so that
future refactors inside khora do not silently break published
releases of its consumers.

This ADR is documentation-only. It does not add, rename, or change any
runtime behaviour; it describes the symbols that are **already** in
`__all__` on the top-level `khora` package plus the modules those
consumers import today.

### Observed downstream imports

- `genesis`: `from khora import MemoryLake`;
  `from khora.core.models import MemoryNamespace`;
  `from khora.config import KhoraConfig`;
  `from khora.config.llm import LiteLLMConfig, configure_litellm`;
  `from khora.config.schema import LLMSettings, PipelineSettings, StorageSettings`;
  `from khora.chat import ChatEngine, ChatResponse, PersonaConfig, load_persona_config`.
- `khora-benchmarks`: `from khora import MemoryLake`;
  `from khora.config import KhoraConfig`;
  `from khora.config.schema import LLMSettings, PipelineSettings, QuerySettings`;
  `from khora.query import SearchMode`;
  plus private imports into `khora.engines.vectorcypher` and
  `khora.extraction.chunkers` (documented here as **unstable** —
  benchmarks pin a specific khora version on purpose).
- `khora-explorer`: `from khora.extraction.skills.base import …`
  (covered by ADR-022); `from khora.config.schema import KhoraConfig`.
- `khora-cli` (new, DYT-2655): will import
  `from khora import MemoryLake, KhoraConfig, SearchMode, LLMUsage,
  RememberResult, RecallResult, BatchResult`;
  `from khora.extraction.binary_readers import extract_if_needed`;
  `from khora.logging_config import setup_logging`.

## Decision

The symbols listed below are the **stable Memory Lake public API**.
Additive changes are permitted in minor releases; breaking changes
require a major version bump and prior coordination with the consumer
list above.

### Top-level `khora.*` re-exports

Re-exported from `khora/__init__.py` (`__all__` is the machine-readable
contract):

| Symbol | Source |
| --- | --- |
| `MemoryLake` | `khora.memory_lake.MemoryLake` |
| `SearchMode` | `khora.query.engine.SearchMode` |
| `RememberResult` | `khora.memory_lake.RememberResult` |
| `RecallResult` | `khora.memory_lake.RecallResult` |
| `BatchResult` | `khora.memory_lake.BatchResult` |
| `Stats` | `khora.memory_lake.Stats` |
| `LLMUsage` | `khora.memory_lake.LLMUsage` (DYT-645, also covered by Poros/Peras coordination) |
| `KhoraConfig` | `khora.config.schema.KhoraConfig` |
| `KhoraError` | `khora.exceptions.KhoraError` |
| `DocumentSource` | `khora.core.models.document.DocumentSource` |
| `EventType` | `khora.core.models.event.EventType` |
| `SemanticFilter` | `khora.hooks.SemanticFilter` |
| `create_engine` / `list_engines` / `register_engine` | `khora.engines` |
| `ExpertiseConfig` / `EntityTypeConfig` / `RelationshipTypeConfig` | ADR-022 |

### `khora.MemoryLake` methods

All signatures are keyword-only beyond the first positional; adding a
new keyword argument with a default is additive. Removing a keyword or
changing a default-observable behaviour is breaking.

- `MemoryLake(database_url: str | KhoraConfig | None = None, *, engine: str = "vectorcypher", graph_url: str | None = None, embedding_model: str = "text-embedding-3-small", storage_config: StorageConfig | None = None, engine_kwargs: dict[str, Any] | None = None, run_migrations: bool = False)`
- `async connect() -> None`
- `async disconnect() -> None`
- `async __aenter__() / __aexit__(...)` — context-manager form
- `async create_namespace(*, config_overrides: dict[str, Any] | None = None) -> MemoryNamespace`
- `async get_namespace(namespace_id: UUID) -> MemoryNamespace | None`
- `async get_namespace_by_stable_id(namespace_id: str | UUID) -> MemoryNamespace | None`
- `async remember(content: str, *, namespace: str | UUID, title: str = "", source: str = "", metadata: dict[str, Any] | None = None, skill_name: str = "general_entities", entity_types: list[str], relationship_types: list[str], expertise: ExpertiseConfig | None = None, extraction_config_hash: str | None = None, chunk_strategy: ChunkStrategy | None = None, external_id: str | None = None) -> RememberResult`
- `async remember_batch(documents: list[dict[str, Any]], *, namespace: str | UUID, skill_name: str = "general_entities", max_concurrent: int = 10, deduplicate: bool = True, infer_relationships: bool = True, on_progress: Callable[[int, int], None] | None = None, entity_types: list[str], relationship_types: list[str], expertise: ExpertiseConfig | None = None, extraction_config_hash: str | None = None, chunk_strategy: ChunkStrategy | None = None, extraction_batch_size: int | None = None, extraction_max_tokens: int | None = None) -> BatchResult`
- `async recall(query: str, *, namespace: str | UUID, limit: int = 10, mode: SearchMode = SearchMode.HYBRID, min_similarity: float = 0.0, agentic: bool = False, raw: bool = False, include_sources: bool = False, start_time: datetime | None = None, end_time: datetime | None = None) -> RecallResult`
- `async forget(document_id: UUID, *, namespace: str | UUID) -> bool`
- `async stats(*, namespace: str | UUID) -> Stats`
- `storage` property — returns the underlying `StorageCoordinator`. Stable in that it exists; the `StorageCoordinator` surface is **not** part of this ADR.

There is no `MemoryLake.from_config(...)` classmethod today. The
constructor already accepts a `KhoraConfig` as its first positional
argument, which covers the same ergonomic; adding a classmethod alias
is a candidate for a future additive change but is out of scope here.

### Result dataclasses

All four are `@dataclass(slots=True, frozen=True)`. Fields listed here
are stable; additional fields may be appended in minor releases with
defaults. Renaming or removing a field is a breaking change.

- `RememberResult(document_id: UUID, namespace_id: UUID, chunks_created: int, entities_extracted: int, relationships_created: int, metadata: dict[str, Any] = {}, llm_usage: list[LLMUsage] = [])`
- `BatchResult(total: int, processed: int, skipped: int, failed: int, chunks: int, entities: int, relationships: int, metadata: dict[str, Any] = {}, llm_usage: list[LLMUsage] = [])`
- `RecallResult(query: str, namespace_id: UUID, chunks: list[tuple[Chunk, float]], entities: list[tuple[Entity, float]], context_text: str, metadata: dict[str, Any] = {}, relationships: list[tuple[Relationship, float]] = [], llm_usage: list[LLMUsage] = [])`
- `Stats(documents: int, chunks: int, entities: int, relationships: int, last_activity_at: datetime | None = None)`

`LLMUsage` fields (`operation`, `model`, `prompt_tokens`,
`completion_tokens`, `total_tokens`, `latency_ms`, `batch_size`) are
pinned by DYT-645; that contract remains the authoritative one for
Poros/Peras cost tracking.

### `khora.config.schema`

- `KhoraConfig` — `BaseSettings` subclass. Loads env vars with the
  `KHORA_` prefix automatically (instantiating `KhoraConfig()` already
  reads the environment, so there is no separate `from_env` method).
- `KhoraConfig.from_yaml(path: str | Path) -> KhoraConfig` — classmethod.
- Nested sections also importable from this module: `LLMSettings`,
  `PipelineSettings`, `QuerySettings`, `StorageSettings`,
  `TenancySettings`.

Adding a new field or a new nested settings section is additive.
Renaming or removing a field is breaking.

### `khora.config.llm`

- `LiteLLMConfig` — pydantic `BaseModel` with the fields `model`,
  `api_key_env`, `temperature`, `max_tokens`, `timeout`, `max_retries`,
  `retry_wait`, `max_concurrent_llm_calls`, `model_list`,
  `router_settings`, `embedding_model`, `embedding_api_key_env`,
  `embedding_dimension`, `embed_concurrency` (plus the rest of the
  sidecar fields defined in `llm.py`).
- `configure_litellm(config: LiteLLMConfig | None = None) -> None`
- `acompletion(...)` / `aembedding(...)` — async helpers used by
  `genesis` when it wants to reuse khora's configured LiteLLM router.
- `create_litellm_router(config: LiteLLMConfig) -> Any`

### `khora.core.models`

- `MemoryNamespace` — the dataclass returned from
  `MemoryLake.create_namespace()` and consumed by `genesis` TUI.
  Public fields: `id`, `namespace_id`, `tenancy_mode`, `version`,
  `is_active`, `config_overrides`, `sync_checkpoints`, `metadata`,
  `created_at`, `updated_at`.
- `TenancyMode` enum, values `SHARED`, `ISOLATED`.
- `DocumentSource` (also re-exported from `khora`).
- `EventType` (also re-exported from `khora`).

`Document`, `DocumentMetadata`, `Chunk`, `ChunkMetadata`, `Entity`,
`Episode`, `Relationship`, and `MemoryEvent` are exported from this
module but are **not** part of this ADR — their shapes are dictated by
storage schemas and may evolve. Consumers that persist these objects
should copy them into their own types.

### `khora.chat`

- `ChatResponse` — dataclass with `content: str`,
  `conversation_id: UUID`, `message_id: UUID`,
  `sources: list[dict] = []`, `metadata: dict[str, Any] = {}`. Consumed
  by `genesis` TUI (`from khora.chat import ChatResponse`).

`ChatEngine`, `ChatMessage`, `ConversationHistory`, `HistoryManager`,
`PersonaConfig`, `load_persona_config`, and `PromptGenerator` are
exported from the module and used by `genesis`, but the chat surface
is still evolving. We commit to keeping `ChatResponse` stable (since
genesis pattern-matches on its fields) and to announcing any breaking
change to the rest of the chat API in the CHANGELOG under
`### Deprecated` one minor release before removal.

### Support modules

- `khora.extraction.binary_readers.extract_if_needed(path: Path) -> Path | None` —
  used by `khora extract` CLI and the forthcoming `khora-cli` to run
  PDF/Excel/Word/Parquet readers before text ingestion.
- `khora.logging_config.setup_logging(level: str = "INFO", json_logs: bool = False, log_file: Path | None = None) -> None` —
  library consumers should call this (or configure loguru with
  `enqueue=True` themselves; see CLAUDE.md § Logging).

## Change Policy

- **Additive** (patch or minor): new optional dataclass fields with
  defaults, new keyword-only parameters with defaults, new methods,
  new nested settings sections, new re-exports.
- **Breaking** (major bump + coordinated release): renaming or removing
  any symbol listed above, changing a field type, tightening a
  parameter's type, changing a default in a way that alters observable
  behaviour, changing method semantics.
- Deprecation must be announced in `CHANGELOG.md` under
  `### Deprecated` in a minor release **before** the symbol is removed
  in the next major release. The deprecation entry must name the
  replacement symbol.
- `__all__` in `src/khora/__init__.py` is the machine-readable source
  of truth for the top-level surface. For submodules, the `__all__`
  list in that submodule is authoritative.

## Relationship to ADR-022

ADR-024 is **additive** to ADR-022. The seven extraction-skills
dataclasses enumerated in ADR-022 (`ExpertiseConfig`,
`EntityTypeConfig`, `RelationshipTypeConfig`, `ConfidenceConfig`,
`ExpansionConfig`, `CorrelationRule`, `InferenceRule`) remain governed
by ADR-022 and are the `expertise` argument to `MemoryLake.remember()`
and `MemoryLake.remember_batch()`. ADR-024 codifies the rest of the
surface the same consumers already touch.

## Compatibility with consumers

- `genesis` — imports `MemoryLake`, `KhoraConfig`, `MemoryNamespace`,
  `LiteLLMConfig`, `configure_litellm`, `ChatResponse` (and other
  `khora.chat` names). All are covered above. No migration needed.
- `khora-benchmarks` — imports `MemoryLake`, `KhoraConfig`,
  `SearchMode` plus internal modules. The public imports are covered;
  benchmarks' use of `khora.engines.vectorcypher` internals remains
  explicitly private — benchmarks pin a khora version to manage
  churn.
- `khora-explorer` — imports from `khora.extraction.skills.base`
  (ADR-022) and `khora.config.schema`. Both covered.
- `khora-cli` (new, DYT-2655) — will import `MemoryLake`,
  `KhoraConfig`, `SearchMode`, `LLMUsage`, `RememberResult`,
  `RecallResult`, `BatchResult`, `extract_if_needed`, `setup_logging`.
  All covered; no API additions required.
- `Poros` / `Peras` — consume `LLMUsage` only (DYT-645). Already
  covered; unchanged by this ADR.

## Consequences

**Positive**

- Four downstream packages can pin a khora minor version and trust
  that patch releases will not break their imports.
- Contributors editing `khora/__init__.py`, `memory_lake.py`, or
  `config/schema.py` have a single document to consult before
  renaming.
- `khora-cli` can proceed without litigating "is this public?" for
  each symbol it imports.

**Negative**

- Renaming any of the listed symbols now requires a coordinated
  release. In practice khora does not ship majors often, so this cost
  falls on a small number of PRs per year.
- `RecallResult.metadata` is a free-form dict in the contract; engines
  may add keys that downstream consumers come to rely on. We note
  this risk here but do not enumerate metadata keys — engines remain
  free to evolve their own metadata shape.

**Neutral**

- This ADR documents existing behaviour. No migration is required.
  `__all__` in `src/khora/__init__.py` already lists the top-level
  surface; no re-export is being added or removed by this ADR.

## Related

- ADR-022 — Extraction skills public API (the `expertise` argument).
- DYT-645 — `LLMUsage` contract with Poros/Peras.
- DYT-2655 — parent initiative introducing `khora-cli`.
- `CLAUDE.md` § Downstream — summarizes the contract for agents.
