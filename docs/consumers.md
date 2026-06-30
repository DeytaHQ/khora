# Public API

Khora is a library. Its stable public API is consumed by services and notebooks. This page documents the public surface that downstream code may rely on.

## No in-library CLI

The CLI commands (`khora extract`, `khora search`, `khora ontology …`) were removed from the `khora` package so the library has no CLI dependencies. The `khora` top-level imports (`Khora`, `KhoraConfig`, `SearchMode`, `ExpertiseConfig`, etc.) are unchanged - call them directly from your service or notebook. See [migrations.md](migrations.md#v080---cli-extraction) for the removal notice.

The one piece of CLI-flavoured functionality available inside the library is binary-document text extraction. Install with `pip install khora[binary-readers]` and import directly:

```python
from khora.extraction.binary_readers import extract_if_needed
```

**Failure contract:** `extract_if_needed` raises `ExtractionError` on genuine parse/open failures (xlsx, docx, parquet). Passing a `.pdf` path raises `NotImplementedError` - preprocess PDFs upstream or use `khora-cli`'s PDF preprocessing.

## Stability contract

Two public API surfaces are pinned as stable.

### Memory-lake surface - `from khora import …`

| Category | Symbols |
| --- | --- |
| Entry point | `Khora`, `KhoraConfig` |
| Operation results | `RememberResult`, `RecallResult`, `BatchResult`, `BatchHandle`, `DocumentResult`, `Stats`, `LLMUsage` |
| Query types | `SearchMode`, `SemanticFilter` |
| Helpers | `context_text` (render a `RecallResult` as an LLM context string) |
| Errors | `KhoraError`, `EngineCapabilityError` |
| Domain enums at the boundary | `DocumentSource`, `EventType` |
| Engine registry | `create_engine`, `list_engines`, `register_engine` |
| Re-exported from the extraction-skill surface | `ExpertiseConfig`, `EntityTypeConfig`, `RelationshipTypeConfig` |

Canonical machine-readable contract: `__all__` in `src/khora/__init__.py`.

### Extraction-skill surface - `from khora.extraction.skills import …`

| Category | Symbols |
| --- | --- |
| Domain definition | `ExpertiseConfig`, `EntityTypeConfig`, `RelationshipTypeConfig` |
| Tuning | `ConfidenceConfig`, `ExpansionConfig` |
| Cross-tool reconciliation | `CorrelationRule`, `InferenceRule`, `InferenceCondition` (supporting type used as `InferenceRule.when`) |
| Chronicle extraction toggles | `EventExtractionConfig`, `FactExtractionConfig` |
| Legacy back-compat exports | `ConfidenceLevel` (enum), `ExtractionSkill` (legacy class) - listed in `__all__` and importable for existing consumers, but not extended; new code should not rely on them |

Canonical machine-readable contract: `__all__` in `src/khora/extraction/skills/base.py`.

### Versioning policy

- **Additive changes** are permitted in minor and patch releases: new optional dataclass fields with defaults, new optional keyword arguments to existing methods, new helper modules. Adding a field must preserve existing `from_dict` / `to_dict` round-trips for older payloads.
- **Breaking changes** require a major version bump: renaming or removing a field/method, changing a type, changing a default in a way that alters observable behaviour, removing a class. Breaking changes are recorded in CHANGELOG.md.
- **Security exception.** Breaking changes that close a confidentiality or integrity vulnerability may land in a patch release without a major bump. The patch CHANGELOG entry calls out the affected signatures under `### Changed (breaking)` and the corresponding security finding under `### Security`. The cross-namespace IDOR close-out (`namespace_id` required on every storage read/write) was landed under this exception - see [migrations.md](migrations.md) and [CHANGELOG.md](../CHANGELOG.md).
- `from_dict` for extraction-skill dataclasses preserves backward compatibility with historical YAML/JSON payloads for at least one major version after schema evolution.

### What's NOT pinned

- Anything not listed in either `__all__`.
- Internal models exported from `khora.core.models` - `Document`, `Chunk`, `Entity`, `Episode`, `Relationship`, `MemoryEvent`. Their shapes are dictated by storage schemas and may evolve; consumers that persist these objects should copy them into their own types.
- The `khora.chat` module aside from `ChatResponse`. The rest (`ChatEngine`, `ChatMessage`, `ConversationHistory`, `HistoryManager`, `PersonaConfig`, `load_persona_config`, `PromptGenerator`) is evolving; breaking changes are announced in CHANGELOG `### Deprecated` one minor release before removal.
- The `khora.storage.StorageCoordinator` surface exposed via the `Khora.storage` property. The property exists; its API is not pinned. `StorageCoordinator.{relational,vector,graph,event_store}` are `NamespaceRequiredProxy` wrappers - they emit a `DeprecationWarning` on first access per role per process and refuse dispatch on read methods missing `namespace_id=`. Downstream code that needs the unwrapped backends should call coordinator-level methods (all of which take `namespace_id=` as a kwarg-only argument) instead.
- Anything whose name starts with an underscore.

For full method signatures and dataclass field lists, read the symbols directly (`help(Khora)`, the source on GitHub, or your IDE's type-stub navigation).

## Integration checklist

For a new downstream consumer:

1. `pip install khora[<backend-of-choice>]` - pick the backend matrix that matches your infra (`postgres`-default for PG, `surrealdb` for zero-infra, `all-backends` if unsure). For binary-document ingestion, add `khora[binary-readers]`.
2. Either call `khora.logging_config.setup_logging()` once per process, or configure your own loguru sinks with `enqueue=True`. The default loguru sink blocks the event loop in async code - see [configuration.md](configuration.md#logging).
3. Run migrations (for PostgreSQL) - pass `run_migrations=True` to `Khora` or invoke `alembic upgrade head` out-of-band. See [migrations.md](migrations.md).
4. Import only from the symbols listed in [api-reference.md](api-reference.md) unless you are willing to follow khora's internal churn.
5. Pin Khora by major version. Minor-version upgrades are safe; majors may include breaking API changes documented in CHANGELOG.md.
6. **Reading credentials back out of `KhoraConfig`?** Credential fields are `pydantic.SecretStr` - call `.get_secret_value()` for the cleartext. `str(cfg.storage.postgresql_url)` returns `'**********'` by design. See [configuration.md](configuration.md#secretstr-typed-credential-fields).
7. **Need spans/metrics?** Install `khora[otel]` and call `khora.telemetry.configure_telemetry()` at process startup, or rely on env-based auto-bootstrap. See [observability.md](observability.md) for the precedence rules and vendor recipes.
