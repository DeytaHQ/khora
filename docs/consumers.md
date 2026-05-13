# Downstream Consumers

Khora is a library. Its stable public API is consumed by several first-party packages. This page is the quick map for anyone asking "where did `khora extract` go?" or "how do I plug khora into my service?"

## Sibling packages

### khora-cli

khora-cli (to be released soon) — extract and search CLI.

```bash
uv pip install khora-cli
uv run khora-cli extract report.pdf
uv run khora-cli search "Who worked on the API design?" -n <namespace-id>
```

Prior to khora v0.8.0, these commands shipped as `uv run khora extract` / `uv run khora search`. The package was split out so that the library has no CLI dependencies (no `click`, no `rich`, no PDF/Excel readers by default). If you need the binary readers without the CLI, install `pip install khora[binary-readers]` and import `from khora.extraction.binary_readers import extract_if_needed`.

khora-cli imports a narrow slice of khora's public API:

```python
from khora import (
    Khora, KhoraConfig, SearchMode, LLMUsage,
    RememberResult, RecallResult, BatchResult,
)
from khora.extraction.binary_readers import extract_if_needed
from khora.logging_config import setup_logging
```

### khora-explorer

khora-explorer (to be released soon) — ontology construction, validation, and preview.

```bash
uv pip install khora-explorer
uv run khora-explorer construct --source <path-or-glob>
uv run khora-explorer validate my_ontology.yaml
uv run khora-explorer preview my_ontology.yaml
```

Prior to v0.7.52 these lived under `uv run khora ontology …` and the underlying `khora.discovery` package. They were extracted because ontology construction needs a heavier dependency set (PDF/HTML scraping, Firecrawl fallback, multi-provider LLM routing) than this library should carry.

khora-explorer imports:

```python
from khora.extraction.skills.base import (
    ExpertiseConfig, EntityTypeConfig, RelationshipTypeConfig,
    ConfidenceConfig, ExpansionConfig, CorrelationRule, InferenceRule,
)
from khora.config.schema import KhoraConfig
```

## Migration from pre-v0.8 khora

If your project installed `khora` before v0.8.0 and used the CLI:

| Before (khora ≤ 0.7.51) | After (khora ≥ 0.8.0) |
|---|---|
| `uv run khora extract file.pdf` | `uv pip install khora-cli && uv run khora-cli extract file.pdf` |
| `uv run khora search "query" -n ns` | `uv run khora-cli search "query" -n ns` |
| `uv run khora ontology construct --source …` | `uv pip install khora-explorer && uv run khora-explorer construct --source …` |
| `uv run khora ontology validate file.yaml` | `uv run khora-explorer validate file.yaml` |
| `uv run khora ontology preview file.yaml` | `uv run khora-explorer preview file.yaml` |
| `from khora.discovery.extraction import extract_if_needed` | `from khora.extraction.binary_readers import extract_if_needed` |
| `from khora.discovery import …` | Install `khora-explorer`. Module has moved. |
| `from khora.cli import …` | Install `khora-cli`. Module has moved. |

The `khora` top-level imports (`Khora`, `KhoraConfig`, `SearchMode`, `ExpertiseConfig`, etc.) are unchanged and continue to work.

## Stability contract

Two public API surfaces are pinned as stable.

### Memory-lake surface — `from khora import …`

| Category | Symbols |
| --- | --- |
| Entry point | `Khora`, `KhoraConfig` |
| Operation results | `RememberResult`, `RecallResult`, `BatchResult`, `BatchHandle`, `DocumentResult`, `Stats`, `LLMUsage` |
| Query types | `SearchMode`, `SemanticFilter` |
| Errors | `KhoraError` |
| Domain enums at the boundary | `DocumentSource`, `EventType` |
| Engine registry | `create_engine`, `list_engines`, `register_engine` |
| Re-exported from the extraction-skill surface | `ExpertiseConfig`, `EntityTypeConfig`, `RelationshipTypeConfig` |

Canonical machine-readable contract: `__all__` in `src/khora/__init__.py`.

### Extraction-skill surface — `from khora.extraction.skills import …`

| Category | Symbols |
| --- | --- |
| Domain definition | `ExpertiseConfig`, `EntityTypeConfig`, `RelationshipTypeConfig` |
| Tuning | `ConfidenceConfig`, `ExpansionConfig` |
| Cross-tool reconciliation | `CorrelationRule`, `InferenceRule`, `InferenceCondition` (supporting type used as `InferenceRule.when`) |
| Chronicle extraction toggles | `EventExtractionConfig`, `FactExtractionConfig` |
| Legacy back-compat exports | `ConfidenceLevel` (enum), `ExtractionSkill` (legacy class) — listed in `__all__` and importable for existing consumers, but not extended; new code should not rely on them |

Canonical machine-readable contract: `__all__` in `src/khora/extraction/skills/base.py`.

### Versioning policy

- **Additive changes** are permitted in minor and patch releases: new optional dataclass fields with defaults, new optional keyword arguments to existing methods, new helper modules. Adding a field must preserve existing `from_dict` / `to_dict` round-trips for older payloads.
- **Breaking changes** require a major version bump: renaming or removing a field/method, changing a type, changing a default in a way that alters observable behaviour, removing a class. Breaking changes coordinate with the published consumer packages (khora-cli, khora-explorer).
- `from_dict` for extraction-skill dataclasses preserves backward compatibility with historical YAML/JSON payloads for at least one major version after schema evolution.

### What's NOT pinned

- Anything not listed in either `__all__`.
- Internal models exported from `khora.core.models` — `Document`, `DocumentMetadata`, `Chunk`, `ChunkMetadata`, `Entity`, `Episode`, `Relationship`, `MemoryEvent`. Their shapes are dictated by storage schemas and may evolve; consumers that persist these objects should copy them into their own types.
- The `khora.chat` module aside from `ChatResponse`. The rest (`ChatEngine`, `ChatMessage`, `ConversationHistory`, `HistoryManager`, `PersonaConfig`, `load_persona_config`, `PromptGenerator`) is evolving; breaking changes are announced in CHANGELOG `### Deprecated` one minor release before removal.
- The `khora.storage.StorageCoordinator` surface exposed via the `Khora.storage` property. The property exists; its API is not pinned.
- Anything whose name starts with an underscore.

For full method signatures and dataclass field lists, read the symbols directly (`help(Khora)`, the source on GitHub, or your IDE's type-stub navigation).

## Integration checklist

For a new downstream consumer:

1. `pip install khora[<backend-of-choice>]` — pick the backend matrix that matches your infra (`postgres`-default for PG, `surrealdb` for zero-infra, `all-backends` if unsure).
2. Either call `khora.logging_config.setup_logging()` once per process, or configure your own loguru sinks with `enqueue=True`. The default loguru sink blocks the event loop in async code — see [configuration.md](configuration.md#logging).
3. Run migrations (for PostgreSQL) — pass `run_migrations=True` to `Khora` or invoke `alembic upgrade head` out-of-band. See [migrations.md](migrations.md).
4. Import only from the symbols listed in [api-reference.md](api-reference.md) unless you are willing to follow khora's internal churn.
5. Pin Khora by major version. Minor-version upgrades are safe; majors may require coordinated releases.
