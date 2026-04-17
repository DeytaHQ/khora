# Downstream Consumers

Khora is a library. Its stable public API is consumed by several first-party packages. This page is the quick map for anyone asking "where did `khora extract` go?" or "how do I plug khora into my service?"

## Sibling packages

### khora-cli

[github.com/DeytaHQ/khora-cli](https://github.com/DeytaHQ/khora-cli) — extract and search CLI.

```bash
uv pip install khora-cli
uv run khora-cli extract report.pdf
uv run khora-cli search "Who worked on the API design?" -n <namespace-id>
```

Prior to khora v0.8.0, these commands shipped as `uv run khora extract` / `uv run khora search`. The package was split out so that the library has no CLI dependencies (no `click`, no `rich`, no PDF/Excel readers by default). If you need the binary readers without the CLI, install `pip install khora[binary-readers]` and import `from khora.extraction.binary_readers import extract_if_needed`.

khora-cli imports a narrow slice of khora's public API:

```python
from khora import (
    MemoryLake, KhoraConfig, SearchMode, LLMUsage,
    RememberResult, RecallResult, BatchResult,
)
from khora.extraction.binary_readers import extract_if_needed
from khora.logging_config import setup_logging
```

### khora-explorer

[github.com/DeytaHQ/khora-explorer](https://github.com/DeytaHQ/khora-explorer) — ontology construction, validation, and preview.

```bash
uv pip install khora-explorer
uv run khora-explorer construct --source <path-or-glob>
uv run khora-explorer validate my_ontology.yaml
uv run khora-explorer preview my_ontology.yaml
```

Prior to v0.7.52 these lived under `uv run khora ontology …` and the underlying `khora.discovery` package. They were extracted because ontology construction needs a heavier dependency set (PDF/HTML scraping, Firecrawl fallback, multi-provider LLM routing) than a memory-lake library should carry.

khora-explorer imports:

```python
from khora.extraction.skills.base import (
    ExpertiseConfig, EntityTypeConfig, RelationshipTypeConfig,
    ConfidenceConfig, ExpansionConfig, CorrelationRule, InferenceRule,
)
from khora.config.schema import KhoraConfig
```

This surface is codified by [ADR-022](adrs/adr-022-extraction-skills-public-api.md).

## Internal consumers

### genesis

Uses `MemoryLake` through the stable top-level surface (ADR-024) plus `lake.storage` for direct backend access. Also imports `LLMUsage` for cost tracking (DYT-645 contract shared with Poros/Peras). Pins a specific khora version per deploy; follows khora's major releases.

### khora-benchmarks

Benchmarks khora's retrieval engines. Imports include private modules (`khora.engines.vectorcypher`, `khora.extraction.chunkers`) documented as **unstable** in ADR-024 — benchmarks pin an exact khora version on purpose.

### anima, ttoj

Thin consumers of the top-level `MemoryLake` surface. No private imports.

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

The `khora` top-level imports (`MemoryLake`, `KhoraConfig`, `SearchMode`, `ExpertiseConfig`, etc.) are unchanged and continue to work.

## Stability contract

Two ADRs formalise what you can depend on:

- [ADR-022](adrs/adr-022-extraction-skills-public-api.md) — `ExpertiseConfig` and friends from `khora.extraction.skills.base`.
- [ADR-024](adrs/adr-024-memory-lake-public-api.md) — top-level `khora` re-exports (`MemoryLake`, results, `SearchMode`, etc.).

Both are append-only in minor releases. Breaking changes require a major bump **and** coordinated releases across genesis, khora-benchmarks, khora-explorer, and khora-cli.

## Integration checklist

For a new downstream consumer:

1. `pip install khora[<backend-of-choice>]` — pick the backend matrix that matches your infra (`postgres`-default for PG, `surrealdb` for zero-infra, `all-backends` if unsure).
2. Either call `khora.logging_config.setup_logging()` once per process, or configure your own loguru sinks with `enqueue=True`. The default loguru sink blocks the event loop in async code — see [configuration.md](configuration.md#logging).
3. Run migrations (for PostgreSQL) — pass `run_migrations=True` to `MemoryLake` or invoke `alembic upgrade head` out-of-band. See [migrations.md](migrations.md).
4. Import only from the symbols listed in [api-reference.md](api-reference.md) unless you are willing to follow khora's internal churn.
5. Pin Khora by major version. Minor-version upgrades are safe; majors may require coordinated releases.
