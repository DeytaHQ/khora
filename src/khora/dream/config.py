"""Dream-phase configuration.

``DreamConfig`` nests under :class:`khora.config.KhoraConfig` and is
configurable via the ``KHORA_DREAM_*`` env-var prefix (single-underscore
flat form, plus ``__`` for nested op-level toggles inside
``DreamOpsConfig``).

Stability: ``DreamConfig`` itself is part of the public API. Individual
field names are **internal** during Phase 0 — they may evolve without a
major-version bump until the dream orchestrator stabilizes (#649).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DreamOpsConfig(BaseModel):
    """Op-level toggles for individual dream operations.

    Every destructive op defaults to ``False`` — a fresh ``DreamConfig()``
    cannot delete anything until the operator opts in per-op. Each toggle
    is independently overridable via env var, e.g.
    ``KHORA_DREAM_OPS__DEDUPE_ENTITIES=true``.
    """

    dedupe_entities: bool = Field(
        default=False,
        description="Merge entities the resolver judges duplicates.",
    )
    prune_edges: bool = Field(
        default=False,
        description="Remove low-confidence / orphaned relationship edges.",
    )
    compact_facts: bool = Field(
        default=False,
        description="Collapse superseded fact records.",
    )
    cluster_events: bool = Field(
        default=False,
        description="Group related events into episode summaries.",
    )
    recompute_centroids: bool = Field(
        default=False,
        description="Recompute entity / cluster centroid embeddings.",
    )


class DreamConfig(BaseSettings):
    """Dream-phase configuration.

    Lives at ``KhoraConfig.dream``. Env-var precedence:

    - ``KHORA_DREAM_ENABLED=true``
    - ``KHORA_DREAM_DEFAULT_MODE=apply``
    - ``KHORA_DREAM_LLM_MAX_TOKENS_PER_RUN=200000``
    - ``KHORA_DREAM_OPS__DEDUPE_ENTITIES=true``

    Modelled as :class:`pydantic_settings.BaseSettings` (matching
    :class:`khora.config.PipelineSettings` /
    :class:`khora.hooks.SemanticHooksConfig`) so flat
    ``KHORA_DREAM_*`` env vars populate top-level fields and
    ``KHORA_DREAM_OPS__*`` populates the nested
    :class:`DreamOpsConfig`.
    """

    model_config = SettingsConfigDict(
        env_prefix="KHORA_DREAM_",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    enabled: bool = Field(
        default=False,
        description="Master switch — when False, Khora.dream() is a no-op.",
    )
    default_mode: Literal["dry-run", "apply"] = Field(
        default="dry-run",
        description="Default mode when caller omits the mode= kwarg.",
    )

    # Op-level toggles (all destructive ops default off).
    ops: DreamOpsConfig = Field(
        default_factory=DreamOpsConfig,
        description="Per-op enable flags.",
    )

    # LLM token budgets.
    llm_max_tokens_per_run: int = Field(
        default=200_000,
        ge=0,
        description="Hard cap on LLM tokens spent in a single dream run.",
    )
    llm_max_tokens_per_namespace_per_day: int = Field(
        default=1_000_000,
        ge=0,
        description="Rolling-day token budget per namespace across all runs.",
    )

    # Retention knobs.
    retention_days: int = Field(
        default=30,
        ge=0,
        description="How long to keep dream run records and reports.",
    )
    retention_runs_per_namespace: int = Field(
        default=50,
        ge=0,
        description="Max retained dream-run records per namespace.",
    )

    # Sink toggles.
    report_file_sink_enabled: bool = Field(
        default=False,
        description="Write dream reports to the file sink.",
    )
    report_event_sink_enabled: bool = Field(
        default=False,
        description="Emit dream reports as semantic-hook events.",
    )
    report_collector_sink_enabled: bool = Field(
        default=False,
        description="Forward dream reports to the telemetry collector.",
    )

    redact_text: Literal["none", "summary", "all"] = Field(
        default="summary",
        description=(
            "Free-text redaction policy for dream reports: 'none' keeps "
            "verbatim text, 'summary' keeps a short summary only, 'all' "
            "strips every textual field."
        ),
    )
