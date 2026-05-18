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

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Hard floor on fact-compaction retention. Operators cannot configure
# the dream phase to hard-delete tombstoned facts younger than this many
# days — anything tighter is treated as a misconfiguration. The floor
# protects against accidental data loss during rollout (#667).
_FACT_COMPACTION_RETENTION_FLOOR_DAYS = 7


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

    # Phase 1.1 — chronicle abstention drift report (#652).
    abstention_drift_min_samples: int = Field(
        default=1000,
        ge=0,
        description=(
            "Minimum sample count before the chronicle abstention-drift "
            "report emits a recommendation. Below this floor the op "
            "returns decision='insufficient_data'."
        ),
    )
    abstention_drift_sample_cap: int = Field(
        default=1024,
        ge=1,
        description=(
            "Per-namespace cap on the in-process ring buffer of recall "
            "samples used by the chronicle abstention-drift report."
        ),
    )

    # Phase 2.4 — chronicle memory_facts compaction (#664).
    fact_compaction_retention_days: int = Field(
        default=365,
        ge=0,
        description=(
            "Age threshold for the chronicle fact-compaction op: facts "
            "tombstoned (legacy ``is_active=False`` OR bi-temporal "
            "``invalidated_at``) more than this many days ago are planned "
            "for hard-delete. Apply mode lands in v0.15 (#669)."
        ),
    )

    # Vectorcypher orphan-report knobs (#657).
    cooccurrence_edge_weight: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description=(
            "Weight applied to ASSOCIATED_WITH co-occurrence edges during "
            "the vectorcypher PageRank orphan report. Selective extraction "
            "emits these by default for non-LLM chunks; left at 1.0 they "
            "would dominate PageRank and mask real orphans."
        ),
    )
    orphan_pr_percentile_threshold: float = Field(
        default=5.0,
        ge=0.0,
        le=100.0,
        description=(
            "Bottom-percentile cut-off for orphan-candidate selection. "
            "Entities with a PageRank score at or below this percentile "
            "AND mention_count <= 1 are flagged as archive candidates."
        ),
    )

    # Vectorcypher source_chunk_ids GC knob (#662).
    source_chunk_ids_gc_min_dead: int = Field(
        default=1,
        ge=1,
        description=(
            "Minimum dead-UUID count below which an entity is not emitted "
            "as a GC candidate by the source_chunk_ids GC op. Default 1 "
            "(every entity with at least one dead reference is planned)."
        ),
    )

    # Vectorcypher centroid-recompute knobs (#660).
    centroid_lev_threshold: int = Field(
        default=2,
        ge=0,
        description=(
            "Maximum Levenshtein distance between any two names in a "
            "merge cluster for the centroid path. Above this, the names "
            "are considered lexically distant and the op plans a "
            "re-embed of the canonical name instead."
        ),
    )
    centroid_min_intra_cluster_cosine: float = Field(
        default=0.88,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum pairwise cosine within a merge cluster. Below this "
            "floor the cluster is judged multimodal — the merge itself "
            "is suspect — and the op emits decision='skip_multimodal' "
            "without planning an embedding."
        ),
    )

    # Phase 2.1 — vectorcypher cross-batch entity-resolution dedupe (#658).
    dedupe_entities_default_threshold: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description=(
            "Default cosine-similarity merge threshold used by the "
            "vectorcypher cross-batch dedupe op for any entity_type "
            "not overridden in dedupe_entities_per_type_thresholds. "
            "Deliberately tighter than the online resolver's "
            "DEFAULT_THRESHOLD (0.85)."
        ),
    )
    dedupe_entities_per_type_thresholds: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-entity-type cosine-similarity merge threshold overrides "
            "(e.g. {'PERSON': 0.95}). Missing types fall back to "
            "dedupe_entities_default_threshold."
        ),
    )

    # ------------------------------------------------------------------
    # Apply-mode guardrails (#667)
    # ------------------------------------------------------------------

    @field_validator("fact_compaction_retention_days")
    @classmethod
    def _enforce_fact_compaction_retention_floor(cls, value: int) -> int:
        """Reject retention windows tighter than the hard floor.

        The dream-phase fact compactor hard-deletes tombstoned rows older
        than ``fact_compaction_retention_days``. Operators cannot drop
        the value below the floor — a value smaller than the floor is
        always a misconfiguration (a Phase 4 rollout disabling the floor
        would land as a separate field).
        """
        if value < _FACT_COMPACTION_RETENTION_FLOOR_DAYS:
            raise ValueError(
                f"fact_compaction_retention_days must be >= "
                f"{_FACT_COMPACTION_RETENTION_FLOOR_DAYS} (got {value}). "
                "The hard floor exists to prevent accidental data loss "
                "during dream-phase apply rollout."
            )
        return value

    @model_validator(mode="after")
    def _enforce_apply_mode_retention_floor(self) -> DreamConfig:
        """Re-assert the retention floor when apply mode is the default.

        Even though :meth:`_enforce_fact_compaction_retention_floor`
        always runs, this model-level pass makes the constraint explicit
        for the (enabled, default_mode='apply') configuration — if the
        Pydantic field-validator ever moved or is bypassed via a future
        configurable-floor flag, this last gate still catches an
        apply-mode dream run pointed at a too-tight retention window.
        """
        if (
            self.enabled
            and self.default_mode == "apply"
            and self.fact_compaction_retention_days < _FACT_COMPACTION_RETENTION_FLOOR_DAYS
        ):
            raise ValueError(
                "DreamConfig(enabled=True, default_mode='apply') "
                f"requires fact_compaction_retention_days >= "
                f"{_FACT_COMPACTION_RETENTION_FLOOR_DAYS} "
                f"(got {self.fact_compaction_retention_days})."
            )
        return self

    # Phase 5.2 — vectorcypher edge pruning by weight x recency (#671).
    prune_edges_enabled: bool = Field(
        default=False,
        description=(
            "Master switch for the vectorcypher edge-pruning op "
            "(OpKind.VECTORCYPHER_PRUNE_EDGES). When False (default), the "
            "op is skipped even if requested by scope.op_kinds."
        ),
    )
    prune_edges_target_predicates: list[str] = Field(
        default_factory=lambda: ["ASSOCIATED_WITH"],
        description=(
            "Whitelist of relationship_type values eligible for edge "
            "pruning. Default targets only ASSOCIATED_WITH co-occurrence "
            "edges (the soup that dominates the edge set after months of "
            "ingest). Operators must opt in to broader pruning by adding "
            "more types — narrower than ASSOCIATED_WITH is fine; wider "
            "is a deliberate decision."
        ),
    )
    prune_edges_confidence_threshold: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description=(
            "Edges with confidence < threshold satisfy the first conjunct "
            "of the prune predicate. The other two conjuncts are "
            "valid_to IS NULL and 'all source chunks deleted'. Default "
            "0.4 matches the issue spec."
        ),
    )

    # Phase 5.3 — vectorcypher contradiction detection (log only, #672).
    contradiction_detect_enabled: bool = Field(
        default=False,
        description=(
            "Master switch for the vectorcypher contradiction-detection "
            "op (OpKind.VECTORCYPHER_CONTRADICTION_DETECT, #672). Report "
            "only — never mutates ``relationships``; findings feed a "
            "human triage queue and become the natural source of mapping "
            "recommendations for Phase 5.4 (#673)."
        ),
    )
    contradiction_detect_similarity_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Textual-similarity threshold for the contradiction detector. "
            "Pairs of live relationships in the same (source, target, type) "
            "bucket scoring below this value are flagged as potential "
            "contradictions. Property contradictions are flagged "
            "independently of this threshold."
        ),
    )

    # Phase 2.5 — chronicle event near-duplicate clustering (#665).
    event_clustering_cosine_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description=(
            "SVO-summary cosine threshold above which two chronicle_events "
            "are considered near-duplicates and clustered. Applied within "
            "a (namespace_id, subject) bucket and a sliding referenced_date "
            "window. Pairwise dot product on pre-normalized embeddings."
        ),
    )
    event_clustering_window_days: int = Field(
        default=7,
        ge=0,
        description=(
            "Half-width of the sliding referenced_date window (in days) "
            "inside which chronicle_events with cosine >= the threshold "
            "are clustered. Events outside the window are never merged "
            "even when their SVO summaries are identical."
        ),
    )

    # Phase 5.1 — vectorcypher community detection + per-community LLM
    # summary (#670). OFF by default — first LLM-using dream op.
    community_summary_enabled: bool = Field(
        default=False,
        description=(
            "Master toggle for the community-summary op. Default OFF — "
            "this is the first LLM-using dream op and operators must "
            "opt-in to the cost surface (~$15-25 per dream cycle on a "
            "100k-entity namespace at gpt-4o-mini rates)."
        ),
    )
    community_summary_min_size: int = Field(
        default=5,
        ge=2,
        description=(
            "Minimum community size to emit a summary op for. "
            "Communities smaller than this are skipped — the LLM cost is "
            "not justified."
        ),
    )
    community_summary_model: str = Field(
        default="gpt-4o-mini",
        description=(
            "LLM model used by the community-summary apply handler. "
            "Configurable so operators can swap to a higher-quality "
            "model (gpt-4o) at higher cost, or to a self-hosted model "
            "for air-gapped deployments."
        ),
    )
    community_summary_max_members_per_prompt: int = Field(
        default=20,
        ge=1,
        description=(
            "Per-community cap on member ids carried into the LLM "
            "prompt. Bounds the prompt size so a single community of "
            "1k entities cannot run an unbounded context."
        ),
    )
