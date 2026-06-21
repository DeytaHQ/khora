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

import os
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Canonical / legacy env-var pairs for dream nested-config fields (#789).
# Mirrors ``khora.config.schema._DREAM_ENV_ALIAS_PAIRS`` — duplicated here
# to avoid an import cycle (config.schema imports DreamConfig).
_DREAM_ENV_ALIAS_PAIRS: tuple[tuple[str, str], ...] = (
    ("KHORA_DREAM_OPS_DEDUPE_ENTITIES", "KHORA_DREAM_OPS__DEDUPE_ENTITIES"),
    ("KHORA_DREAM_OPS_PRUNE_EDGES", "KHORA_DREAM_OPS__PRUNE_EDGES"),
    ("KHORA_DREAM_OPS_COMPACT_FACTS", "KHORA_DREAM_OPS__COMPACT_FACTS"),
    ("KHORA_DREAM_OPS_CLUSTER_EVENTS", "KHORA_DREAM_OPS__CLUSTER_EVENTS"),
    ("KHORA_DREAM_OPS_RECOMPUTE_CENTROIDS", "KHORA_DREAM_OPS__RECOMPUTE_CENTROIDS"),
)


def _inject_dream_canonical_env(data: dict[str, Any], legacy_env_name: str, value: str) -> None:
    """Inject a single-underscore dream env var into the nested-dict slot.

    DreamConfig has ``env_prefix="KHORA_DREAM_"`` and
    ``env_nested_delimiter="__"`` — so legacy ``KHORA_DREAM_OPS__DEDUPE_ENTITIES``
    is split into ``data["ops"]["dedupe_entities"]`` by pydantic-settings.
    Canonical ``KHORA_DREAM_OPS_DEDUPE_ENTITIES`` is ignored unless we
    inject it here.
    """
    prefix = "KHORA_DREAM_"
    if not legacy_env_name.startswith(prefix):
        return
    tail = legacy_env_name[len(prefix) :]
    parts = [p.lower() for p in tail.split("__")]
    if not parts:
        return
    node: Any = data
    for key in parts[:-1]:
        existing = node.get(key)
        if existing is None:
            existing = {}
            node[key] = existing
        elif not isinstance(existing, dict):
            return
        node = existing
    leaf_key = parts[-1]
    if node.get(leaf_key) is None:
        node[leaf_key] = value


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
    ``KHORA_DREAM_OPS_DEDUPE_ENTITIES=true`` (single-underscore canonical
    form; the legacy double-underscore form ``KHORA_DREAM_OPS__DEDUPE_ENTITIES``
    is also accepted, see #789).
    """

    model_config = {"populate_by_name": True}

    dedupe_entities: bool = Field(
        default=False,
        description="Merge entities the resolver judges duplicates.",
        validation_alias=AliasChoices(
            "KHORA_DREAM_OPS_DEDUPE_ENTITIES",
            "KHORA_DREAM_OPS__DEDUPE_ENTITIES",
        ),
    )
    prune_edges: bool = Field(
        default=False,
        description="Remove low-confidence / orphaned relationship edges.",
        validation_alias=AliasChoices(
            "KHORA_DREAM_OPS_PRUNE_EDGES",
            "KHORA_DREAM_OPS__PRUNE_EDGES",
        ),
    )
    compact_facts: bool = Field(
        default=False,
        description="Collapse superseded fact records.",
        validation_alias=AliasChoices(
            "KHORA_DREAM_OPS_COMPACT_FACTS",
            "KHORA_DREAM_OPS__COMPACT_FACTS",
        ),
    )
    cluster_events: bool = Field(
        default=False,
        description="Group related events into episode summaries.",
        validation_alias=AliasChoices(
            "KHORA_DREAM_OPS_CLUSTER_EVENTS",
            "KHORA_DREAM_OPS__CLUSTER_EVENTS",
        ),
    )
    recompute_centroids: bool = Field(
        default=False,
        description="Recompute entity / cluster centroid embeddings.",
        validation_alias=AliasChoices(
            "KHORA_DREAM_OPS_RECOMPUTE_CENTROIDS",
            "KHORA_DREAM_OPS__RECOMPUTE_CENTROIDS",
        ),
    )


class DreamConfig(BaseSettings):
    """Dream-phase configuration.

    Lives at ``KhoraConfig.dream``. Env-var precedence:

    - ``KHORA_DREAM_ENABLED=true``
    - ``KHORA_DREAM_DEFAULT_MODE=apply``
    - ``KHORA_DREAM_LLM_MAX_TOKENS_PER_RUN=200000``
    - ``KHORA_DREAM_OPS_DEDUPE_ENTITIES=true`` (single-underscore canonical;
      the legacy double-underscore form ``KHORA_DREAM_OPS__DEDUPE_ENTITIES``
      is also accepted — see #789)

    Modelled as :class:`pydantic_settings.BaseSettings` (matching
    :class:`khora.config.PipelineSettings` /
    :class:`khora.hooks.SemanticHooksConfig`) so flat
    ``KHORA_DREAM_*`` env vars populate top-level fields and
    ``KHORA_DREAM_OPS_*`` populates the nested
    :class:`DreamOpsConfig`.
    """

    model_config = SettingsConfigDict(
        env_prefix="KHORA_DREAM_",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_dream_alias_conflicts(cls, data: Any) -> Any:
        """Reject conflicting single/double-underscore dream env vars.

        Mirrors the behavior of
        :meth:`khora.config.schema.KhoraConfig._reject_alias_conflicts`
        for dream-owned env vars (the ``KHORA_DREAM_*`` namespace). Same
        same-value-on-both is OK / different-values raises rule. Also
        promotes the canonical single-underscore form into the
        nested-dict slot pydantic-settings expects.
        """
        env = os.environ
        conflicts: list[str] = []
        for canonical, legacy in _DREAM_ENV_ALIAS_PAIRS:
            new_val = env.get(canonical)
            old_val = env.get(legacy)
            if new_val is None and old_val is None:
                continue
            if new_val is not None and old_val is not None and new_val != old_val:
                conflicts.append(f"  - {canonical} and {legacy} are both set to different values")
                continue
            if new_val is not None and old_val is None and isinstance(data, dict):
                _inject_dream_canonical_env(data, legacy, new_val)
        if conflicts:
            raise ValueError(
                "Conflicting Khora dream env vars: the same field is configured "
                "via both the new single-underscore form and the legacy "
                "double-underscore form with different values. Pick one — the "
                "single-underscore form is preferred.\n" + "\n".join(conflicts)
            )
        return data

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
    source_chunk_ids_gc_enabled: bool = Field(
        default=False,
        description=(
            "Master switch for the vectorcypher source_chunk_ids GC op "
            "(OpKind.VECTORCYPHER_SOURCE_CHUNK_IDS_GC, #1263). Default OFF "
            "— the apply path rewrites the entity source_chunk_ids array. "
            "When False the op is skipped even if requested by "
            "scope.op_kinds, and a structured skip_reason is recorded."
        ),
    )
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
    centroid_recompute_enabled: bool = Field(
        default=False,
        description=(
            "Master switch for the vectorcypher centroid-recompute op "
            "(OpKind.VECTORCYPHER_CENTROID_RECOMPUTE, #1263). Default OFF "
            "— the apply path rewrites a merged entity's canonical "
            "embedding. When False the op is skipped even if requested by "
            "scope.op_kinds, and a structured skip_reason is recorded."
        ),
    )
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
    dedupe_entities_enabled: bool = Field(
        default=False,
        description=(
            "Master switch for the vectorcypher cross-batch dedupe op "
            "(OpKind.VECTORCYPHER_DEDUPE_ENTITIES, #1263/#1265). Default "
            "OFF — dedupe soft-deletes absorbed entity rows and rewrites "
            "relationship endpoints on apply, so operators opt in. When "
            "False the op is skipped even if requested by scope.op_kinds, "
            "and a structured skip_reason is recorded."
        ),
    )
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

    # Phase 4.1 — two-LLM judge for borderline dedupe merges (#667).
    dedupe_verifier_band_low: float = Field(
        default=0.78,
        ge=0.0,
        le=1.0,
        description=(
            "Lower edge of the borderline-merge band. Candidate pairs with "
            "cosine similarity in [dedupe_verifier_band_low, "
            "dedupe_verifier_band_high) are routed through the two-LLM "
            "judge before applying. Pairs above the band skip the verifier; "
            "pairs below it are rejected by the planner threshold already."
        ),
    )
    dedupe_verifier_band_high: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description=(
            "Upper edge of the borderline-merge band. Pairs at or above "
            "this score skip the verifier and apply directly."
        ),
    )
    dedupe_verifier_model: str = Field(
        default="gpt-4o-mini",
        description=(
            "LiteLLM model id for the verifier (first judge) on borderline "
            "dedupe merges. Configurable per call via DreamConfig."
        ),
    )
    dedupe_auditor_model: str = Field(
        default="claude-haiku-4.5",
        description=(
            "LiteLLM model id for the auditor (second judge, distinct "
            "model family from dedupe_verifier_model). Both judges must "
            "agree on 'merge' before a borderline op applies."
        ),
    )
    dedupe_verifier_timeout_seconds: int = Field(
        default=10,
        gt=0,
        description=(
            "Per-judge LLM timeout in seconds. A timeout / transport error "
            "degrades the joint verdict to decision='defer' (the merge is "
            "not applied)."
        ),
    )
    dedupe_verifier_min_confidence: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description=(
            "Confidence floor each judge must report when voting 'merge' "
            "before the dispatcher returns 'merge'. Below-floor confidence "
            "degrades to defer."
        ),
    )

    @model_validator(mode="after")
    def _enforce_verifier_band_ordering(self) -> DreamConfig:
        """Reject configurations where the low edge exceeds the high edge."""
        if self.dedupe_verifier_band_low >= self.dedupe_verifier_band_high:
            raise ValueError(
                "dedupe_verifier_band_low must be strictly less than "
                f"dedupe_verifier_band_high (got low={self.dedupe_verifier_band_low}, "
                f"high={self.dedupe_verifier_band_high})."
            )
        return self

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

    # Phase 5 (#1281) — vectorcypher contradiction reconciliation. A two-LLM
    # judge promotes the report-only detector to an opt-in mutating op that
    # soft-deletes the losing edge of a judge-agreed contradiction. Default
    # OFF — the judge is expensive (two LLM calls per flagged pair) and the
    # op mutates the graph, so operators must opt in.
    contradiction_reconcile_enabled: bool = Field(
        default=False,
        description=(
            "Master switch for the vectorcypher contradiction-reconciliation "
            "op (OpKind.VECTORCYPHER_CONTRADICTION_RECONCILE, #1281). Default "
            "OFF. When True, the planner emits a reconcile op (instead of the "
            "report-only detect op) that runs a two-LLM judge on each flagged "
            "pair; only judge-AGREED contradictions soft-delete the losing "
            "edge (mirrored to the graph), and defer/keep outcomes write a "
            "triage row to dream_conflicts. The two-LLM judge is budget-gated "
            "via the dream LLM token budgets. Postgres-only on apply."
        ),
    )
    contradiction_reconcile_model: str = Field(
        default="gpt-4o-mini",
        description=(
            "LiteLLM model id for the first judge (verifier) on a borderline "
            "contradiction. Mirrors dedupe_verifier_model."
        ),
    )
    contradiction_reconcile_auditor_model: str = Field(
        default="claude-haiku-4.5",
        description=(
            "LiteLLM model id for the second judge (auditor, distinct model "
            "family from contradiction_reconcile_model). Both judges must "
            "agree before the losing edge is invalidated."
        ),
    )
    contradiction_reconcile_timeout_seconds: int = Field(
        default=10,
        gt=0,
        description=(
            "Per-judge LLM timeout in seconds. A timeout / transport error "
            "degrades the joint verdict to 'defer' (no mutation)."
        ),
    )
    contradiction_reconcile_min_confidence: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description=(
            "Confidence floor each judge must report when voting 'invalidate' "
            "before the dispatcher returns 'invalidate'. Below-floor "
            "confidence degrades to defer."
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

    # Phase 5.4 — vectorcypher schema-drift normalization (#673).
    normalize_schema_enabled: bool = Field(
        default=False,
        description=(
            "Master toggle for the vectorcypher schema-drift normalization "
            "op. Default OFF — type renames touch the consumer contract "
            "(khora-cli, khora-explorer) and require coordinated release."
        ),
    )
    normalize_schema_mapping: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Operator-supplied old_type -> new_type rename mapping for "
            "the vectorcypher schema-drift normalization op. Applies to "
            "both entity_type and relationship_type columns. Never "
            "auto-derived — when empty, the op emits a single DreamOp "
            "with decision='insufficient_input' and refuses to run."
        ),
    )
