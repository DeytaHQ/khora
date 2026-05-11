# ADR-022: Extraction Skills Public API

- **Status:** Accepted
- **Date:** 2026-04-16 (extended Chronicle #1: events/facts toggles)
- **Deciders:** Khora architecture team

## Context

Khora's expertise system (`khora.extraction.skills`) defines the domain-knowledge
dataclasses used to drive entity and relationship extraction. Downstream
consumers construct these objects directly — either from YAML via the loader,
or programmatically — and pass them into `Khora.remember()` /
`remember_batch()` via the `expertise` parameter.

The original ADR-022 (0.5.0 release) stabilized three dataclasses as the
public contract: `ExpertiseConfig`, `EntityTypeConfig`, and
`RelationshipTypeConfig`. In practice, downstream consumers have also taken
direct dependencies on the remaining configuration dataclasses exposed from
`khora.extraction.skills.base`, because:

- `ExpertiseConfig` embeds `ConfidenceConfig` and `ExpansionConfig` as fields.
- `ExpertiseConfig.correlation_rules` is a `list[CorrelationRule]`.
- `ExpertiseConfig.inference_rules` is a `list[InferenceRule]`.

You cannot construct a non-trivial `ExpertiseConfig` without naming these
types. They are already re-exported from `khora.extraction.skills.__init__`
and documented in `docs/extraction/expertise-system.md`. Treating them as
internal implementation details while they form the transitive surface of a
"stable" type is a contradiction we should resolve explicitly rather than
discover through a breaking change.

**Triggering consumer:** khora-explorer (new sibling repo) depends on khora
as a published package and constructs `ExpertiseConfig` instances that set
`confidence`, `expansion`, `correlation_rules`, and `inference_rules`
directly. Earlier consumers (genesis, khora-benchmarks, Poros/Peras for
LLMUsage) established the same pattern.

## Decision

The following nine dataclasses in `khora.extraction.skills.base` are the
**stable public API** for extraction expertise configuration:

1. `ExpertiseConfig` — top-level domain knowledge definition
2. `EntityTypeConfig` — entity type with attributes and identifiers
3. `RelationshipTypeConfig` — relationship type with source/target constraints
4. `ConfidenceConfig` — confidence thresholds (`min_entity`, `min_relationship`, `min_inferred`)
5. `ExpansionConfig` — semantic expansion settings (mode, depth, batch size)
6. `CorrelationRule` — cross-tool entity correlation rule
7. `InferenceRule` — relationship inference rule
8. `EventExtractionConfig` — Chronicle SVO event extraction toggle (added Chronicle #1)
9. `FactExtractionConfig` — Chronicle atomic fact extraction toggle (added Chronicle #1)

`InferenceCondition` is a supporting type (a field of `InferenceRule.when`)
and is also part of the stable surface. `ConfidenceLevel` (enum) and
`ExtractionSkill` (legacy pre-ADR-022 class) remain exported for backward
compatibility but are not extended under this ADR.

These symbols are the canonical import paths:

```python
from khora.extraction.skills import (
    ExpertiseConfig,
    EntityTypeConfig,
    RelationshipTypeConfig,
    ConfidenceConfig,
    ExpansionConfig,
    CorrelationRule,
    InferenceRule,
    InferenceCondition,
    EventExtractionConfig,
    FactExtractionConfig,
)
```

`ExpertiseConfig`, `EntityTypeConfig`, and `RelationshipTypeConfig` are also
re-exported from the top-level `khora` package.

## Change Policy

- **Additive changes** (new optional fields with defaults, new classmethods,
  new helper methods that do not alter existing signatures) are permitted in
  patch and minor releases. Adding a field must preserve existing
  `from_dict` / `to_dict` round-trips for older payloads.
- **Breaking changes** (renaming a field, removing a field, changing a field
  type, changing a default in a way that alters observable behavior, or
  removing a class) require a **major version bump** and prior coordination
  with downstream maintainers: khora-explorer, genesis, khora-benchmarks, and
  Poros/Peras (the latter two for the adjacent `LLMUsage` contract).
- `from_dict` must continue to accept the historical YAML/JSON schema
  documented in `docs/extraction/expertise-system.md` for at least one major
  version after any schema evolution.
- Internal helpers, private attributes (leading underscore), and
  implementation modules outside `khora.extraction.skills.base` remain free
  to change without notice.

The `__all__` export list in `khora/extraction/skills/base.py` is the
machine-readable source of truth for this contract. Any class not listed
there is implementation detail.

## Consequences

**Positive**

- Downstream consumers can depend on these types without fear of silent
  breaking changes between patch releases.
- The contract is explicit and discoverable (`__all__`, this ADR,
  `CLAUDE.md` Downstream section).
- Refactors inside the extraction pipeline remain unconstrained as long as
  the seven dataclasses preserve their shape.

**Negative**

- Adding optional fields to these dataclasses is now a coordinated change.
  Contributors must choose: extend the stable dataclass (permanent
  commitment) or keep new configuration on an internal/adjacent type.
- YAML schema evolution must stay backward-compatible for one major
  version.

**Neutral**

- This ADR ratifies existing downstream usage; it does not change any
  runtime behavior. No migration is required.

## Related

- ADR-035 — CLAUDE.md structure (Downstream section references this ADR).
- `LLMUsage` contract (DYT-645) — separate stable surface consumed by
  Poros/Peras.
- `docs/extraction/expertise-system.md` — user-facing reference for the
  YAML schema and programmatic usage of these types.
