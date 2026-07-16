# Expertise System

The expertise system provides configurable domain knowledge for entity extraction. This document covers `ExpertiseConfig` and related configuration classes.

## Overview

Expertise configurations define:
- Entity types and their attributes
- Relationship types and constraints
- Correlation rules for cross-tool matching
- Inference rules for relationship discovery
- Custom prompts for extraction

## ExpertiseConfig

The complete domain knowledge definition:

```python
@dataclass
class ExpertiseConfig:
    name: str                    # "saas_expert"
    version: str = "1.0.0"
    description: str = ""
    extends: list[str] = []      # Inherit from other configs

    # LLM prompts (Jinja2 templates)
    system_prompt: str | None = None
    extraction_prompt: str | None = None

    # Type definitions
    entity_types: list[EntityTypeConfig] = []
    relationship_types: list[RelationshipTypeConfig] = []

    # Tool-specific schemas
    tool_schemas: dict[str, dict[str, Any]] = {}

    # Correlation rules
    correlation_rules: list[CorrelationRule] = []

    # Inference rules
    inference_rules: list[InferenceRule] = []

    # Confidence thresholds
    confidence: ConfidenceConfig = ConfidenceConfig()

    # Expansion settings
    expansion: ExpansionConfig = ExpansionConfig()

    # Additional metadata
    metadata: dict[str, Any] = {}
```

## EntityTypeConfig

Define entity types and their expected attributes:

```python
@dataclass
class EntityTypeConfig:
    name: str                    # "COMPANY"
    description: str = ""        # "A business organization"
    attributes: dict[str, list[str]] = {}  # {required: [], optional: []}
    identifiers: list[str] = []  # For cross-tool matching (e.g., "domain")
    aliases: list[str] = []      # Alternative names for this type
```

Example:
```python
EntityTypeConfig(
    name="COMPANY",
    description="A business organization or corporation",
    attributes={
        "required": ["name"],
        "optional": ["domain", "industry", "founded_year", "employee_count"],
    },
    identifiers=["domain", "linkedin_url"],
    aliases=["ORGANIZATION", "BUSINESS"],
)
```

## RelationshipTypeConfig

Define relationship types with source/target constraints:

```python
@dataclass
class RelationshipTypeConfig:
    name: str                    # "WORKS_FOR"
    description: str = ""        # "Employment relationship"
    source_types: list[str] = [] # ["PERSON"] or ["*"] for any
    target_types: list[str] = [] # ["COMPANY"] or ["*"] for any
    bidirectional: bool = False  # Create reverse edge?
    properties: list[str] = []   # Expected properties
```

Example:
```python
RelationshipTypeConfig(
    name="WORKS_FOR",
    description="Person is employed by company",
    source_types=["PERSON"],
    target_types=["COMPANY", "ORGANIZATION"],
    properties=["title", "department", "start_date"],
)
```

## CorrelationRule

Rules for matching entities across tools:

```python
@dataclass
class CorrelationRule:
    name: str                    # "email_match"
    description: str = ""
    pattern: str | None = None   # Regex for reference matching
    match_fields: list[str] = [] # Fields to match on (e.g., ["email"])
    entity_types: list[str] = [] # Types this rule applies to
    creates_relationship: str | None = None  # Relationship type to create
    confidence: float = 0.9      # Match confidence
```

Example:
```python
CorrelationRule(
    name="email_match",
    description="Match people by email address",
    match_fields=["email"],
    entity_types=["PERSON"],
    confidence=0.95,
)

CorrelationRule(
    name="jira_reference",
    description="Match JIRA issue references",
    pattern=r"[A-Z]+-\d+",       # Matches "PROJ-123"
    entity_types=["ISSUE", "TASK"],
    creates_relationship="REFERENCES",
)
```

## InferenceRule

Rules for inferring relationships:

```python
@dataclass
class InferenceCondition:
    relationship: str            # Relationship type to match
    source_type: str | None = None  # Source entity type filter
    target_type: str | None = None  # Target entity type filter

@dataclass
class InferenceRule:
    name: str                    # "colleague_inference"
    description: str = ""
    when: list[InferenceCondition] = []  # Conditions to match
    then_relationship: str = "" # Relationship to create
    then_source: str = "first.source"   # Source entity reference
    then_target: str = "second.target"  # Target entity reference
    confidence: float = 0.5     # Inferred relationship confidence
```

Example: Infer COLLEAGUES_WITH from shared WORKS_FOR
```python
InferenceRule(
    name="colleague_inference",
    description="People working for same company are colleagues",
    when=[
        InferenceCondition(relationship="WORKS_FOR", source_type="PERSON"),
        InferenceCondition(relationship="WORKS_FOR", source_type="PERSON"),
    ],
    then_relationship="COLLEAGUES_WITH",
    then_source="first.source",   # First person
    then_target="second.source",  # Second person
    confidence=0.6,
)
```

Example: Infer transitive ownership
```python
InferenceRule(
    name="transitive_ownership",
    description="If A owns B and B owns C, then A owns C",
    when=[
        InferenceCondition(relationship="OWNS"),
        InferenceCondition(relationship="OWNS"),
    ],
    then_relationship="OWNS",
    then_source="first.source",
    then_target="second.target",
    confidence=0.4,
)
```

## ConfidenceConfig

Confidence thresholds for filtering:

```python
@dataclass
class ConfidenceConfig:
    min_entity: float = 0.5        # Minimum entity confidence
    min_relationship: float = 0.5  # Minimum relationship confidence (see note below)
    min_inferred: float = 0.3      # Minimum inferred relationship confidence
```

> **Note:** The `min_relationship` default of 0.5 shown above is the dataclass fallback. The builtin `general.yaml` skill overrides this to `min_relationship: 0.25` for denser relationship extraction. If you are using the `general_entities` skill (the default), the effective threshold is 0.25.

## ExpansionConfig

Settings for semantic expansion:

```python
@dataclass
class ExpansionConfig:
    enabled: bool = True                  # Enable expansion?
    depth: int = 2                        # Inference passes
    cross_tool_unification: bool = True   # Enable entity dedup?
    relationship_inference: bool = True   # Enable inference?
    max_entities_per_expansion: int = 100 # Entity limit

    # "smart" (recommended), "batch", "incremental", "none"
    inference_mode: str = "smart"

    # Smart mode settings:
    preload_existing: bool = True         # Pre-load existing entities into index
    batch_storage_size: int = 50          # Entities per batch upsert
```

`inference_mode` controls when entity resolution and relationship inference happen:

| Mode | Description |
|------|-------------|
| `smart` | Per-doc O(1) dedup during ingestion, single O(n*k) resolution pass after all docs. Uses token-blocked `EntityIndex`. |
| `incremental` | Full expansion per document, fetching existing graph context each time. O(n^2) per doc. |
| `batch` | Skip inference during ingestion, run once on full graph afterward. O(n^2) once. |
| `none` | Unification only, no inference. |

`preload_existing` (smart mode only): When `true`, existing entities from the database are loaded into the in-memory `EntityIndex` before processing new documents. This ensures dedup works against stored entities, not just new ones. Set to `false` for clean namespaces.

`batch_storage_size` (smart mode only): Number of entities per database batch operation during post-ingestion resolution. Larger values reduce round-trips.

## YAML Configuration

Expertise can be defined in YAML:

```yaml
# saas_expert.yaml
name: saas_expert
version: "1.0.0"
description: "SaaS company extraction expertise"

entity_types:
  - name: COMPANY
    description: "A SaaS company"
    attributes:
      required: [name]
      optional: [domain, industry, funding_stage]
    identifiers: [domain]

  - name: PERSON
    description: "An individual"
    attributes:
      required: [name]
      optional: [email, title, linkedin_url]
    identifiers: [email, linkedin_url]

  - name: PRODUCT
    description: "A software product"
    attributes:
      optional: [pricing_model, target_market]

relationship_types:
  - name: WORKS_FOR
    source_types: [PERSON]
    target_types: [COMPANY]
    properties: [title, department]

  - name: BUILDS
    source_types: [COMPANY]
    target_types: [PRODUCT]

correlation_rules:
  - name: email_match
    match_fields: [email]
    entity_types: [PERSON]
    confidence: 0.95

inference_rules:
  - name: colleague_inference
    description: "Same company → colleagues"
    when:
      - relationship: WORKS_FOR
        source_type: PERSON
      - relationship: WORKS_FOR
        source_type: PERSON
    then:
      relationship: COLLEAGUES_WITH
      source: first.source
      target: second.source
    confidence: 0.6

confidence:
  min_entity: 0.6
  min_relationship: 0.5
  min_inferred: 0.3

expansion:
  enabled: true
  depth: 2
  inference_mode: smart            # "smart", "incremental", "batch", "none"
  preload_existing: true           # Load existing entities into index
  batch_storage_size: 50           # Entities per batch upsert
```

## Loading Expertise

### From File

```python
from khora.extraction.skills import load_expertise

# Load from file path
expertise = load_expertise("path/to/saas_expert.yaml")

# Load from built-in
expertise = load_expertise("builtin:general_entities")
```

### Programmatically

```python
from khora.extraction.skills import ExpertiseConfig, EntityTypeConfig

expertise = ExpertiseConfig(
    name="custom",
    entity_types=[
        EntityTypeConfig(name="COMPANY", description="A company"),
        EntityTypeConfig(name="PERSON", description="A person"),
    ],
    system_prompt="You are an expert at extracting company information.",
)
```

## Inheritance (extends)

Expertise can inherit from other configurations:

```yaml
# extended_saas.yaml
name: extended_saas
extends:
  - saas_expert  # Inherits all types, rules from saas_expert

# Add additional types
entity_types:
  - name: INVESTOR
    description: "An investment firm or investor"

# Add additional rules
inference_rules:
  - name: investor_portfolio
    when:
      - relationship: INVESTED_IN
    then:
      relationship: PORTFOLIO_COMPANY
```

## Jinja2 Templates

Prompts support Jinja2 templating:

Prompts render in a Jinja `ImmutableSandboxedEnvironment`; unsafe constructs (dunder/private attribute access, mutating methods) are rejected and raise `SecurityError`.

```yaml
system_prompt: |
  You are an expert at extracting information about {{ domain }} companies.
  Focus on identifying {{ entity_types | join(', ') }}.

extraction_prompt: |
  Extract entities from the following {{ source_type }} content:

  {{ text }}

  {% if company_context %}
  Context: This is about {{ company_context.name }}
  {% endif %}

  Return JSON with entities and relationships.
```

Usage:
```python
result = await extractor.extract(
    text,
    expertise=expertise,
    context={
        "domain": "SaaS",
        "source_type": "email",
        "company_context": {"name": "Acme Corp"},
    },
)
```

## Built-in Skills

Khora ships YAML-defined built-in skills. Load them with `load_expertise("builtin:<name>")`:

```python
from khora.extraction.skills import load_expertise

# General entities
expertise = load_expertise("builtin:general_entities")
# Types: PERSON, ORGANIZATION, CONCEPT, LOCATION

# Slack messages
expertise = load_expertise("builtin:slack")
# Types: PERSON, CHANNEL, TEAM, TOPIC, PROJECT, DECISION
# Extracts DM recipients, conversation threads, and team dynamics
```

The Slack skill (`extraction/skills/builtin/slack.yaml`) is designed for ingesting Slack workspace exports and DM histories. It includes correlation rules for matching users by Slack handle, and inference rules for team membership and collaboration patterns.

> **Legacy:** `ExtractionSkill.general_entities()`, `ExtractionSkill.technical_docs()`, `ExtractionSkill.business_intel()`, and `ExtractionSkill.research_papers()` are legacy classmethods that return a simpler `ExtractionSkill` object without expansion or confidence configuration. Prefer `load_expertise("builtin:...")` for new code. Retirement is tracked in issue #982.

## API Usage

### Via Khora

```python
result = await kb.remember(
    content,
    namespace=ns.namespace_id,
    expertise="saas_expert",  # Name or path
    entity_types=["PERSON", "ORG"],
    relationship_types=["WORKS_AT"],
)
```

### In Pipeline

```python
result = await ingest_documents(
    namespace_id,
    documents,
    storage,
    expertise="path/to/custom.yaml",
    enable_expansion=True,
)
```

### Direct Usage

```python
from khora.extraction.skills import load_expertise
from khora.extraction.extractors import LLMEntityExtractor

expertise = load_expertise("saas_expert")
extractor = LLMEntityExtractor()

result = await extractor.extract(
    text,
    expertise=expertise,
    context={"company_name": "Acme"},
)
```

## Next Steps

- [Extractors](extractors.md) - Entity extraction
- [Semantic Expansion](semantic-expansion.md) - Unification and inference
- [Ingestion Pipeline](ingestion-pipeline.md) - Full pipeline
