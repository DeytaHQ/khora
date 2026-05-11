# Expertise Configuration Guide

This directory contains example expertise configurations for Khora's knowledge graph extraction system. These configurations define domain-specific knowledge that guides entity and relationship extraction.

## Quick Start

1. **Copy an example** that matches your domain:
   ```bash
   cp examples/config/expertise/saas_expert.yaml ~/.khora/expertise/my_expertise.yaml
   ```

2. **Customize** the configuration for your needs

3. **Use with Khora**:
   ```python
   from khora import Khora
   from khora.extraction.skills import ExpertiseLoader

   loader = ExpertiseLoader()
   expertise = loader.load_file("~/.khora/expertise/my_expertise.yaml")

   async with Khora() as lake:
       result = await lake.remember(content, expertise=expertise)
   ```

## Configuration Format

Expertise configurations are YAML files with the following structure:

```yaml
name: my_expertise
version: "1.0.0"
description: "Description of this expertise"

# Inherit from other expertise configs
extends:
  - builtin:general  # Built-in general entities
  - file:./base.yaml # Local file

# LLM system prompt (Jinja2 templates supported)
system_prompt: |
  You are an expert in {{ domain }}...

  {% for entity_type in entity_types %}
  - {{ entity_type.name }}: {{ entity_type.description }}
  {% endfor %}

# Custom extraction prompt (optional)
extraction_prompt: |
  Extract entities from: {{ text }}
  Entity types: {{ entity_types | join(", ") }}

# Entity type definitions
entity_types:
  - name: ENTITY_NAME
    description: "What this entity represents"
    attributes:
      required: [id, name]
      optional: [description, metadata]
    identifiers: [id]  # Fields used for cross-tool matching

# Relationship type definitions
relationship_types:
  - name: RELATIONSHIP_NAME
    description: "What this relationship means"
    source_types: [ENTITY_A]
    target_types: [ENTITY_B]
    bidirectional: false

# Tool-specific schemas
tool_schemas:
  jira:
    issue_key_pattern: "[A-Z]+-\\d+"
    statuses: [backlog, in_progress, done]

# Cross-tool correlation rules
correlation_rules:
  - name: email_match
    description: "Match entities by email"
    match_fields: [email]
    entity_types: [PERSON, CONTACT]
    confidence: 0.95

# Relationship inference rules
inference_rules:
  - name: transitive_ownership
    description: "If A owns B and B owns C, A indirectly owns C"
    when:
      - relationship: OWNS
        source_type: ORGANIZATION
        target_type: ORGANIZATION
      - relationship: OWNS
        source_type: ORGANIZATION
        target_type: ASSET
    then:
      relationship: INDIRECTLY_OWNS
      source: first.source
      target: second.target
    confidence: 0.7

# Confidence thresholds
confidence:
  min_entity: 0.5
  min_relationship: 0.5
  min_inferred: 0.3

# Expansion settings
expansion:
  enabled: true
  depth: 2
  cross_tool_unification: true
  relationship_inference: true
```

## Available Examples

| File | Description | Use Case |
|------|-------------|----------|
| `general.yaml` | Basic entities | General-purpose extraction |
| `saas_expert.yaml` | SaaS tools expertise | Jira, Salesforce, Slack, GitHub, etc. |
| `technical_docs.yaml` | Technical documentation | API docs, code, architecture |
| `business_intel.yaml` | Business intelligence | Reports, metrics, stakeholders |
| `healthcare.yaml` | Healthcare domain | Patients, providers, diagnoses |
| `custom_template.yaml` | Blank template | Starting point for custom expertise |

## Inheritance

Use `extends` to inherit from other configurations:

```yaml
name: my_custom
extends:
  - builtin:general           # Built-in general expertise
  - builtin:saas_expert       # Built-in SaaS expertise
  - file:./company_specific.yaml  # Local file
```

Configurations are merged in order:
- Later configs override earlier ones for same-named items
- Entity types, relationships, and rules are combined
- Prompts from later configs replace earlier ones

## Jinja2 Templates

System and extraction prompts support Jinja2 templating:

```yaml
system_prompt: |
  You are an expert in {{ expertise.description }}.

  Entity types to extract:
  {% for et in entity_types %}
  - {{ et.name }}: {{ et.description }}
  {% endfor %}

  {% if tool_schemas.jira %}
  Jira issue pattern: {{ tool_schemas.jira.issue_key_pattern }}
  {% endif %}
```

Available template variables:
- `expertise`: The full ExpertiseConfig object
- `entity_types`: List of EntityTypeConfig objects
- `relationship_types`: List of RelationshipTypeConfig objects
- `tool_schemas`: Dict of tool schema configurations
- `tools`: List of tool names (keys of tool_schemas)
- `parent_prompt`: Parent's system prompt (for inheritance)
- Any custom `context` passed at runtime

## Programmatic Configuration

You can also define expertise in Python:

```python
from khora.extraction.skills import (
    ExpertiseConfig,
    EntityTypeConfig,
    RelationshipTypeConfig,
    CorrelationRule,
)

expertise = ExpertiseConfig(
    name="my_expertise",
    description="Custom domain expertise",
    entity_types=[
        EntityTypeConfig(
            name="CUSTOM_ENTITY",
            description="My custom entity type",
            attributes={"required": ["id"], "optional": ["name"]},
        ),
    ],
    relationship_types=[
        RelationshipTypeConfig(
            name="CUSTOM_RELATIONSHIP",
            description="My custom relationship",
            source_types=["CUSTOM_ENTITY"],
            target_types=["CUSTOM_ENTITY"],
        ),
    ],
    correlation_rules=[
        CorrelationRule(
            name="id_match",
            match_fields=["id"],
            entity_types=["CUSTOM_ENTITY"],
        ),
    ],
)

# Use directly
async with Khora() as lake:
    result = await lake.remember(content, expertise=expertise)

# Or register for reuse
from khora.extraction.skills import register_expertise
register_expertise(expertise)
```

## Best Practices

1. **Start with an example** - Copy the closest matching example and customize
2. **Use inheritance** - Build on `general` for common entity types
3. **Be specific** - More detailed descriptions lead to better extraction
4. **Add correlation rules** - Help unify entities across different sources
5. **Test iteratively** - Start simple and add complexity as needed
6. **Version your configs** - Use the `version` field to track changes
