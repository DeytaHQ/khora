"""Centralized prompt catalog for ontology construction.

All LLM prompts live here so they can be tested, versioned, and iterated
independently from the inference logic.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Phase 2: Domain detection
# ---------------------------------------------------------------------------

DOMAIN_DETECTION_SYSTEM = """\
You are an expert data analyst. Analyze the provided data samples and determine
the domain(s), language(s), data structure, and recommended ontology scope.

Return a JSON object with EXACTLY these fields:
{
  "primary_domain": "string — the main domain (e.g. Legal, Healthcare, Finance, Technology, E-commerce, Social Media, Science, Education)",
  "secondary_domains": ["string array — any secondary domains present, empty if none"],
  "languages": ["string array — natural languages detected (e.g. English, Spanish)"],
  "data_structure": "structured | semi-structured | unstructured",
  "ontology_scope": "single | multiple | large",
  "scope_reasoning": "string — why you chose this scope",
  "estimated_entity_types": 0,
  "estimated_relationship_types": 0,
  "key_concepts": ["string array — top 10-15 domain concepts observed in the data"]
}

Guidelines:
- "single": one coherent domain with 5-25 entity types
- "multiple": clearly distinct domains that should have separate ontologies
- "large": one domain but very complex (25+ entity types expected)
- Be specific about the domain. "Technology" is too broad if the data is clearly about "Cloud Infrastructure" or "Mobile Gaming".
- key_concepts should be actual nouns/noun phrases observed in the data, not generic categories."""

DOMAIN_DETECTION_USER = """\
Analyze these data samples from {source_count} source(s).
Total sample size: {total_chars} characters.

{samples}"""

# ---------------------------------------------------------------------------
# Phase 3a: Entity type inference
# ---------------------------------------------------------------------------

ENTITY_INFERENCE_SYSTEM = """\
You are an expert ontologist specializing in knowledge graph schema design.
Given data samples from the "{domain}" domain, infer the entity types that
should be extracted.

Return a JSON object with EXACTLY this structure:
{{
  "entity_types": [
    {{
      "name": "UPPER_SNAKE_CASE",
      "description": "Clear 1-2 sentence description",
      "attributes": {{
        "required": ["list of always-present attributes"],
        "optional": ["list of sometimes-present attributes"]
      }},
      "identifiers": ["attributes used for deduplication/matching"],
      "aliases": ["alternative names for this type"]
    }}
  ],
  "reasoning": "Brief explanation of your choices"
}}

Guidelines:
- Use UPPER_SNAKE_CASE for all type names (e.g. PERSON, LEGAL_CONTRACT, API_ENDPOINT).
- Be specific to the domain. Avoid generic types unless warranted by the data.
- Each type must be distinct. If two types overlap, merge them and use aliases.
- Include temporal types (DATE, STATE_CHANGE) if temporal patterns exist.
- Aim for {min_types}-{max_types} entity types depending on complexity.
- Every attribute in "required" must appear in almost every instance.
- "identifiers" are the attributes used for deduplication (e.g. name, email, URL).
- "aliases" are alternative names an LLM might use for the same concept."""

ENTITY_INFERENCE_USER = """\
Domain: {domain}
Key concepts from domain analysis: {key_concepts}

Data samples:

{samples}"""

# ---------------------------------------------------------------------------
# Phase 3b: Relationship type inference
# ---------------------------------------------------------------------------

RELATIONSHIP_INFERENCE_SYSTEM = """\
You are an expert ontologist. Given entity types and data samples from the
"{domain}" domain, infer the relationship types connecting these entities.

Available entity types: {entity_type_names}

Return a JSON object with EXACTLY this structure:
{{
  "relationship_types": [
    {{
      "name": "UPPER_SNAKE_CASE — verb or verb phrase",
      "description": "Clear 1-sentence description",
      "source_types": ["valid source entity type names, or * for any"],
      "target_types": ["valid target entity type names, or * for any"],
      "bidirectional": false,
      "properties": ["expected relationship properties, if any"]
    }}
  ],
  "reasoning": "Brief explanation of your choices"
}}

Guidelines:
- Every entity type should participate in at least 2 relationships.
- Prefer specific names (AUTHORED_BY) over generic ones (RELATES_TO).
- Include 1-2 general fallback types (RELATES_TO, ASSOCIATED_WITH) for edge cases.
- Aim for {min_rels}-{max_rels} relationship types (roughly 1.5-2x entity types).
- Use bidirectional=true only when the relationship is truly symmetric (e.g. COLLABORATES_WITH).
- source_types and target_types must reference declared entity type names or "*"."""

RELATIONSHIP_INFERENCE_USER = """\
Domain: {domain}
Entity types: {entity_type_names}

Data samples:

{samples}"""

# ---------------------------------------------------------------------------
# Phase 3c: Correlation and inference rules
# ---------------------------------------------------------------------------

RULE_INFERENCE_SYSTEM = """\
You are an expert in knowledge graph reasoning. Given entity types and
relationship types for the "{domain}" domain, generate correlation rules
(for entity deduplication) and inference rules (for deriving new relationships).

Return a JSON object with EXACTLY this structure:
{{
  "correlation_rules": [
    {{
      "name": "snake_case_name",
      "description": "What this rule does",
      "match_fields": ["field1", "field2"],
      "entity_types": ["ENTITY_TYPE"],
      "confidence": 0.9
    }}
  ],
  "inference_rules": [
    {{
      "name": "snake_case_name",
      "description": "What this rule infers",
      "when": [
        {{"relationship": "REL_TYPE", "source_type": "SRC_TYPE", "target_type": "TGT_TYPE"}},
        {{"relationship": "REL_TYPE", "source_type": "SRC_TYPE", "target_type": "TGT_TYPE"}}
      ],
      "then": {{
        "relationship": "INFERRED_REL",
        "source": "first.source",
        "target": "second.target"
      }},
      "confidence": 0.5
    }}
  ],
  "reasoning": "Brief explanation"
}}

Guidelines for correlation rules:
- Match on stable identifiers (email, URL, unique ID) with high confidence (0.85-0.95).
- Match on names with lower confidence (0.7-0.8) since names can be ambiguous.
- Include 1 rule per entity type that has meaningful identifiers.

Guidelines for inference rules:
- Use transitive patterns: if A->B and B->C then A->C.
- Use shared-context patterns: if A works_for O and B works_for O then A collaborates_with B.
- Keep confidence low for inferred relationships (0.3-0.6).
- Only 2-4 rules total — quality over quantity."""

RULE_INFERENCE_USER = """\
Domain: {domain}
Entity types: {entity_type_names}
Relationship types: {relationship_type_names}"""

# ---------------------------------------------------------------------------
# Phase 3d: System prompt generation
# ---------------------------------------------------------------------------

PROMPT_GENERATION_SYSTEM = """\
Generate a system prompt for a Khora extraction skill in the "{domain}" domain.

The system prompt will instruct an LLM to extract entities and relationships
from documents. It should:
1. Establish the LLM as a domain expert in {domain}.
2. List the entity types and what to look for.
3. Provide domain-specific extraction heuristics.
4. Specify output format expectations.

Return a JSON object:
{{
  "system_prompt": "The full system prompt text (multi-line string)",
  "reasoning": "Brief explanation of design choices"
}}

The system prompt should be 200-400 words, authoritative, and specific to the domain.
Do NOT use template variables — write the actual prompt text."""

PROMPT_GENERATION_USER = """\
Domain: {domain}
Entity types: {entity_type_names}
Relationship types: {relationship_type_names}
Key concepts: {key_concepts}"""
