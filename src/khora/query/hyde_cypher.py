"""HyDE-Cypher templated query generator (Phase D2, Issue #595).

For structured RECENCY queries — "latest action items from John's
meetings", "recent escalations from Acme", "co-occurrences of Phoenix
and security in the last week" — text-HyDE generates a free-form
hypothetical that *describes* the answer, then embeds it. HyDE-Cypher
takes a different tack: have the LLM pick a parameterized Cypher
template, fill slots, and execute the result against the graph
backend. The retrieved entities are returned as additional candidates
for downstream fusion.

Two-step LLM call:

1. Classify-and-fill: structured output schema fixes the template id
   and slot values. If the LLM returns ``"none"``, the caller falls
   back to text HyDE.
2. (No second step.) Templates are entirely static — the LLM only
   selects + fills slots, never writes Cypher.

Injection safety: slot values are bound via Neo4j parameters
(``$name``), never string-interpolated into the Cypher source.
Slot values are additionally validated against
:class:`ExpertiseConfig` whitelists for entity_type / relationship_type
fields so a hallucinated type cannot widen the query surface.

Default OFF — gated by ``QuerySettings.enable_hyde_cypher``. The
acceptance bar (HyDE-Cypher beats text-HyDE on a hand-curated
structured-query set AND ties on the unstructured set) requires an
eval harness; until that's run, this module is opt-in via config.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from khora.config.llm import LiteLLMConfig
    from khora.extraction.skills import ExpertiseConfig


# ---------------------------------------------------------------------------
# Template registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HyDECypherTemplate:
    """A parameterized Cypher template the LLM may select from.

    The Cypher string uses ``$name`` parameter placeholders only — slot
    values are never string-interpolated. ``slots`` lists the required
    parameter names; ``type_slots`` lists which slots must be validated
    against an entity/relationship-type whitelist.
    """

    id: str
    description: str
    cypher: str
    slots: tuple[str, ...] = field(default_factory=tuple)
    entity_type_slots: tuple[str, ...] = field(default_factory=tuple)
    relationship_type_slots: tuple[str, ...] = field(default_factory=tuple)


# All templates return ``(e:Entity)`` rows ordered most-recent-first.
# The engine treats the returned entities as additional candidates for
# fusion with vector results.
TEMPLATES: dict[str, HyDECypherTemplate] = {
    "recent_by_type": HyDECypherTemplate(
        id="recent_by_type",
        description=(
            "Entities of a specific type touched in the last N days. "
            "Use for queries like 'latest action items', 'recent meetings'."
        ),
        cypher=(
            "MATCH (e:Entity {namespace_id: $namespace_id, entity_type: $entity_type}) "
            "WHERE e.updated_at >= datetime() - duration({days: $days}) "
            "RETURN e ORDER BY e.updated_at DESC LIMIT $limit"
        ),
        slots=("entity_type", "days"),
        entity_type_slots=("entity_type",),
    ),
    "entity_relationships": HyDECypherTemplate(
        id="entity_relationships",
        description=(
            "All entities related to a named entity via a specific relationship "
            "type. Use for queries like 'who works for Acme', 'tasks owned by John'."
        ),
        cypher=(
            "MATCH (source:Entity {namespace_id: $namespace_id, name: $name})"
            "-[r:`RELATES_TO` {relationship_type: $relationship_type}]-(other:Entity) "
            "RETURN other AS e ORDER BY r.updated_at DESC LIMIT $limit"
        ),
        slots=("name", "relationship_type"),
        relationship_type_slots=("relationship_type",),
    ),
    "cooccurrence": HyDECypherTemplate(
        id="cooccurrence",
        description=(
            "Entities co-mentioned with both of two named entities within the "
            "last N days. Use for queries like 'Phoenix and security recently'."
        ),
        cypher=(
            "MATCH (a:Entity {namespace_id: $namespace_id, name: $entity_a})"
            "<-[:MENTIONED_IN]-(c:Chunk)-[:MENTIONED_IN]->(b:Entity {name: $entity_b}) "
            "WHERE c.created_at >= datetime() - duration({days: $days}) "
            "MATCH (c)<-[:MENTIONED_IN]-(e:Entity) "
            "WHERE e.id <> a.id AND e.id <> b.id "
            "RETURN DISTINCT e ORDER BY c.created_at DESC LIMIT $limit"
        ),
        slots=("entity_a", "entity_b", "days"),
    ),
}


# ---------------------------------------------------------------------------
# LLM selector
# ---------------------------------------------------------------------------


def _selector_system_prompt() -> str:
    template_lines = []
    for tpl in TEMPLATES.values():
        slots_csv = ", ".join(tpl.slots) if tpl.slots else "(no slots)"
        template_lines.append(f"- {tpl.id}: {tpl.description} slots: {slots_csv}")
    body = "\n".join(template_lines)
    return (
        "You are a query classifier. Given a user query, pick the single best "
        "templated graph query that retrieves entities answering it. If no "
        'template fits cleanly, return ``"id": "none"`` so the caller falls back '
        "to text-HyDE.\n\n"
        "Templates:\n"
        f"{body}\n\n"
        "Respond ONLY with a single JSON object of the shape "
        '{"id": "<template_id_or_none>", "slots": {"slot_name": "value", ...}}. '
        "No prose, no markdown, no code fences."
    )


@dataclass(frozen=True, slots=True)
class HyDECypherSelection:
    """LLM classifier output: template id + filled slots.

    ``template_id == "none"`` signals "no template applies — fall back to
    text HyDE". An unknown id is treated the same way.
    """

    template_id: str
    slots: dict[str, Any]


async def select_template(
    query: str,
    llm_config: LiteLLMConfig | None = None,
    *,
    model: str = "gpt-4o-mini",
) -> HyDECypherSelection:
    """Ask the LLM which template (if any) answers the query.

    Always returns a :class:`HyDECypherSelection`. On any error
    (timeout, parse failure, unknown id), returns ``template_id="none"``
    so the caller falls back to text HyDE — never raises, never crashes
    the query.
    """
    from khora.config.llm import LiteLLMConfig, acompletion

    config = llm_config or LiteLLMConfig(model=model, temperature=0.0, max_tokens=200)
    try:
        response = await acompletion(
            prompt=query,
            config=config,
            system_prompt=_selector_system_prompt(),
            response_format={"type": "json_object"},
            _telemetry_op="hyde_cypher.select",
        )
    except Exception as exc:  # noqa: BLE001 — degrade gracefully
        logger.warning("HyDE-Cypher template selection failed: {}", exc)
        return HyDECypherSelection(template_id="none", slots={})

    try:
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").lstrip("json").strip()
        payload = json.loads(cleaned)
    except (json.JSONDecodeError, AttributeError):
        return HyDECypherSelection(template_id="none", slots={})

    template_id = str(payload.get("id", "none"))
    slots = payload.get("slots", {})
    if not isinstance(slots, dict):
        slots = {}
    if template_id not in TEMPLATES and template_id != "none":
        # Hallucinated id — degrade.
        template_id = "none"
        slots = {}
    return HyDECypherSelection(template_id=template_id, slots=slots)


# ---------------------------------------------------------------------------
# Validator + generator
# ---------------------------------------------------------------------------


class HyDECypherValidationError(ValueError):
    """Raised when a HyDE-Cypher selection fails validation.

    The engine catches this and falls back to text-HyDE — validation
    errors are never user-facing.
    """


def validate_selection(
    selection: HyDECypherSelection,
    expertise: ExpertiseConfig | None = None,
) -> HyDECypherTemplate:
    """Validate the LLM's template selection against the registry and
    (optionally) the user's ExpertiseConfig whitelist.

    Returns the :class:`HyDECypherTemplate` on success. Raises
    :class:`HyDECypherValidationError` on:

    - Unknown template id
    - Missing required slot
    - Slot value not in the type whitelist (when expertise provided)

    Slot values themselves are validated as plain Python types here —
    actual injection safety comes from binding via Neo4j parameters at
    execution time. This is the schema-drift / authorization gate, not
    the SQL/Cypher-escape gate.
    """
    if selection.template_id not in TEMPLATES:
        raise HyDECypherValidationError(f"unknown template id: {selection.template_id!r}")
    template = TEMPLATES[selection.template_id]

    for slot in template.slots:
        if slot not in selection.slots:
            raise HyDECypherValidationError(f"missing slot {slot!r} for template {template.id!r}")

    if expertise is not None:
        et_whitelist = set(expertise.get_entity_type_names())
        rt_whitelist = set(expertise.get_relationship_type_names())
        for slot in template.entity_type_slots:
            value = str(selection.slots.get(slot, ""))
            if et_whitelist and value not in et_whitelist:
                raise HyDECypherValidationError(
                    f"entity_type {value!r} not in ExpertiseConfig whitelist for slot {slot!r}"
                )
        for slot in template.relationship_type_slots:
            value = str(selection.slots.get(slot, ""))
            if rt_whitelist and value not in rt_whitelist:
                raise HyDECypherValidationError(
                    f"relationship_type {value!r} not in ExpertiseConfig whitelist for slot {slot!r}"
                )

    return template


def generate_cypher(
    selection: HyDECypherSelection,
    namespace_id: Any,
    *,
    limit: int = 20,
    expertise: ExpertiseConfig | None = None,
) -> tuple[str, dict[str, Any]]:
    """Validate and return ``(cypher_string, params_dict)``.

    The Cypher string contains only ``$placeholder`` parameter references
    — slot values land in ``params_dict`` to be bound by the Neo4j driver,
    never string-interpolated. ``limit`` is exposed as a separate parameter
    so callers can size the result set without re-writing the template.

    Raises :class:`HyDECypherValidationError` for invalid selections; the
    engine catches this and falls back to text-HyDE.
    """
    template = validate_selection(selection, expertise=expertise)
    params: dict[str, Any] = {"namespace_id": str(namespace_id), "limit": int(limit)}
    for slot in template.slots:
        value = selection.slots[slot]
        if slot == "days":
            # ``duration({days: $days})`` requires an integer.
            try:
                params[slot] = int(value)
            except (TypeError, ValueError) as exc:
                raise HyDECypherValidationError(f"slot {slot!r} must be an integer, got {value!r}") from exc
        else:
            # Stringify the rest — defends against the LLM returning a
            # nested object/list as a slot value.
            params[slot] = str(value)
    return template.cypher, params
