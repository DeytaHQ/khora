"""Core API — a full ontology as an ``ExpertiseConfig``.

``03_ontology_config.py`` shows the minimum: bare ``entity_types`` /
``relationship_types`` lists. They're only *hints* — the model can emit
types outside them, and empty lists fall back to unbounded extraction.

Once a document carries more than a couple of facts, you usually want
more than two flat lists. An ``ExpertiseConfig`` is one reusable,
versionable object that holds the whole domain ontology:

  * a **system prompt** that tells the model how your domain is shaped,
  * typed entities with **identifiers** (so the same candidate mentioned
    twice — "Priya Patel" and "Priya" — collapses to one node),
  * typed relationships,
  * a **correlation rule** (cross-source dedup), and
  * an **inference rule** — the kind expansion uses to derive edges the text
    never states, e.g.

        CANDIDATE -APPLIED_TO-> ROLE  +  ROLE -ROLE_AT-> EMPLOYER
            =>  CANDIDATE -TARGETS-> EMPLOYER

Pass it via ``expertise=`` and mirror its type names into the still-required
``entity_types`` / ``relationship_types`` kwargs. The run below also **dumps the
exact prompt** the extractor sends — so you can see your ExpertiseConfig become the
system prompt + a list of *only* your declared types. Two pipeline steps add types
you never declared: **event extraction** (EVENT entities + PARTICIPATED_IN), which we
turn **off** here with ``VectorCypherConfig(store_events=False)``, and **co-occurrence**
densification (ASSOCIATED_WITH edges), which is not configurable in VectorCypher and
still appears.

Engine choice: **vectorcypher** — extraction + expansion live here.

Run it
======
uv run python examples/10_core_apis/06_expertise_config.py
python examples/10_core_apis/06_expertise_config.py
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from pathlib import Path

import litellm
from loguru import logger

from khora import EntityTypeConfig, ExpertiseConfig, Khora, RelationshipTypeConfig
from khora.config import KhoraConfig
from khora.engines.vectorcypher import VectorCypherConfig
from khora.extraction.skills import CorrelationRule, InferenceCondition, InferenceRule

logger.remove()
logger.add("khora.log", level="TRACE", enqueue=True)
for _noisy in ("httpx", "httpcore", "LiteLLM", "openai", "sqlalchemy.engine"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

_CONFIG = Path(__file__).parent.parent / "khora.embedded.yaml"

# A bit more text than 03 — enough facts (a repeated candidate, a role, an
# employer, skills, a hiring manager) to make the ontology earn its keep.
_NOTE = (
    "Priya Patel applied to the backend engineering role at Acme Robotics, a "
    "robotics startup in Berlin. She has five years of Python experience and "
    "previously shipped distributed systems at Globex. The hiring manager, "
    "Sam Chen, has scheduled Priya for a system-design interview next week and "
    "flagged her as a strong fit."
)

EXPERTISE = ExpertiseConfig(
    name="recruiting",
    version="1.0.0",
    description="Candidates, roles, employers, and skills in a hiring pipeline.",
    system_prompt=(
        "You extract recruiting information from hiring notes and messages. People "
        "who applied or were sourced are CANDIDATE; companies are EMPLOYER; open "
        "positions are ROLE; technical skills are SKILL; the person running the "
        "process is HIRING_MANAGER. Capture each candidate's skills and which "
        "employer a role belongs to."
    ),
    entity_types=[
        EntityTypeConfig(
            name="CANDIDATE",
            description="A person applying for or sourced into a role.",
            identifiers=["email", "name"],
            aliases=["APPLICANT"],
        ),
        EntityTypeConfig(
            name="EMPLOYER", description="A company that posts roles.", identifiers=["name"], aliases=["COMPANY"]
        ),
        EntityTypeConfig(name="ROLE", description="An open position.", identifiers=["name"]),
        EntityTypeConfig(name="SKILL", description="A technical or professional skill.", identifiers=["name"]),
        EntityTypeConfig(
            name="HIRING_MANAGER", description="The person running the hiring process.", identifiers=["name"]
        ),
    ],
    relationship_types=[
        RelationshipTypeConfig(
            name="APPLIED_TO",
            description="Candidate applied to a role.",
            source_types=["CANDIDATE"],
            target_types=["ROLE"],
        ),
        RelationshipTypeConfig(
            name="ROLE_AT",
            description="A role belongs to an employer.",
            source_types=["ROLE"],
            target_types=["EMPLOYER"],
        ),
        RelationshipTypeConfig(
            name="HAS_SKILL", description="Candidate has a skill.", source_types=["CANDIDATE"], target_types=["SKILL"]
        ),
        RelationshipTypeConfig(
            name="MANAGED_BY",
            description="A role is run by a hiring manager.",
            source_types=["ROLE"],
            target_types=["HIRING_MANAGER"],
        ),
        RelationshipTypeConfig(
            name="TARGETS",
            description="Candidate is pursuing an employer (derived).",
            source_types=["CANDIDATE"],
            target_types=["EMPLOYER"],
        ),
    ],
    correlation_rules=[
        CorrelationRule(
            name="dedupe_candidates",
            description="Same candidate seen in more than one note.",
            match_fields=["email", "name"],
            entity_types=["CANDIDATE"],
            confidence=0.85,
        ),
    ],
    inference_rules=[
        InferenceRule(
            name="candidate_targets_employer",
            description="Applied to a role at an employer => pursuing that employer.",
            when=[
                InferenceCondition(relationship="APPLIED_TO", source_type="CANDIDATE", target_type="ROLE"),
                InferenceCondition(relationship="ROLE_AT", source_type="ROLE", target_type="EMPLOYER"),
            ],
            then_relationship="TARGETS",
            then_source="first.source",
            then_target="second.target",
            confidence=0.6,
        ),
    ],
)


async def main() -> None:
    config = KhoraConfig.from_yaml(_CONFIG)

    # Turn OFF event extraction so the graph stays on-ontology — no EVENT entities
    # and no PARTICIPATED_IN edges from the LLM's `events` array. (Co-occurrence
    # ASSOCIATED_WITH edges are NOT configurable in VectorCypher and are still added.)
    vc_config = VectorCypherConfig(store_events=False)

    async with Khora(
        config,
        engine="vectorcypher",
        run_migrations=True,
        engine_kwargs={"vectorcypher_config": vc_config},
    ) as kb:
        ns_id = (await kb.create_namespace()).namespace_id

        # The ontology object carries far more than two flat lists:
        print(f"ExpertiseConfig '{EXPERTISE.name}' v{EXPERTISE.version}")
        print(f"  entity types       : {EXPERTISE.get_entity_type_names()}")
        print(f"  relationship types : {EXPERTISE.get_relationship_type_names()}")
        print(f"  correlation rules  : {[r.name for r in EXPERTISE.correlation_rules]}")
        print(f"  inference rules    : {[r.name for r in EXPERTISE.inference_rules]}")
        print(f"  system_prompt      : {'set' if EXPERTISE.system_prompt else 'none'}")

        # ── Dump the exact prompt the extractor sends to the LLM ─────────────
        # Instrumentation only: wrap litellm.acompletion to print the first
        # extraction call's messages, then restore and call through. This shows
        # what your ExpertiseConfig actually becomes — the system message is
        # *your* system_prompt, and the user message lists *only* your declared
        # types (the default broad prompt is fully replaced).
        _orig_acompletion = litellm.acompletion

        async def _show_prompt(*args, **kwargs):
            litellm.acompletion = _orig_acompletion  # print once, then restore
            print("\n===== prompt sent to the extractor LLM =====")
            for m in kwargs.get("messages", []):
                print(f"\n--- {m['role']} ---\n{m['content']}")
            print("===== end prompt =====")
            return await _orig_acompletion(*args, **kwargs)

        litellm.acompletion = _show_prompt

        # Pass the object via expertise=; the type-name kwargs are still required.
        result = await kb.remember(
            _NOTE,
            namespace=ns_id,
            title="hiring-note",
            expertise=EXPERTISE,
            entity_types=EXPERTISE.get_entity_type_names(),
            relationship_types=EXPERTISE.get_relationship_type_names(),
        )
        litellm.acompletion = _orig_acompletion  # ensure restored
        print(f"\nextracted {result.entities_extracted} entities, {result.relationships_created} relationships")

        entities = await kb.list_entities(namespace=ns_id, limit=50)
        histogram = Counter(e.entity_type for e in entities)
        print("\nentity-type histogram (driven by the ExpertiseConfig):")
        for etype, count in sorted(histogram.items(), key=lambda kv: -kv[1]):
            print(f"  {etype:16s} {count}")
        for e in entities[:10]:
            print(f"    • {e.name:34s} → {e.entity_type}")

        # The typed relationship graph the ontology produced (ids resolved to names).
        id2name = {e.id: e.name for e in entities}
        result = await kb.recall("Priya Patel Acme Robotics role", namespace=ns_id, limit=10)
        if result.relationships:
            print("\nrelationships (source → type → target):")
            for r in result.relationships:
                src = id2name.get(r.source_entity_id, str(r.source_entity_id)[:8])
                tgt = id2name.get(r.target_entity_id, str(r.target_entity_id)[:8])
                print(f"    {src} -[{r.relationship_type}]-> {tgt}")

        # The prompt above lists only your ontology's types, and the LLM returns
        # only those. Two pipeline steps can still add types you never declared:
        #   • event extraction → an EVENT entity + PARTICIPATED_IN edges. Turned OFF
        #     here via VectorCypherConfig(store_events=False), so they won't appear.
        #   • co-occurrence → ASSOCIATED_WITH edges between entities sharing a chunk.
        #     NOT configurable in VectorCypher, so these still appear above.
        print(
            "\nNote: event extraction is OFF (store_events=False) — no EVENT / "
            "PARTICIPATED_IN. The remaining ASSOCIATED_WITH edges are co-occurrence "
            "links khora adds during expansion; they are not configurable in VectorCypher."
        )


if __name__ == "__main__":
    asyncio.run(main())
