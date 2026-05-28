"""Core API — constraining extraction with entity_types / relationship_types.

Every ``remember()`` call requires ``entity_types`` and
``relationship_types`` — there is no default. khora won't infer a
taxonomy from the content. That's a deliberate design choice: the
schema is part of the API contract, so two ``remember`` calls from
different code paths can't quietly disagree on what counts as a
PERSON vs an EMPLOYEE.

This demo runs the same paragraph through two ontologies:

  * **Generic** — broad types (PERSON, ORGANIZATION, CONCEPT, …) and
    catch-all RELATES_TO edges. You get something, but the labels are
    blunt and you have to disambiguate downstream.
  * **Domain-specific** — typed for a recruiting workflow (CANDIDATE,
    EMPLOYER, SKILL, ROLE) and edges that match how a recruiter
    thinks (HAS_SKILL, WORKED_AT, APPLIED_TO).

Same model, same text — but the domain-specific run produces an
entity-type histogram you can actually query against. This is the
groundwork for an iterative-ontology workflow (see Tier 40 in the
example index — open extraction → inspect → refine → re-extract).

Engine choice: **vectorcypher** — extraction lives here.

Run it
======
uv run python examples/10_core_apis/03_ontology_config.py
python examples/10_core_apis/03_ontology_config.py
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from pathlib import Path

from loguru import logger

from khora import Khora
from khora.config import KhoraConfig

logger.remove()
logger.add("khora.log", level="TRACE", enqueue=True)
for _noisy in ("httpx", "httpcore", "LiteLLM", "openai", "sqlalchemy.engine"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

_CONFIG = Path(__file__).parent.parent / "khora.embedded.yaml"

_PARAGRAPH = (
    "Priya Patel applied to a backend engineering role at Acme Robotics. "
    "She holds five years of Python experience, has shipped distributed "
    "systems at Globex, and is fluent in Kubernetes. The hiring manager, "
    "Sam Chen, has scheduled her for a system-design loop next week."
)


async def extract_with(kb, label: str, entity_types: list[str], relationship_types: list[str]):
    ns_id = (await kb.create_namespace()).namespace_id
    await kb.remember(
        _PARAGRAPH,
        namespace=ns_id,
        title=f"recruiting-paragraph-{label}",
        entity_types=entity_types,
        relationship_types=relationship_types,
    )
    entities = await kb.list_entities(namespace=ns_id, limit=50)

    histogram = Counter(e.entity_type for e in entities)
    print(f"\n[{label}] entity-type histogram:")
    for etype, count in sorted(histogram.items(), key=lambda kv: -kv[1]):
        print(f"  {etype:15s}  {count}")
    print(f"  total: {len(entities)} entities")
    for e in entities[:8]:
        print(f"    • {e.name:30s}  → {e.entity_type}")


async def main() -> None:
    config = KhoraConfig.from_yaml(_CONFIG)

    async with Khora(config, engine="vectorcypher", run_migrations=True) as kb:
        # ── Run 1: generic ontology ────────────────────────────────────
        await extract_with(
            kb,
            "generic",
            entity_types=["PERSON", "ORGANIZATION", "CONCEPT", "LOCATION", "EVENT", "PRODUCT", "TECHNOLOGY"],
            relationship_types=["RELATES_TO", "PART_OF", "MENTIONS"],
        )

        # ── Run 2: domain-specific recruiting ontology ─────────────────
        await extract_with(
            kb,
            "recruiting",
            entity_types=["CANDIDATE", "EMPLOYER", "ROLE", "SKILL", "HIRING_MANAGER", "INTERVIEW_LOOP"],
            relationship_types=["APPLIED_TO", "WORKED_AT", "HAS_SKILL", "SCHEDULED_FOR", "MANAGED_BY"],
        )


if __name__ == "__main__":
    asyncio.run(main())
