"""Flagship 40 — iterative ontology refinement on a literary excerpt.

This is the sketch version: short text, two ontologies, no
killer-query phase, no point-in-time queries. Enough to show the
core loop:

    open extraction → see what you got → refine the ontology → re-run

The loop is what no competitor in the memory-library category has
shipped as a packaged demo (mid-2026). Mem0/Zep/Letta gesture at it
in docs; Cognee has separate ``ontology_quickstart`` and
``improve_quickstart`` but doesn't combine them; Neo4j's
LLM-Graph-Builder ships it as a UI toggle, not a teachable narrative.

Why this corpus? A 250-word Anna Karenina excerpt is dense with the
extraction-failure modes that matter on real text:

  * **Aliases** — Stepan Arkadyevitch / Stiva / Oblonsky are the same
    person. Naive extraction makes them separate nodes.
  * **Patronymics** — Russian names carry social context (formal vs
    intimate). The same individual appears under many surface forms.
  * **Untyped relationships** — "carrying on an intrigue," "old
    friend," "housekeeper" — vector-only RAG dumps all of this into
    chunks and asks the LLM to figure it out at query time. A typed
    graph stores them as edges with semantics.

Engine: **vectorcypher** — only engine that runs extraction.

Run it
======
uv run python examples/40_flagship/anna_karenina_iterative_ontology.py
python examples/40_flagship/anna_karenina_iterative_ontology.py

Expected runtime: ~30s on gpt-4o-mini (two extraction passes over a
single short chunk).
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

# Constance Garnett's translation of Anna Karenina, opening of Part I.
# Public domain. Roughly 250 words — one chunk under the default
# semantic chunker, which keeps the two extraction passes comparable
# (same text, different ontology, no chunk-boundary noise).
_EXCERPT = """\
All happy families are alike; each unhappy family is unhappy in its own way.

Everything was in confusion in the Oblonskys' house. The wife had discovered \
that the husband was carrying on an intrigue with a French girl, who had been \
a governess in their family, and she had announced to her husband that she \
could not go on living in the same house with him. This position of affairs \
had now lasted three days, and not only the husband and wife themselves, but \
all the members of their family and household, were painfully conscious of it.

Stepan Arkadyevitch Oblonsky — Stiva, as he was called in the fashionable \
world — woke up at his usual hour, that is, at eight o'clock in the morning, \
not in his wife's bedroom, but on the leather-covered sofa in his study. He \
turned over his stout, well-cared-for person on the springy sofa, as though \
he would sink into a long sleep again; he vigorously embraced the pillow on \
the other side and buried his face in it; but all at once he jumped up, sat \
up on the sofa, and opened his eyes.

His wife, Darya Alexandrovna — Dolly, his old friend Konstantin Levin would \
call her — was still in tears in the bedroom. The housekeeper Matrona \
Filimonovna had taken charge of the children. Down the corridor, the cook \
Matvey waited for his master's morning instructions.
"""

# ── Two ontologies to compare ──────────────────────────────────────────
# Open: khora's canonical generic taxonomy. The default a quickstart
# reader would reach for, and what most "RAG demo" code shows.
OPEN_ENTITY_TYPES = ["PERSON", "ORGANIZATION", "CONCEPT", "LOCATION", "EVENT", "PRODUCT", "TECHNOLOGY"]
OPEN_RELATIONSHIP_TYPES = ["RELATES_TO", "PART_OF", "MENTIONS"]

# Refined: domain-aware. CHARACTER replaces the blunt PERSON; the
# servant / family / affair edges encode social structure the open
# ontology would have lumped under RELATES_TO.
REFINED_ENTITY_TYPES = ["CHARACTER", "FAMILY", "SERVANT", "AFFAIR", "ESTATE", "ROLE"]
REFINED_RELATIONSHIP_TYPES = [
    "MARRIED_TO",
    "RELATED_TO",
    "EMPLOYS",
    "HAS_AFFAIR_WITH",
    "FRIENDS_WITH",
    "WORKS_AT",
]


async def _extract_with(kb, label: str, entity_types, relationship_types) -> None:
    """Run a single extraction pass and print the entity report."""
    ns_id = (await kb.create_namespace()).namespace_id
    await kb.remember(
        _EXCERPT,
        namespace=ns_id,
        title=f"anna-karenina-opening-{label}",
        entity_types=entity_types,
        relationship_types=relationship_types,
    )
    entities = await kb.list_entities(namespace=ns_id, limit=100)

    print(f"\n{'=' * 72}")
    print(f"  ONTOLOGY: {label}   ({len(entities)} entities extracted)")
    print(f"{'=' * 72}")

    histogram = Counter(e.entity_type for e in entities)
    print("\n  entity-type histogram:")
    for etype, count in sorted(histogram.items(), key=lambda kv: -kv[1]):
        print(f"    {etype:14s}  {count}")

    print("\n  all extracted entities:")
    # Group by entity_type, print in stable order
    by_type: dict[str, list[str]] = {}
    for e in entities:
        by_type.setdefault(e.entity_type, []).append(e.name)
    for etype in sorted(by_type):
        for name in sorted(by_type[etype]):
            print(f"    • {name:38s}  → {etype}")


async def main() -> None:
    config = KhoraConfig.from_yaml(_CONFIG)
    async with Khora(config, engine="vectorcypher", run_migrations=True) as kb:
        # Phase 1 — open extraction. The expected failure mode: aliases
        # like "Stepan Arkadyevitch", "Stiva", "Oblonsky" appear as
        # separate PERSON nodes; relationships collapse into RELATES_TO.
        await _extract_with(kb, "open", OPEN_ENTITY_TYPES, OPEN_RELATIONSHIP_TYPES)

        # Phase 2 — refined ontology. Same text, narrower types. Watch
        # whether CHARACTER (not PERSON), SERVANT (housekeeper, cook),
        # and AFFAIR (the French-girl intrigue) bubble up as their own
        # categories — and whether MARRIED_TO / HAS_AFFAIR_WITH /
        # EMPLOYS land as typed edges instead of catch-all RELATES_TO.
        await _extract_with(kb, "refined", REFINED_ENTITY_TYPES, REFINED_RELATIONSHIP_TYPES)

        print("\n" + "=" * 72)
        print("  Next steps (left for the full Tier 40 example to build out):")
        print("=" * 72)
        print("""
  • Inspect the relationships — list edges by type, compare RELATES_TO
    blob (open) vs the typed graph (refined).
  • Pick an entity whose name appears under multiple aliases (Stepan
    Arkadyevitch / Stiva / Oblonsky) and check whether they're the
    same node or separate. Refined ontologies sometimes consolidate
    aliases the open pass split.
  • Add a recall query that only the refined graph can answer well —
    e.g., "who employs Matrona?" — and contrast with semantic recall.
""")


if __name__ == "__main__":
    asyncio.run(main())
