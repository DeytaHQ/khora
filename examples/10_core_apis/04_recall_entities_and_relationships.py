"""Core API — inspecting the graph khora built at remember time.

On the Skeleton engine, ``recall()`` returns chunks only — the
``entities`` and ``relationships`` lists are empty by construction
because Skeleton doesn't extract a graph. VectorCypher *does* build
one: a single LLM call per remember produces an entity + relationship
graph keyed back to the source chunks.

This demo shows how to read that graph back. Three API surfaces
working together:

  * ``kb.recall(...)``        — chunks + entities + relationships in
                                one shot (graph-aware retrieval)
  * ``kb.list_entities(...)`` — enumerate every extracted node
  * ``kb.find_related_entities(...)`` — walk the edges from an anchor

The full triple: chunks (textual evidence) + entities (extracted
nodes) + relationships (edges, scored alongside the chunks).

WHY THIS DEMO DEFAULTS TO postgres+neo4j
========================================
On the embedded ``sqlite_lance`` backend, VectorCypher's recall
returns ``result.entities == []`` and ``result.relationships == []``
even though the graph is built — entity vectors aren't written to
the LanceDB entities table, so the graph-channel read path can't
surface them. Tracked as
https://github.com/DeytaHQ/khora/issues/857. ``list_entities`` and
``find_related_entities`` work on either backend; only the inline
``recall().entities`` / ``recall().relationships`` lists are broken
on embedded.

To honestly show all three API surfaces working together — which is
the demo's headline — we default to postgres+neo4j. Pass
``--config examples/khora.embedded.yaml`` to run on the embedded
backend; the inline-graph block prints a note pointing at #857 and
falls through to the list/traverse APIs that work everywhere.

Prereq: bring the docker-compose stack up first::

    make dev                  # from the khora-latest root

URLs default to the docker-compose ports documented in workspace
``CLAUDE.md`` (postgres :5434, neo4j :7688). Override via env::

    export KHORA_DATABASE_URL=postgresql://user:pw@host:port/db
    export KHORA_NEO4J_URL=bolt://user:pw@host:port

Engine choice: **vectorcypher** — only engine with a real graph.

Run it
======
uv run python examples/10_core_apis/04_recall_entities_and_relationships.py
python examples/10_core_apis/04_recall_entities_and_relationships.py

# Embedded (recall().entities/.relationships will be empty, see #857):
uv run python examples/10_core_apis/04_recall_entities_and_relationships.py --config examples/khora.embedded.yaml
python examples/10_core_apis/04_recall_entities_and_relationships.py --config examples/khora.embedded.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path

from loguru import logger

from khora import Khora
from khora.config import KhoraConfig

logger.remove()
logger.add("khora.log", level="TRACE", enqueue=True)
for _noisy in ("httpx", "httpcore", "LiteLLM", "openai", "sqlalchemy.engine"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

_PG_URL = os.environ.get("KHORA_DATABASE_URL", "postgresql://khora:khora@localhost:5434/khora")
_NEO4J_URL = os.environ.get("KHORA_NEO4J_URL", "bolt://neo4j:pleaseletmein@localhost:7688")


def _default_postgres_neo4j_config() -> KhoraConfig:
    """Build the pg+neo4j config inline.

    We don't use ``khora.standard.yaml`` because in v0.17 its
    documented env-var override (``KHORA_DATABASE_URL`` /
    ``KHORA_NEO4J_URL``) doesn't actually replace the YAML field
    values — ``from_yaml(...)`` returns the YAML's defaults regardless
    of what's exported (GH #859 family of config-surface bugs).
    Inline ``model_validate({...})`` respects env vars correctly.
    """
    return KhoraConfig.model_validate(
        {
            "storage": {"backend": "postgres", "embedding_dimension": 1536},
            "database_url": _PG_URL,
            "neo4j_url": _NEO4J_URL,
            "llm": {
                "model": "gpt-4o-mini",
                "api_key_env": "OPENAI_API_KEY",
                "embedding_model": "text-embedding-3-small",
                "embedding_dimension": 1536,
            },
        }
    )


def _load_config() -> KhoraConfig:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML override. Default: inline postgres+neo4j config.",
    )
    args = parser.parse_args()
    if args.config is not None:
        return KhoraConfig.from_yaml(args.config)
    return _default_postgres_neo4j_config()


_FACTS = [
    "Marie Curie won the Nobel Prize in Physics in 1903 with her husband Pierre Curie.",
    "Marie Curie discovered the element radium while working at the University of Paris.",
    "Pierre Curie taught physics at the Sorbonne in Paris.",
    "Their daughter Irène Joliot-Curie also won a Nobel Prize, in Chemistry in 1935.",
]


async def main() -> None:
    config = _load_config()
    async with Khora(config, engine="vectorcypher", run_migrations=True) as kb:
        ns_id = (await kb.create_namespace()).namespace_id

        for fact in _FACTS:
            await kb.remember(
                fact,
                namespace=ns_id,
                entity_types=["PERSON", "ORGANIZATION", "CONCEPT", "LOCATION", "EVENT", "DATE"],
                relationship_types=["WORKS_AT", "MARRIED_TO", "DISCOVERED", "WON", "RELATES_TO"],
            )

        recall = await kb.recall("Marie Curie discoveries and family", namespace=ns_id, limit=5)

        # ── [1] chunks — textual evidence ──────────────────────────────
        print(f"[1] chunks ({len(recall.chunks)}):")
        for c in recall.chunks[:3]:
            print(f"     [{c.score:.2f}] {c.content[:80]}…")

        # ── [2] entities — inline on the recall projection ─────────────
        # On postgres+neo4j these populate from the graph channel during
        # recall. On sqlite_lance the list comes back empty (#857) and we
        # fall through to list_entities below.
        print(f"\n[2] recall.entities ({len(recall.entities)}):")
        if recall.entities:
            for e in recall.entities[:8]:
                print(f"     • {e.name:35s}  ({e.entity_type})  score {e.score:.2f}")
        else:
            print("     (empty — likely sqlite_lance + #857; falling through to list_entities)")

        # ── [3] relationships — inline on the recall projection ────────
        print(f"\n[3] recall.relationships ({len(recall.relationships)}):")
        if recall.relationships:
            name_by_id = {e.id: e.name for e in recall.entities}
            for r in recall.relationships[:8]:
                src = name_by_id.get(r.source_entity_id, str(r.source_entity_id)[:8])
                dst = name_by_id.get(r.target_entity_id, str(r.target_entity_id)[:8])
                print(f"     • {src} —[{r.relationship_type}]→ {dst}  (score {r.score:.2f})")
        else:
            print("     (empty — likely sqlite_lance + #857; falling through to find_related_entities)")

        # ── [4] list_entities — works on either backend ────────────────
        # The dedicated "show me everything in the graph" API. A real
        # app would filter by entity_type or query through
        # search_entities. Works identically on sqlite_lance and
        # postgres+neo4j, which is what makes it the fallback when the
        # inline recall projection is empty.
        all_entities = await kb.list_entities(namespace=ns_id, limit=20)
        print(f"\n[4] list_entities (full graph, {len(all_entities)} nodes):")
        for e in all_entities[:8]:
            print(f"     • {e.name:35s}  ({e.entity_type})")

        # ── [5] find_related_entities — also works on either backend ───
        marie = next(
            (e for e in all_entities if e.entity_type == "PERSON" and "marie" in e.name.lower()),
            None,
        )
        if marie is None:
            print("\n(Marie Curie not in extracted PERSON entities — extraction was light)")
            return
        neighbours = await kb.find_related_entities(marie.id, namespace=ns_id, max_depth=1, limit=10)
        print(f"\n[5] find_related_entities({marie.name!r}, depth=1) — {len(neighbours)} edge(s):")
        for entity, score in neighbours:
            print(f"     {marie.name} —→ {entity.name:25s}  ({entity.entity_type})  score {score:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
