"""Core API — exploring the entity / relationship graph.

VectorCypher's ``remember()`` extracts entities + relationships at
write time and stores them in the graph backend. This demo walks the
**full toolkit** for reading that graph back, so you can pick the
right call for the task at hand:

  1. ``list_entities(namespace)`` — enumerate everything
  2. ``list_entities(entity_type=...)`` — filter to one type
  3. ``search_entities(query, namespace)`` — semantic lookup by name
  4. ``get_entity(entity_id, namespace)`` — fetch one node by id
  5. ``find_related_entities(entity_id, max_depth)`` — walk the edges
  6. ``recall(query, mode=SearchMode.GRAPH)`` — graph-channel retrieval
     (returns chunks + entities + relationships inline)
  7. ASCII tree view, built from (1) + (5)

Engine: **vectorcypher** — only engine that builds a graph.

WHY THIS DEMO DEFAULTS TO postgres+neo4j
========================================
This is the only tier-10 example that doesn't default to the embedded
sqlite_lance backend. Reason: https://github.com/DeytaHQ/khora/issues/857
prevents two of the seven steps from returning anything on the
embedded path. Entity embeddings aren't written to the LanceDB
entities table on sqlite_lance, so:

* ``search_entities(...)`` returns ``[]`` regardless of query
* ``recall(mode=SearchMode.GRAPH).entities`` / ``.relationships`` are empty

The graph itself IS built — ``list_entities`` / ``get_entity`` /
``find_related_entities`` read from the graph store directly and work
on both backends. But for a demo whose goal is "exercise the full
toolkit," running on a backend where 2 of 7 steps degrade to empty
output is dishonest. So we default to postgres+neo4j.

Prereq: bring the docker-compose stack up first::

    make dev                  # from the khora-latest root

The demo builds its postgres+neo4j config inline (matching the
docker-compose default ports — 5434 for postgres, 7688 for neo4j —
documented in workspace ``CLAUDE.md``). Override via env vars if your
stack uses different URLs::

    export KHORA_DATABASE_URL=postgresql://user:pw@host:port/db
    export KHORA_NEO4J_URL=bolt://user:pw@host:port

You CAN still pass ``--config examples/khora.embedded.yaml`` to run on
the embedded backend; steps 3 and 6 will print the documented
limitation note rather than the expected data.

Run it
======
uv run python examples/10_core_apis/05_find_related_entities.py
python examples/10_core_apis/05_find_related_entities.py

# Embedded (steps 3 + 6 will be empty, see #857):
uv run python examples/10_core_apis/05_find_related_entities.py --config examples/khora.embedded.yaml
python examples/10_core_apis/05_find_related_entities.py --config examples/khora.embedded.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from collections import defaultdict
from pathlib import Path
from uuid import UUID

from loguru import logger

from khora import Khora
from khora.config import KhoraConfig
from khora.query.engine import SearchMode

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
    of what's exported. Inline ``model_validate({...})`` respects env
    vars correctly.
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


_FACTS = [
    "Marie Curie won the Nobel Prize in Physics in 1903 with her husband Pierre Curie.",
    "Marie Curie discovered the element radium while working at the University of Paris.",
    "Marie Curie won a second Nobel Prize, in Chemistry, in 1911.",
    "Marie Curie founded the Curie Institute in Paris in 1909.",
    "Pierre Curie taught physics at the Sorbonne in Paris.",
    "Their daughter Irène Joliot-Curie won the Nobel Prize in Chemistry in 1935.",
    "Irène Joliot-Curie married Frédéric Joliot, a fellow physicist.",
    "Frédéric Joliot served as France's first High Commissioner for Atomic Energy.",
]


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


def _render_tree(
    anchor_name: str,
    depth1: list[tuple[str, str, UUID]],
    depth1_children: dict[UUID, list[tuple[str, str]]],
) -> None:
    """Print an ASCII tree two levels deep, anchored on one entity.

    ``depth1`` is a list of (name, type, id) for direct neighbours.
    ``depth1_children[id]`` is the list of (name, type) for each
    neighbour's own neighbours (depth-2 nodes), already deduped against
    the anchor and the depth-1 set.
    """
    print(f"\n{anchor_name}")
    for i, (name, etype, child_id) in enumerate(depth1):
        last = i == len(depth1) - 1
        prefix = "└─ " if last else "├─ "
        print(f"{prefix}{name}  ({etype})")
        children = depth1_children.get(child_id, [])
        for j, (cn, ct) in enumerate(children):
            cprefix = "   " if last else "│  "
            cmark = "└─ " if j == len(children) - 1 else "├─ "
            print(f"{cprefix}{cmark}{cn}  ({ct})")


async def main() -> None:
    config = _load_config()
    async with Khora(config, engine="vectorcypher", run_migrations=True) as kb:
        ns_id = (await kb.create_namespace()).namespace_id

        for fact in _FACTS:
            await kb.remember(
                fact,
                namespace=ns_id,
                entity_types=["PERSON", "ORGANIZATION", "CONCEPT", "LOCATION", "EVENT"],
                relationship_types=[
                    "WORKS_AT",
                    "MARRIED_TO",
                    "DISCOVERED",
                    "WON",
                    "TAUGHT_AT",
                    "FOUNDED",
                    "RELATES_TO",
                ],
            )

        # ── 1. list everything ─────────────────────────────────────────
        all_entities = await kb.list_entities(namespace=ns_id, limit=100)
        by_type: dict[str, list[str]] = defaultdict(list)
        for e in all_entities:
            by_type[e.entity_type].append(e.name)
        print(f"[1] list_entities()  — {len(all_entities)} total")
        for etype in sorted(by_type):
            names = ", ".join(sorted(by_type[etype])[:5])
            extra = f" (+{len(by_type[etype]) - 5} more)" if len(by_type[etype]) > 5 else ""
            print(f"     {etype:14s} {names}{extra}")

        # ── 2. filter by entity_type ───────────────────────────────────
        persons = await kb.list_entities(namespace=ns_id, entity_type="PERSON", limit=50)
        print(f"\n[2] list_entities(entity_type='PERSON')  — {len(persons)} PERSON nodes")
        for p in persons:
            print(f"     • {p.name}")

        # ── 3. semantic search by name/description ─────────────────────
        # search_entities embeds the query and runs cosine similarity
        # against entity vectors (the embedded entity name + short
        # description, NOT the source chunks). On postgres+neo4j this
        # is the right API when you know roughly what the entity is
        # called but not its exact extracted form.
        #
        # Caveat: same underlying bug as #857. On sqlite_lance, entity
        # embeddings aren't written (the LanceDB entities table stays
        # empty), so search_entities also returns no results on
        # embedded. list_entities + get_entity work because they read
        # from the graph backend, not the vector backend.
        hits = await kb.search_entities("physicist who discovered radium", namespace=ns_id, limit=3)
        print(f"\n[3] search_entities('physicist who discovered radium')  — top {len(hits)}")
        if not hits:
            print("     (no results — likely sqlite_lance + #857; pg+neo4j returns the expected matches)")
        for h in hits:
            print(f"     • {h.name}  ({h.entity_type})")

        # ── 4. fetch one by id ─────────────────────────────────────────
        marie = next((e for e in persons if "marie" in e.name.lower() and "curie" in e.name.lower()), None)
        if marie is None:
            print("\n  (Marie Curie wasn't extracted as a PERSON — aborting graph walk)")
            return
        fetched = await kb.get_entity(marie.id, namespace=ns_id)
        print(f"\n[4] get_entity({str(marie.id)[:8]}…)  → {fetched.name}  ({fetched.entity_type})")
        if fetched.description:
            print(f"     description: {fetched.description[:120]}{'…' if len(fetched.description) > 120 else ''}")

        # ── 5. walk the edges outward ──────────────────────────────────
        depth1 = await kb.find_related_entities(marie.id, namespace=ns_id, max_depth=1, limit=20)
        depth2 = await kb.find_related_entities(marie.id, namespace=ns_id, max_depth=2, limit=40)
        print(f"\n[5] find_related_entities(marie, max_depth=1)  — {len(depth1)} direct neighbours")
        print(f"    find_related_entities(marie, max_depth=2)  — {len(depth2)} reachable within 2 hops")

        # ── 6. graph-channel recall ────────────────────────────────────
        # On vectorcypher + postgres+neo4j this returns chunks + entities +
        # relationships inline. On vectorcypher + sqlite_lance the
        # entities/relationships slots come back empty (issue #857).
        gr = await kb.recall("Marie Curie discoveries and family", namespace=ns_id, mode=SearchMode.GRAPH, limit=5)
        print(
            f"\n[6] recall(mode=GRAPH)  — chunks={len(gr.chunks)}, "
            f"entities={len(gr.entities)}, relationships={len(gr.relationships)}"
        )
        if not gr.entities and not gr.relationships:
            print(
                "     (empty entities/relationships — likely sqlite_lance + #857;\n      pg+neo4j returns them inline)"
            )
        else:
            for rel in gr.relationships[:5]:
                # Map id → name from gr.entities for legibility.
                name_by_id = {e.id: e.name for e in gr.entities}
                src = name_by_id.get(rel.source_entity_id, str(rel.source_entity_id)[:8])
                dst = name_by_id.get(rel.target_entity_id, str(rel.target_entity_id)[:8])
                print(f"     • {src} —[{rel.relationship_type}]→ {dst}")

        # ── 7. small ASCII tree from Marie Curie ───────────────────────
        # Built from (1) + (5). Edge type labels would come from (6); on
        # sqlite_lance we render structure only.
        anchor_id = marie.id
        seen = {anchor_id} | {e.id for e, _ in depth1}
        d1_list = [(e.name, e.entity_type, e.id) for e, _ in depth1]
        d1_children: dict[UUID, list[tuple[str, str]]] = {}
        for entity, _score in depth1:
            kids = await kb.find_related_entities(entity.id, namespace=ns_id, max_depth=1, limit=6)
            d1_children[entity.id] = [(k.name, k.entity_type) for k, _ in kids if k.id not in seen][
                :4
            ]  # cap each branch's fan-out for readability
        print("\n[7] graph around Marie Curie (depth=2, structure only):")
        _render_tree(marie.name, d1_list, d1_children)


if __name__ == "__main__":
    asyncio.run(main())
