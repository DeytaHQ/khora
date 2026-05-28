"""Demo 04 — Support tickets → knowledge graph (entity-mediated multi-hop).

Ingest ~100 customer-support tickets and then ask two different shapes
of question: one that vector search alone can answer, and one that
needs the graph. Three pedagogical points:

  1. **Vector search alone is enough for "what tickets are about X"**
     — direct similarity on the ticket text returns the obvious hits.
  2. **Graph traversal is the right tool for "give me context around
     customer Acme"** — the answer requires walking through Acme's
     tickets, the products Acme uses, similar tickets from other
     customers on the same products, and the related error
     categories. Undirected 2-hop neighborhood expansion is the
     correct shape for this — no "directional dependency" overselling.
  3. **Off-topic queries** demonstrate the no-signal case.

WHY VECTORCYPHER
================
Hybrid retrieval + entity extraction + graph traversal — the three
features customer-support workloads benefit from most. Customer names
(ORGANIZATION), product SKUs (PRODUCT), error categories (CONCEPT),
and support agents (PERSON) all surface as named entities at ingest,
so the multi-hop walk from any seed entity has real structure to
follow. The 100-ticket corpus has enough overlap — most customers use
2-3 products, most products span multiple customers — that the graph
clusters cleanly without being trivially small.


DUAL-BACKEND SUPPORT
====================
VectorCypher is **production-ready** on PostgreSQL + pgvector + Neo4j
and **Experimental** on the embedded SQLite + LanceDB stack. The
100-ticket corpus is well within both backends' scale limits. The
multi-hop traversal API does not currently expose direction or
edge-type filters — see ``find_related_entities`` for the full
contract.

**WARNING**
The whole corpus is 100 documents, so it may tike some time to load.
You can limit this by picking smaller subset and specifying it via --data

Run it
======
uv run python examples/30_workloads/03_support_ticket_graph.py
python examples/30_workloads/03_support_ticket_graph.py
# or run via PG+Neo4J stack
uv run python examples/30_workloads/03_support_ticket_graph.py --config examples/khora.standard.yaml
python examples/30_workloads/03_support_ticket_graph.py --config examples/khora.standard.yaml
# if you want to run your own dataset
uv run python examples/30_workloads/03_support_ticket_graph.py --data path/to/your_tickets.jsonl
python examples/30_workloads/03_support_ticket_graph.py --data path/to/your_tickets.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from loguru import logger

from khora import Khora
from khora.config import KhoraConfig

# ── Logging setup ───────────────────────────────────────────────────────
# Khora uses loguru (not stdlib `logging`) for its own output. The default
# loguru sink writes to stderr, which floods the terminal with extraction
# and recall traces. Route the noise to a file and keep the terminal
# showing only warnings/errors plus this script's `print()` output. Drop
# these lines if you'd rather see everything; tighten the file-level
# threshold (e.g. `level="INFO"`) if `khora.log` itself gets too big.
logger.remove()  # drop default stderr sink
logger.add("khora.log", level="TRACE", enqueue=True)  # every level → file (TRACE is the floor)
# logger.add(sys.stderr, level="WARNING")                      # only warn+ → terminal

# Tame third-party stdlib loggers that bypass loguru.
for _noisy in ("httpx", "httpcore", "LiteLLM", "openai", "sqlalchemy.engine"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)


_DEFAULT_CONFIG = Path(__file__).parent.parent / "khora.embedded.yaml"
_DEFAULT_DATA = Path(__file__).parent.parent / "data" / "support_tickets.jsonl"

# Taxonomy for support-ticket extraction. Picked to match what actually
# matters for the multi-hop story:
#
#   ORGANIZATION → the customer (Acme Logistics, Globex Industries, …)
#   PRODUCT      → the SKU mentioned in the ticket (DataSync Pro, …)
#   CONCEPT      → the error category (login failures, sync timeouts, …)
#   PERSON       → the assigned support agent (Lisa Park, Tomas Garcia, …)
#
# Stdlib-style noise (LOCATION, EVENT, TECHNOLOGY) is deliberately
# omitted: the LLM would otherwise extract dates and HS codes as
# entities and clutter the graph without paying for it.
_ENTITY_TYPES = ["ORGANIZATION", "PRODUCT", "CONCEPT", "PERSON"]
_RELATIONSHIP_TYPES = ["AFFECTS", "ASSIGNED_TO", "RELATES_TO", "MENTIONS"]


def _load_args() -> tuple[KhoraConfig, Path]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument(
        "--data",
        type=Path,
        default=_DEFAULT_DATA,
        help=f"JSONL corpus path (default: {_DEFAULT_DATA.name}).",
    )
    args = parser.parse_args()
    return KhoraConfig.from_yaml(args.config), args.data


def _load_tickets(path: Path) -> list[dict]:
    """Load support tickets from a JSONL file.

    Each line is a JSON object with ``ticket_id``, ``title``, ``source``,
    and ``content`` fields. The shipped corpus
    (``examples/data/support_tickets.jsonl``) has 100 tickets across
    10 named customers, 8 products, and 12 error categories.
    """
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _find_entity(entities, *needles: str):
    """Find the first entity whose name matches one of ``needles`` (case-insensitive).

    LLM extraction is not deterministic — "Acme Logistics" might surface
    as "Acme Logistics", "Acme", or "Acme Logistics (logistics)" depending
    on the model. Walk a list of plausible names rather than hard-failing.
    """
    lowered = [n.lower() for n in needles]
    for e in entities:
        if e.name.lower() in lowered:
            return e
    return None


def _print_chunk(doc_by_id: dict, chunk) -> None:
    """One-line preview of a recalled chunk plus its source-document title."""
    doc = doc_by_id.get(chunk.document_id)
    title = doc.title if doc and doc.title else "?"
    preview = " ".join(chunk.content.split())[:120]
    print(f"  [{chunk.score:.3f}] {title}")
    print(f"          {preview}{'…' if len(chunk.content) > 120 else ''}")


async def main() -> None:
    config, data_path = _load_args()
    tickets = _load_tickets(data_path)
    if not tickets:
        print(f"no tickets found in {data_path}")
        return

    async with Khora(config, engine="vectorcypher", run_migrations=True) as kb:
        namespace = await kb.create_namespace()
        ns_id = namespace.namespace_id

        # ── Bulk ingest ──────────────────────────────────────────────
        # remember_batch shares the embedder LRU cache across docs and
        # dedupes entities cross-document — both essential when the
        # same customer (Acme) appears in 10 different ticket bodies.
        print(f"ingesting {len(tickets)} tickets from {data_path.name}…")
        docs = [{"title": t["title"], "source": t["source"], "content": t["content"]} for t in tickets]
        batch = await kb.remember_batch(
            docs,
            namespace=ns_id,
            entity_types=_ENTITY_TYPES,
            relationship_types=_RELATIONSHIP_TYPES,
        )
        print(
            f"  {batch.processed}/{batch.total} processed, "
            f"{batch.chunks} chunks, "
            f"{batch.entities} entities, "
            f"{batch.relationships} relationships"
        )

        # ── Entity sanity check ─────────────────────────────────────
        # Show the top customer (ORGANIZATION) entities by mention
        # count. If a known customer like "Acme Logistics" doesn't
        # appear here, the LLM picked a different label and the rest
        # of the demo's seed lookup will fall back to entity #0.
        customers = await kb.list_entities(namespace=ns_id, entity_type="ORGANIZATION", limit=15)
        print(f"\nORGANIZATION entities surfaced ({len(customers)}):")
        for c in customers[:10]:
            print(f"  - {c.name}  (mentions={c.mention_count})")

        products = await kb.list_entities(namespace=ns_id, entity_type="PRODUCT", limit=15)
        print(f"\nPRODUCT entities surfaced ({len(products)}):")
        for p in products[:8]:
            print(f"  - {p.name}  (mentions={p.mention_count})")

        # ── Q1: vector search wins ──────────────────────────────────
        # "tickets about login failures" — the literal phrase appears
        # in many ticket bodies, semantic similarity ranks them
        # cleanly. No graph traversal needed.
        print("\nQ1: 'tickets about login failures'  (vector search shape)")
        result = await kb.recall("tickets about login failures", namespace=ns_id, limit=5)
        docs_by_id = {d.id: d for d in result.documents}
        for chunk in result.chunks[:5]:
            _print_chunk(docs_by_id, chunk)

        # ── Q2: graph traversal wins ────────────────────────────────
        # "Give me context around Acme Logistics" — the answer is a
        # NEIGHBORHOOD, not a ranked list of text matches. Vector
        # search returns only tickets that literally mention Acme;
        # the graph walks from Acme outward to:
        #   - the products Acme uses
        #   - the support agents who handle Acme's tickets
        #   - the error categories that affect Acme
        #   - other customers using the same products  (2-hop)
        #   - error categories on those shared products (2-hop)
        #
        # Undirected, all-edge-types — which is honest because the
        # question is "context", not "directional dependency".
        print("\nQ2: 'context around Acme Logistics'  (graph traversal shape)")
        acme = _find_entity(customers, "Acme Logistics", "Acme")
        if acme is None and customers:
            print(
                f"  (no canonical 'Acme Logistics' ORGANIZATION entity surfaced — "
                f"the LLM may have picked a different label. Using "
                f"{customers[0].name!r} as the seed.)"
            )
            acme = customers[0]
        if acme is None:
            print("  no ORGANIZATION entities at all — extraction returned empty.")
            return

        print(f"  seed entity: {acme.name} ({acme.entity_type})")
        related = await kb.find_related_entities(
            acme.id,
            namespace=ns_id,
            max_depth=2,
            limit=25,
        )
        # The neighborhood spans many entity types. Group by type so
        # the structure is visible: customers near Acme via shared
        # products are ORGANIZATION; products Acme uses are PRODUCT;
        # error categories are CONCEPT; assigned agents are PERSON.
        by_type: dict[str, list[tuple]] = {}
        for ent, score in related:
            by_type.setdefault(ent.entity_type, []).append((ent.name, score))
        for etype in sorted(by_type):
            print(f"  {etype}:")
            for name, score in by_type[etype][:8]:
                print(f"    [{score:.3f}] {name}")

        # ── Q3: off-topic ──────────────────────────────────────────
        # The corpus has nothing about baking. Top chunk will be some
        # marginal lexical match. Useful as a sanity check: vector
        # search always returns *something*; the caller has to look
        # at the score and recognise that the top match is weak.
        print("\nQ3: 'How do I bake a sourdough loaf?'  (off-topic)")
        offtopic = await kb.recall("How do I bake a sourdough loaf?", namespace=ns_id, limit=2)
        docs_by_id = {d.id: d for d in offtopic.documents}
        for chunk in offtopic.chunks[:2]:
            _print_chunk(docs_by_id, chunk)


if __name__ == "__main__":
    asyncio.run(main())
