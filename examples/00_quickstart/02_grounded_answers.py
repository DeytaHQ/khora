"""Quickstart 02 — grounded answers and abstention.

The most common failure mode in production RAG isn't bad answers — it's
refusing answers when the corpus has nothing to say. This demo shows
how khora's VectorCypher engine surfaces abstention signals so your
application can refuse to answer when it should, without having to
roll your own confidence threshold or trust the LLM to know what it
doesn't know.

The displayed ``chunk.score`` is a normalized rank-within-result and is
**not** the right signal for "is this corpus actually relevant?". The
right signal lives in ``result.engine_info``:

* ``max_raw_vector_score`` — the **pre-rerank, pre-normalize raw cosine**
  of the top semantic-channel hit. This is what your abstention threshold
  should compare against. Below ~0.3 means the corpus has nothing
  meaningfully on-topic; above ~0.5 is a confident match.
* ``abstention_signals`` — a precomputed dict with ``chunks_empty``,
  ``top_score_low``, ``entities_empty``, ``chunks_below_min``, plus a
  weighted ``combined_score`` and a default ``should_abstain`` boolean.
  Use these directly if the default weighting fits your workload; roll
  your own threshold off ``max_raw_vector_score`` if it doesn't.

There is no need to use LLM for this - you can, but it consumes tokens and time.

Engine choice: **vectorcypher** — khora's default engine, and one of the
two (with Chronicle) that emit abstention signals. It runs here on the
embedded ``sqlite_lance`` backend.

Run it
======
uv run python examples/00_quickstart/02_grounded_answers.py
python examples/00_quickstart/02_grounded_answers.py

"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from loguru import logger

from khora import Khora
from khora.config import KhoraConfig

logger.remove()
logger.add("khora.log", level="TRACE", enqueue=True)
for _noisy in ("httpx", "httpcore", "LiteLLM", "openai", "sqlalchemy.engine"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

_CONFIG = Path(__file__).parent.parent / "khora.embedded.yaml"

_POLICIES = [
    ("PTO policy", "Employees accrue 1.5 days of PTO per month, up to 25 days/year."),
    ("Expenses", "Meals over $50 require manager approval before submission."),
    ("Remote work", "Remote work is allowed up to three days per week by default."),
    ("Equipment", "New hires get a laptop refresh every 36 months."),
]


_ABSTAIN_BELOW = 0.45  # tune to your corpus + tolerance for false abstentions


async def answer(kb, namespace, question: str) -> None:
    print(f"\nQ: {question}")
    result = await kb.recall(question, namespace=namespace, limit=3)

    # The raw cosine of the strongest semantic hit.
    # The ``chunk.score`` is a rank of results normalized to [0,1]
    raw_top = result.engine_info.get("max_raw_vector_score", 0.0)

    if not result.chunks or raw_top < _ABSTAIN_BELOW:
        print(f"  → I don't know. (raw_top={raw_top:.2f} < {_ABSTAIN_BELOW})")
        return

    top = result.chunks[0]
    print(f"  [raw_top {raw_top:.2f}] {top.content}")
    # In a real app, hand top.content to an LLM with a "answer only from
    # this context" system prompt. khora supplies grounded context; the
    # LLM call is yours.


async def main() -> None:
    config = KhoraConfig.from_yaml(_CONFIG)
    async with Khora(config, engine="vectorcypher", run_migrations=True) as kb:
        namespace = await kb.create_namespace()
        ns_id = namespace.namespace_id

        for title, content in _POLICIES:
            await kb.remember(
                content,
                namespace=ns_id,
                title=title,
                entity_types=["CONCEPT"],
                relationship_types=["RELATES_TO"],
            )

        # In-corpus: grounded answer with high raw_top
        await answer(kb, ns_id, "How many vacation days do I get per year?")
        # Adjacent phrasing — semantic recall still hits.
        await answer(kb, ns_id, "Can I work from home?")
        # Out-of-corpus — abstention signals should fire (low raw_top).
        await answer(kb, ns_id, "What's the parental leave policy?")
        # Way off-topic — strongest abstention signal.
        await answer(kb, ns_id, "Who won the World Cup in 2022?")


if __name__ == "__main__":
    asyncio.run(main())
