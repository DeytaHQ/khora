"""Workload 04 — Memory as a tool for an LLM agent.

The "memory as a tool" pattern: an LLM-driven agent exposes two callable
tools, ``remember(text)`` and ``recall(query)``, both scoped to a single
user's namespace. The agent (any framework — LangGraph, OpenAI Agents
SDK, CrewAI, Pydantic AI, plain function-calling) decides when to call
which.

This demo simulates a small multi-turn conversation by orchestrating the
tool calls directly. We don't pull in LangGraph or another agent
framework — the goal is to show the **integration contract** (what each
tool returns, when the agent should defer to abstention), not to pick a
favourite framework.

WHY CHRONICLE
=============
Two properties matter for an agent loop:

  1. Abstention signals on a cold namespace. A confident "I do not have
     that" beats a low-similarity guess. Plumb ``should_abstain`` into
     the tool return value and the agent's prompt can branch on it.
     ``should_abstain`` is the canonical gate — read it instead of
     thresholding on ``chunk.score``, which is post-fusion min-max
     normalized within the result set and always 1.00 on the top hit.
  2. No graph backend. The chat-sidecar pattern doesn't need Neo4j;
     skipping it removes operational overhead the use case doesn't
     justify.

Two operational reminders Chronicle won't do for you:

  * **Compression doesn't run automatically.** Chronicle ships fact /
    event compression but it's driven by the dream phase, not by the
    ingest path. See ``05_dream_phase_consolidation.py`` — run that on
    a schedule, not per-turn, to keep a long-lived chat history from
    accumulating near-duplicate facts.
  * **Updates take an explicit forget + remember.** When a user changes
    their mind ("actually my talk is in NYC, not Pittsburgh"), the old
    statement stays in the index unless you delete it. See
    ``01_per_user_preferences.py`` for the canonical pattern.

DUAL-BACKEND SUPPORT
====================
Chronicle works identically on both backends. The embedded path is the
realistic default for a chat-sidecar app — one file per user instead of
a hosted DB.

Run it
======
uv run python examples/30_workloads/04_agent_chat_with_memory.py
python examples/30_workloads/04_agent_chat_with_memory.py
uv run python examples/30_workloads/04_agent_chat_with_memory.py --config examples/khora.standard.yaml
python examples/30_workloads/04_agent_chat_with_memory.py --config examples/khora.standard.yaml
"""

from __future__ import annotations

import argparse
import asyncio
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
_ENTITY_TYPES = ["PERSON", "ORGANIZATION", "CONCEPT", "LOCATION", "EVENT", "PRODUCT", "TECHNOLOGY"]
_RELATIONSHIP_TYPES = ["RELATES_TO", "PART_OF", "MENTIONS"]


def _load_config() -> KhoraConfig:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    args = parser.parse_args()
    return KhoraConfig.from_yaml(args.config)


# ── Tool definitions ─────────────────────────────────────────────────
# In a real LangGraph / OpenAI Agents SDK setup these would be decorated
# (@tool / Tool(...)) and registered with the agent runner. The body is
# the same regardless of framework — the tool just calls into Khora.


async def tool_remember(kb: Khora, namespace, text: str) -> str:
    """Tool: store a fact about the user.

    Returns a short status string the agent can use to decide what to
    say next. Keeping return values terse matters — they get pickled
    into the next prompt as observation/tool-result.
    """
    result = await kb.remember(
        text,
        namespace=namespace,
        title=text[:60],
        entity_types=_ENTITY_TYPES,
        relationship_types=_RELATIONSHIP_TYPES,
    )
    return f"OK — stored ({result.chunks_created} chunks, {result.entities_extracted} entities)"


async def tool_recall(kb: Khora, namespace, query: str) -> dict:
    """Tool: look up what the agent knows about the user.

    Returns a dict instead of a plain string so the agent can branch on
    ``should_abstain``. This is the pattern that prevents the most
    common agent failure mode — confidently answering from empty
    context.

    We expose ``raw_top_score`` (the pre-fusion raw cosine of the
    strongest semantic hit, from ``engine_info["max_raw_vector_score"]``)
    rather than ``chunk.score``. ``chunk.score`` is post-fusion min-max
    normalized within the returned set: the top hit is always 1.00, the
    bottom is always 0.00, and thresholding on it is meaningless. The
    raw cosine is the actual confidence measure.
    """
    result = await kb.recall(query, namespace=namespace, limit=3)
    signals = result.engine_info.get("abstention_signals", {})
    return {
        "context": "\n\n".join(c.content for c in result.chunks),
        # Read this. It's the canonical "should I refuse?" signal.
        "should_abstain": signals.get("should_abstain", False),
        # Diagnostic. 0.0 means no chunks; > ~0.5 is a confident match,
        # < ~0.3 means there's nothing actually on-topic in the corpus.
        "raw_top_score": result.engine_info.get("max_raw_vector_score", 0.0),
        "chunks_returned": len(result.chunks),
    }


# ── Simulated agent loop ──────────────────────────────────────────────
# A real agent would call into an LLM here. We hard-code the calls so
# the demo runs without any extra API hops beyond Khora's own embedding
# and extraction. The PROMPT FRAGMENTS below show what a real agent
# would see in its tool-result observations.


async def main() -> None:
    config = _load_config()

    async with Khora(config, engine="chronicle", run_migrations=True) as kb:
        # One namespace per user — same pattern as demo 02. In a multi-
        # tenant chat app you would resolve user_id -> namespace_id from
        # a session store.
        ns = await kb.create_namespace()
        ns_id = ns.namespace_id

        # ── Turn 1: cold start (no memory yet) ────────────────────
        # The user opens with a question that requires memory. The
        # agent's first action: call recall(). Expected:
        # should_abstain=True (empty namespace), and the agent's
        # response is "I don't know yet" rather than a hallucinated
        # guess.
        print("=== Turn 1 — user asks before telling anything ===")
        user_msg = "Do you remember what conference I am speaking at next month?"
        print(f"USER: {user_msg}")
        memory = await tool_recall(kb, ns_id, user_msg)
        print(
            f"[tool recall -> abstain={memory['should_abstain']}, "
            f"raw_top_score={memory['raw_top_score']:.2f}, "
            f"chunks={memory['chunks_returned']}]"
        )
        # Agent's branch on the tool result. In a LangGraph node you
        # would encode this branch as conditional edges, not an if/else.
        if memory["should_abstain"]:
            print("AGENT: I do not have anything on file about that yet — could you tell me?")
        else:
            print(f"AGENT (would answer from context): {memory['context'][:120]}…")

        # ── Turn 2: user supplies a fact ──────────────────────────
        # The agent recognises this is new information and calls
        # remember() before answering. Two valuable things to call out:
        #
        #   • The agent SHOULD NOT recall-then-store; it stores first,
        #     then recall sees it on the next turn. Some agents try to
        #     "verify" by recall()-ing what they just stored; that
        #     wastes an LLM call.
        #   • The remember() return string is terse on purpose — the
        #     agent's prompt picks it up as a tool observation.
        print("\n=== Turn 2 — user supplies the fact ===")
        user_msg = "I am speaking at PyCon US 2026 in Pittsburgh on May 21st. My talk is about Khora."
        print(f"USER: {user_msg}")
        ack = await tool_remember(kb, ns_id, user_msg)
        print(f"[tool remember -> {ack}]")
        print("AGENT: Got it — I will remember PyCon US 2026 in Pittsburgh on May 21st.")

        # ── Turn 3: callback — does memory survive across turns? ──
        # Simulates the user returning to the same question a few turns
        # later. Now the namespace has the fact; abstention should NOT
        # fire; the agent answers from context.
        print("\n=== Turn 3 — user re-asks the original question ===")
        user_msg = "Hey — what was that conference I told you about?"
        print(f"USER: {user_msg}")
        memory = await tool_recall(kb, ns_id, user_msg)
        print(
            f"[tool recall -> abstain={memory['should_abstain']}, "
            f"raw_top_score={memory['raw_top_score']:.2f}, "
            f"chunks={memory['chunks_returned']}]"
        )
        if memory["should_abstain"]:
            print("AGENT: I still do not have anything — could you remind me?")
        else:
            # In a real agent loop this `context` string is what would
            # be injected into the LLM prompt as background. Here we
            # just print it so the demo is observable.
            preview = memory["context"][:300]
            ellipsis = "…" if len(memory["context"]) > 300 else ""
            print(f"AGENT (LLM context):\n{preview}{ellipsis}")


if __name__ == "__main__":
    asyncio.run(main())
