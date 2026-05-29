"""Chat agent with long-term memory via LangGraph + khora.

A multi-turn chat agent backed by ``KhoraStore`` from the LangGraph
adapter. The agent recalls relevant prior context before each reply
and persists the new turn afterward. Memory survives the StateGraph's
lifecycle — drop the graph, build a new one with the same store, and
the user's history is still there.

The graph itself is intentionally minimal: one "respond" node that
reads memory + the user message and writes back the assistant reply.
A real production agent would add LLM tool-use, routing, and
guardrails on top — but those are LangGraph concerns, not memory
concerns. This example focuses on the memory contract.

Why VectorCypher (the default engine):
The agent surfaces "what did Alice say last week" style queries that
benefit from temporal scoring + entity recall. VectorCypher gives both
out of the box. For a chat-only deployment on Postgres-alone, swap to
Chronicle (``engine="chronicle"``) — same code, simpler infra.

Configuration:
Loads a YAML config via ``--config`` (default ``khora.embedded.yaml``
— in-memory ``sqlite_lance``, zero infra). Switch to PostgreSQL +
pgvector + Neo4j with ``--config examples/khora.standard.yaml``.
Requires ``OPENAI_API_KEY``.

Run it
======
uv run python examples/20_integrations/01_langgraph.py
python examples/20_integrations/01_langgraph.py
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph  # noqa: E402
from langgraph.graph.message import add_messages  # noqa: E402
from loguru import logger

from khora import Khora  # noqa: E402
from khora.config import KhoraConfig  # noqa: E402
from khora.integrations.langgraph import KhoraStore  # noqa: E402

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


_DEFAULT_CONFIG = Path(__file__).parent / "khora.embedded.yaml"


def _load_config() -> KhoraConfig:
    """Parse ``--config`` and load the named Khora YAML.

    Kept inline (rather than in a shared helper) so each example is
    readable on its own — copy / paste a file into your project and it
    works without dragging an ``examples/_common.py`` along. Matches the
    convention used by the numbered tutorials (01_hello_memory.py through
    08_slack_archive_bulk.py).
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help=f"Khora YAML config path (default: {_DEFAULT_CONFIG.name}).",
    )
    args = parser.parse_args()
    return KhoraConfig.from_yaml(args.config)


# Stable user id ≥ 8 chars — the adapter rejects empty / "default" /
# short ids to prevent silent cross-user reads (#618 disaster mode).
USER_ID = "alice-acme-prod"

# A whole conversation lives under one namespace tuple. The tuple is
# the LangGraph namespace; KhoraStore flattens it onto a khora namespace
# scoped to USER_ID.
MEMORY_NS = ("chat", "alice", "messages")


class ChatState(TypedDict):
    """State threaded through the StateGraph on each turn.

    ``messages`` is the running conversation transcript. ``recalled``
    carries the memory hits from the recall step into the respond step
    (printed for visibility; a real agent would splice them into the LLM
    prompt).
    """

    messages: Annotated[list[dict], add_messages]
    recalled: list[str]


async def _recall_relevant(state: ChatState, store: KhoraStore) -> ChatState:
    """Pull memories relevant to the latest user message.

    The store's ``asearch`` does a semantic match over previously stored
    items in the namespace. Top hits go into ``state["recalled"]`` so
    the next node can surface them in the reply.
    """
    user_msg = state["messages"][-1]["content"]
    hits = await store.asearch(MEMORY_NS, query=user_msg, limit=3)
    state["recalled"] = [hit.value.get("text", "") for hit in hits]
    return state


async def _respond_and_persist(state: ChatState, store: KhoraStore) -> ChatState:
    """Generate a reply (mocked) and persist the new exchange."""
    user_msg = state["messages"][-1]["content"]

    # In a real agent: feed user_msg + state["recalled"] into the LLM.
    # With the mock LLM we just acknowledge.
    reply = f"(noted) {user_msg}"

    # Persist the user turn so future queries can recall it. We use the
    # message text as the value and a timestamp-derived key so each turn
    # gets its own row.
    turn_id = f"turn-{len(state['messages']):04d}"
    await store.aput(MEMORY_NS, turn_id, {"text": user_msg, "role": "user"})

    state["messages"].append({"role": "assistant", "content": reply})
    return state


def _build_graph(store: KhoraStore):
    """Wire the two-node graph: recall → respond."""
    builder = StateGraph(ChatState)
    builder.add_node("recall", lambda s: _recall_relevant(s, store))
    builder.add_node("respond", lambda s: _respond_and_persist(s, store))
    builder.set_entry_point("recall")
    builder.add_edge("recall", "respond")
    builder.set_finish_point("respond")
    return builder.compile(store=store)


async def main() -> None:
    config = _load_config()

    async with Khora(config, run_migrations=True) as kb:
        store = KhoraStore(kb, user_id=USER_ID)
        graph = _build_graph(store)

        # ── Turn 1 ───────────────────────────────────────────────────
        state: ChatState = {
            "messages": [{"role": "user", "content": "I prefer Postgres for new services."}],
            "recalled": [],
        }
        state = await graph.ainvoke(state)
        print(f"Turn 1 reply: {state['messages'][-1]['content']!r}")
        print(f"Turn 1 recalled: {state['recalled']}")

        # ── Turn 2 (different topic; recall should still find turn 1) ─
        state = {"messages": [{"role": "user", "content": "What database did I say I liked?"}], "recalled": []}
        state = await graph.ainvoke(state)
        print(f"\nTurn 2 reply: {state['messages'][-1]['content']!r}")
        print(f"Turn 2 recalled: {state['recalled']}")

        # The recall on turn 2 surfaces the turn-1 message because the
        # store persists across graph invocations — that's the whole
        # point. A new ``_build_graph(store)`` would still see it.


if __name__ == "__main__":
    asyncio.run(main())
