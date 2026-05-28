"""Tool-router learning — memory as a routing oracle.

Two tools that look interchangeable but resolve different request
types. The agent doesn't know upfront which tool handles which kind of
request — it has to *learn* by recording outcomes and biasing future
selections.

Pattern from Hindsight's ``tool-learning-demo``: every tool call is
captured as a ``(request_signature, tool_used, outcome)`` event. When
a new request arrives, the agent recalls similar past requests,
extracts the tools that succeeded, and routes accordingly. After
enough rounds the router converges — high-weight prior routes
dominate, errant routes fade via Chronicle's decay.

Why Chronicle (engine="chronicle"):
The pattern is fundamentally about *event sequences* — every tool
call is a temporal event. Chronicle's bi-temporal storage + recency
decay + abstention signals give us the right primitives: high-weight
recent successes win, old failures fade. No graph needed; pure
event-shaped memory.

What you're seeing:

- A simulated stream of incoming requests of two kinds — "billing"
  and "infra". The two tools internally know which they handle, but
  the *agent* doesn't.
- Round 1: agent picks tools at random (or by hash), records outcomes.
- Round 2+: agent recalls similar prior requests, picks the tool with
  the highest aggregate success score.
- After ~10 requests the routing converges.

The pattern transfers to: which RAG corpus to query, which API
endpoint to hit, which sub-agent to delegate to. Anywhere you have an
oracle problem and outcome feedback.

Loads YAML config via ``--config`` (default ``khora.embedded.yaml``); switch to PG+Neo4j with ``--config examples/khora.standard.yaml``. Requires ``OPENAI_API_KEY``.

Run it
======
uv run python examples/30_workloads/10_tool_router_learning.py
python examples/30_workloads/10_tool_router_learning.py
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from loguru import logger

from khora import Khora  # noqa: E402
from khora.config import KhoraConfig  # noqa: E402

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


_ENTITY_TYPES = ["TOOL", "REQUEST_TYPE", "OUTCOME"]
_RELATIONSHIP_TYPES = ["RELATES_TO", "MENTIONS"]


# ── Simulated tools ─────────────────────────────────────────────────
# Real implementations would call an API; here they just inspect the
# request text and return success / failure based on category match.


def _tool_alpha(request: str) -> tuple[bool, str]:
    """Alpha handles billing requests."""
    if "invoice" in request.lower() or "billing" in request.lower():
        return True, f"alpha resolved: {request}"
    return False, f"alpha failed: {request} (not a billing request)"


def _tool_omega(request: str) -> tuple[bool, str]:
    """Omega handles infrastructure requests."""
    if "server" in request.lower() or "deploy" in request.lower() or "infra" in request.lower():
        return True, f"omega resolved: {request}"
    return False, f"omega failed: {request} (not an infra request)"


_TOOLS = {"alpha": _tool_alpha, "omega": _tool_omega}


# ── Memory-driven routing ───────────────────────────────────────────


async def _recall_route(kb: Khora, namespace_id: UUID, request: str) -> str | None:
    """Find the tool that historically succeeded for similar requests.

    Returns the tool name with the most prior successes within the
    recalled context, or None if there's no signal yet.
    """
    result = await kb.recall(
        f"Which tool resolved a request like: {request}",
        namespace=namespace_id,
        limit=5,
    )
    # Tally tool wins from prior outcome events. Each event was
    # written as "tool <name> succeeded: <request>" or "... failed:".
    tally: dict[str, float] = {}
    for chunk in result.chunks:
        for tool_name in _TOOLS:
            marker_succ = f"tool {tool_name} succeeded"
            marker_fail = f"tool {tool_name} failed"
            if marker_succ in chunk.content:
                tally[tool_name] = tally.get(tool_name, 0.0) + chunk.score
            elif marker_fail in chunk.content:
                tally[tool_name] = tally.get(tool_name, 0.0) - chunk.score * 0.5
    if not tally:
        return None
    # Pick the tool with the highest aggregate score.
    return max(tally.items(), key=lambda kv: kv[1])[0]


async def _record_outcome(
    kb: Khora,
    namespace_id: UUID,
    *,
    request: str,
    tool: str,
    success: bool,
) -> None:
    """Persist (request, tool, outcome) as a Chronicle event."""
    verdict = "succeeded" if success else "failed"
    body = f"tool {tool} {verdict}: handled request '{request}'. Outcome: {'resolved' if success else 'unresolved'}."
    await kb.remember(
        body,
        namespace=namespace_id,
        title=f"{tool}-{verdict}: {request[:40]}",
        metadata={
            "occurred_at": datetime.now(UTC).isoformat(),
            "tool": tool,
            "outcome": verdict,
        },
        entity_types=_ENTITY_TYPES,
        relationship_types=_RELATIONSHIP_TYPES,
    )


async def _handle(kb: Khora, namespace_id: UUID, request: str, *, round_idx: int) -> None:
    """One request → recall route → call tool → record outcome."""
    suggested = await _recall_route(kb, namespace_id, request)
    if suggested is None:
        # Cold start — pick deterministically by request hash so the
        # demo is reproducible. Real code might use random choice or
        # round-robin.
        suggested = "alpha" if hash(request) % 2 == 0 else "omega"
        chosen_by = "cold-start"
    else:
        chosen_by = "memory"

    success, _detail = _TOOLS[suggested](request)
    await _record_outcome(kb, namespace_id, request=request, tool=suggested, success=success)

    arrow = "✓" if success else "✗"
    print(f"  round {round_idx:>2}: {arrow} via {suggested:>5} (by {chosen_by:>10}) — {request}")


async def main() -> None:
    # Chronicle: events + decay + abstention. Exactly the shape this
    # learning loop wants.
    config = _load_config()

    async with Khora(config, engine="chronicle", run_migrations=True) as kb:
        namespace = await kb.create_namespace()
        ns_id = namespace.namespace_id

        # ── Request stream ───────────────────────────────────────────
        # Mixed billing + infra requests. The agent should converge
        # to: billing → alpha, infra → omega.
        requests = [
            "Resend the invoice for account 4501",  # billing
            "Restart the staging server in us-east",  # infra
            "Update the deploy pipeline for prod",  # infra
            "Bill the customer for January usage",  # billing
            "Refund invoice 7820",  # billing
            "Roll back the latest deploy",  # infra
            "Issue a credit memo against invoice 3104",  # billing
            "Scale the infra group to 8 nodes",  # infra
            "Apply a discount to the next invoice",  # billing
            "Drain traffic from the canary deploy",  # infra
        ]

        print(f"Processing {len(requests)} requests…\n")
        for idx, req in enumerate(requests, start=1):
            await _handle(kb, ns_id, req, round_idx=idx)

        # ── Verify convergence ───────────────────────────────────────
        # Recall against the namespace to see what the router has
        # "learned". With a real LLM the natural-language summary
        # surfaces the dominant routes; with mock LLM we just count
        # the recorded outcomes by tool.
        stats = await kb.stats(namespace=ns_id)
        print(f"\nNamespace stats: {stats.documents} events recorded.")

        for tool_name in _TOOLS:
            wins = await kb.recall(
                f"requests that {tool_name} resolved",
                namespace=ns_id,
                limit=20,
            )
            success_count = sum(1 for c in wins.chunks if f"tool {tool_name} succeeded" in c.content)
            fail_count = sum(1 for c in wins.chunks if f"tool {tool_name} failed" in c.content)
            print(f"  {tool_name}: {success_count} success / {fail_count} fail (from top-20 recall)")


if __name__ == "__main__":
    asyncio.run(main())
