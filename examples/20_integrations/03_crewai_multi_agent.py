"""Multi-agent shared memory — CrewAI + khora.

Two agents on the same team, three namespaces:

- **Shared "org knowledge"** — both agents read and write here.
  Decisions, facts, anything the team should converge on.
- **Per-agent private notebook** — each agent's own scratch space.
  Agent-A can't see Agent-B's private notes and vice versa.

This shows namespace scoping the way agentic teams actually need it:
shared context where collaboration matters, isolated context where it
doesn't. The same primitive (khora namespaces) does both — you just
hand each agent a different ``KhoraMemory`` instance.

Pattern from Hindsight's ``support-agent-shared-knowledge`` and mem0's
autogen cookbook, ported to CrewAI + khora.

Why VectorCypher (the default):
The agents converge on entities (e.g. "the user wants Postgres" →
PERSON / TECHNOLOGY / PREFERENCE entities). Vector + graph fusion
gives both teams good recall over those entities. Chronicle would also
work but loses the entity-walking story.

Configuration:
Loads a YAML config via ``--config`` (default ``khora.embedded.yaml``
— in-memory ``sqlite_lance``, zero infra). Switch to PostgreSQL +
pgvector + Neo4j with ``--config examples/khora.standard.yaml``.
Requires ``OPENAI_API_KEY``.

Run it
======
uv run python examples/20_integrations/03_crewai_multi_agent.py
python examples/20_integrations/03_crewai_multi_agent.py
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
from uuid import UUID

from loguru import logger

from khora import Khora  # noqa: E402
from khora.config import KhoraConfig  # noqa: E402
from khora.integrations.crewai import KhoraMemory  # noqa: E402

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


# Stable, distinct user_ids per agent — ≥ 8 chars, never "default". The
# adapter rejects empty / generic ids to prevent silent cross-agent
# reads (#618 disaster mode).
RESEARCHER_ID = "agent-researcher-001"
WRITER_ID = "agent-writer-001"
TEAM_ID = "team-acme-prod"


def _make_memory(kb, namespace: UUID, user_id: str, *, scope_root: str = "/"):
    """Build a ``crewai.Memory`` for one agent against one namespace.

    Each ``KhoraMemory`` instance binds (kb, namespace, user_id). Two
    agents calling the same factory with the same namespace + different
    user_ids share storage but distinguish their writes by user_id
    metadata. Two agents calling with *different* namespaces are fully
    isolated.
    """
    return KhoraMemory(kb=kb, namespace=namespace, user_id=user_id, scope_root=scope_root)


async def main() -> None:
    config = _load_config()

    async with Khora(config, run_migrations=True) as kb:
        # ── Namespaces ───────────────────────────────────────────────
        # One shared namespace for org knowledge, one private namespace
        # per agent. All three live in the same khora instance.
        shared_ns = await kb.create_namespace()
        researcher_ns = await kb.create_namespace()
        writer_ns = await kb.create_namespace()

        print(f"Shared namespace:     {shared_ns.namespace_id}")
        print(f"Researcher private:   {researcher_ns.namespace_id}")
        print(f"Writer private:       {writer_ns.namespace_id}")

        # ── Memory handles ───────────────────────────────────────────
        # Each agent gets two handles: one onto the shared pool, one
        # onto its private notebook. Both wrap the same Khora instance.
        researcher_shared = _make_memory(kb, shared_ns.namespace_id, RESEARCHER_ID, scope_root="/team")
        researcher_private = _make_memory(kb, researcher_ns.namespace_id, RESEARCHER_ID, scope_root="/scratch")

        writer_shared = _make_memory(kb, shared_ns.namespace_id, WRITER_ID, scope_root="/team")
        writer_private = _make_memory(kb, writer_ns.namespace_id, WRITER_ID, scope_root="/scratch")

        # ── Researcher writes ────────────────────────────────────────
        # A shared finding (the whole team should know) + a private
        # WIP note (only the researcher should see).
        researcher_shared.remember(
            "Customer interviews show 80% want self-hosted Postgres.",
            scope="/team/findings",
            importance=0.9,
        )
        researcher_private.remember(
            "TODO: re-check the 80% number with Q1's larger cohort.",
            scope="/scratch/todos",
            importance=0.4,
        )

        # ── Writer reads from shared ─────────────────────────────────
        # The writer needs the researcher's finding to draft the blog
        # post — pulls it from shared.
        shared_hits = writer_shared.recall("what do customers want?", limit=5)
        print(f"\nWriter sees {len(shared_hits)} shared finding(s):")
        for hit in shared_hits:
            print(f"  [{hit.score:.2f}] {hit.record.content}")

        # ── Writer reads from researcher's PRIVATE — should be empty ─
        # writer's *own* private namespace doesn't have researcher's
        # TODOs. This is the isolation guarantee.
        own_private = writer_private.recall("what's on the TODO list?", limit=5)
        print(f"\nWriter's own private has {len(own_private)} hit(s) (expected: 0)")

        # ── Writer adds its own private notes + a shared decision ────
        writer_private.remember(
            "Draft outline: lead with the 80% stat, then talk price.",
            scope="/scratch/drafts",
            importance=0.5,
        )
        writer_shared.remember(
            "Blog post angle agreed: lead with self-hosted Postgres demand.",
            scope="/team/decisions",
            importance=0.8,
        )

        # ── Researcher comes back, pulls writer's decision from shared ─
        decision_hits = researcher_shared.recall("what's the blog post angle?", limit=3)
        print(f"\nResearcher sees {len(decision_hits)} shared decision(s):")
        for hit in decision_hits:
            print(f"  [{hit.score:.2f}] {hit.record.content}")

        # ── Stats per namespace ──────────────────────────────────────
        # Sanity check that the writes landed in the namespaces we
        # expect — shared has both contributions, privates have one each.
        shared_stats = await kb.stats(namespace=shared_ns.namespace_id)
        researcher_priv_stats = await kb.stats(namespace=researcher_ns.namespace_id)
        writer_priv_stats = await kb.stats(namespace=writer_ns.namespace_id)
        print(f"\nShared namespace docs:        {shared_stats.documents}")
        print(f"Researcher private docs:      {researcher_priv_stats.documents}")
        print(f"Writer private docs:          {writer_priv_stats.documents}")


if __name__ == "__main__":
    asyncio.run(main())
