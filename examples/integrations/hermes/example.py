"""Hermes + khora example - long-term memory via ``KhoraMemoryProvider``.

Runs without Postgres, Neo4j, ``hermes-agent``, or an API key. The mock
LLM patches ``litellm.acompletion`` / ``litellm.aembedding`` so the
example is hermetic. The khora fixture spins up an in-memory
``sqlite_lance`` backend in a tmp dir.

The example does NOT spin up a real Hermes runtime - that would pull in
``hermes-agent`` (and its ``requests==2.33.0`` pin which clashes with
khora's CVE-2026-25645 floor). Instead we stand up a minimal stand-in
for Hermes's ``MemoryProvider`` ABC so ``KhoraMemoryProvider`` builds
cleanly, then drive the lifecycle directly. Each step is what real
Hermes calls into.

Lifecycle exercised: ``initialize`` → ``queue_prefetch`` → ``sync_turn``
×3 → ``prefetch`` (hot path, abstention on cold cache) →
``handle_tool_call("memory_search")`` (blocking dispatch, real result)
→ ``on_pre_compress`` → ``on_session_end`` (drain) → ``shutdown``.

Kept light (3 turns) to fit the 30s CI smoke budget.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import types
from pathlib import Path

# Add repo root to sys.path so ``examples._helpers`` is importable when
# this script is run from its own directory (CI smoke loop does that).
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Stub the Hermes ABC before importing the adapter. The real
# ``hermes_agent.agent.memory_provider.MemoryProvider`` carries an
# ABCMeta with a handful of abstract methods; for the example we only
# need a class the dynamic subclass in the factory can extend.
_hermes_agent = types.ModuleType("hermes_agent")
_hermes_agent_agent = types.ModuleType("hermes_agent.agent")
_hermes_agent_agent_mp = types.ModuleType("hermes_agent.agent.memory_provider")


class _StubMemoryProvider:
    """Stand-in for hermes_agent.agent.memory_provider.MemoryProvider."""


_hermes_agent_agent_mp.MemoryProvider = _StubMemoryProvider
_hermes_agent_agent.memory_provider = _hermes_agent_agent_mp
_hermes_agent.agent = _hermes_agent_agent
sys.modules.setdefault("hermes_agent", _hermes_agent)
sys.modules.setdefault("hermes_agent.agent", _hermes_agent_agent)
sys.modules.setdefault("hermes_agent.agent.memory_provider", _hermes_agent_agent_mp)

from examples._helpers import embedded_khora, install_mock_llm  # noqa: E402
from khora.integrations.hermes import KhoraMemoryProvider  # noqa: E402


async def main() -> None:
    install_mock_llm()

    async with embedded_khora() as kb:
        # 1) Build the provider. ``kb`` is REQUIRED - no Khora.shared()
        #    fallback in the factory; the example plugin dir is the
        #    only place that defaults to it.
        provider = KhoraMemoryProvider(kb=kb, drain_timeout_s=20.0)

        # 2) Bind to a (agent_identity, session_id) pair. This is the
        #    Hermes-side tenancy key; the adapter derives the khora
        #    namespace UUID5 from it.
        hermes_home = tempfile.mkdtemp(prefix="hermes-example-")
        provider.initialize(
            session_id="demo-session",
            agent_identity="example-agent",
            hermes_home=hermes_home,
            platform="cli",
        )
        assert provider.namespace_id is not None, "namespace_id should be set after initialize"
        print(f"Provider bound to namespace_id={provider.namespace_id}")

        # 3) Warm the prefetch cache for the next turn. Fire-and-forget;
        #    returns immediately, runs against the (currently empty) kb.
        provider.queue_prefetch("What did Alice say about Phoenix?")

        # 4) Persist three conversation turns. ``sync_turn`` returns
        #    immediately - the runtime owns the actual kb.remember call
        #    and writes serialise in submission order (FIFO worker).
        provider.sync_turn(
            "Alice picked PostgreSQL for the Phoenix database.",
            "Got it - PostgreSQL for Phoenix.",
        )
        provider.sync_turn(
            "Bob said the release window is March 2026.",
            "Noted - Phoenix launches in March 2026.",
        )
        provider.sync_turn(
            "Alice mentioned the team grew to 12 engineers.",
            "Acknowledged: team size is 12.",
        )

        # 5) prefetch() is the LLM hot path - bounded by
        #    prefetch_timeout_s (default 0.8s). With writes still
        #    in-flight it returns the abstention payload rather than
        #    blocking the model. This is by design: better an empty
        #    context than a stalled turn.
        cold = provider.prefetch("Tell me about Phoenix")
        print("--- prefetch (hot path, writes still queued) ---")
        print(cold.strip())

        # 6) Tool dispatch is the blocking path - the LLM is already
        #    waiting on a tool result so we serve real data. The
        #    runtime FIFO ensures the writes drain before recall runs.
        result = provider.handle_tool_call(
            "memory_search",
            {"query": "Phoenix database", "top_k": 3},
        )
        assert isinstance(result, str) and len(result) > 0, "tool call returned empty result"
        print("--- memory_search tool call (blocking dispatch) ---")
        print(result.strip())

        # 7) Compress-time flush. Hermes calls this just before dropping
        #    old messages; we batch-remember whatever (user, assistant)
        #    pairs the caller surfaces.
        provider.on_pre_compress(
            messages=[
                {"role": "user", "content": "One more thing - Carol joined the Phoenix team."},
                {"role": "assistant", "content": "Carol joined Phoenix. Team is now 13."},
            ]
        )

        # 8) Drain the background queue and surface any elevated failure
        #    rate. Must run before the ``async with`` exits so writes
        #    don't race ``kb.disconnect()``.
        provider.on_session_end(messages=[])

        # 9) Tear down the runtime. Idempotent - safe even if
        #    on_session_end already shut things down internally.
        provider.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
