# Hermes integration

> **Stability warning.** The upstream `hermes-agent` SDK is pre-1.0 and
> the `MemoryProvider` ABC has reshaped on several minor bumps. The
> adapter pins tightly (`hermes-agent>=0.13,<0.14`) and is tagged
> **experimental**: plan one maintenance PR per upstream minor. A
> nightly skew job opens auto-issues when the latest upstream breaks
> the adapter contract.

`khora.integrations.hermes` plugs khora into Hermes as a long-term
memory plane. Hermes owns the agent loop, the model call, the tool
router, and the context-compression policy; khora owns the storage -
vector recall, entity graph, temporal-aware retrieval, abstention
signals.

The integration is a single primitive: `KhoraMemoryProvider`. Hand it
a connected `Khora` instance and pass the returned object to Hermes
through its plugin discovery (`$HERMES_HOME/plugins/khora/`). Hermes
calls into the provider for prefetch on every turn, persists each
turn through `sync_turn`, and routes the two LLM-callable tools
(`memory_search` / `memory_recall`) through `handle_tool_call`.

## Install

Hermes ships an exact pin on `requests==2.33.0`, and khora floors
`requests>=2.33.1` to pick up CVE-2026-25645. Those two are
incompatible, so the `[hermes]` extra was removed from `pyproject.toml`
during Wave C. Install `hermes-agent` yourself:

```bash
pip install hermes-agent
```

khora's adapter is still registered under the `khora.integrations`
entry-point group (factory: `KhoraMemoryProvider`), so
`khora.integrations.discover()` resolves it at runtime whenever
`hermes-agent` is also importable in the active environment. No
`pip install 'khora[hermes]'` step.

If your project must enforce the CVE floor, vendor or fork
`hermes-agent` to relax its pin; track upstream's progress on
[the pin relaxation](https://github.com/nousresearch/hermes-agent/issues)
before the adapter can move back to a single-line install.

## Quick start

Two ways to wire the provider into Hermes:

### A. The example plugin directory

`examples/integrations/hermes/plugin/` is a runnable Hermes plugin.
Copy it into Hermes's plugin search root:

```bash
cp -r examples/integrations/hermes/plugin "$HERMES_HOME/plugins/khora"
```

The plugin's `register(ctx)` calls `KhoraMemoryProvider(kb=Khora.shared())`
by default. Override the Khora source with the
`KHORA_HERMES_KB_FACTORY` env var (`module.path:attr` form, zero-arg
callable returning a `Khora` instance):

```bash
export KHORA_HERMES_KB_FACTORY=myapp.memory:build_kb
```

### B. Direct construction

For embedded uses where you control the Hermes session lifecycle:

```python
from khora import Khora
from khora.integrations.hermes import KhoraMemoryProvider

kb = Khora()
await kb.connect()
provider = KhoraMemoryProvider(kb=kb)
# ... hand provider to your Hermes context.register_memory_provider(...)
```

`kb` is REQUIRED. The factory does not silently call `Khora.shared()`
- adapter lifecycle is the caller's problem. The example plugin is
the only place `Khora.shared()` is wired in.

## Namespace mapping

Each `(agent_identity, session_id)` pair maps to a deterministic khora
namespace UUID5, so two providers for the same agent + session share
memory across processes without a shared registry. The derivation
lives in `khora.integrations.hermes._mapping`:

```python
# UUID_NAMESPACE_HERMES = uuid5(NAMESPACE_OID, "khora.integrations.hermes")

def derive_namespace_uuid(agent_identity: str, session_id: str) -> UUID:
    return uuid5(UUID_NAMESPACE_HERMES, f"hermes:{agent_identity}:{session_id}")
```

`agent_identity` is the tenancy key - different agents stay isolated
even when they share a `session_id`. The session id is the
conversation scope; switching sessions for the same agent (via
`on_session_switch`) re-binds the provider to a fresh namespace.

The Hermes provider also stamps the same `session_id` onto every
stored document so `Khora.forget_session(namespace, session_id)`
cleanly drops a whole conversation (see #620 for the
session-id-as-first-class-column work).

## The threading model

The adapter is the only one in `khora.integrations` that bridges a
sync caller (Hermes drives `MemoryProvider` from one thread per
session) onto khora's async write path. The substrate lives in
`_runtime.py`:

- One `ThreadPoolExecutor(max_workers=1)` per provider. Single worker
  = strict FIFO; writes serialise in submission order so ingestion
  order matches turn order.
- All async work routes through the process-wide
  `khora.integrations._sync.run_sync` bridge loop. The runtime does
  NOT own its own event loop.
- A TTL-bounded prefetch cache keyed on
  `(namespace_id, session_id, bounded_text_hash(query))` absorbs the
  "prefetch on every turn" pattern without firing N concurrent
  recalls for the same question. The cache slot is populated with
  the in-flight `Future` _before_ the worker submits - readers
  arriving mid-flight wait on the same Future instead of racing.
- **Shed-oldest** when the queue is at `queue_max_size`. The oldest
  pending Future is cancelled and `khora.hermes.queue.shed_total` is
  incremented. WARN log fires at most once per 10 seconds - sustained
  shedding indicates ingest throughput cannot keep up.
- **Bounded error ring** (16 entries, each truncated to 200 chars).
  `failure_rate_pct()` powers the WARN at `on_session_end` when the
  ratio exceeds `failure_threshold_pct`. The ring is the operator's
  forensic surface - no raw exception text on spans (cardinality
  rule).
- **Cache coherency on failure**: if a recall Future raises, the
  cache slot is dropped so the next caller retries instead of pinning
  a Future that will keep raising. The done-callback checks `is …`
  identity before evicting so it doesn't kick out a fresh enqueue
  that replaced the failed entry.
- **Lock order**: `_cache_lock` → `_pending_lock`. Never inverted.
  `_submit_write` only takes `_pending_lock`; `enqueue_recall` takes
  `_cache_lock` then calls into `_submit_write`. No deadlock surface.

Honest caveats:

- **Chat memory is best-effort, not durable.** A SIGKILL or hard
  process crash mid-drain loses whatever is still in the executor
  queue. SIGTERM with a clean shutdown drains up to `drain_timeout_s`.
- **`prefetch()` is allowed to return abstention.** When khora is
  slow or writes haven't drained, the LLM gets an empty
  `<memory-context>` block rather than blocking the turn. Better an
  empty context than a stalled agent.
- **Fork safety is a follow-up** (#790). Don't `os.fork()` after
  constructing a provider - the executor and bridge loop don't survive.

## Tool calls

Two LLM-callable tools. Hermes registers both via
`provider.get_tool_schemas()`:

| Tool | When the LLM picks it | Returns |
|---|---|---|
| `memory_search` | "What did Alice say about Phoenix?" - pure semantic recall. | Top-K chunks + entity hits formatted as a `<memory-context>` block. |
| `memory_recall` | "What did we discuss last week?" - same plus `before` / `after` ISO 8601 bounds. | Same shape, filtered by the time window. |

Both tools accept `query` (required, non-empty), `top_k` (default 10,
hard cap 50), and `min_similarity` (default 0.1, range [0.0, 1.0]).
`memory_recall` adds `before` / `after`; sending `after > before`
raises a `ValueError` back to the model.

The returned `<memory-context>` block lists up to five chunks ranked
by score, each prefixed with `[score: X.XX, YYYY-MM-DD]` and capped
at 500 characters. Empty result → `"No prior memories found."` -
the explicit abstention payload, so the model knows not to confabulate.

Tool dispatch is the blocking path (via `runtime.dispatch_sync`) -
the LLM is already waiting on a tool result, so we serve real data
even if it means waiting on the FIFO queue.

## Knobs

All constructor kwargs on `KhoraMemoryProvider`:

| arg | default | notes |
|---|---|---|
| `kb` | - | REQUIRED. A connected `Khora` instance. |
| `runtime` | `None` | Inject a custom `_KhoraRuntime` (tests only). |
| `prefetch_timeout_s` | `0.8` | Upper bound on `prefetch()`. On timeout, returns the abstention payload. Raise for slow backends; lower if your agent's turn budget is tight. |
| `prefetch_cache_ttl_s` | `30.0` | Lifetime of a cached recall. Raise when the same question repeats often; lower when freshness matters more than dedup. |
| `queue_max_size` | `256` | FIFO queue cap. Beyond this, the oldest pending write is cancelled (shed-oldest). Raise if your conversation rate is bursty; lower to fail fast on backpressure. |
| `drain_timeout_s` | `5.0` | Wall clock budget for `on_session_end` to drain the queue. Raise it for slow extraction pipelines. |
| `failure_threshold_pct` | `1.0` | `on_session_end` logs WARN when the background failure rate exceeds this. Tune to your operational tolerance. |

## Telemetry

All under the public stability tag in `docs/telemetry-contract.json`:

| Name | Kind | Labels | Notes |
|---|---|---|---|
| `khora.integrations.hermes.initialize` | span | `hermes.agent_identity_hash`, `hermes.session_id_hash`, `hermes.platform` | One span per provider bind. |
| `khora.integrations.hermes.prefetch` | span | `cache_hit`, `result_count`, `hermes.query_hash` | Wraps the sync prefetch entry point. |
| `khora.integrations.hermes.sync_turn` | span | `hermes.user_content_hash`, `hermes.assistant_content_hash` | Wraps the enqueue step; real I/O happens after the span closes. |
| `khora.hermes.remember.success_total` | counter | `op` ∈ {`remember`, `remember_batch`} | Successful background writes. |
| `khora.hermes.remember.failed_total` | counter | `op` ∈ {`remember`, `remember_batch`} | Failed background writes. Feeds the failure-rate gauge. |
| `khora.hermes.queue.shed_total` | counter | `op` ∈ {`remember`, `remember_batch`, `recall`} | Shed-oldest evictions. Sustained = ingest throughput too low. |
| `khora.hermes.tool_call_total` | counter | `tool` ∈ {`memory_search`, `memory_recall`, `unknown`, `uninitialized`} | Tool dispatch counts. |

**Cardinality rule** (CLAUDE.md): no `namespace_id` label on any
metric. Free-text values (queries, content) appear only as
`bounded_text_hash` span attributes - never as labels, never raw.

Dashboards / alerting hooks live alongside the rest of the khora
telemetry surface; the same OTel exporter chain picks these up
without extra wiring.

## Quickstart example

The block below is byte-identical to
[`examples/integrations/hermes/example.py`](../../examples/integrations/hermes/example.py)
- CI fails if they diverge.

```python title="example.py"
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

Kept light (3 turns) to fit the 30s CI smoke budget - every
``sync_turn`` triggers a full extraction pipeline that retries 3× on
the mock LLM's non-JSON output.
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
```

The block above is enforced byte-identical against
`examples/integrations/hermes/example.py` by
`tools/check_examples_drift.py` (CI gate).

## Stability

Tagged **experimental**. The Hermes `MemoryProvider` ABC is still
pre-1.0; we expect rework per upstream minor. The adapter pin is
`hermes-agent>=0.13,<0.14` - one minor wide. Bump in a deliberate PR
per upstream minor; the nightly skew job flags breakage against the
latest tagged minor.

The integration is promoted to **stable** once `hermes-agent` ships
one full minor without reshaping `MemoryProvider` and the khora
adapter requires no API change to track it.

## Known issues

- **`requests==2.33.0` exact pin in `hermes-agent`** clashes with
  khora's `requests>=2.33.1` floor (CVE-2026-25645). The `[hermes]`
  extra was removed during Wave C because of this. Users install
  `hermes-agent` themselves and accept the lower `requests` version
  in their resolver, or fork the upstream pin.
- **Fork safety (#790).** Constructing a provider before `os.fork()`
  is unsupported - the FIFO executor and the `run_sync` bridge loop
  don't survive the fork. Build providers in worker processes after
  the fork.
- **Plugin discovery is filesystem-based.** Hermes scans
  `$HERMES_HOME/plugins/<name>/`. The example dir must be copied in;
  there is no `pip install` step that wires the plugin automatically.
  An entry-point-based discovery model is a candidate for v2.
- **`prefetch()` abstention is not a bug.** When the hot path returns
  `"No prior memories found."`, it means writes hadn't drained yet
  or khora was slow. Use the tool-call path (`memory_search` /
  `memory_recall`) for guaranteed retrieval.
