# Semantic Hooks

Semantic hooks let you subscribe to events during document ingestion and recall and receive callbacks when the events - or the entities, relationships, and chunks they carry - match your criteria. Use them to build real-time notifications, dashboards, or downstream processing pipelines.

## Quick Example

```python
from khora import Khora
from khora.hooks import SemanticFilter

async with Khora() as kb:
    # Simple: get notified on every new entity
    async def on_entity(event):
        print(f"New entity: {event.data.get('name')} ({event.data.get('entity_type')})")

    kb.subscribe("entity.created", on_entity)

    # Type pre-filter - Level 0
    filter = SemanticFilter(
        name="competitor_mentions",
        description="Any mention of a competitor company",
        entity_types=["ORGANIZATION"],
    )
    kb.subscribe("entity.created", on_entity, filter=filter)

    # Ingestion triggers callbacks
    ns = await kb.create_namespace()
    await kb.remember(
        "Acme Corp announced a partnership with Globex.",
        namespace=ns.namespace_id,
        entity_types=["ORGANIZATION"],
        relationship_types=["PARTNERSHIP_WITH"],
    )
```

## 3-Level Filter Cascade

Hooks use a cost-efficient cascade - each level is only evaluated if the previous level passes:

| Level | Method | Cost | Purpose |
|-------|--------|------|---------|
| 0 | Type pre-filter + structural `match` DSL | Free | Match by entity/relationship types or structural patterns on `event.data` |
| 1 | Embedding similarity | Sub-millisecond | Binary-quantized cosine similarity gate |
| 2 | LLM evaluation | Per-call LLM cost | Nano-model yes/no for ambiguous cases |

### Level 0: Type Filter

Filters by entity type, relationship type, dream op type, or dream
decision. No computation required:

```python
SemanticFilter(
    name="people_only",
    entity_types=["PERSON"],
)
```

Dream-phase pre-filters narrow on `dream.op.decided` events only:

```python
SemanticFilter(
    name="merge_ops",
    dream_op_types=["merge"],        # op_type in event.data
    dream_decisions=["apply"],       # decision in event.data
)
```

To scope any filter to a single namespace, set `namespace_id`:

```python
SemanticFilter(
    name="tenant_scoped",
    entity_types=["PERSON"],
    namespace_id=some_uuid,  # None (default) = all namespaces
)
```

Namespace-scoped subscriptions survive `create_namespace_version()`: the
dispatcher caches the stable-to-row-id mapping and self-heals it on a failed
scope comparison (#1427), so a scoped subscription keeps firing after a
namespace version bump - it does not silently go quiet.

### Level 0: Structural matching (`match` DSL)

`SemanticFilter.match` accepts an [EventBridge-style](https://docs.aws.amazon.com/eventbridge/latest/userguide/eb-create-pattern-operators.html) filter pattern that runs against `event.data`. Patterns are pure-data (no code execution), evaluated in the dispatcher hot path, and cached for repeated keys.

```python
SemanticFilter(
    name="high_confidence_acme",
    entity_types=["ORGANIZATION"],
    match={
        "name": [{"prefix": "Acme"}],
        "confidence": [{"numeric": [">=", 0.8]}],
    },
)
```

**Operators** (in `value` position):

| Operator | Example | Notes |
|----------|---------|-------|
| `String literal` | `"key": ["v1", "v2"]` | OR over listed values |
| `prefix` | `"key": [{"prefix": "Ac"}]` | String starts-with |
| `suffix` | `"key": [{"suffix": "rp"}]` | String ends-with |
| `equals-ignore-case` | `"key": [{"equals-ignore-case": "AcmE"}]` | Case-insensitive equality |
| `wildcard` | `"key": [{"wildcard": "Ac*me?"}]` | `*` = 0+ chars, `?` = 1 char |
| `numeric` | `"key": [{"numeric": [">=", 0.8]}]` | Single op or multi-op AND: `[">=", 0.5, "<", 0.9]` |
| `anything-but` | `"key": [{"anything-but": ["a", "b"]}]` | Negation; accepts nested `{"prefix": ...}` |
| `exists` | `"key": [{"exists": True}]` | Key presence check (True/False) |
| `contains-all` | `"key": [{"contains-all": ["x", "y"]}]` | All listed items must be in the (list-valued) field |

**Top-level operators**:

- `match["$or"] = [<pattern1>, <pattern2>, ...]` - disjunction across whole patterns
- Other top-level keys combine with implicit AND

**Important**: nested keys via dot-notation are intentionally **not** supported - pre-flatten any nested data you want to match into `event.data` before dispatch.

### Level 1: Embedding pre-screen

Compares the filter's description embedding against the entity/relationship embedding using binary-quantized Hamming distance, then full cosine on survivors:

```python
SemanticFilter(
    name="ai_research",
    description="Research related to artificial intelligence and machine learning",
    similarity_threshold=0.5,  # cosine similarity cutoff (per-filter; default 0.5)
)
```

### Level 2: LLM evaluation

> **Default OFF.** Level 2 is gated by `KHORA_HOOKS_LLM_EVALUATION_ENABLED=true`. **This incurs real LLM cost** - every filter with `examples` evaluated against every passing event makes a nano-LLM call (rate-limited and budgeted; see [Cost Controls](#cost-controls)). Audit your filter set and confirm the per-hour budget before flipping it on in production.

Level 2 **only fires when the filter supplies `examples`** - without them the LLM has no calibration and produces noise, so the dispatcher silently skips Level 2 (Level 1 result is final). To enable the LLM tier on a filter:

```python
SemanticFilter(
    name="strategic_mentions",
    description="Any mention of strategic business decisions",
    entity_types=["EVENT", "CONCEPT"],
    examples=[
        "The board approved a $50M acquisition.",
        "Q3 strategy: pivot to enterprise.",
    ],
    anti_examples=[
        "Lunch was tasty today.",
    ],
    llm_confidence_threshold=0.5,
)
```

## Event Types

Subscribe to any of these event types. Names are stable strings - the `EventType` enum in `khora.core.models.event` is canonical:

| Event | Fires When | Carries |
|-------|------------|---------|
| `entity.created` | New entity extracted during ingestion | `name`, `entity_type`, `description`, `embedding`, `chunk_id` |
| `entity.updated` | Existing entity mutated | Same shape as created |
| `entity.merged` | Two entities collapsed by cross-tool unification | Surviving entity + `merged_from_ids` |
| `entity.deleted` | Entity removed | `entity_id`, `namespace_id` |
| `relationship.created` | New relationship extracted | `source_entity_id`, `target_entity_id`, `relationship_type`, `confidence` |
| `relationship.updated` / `relationship.deleted` | Relationship mutation/removal | - |
| `chunk.created` | Chunk stored | `chunk_id`, `document_id`, content metadata |
| `chunk.embedded` | Chunk embedding generated | `embedding`, `embedding_model` |
| `chunk.entities_resolved` | All entity events for one chunk have been dispatched | `chunk_id`, `document_id`, `entity_ids` (capped at 50, sorted by entity id; `truncated: True` when capped), `entity_names_by_type`, `entity_count`, `occurred_at` |
| `document.created` / `document.updated` / `document.deleted` | Document lifecycle | - |
| `document.processed` / `document.failed` | Document finished or errored mid-pipeline | `status`, optional `error` |
| `episode.created` / `episode.updated` / `episode.deleted` | Episode lifecycle | - |
| `namespace.created` / `namespace.updated` / `namespace.deleted` | Namespace lifecycle | - |
| `sync.started` / `sync.completed` / `sync.failed` / `sync.checkpoint` | Connector sync lifecycle (Phase B) | - |
| `recall.requested` / `recall.results_ready` / `recall.completed` | Recall pipeline (query-side hooks) | `query`, result counts |
| `dream.run.started` / `dream.run.completed` / `dream.run.failed` | Dream run lifecycle | `run_id`, phase counts / error |
| `dream.phase.started` / `dream.phase.completed` | Dream phase boundary | `phase`, `run_id` |
| `dream.op.decided` | Dream orchestrator committed one memory op | `op_type`, `decision`, `run_id`, `entity_id` |

`chunk.entities_resolved` is the event to subscribe to for **co-occurrence filtering** - see [Co-occurrence example](#co-occurrence-filtering) below.

## Cost Controls

Level 2 (LLM) evaluations cost money. khora ships three layers of cost control, all OTel-instrumented:

### 1. Default-OFF gate

`KHORA_HOOKS_LLM_EVALUATION_ENABLED=false` (default). When false, Level 1 is final - even for filters that supply `examples`.

### 2. Token budgets

Two independent rolling-hour budgets enforce caps on Level 2 token spend:

- **Per-namespace cap** (`llm_max_tokens_per_namespace_per_hour`, default `10000`) - global ceiling per namespace.
- **Per-subscription cap** (`llm_max_tokens_per_subscription_per_hour`, default `0` = disabled) - a single noisy filter can no longer drain its namespace's hourly allowance. Recommended setting: `namespace_cap / expected_subscription_count`.

When **either** cap is exceeded, the affected batch **fails open** (returns `True` so the Level 1 match is preserved) and emits `khora.hooks.llm.throttled_total`. A warn-once log fires per window so operators see the breach.

### 3. Decision cache + intra-batch coalescing

- **Cross-batch cache** keyed on `(filter_id, bounded_text_hash(event_summary))` - `event_summary` is `entity_type | name | description`, truncated and SHA1-hashed so no raw text escapes the dispatcher. Subsequent events that hash to the same key short-circuit the LLM. Cache uses TTL + LRU eviction; configurable via `llm_cache_size` (default `2048`, `0` disables) and `llm_cache_ttl_seconds` (default `3600`, `0` = no expiry).
- **Intra-batch coalescing**: within a single LLM batch (default flush window `100ms`, batch size `10`), the evaluator deduplicates pending pairs by `event_summary_hash` before building the prompt. **A burst of 50 identical events → 1 LLM call** (first batch) → all remaining events hit cache. Without the cache primed, a batch of 10 identical events spends 1 prompt slot, not 10, and the result fans out to all 10 awaiting futures.

The two layers compose: cross-batch cache covers bursts that span flush windows; intra-batch coalescing covers a single flush window's duplicates.

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `KHORA_HOOKS_ENABLED` | Enable the hook system | `true` |
| `KHORA_HOOKS_FILTER_MODEL` | LLM for Level 2 evaluation | `gpt-4.1-nano` |
| `KHORA_HOOKS_DEFAULT_SIMILARITY_THRESHOLD` | Level 1 cosine threshold (filter default) | `0.5` |
| `KHORA_HOOKS_MAX_CONCURRENT_CALLBACKS` | Max concurrent callback executions | `10` |
| `KHORA_HOOKS_CALLBACK_TIMEOUT_SECONDS` | Per-callback timeout | `30.0` |
| `KHORA_HOOKS_LLM_EVALUATION_ENABLED` | Enable Level 2 (LLM) evaluation | `false` |
| `KHORA_HOOKS_LLM_BATCH_SIZE` | Pairs per LLM call | `10` |
| `KHORA_HOOKS_LLM_BATCH_FLUSH_MS` | Max ms to wait before flushing a partial batch | `100.0` |
| `KHORA_HOOKS_LLM_MAX_TOKENS_PER_NAMESPACE_PER_HOUR` | Per-namespace hourly token cap | `10000` |
| `KHORA_HOOKS_LLM_MAX_TOKENS_PER_SUBSCRIPTION_PER_HOUR` | Per-subscription hourly token cap (0 = disabled) | `0` |
| `KHORA_HOOKS_LLM_CACHE_SIZE` | Decision-cache LRU capacity (0 disables) | `2048` |
| `KHORA_HOOKS_LLM_CACHE_TTL_SECONDS` | Decision-cache entry TTL (0 = no expiry) | `3600.0` |

Per-filter model override:

```python
SemanticFilter(
    name="custom",
    description="...",
    filter_model="claude-haiku-4-5-20251001",  # override global model
)
```

## API

### `kb.subscribe(event_type, callback, filter=None)`

Register a callback for an event type. Returns a subscription ID.

### `kb.unsubscribe(subscription_id)`

Remove a subscription.

### `kb.hooks`

Access the underlying `HookDispatcher` for advanced usage (e.g., `kb.hooks.clear()`).

### Persistent subscriptions

In-process callbacks (`kb.subscribe`) live only for the duration of
the process. For subscriptions that must survive restarts, use the
persistent API. Persistent subscriptions store their delivery
configuration in PostgreSQL (`khora_hook_subscriptions` table, added
in migration 049) and are reloaded automatically on the next
`Khora.connect()` via `_wire_persistent_hooks`.

```python
sub_id = await kb.subscribe_persistent(
    "entity.created",
    delivery={"type": "webhook", "url": "https://my-service/hooks"},
    filter=SemanticFilter(name="strategic", description="strategic events"),
    namespace_id=my_ns_id,  # optional - scope to one namespace
)

# Later, to remove:
await kb.unsubscribe_persistent(sub_id)
```

Key points:

- Requires a SQL backend (any PostgreSQL-backed stack). Raises
  `RuntimeError` on store-less stacks.
- Delivery targets are opaque dicts; interpreting them is the
  responsibility of the hook dispatcher's delivery backend (webhook
  dispatch, queue publish, etc.).
- The `filter` and `namespace_id` arguments follow the same semantics
  as `kb.subscribe`.
- On restart, khora reloads persistent subscriptions before the first
  `remember` / `recall` call, so no events are missed after a clean
  reconnect.

## Co-occurrence filtering

The Level 0 `match` DSL operates on a single event's `event.data` payload, so it cannot express "alert when X and Y appear in the same chunk" with the per-entity `entity.created` event - each `entity.created` carries one entity at a time. Subscribe to **`chunk.entities_resolved`** instead: that event fires once per chunk after every entity from that chunk has been resolved, carrying the full set under `event.data["entity_ids"]` and `event.data["entity_names_by_type"]`.

Because the `match` DSL deliberately does **not** support nested dot-notation (see [Structural matching](#level-0-structural-matching-match-dsl)), co-occurrence checks that need to look into the `entity_names_by_type` dict run as a custom callback rather than a `match` pattern:

```python
async def flag_acme_security_cooccurrence(event):
    by_type = event.data.get("entity_names_by_type", {})
    persons = set(by_type.get("PERSON", []))
    concepts = set(by_type.get("CONCEPT", []))
    if "Acme" in persons and any("security" in c.lower() for c in concepts):
        await alert(event.data["chunk_id"])

kb.subscribe("chunk.entities_resolved", flag_acme_security_cooccurrence)
```

For simpler "any entity of type X appears in a chunk that also has entity name Y" checks, you can still narrow with a `match` pattern on a flat field - e.g. matching on `entity_count` or `chunk_id` - and do the type-set check inside the callback. The dispatcher only delivers events that pass the `match` cascade, so the callback runs against a pre-filtered stream.

## Related Documentation

- [Extraction Pipeline](../extraction/ingestion-pipeline.md) - where hooks fire during ingestion
- [Event Sourcing](../architecture/event-sourcing.md) - the event model hooks build on
- [Telemetry contract](../telemetry-contract.json) - the canonical list of `khora.hooks.*` metric names (`evaluations_total`, `tokens_total`, `throttled_total`, `cache_hits_total`, `cache_misses_total`)
