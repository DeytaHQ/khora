# Failure-observability contract (ADR-001)

Khora has a recurring shape of bug: a function catches an exception,
logs at DEBUG, returns a default, and the caller has no machine-readable
way to tell that the result is degraded. By the time someone notices
("our recall is empty / our scoring looks wrong"), the original exception
is gone from the logs.

This document describes the convention every new khora module that
catches-and-degrades MUST follow. Existing modules are migrated
incrementally; the chronicle engine recall path (PR #901, PR #906) is
the reference implementation.

## The three metadata keys

A function that returns a result with an observability dict (typically
`metadata` or `engine_info`) SHOULD attach one or more of these keys
when it knowingly took a non-happy path:

```python
metadata["degradations"]: list[Degradation]   # fallback path taken / partial data, still returned a result
metadata["errors"]:       list[ErrorRecord]   # exception caught and swallowed, still returned a result
metadata["skipped"]:      list[SkipReason]    # op kind deliberately not run for a declared reason
```

All three are `TypedDict`s defined in `khora.core.diagnostics`. All keys
are optional; consumers SHOULD treat a missing key as an empty list. No
new dataclass is introduced - the architect is waiting until 6+ working
examples land before considering a typed `Diagnostics` wrapper.

### Schemas

```python
class Degradation(TypedDict, total=False):
    component: str               # "chronicle.bm25", "chronicle.temporal_channel", ...
    reason: str                  # low-cardinality enum: "channel_exception", "events_query_failed", ...
    detail: str | None           # free-form one-line explanation
    exception: str | None        # type(exc).__name__ when caused by a caught exception

class ErrorRecord(TypedDict, total=False):
    component: str
    reason: str
    exception: str               # always present
    detail: str | None

class SkipReason(TypedDict, total=False):
    op_kind: str                 # e.g. "community_summary", "entity_inference"
    reason: str                  # "op_not_supported_by_engine", "no_candidates", ...
    detail: str | None
```

`SkipReason` matches the pre-existing shape of
`DreamResult.metadata["skip_reasons"]` from #880 so dream entries are
valid `SkipReason` values without translation.

## Logging level convention

| Severity                             | Level     |
| ------------------------------------ | --------- |
| Degradation (fallback path taken)    | `WARNING` |
| Error (caught and swallowed)         | `ERROR`   |
| Skip (declared, expected to be rare) | `INFO`    |
| Skip (declared, expected on cold-start, e.g. empty namespace) | `DEBUG` |

When a degradation is caused by a caught exception, the log call SHOULD
pass `exc_info=True` so the traceback survives.

## Metric naming

Engines that want a counter for degradations follow this shape:

```
khora.{engine}.{component}.degraded_total{channel, reason}
```

- `engine`: chronicle, vectorcypher, ...
- `component`: usually the same tail token as in the `Degradation`
  entry (`channel.degraded_total{channel="bm25"}` matches
  `component="chronicle.bm25"`).
- `reason`: low-cardinality enum string.
- **No `namespace_id` label**. See the cardinality rule in
  `docs/telemetry-contract.md`.

The metric MUST be declared in `docs/telemetry-contract.json` (the
drift gate in `tests/unit/telemetry/test_contract.py` will reject the
PR otherwise).

## Reference implementations

### #871 - storage partial-failure counter

The coordinator's cross-store write helpers emit per-operation
`partial_failure` counters when one backend succeeds and another
fails (cross-store divergence), so callers see "success" but the
stores are not in sync. Four counters cover the four write paths:

- `khora.storage.create_entity.partial_failure` - entity creation
- `khora.storage.update_entity.partial_failure` - entity update
- `khora.storage.upsert_entities_batch.partial_failure` - batch upsert
- `khora.storage.replace_document.partial_failure` - document replace

### #880 - `DreamResult.metadata["skip_reasons"]`

`khora.dream.engines.registry._resolve_op_kinds` populates
`skip_reasons` when a planner-requested op kind is not in the active
engine plugin's `dream_capabilities`. Each entry is shaped
`{"op_kind", "reason", "detail"}` - the original prior art for the
`SkipReason` TypedDict. Reasons used today: `op_not_supported_by_engine`,
`no_candidates`, `op_disabled_at_runtime`, `guardrail_tripped:<which>`,
`backend_unsupported`, `schema_drift` (vectorcypher schema drift op skipped),
`abstention_drift` (chronicle abstention drift op skipped).

### #901 + #906 - chronicle channel-failure observability (this PR)

`ChronicleEngine.recall()` initializes a local `degradations: list[Degradation]`,
threads it through `_temporal_channel` and `_temporal_channel_chunks_fallback`,
and attaches the populated list to `RecallResult.engine_info["degradations"]`.

There are ~25 `_record_channel_degradation` call sites across eight component
groups. The shared counter is `khora.chronicle.channel.degraded_total{channel, reason}`.

| Component group                 | `component` prefix              | Representative reasons                                  |
| ------------------------------- | ------------------------------- | ------------------------------------------------------- |
| Events channel                  | `chronicle.events`              | `channel_exception`                                     |
| Facts channel                   | `chronicle.facts`               | `channel_exception`                                     |
| BM25 channel                    | `chronicle.bm25`                | `fulltext_backend_unavailable`, `channel_exception`     |
| Temporal channel                | `chronicle.temporal_channel`    | `events_query_failed`, `cosine_batch_failed`, `chunk_fetch_failed`, `chunk_fallback_failed`, `events_no_usable_signal` |
| Entity channel                  | `chronicle.entity`              | `channel_exception`                                     |
| Cross-session channel           | `chronicle.cross_session`       | `channel_exception`                                     |
| Doc hydration                   | `chronicle.doc_hydration`       | `channel_exception`                                     |
| Namespace overrides             | `chronicle.namespace_overrides` | `channel_exception`                                     |

The "no events yet" path is deliberately NOT recorded - that is the
expected cold-start condition for a namespace, not a degradation.

## Live degradation channel inventory

All channels that currently emit `degraded_total` counters or attach
`Degradation` entries to results, as of v0.21. Update this table when
adding a new channel.

| Counter / channel                                      | Attach point                           | Notes                                           |
| ------------------------------------------------------ | -------------------------------------- | ----------------------------------------------- |
| `khora.chronicle.channel.degraded_total`               | `RecallResult.engine_info`             | All chronicle channel failures (8 groups above) |
| `khora.vectorcypher.recency_channel.degraded_total`    | `RecallResult.engine_info`             |                                                 |
| `khora.vectorcypher.version_filter.degraded_total`     | `RecallResult.engine_info`             | Embedded PIT recall degrades here               |
| `khora.vectorcypher.rel_fetch.degraded_total`          | `RecallResult.engine_info`             |                                                 |
| `khora.vectorcypher.cypher_expand.degraded_total`      | `RecallResult.engine_info`             |                                                 |
| `khora.vectorcypher.entity_vector_search.degraded_total` | `RecallResult.engine_info`           |                                                 |
| `khora.vectorcypher.bm25.degraded_total`               | `RecallResult.engine_info`             |                                                 |
| `khora.vectorcypher.community_projection.degraded_total` | `RecallResult.engine_info`           |                                                 |
| `khora.vectorcypher.chunk_mirror.degraded_total`       | `RecallResult.engine_info`             |                                                 |
| `khora.vectorcypher.temporal_semantic_fallback.degraded_total` | `RecallResult.engine_info`   |                                                 |
| `khora.query.hyde.degraded_total`                      | `RecallResult.engine_info`             | HyDE expansion failures                         |
| `extraction.llm.second_pass` (no dedicated counter)    | `RememberResult` / `ExtractionResult.metadata` | Batched relationship second-pass failure (#1412); reason `second_pass_failed` |
| `khora.dream.graph_mirror.partial_failure`             | `DreamResult.metadata`                 | Post-commit Neo4j mirror failures               |
| `khora.dream.graph_unmirror.partial_failure`           | `DreamResult.metadata`                 | Tombstone un-mirror failures                    |
| `khora.forget.cascade.degraded_total`                  | `RememberResult.metadata` or log only  |                                                 |
| `khora.documents.processor.degraded_total`             | `RememberResult.metadata`              |                                                 |
| `khora.hooks.subscription.persist_degraded_total`      | log only                               | Hook dispatcher persist failures                |
| `khora.storage.create_entity.partial_failure`          | counter only                           | Cross-store divergence on create                |
| `khora.storage.update_entity.partial_failure`          | counter only                           | Cross-store divergence on update                |
| `khora.storage.upsert_entities_batch.partial_failure`  | counter only                           | Cross-store divergence on batch upsert          |
| `khora.storage.replace_document.partial_failure`       | counter **and** `Degradation` dicts    | Cross-store divergence on replace. The #1430 replace-mirror reconcile drain reuses this counter and also returns `Degradation` dicts (component `coordinator.replace_mirror.reconcile`; reasons `graph_mirror_pending_read_failed`, `graph_mirror_reconcile_failed`, `graph_mirror_pending_clear_failed`) |

## Where to attach the list

| Result type                                   | Container               |
| --------------------------------------------- | ----------------------- |
| `DreamResult`                                 | `result.metadata`       |
| `RecallResult` (chronicle, vectorcypher, ...) | `result.engine_info`    |
| `RememberResult`                              | `result.metadata`       |
| Future result types                           | `result.metadata`       |

The chronicle engine uses `engine_info` because that is its established
free-form telemetry dict. The convention does not require introducing
a new field - reuse whatever observability dict the result already
carries. The test helper `tests/test_helpers/diagnostics.py`
(`assert_no_silent_degradation`) inspects both `metadata` and
`engine_info` so tests stay portable across result shapes.

## CLAUDE.md sections to update

When adopting this convention on a new module, add a one-line entry to
the relevant `Gotchas` subsection:

- **Extraction & Search** for chronicle / vectorcypher / pipeline degradations.
- **Backend Specifics** for storage-layer degradations (#871).
- **Telemetry** for any new `degraded_total` metric you declare.

## Testing

Use `tests/test_helpers/diagnostics.py::assert_no_silent_degradation(result)`
as a default assertion in any happy-path test:

```python
from tests.test_helpers.diagnostics import assert_no_silent_degradation

result = await engine.recall(...)
assert_no_silent_degradation(result)
assert len(result.chunks) == 10
```

The helper raises when `degradations` or `errors` is non-empty. It
deliberately tolerates `skipped` entries: skipping is a declared
choice, not a silent failure.

## Stability

The TypedDicts in `khora.core.diagnostics` are part of the public
`khora.core` surface (re-exported via `khora.core.__all__`). Breaking
changes to the field names will be recorded in `CHANGELOG.md`.
Individual `reason` strings are NOT part of the public contract - they
may be renamed within the low-cardinality enum without a major bump,
because callers should be filtering on `component` for routing and
treating `reason` as a human-readable label.
