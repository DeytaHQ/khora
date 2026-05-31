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

Pgvector / SurrealDB upsert paths emit
`khora.storage.entity.upsert.partial_failure_total{kind, field}` when
a batched entity upsert truncates provenance lists silently. The
metric is the only signal that the truncation happened; the per-row
data is still written, so callers see "success" without knowing the
provenance arrays were capped.

### #880 - `DreamResult.metadata["skip_reasons"]`

`khora.dream.engines.registry._resolve_op_kinds` populates
`skip_reasons` when a planner-requested op kind is not in the active
engine plugin's `dream_capabilities`. Each entry is shaped
`{"op_kind", "reason", "detail"}` - the original prior art for the
`SkipReason` TypedDict. Reasons used today: `op_not_supported_by_engine`,
`no_candidates`, `op_disabled_at_runtime`, `guardrail_tripped:<which>`,
`backend_unsupported`.

### #901 + #906 - chronicle channel-failure observability (this PR)

`ChronicleEngine.recall()` initializes a local `degradations: list[Degradation]`,
threads it through `_temporal_channel` and `_temporal_channel_chunks_fallback`,
and attaches the populated list to `RecallResult.engine_info["degradations"]`.

Six call sites record entries:

| Site                                      | `component`                     | `reason`                       |
| ----------------------------------------- | ------------------------------- | ------------------------------ |
| BM25 task `RuntimeError`                  | `chronicle.bm25`                | `fulltext_backend_unavailable` |
| BM25 task other exception                 | `chronicle.bm25`                | `channel_exception`            |
| Semantic/temporal/entity gather exception | `chronicle.<channel>`           | `channel_exception`            |
| `query_events` raised                     | `chronicle.temporal_channel`    | `events_query_failed`          |
| Cosine batch raised                       | `chronicle.temporal_channel`    | `cosine_batch_failed`          |
| `get_chunks_batch` raised                 | `chronicle.temporal_channel`    | `chunk_fetch_failed`           |
| Chunk-fallback `search_similar_chunks` raised | `chronicle.temporal_channel` | `chunk_fallback_failed`       |
| Events exist but no usable scores         | `chronicle.temporal_channel`    | `events_no_usable_signal`      |

Each site also bumps `khora.chronicle.channel.degraded_total{channel, reason}`.
The "no events yet" path is deliberately NOT recorded - that is the
expected cold-start condition for a namespace, not a degradation.

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
