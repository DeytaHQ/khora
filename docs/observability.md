# Observability

Khora emits OpenTelemetry spans and metrics through the OTel API.
Where they go — a collector, a vendor (Honeycomb, Datadog, New Relic,
Dynatrace), local Jaeger/Tempo, or nowhere — is determined by which
`TracerProvider` / `MeterProvider` is installed in the process.
Khora doesn't install one at import time.

Install paths:

| Combination | What it installs | What you get |
|---|---|---|
| `pip install khora` | OTel API only (small wheel) | Spans/metrics are silent no-ops. |
| `pip install khora[otel]` | OTel SDK + OTLP/HTTP exporter | Vanilla OTel. Honors `OTEL_*` env vars. |
| `pip install khora[otel-grpc]` | `khora[otel]` + OTLP/gRPC exporter | Use when your collector wants gRPC. |
| `pip install khora[logfire]` | [Logfire](https://logfire.pydantic.dev) — auto-bootstrap | One-call setup, vendor-managed backend. |

You can combine `khora[otel]` and `khora[logfire]` — the precedence rules
below decide which wins.

## Quick start: OTel Collector → Jaeger

The minimum five-minute recipe. Assumes you have a local Jaeger or Tempo
on `localhost:4318` (HTTP) and run khora in the same process as your app.

```bash
pip install 'khora[otel,sqlite-lance]'
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318"
export OTEL_EXPORTER_OTLP_PROTOCOL="http/protobuf"
export OTEL_SERVICE_NAME="my-app"             # YOUR app's identity, not khora
```

```python
import asyncio
from khora import Khora
from khora.telemetry import configure_telemetry

async def main():
    configure_telemetry()        # picks up OTEL_* env vars
    async with Khora() as kb:
        ns = await kb.create_namespace("demo")
        await kb.remember("Marie Curie won the Nobel Prize.", namespace=ns.namespace_id)

asyncio.run(main())
```

Open Jaeger → search service `my-app` — every `khora.recall`,
`khora.remember`, `khora.vectorcypher.*` span appears under the
`khora` instrumentation scope.

## Quick start: Logfire

```bash
pip install 'khora[logfire,sqlite-lance]'
export LOGFIRE_TOKEN=...
```

```python
import logfire
from khora import Khora

logfire.configure(service_name="my-app")    # installs the TracerProvider
# khora picks up the provider automatically — no configure_telemetry() needed.
```

## Configuration via environment

khora respects the standard
[OTel SDK environment variables](https://opentelemetry.io/docs/specs/otel/configuration/sdk-environment-variables/).
The SDK auto-reads most of them; khora reads a few directly. Operators
control everything via these variables — there is no `KHORA_OTEL_*`
shadow.

| Variable | Honored by | Notes |
|---|---|---|
| `OTEL_SERVICE_NAME` | SDK Resource | **You** set this for your service. khora never sets it. |
| `OTEL_RESOURCE_ATTRIBUTES` | SDK Resource | Comma-separated `k=v` pairs. |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP exporter | Where spans/metrics ship. |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | OTLP exporter | `http/protobuf` (default) or `grpc`. |
| `OTEL_EXPORTER_OTLP_HEADERS` | OTLP exporter | Comma-separated `k=v` for auth (URL-encoded values). |
| `OTEL_TRACES_SAMPLER` / `_ARG` | TracerProvider | e.g. `parentbased_traceidratio` + `0.1`. |
| `OTEL_BSP_SCHEDULE_DELAY` / `_MAX_*` | BatchSpanProcessor | Buffering and batch sizing. |
| `OTEL_SDK_DISABLED` | configure_telemetry | When `true`, khora skips bootstrap entirely. |
| `LOGFIRE_TOKEN` | configure_telemetry | When set + `logfire` importable, khora prefers logfire. |
| `KHORA_NEO4J_LOG_LEVEL` | install_neo4j_log_bridge | Routes neo4j driver DEBUG to the active log backend. |

Per-signal overrides (`OTEL_EXPORTER_OTLP_TRACES_*`,
`OTEL_EXPORTER_OTLP_METRICS_*`) take precedence over the generic
variables — useful when traces go to vendor A and metrics to vendor B.

## Programmatic configuration

Hosts that bootstrap their own `TracerProvider` (e.g. an app that wires
OTel for Django, FastAPI, etc. before importing khora) don't need to
call `configure_telemetry()` — khora detects the non-default global
provider and emits through it.

For scripts, notebooks, or services that want khora to drive the
setup, use `configure_telemetry()`:

```python
from khora.telemetry import configure_telemetry

handle = configure_telemetry(
    backend="otel",                                      # or "logfire", "auto", "none"
    endpoint="https://api.honeycomb.io",
    headers={"x-honeycomb-team": "..."},
    protocol="http/protobuf",
    resource_attributes={"team": "platform", "env": "prod"},
)
```

The handle exposes the resolved state:

```python
print(handle.backend)                            # "otel" / "logfire" / "none"
print(handle.khora_installed_tracer_provider)    # True iff khora called set_tracer_provider
print(handle.endpoint)                           # resolved OTLP endpoint
handle.shutdown()                                # flush + shutdown (only providers khora installed)
```

Library contract: **khora never sets `service.name`** on its Resource.
Service identity belongs to the host application. Pass it via
`OTEL_SERVICE_NAME` or include it in your own SDK init. `service.*`
keys in `resource_attributes=` are dropped with a warning.

khora identifies itself via the **instrumentation scope**:
`scope.name = "khora"`, `scope.version = importlib.metadata.version("khora")`.
That's the right slot for "which library produced this span" — your
dashboards can filter on `instrumentation_scope.name = khora` without
colliding with the operator's `service.name`.

## Precedence

`configure_telemetry()` walks this list and stops at the first match:

1. `backend="none"` — explicit no-op.
2. `OTEL_SDK_DISABLED=true` (env) — no-op.
3. Caller-supplied `tracer_provider=` / `meter_provider=` — install as
   global only if no real provider exists yet.
4. A non-default global `TracerProvider` is already installed — defer
   to it. (This is the "host app already configured OTel" path; same
   path applies if `logfire.configure()` already ran.)
5. `backend="logfire"` or (`backend="auto"` and `LOGFIRE_TOKEN` or
   `LOGFIRE_SEND_TO_LOGFIRE` env is set and `logfire` is importable) —
   call `logfire.configure()`.
6. `backend="otel"` or (`backend="auto"` and any `OTEL_*` env var is
   set) — bootstrap a vanilla OTel SDK with OTLP exporters.
7. Otherwise — no-op.

`configure_telemetry()` is idempotent. The first call's decision sticks
for the rest of the process.

## Public spans, metrics, and resource attributes

The complete contract lives at
[`docs/telemetry-contract.json`](telemetry-contract.json) (with explainer
at [`telemetry-contract.md`](telemetry-contract.md)). Items tagged
`stability: public` are part of khora's API surface and follow standard
semver — breaking changes require a major version bump. CI enforces
drift via `tests/unit/telemetry/test_contract.py`.

Highlights:

- **Public spans**: `khora.recall`, `khora.remember`, `khora.forget`,
  `khora.remember_batch`, `khora.vectorcypher.retrieve`,
  `khora.skeleton.{chunk,embed,batch_*}`,
  `khora.extraction.{llm_call,extract_entities}`,
  `khora.query.{embedding,graph_search,hyde,rerank}`,
  `khora.embedder.{api_call,litellm_request}`.
- **Public metrics**: `khora.memory.{recall,ingest}.duration`,
  `khora.llm.tokens`, `khora.llm.cost_usd`,
  `khora.neo4j.pool.{acquire_duration,timeout,connections.*,utilization}`,
  `khora.chronicle.abstention_signal`, `khora.log.queue.depth`.
- **Khora-contributed resource attribute**:
  `khora.telemetry.contract.version` — bumped alongside contract changes
  so dashboards can filter by schema version independently of khora's
  package version.

OTel semantic conventions apply to attributes: `gen_ai.*` for LLM
calls, `db.*` for storage backends, `code.*` for stack info.

## Sampling and cost control

The OTel SDK handles sampling transparently. Set
`OTEL_TRACES_SAMPLER=parentbased_traceidratio` and
`OTEL_TRACES_SAMPLER_ARG=0.1` to ship 10% of traces — khora needs no
code change. For high-volume operations, gate expensive attribute
computation on `span.is_recording()`:

```python
from khora.telemetry import bounded_text_hash, trace_span

with trace_span("khora.my_op") as span:
    if span.is_recording():                            # only when the span will actually export
        span.set_attribute("query_hash", bounded_text_hash(big_string))
```

**Cardinality rule**: never put a high-cardinality attribute (e.g.
`namespace_id`, `tenant_id`, `user_id`) on a *metric*. It's fine on a
span. Phase-0 audit measured ~438 distinct namespaces over the
production retention window in one deployment; Logfire and Prometheus
bill per series, so an unbounded label is an unbounded bill.

**Free-text rule**: pre-hash with `khora.telemetry.bounded_text_hash`
before setting any free-text value (raw user query, document content,
chunk text) as a span attribute. It returns a SHA1[:8] hash that
bounds cardinality and avoids leaking PII.

## Vendor recipes

### Honeycomb

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="https://api.honeycomb.io"
export OTEL_EXPORTER_OTLP_PROTOCOL="http/protobuf"
export OTEL_EXPORTER_OTLP_HEADERS="x-honeycomb-team=YOUR_API_KEY"
```

### Grafana Cloud (Tempo + Mimir)

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="https://otlp-gateway-prod-eu-west-0.grafana.net/otlp"
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Basic%20BASE64_CREDS"
```

### Datadog

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="https://trace.agent.datadoghq.com"
export OTEL_EXPORTER_OTLP_HEADERS="DD-API-KEY=YOUR_API_KEY"
```

### Local Jaeger / Tempo (docker)

```bash
docker run -d --name jaeger -p 16686:16686 -p 4318:4318 jaegertracing/all-in-one:latest
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318"
```

Open `http://localhost:16686` to browse spans.

## Migrating from `khora[logfire]`

No migration needed. `pip install khora[logfire]` keeps working. If
your app calls `logfire.configure()`, khora detects the resulting
`TracerProvider` and emits through it. You can install both extras
side-by-side — when `LOGFIRE_TOKEN` is set, the logfire path wins;
otherwise vanilla OTel takes over.

The one-line rename: `install_neo4j_logfire_handler` is now
`install_neo4j_log_bridge` (it still picks the logfire handler when
logfire is installed). The old name is kept as a deprecated alias for
khora 0.10.x and will be removed in 0.12.

## Troubleshooting

"I see no spans":

1. Run `khora.telemetry.diagnostics()` — prints the active provider
   class, whether khora bootstrapped it, the endpoint, and the
   resource attributes. The output is the first thing to share when
   filing a bug.
2. Check `OTEL_EXPORTER_OTLP_ENDPOINT` is reachable from the process.
3. Check `OTEL_TRACES_SAMPLER` isn't `always_off` or a zero ratio.
4. Check `OTEL_SDK_DISABLED` isn't set to `true`.
5. If you call `configure_telemetry()` *after* khora-importing code
   has already opened spans, those spans went to the proxy provider
   and were dropped. Move `configure_telemetry()` to process startup,
   before any khora call.

"Spans appear but `service.name` is wrong":

This is correctly your host application's concern — khora never sets
it. Either set `OTEL_SERVICE_NAME` or include `service.name` in your
own Resource when you bootstrap OTel manually.

## Telemetry Collector (structured event recording)

Separate from span/metric export, khora also writes structured
`LLMEvent` / `StorageEvent` / `PipelineEvent` rows to a dedicated
PostgreSQL database when `KHORA_TELEMETRY_DATABASE_URL` is set.
Useful for downstream cost tracking and incident reconstruction.
When the variable isn't set, a zero-cost `NoOpCollector` is used.
This collector is wired by `khora.telemetry.init_telemetry()`, which
is independent of `configure_telemetry()`.

## Async logging caveat

Library consumers that import khora without configuring loguru sinks
inherit loguru's default sync stderr sink, which blocks the event loop
on every log call inside `async def`. Either call
`khora.logging_config.setup_logging()` (which configures sinks with
`enqueue=True` and registers an `atexit` drain) or configure your own
loguru sinks with `enqueue=True` explicitly.
