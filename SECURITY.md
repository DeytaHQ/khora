# Security Policy

khora is a Python library for agentic memory. It runs as part of the host
application's process - there is no server, no listening socket, no
multi-tenant boundary inside khora itself. The threat model is **library
embedded in a trusted host**: secrets, network calls, and database
credentials all live in the host's environment.

This document covers what counts as a vulnerability, how to report one,
and what to expect afterwards.

## Reporting a vulnerability

**Report privately.** Do not open a public GitHub issue, PR, or
discussion thread for a suspected vulnerability.

Use **[GitHub Private Vulnerability Reporting](https://github.com/DeytaHQ/khora/security/advisories/new)**.
That route is end-to-end private, assigns a CVE if warranted, and gives
us a coordinated-disclosure track.

If for any reason you can't use the GitHub route, email
[security@deytahq.com](mailto:security@deytahq.com) with the subject
line `[khora security]` and a short description; we'll move the
discussion to a private channel.

Please include, where possible:

- Affected version(s) (the output of `pip show khora`).
- The shortest reproducer you can produce (a Python snippet, a config
  fragment, or a minimal docker-compose).
- The impact you observed and the impact you believe is possible.
- Whether the issue affects production-supported configurations (see
  below) or only experimental extras.
- Whether you intend to publish (and on what timeline) so we can
  coordinate disclosure.

## What we treat as a vulnerability

In scope, with priority:

- **Credential leakage.** A path through khora's public API that
  exposes a `SecretStr`, database URL, OAuth token, or API key (in
  exceptions, logs, telemetry attributes, returned objects, repr, or
  pickle output).
- **Unauthenticated SQL/Cypher/SurrealQL injection.** Any code path
  where caller-supplied data lands unsanitized inside a SQL,
  Cypher, or SurrealQL fragment - even when the surrounding test
  context "trusts" the caller.
- **Path traversal.** Caller-supplied paths that resolve outside the
  intended directory (file sink for dream-phase reports, on-disk
  Lance store paths, embedded SQLite database paths).
- **Deserialization of attacker-controlled bytes** with side effects
  beyond returning a parsed dataclass.
- **Cross-tenant data leakage** in a single-process khora instance -
  any path where data ingested under namespace A surfaces in a recall
  under namespace B, given correct caller code.
- **Vulnerabilities in vendored or pinned dependencies** that khora's
  configuration leaves exploitable (we triage these via the pip-audit
  job in CI and respond to advisories that affect a default install).

Out of scope (please don't report these to security@):

- **Issues that require an attacker on the host.** If the attacker can
  already read your environment, your code, or your filesystem, they
  don't need a khora bug.
- **Bugs in the host's LLM provider or in the model itself.** Prompt
  injection that fools an LLM into surfacing data that khora correctly
  retrieved is an application-design problem upstream of khora;
  evaluate it in your prompt scaffolding and recall-filter logic.
- **Misconfiguration.** A `KHORA_DATABASE_URL` left as plaintext in a
  committed file is a host-side hygiene problem. We do enforce
  `pydantic.SecretStr` on credential fields so the repr is safe, but
  storage and rotation are the host application's responsibility.
- **Denial of service from oversized input** at the public API.
  khora's input bounds (chunk size, embedding dimension, plan size)
  are caller-controlled by design; if you set them unreasonably large
  and the process OOMs, that's the caller's contract.
- **Bugs in optional integrations** (`khora[crewai]`, `khora[langgraph]`,
  `khora[google-adk]`, `khora[openai-agents]`, `khora[llamaindex]`)
  that originate in the upstream framework. Report those to the
  framework's maintainers; cc us if khora's adapter layer materially
  contributes.

## Supported versions

Khora is pre-1.0. We support **the latest minor release plus the
one immediately preceding it**.

| Version | Supported | Notes |
|---------|-----------|-------|
| `0.15.x` | ✅ current minor | Active security and bug fixes |
| `0.14.x` | ✅ previous minor | Security fixes only |
| `< 0.14` | ❌ | Please upgrade |

Patch releases land on the latest minor only. We will issue a backport
to the previous minor when a fix is genuinely security-relevant (CVE
issued or material exposure).

Production-supported configurations:

- **PostgreSQL + pgvector + Neo4j** (the default VectorCypher engine).
- **PostgreSQL + pgvector** (the Chronicle engine, graph-less).

Experimental configurations - fixes are best-effort, security severity
is judged in context:

- `khora[sqlite-lance]` (embedded).
- `khora[surrealdb]` (unified).
- `khora[weaviate]`.
- `khora[age]`.

## Response targets

These are commitments we work to, not strict SLAs. The numbers
assume reports filed through the GitHub Private Vulnerability route
or the security email.

| Stage | Target |
|-------|--------|
| Acknowledgement of receipt | 2 business days |
| Initial triage (severity + scope confirmed) | 5 business days |
| Fix or risk-accepted decision | 30 days for high/critical, 90 days for medium/low |
| Coordinated disclosure window | 90 days from triage, extendable by mutual agreement |

For widely-exploited issues we will move faster than the targets above
and may publish a security advisory before the full patch ships.

## What khora already does

These are existing controls in the repository - pointers for
researchers and integrators evaluating khora.

- **Credential fields are `pydantic.SecretStr`.** Database URLs,
  passwords, and API keys never appear in `repr()`, JSON dumps, or
  config-dump output. Call `.get_secret_value()` to retrieve
  plaintext at the boundary.
- **Telemetry attributes use `khora.telemetry.bounded_text_hash`** for
  free-text values (raw queries, document text, chunk content). Spans
  never carry raw user content as an attribute. See
  [`docs/telemetry-contract.md`](docs/telemetry-contract.md).
- **Cardinality rule** prevents `namespace_id` from becoming a metric
  label - metric series stay bounded even with millions of tenants.
- **Pre-commit secret-typing check** (`.semgrep.yml`) flags
  `str`-typed fields in Pydantic models / dataclasses that look like
  secrets, so a casual rename can't accidentally widen the secret
  surface.
- **Dependency scanning.** Every PR runs `pip-audit` in CI against
  the resolved dependency graph; advisories with no exploitable path
  are documented inline (see `.github/workflows/ci.yml`).
- **No raw-string SQL/Cypher composition** in the public API.
  Parameterized queries throughout SQLAlchemy and the Neo4j driver;
  HyDE-Cypher template slot values are validated against an
  `ExpertiseConfig`-derived whitelist before binding.
- **bi-temporal soft-delete** for `relationships`, `memory_facts`, and
  (since v0.15) `chronicle_events`. Mutation history is preserved;
  deletes are reversible within the retention window via the
  dream-phase `undo.json` artifacts.

## Acknowledgements

We credit researchers in the security advisory associated with each
fix (with your name, alias, or anonymous - your choice). We do not
operate a paid bug-bounty program.

## Versioning of this policy

This file is checked into the repository. If you spot something
out-of-date (a project URL change, a contact path that no longer
works, a supported-version row that hasn't been updated for a
release), open a regular issue - that part is not sensitive.
