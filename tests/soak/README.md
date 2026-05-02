# Soak / burn-in harness (DYT-3545)

Long-running ingest + recall workload that asserts release-blocking SLOs:

- **Memory ceiling:** RSS at end ≤ 1.5× steady-state RSS (warmup excluded).
- **Loguru queue health:** zero handler errors over the run (proxy for queue
  saturation; loguru 0.7.3's `multiprocessing.SimpleQueue` exposes no `qsize`).
- **Recall p95 drift:** final-window p95 ≤ 2× warmup-window p95.
- **Zero exceptions** in the workload loop.

## Run a 5-minute smoke (embedded, no Docker)

```bash
uv run pytest tests/soak/test_soak.py -m soak --no-cov
```

Drives the SQLite + LanceDB stack with a deterministic embedder stub —
no `OPENAI_API_KEY` required.

## Run the 4-hour release gate

```bash
KHORA_SOAK_DURATION_S=14400 uv run pytest tests/soak/test_soak.py -m soak --no-cov
```

This is **not** wired into PR CI. It runs as a manual `workflow_dispatch`
job (`.github/workflows/soak.yml`) before tagging a release.

## Drive the PostgreSQL + Neo4j stack

```bash
make dev   # spin up postgres + neo4j via compose.yaml
KHORA_SOAK_PG_URL=postgresql+asyncpg://khora:khora@localhost:5432/khora \
KHORA_GRAPH_URL=bolt://neo4j:khora@localhost:7687 \
KHORA_SOAK_DURATION_S=300 \
  uv run pytest tests/soak/test_soak.py::test_soak_postgres_neo4j -m soak --no-cov
```

## Output

Each run writes a JSON summary to `/tmp/khora-soak-{stack}-{ts}.json` and
prints a tabular summary to stdout. The JSON file is what the GitHub
Actions job uploads as a build artifact.
