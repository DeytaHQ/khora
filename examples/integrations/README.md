# Khora integration examples

This directory holds one runnable example per agentic-framework adapter. Each
adapter ships a self-contained scenario that proves the integration works end
to end, without any external service or API key.

## Convention

```
examples/integrations/<framework>/
├── README.md         # one paragraph: what the scenario demonstrates
├── pyproject.toml    # minimal: dependencies = ["khora[<framework>]", ...]
└── example.py        # < 80 LOC, exits 0, no API key required
```

Rules every example must follow:

- **No external services.** Use the in-memory `sqlite_lance` khora fixture
  from `examples/_helpers/khora_fixtures.py`. No Postgres, no Neo4j, no
  Docker - examples run on a fresh checkout with `uv run python example.py`.
- **No API keys.** Use the mock LLM from `examples/_helpers/mock_llm.py`
  (monkeypatches `litellm.acompletion` / `litellm.aembedding`). Embeddings
  are deterministic-by-text-hash; completions return a configurable stub.
- **Byte-identical doc snippets.** The `python title="example.py"` block in
  the matching `docs/integrations/<framework>.md` must equal the example
  file byte-for-byte. CI enforces this via `tools/check_examples_drift.py`.
- **30-second timeout.** Each example is smoke-tested under a 30s budget
  in the `examples-smoke` CI job. Wall-clock target is < 2 min for the
  whole loop across all shipping adapters.

## How CI gates this

The `examples-smoke` job in `.github/workflows/ci.yml` runs after `install`:

1. `python tools/check_examples_drift.py` - fails if any doc snippet diverges
   from its `example.py`.
2. For each `examples/integrations/<framework>/` directory, runs
   `uv sync` inside that dir (to pull the adapter's own extra) and then
   `uv run python example.py` under a 30s timeout.

The smoke loop exits 0 when no adapter directories exist yet (foundation
state). As soon as an adapter merges `examples/integrations/<adapter>/`,
its example is picked up automatically - no workflow change needed.

## Helpers

- `examples/_helpers/mock_llm.py` - `install_mock_llm(monkeypatch=None,
  responses=None, dim=1536)`: patches `litellm.acompletion` and
  `litellm.aembedding`. Embeddings hash-derived (SHA1 → seed → normalised
  vector). Completions cycle through `responses` (default `["stub
  response"]`).
- `examples/_helpers/khora_fixtures.py` - `embedded_khora(...)`: async
  context manager yielding a `Khora` bound to a temp-dir `sqlite_lance`
  backend with migrations applied. Zero infrastructure.
