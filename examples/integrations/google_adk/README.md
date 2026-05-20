# Google ADK + khora example

Smoke example for `khora.integrations.google_adk`. Wraps khora as a
`google.adk.memory.BaseMemoryService` (`KhoraMemoryService`), feeds it
a two-turn synthetic `Session`, and reads back the indexed memories via
`search_memory` - no external services, no Google Cloud project, no
API keys required.

The example runs against an in-memory `sqlite_lance` khora; the mock
LLM helper patches `litellm.acompletion` / `litellm.aembedding` so the
extraction/embedding pipeline is hermetic.

## Run it

```bash
uv sync
uv run python example.py
```

## See also

- `docs/integrations/google_adk.md` - quickstart byte-identical to this
  directory's `example.py` (CI enforces drift via
  `tools/check_examples_drift.py`).
- `src/khora/integrations/google_adk/` - adapter source.
