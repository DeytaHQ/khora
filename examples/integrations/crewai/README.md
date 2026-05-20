# CrewAI integration example

Smoke example for `khora.integrations.crewai`. Constructs a
`crewai.Memory` backed by an in-memory sqlite_lance khora, saves two
records, and runs a recall - no external services or API keys required.

Run it with:

```
uv run python example.py
```

The `examples-smoke` CI job runs this file under a 30s timeout. The
`docs/integrations/crewai.md` quickstart snippet is byte-identical to
`example.py` (gated by `tools/check_examples_drift.py`).
