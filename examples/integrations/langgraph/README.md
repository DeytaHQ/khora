# LangGraph + khora example

Demonstrates wiring khora's `KhoraStore` into a LangGraph `StateGraph`
as the long-term memory store. A single-node graph writes a memory; the
example then reads it back through the same store handle.

The example runs with no infrastructure and no API keys:

- `examples._helpers.embedded_khora` provides an in-memory `sqlite_lance`
  khora in a tmp directory.
- `examples._helpers.install_mock_llm` patches `litellm.acompletion`
  and `litellm.aembedding` with deterministic stubs.

## Run it

```bash
uv sync
uv run python example.py
```

Expected output:

```
Stored memory: 'the sky is blue today'
Namespaces in store: [('memories',)]
```

## See also

- https://docs.deyta.ai/khora/integrations/langgraph - the published
  quickstart, mirroring this directory's `example.py`.
- `src/khora/integrations/langgraph/` - adapter source.
