# OpenAI Agents SDK + khora example

Demonstrates the three khora primitives the adapter exposes against the
OpenAI Agents SDK: `KhoraSession` (SessionABC), `khora_recall_tool`
(FunctionTool factory), and `KhoraMemoryHooks` (RunHooks-shaped).

The example runs with no infrastructure and no API keys:

- `examples._helpers.embedded_khora` provides an in-memory `sqlite_lance`
  khora in a tmp directory.
- `examples._helpers.install_mock_llm` patches `litellm.acompletion`
  and `litellm.aembedding` with deterministic stubs.

It deliberately exercises only the construction surface of the recall
tool and hooks (no `Runner.run` is invoked) - that would require a live
LLM. Session writes are kept to one item so the example finishes well
under the 30s CI smoke budget.

## Run it

```bash
uv sync
uv run python example.py
```

Expected output:

```
Session has 1 item(s); latest: 'We picked PostgreSQL for the user DB.'
Built recall tool: name='recall_memory'
Built memory hooks: app_id='example'
```

## See also

- https://docs.deyta.ai/khora/integrations/openai-agents - the published
  quickstart, mirroring this directory's `example.py`.
- `src/khora/integrations/openai_agents/` - adapter source.
