# Per-provider LLM config

Each file here is the **`llm:` block** of a Khora config — the fields of
`KhoraConfig.llm` (`LLMSettings`). Pick the provider you use and drop the block
under `llm:` in your `khora.yaml`:

```yaml
# khora.yaml
llm:
  model: gpt-4o-mini          # ← from openai.yaml
  api_key_env: OPENAI_API_KEY
  # ...
storage:
  backend: sqlite_lance       # or postgres + neo4j
```

Then `KhoraConfig.from_yaml("khora.yaml")`. To check a block on its own:

```python
import yaml
from khora.config import LLMSettings

LLMSettings.model_validate(yaml.safe_load(open("openai.yaml")))
```

## Files

| File | Chat model | Embeddings | Env vars |
|------|-----------|------------|----------|
| `openai.yaml` | `gpt-4o-mini` | OpenAI `text-embedding-3-small` | `OPENAI_API_KEY` |
| `claude.yaml` | `claude-sonnet-4` | OpenAI (Anthropic has none) | `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` |
| `gemini.yaml` | `gemini/gemini-2.0-flash` | OpenAI | `GEMINI_API_KEY`, `GOOGLE_API_KEY`, `OPENAI_API_KEY` |

`embedding_dimension` is `1536` in every file because the Postgres backend ships
a 1536-wide vector column; keep it at `1536` unless you've sized the column for a
different embedding model.

When the chat model and the embedding model are on different providers (Claude,
Gemini), Khora resolves each provider's key from the environment per call —
`api_key_env` names the **chat** model's key, and the embedding provider's key is
read from the environment, so export both.
