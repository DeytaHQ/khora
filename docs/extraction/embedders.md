# Embedders

Embedders generate vector representations of text for semantic similarity search. Khora uses LiteLLM for unified access to multiple embedding providers.

## Overview

Embeddings enable:
- **Semantic search**: Find content by meaning, not just keywords
- **Entity matching**: Identify similar entities for deduplication
- **Query understanding**: Match queries to relevant content

## LiteLLMEmbedder

The primary embedder uses LiteLLM for provider-agnostic embedding generation.

### Configuration

```python
from khora.extraction.embedders import LiteLLMEmbedder

embedder = LiteLLMEmbedder(
    model="text-embedding-3-small",  # OpenAI default
    dimension=1536,
    timeout=30,          # Request timeout (seconds)
    max_retries=3,       # Retry count on failure
    batch_size=100,      # Max texts per batch
)
```

### From LiteLLM Config

```python
from khora.config import LiteLLMConfig

config = LiteLLMConfig(
    embedding_model="text-embedding-3-small",
    embedding_dimension=1536,
    timeout=30,
    max_retries=3,
)

embedder = LiteLLMEmbedder.from_config(config)
```

## Embedding Generation

### Single Text

```python
embedding = await embedder.embed("Hello, world!")
# Returns: list[float] with 1536 dimensions
```

### Batch Embedding

```python
texts = [
    "First document content",
    "Second document content",
    "Third document content",
]

embeddings = await embedder.embed_batch(texts)
# Returns: list[list[float]], one embedding per text
```

### Automatic Batching

Large lists are automatically split into batches:

```python
# 500 texts with batch_size=100
# Internally splits into 5 batches
embeddings = await embedder.embed_batch(large_text_list)
```

## Supported Providers

LiteLLM supports multiple embedding providers:

### OpenAI

```python
embedder = LiteLLMEmbedder(
    model="text-embedding-3-small",  # Recommended
    dimension=1536,
)

# Other OpenAI models
# - text-embedding-3-large (3072 dimensions)
# - text-embedding-ada-002 (legacy, 1536 dimensions)
```

### Cohere

```python
embedder = LiteLLMEmbedder(
    model="cohere/embed-english-v3.0",
    dimension=1024,
)
```

### Voyage

```python
embedder = LiteLLMEmbedder(
    model="voyage/voyage-02",
    dimension=1024,
)
```

### Self-Hosted

```python
embedder = LiteLLMEmbedder(
    model="ollama/nomic-embed-text",
    dimension=768,
)
```

## Retry Logic

Embedders implement exponential backoff:

```python
async def _embed_batch_internal(self, texts: list[str]) -> list[list[float]]:
    for attempt in range(self._max_retries):
        try:
            response = await litellm.aembedding(
                model=self._model,
                input=texts,
                timeout=self._timeout,
            )
            return [item["embedding"] for item in response.data]
        except Exception as e:
            if attempt < self._max_retries - 1:
                wait_time = 2 ** attempt  # 1s, 2s, 4s
                await asyncio.sleep(wait_time)
            else:
                raise
```

## Environment Variables

Set provider API keys:

```bash
# OpenAI
export OPENAI_API_KEY=sk-...

# Cohere
export COHERE_API_KEY=...

# Voyage
export VOYAGE_API_KEY=...
```

## Dimension Matching

Ensure embedding dimension matches your pgvector schema:

```sql
-- Create table with matching dimension
CREATE TABLE chunk_embeddings (
    id UUID PRIMARY KEY,
    embedding vector(1536),  -- Must match embedding_dimension
    ...
);
```

Khora uses 1536 dimensions by default (OpenAI text-embedding-3-small).

## Performance Considerations

### Batch Size

```python
# Default: 100 texts per batch
batch_size = 100

# Larger batches: fewer API calls, more memory
batch_size = 500

# Smaller batches: more API calls, less memory
batch_size = 25
```

### Concurrency

Embedding is parallelized at the document level:

```python
# Each document's chunks are embedded together
chunks = await embed_chunks(document_chunks, model=...)

# Multiple documents embed in parallel
await asyncio.gather(*[
    process_document(doc)  # Each calls embed_chunks
    for doc in documents
])
```

## API Usage

### Via MemoryLake

```python
result = await lake.remember(
    content,
    embedding_model="text-embedding-3-small",
)
```

### In Pipeline Tasks

```python
from khora.pipelines.tasks import embed_chunks

chunks = await embed_chunks(
    chunks,
    model="text-embedding-3-small",
)

# Chunks now have embedding field populated
for chunk in chunks:
    assert chunk.embedding is not None
    assert len(chunk.embedding) == 1536
```

### Direct Embedder Usage

```python
from khora.extraction.embedders import LiteLLMEmbedder

embedder = LiteLLMEmbedder(model="text-embedding-3-small")

# Embed query for search
query_embedding = await embedder.embed(query)

# Search similar chunks
results = await storage.search_similar_chunks(
    namespace_id,
    query_embedding,
    limit=10,
)
```

## Embedder Protocol

Custom embedders can implement the base protocol:

```python
from khora.extraction.embedders.base import Embedder

class Embedder(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get the embedding dimension."""
        ...

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate embedding for single text."""
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        ...
```

## Embedding Models Comparison

| Model | Provider | Dimensions | Speed | Quality |
|-------|----------|------------|-------|---------|
| text-embedding-3-small | OpenAI | 1536 | Fast | Good |
| text-embedding-3-large | OpenAI | 3072 | Medium | Better |
| embed-english-v3.0 | Cohere | 1024 | Fast | Good |
| voyage-02 | Voyage | 1024 | Medium | Better |

## Next Steps

- [Extractors](extractors.md) - Entity extraction
- [Chunkers](chunkers.md) - Text splitting
- [Query Engine](../query-engine/overview.md) - Semantic search
