# Embedders

Embeddings are what make semantic search work. An embedder converts text into vectors - lists of numbers that capture meaning. Similar concepts get similar vectors, enabling "find things like this" queries even when the exact words differ.

## What Embeddings Do

```
Text:    "Einstein developed the theory of relativity"
           |
           v
Embedder
           |
           v
Vector:  [0.021, -0.156, 0.089, 0.334, -0.027, ...]
         (1536 numbers that capture the meaning)
```

Now you can find similar content:

```
Query: "physicist's work on space-time"
  → similar vector
  → matches Einstein chunk
  → even though no words overlap!
```

## The LiteLLM Embedder

Khora uses LiteLLM for embedding, giving you access to multiple providers through one interface.

### Basic Setup

```python
from khora.extraction.embedders import LiteLLMEmbedder

embedder = LiteLLMEmbedder(
    model="text-embedding-3-small",  # Default: OpenAI
    dimension=1536                    # Output dimension
)
```

### From Configuration

```python
from khora.config import LiteLLMConfig

config = LiteLLMConfig(
    embedding_model="text-embedding-3-small",
    embedding_dimension=1536,
    timeout=30,
    max_retries=3
)

embedder = LiteLLMEmbedder.from_config(config)
```

## Generating Embeddings

### Single Text

```python
embedding = await embedder.embed("Hello, world!")
# Returns: [0.012, -0.034, 0.089, ...] (1536 floats)
```

### Batch (Efficient)

```python
texts = [
    "First document about machine learning",
    "Second document about neural networks",
    "Third document about data science"
]

embeddings = await embedder.embed_batch(texts)
# Returns: [[...], [...], [...]] (one embedding per text)
```

Batching is important - 100 individual API calls is much slower than 1 call with 100 texts.

### Automatic Batching

Large lists are automatically split:

```python
# 500 texts with batch_size=200
# → 3 API calls instead of 500
embeddings = await embedder.embed_batch(large_list)
```

## Supported Providers

### OpenAI (Default)

```python
# Recommended for most use cases
embedder = LiteLLMEmbedder(
    model="text-embedding-3-small",  # Fast, good quality
    dimension=1536
)

# Higher quality, more dimensions
embedder = LiteLLMEmbedder(
    model="text-embedding-3-large",
    dimension=3072
)
```

Set your API key:
```bash
export OPENAI_API_KEY=sk-...
```

### Cohere

```python
embedder = LiteLLMEmbedder(
    model="cohere/embed-english-v3.0",
    dimension=1024
)
```

```bash
export COHERE_API_KEY=...
```

### Voyage AI

```python
embedder = LiteLLMEmbedder(
    model="voyage/voyage-02",
    dimension=1024
)
```

```bash
export VOYAGE_API_KEY=...
```

### Local (Ollama)

```python
embedder = LiteLLMEmbedder(
    model="ollama/nomic-embed-text",
    dimension=768
)
```

No API key needed - just run Ollama locally.

## Model Comparison

| Model | Provider | Dimensions | Speed | Quality | Cost |
|-------|----------|------------|-------|---------|------|
| text-embedding-3-small | OpenAI | 1536 | Fast | Good | Low |
| text-embedding-3-large | OpenAI | 3072 | Medium | Better | Medium |
| embed-english-v3.0 | Cohere | 1024 | Fast | Good | Low |
| voyage-02 | Voyage | 1024 | Medium | Better | Medium |
| nomic-embed-text | Ollama | 768 | Varies | Good | Free |

**Recommendation:** Start with `text-embedding-3-small`. It's fast, cheap, and good enough for most use cases. Upgrade if you need better quality.

## Usage in Khora

### Via Khora

```python
# Uses configured default embedding model
await kb.remember(
    "Your content...",
    namespace=ns.namespace_id,
    entity_types=["PERSON", "ORG"],
    relationship_types=["MENTIONS"],
)
```

`embedding_model` isn't a per-call kwarg on `kb.remember()`. Configure
the embedding model globally via `KhoraConfig.llm.embedding_model` (or
env var `KHORA_LLM_EMBEDDING_MODEL`, or pass
`embedding_model="text-embedding-3-large"` to the `Khora(...)`
constructor).

### In Pipelines

```python
from khora.pipelines.tasks import embed_chunks

chunks = await embed_chunks(
    chunks,
    model="text-embedding-3-small"
)

# Chunks now have embeddings
for chunk in chunks:
    assert len(chunk.embedding) == 1536
```

### For Search

```python
# Embed query with same model as content
query_embedding = await embedder.embed("search query")

# Find similar chunks
results = await storage.search_similar_chunks(
    namespace_id,
    query_embedding,
    limit=10
)
```

## Token-Aware Truncation

The embedder automatically truncates texts that exceed the model's token limit before sending them to the API:

```python
# Handled automatically - no configuration needed
embedding = await embedder.embed(very_long_text)
# Text is truncated to the token limit if too long
```

**How it works:**
1. Encodes the text with `tiktoken` (when available) using the `cl100k_base` encoding
2. If the token count exceeds the model's limit (e.g., 8191 tokens for `text-embedding-3-small`), keeps the first `max_tokens` tokens and decodes them back to text (a plain token slice - no sentence or word boundary detection)
3. If `tiktoken` is unavailable, falls back to a conservative character slice (~3.5 chars/token)

This eliminates API errors from oversized inputs.

## Pre-Normalized Embeddings

All embeddings are **L2-normalized at ingest time**. This enables using dot product instead of cosine similarity for scoring:

```
dot_product(unit_vector_a, unit_vector_b) = cosine_similarity(a, b)
```

Dot product is ~3x faster than cosine similarity because it skips the norm computation step. The Rust acceleration layer provides `batch_dot_product` which takes advantage of this.

```python
from khora._accel import batch_dot_product

# Pre-normalized embeddings → dot product = cosine similarity
results = batch_dot_product(query_embedding, candidate_embeddings, threshold=0.3)
```

## Important: Dimension Matching

On Postgres the embedding column and its HNSW index are sized from
`llm.embedding_dimension` when a **fresh** database is migrated. The default
stays `1536`; other models work too — including `text-embedding-3-large` at its
full `3072` width, which previously failed on Postgres:

```python
KhoraConfig(
    storage={"backend": "postgres"},  # use_halfvec defaults to True
    llm={"embedding_model": "text-embedding-3-large", "embedding_dimension": 3072},
)
```

Constraints (enforced at config time by a Postgres-only guardrail):

- Dimensions above **2000** require **halfvec** (`storage.use_halfvec`, on by
  default), which raises the HNSW-indexable ceiling to **4000**. With halfvec
  disabled the ceiling is **2000**.
- Above 4000 (or above 2000 with halfvec off), request a shortened dimension via
  the model's `dimensions` parameter (e.g. `text-embedding-3-large` supports
  256–3072).

The dimension is **fixed at fresh-DB creation** — an existing, populated
database cannot be resized in place. To change embedding dimension:

1. Migrate a **fresh** database at the new dimension (or drop and re-create), and
2. Re-embed all content.

This is why choosing a model upfront matters.

## Error Handling

The embedder handles transient failures automatically:

```python
# Exponential backoff: 1s, 2s, 4s
max_retries = 3

# Custom timeout
timeout = 30  # seconds
```

If all retries fail, the exception propagates. Failed documents are marked as FAILED in the ingestion pipeline.

## Performance Tips

### Batch Size

```python
# Default: 200 texts per batch
embedder = LiteLLMEmbedder(batch_size=200)

# Larger = fewer API calls, more memory
embedder = LiteLLMEmbedder(batch_size=500)

# Smaller = more API calls, less memory
embedder = LiteLLMEmbedder(batch_size=25)
```

### Parallel Processing

Embedding happens in parallel at multiple levels:

```python
# Document level: multiple documents process concurrently
results = await asyncio.gather(*[
    process_document(doc)
    for doc in documents
])

# Sub-batch level: when a document has many chunks,
# sub-batches run concurrently instead of sequentially
embedder = LiteLLMEmbedder(
    batch_size=200,
    embed_concurrency=20,  # Up to 20 API calls in flight
)
```

When there are more texts than the batch size, the embedder splits them into sub-batches and runs up to `embed_concurrency` (default 20) API calls concurrently. For a document with 1000 chunks and batch_size=200, that's 5 sub-batches with up to 20 running at a time instead of 5 sequential calls.

### Caching

The embedder includes an in-memory LRU cache that avoids redundant API calls for repeated text:

```python
# Identical texts hit the cache automatically
embedding1 = await embedder.embed("same text")
embedding2 = await embedder.embed("same text")  # Cache hit, no API call
```

The cache uses an `OrderedDict` with configurable max size. Frequently embedded strings (like entity descriptions that appear across documents) benefit the most.

## Custom Embedders

You can implement your own embedder:

```python
from khora.extraction.embedders.base import Embedder

class MyEmbedder(Embedder):
    @property
    def model_name(self) -> str:
        return "my-custom-model"

    @property
    def dimension(self) -> int:
        return 768

    async def embed(self, text: str) -> list[float]:
        # Your implementation
        return await my_embedding_api(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Your batch implementation
        return await my_batch_api(texts)
```

## What's Next?

- **[Chunkers](chunkers.md)** - Split text before embedding
- **[Extractors](extractors.md)** - Extract entities from chunks
- **[Query Engine](../query-engine/overview.md)** - Use embeddings for search
