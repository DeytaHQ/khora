# Chunkers

Chunking is deceptively important. Split your documents wrong, and your retrieval suffers - fragments lose context, or bloated chunks dilute relevance. Split them right, and each chunk becomes a self-contained piece of knowledge.

## Why Chunking Matters

Embedding models have token limits (typically 512-8192). More importantly, retrieval works best when chunks are:

- **Coherent** - A complete thought, not a sentence fragment
- **Focused** - About one thing, not a rambling mixture
- **Contextual** - Enough surrounding information to make sense

Too small:
```
"Einstein developed"      <- Useless fragment
```

Too large:
```
"Einstein developed relativity. He also liked music.
 Meanwhile, Marie Curie was working on radioactivity.
 The periodic table was expanded that decade..."
                          <- Too many topics, diluted embedding
```

Just right:
```
"Albert Einstein developed the theory of special relativity in 1905
 while working at the Swiss patent office. This groundbreaking work
 introduced the famous equation E=mc² and fundamentally changed our
 understanding of space, time, and energy."
                          <- Coherent, focused, contextual
```

## The Three Chunking Strategies

### Fixed Chunker

The simplest approach: split by token count with overlap.

```python
from khora.extraction.chunkers import FixedChunker

chunker = FixedChunker(chunk_size=512, chunk_overlap=50)
chunks = chunker.chunk(document_text)
```

**How it works:**

```
Document: |=========================================|
          0       512      1024     1536     2048

Chunk 0:  |==========|
          0         512

Chunk 1:       |==========|
              462        974
               ^
               overlap starts at 462 (512-50)

Chunk 2:            |==========|
                   924       1436

Chunk 3:                 |======|
                        1386   1800 (end of doc)
```

**Pros:**
- Predictable, consistent chunk sizes
- Simple to reason about
- Fast

**Cons:**
- Cuts mid-sentence, mid-paragraph
- Context can be awkwardly split

**Best for:**
- Content without clear structure
- When you need consistent sizing
- Large-scale processing where speed matters

### Semantic Chunker

Respects natural language boundaries - never splits mid-sentence.

```python
from khora.extraction.chunkers import SemanticChunker

chunker = SemanticChunker(chunk_size=512, chunk_overlap=50)
chunks = chunker.chunk(document_text)
```

**How it works:**

1. Split text into sentences
2. Group sentences until approaching `chunk_size`
3. Add the next sentence only if it fits
4. Overlap by including trailing sentences from previous chunk

```
Document:
"Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
      |              |             |              |              |
     80t           120t          150t           100t           130t
    (tokens)

Chunk 0: "Sentence one. Sentence two. Sentence three."
         = 350 tokens (under 512, but adding #4 would exceed)

Chunk 1: "Sentence three. Sentence four. Sentence five."
         = 380 tokens (overlap includes sentence three)
```

**Pros:**
- Natural boundaries preserve meaning
- Better embeddings (complete thoughts)
- More readable chunks

**Cons:**
- Variable chunk sizes
- Very long sentences can cause issues

**Best for:**
- Natural language content
- Articles, reports, documentation
- Content you might show to users

#### spaCy-Enhanced Sentence Splitting

The semantic chunker can optionally use spaCy's `sentencizer` component for more accurate sentence boundary detection. The sentencizer is a rule-based component that ships with spaCy core — no separate model download needed.

```bash
# Install the optional NLP extra
pip install khora[nlp]
```

When spaCy is installed, the semantic chunker uses it automatically — no code changes needed. If spaCy is not installed, the chunker falls back to its regex-based splitter transparently. The `_HAS_SPACY` flag in the chunker module controls this behavior.

### Recursive Chunker

Tries increasingly fine-grained splits until chunks fit.

```python
from khora.extraction.chunkers import RecursiveChunker

chunker = RecursiveChunker(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "]
)
chunks = chunker.chunk(document_text)
```

**How it works:**

```
Level 1: Try splitting on paragraphs (\n\n)
         |Paragraph 1|    |Paragraph 2|    |Paragraph 3|
              |               |                |
         (fits!)         (too big!)         (fits!)
                              |
                              v
Level 2: Split big paragraph on lines (\n)
         |Line 1|  |Line 2|  |Line 3|  |Line 4|
             |         |         |         |
         (fits!)   (fits!)   (too big!)  (fits!)
                                  |
                                  v
Level 3: Split big line on sentences (. )
         |Sent 1|  |Sent 2|  |Sent 3|
             |         |         |
         (fits!)   (fits!)   (fits!)
```

**Default separators:**
```python
DEFAULT_SEPARATORS = [
    "\n\n",   # Paragraphs
    "\n",     # Lines
    ". ",     # Sentences
    ", ",     # Clauses
    " ",      # Words
    ""        # Characters (last resort)
]
```

**Pros:**
- Respects document structure
- Adapts to content hierarchy
- Works well with Markdown, code, structured text

**Cons:**
- More complex logic
- May produce uneven results on unstructured content

**Best for:**
- Markdown documents
- Code files
- Technical documentation
- Any content with clear hierarchy

### Conversation Chunker

Groups Slack messages into coherent conversation chunks using thread-awareness, temporal proximity, and optional semantic similarity. Unlike text chunkers, it operates on structured `SlackMessage` objects and preserves per-message metadata (author, timestamp, character offsets) for individual message retrieval.

```python
from khora.extraction.chunkers import ConversationChunker, ConversationChunkerConfig

config = ConversationChunkerConfig(time_gap_minutes=15, max_group_size=50)
chunker = ConversationChunker(config=config)
chunks = chunker.chunk_messages(messages)
```

**How it works:**

1. **Thread grouping** — messages sharing a `thread_ts` are kept together
2. **Temporal windowing** — top-level messages split on time gaps (default: 15 min)
3. **Size enforcement** — groups exceeding `max_group_size` are split; tiny groups merge with neighbours

**Best for:**
- Slack messages and threads
- Any multi-author conversation data
- When you need individual message retrieval from search results

See [Conversation Chunking](conversation-chunking.md) for full details.

## Choosing the Right Chunker

| Content Type | Recommended | Why |
|--------------|-------------|-----|
| Blog posts, articles | Semantic | Natural language with paragraphs |
| Technical docs | Recursive | Headers, code blocks, structure |
| Slack conversations | Conversation | Thread-aware, per-message metadata |
| Chat logs | Semantic | Conversational, sentence-focused |
| Code files | Recursive | Functions, classes, blocks |
| CSV/structured data | Fixed | No natural boundaries |
| Books/long-form | Semantic | Chapter/paragraph structure |
| API responses | Fixed or Recursive | Depends on format |

## Configuration

### Chunk Size

The `chunk_size` is measured in tokens, not characters. Khora uses tiktoken with the `cl100k_base` encoding (compatible with GPT-4, Claude).

```python
# Rough conversion: 1 token ≈ 4 characters (English)
chunk_size=512   # ~2000 characters
chunk_size=1024  # ~4000 characters
chunk_size=256   # ~1000 characters
```

**Guidelines:**
- `256` - Short, focused chunks (good for Q&A)
- `512` - Balanced (default, works for most cases)
- `1024` - More context (good for complex topics)
- `2048+` - Long context (needs large embedding models)

### Chunk Overlap

Overlap prevents context loss at boundaries:

```python
chunk_overlap=50   # ~10% of 512 (good default)
chunk_overlap=100  # More overlap for complex content
chunk_overlap=0    # No overlap (fastest, may lose context)
```

More overlap = more redundancy = better boundary handling, but larger storage.

## Usage Examples

### Via MemoryLake

```python
# Use semantic chunking (default)
await lake.remember(content, chunk_strategy="semantic")

# Use recursive for structured docs
await lake.remember(content, chunk_strategy="recursive", chunk_size=1024)

# Use fixed for speed
await lake.remember(content, chunk_strategy="fixed", chunk_size=512)
```

### Direct Usage

```python
from khora.extraction.chunkers import SemanticChunker

chunker = SemanticChunker(chunk_size=512, chunk_overlap=50)
results = chunker.chunk(text)

for result in results:
    print(f"Chunk {result.index}:")
    print(f"  Tokens: {result.token_count}")
    print(f"  Position: chars {result.start_char}-{result.end_char}")
    print(f"  Content: {result.content[:100]}...")
```

### Token Counting

All chunkers use tiktoken for accurate counts:

```python
from khora.extraction.chunkers import Chunker

chunker = SemanticChunker()
token_count = chunker.count_tokens("Hello, world!")
# Returns: 4
```

## The ChunkResult Object

Chunkers return `ChunkResult` objects:

```python
ChunkResult(
    content="The actual chunk text...",
    index=0,              # Position in document (0-indexed)
    start_char=0,         # Character offset in original
    end_char=2048,        # End character offset
    token_count=512,      # Actual token count
    metadata={            # Additional info
        "sentences": 5,   # (semantic chunker adds this)
    }
)
```

The character offsets let you map chunks back to the original document - useful for highlighting or citation.

## Empty Chunk Filtering

All chunkers automatically filter out empty or near-empty chunks before returning results. This prevents whitespace-only chunks from polluting the retrieval index.

**How it works:**

- `MIN_CHUNK_CHARS = 10` — Chunks shorter than 10 characters (after stripping) are discarded
- `filter_empty_chunks()` — Base class method called by all chunkers before returning
- `.strip()` on tokenizer decode — `FixedChunker` and `SemanticChunker._fixed_split()` strip whitespace from decoded token sequences

```python
# In Chunker base class (extraction/chunkers/base.py)
MIN_CHUNK_CHARS = 10

def filter_empty_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
    """Remove chunks with fewer than MIN_CHUNK_CHARS characters."""
    return [c for c in chunks if len(c.content.strip()) >= MIN_CHUNK_CHARS]
```

This filtering happens at chunk creation time rather than query time, so empty chunks never enter the vector index. Previously, ~17% of retrieval queries encountered sub-10-character chunks that had to be filtered during search.

## Performance Tips

1. **Batch processing**: Chunkers are synchronous and fast. The bottleneck is usually embedding, not chunking.

2. **Reuse chunkers**: Create one chunker instance and reuse it:
   ```python
   chunker = SemanticChunker(chunk_size=512)
   for doc in documents:
       chunks = chunker.chunk(doc.content)
   ```

3. **Pre-filter content**: Remove boilerplate (headers, footers) before chunking.

4. **Match embedding model**: If your embedding model handles 8192 tokens, you can use larger chunks:
   ```python
   # For models with large context
   chunker = SemanticChunker(chunk_size=2048)
   ```

## What's Next?

- **[Embedders](embedders.md)** - Turn chunks into vectors
- **[Extractors](extractors.md)** - Extract entities from chunks
- **[Ingestion Pipeline](ingestion-pipeline.md)** - The full processing flow
