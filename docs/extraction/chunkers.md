# Chunkers

Chunkers split documents into segments optimized for embedding and retrieval. This document covers the available chunking strategies.

## Overview

Chunking is a critical step that affects retrieval quality:

- **Too small**: Lose context, fragments of information
- **Too large**: Diluted embeddings, exceed model limits
- **Optimal**: Self-contained, coherent segments with preserved context

## Base Chunker

All chunkers extend the abstract `Chunker` base class:

```python
from abc import ABC, abstractmethod

class Chunker(ABC):
    def __init__(self, *, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def chunk(self, text: str) -> list[ChunkResult]:
        """Split text into chunks."""
        ...

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
```

### ChunkResult

```python
@dataclass
class ChunkResult:
    content: str              # Chunk text
    index: int                # Position in document
    start_char: int           # Character offset start
    end_char: int             # Character offset end
    token_count: int          # Token count
    metadata: dict[str, Any]  # Additional metadata
```

## Fixed Chunker

Splits text by token count with overlap.

### Configuration

```python
from khora.extraction.chunkers import FixedChunker

chunker = FixedChunker(
    chunk_size=512,       # Target tokens per chunk
    chunk_overlap=50,     # Overlap tokens between chunks
)

results = chunker.chunk(text)
```

### Behavior

```
Document: [====================================================]
                    Token positions

Chunk 0:  [===========]
                      ↓ overlap
Chunk 1:           [===========]
                             ↓ overlap
Chunk 2:                  [===========]
                                    ↓ overlap
Chunk 3:                         [=======]  (smaller, end of doc)
```

### Use Cases

- Consistent chunk sizes needed
- Simple, predictable chunking
- When context preservation is less critical

## Semantic Chunker

Respects sentence and paragraph boundaries.

### Configuration

```python
from khora.extraction.chunkers import SemanticChunker

chunker = SemanticChunker(
    chunk_size=512,       # Target tokens per chunk
    chunk_overlap=50,     # Overlap tokens
)

results = chunker.chunk(text)
```

### Behavior

1. Split text into sentences (using sentence boundary detection)
2. Group sentences until reaching `chunk_size`
3. Apply overlap at sentence boundaries
4. Never split mid-sentence

```
Document:  Sentence1. Sentence2. Sentence3. Sentence4. Sentence5.
                     ↓           ↓           ↓           ↓

Chunk 0:   [Sentence1. Sentence2.]
                       ↓ overlap (full sentence)
Chunk 1:   [Sentence2. Sentence3. Sentence4.]
                                  ↓ overlap
Chunk 2:   [Sentence4. Sentence5.]
```

### Use Cases

- Natural language content
- When sentence context is important
- Documents with clear sentence structure

## Recursive Chunker

Hierarchically splits using multiple separators (LangChain-style).

### Configuration

```python
from khora.extraction.chunkers import RecursiveChunker

chunker = RecursiveChunker(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "],  # Default separators
)

results = chunker.chunk(text)
```

### Behavior

1. Try to split on first separator (`\n\n` = paragraphs)
2. If chunks still too large, split on next separator (`\n` = lines)
3. Continue until chunks fit within `chunk_size`
4. Apply overlap at the chosen boundary level

```
Level 1: Split on paragraphs (\\n\\n)
         [Paragraph 1]  [Paragraph 2]  [Paragraph 3]
                ↓ too large
Level 2: Split on lines (\\n)
         [Line 1, Line 2]  [Line 3, Line 4]
                ↓ too large
Level 3: Split on sentences (. )
         [Sent1. Sent2.]  [Sent3. Sent4.]
```

### Default Separators

```python
DEFAULT_SEPARATORS = [
    "\n\n",     # Paragraphs
    "\n",       # Lines
    ". ",       # Sentences
    ", ",       # Clauses
    " ",        # Words
    "",         # Characters (fallback)
]
```

### Use Cases

- Markdown documents
- Code files
- Structured text with clear hierarchy
- General-purpose chunking

## Token Counting

All chunkers use tiktoken for accurate token counting:

```python
import tiktoken

def count_tokens(text: str) -> int:
    # Uses cl100k_base encoding (GPT-4, Claude compatible)
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))
```

## Chunker Selection Guide

| Content Type | Recommended Chunker |
|--------------|---------------------|
| General text | `semantic` |
| Markdown/docs | `recursive` |
| Fixed-size needs | `fixed` |
| Unstructured data | `fixed` |
| Technical docs | `recursive` |
| Conversations | `semantic` |

## Configuration Recommendations

### General Use

```python
chunk_size = 512      # Good balance
chunk_overlap = 50    # ~10% overlap
strategy = "semantic"
```

### Long Documents

```python
chunk_size = 1024     # Larger context
chunk_overlap = 100   # More overlap
strategy = "recursive"
```

### Short Documents

```python
chunk_size = 256      # Smaller chunks
chunk_overlap = 25    # Proportional overlap
strategy = "semantic"
```

## API Usage

### Via MemoryLake

```python
result = await lake.remember(
    content,
    chunk_strategy="semantic",
    chunk_size=512,
)
```

### Direct Chunker Usage

```python
from khora.extraction.chunkers import SemanticChunker

chunker = SemanticChunker(chunk_size=512)
chunk_results = chunker.chunk(document_text)

for result in chunk_results:
    print(f"Chunk {result.index}: {result.token_count} tokens")
    print(f"  Position: {result.start_char}-{result.end_char}")
    print(f"  Content: {result.content[:100]}...")
```

### In Pipeline Tasks

```python
from khora.pipelines.tasks import chunk_document

chunks = await chunk_document(
    document,
    strategy="recursive",
    chunk_size=1024,
)
```

## Output

Chunkers produce `Chunk` model instances with:

```python
Chunk(
    id=uuid4(),
    document_id=document.id,
    namespace_id=document.namespace_id,
    content=chunk_result.content,
    index=chunk_result.index,
    start_char=chunk_result.start_char,
    end_char=chunk_result.end_char,
    token_count=chunk_result.token_count,
    metadata={
        "chunker": "semantic",
        "overlap_tokens": 50,
        **chunk_result.metadata,
    },
    created_at=document.created_at,  # Inherit timestamp
)
```

## Next Steps

- [Embedders](embedders.md) - Embedding generation
- [Extractors](extractors.md) - Entity extraction
- [Ingestion Pipeline](ingestion-pipeline.md) - Full pipeline
