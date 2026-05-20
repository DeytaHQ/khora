"""LlamaIndex + khora example - async retrieval via ``KhoraRetriever``.

Runs without Postgres, Neo4j, or an API key. The mock LLM patches
``litellm.acompletion`` / ``litellm.aembedding`` so the example is
hermetic. The khora fixture spins up an in-memory ``sqlite_lance``
backend in a tmp dir.

Demonstrates:

* Stash a couple of documents into khora via ``Khora.remember``.
* Wrap khora in ``KhoraRetriever`` and call ``aretrieve(...)``.
* Each returned ``NodeWithScore`` carries chunk text + khora metadata
  (chunk_id, document_id, abstention signal).

``KhoraMemoryBlock`` and ``KhoraChatStore`` are also exported by the
adapter; see ``docs/integrations/llamaindex.md`` for usage notes.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Add repo root to sys.path so ``examples._helpers`` is importable when
# this script is run from its own directory (CI smoke loop does that).
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples._helpers import embedded_khora, install_mock_llm  # noqa: E402
from khora.integrations.llamaindex import KhoraRetriever  # noqa: E402


async def main() -> None:
    install_mock_llm()

    async with embedded_khora() as kb:
        namespace = await kb.create_namespace()
        ns_id = namespace.namespace_id

        # Stash two documents through khora's normal write path.
        memory_one = "We decided to use PostgreSQL for the user database."
        memory_two = "The release window is the third week of every month."
        await kb.remember(memory_one, namespace=ns_id, entity_types=[], relationship_types=[])
        await kb.remember(memory_two, namespace=ns_id, entity_types=[], relationship_types=[])

        retriever = KhoraRetriever(kb, namespace_id=ns_id, similarity_top_k=3)

        # Verbatim recall: the mock LLM's hash-derived embeddings give
        # an exact match (cosine = 1.0) for the stored text.
        nodes = await retriever.aretrieve(memory_one)
        for node in nodes:
            text = node.node.text.replace("\n", " ")
            print(f"[{node.score:.2f}] {text}")


if __name__ == "__main__":
    asyncio.run(main())
