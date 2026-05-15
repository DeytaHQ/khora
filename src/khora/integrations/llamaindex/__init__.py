"""LlamaIndex adapter — ``KhoraRetriever``, ``KhoraMemoryBlock``, ``KhoraChatStore``.

End-user surface:

* :class:`KhoraRetriever` — async-only ``BaseRetriever`` for plugging
  khora into a LlamaIndex ``QueryEngine`` / agent pipeline. The sync
  ``_retrieve`` path is intentionally not implemented (raises
  ``NotImplementedError``); use ``aretrieve``.
* :func:`KhoraMemoryBlock` — factory that returns a
  ``BaseMemoryBlock[str]`` instance backed by khora. Wraps
  ``Khora.recall`` (read) and ``Khora.remember`` (write); use it inside
  ``llama_index.core.memory.Memory(memory_blocks=[...])`` for an agent's
  long-term semantic memory.
* :func:`KhoraChatStore` — **deprecated** factory for the legacy
  ``ChatMemoryBuffer`` path. Emits ``DeprecationWarning`` on call. New
  code should use ``KhoraMemoryBlock``.

Module-load discipline: this package imports nothing from
``llama_index`` at module top level. The framework imports live inside
function bodies (see ``retriever.py``, ``memory.py``, ``chat_store.py``)
so ``import khora.integrations.llamaindex`` works without the optional
``[llamaindex]`` extra installed. Verified by
``tools/check_optional_imports.py`` (AST lint) plus the subprocess probe
in ``tests/unit/integrations/test_no_eager_imports.py``.

Stability: experimental. LlamaIndex has shipped breaking changes on
minor bumps (``BaseMemoryBlock`` API reshape between 0.11 → 0.12 → 0.14,
``BaseMemory.put`` → ``aput``). The ``[llamaindex]`` extra pins
``llama-index-core>=0.14,<0.15`` deliberately; quarterly maintenance
reserved for the next minor bump.
"""

from __future__ import annotations

from khora.integrations.llamaindex.chat_store import KhoraChatStore
from khora.integrations.llamaindex.memory import KhoraMemoryBlock
from khora.integrations.llamaindex.retriever import KhoraRetriever

__all__ = ["KhoraChatStore", "KhoraMemoryBlock", "KhoraRetriever"]
