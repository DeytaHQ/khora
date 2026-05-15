"""``khora.integrations.crewai`` — CrewAI memory adapter.

End-user surface: :func:`KhoraMemory`. One call returns a fully wired
``crewai.Memory(storage=KhoraStorageBackend(...))`` instance that can
be passed straight into ``Agent(memory=...)``.

The module level intentionally avoids ``import crewai``. Only the
factory function loads CrewAI types — kept lazy so ``pip install
khora`` without the ``[crewai]`` extra still lets users construct a
``KhoraStorageBackend`` for testing.

Stability: experimental until one full khora minor ships without a
breaking change to this adapter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

if TYPE_CHECKING:  # pragma: no cover - typing only
    from khora.khora import Khora


__all__ = ["KhoraMemory", "KhoraStorageBackend"]


def KhoraMemory(  # noqa: N802 — public factory matches PEP-8 by convention for "class-shaped"
    *,
    kb: Khora,
    namespace: UUID,
    user_id: str,
    app_id: str = "crewai",
    scope_root: str | None = "/",
    **memory_kwargs: Any,
) -> Any:
    """Build a ``crewai.Memory`` instance backed by khora.

    The factory does three things:

    1. Validates ``user_id`` — empty, ``"default"``, or any value
       shorter than 8 characters is rejected with
       :class:`khora.exceptions.KhoraIntegrationError`. Silent
       cross-user reads are the dominant disaster mode for agent
       memory adapters (#618 risk list).
    2. Constructs a :class:`KhoraStorageBackend` bound to
       ``(kb, namespace, user_id, app_id)``.
    3. Wires a text-stashing embedder into ``crewai.Memory`` so the
       ``StorageBackend.search()`` callback can recover the original
       query text. CrewAI's ``Memory`` passes only the pre-computed
       embedding to the backend, so without this stash we'd have no
       way to feed khora's HyDE / rerank pipeline with text. The
       embedder still returns a deterministic dummy vector — CrewAI
       needs *something*, but :meth:`KhoraStorageBackend.search`
       discards it.

    Args:
        kb: A connected :class:`khora.Khora` instance. Caller owns the
            lifecycle.
        namespace: Stable khora namespace UUID.
        user_id: Stable end-user identifier (≥ 8 chars, not
            ``"default"``).
        app_id: Free-form app identifier (default ``"crewai"``).
        scope_root: Forwarded to ``crewai.Memory(root_scope=...)`` so
            every record this adapter saves lives under the given
            scope prefix. Default ``"/"`` keeps the tree flat.
        **memory_kwargs: Passed through to ``crewai.Memory(...)``.
            Use this for the ``llm``, recency/semantic/importance
            weights, ``read_only``, etc.

    Returns:
        A ``crewai.Memory`` instance with ``storage`` and ``embedder``
        already wired.

    Raises:
        KhoraIntegrationError: If ``user_id`` fails validation.
        ImportError: If the ``[crewai]`` extra is not installed.

    Reentrancy note:
        ``KhoraStorageBackend`` dispatches every async call through
        :func:`khora.integrations._sync.run_sync`, which refuses to
        run from inside an already-running asyncio event loop. Do not
        construct or invoke this adapter from within an async handler
        — call it from a sync entry point or a worker thread.
    """
    # Lazy framework imports — see module docstring.
    from crewai.memory.types import MemoryRecord  # noqa: PLC0415 — required to be lazy
    from crewai.memory.unified_memory import Memory  # noqa: PLC0415

    from khora.integrations.crewai.storage import (  # noqa: PLC0415 — adapter-local
        KhoraStorageBackend,
        _raise_invalid_user_id,
        _stash_query_text,
    )

    _raise_invalid_user_id(user_id)

    storage = KhoraStorageBackend(
        kb=kb,
        namespace_id=namespace,
        user_id=user_id,
        app_id=app_id,
        memory_record_cls=MemoryRecord,
    )

    def _stashing_embedder(texts: list[str]) -> list[list[float]]:
        """Capture CrewAI's embed-text payload and return a dummy vector.

        CrewAI's ``Memory.recall`` flow calls ``embed_text(self._embedder,
        query)``, which calls ``embedder([query])`` (see
        ``crewai.memory.types.embed_text``). The vector this function
        returns is forwarded to ``KhoraStorageBackend.search`` and
        discarded — we only need the side-effect of stashing the text.
        """
        if texts:
            _stash_query_text(texts[-1])
        # Return a 1-dim zero vector per input. CrewAI does shape
        # sanity checks (``if not embedding: results = []``) but does
        # not inspect the vector content; our search() discards it.
        return [[0.0] for _ in texts]

    return Memory(
        storage=storage,
        embedder=_stashing_embedder,
        root_scope=scope_root,
        **memory_kwargs,
    )


def _import_storage_backend_for_typing() -> type:
    """Internal helper: late-load ``KhoraStorageBackend`` for re-export.

    ``__all__`` lists ``KhoraStorageBackend`` for IDE / static-analysis
    convenience, but we don't want a top-level ``import`` of the
    backend module on package load — it's the call sites that should
    pay the import cost. The class is exposed via ``__getattr__``.
    """
    from khora.integrations.crewai.storage import KhoraStorageBackend  # noqa: PLC0415

    return KhoraStorageBackend


def __getattr__(name: str) -> Any:
    """Lazy export for ``KhoraStorageBackend``.

    Allows ``from khora.integrations.crewai import KhoraStorageBackend``
    without forcing a load of ``storage.py`` (which itself imports
    ``run_sync`` etc.) at package import time.
    """
    if name == "KhoraStorageBackend":
        return _import_storage_backend_for_typing()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
