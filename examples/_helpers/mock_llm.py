"""Deterministic mock LLM for the integration examples.

Patches ``litellm.acompletion`` and ``litellm.aembedding`` so the
examples-smoke job does not need an OpenAI key. Embeddings are
hash-derived (SHA1 → numpy seed → L2-normalised float vector) so the
same text always produces the same embedding across runs. Completions
cycle through a configurable response list (default ``["stub
response"]``).

Usage in an ``example.py``::

    from examples._helpers import install_mock_llm

    install_mock_llm()  # patches litellm before the first Khora call

Or with pytest-style monkeypatching::

    install_mock_llm(monkeypatch=monkeypatch, responses=["yes", "no"])
"""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any


def _hash_to_unit_vector(text: str, dim: int) -> list[float]:
    """Map text → deterministic L2-normalised float vector of length ``dim``.

    SHA-256 stream chunked into 8-byte little-endian unsigned ints, each
    rescaled to [-1, 1], padded/truncated to ``dim``, then L2-normalised.
    Pure stdlib, no numpy dep so the helper stays importable in a thin
    examples venv.
    """
    out: list[float] = []
    counter = 0
    while len(out) < dim:
        digest = hashlib.sha256(f"{text}|{counter}".encode()).digest()
        for i in range(0, len(digest), 8):
            chunk = digest[i : i + 8]
            if len(chunk) < 8:
                break
            value = int.from_bytes(chunk, "little", signed=False)
            # uint64 range → [-1, 1]
            out.append((value / (2**64 - 1)) * 2.0 - 1.0)
            if len(out) >= dim:
                break
        counter += 1

    # L2 normalise
    norm = sum(v * v for v in out) ** 0.5
    if norm == 0.0:
        return [0.0] * dim
    return [v / norm for v in out[:dim]]


@dataclass
class _StubMessage:
    role: str = "assistant"
    content: str = ""


@dataclass
class _StubChoice:
    message: _StubMessage = field(default_factory=_StubMessage)
    finish_reason: str = "stop"
    index: int = 0


@dataclass
class _StubUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class _StubCompletion:
    choices: list[_StubChoice]
    usage: _StubUsage = field(default_factory=_StubUsage)
    model: str = "mock"


@dataclass
class _StubEmbeddingItem:
    """Embedding row.

    litellm returns dict-shaped rows under ``response.data``, so callers
    in khora reach for ``item["embedding"]``. We support both attribute
    and subscript access so the stub is drop-in compatible.
    """

    embedding: list[float]
    index: int = 0

    def __getitem__(self, key: str) -> Any:
        if key == "embedding":
            return self.embedding
        if key == "index":
            return self.index
        raise KeyError(key)


@dataclass
class _StubEmbedding:
    data: list[_StubEmbeddingItem]
    usage: _StubUsage = field(default_factory=_StubUsage)
    model: str = "mock-embedding"


class MockLLM:
    """Cycles through stub completions and produces hashed embeddings."""

    def __init__(self, responses: Sequence[str] | None = None, dim: int = 1536) -> None:
        self._responses: list[str] = list(responses) if responses else ["stub response"]
        if not self._responses:
            raise ValueError("MockLLM requires at least one response")
        self._dim = dim
        self._call_index = 0
        self.completion_calls: list[dict[str, Any]] = []
        self.embedding_calls: list[dict[str, Any]] = []

    async def acompletion(self, *args: Any, **kwargs: Any) -> _StubCompletion:
        self.completion_calls.append(kwargs)
        text = self._responses[self._call_index % len(self._responses)]
        self._call_index += 1
        return _StubCompletion(
            choices=[_StubChoice(message=_StubMessage(content=text))],
            usage=_StubUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            model=kwargs.get("model", "mock"),
        )

    async def aembedding(self, *args: Any, **kwargs: Any) -> _StubEmbedding:
        self.embedding_calls.append(kwargs)
        inputs = kwargs.get("input", [])
        if isinstance(inputs, str):
            inputs = [inputs]
        elif not isinstance(inputs, Iterable):
            inputs = [str(inputs)]
        items = [
            _StubEmbeddingItem(embedding=_hash_to_unit_vector(str(text), self._dim), index=i)
            for i, text in enumerate(inputs)
        ]
        return _StubEmbedding(
            data=items,
            usage=_StubUsage(prompt_tokens=len(items), completion_tokens=0, total_tokens=len(items)),
            model=kwargs.get("model", "mock-embedding"),
        )


def install_mock_llm(
    *,
    monkeypatch: Any = None,
    responses: Sequence[str] | None = None,
    dim: int = 1536,
) -> MockLLM:
    """Patch ``litellm.acompletion`` and ``litellm.aembedding``.

    Args:
        monkeypatch: pytest ``MonkeyPatch`` fixture. If provided, patches
            are scoped to the test. If ``None``, patches are applied
            globally (suitable for ``example.py`` scripts).
        responses: completion strings to cycle through. Default ``["stub
            response"]``.
        dim: embedding dimension. Default 1536 (matches text-embedding-3-small).

    Returns:
        The ``MockLLM`` instance, so callers can inspect ``completion_calls``
        / ``embedding_calls`` or reset ``responses`` mid-run.
    """
    try:
        import litellm
    except ImportError as exc:  # pragma: no cover - examples require litellm via khora
        raise RuntimeError("litellm must be installed to use the mock LLM helper") from exc

    mock = MockLLM(responses=responses, dim=dim)

    if monkeypatch is not None:
        monkeypatch.setattr(litellm, "acompletion", mock.acompletion)
        monkeypatch.setattr(litellm, "aembedding", mock.aembedding)
    else:
        litellm.acompletion = mock.acompletion  # type: ignore[assignment]
        litellm.aembedding = mock.aembedding  # type: ignore[assignment]
        # Some khora call sites import these as module-level symbols; refresh.
        for module_name in list(sys.modules):
            module = sys.modules[module_name]
            if module is None or not module_name.startswith(("khora.", "litellm")):
                continue
            if getattr(module, "acompletion", None) is not None and module_name.startswith("litellm"):
                module.acompletion = mock.acompletion  # type: ignore[attr-defined]
            if getattr(module, "aembedding", None) is not None and module_name.startswith("litellm"):
                module.aembedding = mock.aembedding  # type: ignore[attr-defined]

    return mock
