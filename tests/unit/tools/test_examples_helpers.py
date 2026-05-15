"""Smoke tests for examples/_helpers.

These are the harness layer the adapter examples rely on. If the mock
LLM stops returning the right shape or the embedded khora fixture stops
spinning up, every example.py will fail in CI without a clear signal.
These tests catch breakage at the helper layer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the top-level ``examples`` package importable in unit tests. It is
# not a regular installed package — it lives alongside src/ at the repo
# root so it has to be added to sys.path manually.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples._helpers import embedded_khora, install_mock_llm  # noqa: E402


@pytest.mark.embedded
async def test_install_mock_llm_returns_deterministic_embeddings() -> None:
    import litellm

    mock = install_mock_llm(dim=64)
    try:
        r1 = await litellm.aembedding(model="x", input="hello")
        r2 = await litellm.aembedding(model="x", input="hello")
        r3 = await litellm.aembedding(model="x", input="world")
    finally:
        # Restore litellm symbols (the unset call is in __init__).
        pass

    v1 = r1.data[0].embedding
    v2 = r2.data[0].embedding
    v3 = r3.data[0].embedding
    assert len(v1) == 64
    assert v1 == v2  # deterministic by text
    assert v1 != v3  # text-sensitive
    # L2 normalised
    norm = sum(x * x for x in v1) ** 0.5
    assert abs(norm - 1.0) < 1e-6
    assert mock.embedding_calls  # call observed


@pytest.mark.embedded
async def test_install_mock_llm_cycles_completion_responses() -> None:
    import litellm

    install_mock_llm(responses=["alpha", "beta"])
    r1 = await litellm.acompletion(model="x", messages=[{"role": "user", "content": "?"}])
    r2 = await litellm.acompletion(model="x", messages=[{"role": "user", "content": "?"}])
    r3 = await litellm.acompletion(model="x", messages=[{"role": "user", "content": "?"}])
    assert r1.choices[0].message.content == "alpha"
    assert r2.choices[0].message.content == "beta"
    assert r3.choices[0].message.content == "alpha"  # cycles


@pytest.mark.embedded
async def test_embedded_khora_round_trip(tmp_path: Path) -> None:
    """remember() + recall() through the sqlite_lance fixture."""
    try:
        import aiosqlite  # noqa: F401
        import lancedb  # noqa: F401
    except ImportError:
        pytest.skip("sqlite_lance optional deps not installed")

    install_mock_llm(dim=64)
    async with embedded_khora(embedding_dimension=64) as kb:
        ns = await kb.create_namespace()
        await kb.remember(
            "Alice met Bob at the conference.",
            namespace=ns.namespace_id,
            entity_types=["PERSON"],
            relationship_types=["MET"],
        )
        result = await kb.recall("Alice", namespace=ns.namespace_id)
        # We don't assert recall quality with a hash-mock embedder — just
        # that the call sequence completes without raising.
        assert result is not None
