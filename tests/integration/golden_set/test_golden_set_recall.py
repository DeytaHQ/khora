"""Golden-set retrieval regression tests (#1479).

A hermetic CI safety net that pins retrieval RANK POSITIONS so any future
ranking change that demotes a known-correct chunk fails loudly - without a paid
benchmark run. Corpus + expected ranks live in ``corpus.json`` (see
``README.md`` in this directory for how to add a case).

Determinism / no-flakiness:

* **No LLM, no network.** The embedder and entity extractor are patched to
  deterministic in-process functions (``stub_llm`` from the shared test
  helpers stages the extraction registry; ``_patch_vocab_embedder`` swaps in a
  token-vocabulary embedding). Nothing here calls out.
* **Semantic embedding.** Unlike the SHA-hash ``fake_embedding`` helper (which
  is deterministic but NOT semantic - a query and its answer hash to unrelated
  vectors), ``vocab_embedding`` is a normalized bag-of-words over a FIXED
  vocabulary built from the corpus, so cosine similarity tracks lexical
  overlap. That makes rank assertions meaningful and stable: the same corpus +
  same query always yields the same order.
* **Full engine recall path.** We drive ``Khora.recall`` (the VectorCypher
  engine) so the assertions cover fusion + boosts + rerank + MMR - the exact
  pipeline the #1463 MMR bug and the #1433 score/order break lived in. A
  regression that demotes a gold chunk in the RETURNED ORDER trips these tests.

Assertions are on ORDER (#1433: the returned order is the authoritative
ranking), pinned loose enough not to be brittle (top-N, not exact position) but
tight enough to catch a real demotion (a gold doc falling out of top-N fails).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

try:
    import aiosqlite  # noqa: F401
    import lancedb  # noqa: F401

    _HAS_EMBEDDED = True
except ImportError:
    _HAS_EMBEDDED = False

from tests.test_helpers.filter_spy import stub_llm

pytestmark = [
    pytest.mark.integration,
    pytest.mark.embedded,
    pytest.mark.skipif(not _HAS_EMBEDDED, reason="aiosqlite/lancedb not installed"),
]

_CORPUS_PATH = Path(__file__).with_name("corpus.json")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _load_corpus() -> dict[str, Any]:
    return json.loads(_CORPUS_PATH.read_text(encoding="utf-8"))


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _build_vocabulary(corpus: dict[str, Any]) -> dict[str, int]:
    """Fixed token -> dimension index map built from every document + query.

    Sorted for a stable, order-independent index assignment so the vocabulary
    (and therefore every embedding) is identical across runs and machines.
    """
    vocab: set[str] = set()
    for doc in corpus["documents"]:
        vocab.update(_tokens(doc["content"]))
    for q in corpus["queries"]:
        vocab.update(_tokens(q["query"]))
    return {tok: i for i, tok in enumerate(sorted(vocab))}


def _make_vocab_embedding(vocab: dict[str, int]):
    """Return an L2-normalized bag-of-words embedding function over ``vocab``.

    Each dimension is a token count; the vector is L2-normalized so cosine
    similarity between a query and a document rises with shared-token overlap.
    Deterministic and semantic-enough for stable rank assertions. Unknown
    tokens (present in neither corpus nor queries - impossible here since the
    vocab is built from both) are ignored.
    """
    dim = len(vocab)

    def vocab_embedding(text: str) -> list[float]:
        vec = [0.0] * dim
        for tok in _tokens(text):
            idx = vocab.get(tok)
            if idx is not None:
                vec[idx] += 1.0
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]

    return vocab_embedding, dim


def _patch_vocab_embedder(monkeypatch: pytest.MonkeyPatch, vocab_embedding) -> None:
    """Swap the LiteLLM embedder for the deterministic vocab embedding.

    Patches the SAME class methods ``stub_llm`` does, so every engine code path
    (ingest + query) picks up the semantic embedding. Call AFTER ``stub_llm``
    (which patches the SHA embedder) so this overrides it.
    """

    async def _embed_batch(self: Any, texts: list[str]) -> list[list[float]]:
        return [vocab_embedding(t) for t in texts]

    async def _embed(self: Any, text: str) -> list[float]:
        return vocab_embedding(text)

    monkeypatch.setattr(
        "khora.extraction.embedders.litellm.LiteLLMEmbedder.embed_batch",
        _embed_batch,
    )
    monkeypatch.setattr(
        "khora.extraction.embedders.litellm.LiteLLMEmbedder.embed",
        _embed,
    )


def _config(tmp_path: Path, dim: int):
    from khora.config import KhoraConfig
    from khora.config.schema import SQLiteLanceConfig

    config = KhoraConfig()
    config.storage.backend = "sqlite_lance"
    config.storage.sqlite_lance = SQLiteLanceConfig(
        db_path=str(tmp_path / "golden.db"),
        lance_path=str(tmp_path / "golden.lance"),
        embedding_dimension=dim,
    )
    config.storage.embedding_dimension = dim
    config.llm.embedding_dimension = dim
    # Every chunk goes to the (stub) extractor so the graph channel has entities
    # to traverse for the multi-hop / multi-entity archetypes.
    config.pipelines.selective_extraction = False
    # Determinism: the cross-encoder reranker downloads a real model whose
    # near-tied tail scores resolve by per-ingest-random UUID, so two ingests of
    # the same corpus can differ at the tail. Disable it so the golden set pins
    # the pure (deterministic) vector + graph fusion + MMR pipeline. MMR stays
    # ON so the #1463-class demotion this set guards against is still exercised.
    config.query.enable_reranking = False
    config.query.enable_llm_reranking = False
    # HyDE would issue a real (unstubbed) chat LLM call to synthesize a
    # hypothetical document, adding a network dependency and non-determinism.
    # Force it off so the golden set is fully hermetic and stable.
    config.query.enable_hyde = "never"
    return config


def _rank_of_doc(chunks: list[Any], doc_id: UUID) -> int | None:
    """1-based rank of the first chunk whose document_id == ``doc_id``.

    Assertions use the RETURNED chunk order (the authoritative ranking, #1433),
    NOT a re-sort by score. Deduplicates on document_id so a multi-chunk doc
    counts once at its best position (our corpus is single-chunk, but this keeps
    the helper honest).
    """
    seen: set[UUID] = set()
    ordinal = 0
    for chunk in chunks:
        if chunk.document_id in seen:
            continue
        seen.add(chunk.document_id)
        ordinal += 1
        if chunk.document_id == doc_id:
            return ordinal
    return None


async def _seed_and_recall(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Ingest the golden corpus, run every golden query, return (results, id_map).

    ``id_map`` maps corpus ``doc_id`` -> stored document UUID so assertions can
    translate the fixture's stable string ids to the recall result's UUIDs.
    """
    from khora import Khora, SearchMode

    corpus = _load_corpus()
    vocab = _build_vocabulary(corpus)
    vocab_embedding, dim = _make_vocab_embedding(vocab)

    # stub_llm first (patches SHA embedder + extractor + resets registry),
    # then override the embedder with the semantic vocab embedding.
    stub_llm(monkeypatch, dim=dim)
    _patch_vocab_embedder(monkeypatch, vocab_embedding)

    entity_types = ["PERSON", "PLACE", "ELEMENT", "PROJECT", "CONCEPT"]
    relationship_types = ["RELATED_TO", "DISCOVERED", "OWNS", "MENTIONS"]

    tmp_path.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    id_map: dict[str, UUID] = {}
    async with Khora(_config(tmp_path, dim), run_migrations=True) as kb:
        ns = await kb.create_namespace()
        for doc in corpus["documents"]:
            remembered = await kb.remember(
                doc["content"],
                namespace=ns.namespace_id,
                entity_types=entity_types,
                relationship_types=relationship_types,
            )
            id_map[doc["doc_id"]] = remembered.document_id

        for q in corpus["queries"]:
            results[q["query_id"]] = await kb.recall(
                q["query"],
                namespace=ns.namespace_id,
                limit=10,
                mode=SearchMode.HYBRID,
            )
    return corpus, results, id_map


async def test_golden_queries_gold_docs_within_pinned_rank(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every golden query's gold doc(s) must appear at or above the pinned rank.

    One ingest (~18 docs) + one recall per query, all assertions collected so a
    regression reports EVERY archetype that broke (not just the first). Runs in
    a couple seconds - migrations + ingest are paid once.
    """
    corpus, results, id_map = await _seed_and_recall(tmp_path, monkeypatch)

    failures: list[str] = []
    for spec in corpus["queries"]:
        query_id = spec["query_id"]
        chunks = results[query_id].chunks
        if not chunks:
            failures.append(f"[{query_id}] recall returned no chunks at all")
            continue
        max_rank = spec["max_rank"]
        for gold_doc_id in spec["gold_doc_ids"]:
            rank = _rank_of_doc(chunks, id_map[gold_doc_id])
            if rank is None:
                failures.append(
                    f"[{query_id}] ({spec['archetype']}) gold doc '{gold_doc_id}' absent from the "
                    f"returned {len(chunks)}-chunk result for query {spec['query']!r}"
                )
            elif rank > max_rank:
                failures.append(
                    f"[{query_id}] ({spec['archetype']}) gold doc '{gold_doc_id}' ranked #{rank} "
                    f"but must be within top-{max_rank} for query {spec['query']!r}"
                )

    assert not failures, (
        "Golden-set rank regression - a ranking change demoted a known-correct chunk "
        "(the #1463-class regression this golden set guards against):\n  " + "\n  ".join(failures)
    )


async def test_golden_set_recall_is_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated identical recalls on one ingest produce a byte-identical order.

    Guards the no-flakiness contract: with the deterministic embedder + a
    hermetic config (no HyDE / no cross-encoder - both would add a non-stubbed
    LLM call and stochastic ordering), the same query on the same data must
    return the exact same chunk order every time. Two INDEPENDENT ingests are
    NOT compared: candidate ties tie-break on the per-ingest-random document
    UUID, so the tail legitimately reorders across ingests - but a single
    ingest is fully reproducible, which is what a golden run in CI relies on.
    """
    from khora import Khora, SearchMode

    corpus = _load_corpus()
    vocab = _build_vocabulary(corpus)
    vocab_embedding, dim = _make_vocab_embedding(vocab)
    stub_llm(monkeypatch, dim=dim)
    _patch_vocab_embedder(monkeypatch, vocab_embedding)

    query = next(q for q in corpus["queries"] if q["query_id"] == "multi_entity_alice_bob")["query"]

    tmp_path.mkdir(parents=True, exist_ok=True)
    async with Khora(_config(tmp_path, dim), run_migrations=True) as kb:
        ns = await kb.create_namespace()
        for doc in corpus["documents"]:
            await kb.remember(
                doc["content"],
                namespace=ns.namespace_id,
                entity_types=["PERSON", "PLACE", "ELEMENT", "PROJECT", "CONCEPT"],
                relationship_types=["RELATED_TO", "DISCOVERED", "OWNS", "MENTIONS"],
            )
        orders = []
        for _ in range(3):
            res = await kb.recall(query, namespace=ns.namespace_id, limit=10, mode=SearchMode.HYBRID)
            orders.append([str(c.id) for c in res.chunks])

    assert orders[0] == orders[1] == orders[2], (
        "recall order is not deterministic across repeated identical recalls:\n"
        + "\n".join(f"  run {i}: {o}" for i, o in enumerate(orders))
    )
