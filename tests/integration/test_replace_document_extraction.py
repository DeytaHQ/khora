"""End-to-end replace-via-remember() integration tests.

Exercises the document-replacement lifecycle through the public
``Khora.remember()`` / ``VectorCypherEngine.remember()`` API surface,
against a real Postgres + Neo4j stack. The sibling file
``test_coordinator_replace_document_extraction.py`` tests the coordinator
primitive directly; this file tests the public entry point where an
``external_id`` collision dispatches to ``_remember_via_replace``.

Scenarios (one test method each):

1. Golden path: same ``external_id`` replaces in-place (orphan retire,
   survivor remap, net-new upsert, relationship retire).
2. Co-sourced entity survives retirement when one of two source docs is
   replaced.
3. Co-sourced relationship survives retirement likewise.
4. Atomic chunk replacement: Postgres failure rolls back the chunk write
   and graph mutations do not leak through.
5. Graph-side failure marks the document ``FAILED``; the next successful
   replace self-heals it to ``COMPLETED``.
6. Concurrent ``remember()`` calls with the same ``external_id`` converge
   on a single document (engine.py:641 IntegrityError -> _remember_via_replace).
7. ``prefer_current`` filter symmetry — retired edges are excluded when
   the traversal applies the ``valid_until`` predicate and visible
   otherwise. Anchored to raw Cypher that mirrors
   ``DualNodeManager.get_entity_neighborhoods`` (dual_nodes.py:595-600)
   because neither ``Khora.recall()`` nor
   ``GraphBackend.get_neighborhoods_batch`` expose ``prefer_current``
   publicly.
8. Backward compat: when ``external_id=None``, checksum dedup still
   returns ``duplicate=True`` without routing through the replace path.

Gated by ``NEO4J_INTEGRATION_TEST=1`` because CI does not provision Neo4j.

How to run locally::

    make dev  # postgres + neo4j via docker compose
    NEO4J_INTEGRATION_TEST=1 uv run pytest \\
        tests/integration/test_replace_document_extraction.py -v

Connection parameters (env overrides, sensible ``make dev`` defaults)::

    KHORA_NEO4J_URL          (default: bolt://localhost:7687)
    KHORA_NEO4J_USERNAME     (default: neo4j)
    KHORA_NEO4J_PASSWORD     (default: password)
    KHORA_DATABASE_URL       (default: postgresql+asyncpg://khora:khora@localhost:5432/khora)

Embedding dimension is pinned to 4 (matching the sibling test) via an
``embed_batch`` monkeypatch. All entity extraction is driven by a
registry-based stub that matches markers in the document content.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy.exc import IntegrityError

from khora.config import KhoraConfig
from khora.core.models.document import DocumentStatus
from khora.extraction.extractors.base import (
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
)
from khora.khora import Khora, RememberResult

EMBED_DIM = 4

# Module-scoped extraction registry. ``plan_extraction`` stages the
# ExtractionResult returned for any text containing ``marker``; the
# stub scans registered markers in insertion order and returns the
# first match. Entities/relationships use confidence=0.99 to clear the
# default 0.5 min_entity_confidence / min_relationship_confidence gate.
_EXTRACTION_REGISTRY: dict[str, ExtractionResult] = {}


def plan_extraction(
    marker: str,
    entities: list[tuple[str, str]],
    relationships: list[tuple[str, str, str]],
) -> None:
    """Stage an extraction result for texts containing ``marker``.

    Args:
        marker: substring that identifies which documents return this result.
        entities: list of ``(name, entity_type)`` tuples.
        relationships: list of ``(source_name, target_name, rel_type)`` tuples.
    """
    _EXTRACTION_REGISTRY[marker] = ExtractionResult(
        entities=[ExtractedEntity(name=n, entity_type=t, confidence=0.99) for n, t in entities],
        relationships=[
            ExtractedRelationship(
                source_entity=s,
                target_entity=t,
                relationship_type=rt,
                confidence=0.99,
            )
            for s, t, rt in relationships
        ],
    )


async def _stub_extract_multi(
    self: Any,
    texts: list[str],
    **_kwargs: Any,
) -> list[ExtractionResult]:
    """Registry-based stub for ``LLMEntityExtractor.extract_multi``.

    Returns the first registered result whose marker appears in each
    input text; otherwise returns an empty ExtractionResult.
    """
    out: list[ExtractionResult] = []
    for text in texts:
        matched = next(
            (result for marker, result in _EXTRACTION_REGISTRY.items() if marker in text),
            None,
        )
        out.append(matched if matched is not None else ExtractionResult())
    return out


async def _stub_embed_batch(self: Any, texts: list[str]) -> list[list[float]]:
    """Deterministic unit vector embedder stub."""
    unit = [1.0] + [0.0] * (EMBED_DIM - 1)
    return [unit[:] for _ in texts]


# ----------------------------------------------------------------------
# Cypher helpers
# ----------------------------------------------------------------------


async def _run_cypher(driver: Any, query: str, **params: Any) -> list[dict[str, Any]]:
    async with driver.session() as session:
        result = await session.run(query, **params)
        return await result.data()


async def _get_entity_versions(driver: Any, ns: str, name: str) -> list[dict[str, Any]]:
    return await _run_cypher(
        driver,
        """
        MATCH (v:EntityVersion {namespace_id: $ns, name: $name})
        RETURN v.id AS id,
               v.retirement_reason AS reason,
               v.version_valid_to AS version_valid_to,
               v.source_document_ids AS sdids
        """,
        ns=ns,
        name=name,
    )


async def _get_supersedes_edges(driver: Any, ns: str, name: str) -> list[dict[str, Any]]:
    return await _run_cypher(
        driver,
        """
        MATCH (current:Entity {namespace_id: $ns, name: $name})
              -[s:SUPERSEDES]->(old:EntityVersion)
        RETURN s.reason AS reason,
               s.superseded_at AS superseded_at,
               old.id AS old_id,
               current.valid_until AS current_valid_until
        """,
        ns=ns,
        name=name,
    )


async def _get_entity_by_name(driver: Any, ns: str, name: str) -> dict[str, Any] | None:
    rows = await _run_cypher(
        driver,
        """
        MATCH (e:Entity {namespace_id: $ns, name: $name})
        RETURN e.id AS id,
               e.source_document_ids AS sdids,
               e.valid_until AS valid_until
        """,
        ns=ns,
        name=name,
    )
    return rows[0] if rows else None


async def _get_relationship(
    driver: Any,
    ns: str,
    src_name: str,
    tgt_name: str,
    rel_type: str | None = None,
) -> dict[str, Any] | None:
    """Fetch a relationship between two entities.

    Filters by ``rel_type`` when provided to defeat co-occurrence noise:
    ``_build_cooccurrence_relationships`` adds ``ASSOCIATED_WITH`` edges
    when 2+ entities appear in the same chunk, which can shadow a
    specific KNOWS edge the caller wanted to inspect.
    """
    rows = await _run_cypher(
        driver,
        """
        MATCH (s:Entity {namespace_id: $ns, name: $src_name})
              -[r]->(t:Entity {namespace_id: $ns, name: $tgt_name})
        WHERE ($rel_type IS NULL OR type(r) = $rel_type)
        RETURN r.id AS id,
               type(r) AS type,
               r.valid_until AS valid_until,
               r.source_document_ids AS sdids
        LIMIT 1
        """,
        ns=ns,
        src_name=src_name,
        tgt_name=tgt_name,
        rel_type=rel_type,
    )
    return rows[0] if rows else None


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("NEO4J_INTEGRATION_TEST"),
    reason="set NEO4J_INTEGRATION_TEST=1 to run against real backends (requires make dev)",
)
class TestReplaceViaRememberIntegration:
    """End-to-end replace lifecycle through ``Khora.remember()``."""

    # ------------------------------------------------------------------
    # Fixtures
    # ------------------------------------------------------------------

    @pytest.fixture(autouse=True)
    def _stub_extractor_and_embedder(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Install the registry stub + deterministic embedder.

        ``extract_multi`` is patched at the class level because
        ``extract_entities`` instantiates a fresh ``LLMEntityExtractor`` per
        call (extract.py:142). The embedder is likewise patched at the
        class level so the engine-owned instance picks it up.
        """
        _EXTRACTION_REGISTRY.clear()
        monkeypatch.setattr(
            "khora.extraction.extractors.llm.LLMEntityExtractor.extract_multi",
            _stub_extract_multi,
        )
        monkeypatch.setattr(
            "khora.extraction.embedders.litellm.LiteLLMEmbedder.embed_batch",
            _stub_embed_batch,
        )

    @pytest.fixture(scope="class")
    async def kb(self) -> AsyncIterator[Khora]:
        database_url = os.environ.get(
            "KHORA_DATABASE_URL",
            "postgresql+asyncpg://khora:khora@localhost:5432/khora",
        )
        neo4j_url = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7687")
        neo4j_user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
        neo4j_password = os.environ.get("KHORA_NEO4J_PASSWORD", "password")

        config = KhoraConfig(database_url=database_url, neo4j_url=neo4j_url)
        # The top-level ``neo4j_url`` shortcut reads credentials from
        # ``storage.neo4j_user`` / ``storage.neo4j_password`` when the URL
        # itself does not embed them (``_parse_neo4j_url``).
        config.storage.neo4j_user = neo4j_user
        config.storage.neo4j_password = neo4j_password
        # Match the sibling's 4-dim stub embeddings so storage + search agree.
        config.llm.embedding_dimension = EMBED_DIM
        config.storage.embedding_dimension = EMBED_DIM
        # Keep documents to a single chunk so co-occurrence edges stay predictable.
        config.pipeline.chunk_size = 1024
        config.pipeline.extract_entities = True
        # Skip selective extraction threshold so every chunk is extracted.
        config.pipeline.selective_extraction = False

        kb = Khora(config, run_migrations=False)
        await kb.connect()
        try:
            yield kb
        finally:
            await kb.disconnect()

    @pytest.fixture
    async def namespace_id(self, kb: Khora) -> UUID:
        ns = await kb.create_namespace()
        return ns.namespace_id

    # ------------------------------------------------------------------
    # Scenario helpers
    # ------------------------------------------------------------------

    def _graph_driver(self, kb: Khora) -> Any:
        graph = kb.storage.graph
        assert graph is not None, "graph backend must be configured"
        driver = getattr(graph, "_driver", None)
        assert driver is not None, "Neo4j driver must be connected"
        return driver

    async def _remember(
        self,
        kb: Khora,
        *,
        namespace_id: UUID,
        content: str,
        external_id: str | None,
    ) -> RememberResult:
        return await kb.remember(
            content=content,
            namespace=namespace_id,
            entity_types=["PERSON", "CONCEPT"],
            relationship_types=["KNOWS", "RELATES_TO"],
            external_id=external_id,
        )

    # ------------------------------------------------------------------
    # 1. Golden path
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_golden_path_replaces_with_same_external_id(self, kb: Khora, namespace_id: UUID) -> None:
        """v1 (alice, bob, KNOWS) → v2 (alice, carol): bob + KNOWS retire, alice remaps, carol net-new."""
        driver = self._graph_driver(kb)
        ns = str(await kb.storage.resolve_namespace(namespace_id))
        ext = f"golden-{uuid4().hex[:8]}"
        alice = f"alice-{uuid4().hex[:6]}"
        bob = f"bob-{uuid4().hex[:6]}"
        carol = f"carol-{uuid4().hex[:6]}"

        plan_extraction(
            "v1-golden",
            entities=[(alice, "PERSON"), (bob, "PERSON")],
            relationships=[(alice, bob, "KNOWS")],
        )
        v1 = await self._remember(
            kb,
            namespace_id=namespace_id,
            content=f"v1-golden story about {alice} and {bob}.",
            external_id=ext,
        )

        plan_extraction(
            "v2-golden",
            entities=[(alice, "PERSON"), (carol, "PERSON")],
            relationships=[],
        )
        v2 = await self._remember(
            kb,
            namespace_id=namespace_id,
            content=f"v2-golden note mentioning {alice} and {carol}.",
            external_id=ext,
        )

        # Same document id reused across replacements.
        assert v1.document_id == v2.document_id

        doc = await kb.get_document(v2.document_id)
        assert doc is not None
        assert doc.status == DocumentStatus.COMPLETED
        assert doc.external_id == ext

        # bob is retired: a :EntityVersion snapshot exists with the
        # 'document_replaced' retirement_reason, plus a SUPERSEDES edge
        # from the current :Entity row whose valid_until is stamped.
        bob_versions = await _get_entity_versions(driver, ns, bob)
        assert len(bob_versions) == 1
        assert bob_versions[0]["reason"] == "document_replaced"

        bob_supersedes = await _get_supersedes_edges(driver, ns, bob)
        assert len(bob_supersedes) == 1
        assert bob_supersedes[0]["reason"] == "document_replaced"
        assert bob_supersedes[0]["current_valid_until"] is not None
        assert bob_supersedes[0]["superseded_at"] is not None
        # SUPERSEDES edge must point at the EntityVersion snapshot — not a
        # different (or missing) row. Guards against a producer bug that
        # creates one without the other.
        assert bob_supersedes[0]["old_id"] == bob_versions[0]["id"]

        # alice survives and has been remapped off the old document id.
        # v1 and v2 share the same document id (in-place replace), so the
        # survivor's source_document_ids is unchanged in UUID content but
        # the remap path still executes. Assert alice is alive and
        # associated with the surviving doc id.
        alice_row = await _get_entity_by_name(driver, ns, alice)
        assert alice_row is not None
        assert alice_row["valid_until"] is None
        assert str(v2.document_id) in alice_row["sdids"]

        # carol is net-new.
        carol_row = await _get_entity_by_name(driver, ns, carol)
        assert carol_row is not None
        assert carol_row["valid_until"] is None
        assert str(v2.document_id) in carol_row["sdids"]

        # The KNOWS relationship between alice and bob is retired
        # (valid_until stamped). Filter by rel_type to skip the
        # co-occurrence ASSOCIATED_WITH edge emitted for same-chunk pairs.
        knows = await _get_relationship(driver, ns, alice, bob, rel_type="KNOWS")
        assert knows is not None
        assert knows["valid_until"] is not None

        # Single document in the namespace (list_documents is paginated
        # but we only wrote one external_id).
        docs = await kb.list_documents(namespace=namespace_id)
        matching = [d for d in docs if d.external_id == ext]
        assert len(matching) == 1

    # ------------------------------------------------------------------
    # 2. Co-sourced entity survives retirement
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_cosourced_entity_survives_retirement(self, kb: Khora, namespace_id: UUID) -> None:
        """Dave is written by doc A and doc B; replacing A (minus Dave) leaves Dave alive and sole-sourced to B."""
        driver = self._graph_driver(kb)
        ns = str(await kb.storage.resolve_namespace(namespace_id))
        ext_a = f"cosource-a-{uuid4().hex[:8]}"
        ext_b = f"cosource-b-{uuid4().hex[:8]}"
        dave = f"dave-{uuid4().hex[:6]}"
        eve = f"eve-{uuid4().hex[:6]}"

        plan_extraction(
            "cs-doc-a",
            entities=[(dave, "PERSON")],
            relationships=[],
        )
        result_a = await self._remember(
            kb,
            namespace_id=namespace_id,
            content=f"cs-doc-a mentions {dave}.",
            external_id=ext_a,
        )

        plan_extraction(
            "cs-doc-b",
            entities=[(dave, "PERSON")],
            relationships=[],
        )
        result_b = await self._remember(
            kb,
            namespace_id=namespace_id,
            content=f"cs-doc-b also mentions {dave}.",
            external_id=ext_b,
        )
        # Confirm co-sourcing before replacing.
        dave_pre = await _get_entity_by_name(driver, ns, dave)
        assert dave_pre is not None
        assert str(result_a.document_id) in dave_pre["sdids"]
        assert str(result_b.document_id) in dave_pre["sdids"]

        # Replace doc A with extraction that drops Dave entirely.
        plan_extraction(
            "cs-doc-a-v2",
            entities=[(eve, "PERSON")],
            relationships=[],
        )
        await self._remember(
            kb,
            namespace_id=namespace_id,
            content=f"cs-doc-a-v2 no longer mentions the original person; {eve} stepped in.",
            external_id=ext_a,
        )

        # Dave is still alive (not retired) because doc B still sources him.
        dave_versions = await _get_entity_versions(driver, ns, dave)
        assert dave_versions == []

        dave_row = await _get_entity_by_name(driver, ns, dave)
        assert dave_row is not None
        assert dave_row["valid_until"] is None
        assert str(result_a.document_id) not in dave_row["sdids"]
        assert str(result_b.document_id) in dave_row["sdids"]

    # ------------------------------------------------------------------
    # 3. Co-sourced relationship survives retirement
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_cosourced_relationship_survives_retirement(self, kb: Khora, namespace_id: UUID) -> None:
        """KNOWS(alice,bob) is asserted by docs A + B; replacing A without KNOWS leaves the edge alive."""
        driver = self._graph_driver(kb)
        ns = str(await kb.storage.resolve_namespace(namespace_id))
        ext_a = f"rel-a-{uuid4().hex[:8]}"
        ext_b = f"rel-b-{uuid4().hex[:8]}"
        alice = f"alice-{uuid4().hex[:6]}"
        bob = f"bob-{uuid4().hex[:6]}"

        plan_extraction(
            "rel-doc-a",
            entities=[(alice, "PERSON"), (bob, "PERSON")],
            relationships=[(alice, bob, "KNOWS")],
        )
        await self._remember(
            kb,
            namespace_id=namespace_id,
            content=f"rel-doc-a: {alice} knows {bob}.",
            external_id=ext_a,
        )

        plan_extraction(
            "rel-doc-b",
            entities=[(alice, "PERSON"), (bob, "PERSON")],
            relationships=[(alice, bob, "KNOWS")],
        )
        result_b = await self._remember(
            kb,
            namespace_id=namespace_id,
            content=f"rel-doc-b: {alice} also knows {bob}.",
            external_id=ext_b,
        )

        # Replace A keeping alice+bob but dropping KNOWS.
        plan_extraction(
            "rel-doc-a-v2",
            entities=[(alice, "PERSON"), (bob, "PERSON")],
            relationships=[],
        )
        await self._remember(
            kb,
            namespace_id=namespace_id,
            content=f"rel-doc-a-v2: {alice} and {bob} appear but the edge is gone.",
            external_id=ext_a,
        )

        # The KNOWS edge must survive because doc B still asserts it. The
        # post-replace state has valid_until IS NULL and source_document_ids
        # contains the doc B id only. Filter by rel_type to skip the
        # co-occurrence ASSOCIATED_WITH edge emitted for same-chunk pairs.
        knows = await _get_relationship(driver, ns, alice, bob, rel_type="KNOWS")
        assert knows is not None
        assert knows["valid_until"] is None
        assert str(result_b.document_id) in knows["sdids"]

    # ------------------------------------------------------------------
    # 4. Atomic chunk replacement — Postgres rolls back
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_atomic_chunk_replacement_pg_rolls_back(
        self,
        kb: Khora,
        namespace_id: UUID,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Raise inside the PG replace txn; chunks table and graph must be unchanged."""
        driver = self._graph_driver(kb)
        ns = str(await kb.storage.resolve_namespace(namespace_id))
        ext = f"rollback-{uuid4().hex[:8]}"
        alice = f"alice-{uuid4().hex[:6]}"

        plan_extraction(
            "rollback-v1",
            entities=[(alice, "PERSON")],
            relationships=[],
        )
        v1 = await self._remember(
            kb,
            namespace_id=namespace_id,
            content=f"rollback-v1 doc with {alice}.",
            external_id=ext,
        )

        # Snapshot PG chunk count BEFORE the replace attempt. The v1
        # create path writes to the engine-owned ``khora_chunks`` table
        # via TemporalVectorStore, NOT to the coordinator's ``chunks``
        # table, so this count starts at 0 and must stay at 0 after the
        # aborted replace (PG rollback is the invariant).
        chunks_before = await kb.storage.get_chunks_by_document(v1.document_id, namespace_id=namespace_id)

        vector_backend = kb.storage.vector
        assert vector_backend is not None
        call_count = {"n": 0}

        async def flaky_create(chunks: Any, *args: Any, **kwargs: Any) -> Any:
            # The replace path always passes session=... via the coordinator's
            # transaction context; the v1 create path writes through the
            # temporal store and does NOT go through this method. Raising
            # unconditionally is safe.
            call_count["n"] += 1
            raise RuntimeError("injected vector failure")

        monkeypatch.setattr(vector_backend, "create_chunks_batch", flaky_create)

        plan_extraction(
            "rollback-v2",
            entities=[(alice, "PERSON")],
            relationships=[],
        )
        with pytest.raises(RuntimeError, match="injected vector failure"):
            await self._remember(
                kb,
                namespace_id=namespace_id,
                content=f"rollback-v2 replacement content with {alice} again.",
                external_id=ext,
            )

        # Prove the coordinator txn actually reached the vector write:
        # the engine performs a pre-coordinator chunk wipe at
        # engine.py:1391-1405, and if that threw first the test would
        # silently pass without exercising the PG rollback path.
        assert call_count["n"] >= 1, "coordinator txn never invoked vector.create_chunks_batch"

        # PG chunks: unchanged by the rolled-back transaction.
        chunks_after = await kb.storage.get_chunks_by_document(v1.document_id, namespace_id=namespace_id)
        assert len(chunks_after) == len(chunks_before)

        # Document status is FAILED (coordinator's error path marks the row
        # on exception).
        failed = await kb.get_document(v1.document_id)
        assert failed is not None
        assert failed.status == DocumentStatus.FAILED

        # Graph: alice must not be retired. No :EntityVersion snapshot,
        # valid_until still NULL on the current row. Retirement runs
        # AFTER the PG txn commits (coordinator.py:597-607), so a failure
        # inside the txn must prevent graph mutation entirely.
        alice_versions = await _get_entity_versions(driver, ns, alice)
        assert alice_versions == []
        alice_row = await _get_entity_by_name(driver, ns, alice)
        assert alice_row is not None
        assert alice_row["valid_until"] is None

    # ------------------------------------------------------------------
    # 5. Graph failure marks FAILED and the next replace self-heals
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_graph_failure_marks_failed_then_heals(
        self,
        kb: Khora,
        namespace_id: UUID,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Inject a retire failure, observe FAILED, then let a retry succeed and heal to COMPLETED."""
        ext = f"heal-{uuid4().hex[:8]}"
        doomed = f"doomed-{uuid4().hex[:6]}"

        plan_extraction(
            "heal-v1",
            entities=[(doomed, "PERSON")],
            relationships=[],
        )
        v1 = await self._remember(
            kb,
            namespace_id=namespace_id,
            content=f"heal-v1 with {doomed}.",
            external_id=ext,
        )

        graph = kb.storage.graph
        assert graph is not None
        orig_retire = graph.retire_orphaned_entities_batch  # type: ignore[unresolved-attribute]
        call_count = {"n": 0}

        async def flaky_retire(*args: Any, **kwargs: Any) -> int:
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("injected graph failure")
            return await orig_retire(*args, **kwargs)

        monkeypatch.setattr(graph, "retire_orphaned_entities_batch", flaky_retire)

        # v2 retirement fails mid-way → document lands in FAILED.
        plan_extraction(
            "heal-v2",
            entities=[],  # orphans doomed
            relationships=[],
        )
        with pytest.raises(RuntimeError, match="injected graph failure"):
            await self._remember(
                kb,
                namespace_id=namespace_id,
                content="heal-v2 no entities at all.",
                external_id=ext,
            )

        failed = await kb.get_document(v1.document_id)
        assert failed is not None
        assert failed.status == DocumentStatus.FAILED

        # v3 retirement succeeds (second call delegates to original) →
        # document self-heals back to COMPLETED.
        plan_extraction(
            "heal-v3",
            entities=[],
            relationships=[],
        )
        await self._remember(
            kb,
            namespace_id=namespace_id,
            content="heal-v3 still no entities; retry should heal.",
            external_id=ext,
        )

        healed = await kb.get_document(v1.document_id)
        assert healed is not None
        assert healed.status == DocumentStatus.COMPLETED
        # mark_completed() must also null out the stale error_message from
        # the prior FAILED state; detect regressions where healing leaves
        # the old message behind.
        assert healed.error_message is None

    # ------------------------------------------------------------------
    # 6. Concurrent remember() calls converge on a single document
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_concurrent_external_id_converges_to_single_document(self, kb: Khora, namespace_id: UUID) -> None:
        """Two parallel remember() calls with the same external_id converge on one document.

        Per engine.py:641-652, the loser catches IntegrityError and
        re-lookups ``get_document_by_external_id``; if the winner has not
        committed yet, the handler re-raises the IntegrityError. So under
        adversarial race timing one task may legitimately surface
        IntegrityError. Accept either outcome:
          (a) both tasks return RememberResult, OR
          (b) one RememberResult + one IntegrityError.
        The invariant in either case is exactly one COMPLETED document
        row carries this external_id.
        """
        ext = f"conc-{uuid4().hex[:8]}"
        alice = f"alice-{uuid4().hex[:6]}"
        bob = f"bob-{uuid4().hex[:6]}"

        plan_extraction(
            "conc-left",
            entities=[(alice, "PERSON")],
            relationships=[],
        )
        plan_extraction(
            "conc-right",
            entities=[(bob, "PERSON")],
            relationships=[],
        )

        results = await asyncio.gather(
            self._remember(
                kb,
                namespace_id=namespace_id,
                content=f"conc-left about {alice}.",
                external_id=ext,
            ),
            self._remember(
                kb,
                namespace_id=namespace_id,
                content=f"conc-right about {bob}.",
                external_id=ext,
            ),
            return_exceptions=True,
        )

        # Every outcome must be either a RememberResult or an IntegrityError.
        unexpected = [r for r in results if not isinstance(r, (RememberResult, IntegrityError))]
        assert unexpected == [], f"unexpected exception bubbled up: {unexpected}"
        successes = [r for r in results if isinstance(r, RememberResult)]
        assert len(successes) >= 1, f"at least one call must succeed, got {results!r}"

        # Invariant: exactly one document row carries this external_id,
        # in COMPLETED status.
        docs = await kb.list_documents(namespace=namespace_id)
        matching = [d for d in docs if d.external_id == ext]
        assert len(matching) == 1
        assert matching[0].status == DocumentStatus.COMPLETED

        # Every RememberResult references that single document id.
        for r in successes:
            assert r.document_id == matching[0].id
        # If both branches succeeded, their document ids agree trivially
        # (both equal matching[0].id above).

    # ------------------------------------------------------------------
    # 7. Recall filter symmetry — prefer_current
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_recall_filter_symmetry_prefer_current(self, kb: Khora, namespace_id: UUID) -> None:
        """Retired edges are hidden when prefer_current=True, visible otherwise.

        ``Khora.recall()`` does not expose ``prefer_current`` as a
        public kwarg (see plan §6 open question #1). The graph-backend
        ``get_neighborhoods_batch`` also does not take ``prefer_current``;
        only ``DualNodeManager.get_entity_neighborhoods`` (used by the
        VectorCypher retriever for temporal queries) does. Rather than
        reach into engine internals, this test asserts the underlying
        invariant with raw Cypher that mirrors the
        ``DualNodeManager.get_entity_neighborhoods`` temporal predicate
        (``dual_nodes.py:595-600``): a retired edge must filter out when
        ``valid_until`` has passed.
        """
        driver = self._graph_driver(kb)
        ns_str = str(await kb.storage.resolve_namespace(namespace_id))
        ext = f"temporal-{uuid4().hex[:8]}"
        alice = f"alice-{uuid4().hex[:6]}"
        bob = f"bob-{uuid4().hex[:6]}"

        plan_extraction(
            "temporal-v1",
            entities=[(alice, "PERSON"), (bob, "PERSON")],
            relationships=[(alice, bob, "KNOWS")],
        )
        await self._remember(
            kb,
            namespace_id=namespace_id,
            content=f"temporal-v1: {alice} knows {bob}.",
            external_id=ext,
        )

        # Replace with v2 that drops bob and KNOWS → KNOWS is retired.
        plan_extraction(
            "temporal-v2",
            entities=[(alice, "PERSON")],
            relationships=[],
        )
        await self._remember(
            kb,
            namespace_id=namespace_id,
            content=f"temporal-v2: only {alice} now.",
            external_id=ext,
        )

        # Confirm KNOWS is retired before inspecting neighborhoods.
        # Filter by rel_type to skip the co-occurrence ASSOCIATED_WITH edge.
        knows = await _get_relationship(driver, ns_str, alice, bob, rel_type="KNOWS")
        assert knows is not None
        assert knows["valid_until"] is not None

        # prefer_current=False equivalent: traverse without a temporal
        # predicate. The retired KNOWS edge is still traversable, so bob
        # is reachable from alice.
        rows_all = await _run_cypher(
            driver,
            """
            MATCH (e:Entity {namespace_id: $ns, name: $alice})
            OPTIONAL MATCH (e)-[r]-(related:Entity)
            WHERE related.namespace_id = $ns
              AND related.id <> e.id
            RETURN collect(DISTINCT related.name) AS names
            """,
            ns=ns_str,
            alice=alice,
        )
        names_all = set(rows_all[0]["names"]) if rows_all else set()
        assert bob in names_all

        # prefer_current=True equivalent: apply the same temporal
        # predicate that ``DualNodeManager.get_entity_neighborhoods``
        # emits — a relationship is excluded if its ``valid_until`` has
        # passed. The retired KNOWS disappears, so bob is unreachable.
        rows_current = await _run_cypher(
            driver,
            """
            MATCH (e:Entity {namespace_id: $ns, name: $alice})
            WITH e, datetime() AS _now
            OPTIONAL MATCH path = (e)-[*1..1]-(related:Entity)
            WHERE related.namespace_id = $ns
              AND related.id <> e.id
              AND (related.valid_until IS NULL OR related.valid_until > _now)
              AND all(r IN relationships(path) WHERE r.valid_until IS NULL OR r.valid_until > _now)
            RETURN collect(DISTINCT related.name) AS names
            """,
            ns=ns_str,
            alice=alice,
        )
        names_current = set(rows_current[0]["names"]) if rows_current else set()
        assert bob not in names_current

    # ------------------------------------------------------------------
    # 8. Backward compat — external_id=None → checksum dedup, no replace
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_backward_compat_external_id_none_checksum_dedup(
        self,
        kb: Khora,
        namespace_id: UUID,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Without external_id, identical content still dedups via checksum and never reaches replace."""
        content = f"legacy-{uuid4().hex[:8]} same content twice"
        alice = f"alice-{uuid4().hex[:6]}"
        plan_extraction(
            "legacy-",
            entities=[(alice, "PERSON")],
            relationships=[],
        )

        # Spy on coordinator.replace_document_extraction to ensure the
        # checksum-dedup path never invokes it.
        storage = kb.storage
        orig_replace = storage.replace_document_extraction
        replace_calls = {"n": 0}

        async def spy_replace(*args: Any, **kwargs: Any) -> Any:
            replace_calls["n"] += 1
            return await orig_replace(*args, **kwargs)

        monkeypatch.setattr(storage, "replace_document_extraction", spy_replace)

        first = await self._remember(
            kb,
            namespace_id=namespace_id,
            content=content,
            external_id=None,
        )
        second = await self._remember(
            kb,
            namespace_id=namespace_id,
            content=content,
            external_id=None,
        )

        # Second call hits checksum dedup and returns duplicate=True.
        assert second.document_id == first.document_id
        assert second.metadata.get("duplicate") is True

        # Only one document row for this checksum in the namespace.
        docs = await kb.list_documents(namespace=namespace_id)
        matching = [d for d in docs if d.id == first.document_id]
        assert len(matching) == 1

        # No EntityVersion snapshots were produced — the second
        # remember() did NOT route through the replace path.
        driver = self._graph_driver(kb)
        ns_str = str(await kb.storage.resolve_namespace(namespace_id))
        versions = await _get_entity_versions(driver, ns_str, alice)
        assert versions == []

        assert replace_calls["n"] == 0
