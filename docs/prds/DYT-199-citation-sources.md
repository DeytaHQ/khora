# DYT-199: Citation Sources on Chunk and Entity

**Status:** Draft
**Author:** Filip
**Date:** 2026-03-03
**Linear Ticket:** DYT-199

---

### Problem Statement

Downstream consumers call `lake.recall()`, `lake.search_entities()`, and `lake.find_related_entities()` to power synthesis and retrieval features. These consumers need to build deterministic citation lists from the results — linking each piece of retrieved content back to its origin document with a human-readable title and clickable URL.

Today, chunks only expose a raw `document_id` (UUID) and entities expose `source_document_ids` (a UUID array). There is no way to get a document's title, URL, or source tool without a separate database lookup, which adds latency and forces every consumer to implement the same resolution logic.

A secondary issue compounds this: the `source_tool` field (canonical SaaS tool identifier like "slack", "linear") exists on the `DocumentMetadata` and `Entity` dataclasses but is silently dropped during persistence — never written to PostgreSQL or Neo4j. Even if consumers did their own resolution, the tool identifier would be empty.

### Goals

- Every `Chunk` returned by `recall()` carries a `source: Source` object with `document_id`, `title`, `url`, `source_type`, and `source_tool` — populated at ingest time with zero additional queries on read
- Every `Entity` returned by `search_entities()` and `find_related_entities()` carries a `source_documents: list[Source]` with the same fields — resolved with < 5ms added latency
- A single `Source` dataclass provides a consistent citation format across all khora API responses
- The `source_tool` persistence bug is fixed: values round-trip correctly through PostgreSQL (documents) and Neo4j (entities)
- The `remember()` API accepts `source_tool` as a first-class parameter so consumers can pass it directly

### Non-Goals

- **Full denormalization of entity sources** — Entity source resolution happens at read time via a single batch query. Denormalizing into Neo4j would require complex Cypher merge logic (APOC dependency or multi-pass upserts) for a < 1ms improvement; not justified now.
- **Citation formatting or rendering** — Khora provides structured `Source` objects. Formatting them into markdown, HTML, or numbered references is the consumer's responsibility.
- **Retroactive source_tool backfill** — Existing documents ingested without `source_tool` will have it empty. Consumers can re-ingest to populate. The migration backfills chunk source metadata from parent documents but cannot recover `source_tool` data that was never stored.
- **Breaking RecallResult structure** — No new `sources` list on RecallResult. Sources live directly on Chunk and Entity objects.

### Requirements

#### Functional Requirements

1. **FR-1:** The `Chunk` dataclass must include a `source: Source | None` field containing `document_id`, `title`, `url`, `source_type`, and `source_tool` from the parent document.
2. **FR-2:** The `Entity` dataclass must include a `source_documents: list[Source]` field, resolved from the entity's `source_document_ids` array.
3. **FR-3:** A frozen, slotted `Source` dataclass must be defined in `core/models/source.py` and exported from `core/models/__init__.py`.
4. **FR-4:** `chunk.source` must be populated during document ingestion (in the chunking pipeline), denormalized into the `chunks` database table, and read back without additional queries.
5. **FR-5:** `entity.source_documents` must be populated at read time in the `MemoryLake` facade methods (`recall`, `search_entities`, `find_related_entities`) using a single batched document fetch.
6. **FR-6:** `DocumentModel` must persist `source_tool` to PostgreSQL. The `_create_document_with`, `_update_document_with`, and `_document_model_to_domain` methods must include `source_tool`.
7. **FR-7:** Neo4j entity operations must persist and read `source_tool` via `_entity_to_cypher_params`, `_record_to_entity`, `update_entity`, and `upsert_entities_batch`.
8. **FR-8:** `MemoryLake.remember()` and `remember_batch()` must accept a `source_tool` parameter and pass it through to `DocumentMetadata`.
9. **FR-9:** An Alembic migration must add `source_title`, `source_url`, `source_type`, `source_tool` columns to the `chunks` table and `source_tool` to the `documents` table, with a backfill step that populates existing chunks from their parent document.

#### Non-Functional Requirements

1. **NFR-1:** Chunk source retrieval must add 0ms latency to `recall()` — data is denormalized at ingest.
2. **NFR-2:** Entity source resolution must add < 5ms (p95) latency to `recall()`, `search_entities()`, and `find_related_entities()`.
3. **NFR-3:** The migration must be reversible (downgrade drops added columns).
4. **NFR-4:** All existing tests must continue to pass. New test coverage for citation resolution and source_tool persistence.
5. **NFR-5:** `make lint` (ruff + ty) must pass clean after all changes.

### User Stories

- As a **downstream developer building synthesis**, I want `recall()` results to include document title and URL on each chunk so that I can build a citation list without extra database lookups.
- As a **downstream developer building synthesis**, I want entities from `search_entities()` to include their source documents so that I can attribute extracted knowledge to its origin.
- As a **khora consumer**, I want to pass `source_tool="slack"` to `remember()` so that recalled chunks carry the tool provenance for display in citations.
- As a **khora consumer**, I want a consistent `Source` type across all API responses so that my citation rendering code works uniformly regardless of whether the source is a chunk or an entity.
- As a **khora library maintainer**, I want the `source_tool` persistence bug fixed so that provenance data is no longer silently lost.

### Technical Approach

**Source model** — Frozen, slotted `Source` dataclass in `core/models/source.py` alongside the existing `SourceTool` enum. Fields: `document_id`, `title`, `url`, `source_type` (transport: file/url/api), `source_tool` (canonical SaaS tool: slack/linear).

**Chunk denormalization (write-time)** — During `chunk_document()`, build a `Source` from the parent `Document`'s metadata and attach to each `Chunk`. Store as four columns on `ChunkModel` (`source_title`, `source_url`, `source_type`, `source_tool`). Read back in `_chunk_model_to_domain()`. Zero additional queries on read.

**Entity resolution (read-time)** — In `MemoryLake._resolve_entity_sources()`, collect all `source_document_ids` from returned entities, batch-fetch via `StorageCoordinator.get_documents_batch()` (single SQL query), and populate `entity.source_documents`. Called from `recall()`, `search_entities()`, and `find_related_entities()`.

**source_tool persistence fix** — Add `source_tool` column to `DocumentModel`, fix write/read in PostgreSQL backend. Add `source_tool` to Neo4j entity serialization/deserialization and upsert Cypher.

**Migration** — Single Alembic migration adds columns to `chunks` and `documents` tables, backfills existing chunks from parent documents.

See the implementation plan at `.claude/plans/clever-snacking-pie.md` for file-level detail.

### Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Chunk source availability | 100% of chunks from `recall()` have non-null `source` | Unit + integration tests |
| Entity source availability | 100% of entities from `search_entities()`/`find_related_entities()` have populated `source_documents` | Unit + integration tests |
| Entity resolution latency | < 5ms p95 added | Logfire span measurement on `_resolve_entity_sources` |
| source_tool round-trip | Values survive write → read for both documents (PG) and entities (Neo4j) | Persistence tests |
| Test suite | All existing tests pass, new citation tests added | `make test` |
| Downstream unblocked | Consumers can build citation lists from khora responses | Manual integration verification |

### Open Questions

- [x] Should entities with deleted source documents show a stub `Source(document_id=...)` with empty metadata, or should those entries be filtered out? Answer: stub with empty fields — preserves reference count, no silent data loss, consumer can detect and handle
- [x] Should the chunk backfill migration also populate `source_tool` from `documents.metadata_->>'source_tool'` (JSONB) for documents that stored it in custom metadata before the column existed? Answer: yes

### Timeline

| Phase | Milestone | Scope |
|-------|-----------|-------|
| 1 | Foundation | Source dataclass, model fields on Chunk/Entity, exports |
| 2 | Persistence | Migration, DocumentModel source_tool, chunk denormalization in ingest + storage |
| 3 | Read enrichment | Entity resolution in MemoryLake facade, source_tool Neo4j fix |
| 4 | API | `remember()` source_tool parameter |
| 5 | Tests + verification | Unit tests, lint, migration verification |
