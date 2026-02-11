# Namespace Optimization: Final Implementation Plan v2

**Author:** architect (synthesized from team analysis)
**Date:** 2026-02-11
**Branch:** `ib/namespace-optimizations`
**Status:** Approved for implementation

> **Note:** This is v2, superseding the initial plan. The team converged on the
> `accessible_doc_ids` pattern (post-filter with existing columns) rather than
> adding `namespace_ids[]` columns and Neo4j schema changes.

---

## 1. Architecture Summary

### Current State
- **TenancyMode** enum (SHARED/ISOLATED) exists but is dead code — never checked by any engine, backend, or query path.
- **Namespace isolation** relies exclusively on `namespace_id` foreign keys in PostgreSQL and property filters in Neo4j. This works for ISOLATED mode by definition (each namespace = separate silo).
- **ACL framework** (`ACLChecker`, `ACLEnforcer`) is fully implemented but disabled at runtime (`api/deps.py` disables enforcement).
- **Orphan entity bug:** `StorageCoordinator.delete_document()` and all engine `forget()` methods leave orphaned entities and relationships in Neo4j when documents are deleted.
- **Bi-temporal model:** `TemporalEdgeModel`, `TimeNodeModel`, and time hierarchy code exist but are never called by any engine. The ingest pipeline discards LLM-extracted temporal info.

### Target State
Two namespace modes, selectable per-namespace:

| Mode | Storage Model | Access Control | Entity Dedup | Overhead |
|------|--------------|----------------|--------------|----------|
| **ISOLATED** (default) | Separate `namespace_id` per silo | None needed (namespace = boundary) | Within namespace | **Zero overhead** vs today |
| **SHARED** (opt-in) | Shared `namespace_id` per workspace | `accessible_doc_ids` post-filter | Within namespace (Phase 1), cross-namespace (future) | ~5-20ms per query |

### Key Design Decision: `accessible_doc_ids` Pattern

The universal interface for shared-mode access filtering:
- **`accessible_doc_ids: set[UUID] | None`** passed to all search methods
- **`None`** = ISOLATED mode → no filtering, existing behavior unchanged
- **`set()`** (empty) = no access → short-circuit, return `[]`
- **`set(...)`** = SHARED mode → filter by document provenance

**Why this approach:**
1. **No schema migration on existing tables** — uses `source_document_ids` (entities) and `document_id` (chunks) already present
2. **No Neo4j changes** — post-filter in Python, zero Cypher modifications
3. **Document-level granularity** — more precise than namespace-level; a user can access specific documents within a shared namespace
4. **O(1) access changes** — insert/delete a row in `document_access`, no entity/chunk rewrite
5. **Single source of truth** — `document_access` table, not replicated across stores

**Trade-off accepted:** Post-filtering means Neo4j may traverse through inaccessible entities to discover accessible ones. This is correct under the Union strategy: entity visibility is determined by data provenance (source documents), not discovery path. The graph is a discovery mechanism; access is data-level.

---

## 2. Data Model Changes

### 2.1 New Table: `document_access`

```sql
CREATE TABLE document_access (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    principal_type VARCHAR(20) NOT NULL,  -- 'user', 'role', 'api_key'
    principal_id VARCHAR(255) NOT NULL,
    permission VARCHAR(20) NOT NULL DEFAULT 'read',  -- 'read', 'write', 'admin'
    granted_by VARCHAR(255),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, principal_type, principal_id, permission)
);

CREATE INDEX ix_document_access_principal
ON document_access (principal_type, principal_id);

CREATE INDEX ix_document_access_document
ON document_access (document_id);
```

### 2.2 New Column: `tenancy_mode` on `memory_namespaces`

```sql
ALTER TABLE memory_namespaces
ADD COLUMN tenancy_mode VARCHAR(10) NOT NULL DEFAULT 'isolated'
  CHECK (tenancy_mode IN ('isolated', 'shared'));
```

### 2.3 New GIN Indexes on Existing Columns

```sql
-- Entity source_document_ids for array overlap queries in shared mode
CREATE INDEX ix_entities_source_doc_ids_gin
ON entities USING GIN (source_document_ids);

-- Relationship source_document_ids for relationship access filtering
CREATE INDEX ix_relationships_source_doc_ids_gin
ON relationships USING GIN (source_document_ids);
```

No GIN index needed for chunks — they have scalar `document_id` with existing B-tree index.

### 2.4 Neo4j Schema Changes

**None for access control.**

Only change: `get_entity_neighborhoods()` in `dual_nodes.py` should return `source_document_ids` in query results to enable Python post-filtering without an extra PostgreSQL round-trip:

```cypher
-- Add to RETURN clause:
related.source_document_ids  -- NEW: enables Python-side access filtering
```

### 2.5 Summary of Schema Impact

| Table | New Columns | New Indexes | New Tables |
|-------|-------------|-------------|------------|
| `memory_namespaces` | `tenancy_mode` | — | — |
| `entities` | — | GIN on `source_document_ids` | — |
| `relationships` | — | GIN on `source_document_ids` | — |
| `chunks` | — | — | — |
| `documents` | — | — | — |
| — | — | — | `document_access` |

**Total: 1 new column, 2 new indexes, 1 new table. Zero changes to existing data.**

---

## 3. AccessFilter Abstraction

New file: `khora/acl/access_filter.py`

```python
@dataclass(frozen=True)
class AccessFilter:
    """Resolved access context for a query.

    In ISOLATED mode: accessible_doc_ids is None (no filtering needed).
    In SHARED mode: accessible_doc_ids is the set of document UUIDs
    the calling principal can access within the workspace.
    """
    mode: TenancyMode
    namespace_id: UUID
    workspace_id: UUID
    accessible_doc_ids: frozenset[UUID] | None  # None = ISOLATED, set = SHARED

    @property
    def is_shared(self) -> bool:
        return self.mode == TenancyMode.SHARED

    @property
    def is_empty(self) -> bool:
        """True if shared mode with no access — short-circuit all queries."""
        return self.accessible_doc_ids is not None and len(self.accessible_doc_ids) == 0
```

**Resolution flow:**
1. `MemoryLake` receives `namespace_id` + optional `principal`
2. Looks up namespace → `tenancy_mode`
3. If ISOLATED: `AccessFilter(accessible_doc_ids=None)` — zero overhead
4. If SHARED: Query `document_access` table for all doc_ids the principal can READ within the workspace → `AccessFilter(accessible_doc_ids=frozenset(...))`
5. Cache with 30s TTL to avoid repeated ACL lookups within a session

**`accessible_doc_ids` resolution query:**
```sql
SELECT DISTINCT da.document_id
FROM document_access da
JOIN documents d ON da.document_id = d.id
JOIN memory_namespaces ns ON d.namespace_id = ns.id
WHERE da.principal_type = $principal_type
  AND da.principal_id = $principal_id
  AND da.permission >= 'read'
  AND ns.workspace_id = $workspace_id;
```

**For large sets (>1000 docs):** Use a CTE or temp table in subsequent queries instead of IN-clause to avoid query plan degradation.

---

## 4. API Changes

### 4.1 MemoryEngineProtocol

```python
async def recall(
    self, query: str, namespace_id: UUID, *,
    access_filter: AccessFilter | None = None,  # NEW — backwards compatible
    limit: int = 10, mode: SearchMode = ..., ...
) -> RecallResult: ...

async def remember(
    self, content: str, namespace_id: UUID, *,
    access_filter: AccessFilter | None = None,  # NEW
    ...
) -> RememberResult: ...
```

When `access_filter is None`: identical to today. When `access_filter.is_shared`: apply document-level filtering.

### 4.2 MemoryLake Facade

```python
async def recall(
    self, query: str, *, namespace: str | UUID | None = None,
    principal: str | None = None,  # NEW
    ...
) -> RecallResult:
    ns_id = await self._resolve_namespace(namespace)
    access_filter = await self._resolve_access(ns_id, principal)
    if access_filter and access_filter.is_empty:
        return RecallResult(memories=[], ...)  # Short-circuit
    return await engine.recall(query, ns_id, access_filter=access_filter, ...)

# New methods
async def grant_access(
    self, document_id: UUID, principal_type: str, principal_id: str,
    permission: str = "read"
) -> None: ...

async def revoke_access(
    self, document_id: UUID, principal_type: str, principal_id: str,
    permission: str = "read"
) -> None: ...
```

### 4.3 Namespace Creation

```python
async def create_namespace(
    self, name: str, workspace_id: UUID, *,
    tenancy_mode: TenancyMode = TenancyMode.ISOLATED,  # NEW
    ...
) -> MemoryNamespace:
```

### 4.4 Document Update API

```python
async def update(
    self, document_id: UUID, *,
    content: str | None = None,     # None = metadata-only update
    namespace: str | UUID | None = None,
    title: str | None = None,
    source: str | None = None,
    metadata: dict[str, Any] | None = None,
    skill_name: str = "general_entities",
) -> RememberResult:
    """Update a document. If content is provided, re-processes fully.
    If content is None, updates only metadata fields."""
```

---

## 5. Query Path Changes (Per Engine)

### 5.1 ISOLATED Mode (all engines)

**Absolutely zero changes.** `access_filter` is `None`, all existing `WHERE namespace_id = $ns` filters remain. This is the critical invariant — default mode must not regress.

### 5.2 SHARED Mode — pgvector Search

**Chunk vector search** (`PgVectorBackend.search_similar`):
```python
if access_filter and access_filter.is_shared:
    query = query.where(
        ChunkModel.document_id.in_([str(d) for d in access_filter.accessible_doc_ids])
    )
```
Uses existing B-tree index on `document_id`. For >1000 docs, use temp table + JOIN.

**Entity vector search** (`PgVectorBackend.search_similar_entities`):
```python
if access_filter and access_filter.is_shared:
    query = query.where(
        EntityModel.source_document_ids.overlap(
            [str(d) for d in access_filter.accessible_doc_ids]
        )
    )
```
Uses new GIN index on `source_document_ids`. Enable iterative scan:
```python
if access_filter and access_filter.is_shared:
    await session.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
```

**Full-text search** (`search_fulltext`):
```python
if access_filter and access_filter.is_shared:
    query = query.where(
        ChunkModel.document_id.in_([str(d) for d in access_filter.accessible_doc_ids])
    )
```
PostgreSQL combines GIN (tsv) + B-tree (document_id) via bitmap AND — efficient.

### 5.3 SHARED Mode — Neo4j Graph Traversal

**No Cypher query changes.** All access filtering is post-filter in Python.

**Post-filter in retriever** (`retriever.py._cypher_expand`):
```python
async def _cypher_expand(self, entry_entity_ids, namespace_id, depth,
                          access_filter=None):
    neighborhoods = await self._dual_nodes.get_entity_neighborhoods(...)

    for source_id, related in neighborhoods.items():
        for entity_info in related:
            if access_filter and access_filter.is_shared:
                entity_doc_ids = set(entity_info.get("source_document_ids", []))
                if not entity_doc_ids & access_filter.accessible_doc_ids:
                    continue  # Entity not visible
```

### 5.4 Multi-Hop Security Analysis

With post-filter (not in-graph), Neo4j may traverse through entities from inaccessible documents:

```
Entity_A (doc1 ✓) --WORKS_WITH→ Entity_B (doc2 ✗) --KNOWS→ Entity_C (doc3 ✓)
```

User with `accessible_doc_ids = {doc1, doc3}`:
- Entity_B filtered out (doc2 not accessible) ✓
- Entity_C returned (doc3 accessible) ✓
- User does NOT learn Entity_B exists or is the connector ✓

This is correct under the **Union strategy**: visibility is based on entity provenance, not discovery path.

---

## 6. Document Update Strategy

### 6.1 Content Update (Diff-and-Reconcile)

```
Phase 1: Snapshot old state
  ├─ Fetch existing document (verify namespace)
  ├─ Fetch old chunks for this document
  ├─ Fetch old entities with source_document_ids containing this doc
  └─ Record old entity IDs and their mention_counts from this doc

Phase 2: Remove old artifacts
  ├─ Delete old chunks from pgvector
  ├─ Delete old Chunk nodes from Neo4j (VectorCypher)
  ├─ Delete old chunks from temporal store (Skeleton/VectorCypher)
  └─ Mark document as status=PROCESSING

Phase 3: Re-process with new content
  ├─ Update document content and checksum in PostgreSQL
  ├─ Re-chunk new content
  ├─ Re-embed new chunks
  ├─ Re-extract entities from new chunks
  ├─ Store new chunks in pgvector / temporal store
  └─ Store new Chunk nodes in Neo4j (VectorCypher)

Phase 4: Entity reconciliation
  ├─ For each OLD entity from this document:
  │   ├─ Remove this document_id from source_document_ids
  │   ├─ Remove old chunk_ids from source_chunk_ids
  │   ├─ Decrement mention_count by the old contribution
  │   └─ If source_document_ids is now empty → DELETE entity (orphan cleanup)
  ├─ For each NEW entity from re-extraction:
  │   └─ MERGE using existing upsert pattern (handles create + update)
  └─ Update document status = PROCESSED, set new chunk_count/entity_count

Phase 5: Access propagation (SHARED mode only)
  └─ New chunks inherit the same access (derived from document at query time)
```

### 6.2 Metadata-Only Update

Updates DocumentModel fields (title, source, metadata) without re-extraction.

### 6.3 Access Rights Update (SHARED Mode)

Grant/revoke operations modify `document_access` table only. No entity/chunk rewriting — visibility computed at query time.

### 6.4 Consistency Guarantees

- **PostgreSQL as source of truth**: If Neo4j is inconsistent, it can be rebuilt
- **Eventual consistency**: Neo4j entity counts may be slightly stale during update
- **Background reconciliation job**: Periodic fix for entity reference drift
- **Idempotent operations**: `GREATEST(0, ...)` and `array_remove()` safe to retry

---

## 7. Implementation Phases

### Phase 0: Fix Orphan Entity Bug (~2 days)

**Files modified:**
- `storage/coordinator.py` — Add entity cleanup to `delete_document()`
- `storage/backends/neo4j.py` — New `cleanup_entities_for_document()`
- `storage/backends/pgvector.py` — New `cleanup_entities_for_document()`
- `engines/graphrag/engine.py` — Fix `forget()` to pass `namespace_id`
- `engines/vectorcypher/engine.py` — Fix `forget()` to clean entity refs
- `engines/skeleton/engine.py` — Same

**This is a correctness bug affecting BOTH modes. Must ship before any shared-mode work.**

### Phase 1: Data Model + AccessFilter Foundation (~3 days)

**Files created:**
- `acl/access_filter.py` — AccessFilter dataclass + resolution logic
- Alembic migration — `document_access` table, `tenancy_mode` column, GIN indexes

**Files modified:**
- `db/models.py` — Add `DocumentAccessModel`, `tenancy_mode` on `MemoryNamespaceModel`
- `core/models/tenancy.py` — Ensure `tenancy_mode` surfaces on `MemoryNamespace` dataclass
- `config/schema.py` — Activate `TenancySettings`

### Phase 2: ISOLATED Mode Hardening (~2 days)

**Files modified:** `neo4j.py`, `pgvector.py`, `postgresql.py`
**Scope:** Add missing `namespace_id` filters to queries that currently lack them.

### Phase 3: SHARED Mode Core (~4-5 days)

**Files modified:**
- `engines/protocol.py` — Add `access_filter` parameter
- `memory_lake.py` — Add `principal` parameter, `_resolve_access()`, `grant_access()`/`revoke_access()`
- All engine implementations — Thread AccessFilter through remember/recall/forget
- `storage/backends/pgvector.py` — GIN array overlap queries
- `engines/vectorcypher/retriever.py` — Post-filter logic
- `engines/vectorcypher/dual_nodes.py` — Return `source_document_ids`
- `pipelines/flows/ingest.py` — Auto-create `document_access` on shared-mode ingestion

### Phase 4: ACL Enforcement (~2 days)

**Files modified:** `acl/enforcer.py`, `acl/checker.py`, `api/deps.py`

### Phase 5: Rust Acceleration (optional, ~3-4 days)

**P0 — AccessBitmap:** Roaring bitmap for `source_document_ids ∩ accessible_doc_ids`
**P0 — Filtered Vector Search:** Fused filter+distance in Rust
**P1 — Entity Reference Cleanup:** Parallel array manipulation via rayon
**P0 (small) — Access-Aware RRF:** Filter during fusion

---

## 8. Performance Analysis

### ISOLATED Mode
| Metric | Impact |
|--------|--------|
| Ingestion | **Zero overhead** |
| Query | **Zero overhead** |
| Storage | **Zero overhead** |

### SHARED Mode
| Operation | Overhead | Mitigation |
|-----------|----------|------------|
| AccessFilter resolution | ~2-5ms | 30s cache TTL |
| Chunk vector search | ~5% | B-tree document_id, iterative scan |
| Entity vector search | ~10-15% | GIN index, iterative scan |
| Full-text search | ~5% | Bitmap AND: GIN tsv + B-tree doc_id |
| Neo4j traversal | 0% | No Cypher changes |
| Python post-filter | ~1-2ms | O(1) set lookup per entity |
| **Total per-query** | **~5-20ms** | Well within acceptable range |

---

## 9. Testing Strategy

### Unit Tests (~35 new)
- AccessFilter creation, resolution, caching (both modes)
- `document_access` CRUD operations
- Entity cleanup on `forget()` (both Neo4j and pgvector)
- pgvector query generation with GIN overlap
- Post-filter logic for entity and chunk results
- Empty `accessible_doc_ids` short-circuit

### Integration Tests (~20 new)
- ISOLATED: Full regression suite — verify zero behavior change
- SHARED: Ingest → recall with/without access
- SHARED: Grant/revoke → immediate visibility change
- SHARED: Entity Union visibility (multi-doc provenance)
- SHARED: Multi-hop traversal respects access boundaries
- SHARED: Forget → cascade cleanup
- Cross-workspace: Complete isolation

### Performance Benchmarks
- Isolated mode: <1% regression allowed
- Shared mode: Measure at 100-entity, 10K-entity scales

---

## 10. Risk Assessment

| Risk | Severity | Probability | Mitigation |
|------|----------|-------------|------------|
| Orphan entity bug causes data corruption | HIGH | CERTAIN | Phase 0 fixes first |
| ISOLATED mode performance regression | HIGH | LOW | AccessFilter=None short-circuits |
| Large accessible_doc_ids degrades pgvector | MEDIUM | MEDIUM | Temp table for >1000 |
| Graph structure leakage via traversal | LOW | LOW | Union strategy, no content leak |
| ACL cache staleness (30s TTL) | LOW | MEDIUM | Configurable TTL |
| Two-store consistency on partial failure | MEDIUM | LOW | Background reconciliation job |

---

## 11. Out of Scope (Deferred)

1. **Temporal edge wiring** — dead code, separate initiative
2. **TemporalFilter unification** — two incompatible types
3. **Cross-namespace entity merging** — future optimization for shared mode
4. **PostgreSQL RLS** — evaluate after Phase 4
5. **Recency bias normalization** — VectorCypher cross-namespace skew

---

## 12. Critical Dependencies and Ordering

```
Phase 0 (orphan bug) ─────┐
                           ├──→ Phase 2 (isolated hardening)
Phase 1 (data model) ─────┤
                           ├──→ Phase 3 (shared mode core)
                           │         │
                           │         └──→ Phase 4 (ACL enforcement)
                           │
Phase 5 (Rust accel) ←──── can start P0 items once Phase 1 schema stable
```

- Phases 0 and 1 execute **in parallel** (no dependency)
- Phase 2 depends on both 0 and 1
- Phase 3 depends on 2 (can partially overlap — different files)
- Phase 4 depends on 3
- Phase 5 is independent — can start after Phase 1

---

## 13. File Change Summary

| Phase | Files Modified | Files Created |
|-------|---------------|---------------|
| 0 | `coordinator.py`, `neo4j.py`, `pgvector.py`, `graphrag/engine.py`, `vectorcypher/engine.py`, `skeleton/engine.py` | — |
| 1 | `db/models.py`, `tenancy.py`, `schema.py` | `acl/access_filter.py`, Alembic migration |
| 2 | `neo4j.py`, `pgvector.py`, `postgresql.py` | — |
| 3 | `protocol.py`, `memory_lake.py`, `graphrag/engine.py`, `vectorcypher/engine.py`, `retriever.py`, `dual_nodes.py`, `skeleton/engine.py`, `pgvector.py`, `ingest.py` | — |
| 4 | `enforcer.py`, `checker.py`, `api/deps.py` | — |
| 5 (opt) | — | `rust/khora-accel/src/access_bitmap.rs`, PyO3 bindings |

**Total: ~15 files modified, ~3 files created across 5 phases + 1 optional Rust phase.**
