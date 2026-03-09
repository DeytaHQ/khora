# PRD-001: Remove Namespace Slugs

**Status:** Draft
**Author:** Filip
**Date:** 2026-03-08
**Linear:** DYT-315

---

### Problem Statement

Khora's public API accepts both UUIDs and slug strings for namespace identification (`_resolve_namespace()` in `memory_lake.py`). This dual-lookup path adds complexity: slug auto-generation logic, a dedicated `get_namespace_by_slug()` method through every storage layer, a DB column with unique constraints and partial indexes, and branching resolution code. Slug-based lookup is unused by downstream consumers (genesis, khora-benchmarks) and the human-readable slug functionality belongs in the peras service layer, not in the storage library.

### Goals

- All public API functions (`remember`, `recall`, `forget`, `remember_batch`) accept `UUID` or `str` (converted to UUID) for namespace identification — no slug lookup path
- Remove the `slug` column from `memory_namespaces` table and all associated constraints/indexes
- Remove `get_namespace_by_slug()` from storage backends, coordinator, and engine layers
- Remove slug field from `MemoryNamespace` domain model and auto-generation logic
- Reduce code surface area (estimated ~15 files, ~100+ lines removed)

### Non-Goals

- **Database migration** — no new Alembic migration file to drop the DB column. The ORM model removes the `slug` mapped column, so SQLAlchemy ignores it, but the physical column remains in the database until a migration is added separately
- **Backward compatibility** — no deprecation period or shim; clean break
- **Adding slug functionality to peras** — that's a peras concern
- **Changing namespace creation API** — `create_namespace()` still takes a name; it just no longer generates or stores a slug
- **Making namespace_id required** — `namespace` parameter stays optional in public API; `get_or_create_default_namespace()` convenience is preserved

### Requirements

#### Functional Requirements

1. **FR-1:** `MemoryNamespace` domain model (`core/models/tenancy.py`) must not contain a `slug` field
2. **FR-2:** `MemoryNamespaceModel` ORM (`db/models.py`) must not contain a `slug` column, its unique constraint (`uq_namespace_slug_version`), or its partial index (`idx_namespace_slug_active`)
3. **FR-3:** `_resolve_namespace()` in `memory_lake.py` must accept `UUID` or `str` (converted to `UUID`), raising `ValueError` for non-UUID-parseable strings. The slug lookup branch is removed
4. **FR-4:** `get_namespace_by_slug()` must be removed from `RelationalBackend` protocol, `PostgreSQLBackend`, and `StorageCoordinator`
5. **FR-5:** All three engines (GraphRAG, Skeleton, VectorCypher) must replace the `get_namespace_by_slug("default")` call inside `get_or_create_default_namespace()` with a name-based lookup (e.g., `get_namespace_by_name("default")`). The `get_or_create_default_namespace()` method and optional `namespace` parameter in the public API are preserved
6. **FR-6:** `create_namespace_version()` in coordinator must not accept a `slug` parameter
7. **FR-7:** All tests referencing slug behavior must be removed or rewritten to use UUID-only paths

#### Non-Functional Requirements

1. **NFR-1:** All existing unit tests pass after changes
2. **NFR-2:** Coverage remains at or above current threshold (30%)
3. **NFR-3:** `make lint` and `ty check src/` pass clean

### User Stories

- As a **library consumer**, I want namespace identification to use only UUIDs so that there is one unambiguous way to reference a namespace.
- As a **khora maintainer**, I want to remove the slug abstraction so that the storage layer has less code to maintain and fewer edge cases.

### Technical Approach

**Files to modify (source):**

| File | Change |
|------|--------|
| `src/khora/core/models/tenancy.py` | Remove `slug` field and `__post_init__` auto-generation |
| `src/khora/db/models.py` | Remove `slug` column, `uq_namespace_slug_version` constraint, `idx_namespace_slug_active` index |
| `src/khora/memory_lake.py` | Simplify `_resolve_namespace()` to UUID-only; remove slug branch |
| `src/khora/storage/backends/base.py` | Remove `get_namespace_by_slug()` from protocol |
| `src/khora/storage/backends/postgresql.py` | Remove `get_namespace_by_slug()` impl; remove `slug=` mappings in create/update |
| `src/khora/storage/coordinator.py` | Remove `get_namespace_by_slug()` and `slug` param from `create_namespace_version()` |
| `src/khora/engines/graphrag/engine.py` | Replace `get_namespace_by_slug("default")` with `get_or_create_default_namespace()`; remove `slug` param from `create_namespace()` calls |
| `src/khora/engines/skeleton/engine.py` | Same as above |
| `src/khora/engines/vectorcypher/engine.py` | Same as above |

**Files to modify (tests):**

| File | Change |
|------|--------|
| `tests/unit/test_models.py` | Remove `test_namespace_auto_slug` and slug references in namespace creation |
| `tests/unit/test_memory_lake.py` | Remove `test_slug_lookup`, `test_slug_not_found_raises`; update any namespace fixtures |


**Files to modify (docs):**

| File | Change |
|------|--------|
| `docs/architecture/multi-tenancy.md` | Remove slug-based examples |
| `docs/architecture/storage-backends.md` | Remove `slug VARCHAR` from schema docs |

**Default namespace strategy:** Engines currently do `get_namespace_by_slug("default")`. After removal, add a `get_or_create_default_namespace()` method on the coordinator that looks up by `name="default"` and creates if missing. All three engines call this instead of the slug-based path.

### Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Lines of slug-related code removed | ~100+ | Diff stats |
| Files modified | ~13 | Diff stats |
| Tests passing | 100% | `make test` |
| Lint/typecheck clean | 0 errors | `make lint` |

### Open Questions

- [x] **Default namespace resolution:** ~~(a) well-known UUID constant, (b) name lookup, (c) `get_or_create_default_namespace()`.~~ Decision: (c) — new coordinator method that looks up by name and creates if missing
- [x] **Engine `create_namespace()` methods:** ~~Slug param needed?~~ Decision: `name` alone suffices; `slug` param removed

### Timeline

Single implementation phase — all changes in one PR.
