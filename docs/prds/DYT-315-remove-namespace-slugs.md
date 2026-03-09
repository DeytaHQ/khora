# PRD-001: Flatten Namespace Model

**Status:** Implemented
**Author:** Filip
**Date:** 2026-03-08
**Linear:** DYT-315

---

### Problem Statement

Khora's namespace model carries unnecessary complexity: dual UUID/slug identification, optional namespace parameters with implicit default provisioning, and `name`/`description` metadata fields that belong in the service layer. Slug-based lookup is unused by downstream consumers (genesis, khora-benchmarks), default namespace auto-creation masks missing configuration errors, and human-readable metadata belongs in peras — not in the storage library. The namespace should be a lean, required UUID isolation boundary.

### Goals

- All public API functions (`remember`, `recall`, `forget`, `remember_batch`) require a `namespace_id: UUID` parameter — no optional namespace, no slug lookup path
- Remove the `slug` column from `memory_namespaces` table and all associated constraints/indexes
- Remove `name` and `description` columns from `memory_namespaces` table and domain model
- Remove `get_namespace_by_slug()` and `get_or_create_default_namespace()` from all layers
- Remove `get_namespace_by_name()` (no longer needed without name field)
- Namespace becomes a pure UUID isolation boundary: `id`, `version`, `is_active`, `config_overrides`, timestamps
- Reduce code surface area (estimated ~15 files, ~150+ lines removed)

### Non-Goals

- **Backward compatibility** — no deprecation period or shim; clean break
- **Adding slug/name functionality to peras** — that's a peras concern
- **Changing namespace table structure beyond stated columns** — `version`, `is_active`, `config_overrides`, and timestamp columns are preserved

### Requirements

#### Functional Requirements

1. **FR-1:** `MemoryNamespace` domain model (`core/models/tenancy.py`) must not contain `slug`, `name`, or `description` fields. Remove `__post_init__` auto-generation logic
2. **FR-2:** `MemoryNamespaceModel` ORM (`db/models.py`) must not contain `slug`, `name`, or `description` columns, nor the `uq_namespace_slug_version` constraint or `idx_namespace_slug_active` index
3. **FR-3:** `namespace_id` parameter must be **required** (not optional) in all public API methods (`remember`, `recall`, `forget`, `remember_batch`). Accept `UUID` or `str` (converted to `UUID`), raising `ValueError` for non-UUID-parseable strings
4. **FR-4:** `_resolve_namespace()` in `memory_lake.py` must not have a fallback to default namespace. Remove the `get_or_create_default_namespace()` call path
5. **FR-5:** `get_or_create_default_namespace()` must be removed from `MemoryLake`, all three engines (GraphRAG, Skeleton, VectorCypher), and the `MemoryEngineProtocol`
6. **FR-6:** `get_namespace_by_slug()` must be removed from `RelationalBackend` protocol, `PostgreSQLBackend`, and `StorageCoordinator`
7. **FR-7:** `get_namespace_by_name()` must be removed from `RelationalBackend` protocol, `PostgreSQLBackend`, and `StorageCoordinator`
8. **FR-8:** `create_namespace()` and `create_namespace_version()` must not accept `slug`, `name`, or `description` parameters
9. **FR-9:** Alembic migration to drop `slug`, `name`, and `description` columns and add a unique constraint on `(namespace_id)` or equivalent as needed
10. **FR-10:** All tests referencing slug, name, description, or default namespace behavior must be removed or rewritten

#### Non-Functional Requirements

1. **NFR-1:** All existing unit tests pass after changes
2. **NFR-2:** Coverage remains at or above current threshold (30%)
3. **NFR-3:** `make lint` and `ty check src/` pass clean

### User Stories

- As a **library consumer**, I want namespace identification to use only UUIDs so that there is one unambiguous way to reference a namespace.
- As a **library consumer**, I want namespace to be a required parameter so that missing tenant context fails loudly rather than silently creating a default.
- As a **khora maintainer**, I want to remove slug, name, description, and default provisioning so that the namespace model is a minimal isolation boundary with less code to maintain.

### Technical Approach

**Files to modify (source):**

| File | Change |
|------|--------|
| `src/khora/core/models/tenancy.py` | Remove `slug`, `name`, `description` fields and `__post_init__` auto-generation |
| `src/khora/db/models.py` | Remove `slug`, `name`, `description` columns, `uq_namespace_slug_version` constraint, `idx_namespace_slug_active` index |
| `src/khora/memory_lake.py` | Make `namespace_id` required in public API; simplify `_resolve_namespace()` to UUID-only; remove `get_or_create_default_namespace()` and its fallback |
| `src/khora/storage/backends/base.py` | Remove `get_namespace_by_slug()` and `get_namespace_by_name()` from protocol; remove `name`/`description` from `create_namespace()` signature |
| `src/khora/storage/backends/postgresql.py` | Remove `get_namespace_by_slug()`, `get_namespace_by_name()` impls; remove `slug=`/`name=`/`description=` mappings in create/update |
| `src/khora/storage/coordinator.py` | Remove `get_namespace_by_slug()`, `get_namespace_by_name()`; remove `slug`/`name`/`description` params from `create_namespace_version()` |
| `src/khora/engines/graphrag/engine.py` | Remove `get_or_create_default_namespace()`, `_default_namespace_id` cache; remove `name=` from `create_namespace()` calls |
| `src/khora/engines/skeleton/engine.py` | Same as above |
| `src/khora/engines/vectorcypher/engine.py` | Same as above |
| `src/khora/engines/protocol.py` | Remove `get_or_create_default_namespace()` from `MemoryEngineProtocol` |

**Files to modify (tests):**

| File | Change |
|------|--------|
| `tests/unit/test_models.py` | Remove `test_namespace_auto_slug`, slug/name/description references in namespace creation |
| `tests/unit/test_memory_lake.py` | Remove `test_slug_lookup`, `test_slug_not_found_raises`, default namespace tests; update fixtures to always pass `namespace_id` |

**Files to modify (docs):**

| File | Change |
|------|--------|
| `docs/architecture/multi-tenancy.md` | Remove slug-based examples; document required namespace_id; remove name/description references |
| `docs/architecture/storage-backends.md` | Remove `slug VARCHAR`, `name VARCHAR`, `description TEXT` from schema docs |

**Files to add (migrations):**

| File | Change |
|------|--------|
| `src/khora/db/migrations/versions/XXX_drop_namespace_slug_name_desc.py` | Drop `slug`, `name`, `description` columns; drop old constraints/indexes; add new constraints as needed |

### Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Lines of namespace-related code removed | ~150+ | Diff stats |
| Files modified | ~15 | Diff stats |
| Tests passing | 100% | `make test` |
| Lint/typecheck clean | 0 errors | `make lint` |

### Open Questions

- [x] **Default namespace resolution:** ~~(a) well-known UUID constant, (b) name lookup, (c) `get_or_create_default_namespace()`.~~ Decision: none — namespace is always required, no default provisioning
- [x] **Engine `create_namespace()` methods:** ~~Slug param needed?~~ Decision: only `id` and optional `config_overrides`; `slug`, `name`, `description` removed
- [x] **Database migration:** ~~Separate or included?~~ Decision: included in this PR — Alembic migration drops `slug`, `name`, `description` columns

### Timeline

Single implementation phase — all changes in one PR.
