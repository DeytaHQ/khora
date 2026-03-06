# DYT-220: Flatten Multi-Tenancy to Namespace-Only

**Status:** Reviewed
**Author:** Filip
**Date:** 2026-03-05
**Linear Ticket:** DYT-220

---

## Problem Statement

Khora's multi-tenancy model uses a 3-level hierarchy: Organization → Workspace → Namespace. In practice, **namespace is the only meaningful isolation boundary** — all data (documents, chunks, entities, relationships, episodes) is scoped to `namespace_id`. Organization and Workspace exist as structural overhead that:

1. **Adds friction to the API** — creating a namespace requires a workspace, which requires an organization. The `ensure_namespace()` method auto-creates "default" org and workspace to hide this from callers.
2. **Complicates downstream consumers** — genesis, peras, and khora-benchmarks all interact with namespaces. None meaningfully needs the org/workspace hierarchy going forward.
3. **Creates dead code** — `TenancyMode.ISOLATED` (per-org database separation) is defined but not implemented. Organization/Workspace CRUD operations exist in the storage layer but serve only to satisfy FK constraints.
4. **Inflates the schema** — 2 tables (`organizations`, `workspaces`), ~15 columns, multiple indexes, and cascade chains exist solely to support a hierarchy nobody uses.

### Downstream Consumers

| Consumer | Current Usage | Impact |
|----------|--------------|--------|
| **genesis** | Uses `lake.storage` API for full org/workspace/namespace hierarchy — CLI navigates orgs, lists workspaces, browses namespaces | Must rewrite navigation UX to namespace-only; remove all org/workspace creation and lookup calls |
| **peras** | Uses `MemoryLake` facade + accesses `lake.storage` directly for `get_sync_checkpoint()`, `set_sync_checkpoint()`, `get_document()` | Must remove any org/workspace references; storage API changes affect sync checkpoint calls |
| **khora-benchmarks** | Creates test namespaces | Must update setup code to remove `workspace_id` |

---

## Goals

- Namespace becomes the **sole** data isolation boundary — no parent references required
- Public API (`MemoryLake`) requires **zero breaking changes** for callers using `namespace: str | UUID | None`
- Storage layer API removes org/workspace CRUD methods
- Database migration safely drops `organizations` and `workspaces` tables
- ACL system simplified to namespace-only permissions
- ConfigResolver reduced to 2-tier: global defaults → namespace overrides
- `TenancyMode` moves from Organization to Namespace
- All downstream consumers updated with a clean version bump (no deprecation period)

---

## Non-Goals

- Introducing a new grouping/tagging mechanism for namespaces (future work)
- Changing how `namespace_id` is used in data tables (documents, chunks, entities, etc.) — these stay as-is
- Modifying namespace versioning (`version`, `is_active`, `previous_version_id`) — orthogonal to this change
- Changing the graph backend namespace isolation model
- Adding namespace-level billing or quotas
- Implementing `TenancyMode.ISOLATED` — it moves to namespace but remains unimplemented

---

## Requirements

### Functional Requirements

#### FR-1: Remove Organization Model and Table

Remove `Organization` dataclass (`core/models/tenancy.py`), `OrganizationModel` ORM class (`db/models.py`), and the `organizations` database table. Remove all CRUD operations: `create_organization()`, `get_organization()`, `get_organization_by_slug()` from `StorageCoordinator`, backend protocol, and PostgreSQL backend.

**Files affected:**
- `src/khora/core/models/tenancy.py` — delete `Organization` class
- `src/khora/db/models.py` — delete `OrganizationModel`
- `src/khora/core/__init__.py`, `src/khora/core/models/__init__.py` — remove re-exports
- `src/khora/db/__init__.py` — remove re-export
- `src/khora/storage/backends/base.py` — remove protocol methods
- `src/khora/storage/backends/postgresql.py` — remove implementations
- `src/khora/storage/coordinator.py` — remove delegations

#### FR-2: Remove Workspace Model and Table

Remove `Workspace` dataclass, `WorkspaceModel` ORM class, and the `workspaces` database table. Remove all CRUD operations: `create_workspace()`, `get_workspace()`, `list_workspaces()`, `get_workspace_by_slug()`.

**Files affected:** Same set as FR-1.

#### FR-3: Update MemoryNamespace — Remove `workspace_id`, Add `tenancy_mode`

The `MemoryNamespace` dataclass and `MemoryNamespaceModel` lose the `workspace_id` FK and gain a `tenancy_mode` field (default: `SHARED`).

**Before:**
```python
@dataclass
class MemoryNamespace:
    id: UUID
    workspace_id: UUID  # FK to workspaces
    name: str
    slug: str
    ...
```

**After:**
```python
@dataclass
class MemoryNamespace:
    id: UUID
    name: str
    slug: str
    tenancy_mode: TenancyMode = TenancyMode.SHARED
    ...
```

**Database constraint change:**
- Old: `UNIQUE(workspace_id, slug, version)`
- New: `UNIQUE(slug, version)` — slugs become globally unique

**Files affected:**
- `src/khora/core/models/tenancy.py` — remove `workspace_id` field, add `tenancy_mode`
- `src/khora/db/models.py` — update `MemoryNamespaceModel`, drop FK, update constraints
- Alembic migration (new)

#### FR-4: Update `create_namespace()` and `create_namespace_version()` APIs

Remove `workspace_id` parameter from `MemoryLake.create_namespace()`, `create_namespace_version()`, and the engine protocol.

**Before:**
```python
async def create_namespace(self, name: str, workspace_id: UUID, ...) -> MemoryNamespace
async def create_namespace_version(self, workspace_id: UUID, slug: str, ...) -> MemoryNamespace
```

**After:**
```python
async def create_namespace(self, name: str, ...) -> MemoryNamespace
async def create_namespace_version(self, slug: str, ...) -> MemoryNamespace
```

**Files affected:**
- `src/khora/memory_lake.py`
- `src/khora/engines/protocol.py`
- `src/khora/engines/graphrag/engine.py`
- `src/khora/engines/skeleton/engine.py`
- `src/khora/engines/vectorcypher/engine.py`
- `src/khora/storage/coordinator.py` — `create_namespace_version()` (line 344)
- `src/khora/storage/backends/postgresql.py` — `create_namespace_version()` (line 333)

#### FR-5: Simplify `ensure_namespace()` and `get_or_create_default_namespace()`

These methods currently auto-create a "default" Organization and Workspace before creating the namespace. After this change, they create the namespace directly.

**Files affected:**
- `src/khora/memory_lake.py`
- All engine implementations

#### FR-6: Simplify `_resolve_namespace()` Slug Lookup

Currently resolves slugs by looking up the default workspace, then finding the namespace within that workspace. After this change, slugs are globally unique — look up directly by slug.

**Before:**
```python
ns = await storage.get_namespace_by_slug(default_ns.workspace_id, namespace)
```

**After:**
```python
ns = await storage.get_namespace_by_slug(namespace)
```

**Files affected:**
- `src/khora/memory_lake.py` — `_resolve_namespace()`
- `src/khora/storage/backends/base.py` — update protocol signature
- `src/khora/storage/backends/postgresql.py` — remove `workspace_id` from query

#### FR-7: Simplify ACL to Namespace-Only

Remove `workspace_id` and `organization_id` from `ACLContext`. Remove the `parent_ids` property. Remove all org/workspace-related methods and constants:

- `ACLEnforcer.check_workspace_read()` (enforcer.py:152)
- `ACLEnforcer.check_workspace_admin()` (enforcer.py:155)
- `ACLEnforcer.check_organization_read()` (enforcer.py:160)
- `ACLEnforcer.check_organization_admin()` (enforcer.py:164)
- `ACLChecker.HIERARCHY` dict entries for "organization" and "workspace" (checker.py:87-92)
- `ACLChecker.check()` parent walk-up logic that uses `parent_ids`

Update `PermissionModel` resource_type to only allow "namespace".

**Files affected:**
- `src/khora/acl/enforcer.py` — simplify `ACLContext`, remove 4 methods, remove `parent_ids`
- `src/khora/acl/checker.py` — remove `HIERARCHY` org/workspace entries, remove parent walk-up logic
- `src/khora/db/models.py` — update `PermissionModel` resource_type options

#### FR-8: Simplify ConfigResolver to 2-Tier

Reduce config resolution from 4-tier (global → org → workspace → namespace) to 2-tier (global → namespace). Remove all workspace/organization-related code — no backward compatibility.

**Before:** `resolve_for_namespace()` fetches namespace → workspace → organization and merges all configs.
**After:** `resolve_for_namespace()` fetches namespace only and merges with global defaults.

`resolve_immediate()` loses its `organization_config` and `workspace_config` parameters entirely (no no-ops).

**Files affected:**
- `src/khora/config/resolver.py` — remove workspace/org fetch (lines 84-89), remove merge steps, remove `resolve_immediate()` org/workspace params

#### FR-9: Remove `full_path` Property

Remove `MemoryNamespace.full_path` property entirely. Callers should use `slug` directly.

**Files affected:**
- `src/khora/core/models/tenancy.py` — delete `full_path` property (lines 106-109)

#### FR-10: Database Migration

Create an Alembic migration with the following steps **in this exact order**:

1. **Pre-check:** Query for duplicate slugs across workspaces. If collisions exist, **fail fast** with a clear error listing the conflicting slugs. Require manual resolution before re-running. No auto-renaming.
2. **Delete orphaned permission rows:** `DELETE FROM permissions WHERE resource_type IN ('organization', 'workspace')` and `DELETE FROM permissions WHERE inherited_from_type IN ('organization', 'workspace')`
3. **Drop `inherited_from_type` and `inherited_from_id` columns** from `permissions` table (no hierarchy = no inheritance tracking)
4. **Drop old indexes** on `memory_namespaces` that reference `workspace_id` (including `idx_namespace_active`)
5. **Remove `workspace_id` FK and column** from `memory_namespaces`
6. **Add `tenancy_mode` column** to `memory_namespaces` (default: `'shared'`)
7. **Add new unique constraint** `UNIQUE(slug, version)`
8. **Add new index** `idx_namespace_slug_active ON memory_namespaces (slug, version) WHERE is_active = true`
9. **Drop `workspaces` table** (FK to organizations is safe to drop since no other table references workspaces after step 5)
10. **Drop `organizations` table**

**Migration must be reversible** with a downgrade path that recreates the tables with default data.

**Slug collision risk:** <50 namespaces exist in production — global uniqueness is highly likely. The pre-check query in step 1 ensures the migration fails safely if collisions exist.

#### FR-11: Update Engine Implementations

All three engines (graphrag, skeleton, vectorcypher) import and use Organization/Workspace types in their namespace management code. Update all to remove these references.

**Files affected:**
- `src/khora/engines/graphrag/engine.py`
- `src/khora/engines/skeleton/engine.py`
- `src/khora/engines/vectorcypher/engine.py`

#### FR-12: Update `list_namespaces()` Signature

Replace workspace-scoped listing with paginated global listing.

**Before:**
```python
async def list_namespaces(self, workspace_id: UUID, *, active_only: bool = True) -> list[MemoryNamespace]
```

**After:**
```python
async def list_namespaces(self, *, active_only: bool = True, limit: int = 100, offset: int = 0) -> list[MemoryNamespace]
```

Results ordered by `id` always. No workspace filter.

**Files affected:**
- `src/khora/storage/backends/base.py` — update protocol
- `src/khora/storage/backends/postgresql.py` — update query
- `src/khora/storage/coordinator.py` — update delegation
- `src/khora/engines/graphrag/engine.py` — update call sites
- `src/khora/engines/skeleton/engine.py` — update call sites
- `src/khora/engines/vectorcypher/engine.py` — update call sites

### Non-Functional Requirements

1. **NFR-1:** Zero data loss — all existing documents, chunks, entities, and relationships must survive the migration
2. **NFR-2:** Migration pre-check must detect slug collisions and fail fast with actionable error
3. **NFR-3:** Test coverage must remain ≥30% after changes
4. **NFR-4:** `make lint` and `ty check src/` must pass clean after changes
5. **NFR-5:** Public `remember()`/`recall()`/`forget()` signatures remain unchanged — callers using `namespace: str | UUID | None` see no breaking change

---

## User Stories

- As a **library consumer** (genesis/peras), I want to create namespaces without needing to set up organizations and workspaces, so that I can start storing data with a single call.
- As a **library maintainer**, I want fewer tenancy concepts so that the codebase is simpler to understand and maintain.
- As a **downstream developer**, I want namespace slugs to be globally unique so that I can reference namespaces by name without workspace context.
- As a **library consumer**, I want per-namespace config overrides with global defaults so that I can customize behavior without organizational hierarchy.

---

## Technical Approach

### Migration Strategy

1. **Phase 1 — Code changes** (no migration yet):
   - Update all models, remove org/workspace from domain layer
   - Update storage backends, coordinator, engines
   - Update ACL, ConfigResolver
   - Update tests

2. **Phase 2 — Migration**:
   - Write Alembic migration with slug collision pre-check (fail fast, no auto-rename)
   - Drop FKs, add column, update constraints and indexes, drop tables

3. **Phase 3 — Downstream coordination (clean break, no deprecation)**:
   - Update genesis: rewrite namespace navigation to namespace-only
   - Update peras: remove any org/workspace storage API calls
   - Update khora-benchmarks: update setup code
   - All downstream updates happen simultaneously with khora version bump

### Breaking Changes Summary

| Change | Who It Breaks | Migration Path |
|--------|--------------|----------------|
| `create_namespace()` loses `workspace_id` param | Direct callers of this method | Remove the argument |
| `create_namespace_version()` loses `workspace_id` param | Direct callers | Remove the argument |
| `get_namespace_by_slug()` loses `workspace_id` param | Direct callers | Remove the argument |
| `list_namespaces()` loses `workspace_id`, gains `limit`/`offset` | Direct callers | Remove workspace_id, add pagination params |
| `Organization` class removed | Importers of `khora.core.models.Organization` | Delete imports |
| `Workspace` class removed | Importers of `khora.core.models.Workspace` | Delete imports |
| `create_organization()`/`create_workspace()` removed | Storage layer callers (genesis, peras) | Delete calls |
| `get_organization()`/`get_workspace()`/`list_workspaces()` removed | Storage layer callers (genesis) | Delete calls |
| `MemoryNamespace.workspace_id` removed | Code accessing this field | Delete references |
| `MemoryNamespace.full_path` removed | Code using `namespace.full_path` | Use `namespace.slug` |
| ACL `check_workspace_*`/`check_organization_*` removed | ACL consumers | Use `check_namespace_*` |
| `ConfigResolver.resolve_immediate()` loses org/workspace params | `resolve_immediate()` callers | Remove those params |
| `PermissionModel.inherited_from_type/id` columns dropped | Direct SQL queries on permissions | Update queries |
| DB schema: `organizations`/`workspaces` tables dropped | Direct SQL queries against these tables | Update queries |

---

## Risks and Reasons NOT to Remove

These were considered and the decision was made to proceed:

1. **Future multi-tenancy needs**: If a consumer later needs org-level grouping (e.g., billing boundary, team separation), we'd need to re-introduce some hierarchy. **Mitigation:** Namespace metadata can hold arbitrary grouping tags. A lightweight "label" or "group" field can be added later without full org/workspace overhead.

2. **Slug collision on migration**: Two namespaces in different workspaces could have the same slug. **Mitigation:** <50 namespaces exist — global uniqueness is highly likely. Migration includes a pre-check that fails fast with a clear error if collisions exist, requiring manual resolution. No auto-renaming.

3. **Config inheritance loss**: Organizations/workspaces provided config inheritance (set a model once, all child namespaces inherit). After this change, each namespace must be configured individually or rely on global defaults. **Mitigation:** 2-tier (global + namespace) is accepted as sufficient. For bulk config, consumers set `config_overrides` programmatically. No bulk config API is planned.

4. **Permission inheritance loss**: Org-admin granting workspace/namespace access in one grant is no longer possible. **Mitigation:** ACL is not wired into production paths. Namespace-only permissions are simpler and sufficient.

5. **Downstream breakage**: genesis, peras, khora-benchmarks all need updates. Genesis impact is significant (navigation UX rewrite). **Mitigation:** Clean break with simultaneous version bump across all consumers. No deprecation period.

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Tables removed | 2 (`organizations`, `workspaces`) | Schema inspection |
| Columns removed from `memory_namespaces` | 1 (`workspace_id`) | Schema inspection |
| Lines of code removed (net) | 200+ | `git diff --stat` |
| Public API breakage (`remember`/`recall`/`forget`) | 0 | Existing tests pass |
| Test coverage | ≥30% | `make test` |
| ConfigResolver DB queries per call | 1 (down from 3) | Code inspection |

---

## Timeline

| Phase | Milestone | Scope |
|-------|-----------|-------|
| 1 | Code changes | Remove org/workspace from models, storage, engines, ACL, config |
| 2 | Migration | Alembic migration with collision pre-check |
| 3 | Tests | Update all tests, verify coverage |
| 4 | Downstream | Simultaneously update genesis, peras, khora-benchmarks |

---

## Review Findings (2026-03-05)

**Overall Assessment: REQUEST CHANGES → APPROVED WITH REVISIONS**

Reviewers: Security, Performance, Correctness, Test, Docs, Devil's Advocate (6 reviewers, all findings addressed below).

### HIGH Findings

**H1: Genesis actively uses full org/workspace hierarchy.** The devil's advocate found that genesis isn't just "creating namespaces" — its CLI navigates orgs, lists workspaces, and browses namespaces within workspaces. The original PRD understated this as "must update namespace creation calls."
**Resolution:** Accepted scope. Genesis will be rewritten to use namespace-only navigation. Downstream impact table updated to reflect the full scope. No migration shim — clean break.

**H2: Slug collision migration algorithm was unspecified.** Five of six reviewers flagged this. The original PRD said "append workspace slug as suffix" with no algorithm or test.
**Resolution:** No auto-renaming. <50 namespaces exist; global uniqueness is highly likely. Migration includes a pre-check query that fails fast with a clear error listing collisions, requiring manual resolution. FR-10 updated with exact step ordering.

**H3: Migration FK ordering was ambiguous.** Must remove `workspace_id` FK from `memory_namespaces` before dropping `workspaces` table.
**Resolution:** FR-10 rewritten with explicit 10-step ordering. FK removal (step 5) precedes table drops (steps 9-10).

**H4: Orphaned permission rows.** `PermissionModel` has no FK to org/workspace tables. Dropping those tables leaves rows with `resource_type='organization'/'workspace'` as orphans.
**Resolution:** FR-10 step 2 now explicitly deletes these rows. Step 3 drops the `inherited_from_type` and `inherited_from_id` columns entirely.

**H5: `list_namespaces()` becomes unbounded.** Currently filtered by `workspace_id` (indexed). After removal, returns all namespaces with no pagination.
**Resolution:** New FR-12 added. Signature becomes `list_namespaces(*, active_only=True, limit=100, offset=0)`, ordered by `id`.

**H6: ConfigResolver missing from affected files.** `resolve_for_namespace()` calls `storage.get_workspace()` and `storage.get_organization()` — would break at runtime.
**Resolution:** FR-8 updated to include `src/khora/config/resolver.py` and specifies removing all org/workspace code with no backward compatibility.

**H7: Missing index on `(slug, version)`.** Current index `idx_namespace_active` has `workspace_id` as leading column. After migration, slug lookups won't use it.
**Resolution:** FR-10 step 4 drops old indexes, step 8 creates `idx_namespace_slug_active ON memory_namespaces (slug, version) WHERE is_active = true`.

### MEDIUM Findings

**M1: Peras accesses `.storage` API directly.** Uses `get_sync_checkpoint()`, `set_sync_checkpoint()`, `get_document()` — more coupling than originally documented.
**Resolution:** Downstream impact table updated. Peras will be updated as part of the clean break.

**M2: TenancyMode.ISOLATED — remove or move?** Devil's advocate argued against moving an unimplemented feature.
**Resolution:** Move it to namespace. It stays as a placeholder for future implementation. Added to Non-Goals that implementing ISOLATED is out of scope.

**M3: Config inheritance loss understated.** No bulk config API for 50-namespace scenarios.
**Resolution:** Accepted trade-off. 2-tier (global + namespace) is sufficient. Documented honestly in Risks section.

**M4: No deprecation timeline.** No communication plan for downstream consumers.
**Resolution:** Clean break with simultaneous version bump. All downstream consumers (genesis, peras, khora-benchmarks) updated in the same release cycle. No deprecation period.

**M5: `create_namespace_version()` still requires `workspace_id`.** Not addressed in original PRD.
**Resolution:** Added to FR-4. Both `create_namespace()` and `create_namespace_version()` lose `workspace_id`.

**M6: ACL HIERARCHY dict and enforcer methods not explicitly listed.** FR-7 was vague about what to remove.
**Resolution:** FR-7 expanded with explicit list of 4 methods, HIERARCHY dict entries, and parent walk-up logic to remove.

**M7: FR-9 `full_path` was undecided.** Two options listed, no decision.
**Resolution:** Remove entirely. Callers use `slug` directly.

**M8: `PermissionModel.inherited_from_type/id` columns unclear.** Should they be dropped?
**Resolution:** Yes. FR-10 step 3 drops both columns. No hierarchy means no inheritance tracking.

### LOW Findings

**L1: Open questions lacked deadlines.** All three open questions have been resolved during review and removed from the Open Questions section.

**L2: Missing observability section.** No version compatibility matrix.
**Resolution:** Accepted as out of scope for this PRD. Version bump communicates the break.

**L3: ConfigResolver 3→1 query reduction.** Positive performance finding — every `remember()`/`recall()` call saves 2 DB roundtrips. Added to Success Metrics.

**L4: Test coverage impact minimal.** 15 tests affected (8 delete, 7 update). Coverage stays at ~52%, well above 30% threshold.
**Resolution:** No action needed. Migration test should be added as part of Phase 3.
