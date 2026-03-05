# DYT-219: Remove FastAPI Dependency

**Status:** Draft
**Author:** Filip
**Date:** 2026-03-04
**Linear Ticket:** DYT-219

---

### Problem Statement

Khora is a library, not a deployable service. Yet it ships `fastapi>=0.133.1`, `uvicorn[standard]>=0.41.0`, and `httpx>=0.28.1` as **core** dependencies. This means every consumer — including lightweight scripts and downstream libraries like Genesis — must install a full web server stack they never use.

This bloats install size, increases dependency resolution conflicts, and sends a confusing signal about what Khora is. The FastAPI layer (`src/khora/api/`) is not exported in the public API, not used by any downstream consumer, and exists purely as a convenience wrapper around the `MemoryLake` facade.

### Goals

- Remove `fastapi`, `uvicorn`, and `httpx` from core dependencies
- Remove all FastAPI application code, routes, middleware, and the `khora serve` CLI command
- Reduce Khora's install footprint to only what the library needs
- Maintain 100% backward compatibility of the public Python API (`MemoryLake`, `remember()`, `recall()`, `forget()`, `remember_batch()`)
- Keep all non-API tests passing with coverage >= 30%

### Non-Goals

- Preserving the FastAPI layer as an optional extra (`khora[server]`) — full removal, not demotion
- Providing a replacement HTTP server or REST API
- Changing the public `MemoryLake` Python API
- Refactoring other optional dependencies (spaCy, logfire) in this ticket

### Requirements

#### Functional Requirements

1. **FR-1:** Remove `fastapi`, `uvicorn[standard]` from `pyproject.toml` core dependencies
2. **FR-2:** Remove `httpx` from core dependencies (only used by ArcadeDB backend, which already lazy-imports it)
3. **FR-3:** Delete the entire `src/khora/api/` directory (app.py, deps.py, routes/, __init__.py)
4. **FR-4:** Remove the `serve` CLI command from `src/khora/cli/server.py` and its registration in `cli/__init__.py`
5. **FR-5:** Remove `api_host` and `api_port` fields from `KhoraConfig` in `config/schema.py`
6. **FR-6:** Remove `test_client` fixture from `tests/conftest.py` and delete `tests/unit/test_api.py`
7. **FR-7:** Clean up FastAPI/uvicorn log suppression in `logging_config.py`
8. **FR-8:** Update README.md to remove references to `khora serve` and the "deploy as a FastAPI service" positioning
9. **FR-9:** Regenerate `uv.lock` after dependency changes
10. **FR-10:** Audit and update all documentation in `docs/` for FastAPI references — remove or rewrite any mentions of the API layer, `khora serve`, REST endpoints, FastAPI middleware, or HTTP deployment. Ignore unrelated doc issues

#### Non-Functional Requirements

1. **NFR-1:** All remaining tests pass (`make test`)
2. **NFR-2:** Linting and type checking pass (`make lint`)
3. **NFR-3:** Coverage remains >= 30%
4. **NFR-4:** No public API changes — `from khora import MemoryLake` and all `__all__` exports unchanged
5. **NFR-5:** Downstream consumers (Genesis, khora-benchmarks) unaffected

### User Stories

- As a **library consumer**, I want to `pip install khora` without pulling in a web server stack so that my dependency tree stays lean.
- As a **downstream library author** (Genesis), I want khora to avoid transitive dependency conflicts with my own FastAPI version so that I can upgrade independently.
- As a **contributor**, I want the codebase to clearly reflect that khora is a library so that new contributors understand the project's scope.

### Technical Approach

**Scope of deletion:**

| Path | Action |
|------|--------|
| `src/khora/api/` (7 files) | Delete entire directory |
| `src/khora/cli/server.py` | Delete file |
| `src/khora/cli/__init__.py` | Remove `serve` command registration |
| `tests/unit/test_api.py` | Delete file |
| `tests/conftest.py` | Remove `test_client` fixture and FastAPI imports |
| `src/khora/config/schema.py` | Remove `api_host`, `api_port` fields |
| `src/khora/logging_config.py` | Remove uvicorn log suppression lines |
| `pyproject.toml` | Remove fastapi, uvicorn, httpx from dependencies |
| `uv.lock` | Regenerate |
| `README.md` | Update positioning and examples |
| `docs/**/*.md` | Audit and remove all FastAPI/serve/REST/HTTP references |

**Risk assessment:**

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Downstream consumer uses `khora.api` | Very Low | Not in `__all__`, no imports found in Genesis or khora-benchmarks |
| ArcadeDB backend breaks without httpx | Low | Already lazy-imports httpx; add clear error message if missing |
| Test coverage drops below 30% | Low | Only 4 API tests removed; 1339+ other tests remain |
| Config schema change breaks consumers | Low | `api_host`/`api_port` only used by deleted server code |

**Order of operations:**
1. Delete `src/khora/api/` directory and `cli/server.py`
2. Clean up imports and references (cli/__init__.py, conftest.py, config/schema.py, logging_config.py)
3. Remove dependencies from pyproject.toml
4. Delete test_api.py, regenerate lockfile
5. Update README.md and audit all `docs/` for FastAPI references
6. Run `make test && make lint` to verify

### Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Core dependencies reduced | -3 (fastapi, uvicorn, httpx) | `pyproject.toml` diff |
| Files removed | ~9 files | git diff --stat |
| All tests pass | 100% | `make test` |
| Lint/typecheck pass | Clean | `make lint` |
| Coverage | >= 30% | pytest --cov |
| Downstream compat | No breakage | Import test in Genesis |

### Open Questions

- [x] Remove entirely or keep as optional extra? — **Decision: Remove entirely**
- [x] Remove httpx from core? — **Decision: Yes, ArcadeDB lazy-imports it**
- [x] Should ArcadeDB's httpx dependency be documented in a `khora[arcadedb]` extra? — Owner: Filip — Answer: Yes

### Timeline

Single PR, estimated small scope (~1-2 hours). All changes are deletions with minimal refactoring.
