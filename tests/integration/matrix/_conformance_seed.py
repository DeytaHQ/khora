"""One-time seeder for the live-store recall-filter conformance legs.

Run ONCE before a docker-backed pytest leg, naming the backend to seed::

    python -m tests.integration.matrix._conformance_seed            # postgres (default)
    python -m tests.integration.matrix._conformance_seed neo4j      # cypher leg
    python -m tests.integration.matrix._conformance_seed weaviate   # weaviate leg

Each backend builds its live store, seeds every corpus case that targets it (a
random chunk UUID per record), and persists the ``case_id -> {seed_id: chunk_uuid}``
map as a JSON artifact at that backend's seed-map path. That map is the ONLY bridge
between these rows and the strictly READ-ONLY pytest step (a separate process), so
under ``-n auto`` every xdist worker only reads the pre-seeded store — no write
contention against the shared container. This is the ONE-TIME seed (a workflow step
before pytest), NOT a per-xdist-worker fixture.

The embedded legs (sqlite_lance / surrealdb) are NOT seeded here: they run on a
per-worker in-process store (tmp SQLite+LanceDB / embedded ``memory://``) and seed
inside their own pytest fixture, so they need no shared seed-map artifact.

This is a thin driver: each backend's ``build_seed_map`` / ``write_seed_map`` lives
in its sibling ``_conformance_<backend>`` helper (which the test module also imports),
so the rows it writes are byte-for-byte the rows the test reads back.
"""

from __future__ import annotations

import asyncio
import sys
from types import ModuleType

# The backends this driver can seed. Embedded legs (sqlite_lance / surrealdb) are
# absent on purpose — they seed per-worker in their own pytest fixture.
_BACKENDS: frozenset[str] = frozenset({"postgres", "neo4j", "weaviate"})


def _seeder_module(backend: str) -> ModuleType:
    """Lazily import the requested backend's ``_conformance_<backend>`` helper.

    Lazy + per-backend so seeding ``postgres`` never imports the neo4j / weaviate
    SDKs (each helper imports its own optional client at use time). The module
    exposes the ``build_seed_map`` / ``write_seed_map`` / ``SEED_MAP_PATH`` trio.
    """
    if backend == "postgres":
        from tests.integration.matrix import _conformance_pg as module
    elif backend == "neo4j":
        from tests.integration.matrix import _conformance_neo4j as module
    elif backend == "weaviate":
        from tests.integration.matrix import _conformance_weaviate as module
    else:  # pragma: no cover - guarded by the caller's membership check
        raise SystemExit(f"unknown conformance backend {backend!r}; choose one of {sorted(_BACKENDS)}")
    return module


async def main(backend: str = "postgres") -> None:
    """Seed every case targeting ``backend`` once and persist its seed map."""
    if backend not in _BACKENDS:
        raise SystemExit(f"unknown conformance backend {backend!r}; choose one of {sorted(_BACKENDS)}")
    module = _seeder_module(backend)
    seed_map = await module.build_seed_map()
    module.write_seed_map(seed_map)
    print(f"seeded {len(seed_map)} {backend} conformance cases; wrote seed map to {module.SEED_MAP_PATH}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "postgres"))
