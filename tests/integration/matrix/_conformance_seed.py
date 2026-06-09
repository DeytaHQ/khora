"""One-time seeder for the Postgres recall-filter conformance leg.

Run ONCE before the ``KHORA_CONFORMANCE_BACKEND=postgres`` pytest leg::

    python -m tests.integration.matrix._conformance_seed

It builds a live-Postgres coordinator (relational + the skeleton ``khora_chunks``
temporal vector store, sharing one engine) against ``KHORA_DATABASE_URL`` and
calls :func:`khora.filter.conformance.seed_case` once for every corpus case that
targets ``postgres``. This is the ONE-TIME seed (a workflow step before pytest),
NOT a per-xdist-worker fixture — the pytest step is strictly read-only.

``seed_case`` assigns random chunk UUIDs, so the ``seed_id -> chunk_uuid`` map it
returns is the ONLY bridge between these rows and the read-only pytest step (a
separate process). This entrypoint therefore persists that map as a JSON artifact
at ``KHORA_CONFORMANCE_SEED_MAP`` (the one-time write); the test loads it and runs
the compiled predicate against the pre-seeded rows without ever re-seeding.

This is a thin driver: it reuses ``seed_case`` verbatim and shares the same
coordinator-construction helper the conformance test module uses, so the rows it
writes are byte-for-byte the rows the test reads back.
"""

from __future__ import annotations

import asyncio

from tests.integration.matrix._conformance_pg import (
    SEED_MAP_PATH,
    build_seed_map,
    write_seed_map,
)


async def main() -> None:
    """Seed every ``postgres``-targeted case once and persist the seed map."""
    seed_map = await build_seed_map()
    write_seed_map(seed_map)
    print(f"seeded {len(seed_map)} postgres conformance cases; wrote seed map to {SEED_MAP_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
