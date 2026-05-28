"""Scenario — Namespace versioning with the storage API.

Khora's ``MemoryNamespace`` carries two UUIDs:

- ``namespace_id`` — STABLE identifier. Same across every version of a
  namespace. ``kb.create_namespace()`` returns it; the high-level
  facade (``kb.remember`` / ``kb.recall`` / ``kb.list_*``) accepts it
  and resolves to the currently-active version under the hood.
- ``id`` — ROW PRIMARY KEY. Different per version. Every FK column in
  ``entities`` / ``relationships`` / ``chunks`` / ``documents`` holds
  this value, and Neo4j ``(:Entity {namespace_id: <row id>})`` does
  too. The coordinator surface (``kb.storage.*``) takes this id
  because each coordinator call is scoped to a specific version.

WHEN YOU WANT VERSIONING
========================
Real cases for cutting a new version on the same namespace:

- Re-indexing with a better extractor model without losing the prior
  pass's data (compare top-k recall side-by-side).
- A/B testing a new chunking strategy on the same corpus.
- Schema migrations — re-ingest under a new ``ExpertiseConfig`` while
  the old graph stays queryable until you cut over.
- "Snapshot" semantics — freeze the current graph as a historical
  version, ingest fresh material into a new version.

In all of these you'd hold onto the stable ``namespace_id`` in your
application code, then either let the facade resolve to the active
version, or use the storage layer with the specific version's row id.

ONE ACTIVE VERSION AT A TIME
============================
``create_namespace_version`` atomically flips ``is_active`` on the
previous row to ``False`` and inserts the new row as active. The
facade (``kb.remember`` / ``kb.recall``) always operates on whichever
version is currently active. Historical versions are read-only and
addressable only through the storage layer using the version's row id.

DUAL-BACKEND SUPPORT
====================
This script accepts ``--config`` pointing at either:

  • ``examples/khora.embedded.yaml``  (SQLite + LanceDB, zero infra; default)
  • ``examples/khora.standard.yaml``  (PostgreSQL + pgvector + Neo4j)

``create_namespace_version`` is implemented on both backends, so the
flow works identically.

Run it
======
uv run python examples/30_workloads/06_namespace_versioning.py
python examples/30_workloads/06_namespace_versioning.py
uv run python examples/30_workloads/06_namespace_versioning.py --config examples/khora.standard.yaml
python examples/30_workloads/06_namespace_versioning.py --config examples/khora.standard.yaml
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from uuid import UUID

from loguru import logger

from khora import Khora
from khora.config import KhoraConfig

# ── Logging setup ───────────────────────────────────────────────────────
# Khora uses loguru. The default sink writes to stderr, which floods the
# terminal with extraction and recall traces. Route the noise to a file
# and keep the terminal showing only this script's `print()` output.
logger.remove()
logger.add("khora.log", level="TRACE", enqueue=True)

# Default to the embedded config so a fresh clone runs without `make dev`.
_DEFAULT_CONFIG = Path(__file__).parent.parent / "khora.embedded.yaml"


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


def line(label: str, value: object) -> None:
    print(f"  {label:34} {value}")


def print_entities(label: str, entities: list) -> None:
    print(f"  {label}: {len(entities)}")
    for e in entities:
        desc = (e.description or "").splitlines()[0][:60]
        suffix = f"  — {desc}" if desc else ""
        print(f"    - {e.name!r} ({e.entity_type}){suffix}")


def print_relationships(label: str, rels: list) -> None:
    print(f"  {label}: {len(rels)}")
    for r in rels:
        src = r.source_entity_name or str(r.source_entity_id)[:8]
        tgt = r.target_entity_name or str(r.target_entity_id)[:8]
        print(f"    - {src} --[{r.relationship_type}]--> {tgt}")


async def list_versions(kb: Khora, stable_id: UUID) -> list:
    """Return all rows on ``memory_namespaces`` that share the given
    stable ``namespace_id``, sorted oldest → newest. Filtered Python-side
    because ``kb.storage.list_namespaces`` has no by-stable-id filter
    in v0.17.
    """
    page = await kb.storage.list_namespaces(active_only=False, limit=200)
    rows = page.items if hasattr(page, "items") else list(page)
    return sorted(
        (n for n in rows if n.namespace_id == stable_id),
        key=lambda n: n.version,
    )


async def row_id_for_version(kb: Khora, stable_id: UUID, version: int) -> UUID:
    """Look up the row id (version handle) for a specific version of a
    stable namespace.
    """
    versions = await list_versions(kb, stable_id)
    match = next((n for n in versions if n.version == version), None)
    if match is None:
        raise ValueError(f"No namespace row with stable_id={stable_id} and version={version}")
    return match.id


def _load_config() -> KhoraConfig:
    """Parse ``--config`` and load the named Khora YAML.

    Kept inline (rather than in a shared helper) so this file is
    readable on its own — copy/paste it into your project and it works
    without dragging an ``examples/_common.py`` along.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help=f"Khora YAML config path (default: {_DEFAULT_CONFIG.name}).",
    )
    args = parser.parse_args()
    return KhoraConfig.from_yaml(args.config)


async def main() -> None:
    config = _load_config()

    async with Khora(config, run_migrations=True) as kb:
        # ──────────────────────────────────────────────────────────────────
        section("Step 1 — Create v1 of a fresh namespace")
        # ──────────────────────────────────────────────────────────────────
        # `create_namespace()` returns a MemoryNamespace with two UUIDs:
        # `namespace_id` (STABLE — hold onto this in your application code)
        # and `id` (ROW PK — the version handle for v1).
        v1 = await kb.create_namespace()
        stable_id = v1.namespace_id
        v1_row_id = v1.id

        line("stable namespace_id", stable_id)
        line("v1 row id (version handle)", v1_row_id)
        line("v1.version", v1.version)
        line("v1.is_active", v1.is_active)

        # ──────────────────────────────────────────────────────────────────
        section("Step 2 — Store a document in v1 via the facade")
        # ──────────────────────────────────────────────────────────────────
        # The facade takes the STABLE id. It resolves to the active row
        # under the hood, so the write lands in v1 (the only version so far).
        await kb.remember(
            "Marie Curie won the Nobel Prize in Physics in 1903.",
            namespace=stable_id,
            entity_types=["PERSON", "AWARD"],
            relationship_types=["WON"],
        )

        # Read v1's data via the coordinator using the v1 ROW id.
        v1_entities = await kb.storage.list_entities(v1_row_id)
        v1_rels = await kb.storage.list_relationships(v1_row_id)
        print_entities("v1 entities", v1_entities)
        print_relationships("v1 relationships", v1_rels)

        # ──────────────────────────────────────────────────────────────────
        section("Step 3 — Cut v2 of the same namespace")
        # ──────────────────────────────────────────────────────────────────
        # `create_namespace_version` takes the previous-version
        # MemoryNamespace object. It atomically deactivates v1 and creates
        # v2 as the new active row, sharing the same stable id.
        v2 = await kb.storage.create_namespace_version(previous_version=v1)
        v2_row_id = v2.id

        line("v2 stable namespace_id", v2.namespace_id)  # same as v1
        line("v2 row id", v2_row_id)  # different from v1
        line("v2.version", v2.version)  # 2
        line("v2.is_active", v2.is_active)  # True

        # ──────────────────────────────────────────────────────────────────
        section("Step 4 — Store fresh data in v2 via the facade")
        # ──────────────────────────────────────────────────────────────────
        # The facade still takes the stable id; resolution now picks v2.
        await kb.remember(
            "Albert Einstein developed the theory of relativity in 1905.",
            namespace=stable_id,
            entity_types=["PERSON", "CONCEPT"],
            relationship_types=["DEVELOPED"],
        )

        v2_entities = await kb.storage.list_entities(v2_row_id)
        v2_rels = await kb.storage.list_relationships(v2_row_id)
        print_entities("v2 entities", v2_entities)
        print_relationships("v2 relationships", v2_rels)

        # v1's data is untouched.
        print()
        print("  v1 after v2 ingest (should be unchanged):")
        v1_entities_after = await kb.storage.list_entities(v1_row_id)
        v1_rels_after = await kb.storage.list_relationships(v1_row_id)
        print_entities("v1 entities (still)", v1_entities_after)
        print_relationships("v1 relationships (still)", v1_rels_after)

        # ──────────────────────────────────────────────────────────────────
        section("Step 5 — Enumerate versions of this stable namespace")
        # ──────────────────────────────────────────────────────────────────
        # `kb.storage.list_namespaces` returns every namespace row in the
        # database; filter Python-side to the stable id you care about.
        versions = await list_versions(kb, stable_id)
        line("versions of this stable id", len(versions))
        for n in versions:
            line(f"  v{n.version}", f"row_id={n.id} is_active={n.is_active}")

        # ──────────────────────────────────────────────────────────────────
        section("Step 6 — Resolve stable id → active row id")
        # ──────────────────────────────────────────────────────────────────
        # `kb.storage.resolve_namespace(stable_id)` returns the row id of
        # the currently-active version. This is the supported bridge from
        # a stable id (what your application code holds) to a row id
        # (what the coordinator methods need).
        active_row = await kb.storage.resolve_namespace(stable_id)
        line("active row id (via resolve)", active_row)
        line("matches v2.id", active_row == v2_row_id)

        # For a SPECIFIC version's row id (not just the active one), use
        # the `row_id_for_version` helper at the top of this script.
        v1_row_again = await row_id_for_version(kb, stable_id, version=1)
        line("row id for v1 (via helper)", v1_row_again)
        line("matches original v1.id", v1_row_again == v1_row_id)

        # ──────────────────────────────────────────────────────────────────
        section("Step 7 — Read either version's data through kb.storage.*")
        # ──────────────────────────────────────────────────────────────────
        # Given a stable id and a version number, look up that version's
        # row id and pass it to the coordinator's list_* methods.
        for ver in (1, 2):
            row = await row_id_for_version(kb, stable_id, version=ver)
            ents = await kb.storage.list_entities(row)
            rels = await kb.storage.list_relationships(row)
            print(f"\n  --- v{ver} (row_id={row}) ---")
            print_entities("    entities", ents)
            print_relationships("    relationships", rels)

        # ──────────────────────────────────────────────────────────────────
        section("Recap — the version-aware idiom")
        # ──────────────────────────────────────────────────────────────────
        print(
            """
  Hold onto the stable namespace_id in your application code. When you
  need to read a SPECIFIC version's data, look up its row id from
  `list_namespaces`:

      page = await kb.storage.list_namespaces(active_only=False)
      versions = [n for n in page.items if n.namespace_id == stable_id]
      target   = next(n for n in versions if n.version == wanted_version)

      entities = await kb.storage.list_entities(target.id)
      rels     = await kb.storage.list_relationships(target.id)

  When you only need the CURRENTLY-ACTIVE version's data:

      active_row = await kb.storage.resolve_namespace(stable_id)
      entities   = await kb.storage.list_entities(active_row)

  Or just use the facade (kb.list_entities, kb.recall, ...) — it does
  the same resolve under the hood.
            """.rstrip()
        )


if __name__ == "__main__":
    asyncio.run(main())
