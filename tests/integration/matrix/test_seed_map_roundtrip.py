"""Seed-map serialization guard — runs on EVERY conformance leg, no DB.

The read-only postgres conformance design hinges on one artifact: the JSON seed
map the one-time seed step writes (``write_seed_map``) and the read-only test step
loads (``load_seed_map``). If that round-trip drifts — the ``{case_id: {seed_id:
chunk_id}}`` shape, the ``str``↔``UUID`` coercion, the missing-file contract —
the whole postgres leg silently mis-maps survivors. These checks guard exactly
that, with no database, so they run on every leg (including the no-Docker ones)
and catch a regression even when Postgres is unavailable.

Marked ``integration`` + ``filter_conformance`` (NOT ``_pg_reachable``-gated): the
functions under test are pure file I/O over a ``tmp_path`` artifact, so the test
needs no live store and runs on every conformance leg.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from tests.integration.matrix import _conformance_pg

pytestmark = [pytest.mark.integration, pytest.mark.filter_conformance]


@pytest.fixture
def _seed_map_at(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point the seed-map helpers at a ``tmp_path`` file and reset the load cache.

    ``write_seed_map`` / ``load_seed_map`` read the module-level ``SEED_MAP_PATH``
    constant, so the test redirects it via monkeypatch. ``load_seed_map`` is
    ``lru_cache``-d, so the cache is cleared before and after to keep cases isolated.
    """
    path = tmp_path / "seed_map.json"
    monkeypatch.setattr(_conformance_pg, "SEED_MAP_PATH", str(path))
    _conformance_pg.load_seed_map.cache_clear()
    try:
        yield path
    finally:
        _conformance_pg.load_seed_map.cache_clear()


def test_seed_map_round_trip_preserves_ids_and_uuid_type(_seed_map_at: Path) -> None:
    """write_seed_map -> load_seed_map preserves case_ids, seed_ids, and UUID identity."""
    chunk_a, chunk_b, chunk_c = uuid4(), uuid4(), uuid4()
    original = {
        "F-OP-created_at-gt": {"created_at-hit": str(chunk_a), "created_at-miss": str(chunk_b)},
        "F-OP-metadata-tier-eq": {"meta-gold": str(chunk_c)},
    }

    _conformance_pg.write_seed_map(original)
    loaded = _conformance_pg.load_seed_map()

    # Same case_ids and, per case, the same seed_ids.
    assert loaded.keys() == original.keys()
    assert loaded["F-OP-created_at-gt"].keys() == {"created_at-hit", "created_at-miss"}
    assert loaded["F-OP-metadata-tier-eq"].keys() == {"meta-gold"}

    # Chunk ids come back as UUID objects equal to the originals (the str<->UUID
    # coercion the ``id = ANY(:ids)`` bind depends on).
    assert loaded["F-OP-created_at-gt"]["created_at-hit"] == chunk_a
    assert isinstance(loaded["F-OP-created_at-gt"]["created_at-hit"], UUID)
    assert loaded["F-OP-created_at-gt"]["created_at-miss"] == chunk_b
    assert loaded["F-OP-metadata-tier-eq"]["meta-gold"] == chunk_c


def test_load_seed_map_missing_file_raises_actionable_error(_seed_map_at: Path) -> None:
    """A missing map raises FileNotFoundError naming the seed step (the read-only contract)."""
    assert not _seed_map_at.exists()  # nothing written
    with pytest.raises(FileNotFoundError) as exc_info:
        _conformance_pg.load_seed_map()
    message = str(exc_info.value)
    # The error must point an operator at the fix, not just say "no such file".
    assert "_conformance_seed" in message
    assert "read-only" in message
