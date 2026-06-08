"""Compiler-registry drift guard for the recall-filter conformance corpus.

The conformance corpus dispatches every case through ``CompilerRegistry.get``;
this guard is what keeps the registry and the corpus reconciled — if an engine
renames its ``(engine_id, storage_target)`` key, or stops registering a compiler,
this test fails LOUDLY (``UnknownCompilerError``) rather than letting the corpus
silently skip a backend.

Pure import test — it imports the engine/backend modules so their import-time
``CompilerRegistry.register(...)`` calls fire, then asserts every expected key
still resolves. No live store, so it is NOT gated behind ``_pg_reachable`` and
runs on every conformance CI leg (including the no-Docker ones).
"""

from __future__ import annotations

import pytest

# Importing these modules fires their module-level ``CompilerRegistry.register``
# calls (the registry is empty until an engine/backend module imports).
import khora.engines.chronicle.engine  # noqa: F401
import khora.engines.skeleton.backends.pgvector  # noqa: F401
import khora.engines.skeleton.backends.sqlite_lance  # noqa: F401
import khora.engines.skeleton.backends.surrealdb  # noqa: F401
import khora.engines.skeleton.backends.weaviate  # noqa: F401
from khora.filter.registry import CompilerRegistry

pytestmark = [pytest.mark.filter_conformance]


# Single source of truth for the registry keys the conformance corpus relies on.
# Each is verified at its ``CompilerRegistry.register(...)`` call-site in the
# engine/backend module imported above; KEEP THIS LIST IN LOCKSTEP WITH THOSE.
# When a new compiler (e.g. cypher) lands and registers at import time, add its
# ``(engine_id, storage_target)`` key here. Failure to update allows the compiler
# to register silently while the conformance job silently excludes it — a dangerous
# skew. This guard fails loudly (``UnknownCompilerError``) only on a key that
# VANISHES or is renamed; a NEW unlisted key is caught by this comment + review.
# Currently registered: chronicle, skeleton.pgvector, skeleton.sqlite_lance,
# skeleton.surrealdb, skeleton.weaviate.
EXPECTED_KEYS: tuple[tuple[str, str], ...] = (
    ("chronicle", "chunks"),
    ("skeleton.pgvector", "khora_chunks"),
    ("skeleton.sqlite_lance", "khora_chunks"),
    ("skeleton.surrealdb", "temporal_chunk"),
    ("skeleton.weaviate", "KhoraChunk"),
)


@pytest.mark.parametrize(("engine_id", "storage_target"), EXPECTED_KEYS)
def test_expected_compiler_key_resolves(engine_id: str, storage_target: str) -> None:
    """Each expected ``(engine_id, storage_target)`` resolves to a real compiler.

    ``CompilerRegistry.get`` raises ``UnknownCompilerError`` on a missing key, so
    a drifted/renamed registration fails this test loudly instead of degrading
    into a silently-skipped conformance backend.
    """
    compiler = CompilerRegistry.get(engine_id, storage_target)
    assert callable(compiler)
