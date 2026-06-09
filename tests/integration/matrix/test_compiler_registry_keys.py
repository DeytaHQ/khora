"""Compiler-registry drift guard for the recall-filter conformance corpus.

The conformance corpus dispatches every case through ``CompilerRegistry.get``;
this guard is what keeps the registry and the corpus reconciled. It catches drift
in **both** directions:

* a removed/renamed key — :func:`test_expected_compiler_key_resolves` fails LOUDLY
  (``UnknownCompilerError``) instead of letting the corpus silently skip a backend;
* a **new, unlisted** registration — :func:`test_registry_holds_exactly_expected_keys`
  asserts the registry contains *exactly* ``EXPECTED_KEYS``, so a compiler that
  registers without being added here fails the guard rather than being silently
  excluded from the conformance matrix (this is the add-blind gap the earlier
  comment-and-review-only contract left open).

Pure import test — it imports the engine/backend modules so their import-time
``CompilerRegistry.register(...)`` calls fire, then checks the resulting key set.
No live store, so it is NOT gated behind ``_pg_reachable`` and runs on every
conformance CI leg (including the no-Docker ones). The conformance job selects
only ``filter_conformance``-marked tests, so the registry-clearing unit tests in
``tests/recall/test_compiler_registry.py`` never run in the same session.
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
# engine/backend module imported above. When a new compiler (e.g. cypher) lands
# and registers at import time, add its ``(engine_id, storage_target)`` key here:
# ``test_registry_holds_exactly_expected_keys`` fails until you do, so a new
# registration can no longer be silently excluded from the conformance matrix.
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


def test_registry_holds_exactly_expected_keys() -> None:
    """The registry holds *exactly* ``EXPECTED_KEYS`` — no more, no less.

    The per-key resolve test above catches a removed or renamed key, but is
    add-blind: a newly-registered compiler that is not in ``EXPECTED_KEYS`` still
    passes it. This closes that gap — a new registration that lands without being
    added to ``EXPECTED_KEYS`` fails here, so it cannot be silently excluded from
    the conformance matrix. The diff in the message names exactly what drifted.
    """
    registered = CompilerRegistry.registered_keys()
    expected = frozenset(EXPECTED_KEYS)
    unexpected = registered - expected
    missing = expected - registered
    assert registered == expected, (
        f"compiler registry drift — update EXPECTED_KEYS in this file to match the "
        f"registrations: unexpected (registered but unlisted) = {sorted(unexpected)}; "
        f"missing (listed but not registered) = {sorted(missing)}"
    )
