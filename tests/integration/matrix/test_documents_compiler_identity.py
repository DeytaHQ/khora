"""Compiler-identity guard for the four ``documents``-tier registrations.

The sibling drift guard (``test_compiler_registry_keys.py``) asserts WHICH
``(engine_id, storage_target)`` keys the registry holds. It cannot see what each
key is bound TO: a documents key wired to the wrong backend's compiler resolves
to a perfectly callable function and passes it. This module closes that half —
it asserts the compiler FUNCTION OBJECT behind each documents key, so a
copy-paste registration (the plausible mistake, since three of the four blocks
are near-identical and two legitimately share ``compile_lance``) fails here.

Deliberately NOT a second exact-key-set assertion: two copies of that assertion
would drift apart, and the sibling guard already owns it.

**Why this lives in the integration session and not next to the unit tests for
these contexts.** The registry is process-wide class state, and the unit suite
contains a module whose autouse fixture clears it around every test
(``tests/recall/test_compiler_registry.py``). Registration is an import-time
side effect, so once a module is in ``sys.modules`` a later import cannot re-fire
it — a registry read scheduled after that fixture sees an empty registry. The
unit job runs those tests in the same session under ``xdist``, which makes the
ordering nondeterministic; the integration job never collects them. Same reason
the sibling guard sits in a session of its own.
"""

from __future__ import annotations

import pytest

# Importing these modules fires their module-level ``CompilerRegistry.register``
# calls — two are on ``import khora``'s eager path, two are imported lazily by
# ``StorageFactory``, so all four are named explicitly rather than assumed.
import khora.storage.backends.postgresql  # noqa: F401
import khora.storage.backends.sqlite  # noqa: F401
import khora.storage.backends.sqlite_lance.relational  # noqa: F401
import khora.storage.backends.surrealdb.relational  # noqa: F401
from khora.filter.compilers.lance import compile_lance
from khora.filter.compilers.postgres import compile_postgres
from khora.filter.compilers.surrealdb import compile_surrealdb
from khora.filter.registry import CompilerFn, CompilerRegistry

pytestmark = [pytest.mark.integration]


# Each documents-tier registry key and the compiler its store module must bind
# it to. Verified at the ``CompilerRegistry.register(...)`` call-site in each
# backend module imported above. The two SQLite dialects share ``compile_lance``
# by design — they differ in their compile context, not their compiler.
EXPECTED_DOCUMENTS_COMPILERS: tuple[tuple[str, str, CompilerFn], ...] = (
    ("relational.postgresql", "documents", compile_postgres),
    ("relational.sqlite", "documents", compile_lance),
    ("relational.sqlite_lance", "documents", compile_lance),
    ("relational.surrealdb", "documents", compile_surrealdb),
)


@pytest.mark.parametrize(
    ("engine_id", "storage_target", "expected"),
    EXPECTED_DOCUMENTS_COMPILERS,
    ids=[f"{key}" for key, _target, _fn in EXPECTED_DOCUMENTS_COMPILERS],
)
def test_documents_key_resolves_to_its_own_compiler(
    engine_id: str,
    storage_target: str,
    expected: CompilerFn,
) -> None:
    """Identity, not callability — the wrong compiler is also callable."""
    assert CompilerRegistry.get(engine_id, storage_target) is expected


def test_documents_keys_are_not_all_bound_to_one_compiler() -> None:
    """Three distinct compilers back the four documents keys.

    A registration block copy-pasted across the four store modules without
    swapping the compiler import would still satisfy every per-key assertion in
    isolation if the expectations were derived from the registry. This pins the
    shape of the mapping itself: three distinct functions, with the SQLite pair
    sharing one and the other two standing alone.
    """
    resolved = [CompilerRegistry.get(key, target) for key, target, _fn in EXPECTED_DOCUMENTS_COMPILERS]
    assert len({id(fn) for fn in resolved}) == 3
    assert CompilerRegistry.get("relational.sqlite", "documents") is CompilerRegistry.get(
        "relational.sqlite_lance", "documents"
    )
    assert CompilerRegistry.get("relational.postgresql", "documents") is not CompilerRegistry.get(
        "relational.surrealdb", "documents"
    )
