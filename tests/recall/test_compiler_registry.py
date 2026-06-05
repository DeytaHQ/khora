"""Unit tests for the internal compiler registry (Layer 4 seam).

Pins: register/get round-trips on ``(engine_id, storage_target)``, a clear
``KhoraError`` subclass on a miss, idempotent same-fn re-register, a raise on a
conflicting fn, thread-safe concurrent registration, and ``CompiledFilter``
typing/defaults — plus the internal-only rule that the registry is absent from
``khora.__all__``.
"""

from __future__ import annotations

import threading

import pytest

from khora.exceptions import KhoraError
from khora.filter.ast import FilterNode
from khora.filter.context import CompileContext
from khora.filter.model import Op
from khora.filter.registry import (
    CompiledFilter,
    CompilerConflictError,
    CompilerRegistry,
    UnknownCompilerError,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    """Each test starts and ends with an empty registry (process-wide state).

    The registry is process-wide class state; this fixture prevents tests from
    leaking registrations into one another. Tests additionally use unique
    engine_id / storage_target keys so a concurrent or mis-ordered run cannot
    cross-contaminate.
    """
    CompilerRegistry._clear()
    yield
    CompilerRegistry._clear()


def _fake_compiler(ast: FilterNode, ctx: CompileContext) -> CompiledFilter:
    return CompiledFilter(predicate="WHERE 1=1", params={}, consumed_keys=frozenset(), canonical_hash="abc")


def _other_compiler(ast: FilterNode, ctx: CompileContext) -> CompiledFilter:
    return CompiledFilter(predicate="WHERE 2=2", params={}, consumed_keys=frozenset(), canonical_hash="def")


# ---------------------------------------------------------------------------
# register / get round-trip.
# ---------------------------------------------------------------------------


def test_register_then_get_returns_same_fn() -> None:
    CompilerRegistry.register("vectorcypher", "khora_chunks", _fake_compiler)
    assert CompilerRegistry.get("vectorcypher", "khora_chunks") is _fake_compiler


def test_registry_keyed_by_engine_and_storage_target() -> None:
    CompilerRegistry.register("vectorcypher", "khora_chunks", _fake_compiler)
    CompilerRegistry.register("vectorcypher", "neo4j_chunk", _other_compiler)
    assert CompilerRegistry.get("vectorcypher", "khora_chunks") is _fake_compiler
    assert CompilerRegistry.get("vectorcypher", "neo4j_chunk") is _other_compiler


def test_registry_empty_at_import() -> None:
    # After _clear (autouse fixture), nothing is registered — the registry ships
    # empty; engines register at import time, not the registry module.
    with pytest.raises(UnknownCompilerError):
        CompilerRegistry.get("vectorcypher", "khora_chunks")


def test_registry_is_a_non_instantiable_singleton() -> None:
    # The registry is process-wide class state used via classmethods; constructing
    # an instance is a misuse and raises TypeError.
    with pytest.raises(TypeError):
        CompilerRegistry()


def test_registered_compiler_round_trips_and_is_callable() -> None:
    # The CompilerFn contract is usable end-to-end: a registered compiler is
    # fetched and invoked as Callable[[FilterNode, CompileContext], CompiledFilter].
    CompilerRegistry.register("vectorcypher", "khora_chunks", _fake_compiler)
    compiler = CompilerRegistry.get("vectorcypher", "khora_chunks")
    result = compiler(FilterNode(op=Op.AND), CompileContext(backend_target="khora_chunks"))
    assert isinstance(result, CompiledFilter)
    assert result.predicate == "WHERE 1=1"


# ---------------------------------------------------------------------------
# Miss raises a clear KhoraError subclass.
# ---------------------------------------------------------------------------


def test_get_missing_raises_unknown_compiler_error() -> None:
    with pytest.raises(UnknownCompilerError):
        CompilerRegistry.get("nope", "nope")


def test_unknown_compiler_error_is_khora_error() -> None:
    assert issubclass(UnknownCompilerError, KhoraError)
    with pytest.raises(KhoraError):
        CompilerRegistry.get("nope", "nope")


def test_unknown_compiler_error_carries_key() -> None:
    try:
        CompilerRegistry.get("eng", "store")
    except UnknownCompilerError as exc:
        assert exc.engine_id == "eng"
        assert exc.storage_target == "store"
    else:  # pragma: no cover
        pytest.fail("expected UnknownCompilerError")


# ---------------------------------------------------------------------------
# Idempotent same-fn re-register; conflicting fn raises.
# ---------------------------------------------------------------------------


def test_reregister_same_fn_is_idempotent() -> None:
    CompilerRegistry.register("vectorcypher", "khora_chunks", _fake_compiler)
    CompilerRegistry.register("vectorcypher", "khora_chunks", _fake_compiler)  # no raise
    assert CompilerRegistry.get("vectorcypher", "khora_chunks") is _fake_compiler


def test_reregister_different_fn_raises() -> None:
    CompilerRegistry.register("vectorcypher", "khora_chunks", _fake_compiler)
    with pytest.raises(CompilerConflictError):
        CompilerRegistry.register("vectorcypher", "khora_chunks", _other_compiler)
    # The original registration is preserved (never silently overwritten).
    assert CompilerRegistry.get("vectorcypher", "khora_chunks") is _fake_compiler


def test_conflict_error_is_khora_error() -> None:
    assert issubclass(CompilerConflictError, KhoraError)


def test_conflict_error_carries_key() -> None:
    CompilerRegistry.register("eng", "store", _fake_compiler)
    try:
        CompilerRegistry.register("eng", "store", _other_compiler)
    except CompilerConflictError as exc:
        assert exc.engine_id == "eng"
        assert exc.storage_target == "store"
    else:  # pragma: no cover
        pytest.fail("expected CompilerConflictError")


# ---------------------------------------------------------------------------
# Thread-safe concurrent registration (class-level state, class lock).
# ---------------------------------------------------------------------------


def test_concurrent_register_same_fn_is_safe() -> None:
    # Many threads racing to register the SAME function on the SAME key must all
    # succeed (idempotent) with no lost update / spurious conflict.
    errors: list[BaseException] = []
    barrier = threading.Barrier(16)

    def worker() -> None:
        try:
            barrier.wait()
            CompilerRegistry.register("vc", "same_target", _fake_compiler)
        except BaseException as exc:  # noqa: BLE001 - record for the assertion
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert CompilerRegistry.get("vc", "same_target") is _fake_compiler


def test_concurrent_register_distinct_keys_all_land() -> None:
    # Each thread registers on a distinct key; the lock must serialize the dict
    # mutations so every registration survives (no dropped writes under races).
    barrier = threading.Barrier(20)

    def worker(i: int) -> None:
        barrier.wait()
        CompilerRegistry.register("vc", f"target_{i}", _fake_compiler)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for i in range(20):
        assert CompilerRegistry.get("vc", f"target_{i}") is _fake_compiler


def test_concurrent_conflicting_register_keeps_one_winner() -> None:
    # Threads race to register DIFFERENT functions on the SAME key. The lock must
    # serialize so exactly one fn wins and every loser sees a CompilerConflictError
    # (never a corrupted/partial registration). The final state is one of the two.
    conflicts: list[CompilerConflictError] = []
    barrier = threading.Barrier(16)
    compilers = (_fake_compiler, _other_compiler)

    def worker(i: int) -> None:
        barrier.wait()
        try:
            CompilerRegistry.register("vc", "contested", compilers[i % 2])
        except CompilerConflictError as exc:
            conflicts.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    winner = CompilerRegistry.get("vc", "contested")
    assert winner in compilers
    # Everyone who tried to register the OTHER function lost with a clean conflict.
    assert all(isinstance(c, CompilerConflictError) for c in conflicts)


# ---------------------------------------------------------------------------
# CompiledFilter typing + defaults.
# ---------------------------------------------------------------------------


def test_compiled_filter_fields() -> None:
    cf = CompiledFilter(
        predicate="WHERE x = :f_0",
        params={"f_0": 1},
        consumed_keys=frozenset({"source_name"}),
        canonical_hash="deadbeef",
    )
    assert cf.predicate == "WHERE x = :f_0"
    assert cf.params == {"f_0": 1}
    assert cf.consumed_keys == frozenset({"source_name"})
    assert cf.canonical_hash == "deadbeef"


def test_compiled_filter_is_frozen() -> None:
    cf = CompiledFilter(predicate=1, params={}, consumed_keys=frozenset(), canonical_hash="h")
    with pytest.raises((AttributeError, TypeError)):
        cf.predicate = 2  # type: ignore[misc]


def test_compiled_filter_predicate_type_is_generic() -> None:
    # T varies by backend: a SQLAlchemy expr, a Cypher fragment, a callable, ...
    # The dataclass carries whatever predicate type the backend emits verbatim.
    def _callable_predicate(record: object) -> bool:  # pragma: no cover - never called
        return True

    cf = CompiledFilter(
        predicate=_callable_predicate,
        params={},
        consumed_keys=frozenset({"source_name", "title"}),
        canonical_hash="h",
    )
    assert cf.predicate is _callable_predicate
    assert cf.consumed_keys == frozenset({"source_name", "title"})


def test_compiled_filter_consumed_keys_is_frozenset() -> None:
    cf = CompiledFilter(
        predicate="p",
        params={"f_0": 1},
        consumed_keys=frozenset({"a"}),
        canonical_hash="h",
    )
    assert isinstance(cf.consumed_keys, frozenset)


# ---------------------------------------------------------------------------
# Internal-only — registry is NOT exported from khora.__all__.
# ---------------------------------------------------------------------------


# The full set of internal Layer 3/4 seam names that AC2 requires to be absent
# from the top-level ``khora.__all__`` while remaining reachable via
# ``khora.filter`` (and its submodules).
_INTERNAL_SEAM_NAMES = (
    "CompilerRegistry",
    "CompileContext",
    "SchemaCapabilities",
    "FilterNode",
    "parse_to_ast",
    "CompiledFilter",
    "CompilerFn",
)


def test_ac2_internal_seam_absent_from_khora_top_level_all() -> None:
    import khora

    for name in _INTERNAL_SEAM_NAMES:
        assert name not in khora.__all__, f"{name} leaked into khora.__all__"


def test_ac2_internal_seam_not_in_khora_filter_public_all() -> None:
    # khora.filter.__all__ is the PUBLIC surface; the internal names are reachable
    # as attributes but not part of __all__.
    import khora.filter as f

    for name in _INTERNAL_SEAM_NAMES:
        assert name not in f.__all__, f"{name} leaked into khora.filter.__all__"


def test_ac2_registry_importable_from_khora_filter() -> None:
    # `from khora.filter import CompilerRegistry` must work even though it's not
    # in __all__.
    from khora.filter import CompiledFilter as PkgCompiledFilter
    from khora.filter import CompilerRegistry as PkgRegistry

    assert PkgRegistry is CompilerRegistry
    assert PkgCompiledFilter is CompiledFilter


def test_ac2_registry_importable_via_submodule() -> None:
    # ...and via the concrete submodule path.
    from khora.filter.registry import CompilerRegistry as ModRegistry

    assert ModRegistry is CompilerRegistry


def test_registry_reachable_from_khora_filter() -> None:
    import khora.filter as f

    assert f.CompilerRegistry is CompilerRegistry
    assert f.CompiledFilter is CompiledFilter
