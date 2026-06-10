"""Self-tests for the shared filter-spy helper.

The embedded and live-DB enforcement spies trust ``assert_filter_threaded`` and
``spy_on`` to FAIL when a filter is dropped, mutated, or never threaded. If the
helper's own assertion were vacuous, every spy that depends on it would go green
for the wrong reason. These tests pin the helper's non-vacuity directly — no
database, no engine — so the guarantee is committed, not ad-hoc.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from khora.filter import RecallFilter, parse_to_ast
from tests.test_helpers.filter_spy import (
    FilterCallRecord,
    assert_filter_threaded,
    expected_hash,
    seed_corpus,
    spy_on,
)

pytestmark = [pytest.mark.unit, pytest.mark.filter_enforcement]

_FILTER = {"source_name": "linear", "occurred_at": {"$gte": "2026-01-01"}}


def _record_for(doc: dict) -> FilterCallRecord:
    """A record whose positional arg is the AST for ``doc``."""
    return FilterCallRecord(args=(parse_to_ast(RecallFilter.model_validate(doc)),), kwargs={})


def test_correct_filter_passes() -> None:
    assert_filter_threaded([_record_for(_FILTER)], _FILTER, min_calls=1)


def test_filter_in_kwarg_is_captured() -> None:
    """filter_ast passed as a kwarg (the channel-method shape) is captured."""
    ast = parse_to_ast(RecallFilter.model_validate(_FILTER))
    rec = FilterCallRecord(args=(), kwargs={"filter_ast": ast})
    assert_filter_threaded([rec], _FILTER, min_calls=1)


def test_vacuity_guard_fails_on_zero_calls() -> None:
    """An empty record list FAILS — the channel was never exercised."""
    with pytest.raises(AssertionError, match="vacuity guard"):
        assert_filter_threaded([], _FILTER, min_calls=1)


def test_vacuity_guard_respects_min_calls() -> None:
    """Fewer calls than the floor FAILS even if each one is correct."""
    with pytest.raises(AssertionError, match="vacuity guard"):
        assert_filter_threaded([_record_for(_FILTER)], _FILTER, min_calls=2)


def test_dropped_filter_fails() -> None:
    """A call that carried NO filter_ast FAILS (the filter was dropped)."""
    rec = FilterCallRecord(args=(), kwargs={})
    with pytest.raises(AssertionError, match="NO filter_ast"):
        assert_filter_threaded([rec], _FILTER, min_calls=1)


def test_wrong_filter_fails() -> None:
    """A call carrying a different filter FAILS on canonical_hash mismatch."""
    with pytest.raises(AssertionError, match="canonical_hash"):
        assert_filter_threaded([_record_for({"source_name": "slack"})], _FILTER, min_calls=1)


def test_one_good_one_bad_fails() -> None:
    """If ANY call diverges, the assertion FAILS — not a majority vote."""
    records = [_record_for(_FILTER), _record_for({"source_name": "slack"})]
    with pytest.raises(AssertionError):
        assert_filter_threaded(records, _FILTER, min_calls=1)


def test_expected_hash_accepts_instance_and_dict() -> None:
    inst = RecallFilter.model_validate(_FILTER)
    assert expected_hash(_FILTER) == expected_hash(inst)


async def test_spy_on_passes_through_and_records() -> None:
    """spy_on records each call AND returns the real method's result."""

    class _Chan:
        async def search(self, *, filter_ast=None):  # noqa: ANN001, ANN202
            return ("ran", filter_ast)

    chan = _Chan()
    mp = pytest.MonkeyPatch()
    try:
        records = spy_on(mp, chan, "search")
        ast = parse_to_ast(RecallFilter.model_validate(_FILTER))
        out = await chan.search(filter_ast=ast)
        # Pass-through: the real method still ran and returned its value.
        assert out == ("ran", ast)
        # Recorded: one call, captured with the right hash.
        assert_filter_threaded(records, _FILTER, min_calls=1)
    finally:
        mp.undo()


def test_spy_on_records_sync_target() -> None:
    """spy_on works on a SYNC module-level function (the compile_cypher shape).

    qa-graph's path-3/8 spy point is the sync ``compile_cypher(ast, ctx)``; the
    filter is the FIRST POSITIONAL arg, not a kwarg. The sync wrapper must record
    it (via FilterCallRecord's positional-FilterNode fallback) AND pass through
    without awaiting a non-coroutine.
    """
    import types

    mod = types.ModuleType("fake_compiler_mod")

    def compile_like(ast, ctx):  # noqa: ANN001, ANN202 — mimics compile_cypher(ast, ctx)
        return ("compiled", ast, ctx)

    mod.compile_like = compile_like  # type: ignore[attr-defined]

    mp = pytest.MonkeyPatch()
    try:
        records = spy_on(mp, mod, "compile_like")
        ast = parse_to_ast(RecallFilter.model_validate(_FILTER))
        out = mod.compile_like(ast, {"node": "Chunk"})  # type: ignore[attr-defined]
        # Pass-through: sync call returns the real value (not a coroutine).
        assert out == ("compiled", ast, {"node": "Chunk"})
        # Recorded with the positional AST resolved to the right hash.
        assert_filter_threaded(records, _FILTER, min_calls=1)
    finally:
        mp.undo()


async def test_seed_corpus_threads_per_doc_metadata() -> None:
    """seed_corpus passes per-doc metadata/title to remember; plain strings don't.

    Path 8's residual-metadata predicate needs a plantable metadata key on the
    seeded chunk. A dict doc must thread ``metadata=`` (and ``title=``) to
    remember; a plain-string doc must call remember with neither, so the two
    shapes coexist.
    """
    calls: list[dict] = []

    async def fake_remember(**kwargs):  # noqa: ANN003, ANN202
        calls.append(kwargs)

    ns = uuid4()
    await seed_corpus(
        fake_remember,
        ns,
        [
            "plain string doc",
            {"content": "eng doc", "metadata": {"channel": "eng"}, "title": "T"},
        ],
    )

    assert calls[0] == {"content": "plain string doc", "namespace": ns}
    assert calls[1] == {
        "content": "eng doc",
        "namespace": ns,
        "metadata": {"channel": "eng"},
        "title": "T",
    }
