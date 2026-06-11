"""Audit gate: every VectorCypher channel call must carry the recall filter.

The filter-enforcement feature (#1051) threads the caller ``filter_ast``
through every adaptive sub-search so no channel silently drops or mis-applies
it. This meta-test is the structural backstop for that contract: it statically
walks ``src/khora/engines/vectorcypher`` and asserts that every CALL to a
filter-aware channel method passes ``filter_ast=``. Any call that omits it is a
site where a filter-violating chunk could reach RRF fusion — so the omission set
must equal a small, explicitly-justified allowlist. A NEW unguarded call fails
this gate until it is either fixed (thread the filter) or consciously added to
the allowlist with a reason.

Modelled on the ``SYSTEM_KEYS`` coverage meta-test in
``src/khora/filter/conformance.py``: enumerate the surface from source, diff
against a declared set, fail on drift. No database, no imports of the engine —
pure ``ast`` over the source files, so it runs in the main ``test`` job with no
infrastructure.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.filter_enforcement]

# The retriever module is the surface the filter is threaded through.
_RETRIEVER = Path(__file__).resolve().parents[2] / "src" / "khora" / "engines" / "vectorcypher" / "retriever.py"

# Channel / sub-search methods that ACCEPT a ``filter_ast`` parameter. Every
# call to one of these is expected to pass ``filter_ast=`` — that is the
# thread-through contract. Derived from "every method in retriever.py whose
# signature declares a filter_ast parameter" (see test_param_set_is_current).
_FILTER_AWARE_METHODS = frozenset(
    {
        "retrieve",
        "_typed_entity_recent_retrieve",
        "_vectorcypher_retrieve",
        "_simple_retrieve",
        "_fetch_chunks_from_entities",
        "_vector_search_chunks",
        "_recency_channel_chunks",
        "_bm25_search_chunks",
        "_vector_only_fallback",
    }
)

# Calls that deliberately OMIT ``filter_ast`` — each MUST carry a justification
# below. Keyed by the call's line number is brittle across edits, so we key by
# (called_method, enclosing_function) which is stable under line drift.
#
# (method_called, enclosing_function): reason
_ALLOWLISTED_OMISSIONS: dict[tuple[str, str], str] = {
    (
        "_vector_search_chunks",
        "_vectorcypher_retrieve",
    ): (
        "Restrictive-filter unfiltered re-run: re-searches with temporal_filter=None "
        "and intentionally drops the caller filter. GUARDED by an explicit `filter_ast "
        "is None` precondition on the enclosing `if`, so it is unreachable whenever a "
        "caller filter is present — it cannot smuggle filter-violating chunks into RRF. "
        "Covered behaviorally by the PG restrictive-fallback spy (qa-graph) and the "
        "embedded point-in-time fail-fast test."
    ),
}

# Storage-layer boundaries that RECEIVE filter_ast (the retriever threads it
# correctly — wiring verified by the spies) but do NOT compile/enforce it yet.
# These are NOT retriever call-site omissions, so they live here as tracked
# "wiring-done, enforcement-pending" boundaries rather than in the omission
# allowlist. Pillar-2 (the filter REACHES the channel) is satisfied; Pillar-4
# (the channel ENFORCES it) is tracked for follow-up. Each is keyed by
# "module::method" with a reason.
_ENFORCEMENT_PENDING_BOUNDARIES: dict[str, str] = {}


def _retriever_tree() -> ast.Module:
    return ast.parse(_RETRIEVER.read_text())


def _enclosing_function(tree: ast.Module, target: ast.Call) -> str | None:
    """Return the name of the function lexically enclosing ``target``."""
    best: str | None = None
    best_lineno = -1
    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            if node.lineno <= target.lineno <= (node.end_lineno or node.lineno):
                if node.lineno > best_lineno:
                    best, best_lineno = node.name, node.lineno
    return best


def _channel_calls(tree: ast.Module) -> list[tuple[ast.Call, str, str | None]]:
    """Every ``self.<filter_aware_method>(...)`` call with its enclosing func."""
    out: list[tuple[ast.Call, str, str | None]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            method = node.func.attr
            if method in _FILTER_AWARE_METHODS:
                out.append((node, method, _enclosing_function(tree, node)))
    return out


def test_param_set_is_current() -> None:
    """The declared filter-aware method set matches the source signatures.

    Keeps ``_FILTER_AWARE_METHODS`` honest: if someone adds a method with a
    ``filter_ast`` parameter (a new channel) and forgets to register it here,
    this fails — so the channel-call audit below can't silently miss it.
    """
    tree = _retriever_tree()
    declared_in_source: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            params = [a.arg for a in node.args.args] + [a.arg for a in node.args.kwonlyargs]
            if "filter_ast" in params:
                declared_in_source.add(node.name)
    assert declared_in_source == set(_FILTER_AWARE_METHODS), (
        "filter-aware method set drifted from source.\n"
        f"  in source but not declared here: {declared_in_source - set(_FILTER_AWARE_METHODS)}\n"
        f"  declared here but not in source: {set(_FILTER_AWARE_METHODS) - declared_in_source}\n"
        "Update _FILTER_AWARE_METHODS (a new channel must be threaded + audited)."
    )


def test_every_channel_call_threads_filter() -> None:
    """No channel call omits ``filter_ast`` except the justified allowlist.

    This is the gate: if a channel is invoked without the filter, a
    filter-violating chunk could reach fusion. Drift in EITHER direction fails —
    a new unguarded call (must be threaded or justified) OR a stale allowlist
    entry whose call now threads the filter (the omission was fixed; remove it).
    """
    tree = _retriever_tree()
    observed_omissions: dict[tuple[str, str | None], int] = {}
    for call, method, enclosing in _channel_calls(tree):
        threads = any(kw.arg == "filter_ast" for kw in call.keywords)
        if not threads:
            observed_omissions[(method, enclosing)] = call.lineno

    observed_keys = set(observed_omissions)
    allowlisted_keys = set(_ALLOWLISTED_OMISSIONS)

    new_unguarded = observed_keys - allowlisted_keys
    assert not new_unguarded, (
        "NEW channel call(s) omit filter_ast — a filter-violating chunk could reach RRF.\n"
        + "\n".join(
            f"  {_RETRIEVER.name}:{observed_omissions[k]}  self.{k[0]}(...) inside {k[1]}()"
            for k in sorted(new_unguarded)
        )
        + "\nThread filter_ast into the call, or add it to _ALLOWLISTED_OMISSIONS with a reason."
    )

    stale_allowlist = allowlisted_keys - observed_keys
    assert not stale_allowlist, (
        "Stale allowlist entries — these calls now thread filter_ast (gap fixed?).\n"
        + "\n".join(f"  self.{k[0]}(...) inside {k[1]}()" for k in sorted(stale_allowlist))
        + "\nRemove them from _ALLOWLISTED_OMISSIONS."
    )


def test_allowlist_entries_have_reasons() -> None:
    """Every allowlisted omission carries a non-empty justification."""
    for key, reason in _ALLOWLISTED_OMISSIONS.items():
        assert reason and reason.strip(), f"allowlist entry {key} has no justification"


def test_enforcement_pending_boundaries_have_reasons() -> None:
    """Every wiring-done/enforcement-pending boundary carries a reason."""
    for key, reason in _ENFORCEMENT_PENDING_BOUNDARIES.items():
        assert reason and reason.strip(), f"enforcement-pending boundary {key} has no reason"


def test_embedded_search_fulltext_now_enforces_filter_ast() -> None:
    """Pins embedded BM25 filter enforcement to source.

    ``SQLiteLanceTemporalStore.search_fulltext`` declares ``filter_ast`` AND
    forwards it into the inner ``_bm25_search`` (which compiles + post-filters
    it). If someone regresses the forward, this fails — the embedded BM25
    channel would silently drop the caller filter again. Keyed off the source so
    the doc note can't silently drift from reality.
    """
    backend = (
        Path(__file__).resolve().parents[2] / "src" / "khora" / "engines" / "skeleton" / "backends" / "sqlite_lance.py"
    )
    tree = ast.parse(backend.read_text())
    fn = next(
        (
            n
            for n in ast.walk(tree)
            if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef)) and n.name == "search_fulltext"
        ),
        None,
    )
    assert fn is not None, "search_fulltext not found in sqlite_lance temporal store"
    # It declares filter_ast (wiring present) ...
    params = [a.arg for a in fn.args.args] + [a.arg for a in fn.args.kwonlyargs]
    assert "filter_ast" in params, "search_fulltext no longer declares filter_ast — wiring changed"
    # ... AND its body forwards filter_ast into the inner _bm25_search (that's
    # how it enforces). If this stops, the embedded BM25 filter dropped.
    passes_to_bm25 = any(
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "_bm25_search"
        and any(kw.arg == "filter_ast" for kw in call.keywords)
        for call in ast.walk(fn)
    )
    assert passes_to_bm25, (
        "Embedded search_fulltext no longer forwards filter_ast to _bm25_search — the "
        "BM25 channel would drop the caller filter. Restore the `filter_ast=filter_ast` "
        "forward so the embedded BM25 channel enforces the recall filter."
    )
