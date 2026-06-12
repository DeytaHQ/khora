"""Honest filter-pushdown reporting for ``RecallResult.engine_info["filter"]``.

An engine that runs a structured recall filter across one or more retrieval
channels needs to tell the caller, truthfully, *what happened to the filter*:
which constraint leaves were pushed into the backend query (so no Python
post-filter was needed for them) and which were re-checked in memory. Before
#1069 the skeleton engine derived ``pushed_down`` from a hardcoded backend
check (``self._backend_type == "pgvector"``) rather than from what the compiler
actually consumed, so it under-reported on every non-pgvector backend whose
compiler did push the predicate down.

This module is the single, backend-agnostic place that turns per-channel
pushdown facts into the public :class:`FilterPushdownReport`. Each channel hands
in a :class:`ChannelPlan` describing what its compiler consumed
(``pushed_keys``), what it had to re-check in memory (``post_filtered_keys``),
and whether it ran a defensive in-memory re-check over the full predicate even
though every leaf compiled down (``defensive_recheck``). :func:`build_filter_report`
folds those plans into the top-level partition and the per-channel breakdown.

The module is **pure**: it imports only the filter AST + the canonical leaf
enumerator (:func:`khora.filter.execute.filter_leaf_keys`) and never an engine
or a backend. Engines construct the :class:`ChannelPlan` carriers from their own
compile results and call :func:`build_filter_report`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field

from khora.filter.ast import FilterNode
from khora.filter.execute import filter_leaf_keys

__all__ = [
    "ChannelPlan",
    "FilterChannelReport",
    "FilterPushdownReport",
    "build_filter_report",
]


class FilterChannelReport(BaseModel):
    """Per-channel disposition of a recall filter's constraint leaves.

    One entry per retrieval channel that saw the filter. The two lists name the
    dotted constraint-leaf keys (``".".join(clause.path)``) that this channel's
    compiler pushed into its backend query versus the ones it re-checked in
    memory. Both lists are sorted for deterministic, JSON-stable output.
    """

    model_config = ConfigDict(frozen=True)

    pushed_keys: list[str] = Field(default_factory=list)
    """Constraint-leaf keys this channel pushed into its backend query (sorted)."""

    post_filtered_keys: list[str] = Field(default_factory=list)
    """Constraint-leaf keys this channel re-checked with an in-memory predicate (sorted)."""


class FilterPushdownReport(BaseModel):
    """Honest, backend-agnostic summary of how a recall filter was handled.

    Surfaced verbatim as ``RecallResult.engine_info["filter"]``. The top-level
    ``pushed_keys`` / ``post_filtered_keys`` lists *partition* the filter's
    constraint leaves: a leaf is in ``pushed_keys`` only when every channel that
    gates it pushed it into the backend query, and in ``post_filtered_keys`` when
    at least one gating channel had to re-check it in memory. Both lists are
    sorted and JSON-stable.
    """

    model_config = ConfigDict(frozen=True)

    pushed_down: bool = False
    """``True`` only when the filter is fully pushed: ``post_filtered_keys`` is
    empty AND ``pushed_keys`` covers every constraint leaf. ``False`` for a
    constraint-free / no-filter recall (nothing was narrowed)."""

    post_filtered: bool = False
    """``True`` when any constraint leaf was re-checked in memory on a gating
    channel, OR when a channel ran a defensive full-predicate re-check
    (``defensive_recheck``) even though every leaf compiled down."""

    pushed_keys: list[str] = Field(default_factory=list)
    """Constraint-leaf keys pushed into the backend query on every gating channel (sorted)."""

    post_filtered_keys: list[str] = Field(default_factory=list)
    """Constraint-leaf keys re-checked in memory on at least one gating channel (sorted).

    Note the NO-DEMOTE rule: a defensive full-predicate re-check sets
    ``post_filtered=True`` but does NOT move a fully-pushed leaf here."""

    channels: dict[str, FilterChannelReport] = Field(default_factory=dict)
    """Per-channel breakdown, keyed by channel name. One entry per channel the
    engine fed in — the builder never injects or drops a channel, so a
    single-channel engine that always passes its backend channel (even on a
    no-filter recall, with empty key lists) carries one entry here."""


@dataclass(frozen=True, slots=True)
class ChannelPlan:
    """A single channel's pushdown facts, fed into :func:`build_filter_report`.

    ``@internal`` carrier — engines build one per retrieval channel from their
    own compile results; it is not part of the public ``engine_info`` payload.

    * ``pushed_keys`` — dotted constraint-leaf keys this channel's compiler
      consumed into its backend query.
    * ``post_filtered_keys`` — dotted constraint-leaf keys this channel could not
      push and re-checks with an in-memory predicate.
    * ``defensive_recheck`` — ``True`` when the channel runs an in-memory re-check
      over the FULL predicate as a safety net even though every leaf compiled
      down. Sets the top-level ``post_filtered`` flag without demoting any
      fully-pushed leaf (NO-DEMOTE).
    """

    pushed_keys: frozenset[str] = field(default_factory=frozenset)
    post_filtered_keys: frozenset[str] = field(default_factory=frozenset)
    defensive_recheck: bool = False


def build_filter_report(
    filter_ast: FilterNode | None,
    channel_inputs: Mapping[str, ChannelPlan],
) -> FilterPushdownReport:
    """Fold per-channel :class:`ChannelPlan` facts into a :class:`FilterPushdownReport`.

    Pure: enumerates the filter's constraint leaves with the canonical
    :func:`khora.filter.execute.filter_leaf_keys` walk (the same key set the
    compilers build ``consumed_keys`` from) and partitions them.

    A leaf is *gated* by a channel when that channel saw it — i.e. the leaf is in
    the channel's ``pushed_keys`` or ``post_filtered_keys``. Disposition:

    * Top-level ``pushed_keys`` — leaves pushed on *every* gating channel.
    * Top-level ``post_filtered_keys`` — leaves re-checked in memory on *any*
      gating channel.

    These two lists partition the constraint leaves. ``pushed_down`` is ``True``
    only when ``post_filtered_keys`` is empty and ``pushed_keys`` covers all
    constraint leaves. ``post_filtered`` is ``True`` when any leaf was
    post-filtered OR any channel ran a defensive full-predicate re-check (which
    does NOT demote a fully-pushed leaf — NO-DEMOTE).

    A constraint-free filter (``None``, or a root with no children) carries no
    leaves: ``pushed_keys`` / ``post_filtered_keys`` are empty and ``pushed_down``
    is ``False`` (nothing was narrowed); ``post_filtered`` is still ``True`` if a
    channel set ``defensive_recheck``. The ``channels`` map is built one-for-one
    from ``channel_inputs``: the builder never injects or drops a channel, so an
    engine that always passes its single backend channel (even with an empty
    :class:`ChannelPlan`) gets a one-entry ``channels`` map on every recall.
    """
    # One FilterChannelReport per input, always — the builder preserves exactly
    # the channels the engine handed in (no injection, no dropping), so the
    # carrier is faithful even on the no-filter path (an empty ChannelPlan).
    channels = {
        name: FilterChannelReport(
            pushed_keys=sorted(plan.pushed_keys),
            post_filtered_keys=sorted(plan.post_filtered_keys),
        )
        for name, plan in channel_inputs.items()
    }
    any_defensive = any(plan.defensive_recheck for plan in channel_inputs.values())

    # A None root or an empty-AND root (filter={} / RecallFilter()) carries no
    # constraint leaves — nothing was narrowed. A defensive re-check still flips
    # post_filtered; pushed_down stays False (no leaves to push).
    all_leaves = frozenset() if filter_ast is None or not filter_ast.children else filter_leaf_keys(filter_ast)
    if not all_leaves:
        return FilterPushdownReport(post_filtered=any_defensive, channels=channels)

    # Partition the GATED leaves: a leaf goes to pushed_keys iff at least one
    # channel gates it AND every gating channel pushed it; it goes to
    # post_filtered_keys iff any gating channel re-checked it in memory. A leaf no
    # channel gates lands in NEITHER list (defensive_recheck is its only signal).
    pushed: set[str] = set()
    post_filtered: set[str] = set()
    for leaf in all_leaves:
        gating = [plan for plan in channel_inputs.values() if leaf in (plan.pushed_keys | plan.post_filtered_keys)]
        if any(leaf in plan.post_filtered_keys for plan in gating):
            post_filtered.add(leaf)
        elif gating:  # all gating channels pushed it (NO-DEMOTE: stays pushed)
            pushed.add(leaf)

    return FilterPushdownReport(
        # Fully pushed only when nothing was post-filtered AND every constraint
        # leaf landed in pushed_keys (so all leaves were gated + pushed).
        pushed_down=not post_filtered and pushed == set(all_leaves),
        post_filtered=bool(post_filtered) or any_defensive,
        pushed_keys=sorted(pushed),
        post_filtered_keys=sorted(post_filtered),
        channels=channels,
    )
