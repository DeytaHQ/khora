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

Beyond the per-channel plans, the builder honours an optional *surface-coverage*
signal: an engine may report the row counts of each result surface (``chunks`` /
``entities`` / ``relationships``) and the set of surfaces the filter's channels
actually gate. When a surface came back non-empty but is NOT in the covered set,
the filter did not constrain the rows on that surface, so every leaf of the
filter is *unenforced* for that surface — the builder forces those leaves into
``unenforced_keys`` (out of ``pushed_keys`` / ``post_filtered_keys``) and clears
``pushed_down``. This keeps the report honest for multi-surface engines that push
a filter through the chunk channel but emit entities / relationships the filter
never touched. The signal is opt-in: with ``surface_sizes=None`` the builder is
byte-for-byte its legacy self.

The module is **pure**: it imports only the filter AST + the canonical leaf
enumerator (:func:`khora.filter.execute.filter_leaf_keys`) and never an engine
or a backend. Engines construct the :class:`ChannelPlan` carriers from their own
compile results and call :func:`build_filter_report`.
"""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Set as AbstractSet
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
    ``pushed_keys`` / ``post_filtered_keys`` / ``unenforced_keys`` lists form a
    TOTAL partition of the filter's constraint leaves: every leaf lands in exactly
    one of the three. A leaf is in ``pushed_keys`` only when every channel that
    gates it pushed it into the backend query; in ``post_filtered_keys`` when at
    least one gating channel had to re-check it in memory; and in
    ``unenforced_keys`` when no channel gates it at all (the filter constrains it
    but nothing — pushdown nor in-memory re-check — actually enforces it), OR when
    the recall returned a non-empty result surface (``entities`` /
    ``relationships``) that the filter's channels do not cover, so the filter's
    leaves went unenforced against that surface's rows. On a correct recall every
    leaf is enforced, so ``unenforced_keys == []``. A single-channel engine like
    skeleton (whose one channel gates every leaf) always reports
    ``unenforced_keys == []``; the list is defined for multi-channel /
    multi-surface engines where a leaf may slip past every channel or go
    unenforced on an uncovered result surface. All three lists are sorted and
    JSON-stable.
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

    unenforced_keys: list[str] = Field(default_factory=list)
    """Constraint-leaf keys the filter did not enforce (sorted). A leaf lands here
    when NO channel gates it — neither pushed down nor re-checked in memory — or
    when the recall returned a non-empty result surface not covered by the
    filter's channels (an uncovered ``entities`` / ``relationships`` surface makes
    every filter leaf unenforced against that surface's rows). Together with
    ``pushed_keys`` and ``post_filtered_keys`` this forms a total, disjoint
    partition of every constraint leaf. A correct recall enforces every leaf, so
    this list is empty (``unenforced_keys == []``)."""

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
    *,
    surface_sizes: Mapping[str, int] | None = None,
    covered_surfaces: AbstractSet[str] = frozenset({"chunks"}),
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
    * Top-level ``unenforced_keys`` — leaves that no channel gates at all, plus
      any leaf forced unenforced by the surface-coverage rule below.

    These three lists form a TOTAL, disjoint partition of the constraint leaves:
    every leaf lands in exactly one. A leaf that no channel gates lands in
    ``unenforced_keys`` (nothing enforces it; its only other signal would be
    ``defensive_recheck``). ``pushed_down`` is ``True`` only when ``post_filtered_keys`` is empty and
    ``pushed_keys`` covers all constraint leaves (so every leaf was gated and
    pushed). ``post_filtered`` is ``True`` when any leaf was post-filtered OR any
    channel ran a defensive full-predicate re-check (which does NOT demote a
    fully-pushed leaf — NO-DEMOTE).

    **Surface coverage** (opt-in via ``surface_sizes``): an engine may emit rows
    on more than one result surface (``chunks`` / ``entities`` /
    ``relationships``) while its filter channels only gate some of them. Pass
    ``surface_sizes`` mapping each surface to its returned row count and
    ``covered_surfaces`` naming the surfaces the filter actually constrains
    (default ``{"chunks"}``). When any of the three known surfaces came back
    non-empty (``surface_sizes.get(s, 0) > 0``) but is not in ``covered_surfaces``,
    the filter went unenforced against that surface's rows, so ALL of the filter's
    leaves are forced into ``unenforced_keys`` (removed from ``pushed_keys`` /
    ``post_filtered_keys``) and ``pushed_down`` becomes ``False``. With
    ``surface_sizes=None`` this rule is inert and the builder is byte-for-byte its
    legacy self. The rule only applies on the non-empty-leaves path — an empty /
    constraint-free filter has no leaves to force.

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

    # Surface coverage (opt-in): a non-empty result surface the filter's channels
    # do not cover means the filter went unenforced against that surface's rows,
    # so ALL filter leaves are forced unenforced. Inert when surface_sizes is None.
    surface_unenforced: frozenset[str] = frozenset()
    if surface_sizes is not None and any(
        surface_sizes.get(s, 0) > 0 and s not in covered_surfaces for s in ("chunks", "entities", "relationships")
    ):
        surface_unenforced = frozenset(all_leaves)

    # Keep the three top-level lists a total, disjoint partition: forced-unenforced
    # leaves leave both pushed and post_filtered.
    pushed -= surface_unenforced
    post_filtered -= surface_unenforced

    return FilterPushdownReport(
        # Fully pushed only when nothing was post-filtered AND every constraint
        # leaf landed in pushed_keys (so all leaves were gated + pushed). Any
        # surface-forced unenforced leaf drops it out of pushed, so this is False.
        pushed_down=not post_filtered and pushed == set(all_leaves),
        post_filtered=bool(post_filtered) or any_defensive,
        pushed_keys=sorted(pushed),
        post_filtered_keys=sorted(post_filtered),
        unenforced_keys=sorted((all_leaves - pushed - post_filtered) | surface_unenforced),
        channels=channels,
    )
