"""Builder unit tests for the canonical :class:`FilterPushdownReport` (#1069).

``build_filter_report`` is the single backend-agnostic place that folds
per-channel :class:`ChannelPlan` pushdown facts into the public
:class:`FilterPushdownReport` surfaced as ``RecallResult.engine_info["filter"]``.
These are PURE tests of the fold (no engine, no backend, no Postgres): they feed
hand-built ASTs + ``ChannelPlan`` carriers and pin the locked contract:

1. The constraint-free / no-filter carrier is ``pushed_down=False``,
   ``post_filtered=False``, empty key lists, and ONE named empty channel entry
   (the engine always passes a one-entry mapping) — never ``channels={}``.
2. A *gating* channel is one where the leaf appears in ``pushed_keys`` ∪
   ``post_filtered_keys``. A leaf no channel gates lands in NEITHER top-level
   list.
3. NO-DEMOTE: ``defensive_recheck=True`` flips top-level ``post_filtered`` but
   never moves a pushed leaf into ``post_filtered_keys``.
4. ``pushed_down`` is ``True`` iff ``post_filtered_keys == []`` AND
   ``set(pushed_keys) == {all constraint leaves}``.

The model is also pinned for frozen-immutability, deterministic sorting,
dict→model round-trip, and a ``model_json_schema()`` snapshot that catches a
field rename / type change.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from khora.filter import FilterPushdownReport
from khora.filter.ast import FilterClause, FilterNode, FilterOp, parse_to_ast
from khora.filter.execute import filter_leaf_keys
from khora.filter.model import RecallFilter
from khora.filter.report import (
    ChannelPlan,
    FilterChannelReport,
    build_filter_report,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _ast(doc: dict[str, object]) -> FilterNode:
    """Lower a wire filter document into its canonical AST.

    Round-trips through the public validator + ``parse_to_ast`` so the leaf-key
    set the builder enumerates is exactly the production one (the same walk the
    compilers build ``consumed_keys`` from).
    """
    return parse_to_ast(RecallFilter.model_validate(doc))


# A two-leaf filter used across the partition / intersection cases. Its leaves
# are exactly ``{"metadata.tier", "source_name"}``.
_TWO_LEAF_DOC: dict[str, object] = {"source_name": "linear", "metadata.tier": "gold"}


def _two_leaf_keys() -> frozenset[str]:
    return filter_leaf_keys(_ast(_TWO_LEAF_DOC))


# --------------------------------------------------------------------------- #
# Empty / no-filter / constraint-free carriers (rules in docs/api-reference.md
# under engine_info["filter"]).
# --------------------------------------------------------------------------- #


def test_no_filter_carrier_is_canonical_empty_with_one_named_channel() -> None:
    """``filter_ast=None`` → all-False report with ONE named empty channel.

    Per docs/api-reference.md, the canonical no-filter carrier is ``pushed_down=False``,
    ``post_filtered=False``, empty key lists, and ``channels`` carrying the
    single named channel the engine fed in (an empty :class:`ChannelPlan`) — NOT
    ``channels={}``. The builder never injects or drops a channel.
    """
    report = build_filter_report(None, {"sqlite_lance": ChannelPlan()})

    assert report.pushed_down is False
    assert report.post_filtered is False
    assert report.pushed_keys == []
    assert report.post_filtered_keys == []
    # ONE named empty entry — not an empty dict.
    assert report.channels == {"sqlite_lance": FilterChannelReport()}
    assert list(report.channels) == ["sqlite_lance"]


def test_empty_and_root_matches_no_filter_carrier() -> None:
    """A bare ``filter={}`` (empty-AND root) folds to the same empty carrier.

    ``RecallFilter()`` / ``filter={}`` normalize to ``FilterNode(op=AND,
    children=())`` — a match-everything root carrying zero leaves. It is
    constraint-free, so it reports identically to the ``None`` carrier (decision
    1): nothing narrowed, ``pushed_down=False``.
    """
    empty_ast = _ast({})
    assert empty_ast.op == FilterOp.AND and not empty_ast.children

    report = build_filter_report(empty_ast, {"sqlite_lance": ChannelPlan()})

    assert report.model_dump(mode="json") == build_filter_report(None, {"sqlite_lance": ChannelPlan()}).model_dump(
        mode="json"
    )


def test_no_filter_carrier_defensive_recheck_flips_post_filtered_only() -> None:
    """A defensive re-check on a no-filter recall flips ``post_filtered`` alone.

    With no constraint leaves there is nothing to push, so ``pushed_down`` stays
    ``False`` and both key lists stay empty — but a channel that ran a defensive
    full-predicate re-check still sets top-level ``post_filtered=True`` (decision
    3, the no-leaf edge).
    """
    report = build_filter_report(None, {"sqlite_lance": ChannelPlan(defensive_recheck=True)})

    assert report.pushed_down is False
    assert report.post_filtered is True
    assert report.pushed_keys == []
    assert report.post_filtered_keys == []
    assert report.channels == {"sqlite_lance": FilterChannelReport()}


def test_builder_preserves_every_named_channel_with_no_filter() -> None:
    """Multi-channel no-filter recall keeps EVERY named channel, each empty.

    The builder is one-for-one with ``channel_inputs`` even on the no-filter
    path: a three-channel engine that fed three empty plans gets three empty
    channel entries back (no injection, no dropping).
    """
    report = build_filter_report(
        None,
        {"vector": ChannelPlan(), "bm25": ChannelPlan(), "graph": ChannelPlan()},
    )

    assert set(report.channels) == {"vector", "bm25", "graph"}
    assert all(ch == FilterChannelReport() for ch in report.channels.values())


# --------------------------------------------------------------------------- #
# pushed_down derivation (docs/api-reference.md under engine_info["filter"]).
# --------------------------------------------------------------------------- #


def test_pushed_down_true_when_all_leaves_pushed_single_channel() -> None:
    """Every leaf pushed on the sole channel, nothing post-filtered → True.

    Per docs/api-reference.md, ``pushed_down`` is ``True`` iff ``post_filtered_keys`` is empty
    AND ``pushed_keys`` covers every constraint leaf. A single-channel skeleton
    gates every leaf, so when its plan pushes them all the report is fully
    pushed.
    """
    leaves = _two_leaf_keys()
    report = build_filter_report(_ast(_TWO_LEAF_DOC), {"sqlite_lance": ChannelPlan(pushed_keys=leaves)})

    assert report.pushed_down is True
    assert report.post_filtered is False
    assert set(report.pushed_keys) == set(leaves)
    assert report.post_filtered_keys == []
    # Fully enforced: every leaf pushed, nothing slipped past the channel.
    assert report.unenforced_keys == []


def test_pushed_down_false_when_one_leaf_post_filtered() -> None:
    """One leaf pushed, the other re-checked in memory → not fully pushed.

    The JSON1-absent split: ``source_name`` pushes, ``metadata.tier`` defers to
    the in-memory post-filter. ``post_filtered_keys`` is non-empty so
    ``pushed_down`` is ``False``, and the two leaves partition into
    the two top-level lists.
    """
    report = build_filter_report(
        _ast(_TWO_LEAF_DOC),
        {
            "sqlite_lance": ChannelPlan(
                pushed_keys=frozenset({"source_name"}),
                post_filtered_keys=frozenset({"metadata.tier"}),
            )
        },
    )

    assert report.pushed_down is False
    assert report.post_filtered is True
    assert report.pushed_keys == ["source_name"]
    assert report.post_filtered_keys == ["metadata.tier"]


def test_pushed_down_false_when_all_leaves_post_filtered() -> None:
    """ALL leaves post-filtered → ``pushed_down=False`` (not vacuously True).

    The all-post-filtered case is the inverse of the fully-pushed one: every
    constraint leaf was re-checked in memory, so ``post_filtered_keys`` covers
    them all and ``pushed_keys`` is empty. ``pushed_down`` must be ``False`` —
    the report never claims a pushdown when nothing pushed.
    """
    leaves = _two_leaf_keys()
    report = build_filter_report(_ast(_TWO_LEAF_DOC), {"sqlite_lance": ChannelPlan(post_filtered_keys=leaves)})

    assert report.pushed_down is False
    assert report.post_filtered is True
    assert report.pushed_keys == []
    assert set(report.post_filtered_keys) == set(leaves)


def test_pushed_down_false_when_a_leaf_is_gated_by_no_channel() -> None:
    """A constraint leaf no channel gated → it lands in ``unenforced_keys`` → False.

    Per docs/api-reference.md, ``metadata.tier`` appears in no channel's ``pushed_keys`` ∪
    ``post_filtered_keys``, so it lands in ``unenforced_keys`` (nothing enforces
    it) — NOT silently dropped. The pushed set is then a strict subset of all
    leaves, so ``pushed_down`` is ``False`` — the builder does not silently treat
    an unseen leaf as pushed.
    """
    report = build_filter_report(
        _ast(_TWO_LEAF_DOC),
        {"sqlite_lance": ChannelPlan(pushed_keys=frozenset({"source_name"}))},
    )

    assert report.pushed_down is False
    # source_name pushed; metadata.tier ungated -> unenforced_keys.
    assert report.pushed_keys == ["source_name"]
    assert report.post_filtered_keys == []
    assert report.unenforced_keys == ["metadata.tier"]
    # post_filtered is False: no leaf was post-filtered and no defensive re-check.
    assert report.post_filtered is False


# --------------------------------------------------------------------------- #
# NO-DEMOTE (docs/api-reference.md under engine_info["filter"]).
# --------------------------------------------------------------------------- #


def test_defensive_recheck_sets_post_filtered_but_does_not_demote() -> None:
    """``defensive_recheck=True`` flips ``post_filtered`` but keeps pushed leaves.

    NO-DEMOTE: the sqlite_lance backend always runs a
    compile_python post-filter over the full AST as a safety net even when every
    leaf compiled into the WHERE. That sets top-level ``post_filtered=True``, but
    a fully-pushed leaf stays in ``pushed_keys`` — it is NOT moved into
    ``post_filtered_keys``. ``pushed_down`` therefore stays ``True``.
    """
    leaves = _two_leaf_keys()
    report = build_filter_report(
        _ast(_TWO_LEAF_DOC),
        {"sqlite_lance": ChannelPlan(pushed_keys=leaves, defensive_recheck=True)},
    )

    assert report.post_filtered is True  # defensive re-check flipped it
    assert report.post_filtered_keys == []  # NO-DEMOTE: no leaf moved
    assert set(report.pushed_keys) == set(leaves)  # leaves stay pushed
    assert report.pushed_down is True  # still fully pushed


# --------------------------------------------------------------------------- #
# Multi-channel intersection / partition semantics (docs/api-reference.md).
# --------------------------------------------------------------------------- #


def test_multichannel_leaf_pushed_in_one_post_filtered_in_another() -> None:
    """Adversarial split: a leaf pushed on channel A, post-filtered on B → post.

    Per docs/api-reference.md, a leaf re-checked in memory on ANY gating channel goes to the
    top-level ``post_filtered_keys`` (the honest worst case), even though another
    channel pushed it cleanly. The per-channel breakdown still records each
    channel's own disposition faithfully.
    """
    report = build_filter_report(
        _ast({"source_name": "linear"}),
        {
            "chan_a": ChannelPlan(pushed_keys=frozenset({"source_name"})),
            "chan_b": ChannelPlan(post_filtered_keys=frozenset({"source_name"})),
        },
    )

    assert report.pushed_keys == []
    assert report.post_filtered_keys == ["source_name"]
    assert report.pushed_down is False
    assert report.post_filtered is True
    # Per-channel breakdown is faithful to each channel's own facts.
    assert report.channels["chan_a"] == FilterChannelReport(pushed_keys=["source_name"])
    assert report.channels["chan_b"] == FilterChannelReport(post_filtered_keys=["source_name"])


def test_multichannel_leaf_pushed_on_every_gating_channel() -> None:
    """A leaf pushed on every channel that gates it → top-level pushed.

    The mirror of the adversarial case: when both gating channels pushed the
    leaf (neither re-checked it), it lands in the top-level ``pushed_keys`` and
    nothing is post-filtered, so a single-leaf filter is fully pushed.
    """
    report = build_filter_report(
        _ast({"source_name": "linear"}),
        {
            "chan_a": ChannelPlan(pushed_keys=frozenset({"source_name"})),
            "chan_b": ChannelPlan(pushed_keys=frozenset({"source_name"})),
        },
    )

    assert report.pushed_keys == ["source_name"]
    assert report.post_filtered_keys == []
    assert report.pushed_down is True
    assert report.post_filtered is False


def test_multichannel_partition_property_when_all_channels_gate_every_leaf() -> None:
    """Each constraint leaf lands in EXACTLY one top-level list (partition).

    When every channel gates every leaf, the two top-level lists partition the
    constraint leaves: their union is the full leaf set and their intersection
    is empty. Here channel A pushes ``source_name`` & post-filters
    ``metadata.tier``; channel B does the reverse — so each leaf is post-filtered
    on at least one gating channel and both end up in ``post_filtered_keys``,
    with nothing double-counted.
    """
    leaves = _two_leaf_keys()
    report = build_filter_report(
        _ast(_TWO_LEAF_DOC),
        {
            "chan_a": ChannelPlan(
                pushed_keys=frozenset({"source_name"}),
                post_filtered_keys=frozenset({"metadata.tier"}),
            ),
            "chan_b": ChannelPlan(
                pushed_keys=frozenset({"metadata.tier"}),
                post_filtered_keys=frozenset({"source_name"}),
            ),
        },
    )

    union = set(report.pushed_keys) | set(report.post_filtered_keys)
    intersection = set(report.pushed_keys) & set(report.post_filtered_keys)
    assert union == set(leaves), "the two top-level lists must cover every gated constraint leaf"
    assert intersection == set(), "a leaf must not appear in both top-level lists"
    # Both leaves post-filtered on at least one gating channel.
    assert report.pushed_keys == []
    assert set(report.post_filtered_keys) == set(leaves)


# --------------------------------------------------------------------------- #
# Determinism — sorted, JSON-stable key lists.
# --------------------------------------------------------------------------- #


def test_key_lists_are_sorted_regardless_of_input_order() -> None:
    """Top-level and per-channel key lists are sorted (deterministic JSON).

    Frozensets have no order, so the builder must sort every emitted list for a
    JSON-stable ``engine_info["filter"]``. We feed leaves whose natural set
    iteration would not be sorted and assert the output is.
    """
    doc = {"source_name": "linear", "source_type": "ticket", "title": "x", "content_type": "md"}
    ast = _ast(doc)
    leaves = filter_leaf_keys(ast)
    # A mixed split so both lists carry multiple keys.
    pushed = frozenset({"title", "content_type"})
    post = leaves - pushed
    report = build_filter_report(ast, {"sqlite_lance": ChannelPlan(pushed_keys=pushed, post_filtered_keys=post)})

    assert report.pushed_keys == sorted(report.pushed_keys)
    assert report.post_filtered_keys == sorted(report.post_filtered_keys)
    ch = report.channels["sqlite_lance"]
    assert ch.pushed_keys == sorted(ch.pushed_keys)
    assert ch.post_filtered_keys == sorted(ch.post_filtered_keys)


def test_report_is_deterministic_across_repeated_builds() -> None:
    """Two builds from identical inputs produce byte-identical JSON.

    The fold has no hidden order dependence: rebuilding from the same AST + plan
    yields the same ``model_dump(mode="json")`` every time.
    """
    leaves = _two_leaf_keys()
    inputs = {"sqlite_lance": ChannelPlan(pushed_keys=leaves, defensive_recheck=True)}
    a = build_filter_report(_ast(_TWO_LEAF_DOC), inputs).model_dump(mode="json")
    b = build_filter_report(_ast(_TWO_LEAF_DOC), inputs).model_dump(mode="json")
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# --------------------------------------------------------------------------- #
# Model contract — round-trip, immutability, schema snapshot.
# --------------------------------------------------------------------------- #


def test_dict_to_model_round_trip() -> None:
    """``FilterPushdownReport.model_validate(model.model_dump())`` round-trips.

    The report is surfaced as a JSON dict and re-hydrated by downstream callers
    via ``model_validate`` — the two must be equal, including the nested
    per-channel models (dicts coerce back to ``FilterChannelReport``).
    """
    report = build_filter_report(
        _ast(_TWO_LEAF_DOC),
        {
            "sqlite_lance": ChannelPlan(
                pushed_keys=frozenset({"source_name"}),
                post_filtered_keys=frozenset({"metadata.tier"}),
                defensive_recheck=True,
            )
        },
    )

    dumped = report.model_dump(mode="json")
    rehydrated = FilterPushdownReport.model_validate(dumped)
    assert rehydrated == report
    # The nested channel coerces from dict back to the typed model.
    assert isinstance(rehydrated.channels["sqlite_lance"], FilterChannelReport)


def test_report_is_frozen_immutable() -> None:
    """The report and its nested channel reports are frozen (pydantic v2).

    Both models declare ``model_config = ConfigDict(frozen=True)`` — mutating any
    field after construction raises ``ValidationError``. This keeps the surfaced
    ``engine_info["filter"]`` an immutable record.
    """
    report = build_filter_report(_ast(_TWO_LEAF_DOC), {"sqlite_lance": ChannelPlan()})

    with pytest.raises(ValidationError):
        report.pushed_down = True  # type: ignore[misc]
    with pytest.raises(ValidationError):
        report.pushed_keys = ["x"]  # type: ignore[misc]

    channel = FilterChannelReport(pushed_keys=["source_name"])
    with pytest.raises(ValidationError):
        channel.pushed_keys = ["other"]  # type: ignore[misc]


def test_model_json_schema_snapshot() -> None:
    """Pin the full JSON schema so a field rename / type change is caught.

    The report is part of the public ``engine_info["filter"]`` surface; a
    silent field rename or a type change (e.g. ``pushed_down`` from bool to str,
    or dropping the ``channels`` ``$ref``) would break downstream consumers
    without any other test failing. This snapshots the entire schema dict.
    """
    schema = FilterPushdownReport.model_json_schema()

    expected = {
        "$defs": {
            "FilterChannelReport": {
                "description": (
                    "Per-channel disposition of a recall filter's constraint leaves.\n\n"
                    "One entry per retrieval channel that saw the filter. The two lists name the\n"
                    'dotted constraint-leaf keys (``".".join(clause.path)``) that this channel\'s\n'
                    "compiler pushed into its backend query versus the ones it re-checked in\n"
                    "memory. Both lists are sorted for deterministic, JSON-stable output."
                ),
                "properties": {
                    "pushed_keys": {
                        "items": {"type": "string"},
                        "title": "Pushed Keys",
                        "type": "array",
                    },
                    "post_filtered_keys": {
                        "items": {"type": "string"},
                        "title": "Post Filtered Keys",
                        "type": "array",
                    },
                },
                "title": "FilterChannelReport",
                "type": "object",
            }
        },
        "description": (
            "Honest, backend-agnostic summary of how a recall filter was handled.\n\n"
            'Surfaced verbatim as ``RecallResult.engine_info["filter"]``. The top-level\n'
            "``pushed_keys`` / ``post_filtered_keys`` / ``unenforced_keys`` lists form a\n"
            "TOTAL partition of the filter's constraint leaves: every leaf lands in exactly\n"
            "one of the three. A leaf is in ``pushed_keys`` only when every channel that\n"
            "gates it pushed it into the backend query; in ``post_filtered_keys`` when at\n"
            "least one gating channel had to re-check it in memory; and in\n"
            "``unenforced_keys`` when no channel gates it at all (the filter constrains it\n"
            "but nothing — pushdown nor in-memory re-check — actually enforces it). On a\n"
            "correct recall every leaf is enforced, so ``unenforced_keys == []``. A\n"
            "single-channel engine like skeleton (whose one channel gates every leaf)\n"
            "always reports ``unenforced_keys == []``; the list is defined for\n"
            "multi-channel engines where a leaf may slip past every channel. All three\n"
            "lists are sorted and JSON-stable."
        ),
        "properties": {
            "pushed_down": {
                "default": False,
                "title": "Pushed Down",
                "type": "boolean",
            },
            "post_filtered": {
                "default": False,
                "title": "Post Filtered",
                "type": "boolean",
            },
            "pushed_keys": {
                "items": {"type": "string"},
                "title": "Pushed Keys",
                "type": "array",
            },
            "post_filtered_keys": {
                "items": {"type": "string"},
                "title": "Post Filtered Keys",
                "type": "array",
            },
            "unenforced_keys": {
                "items": {"type": "string"},
                "title": "Unenforced Keys",
                "type": "array",
            },
            "channels": {
                "additionalProperties": {"$ref": "#/$defs/FilterChannelReport"},
                "title": "Channels",
                "type": "object",
            },
        },
        "title": "FilterPushdownReport",
        "type": "object",
    }

    assert schema == expected


def test_model_json_schema_required_fields_and_types() -> None:
    """A tighter guard on the property names + scalar types (rename/type catch).

    Complements the full-schema snapshot with an explicit, readable assertion on
    the public field surface: the five canonical fields, their titles, and the
    two boolean / three array types. If the full snapshot above ever needs a
    docstring tweak, this narrower check still pins the load-bearing shape.
    """
    schema = FilterPushdownReport.model_json_schema()
    props = schema["properties"]

    assert set(props) == {
        "pushed_down",
        "post_filtered",
        "pushed_keys",
        "post_filtered_keys",
        "unenforced_keys",
        "channels",
    }
    assert props["pushed_down"]["type"] == "boolean"
    assert props["post_filtered"]["type"] == "boolean"
    assert props["pushed_keys"]["type"] == "array"
    assert props["pushed_keys"]["items"] == {"type": "string"}
    assert props["post_filtered_keys"]["type"] == "array"
    assert props["post_filtered_keys"]["items"] == {"type": "string"}
    assert props["unenforced_keys"]["type"] == "array"
    assert props["unenforced_keys"]["items"] == {"type": "string"}
    assert props["channels"]["type"] == "object"
    assert props["channels"]["additionalProperties"] == {"$ref": "#/$defs/FilterChannelReport"}


# --------------------------------------------------------------------------- #
# Direct-AST construction parity (no RecallFilter round-trip).
# --------------------------------------------------------------------------- #


def test_hand_built_ast_folds_identically_to_wire_round_trip() -> None:
    """A hand-built ``FilterNode`` folds the same as a wire-validated one.

    The builder enumerates leaves with ``filter_leaf_keys`` regardless of how the
    AST was constructed, so a directly-built ``AND([source_name $eq linear])``
    yields the same report as the ``RecallFilter`` round-trip. Guards that the
    fold depends only on the AST, not on the wire model.
    """
    hand = FilterNode(
        op=FilterOp.AND,
        children=(FilterClause(path=("source_name",), op=FilterOp.EQ, operand="linear"),),
    )
    wire = _ast({"source_name": "linear"})

    plan = {"sqlite_lance": ChannelPlan(pushed_keys=frozenset({"source_name"}))}
    assert build_filter_report(hand, plan).model_dump(mode="json") == build_filter_report(wire, plan).model_dump(
        mode="json"
    )
