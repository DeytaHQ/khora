r"""Every text-emitting compiler returns a **splice-safe** predicate fragment.

Splice safety is the contract both text compilers now name in their module
docstrings: ``A AND F`` is equivalent to ``A AND (F)``, i.e. the returned
``predicate`` carries no boolean operator at parenthesis depth 0. It matters
because callers splice the fragment into a conjunction **as text**, and ``AND``
binds tighter than ``OR`` — so an ungrouped top-level ``OR`` absorbs the caller's
namespace predicate into its left disjunct and the right disjunct reads every
tenant's rows. It fails as somebody else's data, not as an error.

**Why this module exists, and what it is NOT a duplicate of.** khora #1587 shipped
two store-side tripwires for the same hazard, and both are subject-limited:

* ``test_ungrouped_or_fragment_cannot_absorb_the_namespace_scope`` (in each
  store's scan module) monkeypatches ``CompilerRegistry`` to install a fake
  compiler that emits the bare ungrouped shape. It pins "IF an ungrouped fragment
  ever arrives, the STORE's parentheses contain it" — a real compiler that
  *started* emitting ungrouped output leaves it green, because the fake displaces
  the real one.
* the keyset-disjunction tripwire pins the store's own hand-written ``OR``, which
  is not compiler output at all.

Neither pins the realistic regression: a compiler emission change. That is this
module's subject, and it is asserted on the **real** ``compile_surrealdb`` /
``compile_lance`` over the real conformance corpus, with nothing patched.

**And the store-side parentheses are not a general backstop, which is why the
compiler-side invariant has to hold on its own.** Counted by hand over
``grep -rn "[.]predicate" src/khora`` in this tree, restricted to the two
text-emitting compilers: **8 splice sites, of which 5 do NOT re-parenthesize.**
Only three wrap it: ``SQLiteRelationalBackend.scan_documents``,
``SQLiteLanceRelationalAdapter.scan_documents`` (via ``_lance_fragment_to_text``)
and ``SurrealDBRelationalAdapter.scan_documents``. The five that splice bare are
``SQLiteLanceTemporalStore._vector_search`` / ``._bm25_search`` /
``.search_recent_chunks`` (``sql += f" AND {ast_sql}"`` onto a host ``WHERE``
carrying ``namespace_id = ?``) and ``SurrealDBTemporalStore._search_inner`` /
``.search_fulltext`` (appended into a clause list whose element 0 IS the namespace
predicate, joined on ``" AND "``). On those five, this module is the only thing
standing between an ungrouped fragment and a cross-namespace read.
``compile_postgres``' four sites are excluded for the reason below.

Symbols, not line numbers, and that is a considered choice rather than a style
preference. The first draft of this paragraph carried eight ``file:line``
references and **two of them were already wrong in the commit that wrote them**
(the two ``scan_documents`` splices had each drifted ~20 lines while this branch
was still being reviewed), while three more pointed at the ``_build_ast_clause``
call rather than the append they quoted. This ticket's own checklist asked to
replace a hardcoded ``filter/compilers/surrealdb.py:260`` reference with a symbol
reference for exactly that reason — "the number rots on the first edit above it" —
so a fresh set of line numbers here would have been the same defect, authored
knowingly. Re-derive the set with
``grep -rn "[.]predicate" src/khora`` rather than trusting any count in prose.

**Which compilers, and why not the third.** ``compile_surrealdb`` and
``compile_lance`` are the two that return a fragment as a *string*
(``compile_lance``'s output is spliced by ``SQLiteRelationalBackend`` and, via
``_lance_fragment_to_text``, by the ``sqlite_lance`` store; ``compile_surrealdb``'s
by ``SurrealDBRelationalAdapter`` and ``khora.storage.temporal.surrealdb``).
``compile_postgres`` is deliberately absent: it emits a SQLAlchemy
``ColumnElement`` that the ORM groups by its own precedence handling, so there is
no string to mis-splice and a test there would assert someone else's library.

**The checker is itself falsifiable, and that is not optional here.** A checker
that always returned "safe" would make every corpus parametrization below pass
while proving nothing — the same defect class the vacuous-parametrization finding
this change also fixes is about. So :func:`_depth0_boolean` is pinned in both
directions by
:func:`test_checker_flags_an_unsafe_fragment` and
:func:`test_checker_passes_a_safe_fragment` before it is trusted on any compiler
output, including the two forms that would make it fail *open*: an unbalanced
fragment, and a doubled ``''`` inside a single-quoted literal (a scanner that
mishandles the doubling mis-tracks depth for the whole remainder). Quote handling
is defensive rather than load-bearing — both compilers bind user values rather
than interpolating them — and the simple toggle-on-every-quote scan below handles
doubling correctly by construction.

Corpus counts, measured in this tree on this branch
(``uv run pytest tests/unit/filter/test_fragment_splice_safety.py -q``, and the
standalone sweep reported in the PR body): the corpus is **209 cases** across the
14 family generators. Per ``(compiler, mode)``, fragments compiled / cases skipped
because the case raised / fragments containing a nested (depth >= 1) boolean node:

| compiler | mode | compiled | skipped | nested |
| --- | --- | --- | --- | --- |
| ``compile_lance`` | ``raise`` | 144 | 65 | 76 |
| ``compile_lance`` | ``split`` | 209 | 0 | 79 |
| ``compile_surrealdb`` | ``raise`` | 192 | 17 | 81 |
| ``compile_surrealdb`` | ``split`` | 209 | 0 | 84 |

Sixteen of the seventeen ``surrealdb``/``raise`` skips are
``RecallFilterUnsupportedError``; the seventeenth is ``F-DOTKEY-dollar-key``
(``metadata.$ref``), the corpus's one ``CompileError`` — the identifier guard,
which under ``"raise"`` has no residual to defer the leaf to. Under ``"split"``
that same case now compiles (to the match-all placeholder) rather than raising,
per this change's §8 rework, which is why this column reads 209/0 where an
earlier revision
of this table measured 208/1. A case that raises produces no fragment, and a
fragment that is never produced cannot be mis-spliced, so skipping is correct —
but skipping *everything* would also be silently green, which is why the
anti-vacuity lower bounds below are asserted alongside.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable
from typing import Any

import pytest

from khora.filter import RecallFilterUnsupportedError
from khora.filter.compilers.lance import compile_lance
from khora.filter.compilers.surrealdb import compile_surrealdb
from khora.filter.conformance import (
    ConformanceCase,
    _resolve_ast,
    f_array_cases,
    f_coerce_cases,
    f_dates_cases,
    f_dotkey_cases,
    f_exists_cases,
    f_impossible_cases,
    f_logic_cases,
    f_nullval_cases,
    f_objeq_cases,
    f_op_cases,
    f_polarity_cases,
    f_sel_cases,
    f_sugar_cases,
    f_unsup_cases,
)
from khora.filter.context import CompileContext, CompileError
from khora.storage.backends.sqlite import _documents_compile_context as sqlite_context
from khora.storage.backends.surrealdb.relational import _documents_compile_context as surrealdb_context

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# The checker
# --------------------------------------------------------------------------- #

# ``AND`` / ``OR`` are matched on word boundaries, so an identifier or builtin
# that merely CONTAINS one cannot false-positive. That is not hypothetical: the
# ``author`` system key ends in ``or``, and SurrealQL's ``CONTAINSANY`` and
# SQLite's ``json_extract`` both appear in real fragments. ``NOT`` / ``!`` are
# deliberately NOT flagged — both bind tighter than ``AND``, so a leading
# negation on a group is splice-safe (``compile_surrealdb`` emits ``!(...)``,
# ``compile_lance`` emits ``(NOT (...))``).
_WORD_CHAR = re.compile(r"[A-Za-z0-9_]")
_BOOL_WORDS = ("AND", "OR")


def _scan(fragment: str) -> tuple[list[tuple[int, int, str]], str | None]:
    """Return ``(boolean operators, structural problem)`` for one fragment.

    Each operator is ``(depth, offset, text)``. Walks the string tracking
    parenthesis depth and skipping single-quoted spans; a doubled ``''`` inside a
    literal is an escaped quote, not a terminator.

    The structural problem is non-``None`` for an unbalanced fragment (either
    direction) or an unterminated quote. Both are splice hazards in their own
    right — a stray ``)`` closes the caller's group and a missing one swallows the
    caller's remaining conjuncts — and both are also how a naive checker fails
    **open**, by mis-tracking depth and reading a top-level ``OR`` as nested.
    """
    operators: list[tuple[int, int, str]] = []
    width = max(len(word) for word in _BOOL_WORDS)
    depth = 0
    i = 0
    n = len(fragment)
    while i < n:
        char = fragment[i]
        if char == "'":
            i += 1
            while i < n:
                if fragment[i] == "'":
                    if i + 1 < n and fragment[i + 1] == "'":
                        i += 2  # an escaped quote inside the literal
                        continue
                    break
                i += 1
            if i >= n:
                return operators, f"unterminated single-quoted literal in {fragment!r}"
            i += 1
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return operators, f"unbalanced ')' at offset {i} in {fragment!r}"
        elif fragment.startswith("&&", i) or fragment.startswith("||", i):
            operators.append((depth, i, fragment[i : i + 2]))
        else:
            upper = fragment[i : i + width].upper()
            for word in _BOOL_WORDS:
                if not upper.startswith(word):
                    continue
                before = fragment[i - 1] if i else " "
                after = fragment[i + len(word)] if i + len(word) < n else " "
                if not _WORD_CHAR.match(before) and not _WORD_CHAR.match(after):
                    operators.append((depth, i, fragment[i : i + len(word)]))
                    break
        i += 1
    if depth != 0:
        return operators, f"unbalanced fragment: ends at depth {depth} in {fragment!r}"
    return operators, None


def _depth0_boolean(fragment: str) -> str | None:
    """Describe the first splice hazard in ``fragment``, or ``None`` if safe."""
    operators, problem = _scan(fragment)
    if problem is not None:
        return problem
    for depth, offset, text in operators:
        if depth == 0:
            return f"depth-0 {text!r} at offset {offset} in {fragment!r}"
    return None


def _has_nested_boolean(fragment: str) -> bool:
    """Whether ``fragment`` contains a boolean operator at depth >= 1.

    The anti-vacuity companion to :func:`_depth0_boolean`: a corpus that stopped
    producing boolean nodes at all would satisfy splice safety trivially.
    """
    operators, _ = _scan(fragment)
    return any(depth >= 1 for depth, _offset, _text in operators)


# --------------------------------------------------------------------------- #
# The checker's own falsifiability
# --------------------------------------------------------------------------- #

# Fragments the checker MUST flag. The first is exactly the shape the store-side
# monkeypatched tripwire installs, so the two tests are demonstrably about the
# same hazard. The last two are the fail-open forms.
_UNSAFE_FRAGMENTS = [
    "title = $f_0 OR content = $f_1",
    "a = 1 AND b = 2",
    "a = 1 && b = 2",
    "a = 1 || b = 2",
    "(a = 1) OR (b = 2)",  # grouped operands, ungrouped join
    "(a=1) OR (b=2)",  # same, unspaced — the token scan must not depend on spacing
    "a = 'it''s' OR b = 2",  # a doubled quote must not hide the operator after it
    # The literal is ``a'`` (open, ``a``, escaped ``''``, close), so the ``OR``
    # that follows is real and at depth 0. A scanner that toggles on the escape
    # reads the rest of the string as quoted and lets this through — failing OPEN.
    "name = 'a''' OR b = 1",
    "a = 1)",
    "(a = 1",
]

# Fragments the checker MUST pass. ``author`` / ``CONTAINSANY`` / ``json_extract``
# are the word-boundary cases; ``'x OR y'`` is the quoted-span case; ``!(...)``
# and ``(NOT (...))`` are the two compilers' actual negation emissions.
_SAFE_FRAGMENTS = [
    "(a = 1 OR b = 2)",
    "a = 1",
    "!(a = 1 OR b = 2)",
    "(NOT (a = 1))",
    "NOT (a = 1)",
    "true",
    "1",
    "(a = 'it''s' OR b = 2)",
    "a = 'x OR y'",
    "name = 'it''s OR fine'",  # escape inside a literal; the OR never leaves it
    "(name = 'it''s OR fine' OR b = 1)",  # …and the real OR after it is grouped
    "author = $f_0",
    "x CONTAINSANY $f_0",
    "json_extract(metadata, '$.tier') = ?",
    "coalesce(json_extract(metadata, '$.x') = ?, 0)",
]


@pytest.mark.parametrize("fragment", _UNSAFE_FRAGMENTS)
def test_checker_flags_an_unsafe_fragment(fragment: str) -> None:
    """Without this, a checker stuck at "safe" makes every case below vacuous."""
    assert _depth0_boolean(fragment) is not None, f"checker missed an unsafe fragment: {fragment!r}"


@pytest.mark.parametrize("fragment", _SAFE_FRAGMENTS)
def test_checker_passes_a_safe_fragment(fragment: str) -> None:
    """And a checker that flags everything would fail the whole corpus for nothing."""
    assert _depth0_boolean(fragment) is None, f"checker false-positived on {fragment!r}"


def test_nested_boolean_detector_is_falsifiable() -> None:
    """The anti-vacuity detector must itself distinguish nested from flat."""
    assert _has_nested_boolean("(a = 1 OR b = 2)")
    assert _has_nested_boolean("!(a = 1 AND b = 2)")
    assert not _has_nested_boolean("a = 1")
    assert not _has_nested_boolean("(a = 1)")
    assert not _has_nested_boolean("(a = 'x OR y')")  # the operator is inside a literal


# --------------------------------------------------------------------------- #
# The corpus
# --------------------------------------------------------------------------- #

_FAMILY_GENERATORS = (
    f_op_cases,
    f_sugar_cases,
    f_impossible_cases,
    f_exists_cases,
    f_coerce_cases,
    f_polarity_cases,
    f_objeq_cases,
    f_dotkey_cases,
    f_array_cases,
    f_logic_cases,
    f_dates_cases,
    f_nullval_cases,
    f_sel_cases,
    f_unsup_cases,
)


def _all_cases() -> list[ConformanceCase]:
    """The whole hand-authored conformance corpus, as ``test_conformance_catalog``
    assembles it — a real corpus rather than a hand-picked fragment list, so a new
    family joins this gate for free."""
    cases: list[ConformanceCase] = []
    for generator in _FAMILY_GENERATORS:
        cases.extend(generator())
    return cases


def _metadata_capable(ctx: CompileContext, **caps: bool) -> CompileContext:
    """The store's own compile context with metadata capabilities forced ON.

    The stores' contexts are used rather than hand-built ones so this gate covers
    the field mapping and mode the scan paths really compile under. The
    capabilities are forced because ``sqlite_json1`` is probed from the running
    interpreter: on a build without JSON1 every metadata leaf would defer, and
    this module would silently stop covering the metadata emissions — a quieter
    version of the vacuity it is guarding against.
    """
    return dataclasses.replace(ctx, schema_capabilities=dataclasses.replace(ctx.schema_capabilities, **caps))


_COMPILERS: dict[str, tuple[Callable[..., Any], CompileContext]] = {
    "lance": (
        compile_lance,
        _metadata_capable(sqlite_context(), sqlite_json1=True),
    ),
    "surrealdb": (
        compile_surrealdb,
        _metadata_capable(surrealdb_context(), native_map_metadata=True, jsonb_path_query=True),
    ),
}

# Lower bounds, not the measured values — the corpus grows and these must not
# turn into a maintenance tax. Re-measured across the four (compiler, mode)
# pairs on this branch; the minimum of each column is 144 fragments compiled
# (lance/raise) and 76 of them nested (also lance/raise). The bounds exist so
# that a lowering or gate change which starts skipping the corpus wholesale
# fails loudly instead of passing with zero work.
#
# ``_MIN_NESTED`` is 40, not 1. A floor of 1 satisfies the letter of "at least
# one nested boolean node" while leaving no teeth: 75 of the 76 nested fragments
# could vanish and this gate would still pass, which is the same
# looks-tested-but-is-not failure the vacuous parametrizations this change
# replaced were.
# 40 is a little over half the measured minimum — comfortably clear of corpus
# churn, but it does not survive the boolean emissions being gated off.
_MIN_FRAGMENTS = 100
_MIN_NESTED = 40


@pytest.mark.parametrize("compiler_name", sorted(_COMPILERS))
@pytest.mark.parametrize("mode", ["raise", "split"])
def test_every_compiled_fragment_is_splice_safe(compiler_name: str, mode: str) -> None:
    """No fragment either text compiler emits carries a depth-0 boolean operator.

    Cases that raise are skipped for this ``(compiler, mode)`` pair — a fragment
    that is never produced cannot be mis-spliced — and the two lower bounds below
    are what stop "skipped everything" from reading as "passed".
    """
    compiler, base_ctx = _COMPILERS[compiler_name]
    ctx = dataclasses.replace(base_ctx, on_unsupported=mode)

    violations: list[str] = []
    compiled_count = 0
    nested_count = 0
    for case in _all_cases():
        try:
            compiled = compiler(_resolve_ast(case.filter), ctx)
        except (RecallFilterUnsupportedError, CompileError):
            # ``CompileError`` is not defensive padding: the corpus really does
            # provoke one, ``F-DOTKEY-dollar-key`` under ``surrealdb``/``raise``
            # (the identifier guard). It is the only one across all four pairs.
            continue
        compiled_count += 1
        problem = _depth0_boolean(compiled.predicate)
        if problem is not None:
            violations.append(f"{case.id}: {problem}")
        if _has_nested_boolean(compiled.predicate):
            nested_count += 1

    assert not violations, f"{compiler_name}/{mode} emitted unsafe fragments:\n" + "\n".join(violations)
    assert compiled_count >= _MIN_FRAGMENTS, (
        f"{compiler_name}/{mode} compiled only {compiled_count} fragments — the corpus stopped "
        f"reaching this compiler, so a green run proves nothing"
    )
    assert nested_count >= _MIN_NESTED, (
        f"{compiler_name}/{mode} produced no fragment with a nested boolean node — splice safety "
        f"is satisfied trivially when nothing boolean is emitted"
    )
