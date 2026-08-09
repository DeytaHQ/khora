"""Page-level oracle for the document enumeration — the RESULT-SURFACE checks.

The enumeration contract has four invariants a caller can actually be hurt by,
and all four are properties of the documents a walk *returns*:

* **INV-E, total order.** Every page, and the concatenation of every walk, is
  strictly descending in ``(created_at, id)``. Strictly, not merely
  non-ascending: a repeat is an exactly-once violation wearing an ordering
  costume, so the same comparison catches both.
* **Completeness.** The walk returns EVERY document the filter matches — see
  :func:`assert_walk_matches_expected`, and read the warning on
  :func:`assert_documents_satisfy` before relying on the soundness leg alone.
* **Soundness: the full filter holds on every returned row.** Weaker than it
  looks on the coordinator path; see below.
* **``next_after is None`` iff ``exhausted``.** The pair a walking caller loops
  on; either half alone is unactionable.

**Soundness and completeness are not interchangeable, and only one of them is
independent evidence here.** The coordinator applies
``compile_python(filter_ast, build_compile_context("documents",
on_unsupported="split")).predicate`` to every row before returning it
(``storage/coordinator.py``, in ``scan_documents_page``). This module compiles the
*same* predicate from the *same* AST, so on any page produced by the coordinator or
the facade, :func:`assert_documents_satisfy` passes **by construction** — it is a
cheap regression guard against the post-filter being removed or given the wrong
context, not proof that the page is right. Worse, the failure it *cannot* see is
the dangerous one: a silently DROPPED row never reaches the surface, so no
per-returned-row predicate can notice it. That is what
:func:`assert_walk_matches_expected` is for, and why a walk test that only asserts
soundness is not testing the thing that hurts. (On a raw ``scan_documents`` step,
where no post-filter has run, the soundness leg is genuinely independent — but a
step's window is a deliberate superset of the matches, so the assertion that
belongs there is the superset property, not this one.)

**Why nothing here reads the report.** The tempting shortcut is to read
``DocumentPage.post_filtered_keys`` and trust that the named leaves were enforced.
That is exactly the mistake this tier guards against: ``post_filtered_keys`` is self-reported
provenance, so a backend that lies about its split produces a *consistent* report
over a *wrong* row set. The split's honesty is checked at the execution seam
instead — :mod:`tests.test_helpers.document_scan_spy`.

Engine- and backend-agnostic on purpose, mirroring
:mod:`khora.filter.provenance`: it takes anything sequence-shaped carrying
``Document``-shaped rows, compiles its own predicate, and knows nothing about
which store produced the page. That is what lets the same four assertions serve
the four per-backend scan modules, the facade walk, and the property-based walk fuzzer.

``backend_target`` is a seam, not a knob: the default ``"documents"`` is the
target every documents-tier compile context declares, and ``compile_python``
reads ``field_mapping`` only for system keys (metadata always resolves through
``record.metadata``), which every documents context identity-maps. So the
identity context built here is compile-equivalent to any of the four backends'
own ``_documents_compile_context()`` for this compiler — the parameter exists for
a future non-``documents`` surface, not to be varied per store.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Any
from uuid import UUID

from khora.filter.compilers.python import compile_python
from khora.filter.execute import build_compile_context
from tests.test_helpers.document_scan import as_utc


def _as_uuid(value: Any) -> UUID:
    """Normalize a document id to :class:`~uuid.UUID` for comparison.

    Both normalizations here exist because a mixed compare *raises* instead of
    failing an assertion, which surfaces as an error in whatever test happens to
    be running rather than as an ordering failure.

    ``UUID`` passes through. A ``str`` is parsed — comparing as UUIDs rather than
    as strings is the semantically correct choice, not a convenience: every store
    orders on a uuid/record-id column, and the dashed string form sorts
    differently from the 16-byte value (``-`` sorts below every hex digit, which is
    the whole hazard the id-ladder seed in :mod:`tests.test_helpers.document_scan`
    exists to expose). A SurrealDB ``RecordID`` carries the uuid on ``.id``, so it
    recurses one level. Anything else fails loudly rather than being coerced.
    """
    if isinstance(value, UUID):
        return value
    if isinstance(value, str):
        return UUID(value)
    inner = getattr(value, "id", None)
    if inner is not None and inner is not value:
        return _as_uuid(inner)
    raise AssertionError(
        f"document id {value!r} ({type(value).__name__}) is not a UUID and cannot be normalized to one"
    )


def _order_key(doc: Any) -> tuple[datetime, UUID]:
    """The enumeration sort key: ``(created_at, id)``, both operands normalized.

    ``as_utc`` is load-bearing rather than tidy — PostgreSQL reads an aware
    ``created_at`` off ``timestamptz`` while the embedded tiers read theirs back
    naive. :func:`_as_uuid` does the same job for the id half. Normalizing keeps
    one oracle usable on every store.
    """
    return (as_utc(doc.created_at), _as_uuid(doc.id))


def assert_total_order(docs: Sequence[Any]) -> None:
    """Assert ``docs`` strictly descends in ``(created_at, id)`` (INV-E).

    STRICTLY descending, so this also catches a repeated document: a duplicate
    compares equal on both key halves and fails here even before an exactly-once
    check runs. Pass a single page's rows, or a whole walk's concatenation.
    """
    for a, b in zip(docs, docs[1:], strict=False):
        assert _order_key(a) > _order_key(b), (
            f"INV-E violated at {a.id} -> {b.id}: {_order_key(a)} does not sort strictly above {_order_key(b)}"
        )


def assert_documents_satisfy(docs: Iterable[Any], filter_ast: Any, *, backend_target: str = "documents") -> None:
    """Assert every returned document satisfies the FULL filter AST (SOUNDNESS only).

    **Not independent of the coordinator's post-filter, and it cannot catch a
    dropped row.** ``scan_documents_page`` applies this exact predicate — same
    compiler, same context — to every row before returning it, so on a coordinator
    or facade page this passes by construction. Keep it as a cheap guard against the
    post-filter being deleted or handed the wrong context; do NOT read a green run
    as evidence the page is correct. Pair it with
    :func:`assert_walk_matches_expected`, which checks the half that can actually
    fail silently: a document the filter matches that the walk never returned.

    Genuinely independent in one place only — a raw ``scan_documents`` step, which
    has had no post-filter applied. Even there the assertion to prefer is the
    superset property (the window must be a superset of the matches), since a
    step's window is deliberately wider than the match set.
    """
    predicate = compile_python(filter_ast, build_compile_context(backend_target, on_unsupported="split")).predicate
    for doc in docs:
        assert predicate(doc), f"returned doc {doc.id} does not satisfy the full filter"


def assert_walk_matches_expected(docs: Iterable[Any], expected_ids: Iterable[Any]) -> None:
    """Assert a walk returned EXACTLY the expected document ids (COMPLETENESS).

    The half of the contract no per-returned-row assertion can reach. A backend (or
    a keyset cursor) that silently skips a matching row produces a page on which
    every returned document satisfies the filter, is correctly ordered, and reports
    a consistent split — and is missing data the caller will never learn about.
    Only a comparison against an independently-known expectation catches it.

    ``expected_ids`` must come from the test's own knowledge of its seed corpus —
    hand-listed, or derived from the seeded ``Document`` objects — and **not** from
    another enumeration call, which would compare the implementation against
    itself. Ids are normalized through :func:`_as_uuid` on both sides, so a caller
    may pass ``Document`` objects' ids in whatever shape its store hands back.

    Set comparison, deliberately: ordering is :func:`assert_total_order`'s job and
    exactly-once is :func:`assert_walk_compliant`'s, so a failure here reports the
    missing and extra ids rather than a diff of two long lists.
    """
    got = {_as_uuid(doc.id) if hasattr(doc, "id") else _as_uuid(doc) for doc in docs}
    want = {_as_uuid(doc.id) if hasattr(doc, "id") else _as_uuid(doc) for doc in expected_ids}
    assert got == want, (
        f"walk did not return exactly the matching set — "
        f"MISSING (silently dropped): {sorted(want - got)}; UNEXPECTED (wrongly kept): {sorted(got - want)}"
    )


def assert_page_compliant(page: Any, filter_ast: Any = None, *, backend_target: str = "documents") -> None:
    """Assert one ``DocumentPage`` is order-correct, sound and walk-actionable.

    ``filter_ast`` is optional so the same helper serves an unfiltered walk; when
    given, every row on the page must satisfy the whole AST — with the soundness
    caveat on :func:`assert_documents_satisfy`, which applies in full here. This
    helper deliberately does NOT check completeness: a single page is *expected* to
    be a prefix of the match set (it is bounded by ``limit``), so the only place
    completeness is well-defined is across a finished walk. Use
    :func:`assert_walk_compliant` with ``expected_ids`` for that.

    The final assertion is the walk-control invariant ``next_after is None``
    **iff** ``exhausted`` — a page that reports neither a resume position nor
    exhaustion is the one pair a walking caller cannot act on, and a page that
    reports both is a walk that never terminates.
    """
    docs = list(page)
    assert_total_order(docs)
    if filter_ast is not None:
        assert_documents_satisfy(docs, filter_ast, backend_target=backend_target)
    assert (page.next_after is None) == page.exhausted, (
        f"next_after/exhausted disagree: next_after={page.next_after!r}, exhausted={page.exhausted!r}"
    )


def assert_walk_compliant(
    pages: Iterable[Any],
    filter_ast: Any = None,
    *,
    backend_target: str = "documents",
    expected_ids: Iterable[Any] | None = None,
) -> list[Any]:
    """Assert a whole walk: every page compliant, exactly-once, one total order.

    Checks each page individually, then the properties that only exist across
    pages: no document appears twice; the concatenation is a single descending run
    (a per-page order that resets at each boundary passes
    :func:`assert_page_compliant` on every page and is still a broken walk); and the
    walk terminated correctly — nothing follows a page that reported
    ``exhausted=True``, and (when any pages were supplied) the last page *is*
    exhausted. That last rule is what makes the completeness leg meaningful: a walk
    stopped early is a prefix of the match set, so judging it for completeness would
    pass on rows that were simply never fetched. The per-page
    ``next_after is None`` iff ``exhausted`` invariant is about a single page and
    cannot see either termination property.

    ``expected_ids`` adds the completeness leg via
    :func:`assert_walk_matches_expected`. **Pass it whenever the test knows its
    corpus** — without it this call proves only that what came back was ordered,
    unique and (per the caveat on :func:`assert_documents_satisfy`) tautologically
    filter-satisfying, none of which can fail when a matching row is dropped.

    Returns the flattened documents in walk order, so a caller can go on to
    compare the ordered list against its own corpus expectation.
    """
    seen: set[UUID] = set()
    flat: list[Any] = []
    saw_any = False
    finished = False
    for page in pages:
        assert not finished, "pages after exhaustion: a page followed one that already reported exhausted=True"
        saw_any = True
        assert_page_compliant(page, filter_ast, backend_target=backend_target)
        for doc in page:
            doc_id = _as_uuid(doc.id)
            assert doc_id not in seen, f"exactly-once violated: {doc_id} repeated"
            seen.add(doc_id)
            flat.append(doc)
        finished = page.exhausted
    if saw_any:
        assert finished, (
            "walk did not finish: the last page reported exhausted=False — a truncated walk "
            "proves nothing about completeness, since a missing match may simply be on a page never fetched"
        )
    assert_total_order(flat)
    if expected_ids is not None:
        assert_walk_matches_expected(flat, expected_ids)
    return flat


__all__ = [
    "assert_documents_satisfy",
    "assert_page_compliant",
    "assert_total_order",
    "assert_walk_compliant",
    "assert_walk_matches_expected",
]
