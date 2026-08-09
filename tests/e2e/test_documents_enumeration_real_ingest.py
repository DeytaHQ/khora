"""Document-enumeration filter sweep over a corpus written by the REAL ingest path.

Every other leg of the enumeration proof seeds its ``documents`` rows through the
storage write API (``create_document``), which is the right seam for a *compiler*
proof — it puts exactly the declared field values in the row and nothing else. This
module seeds the same generated corpus through :meth:`khora.Khora.remember` instead,
so the rows under test carry whatever the ingest pipeline actually persists, and then
walks :meth:`khora.Khora.list_documents` against them. What it can catch that a
write-API leg cannot: an ingest transform that changes the filterable surface
(``""`` collapsing to ``NULL``, a metadata-derived ``source_timestamp``, a
denormalization that never fires) would leave the compiler perfectly correct and the
caller's filter silently wrong.

Nothing here hand-lists a predicate. The corpus is
:func:`khora.filter.conformance.documents_conformance_cases` — the generated
operator × composition product (the per-system-key operator atoms plus every eligible
case from the shared logic / date / array / existence / object-equality / sugar /
dot-key families). The expectation is the case's own ``expected_ids``, which the
implementation-blind Python oracle
(:func:`khora.filter.conformance.documents_oracle_survivors`) recomputes here from
the seed records alone; an expectation is never taken from a second enumeration call.

**Every read is a WALK, never a single fetch.** ``limit`` is deliberately small
(:data:`_WALK_LIMIT`) so any case matching more rows than that pages, and the keyset
cursor is carried across the boundary rather than being skipped; the sweep asserts
that at least one case actually did so, since a corpus that shrank below the limit
would quietly turn every walk back into one fetch. The pages go to
:func:`tests.test_helpers.document_page_oracle.assert_walk_compliant` with
``expected_ids`` always supplied — the completeness leg is the only one that can see
a silently dropped row.

**What real ingest cannot seed, and why the sweep is a subset of the corpus.**
``Khora.remember`` exposes no keyword for ``created_at`` (khora-ops ingest time,
always "now") or ``content_type`` (never written by this path), so a case filtering
either key would be asserted against a corpus that does not carry its seed values.
Those cases are excluded by :func:`_is_ingest_seedable`, which derives the exclusion
from the AST's leaf keys rather than from a hand-maintained case-id list — a new case
on either key drops out on its own. The excluded shapes remain covered by the
write-API legs, where the columns can be stamped directly.

Two legs, one body: the container-free embedded ``sqlite_lance`` stack (always runs)
and live Postgres (self-skips when unreachable, so a no-Docker run collects and skips
cleanly). Both run the Skeleton engine — it has no graph channel and no typed
extraction, so ingest reduces to exactly the document/chunk write this module is
about. The deterministic stub embedder is installed by the fixtures; no network.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

import pytest

from khora import Khora
from khora.core.models.document import DocumentStatus
from khora.filter import RecallFilter
from khora.filter.ast import parse_to_ast
from khora.filter.conformance import (
    ConformanceCase,
    SeedRecord,
    _documents_case_namespace_id,
    documents_conformance_cases,
    documents_oracle_survivors,
    seed_documents_case,
)
from khora.filter.execute import iter_leaf_clauses
from khora.filter.model import SYSTEM_KEYS as _SYSTEM_KEYS
from khora.filter.model import RecallFilterValidationError
from tests.e2e import _harness
from tests.test_helpers.document_page_oracle import assert_walk_compliant
from tests.test_helpers.document_scan import as_utc

pytestmark = [pytest.mark.e2e, pytest.mark.slow]


# --------------------------------------------------------------------------- #
# Leg selection.
# --------------------------------------------------------------------------- #

_EMBEDDED = "embedded"
_POSTGRES = "postgres"

# leg name -> (Khora fixture, module owning that store's documents compile context).
_LEGS: dict[str, tuple[str, str]] = {
    _EMBEDDED: ("skeleton_sqlite_lance_kb", "khora.storage.backends.sqlite_lance.relational"),
    _POSTGRES: ("skeleton_pgvector_kb", "khora.storage.backends.postgresql"),
}


@dataclass(frozen=True, slots=True)
class _Leg:
    """A resolved leg: its name, its connected ``Khora``, and its backend module."""

    name: str
    kb: Khora
    backend_module: str


@pytest.fixture
def leg(request: pytest.FixtureRequest) -> _Leg:
    """Resolve one leg's connected ``Khora``, self-skipping when its store is down.

    The Postgres leg skips on the same socket probe the rest of the e2e suite uses
    (``_harness._pg_reachable``), so a no-Docker run still executes the embedded leg
    in full. CI converts that skip into a hard red via ``KHORA_E2E_PG_REQUIRED=1``
    (see ``tests/e2e/conftest.py``), so a broken container leg cannot pass green.
    """
    name = request.param
    fixture_name, backend_module = _LEGS[name]
    if name == _POSTGRES and not _harness._pg_reachable():
        pytest.skip("start Postgres (make dev) to exercise the live document-enumeration leg")
    return _Leg(name=name, kb=request.getfixturevalue(fixture_name), backend_module=backend_module)


both_legs = pytest.mark.parametrize("leg", list(_LEGS), indirect=True, ids=list(_LEGS))


# --------------------------------------------------------------------------- #
# What the real ingest path can seed.
# --------------------------------------------------------------------------- #

# The ``SeedRecord`` string fields ``Khora.remember`` exposes a keyword for. Unlike
# the recall row-set harness, ``external_id`` is threaded VERBATIM here rather than
# being commandeered as a reconciliation handle: the corpus contains an
# ``external_id`` operator family, so overwriting the column would silently redefine
# those cases' expectations. Reconciliation instead rides on ``RememberResult``'s own
# ``document_id`` (see :func:`_seed_corpus`), which perturbs no filterable field at
# all — not the row, not the metadata blob.
_THREADABLE_STRING_KEYS: tuple[str, ...] = (
    "source_type",
    "source_name",
    "source_url",
    "source",
    "title",
    "external_id",
)

# The filter leaf roots a real-ingest corpus can carry the seed values for: the six
# threadable strings, the user-supplied ``source_timestamp`` column, and the
# ``metadata`` blob (any sub-path — the whole dict is threaded).
#
# Absent, and load-bearing: ``created_at`` is khora-ops ingest time with no
# ``remember`` keyword (every seeded row would read "now", not the case's seed
# value), and ``content_type`` is never written by this path (every row reads NULL).
# ``occurred_at`` is absent from the generated documents corpus already — it is not
# an enumerable key at all, which :func:`test_occurred_at_is_not_enumerable` pins.
_INGEST_SEEDABLE_ROOTS: frozenset[str] = frozenset({*_THREADABLE_STRING_KEYS, "source_timestamp", "metadata"})

# Small on purpose: a corpus of 4+ records then spans pages on every case, so each
# walk exercises the keyset cursor rather than degenerating into one fetch.
_WALK_LIMIT = 3

# A walk over a handful of seeded records cannot legitimately need this many pages;
# tripping it means the cursor is not advancing.
_MAX_PAGES = 64


def _case_ast(case: ConformanceCase) -> Any:
    """Lower a case's filter through the real validator + parser (never a hand-built AST)."""
    model = case.filter if isinstance(case.filter, RecallFilter) else RecallFilter.model_validate(case.filter)
    return parse_to_ast(model)


def _is_ingest_seedable(case: ConformanceCase) -> bool:
    """Whether every leaf key of ``case`` is one the real ingest path can seed.

    Derived from the lowered AST, so a case added later on ``created_at`` or
    ``content_type`` drops out automatically instead of silently being asserted
    against a corpus that never carried its seed values. A filter with no leaf keys
    at all (the empty filter) is vacuously seedable.
    """
    roots = {clause.path[0] for clause in iter_leaf_clauses(_case_ast(case)) if clause.path}
    return roots <= _INGEST_SEEDABLE_ROOTS


def ingest_seedable_cases() -> list[ConformanceCase]:
    """The generated documents corpus, restricted to what real ingest can seed."""
    return [case for case in documents_conformance_cases() if _is_ingest_seedable(case)]


def _assert_sweep_is_not_vacuous(cases: Sequence[ConformanceCase]) -> None:
    """Assert the swept subset still addresses every key real ingest can seed.

    :func:`_is_ingest_seedable` is a filter, and a filter that silently starts
    matching nothing leaves a green sweep that proves nothing. This pins both ends of
    it: the swept cases must collectively address EVERY root in
    :data:`_INGEST_SEEDABLE_ROOTS` (so no key quietly stopped being covered), and the
    keys the exclusion actually costs must still be exactly the two documented ingest
    gaps (so a case is never dropped for a reason nobody wrote down).
    """
    assert cases, "the generated documents corpus produced no ingest-seedable cases"
    swept = {clause.path[0] for case in cases for clause in iter_leaf_clauses(_case_ast(case)) if clause.path}
    assert swept == _INGEST_SEEDABLE_ROOTS, (
        f"the swept corpus addresses {sorted(swept)}, but the ingest-seedable key set is "
        f"{sorted(_INGEST_SEEDABLE_ROOTS)} — coverage drifted"
    )
    forfeited = {
        clause.path[0]
        for case in documents_conformance_cases()
        if not _is_ingest_seedable(case)
        for clause in iter_leaf_clauses(_case_ast(case))
        if clause.path and clause.path[0] not in _INGEST_SEEDABLE_ROOTS
    }
    assert forfeited == {"created_at", "content_type"}, (
        f"the exclusion now forfeits {sorted(forfeited)}; only created_at (no remember keyword) and "
        "content_type (never written by this path) are accounted for"
    )


def _corpus_key(records: Sequence[SeedRecord]) -> str:
    """A stable identity for a seed corpus, so cases sharing one share a namespace.

    Many cases in the generated corpus differ only in their filter and re-declare the
    same records (the eight ``source_name`` operator atoms, for instance). Seeding
    that corpus once and pointing every case at the same namespace turns ~900
    ``remember`` calls into ~180 without weakening anything: the filter is the only
    narrowing force, so a case's expectation over the shared namespace is exactly its
    expectation over a private one.

    Every seed value in the corpus is JSON-native, so a canonical JSON dump is a
    faithful identity; ``default=repr`` keeps a future non-native value from raising
    here (at worst two such corpora would be seeded separately, which is safe).
    """
    return json.dumps(
        [
            [
                rec.id,
                rec.content,
                rec.metadata,
                rec.source_timestamp.isoformat() if rec.source_timestamp is not None else None,
                *[getattr(rec, key) for key in _THREADABLE_STRING_KEYS],
            ]
            for rec in records
        ],
        sort_keys=True,
        default=repr,
    )


# --------------------------------------------------------------------------- #
# Seeding through Khora.remember() — the real write path.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class _SeededCorpus:
    """One seed corpus, written through ``remember`` into its own namespace."""

    namespace_id: UUID
    #: seed id -> the ``RememberResult.document_id`` that seed produced.
    doc_ids: dict[str, UUID]
    #: document id -> seed id (the reverse map the survivor assertions read).
    seed_ids: dict[UUID, str]
    #: document id -> the row's ``updated_at``, for the ``updated_before`` kwarg leg.
    updated_at: dict[UUID, datetime]


def _remember_kwargs(record: SeedRecord) -> dict[str, Any]:
    """The ``remember`` keywords that reproduce one record's filterable surface.

    Threads each populated field VERBATIM — including ``external_id``, which the
    corpus filters on — and omits any field the record leaves ``None`` so the column
    lands NULL, which is what "absent" means to the oracle. Content is made distinct
    per record (the corpus gives every record the same anchor text) so ingest's
    checksum de-duplication cannot collapse two records into one enumerable row;
    ``content`` is not a filterable key, so distinguishing it perturbs nothing.
    """
    kwargs: dict[str, Any] = {
        "content": f"{record.content} record {record.id}",
        "entity_types": [],
        "relationship_types": [],
    }
    for key in _THREADABLE_STRING_KEYS:
        value = getattr(record, key)
        if value is not None:
            kwargs[key] = value
    if record.source_timestamp is not None:
        kwargs["source_timestamp"] = record.source_timestamp
    if record.metadata:
        kwargs["metadata"] = dict(record.metadata)
    return kwargs


def _assert_round_trip(record: SeedRecord, doc: Any) -> None:
    """Assert one seeded row carries the field values its ``SeedRecord`` declared.

    The precondition every assertion downstream rests on: if ingest quietly rewrote a
    field (an empty string where the record said absent, a ``source_timestamp``
    derived from the metadata blob, a dropped metadata key), the sweep would be
    measuring the filter against a corpus nobody declared, and a mismatch would read
    as a filter bug. Checked once per seeded row, before any filter runs.
    """
    for key in _THREADABLE_STRING_KEYS:
        assert getattr(doc, key) == getattr(record, key), (
            f"seed {record.id}: ingest persisted {key}={getattr(doc, key)!r}, record declared "
            f"{getattr(record, key)!r} — the filterable surface does not match the corpus"
        )
    if record.source_timestamp is None:
        assert doc.source_timestamp is None, (
            f"seed {record.id}: ingest invented source_timestamp={doc.source_timestamp!r} for a record that "
            "declared none (a metadata-derived timestamp would do exactly this)"
        )
    else:
        assert as_utc(doc.source_timestamp) == as_utc(record.source_timestamp), (
            f"seed {record.id}: source_timestamp round-tripped as {doc.source_timestamp!r}, "
            f"record declared {record.source_timestamp!r}"
        )
    assert doc.metadata == dict(record.metadata), (
        f"seed {record.id}: metadata round-tripped as {doc.metadata!r}, record declared {record.metadata!r}"
    )
    # ``content_type`` has no ``remember`` keyword; a non-NULL value here would mean
    # ingest started writing the column and the exclusion above needs revisiting.
    assert doc.content_type is None, (
        f"seed {record.id}: ingest wrote content_type={doc.content_type!r}; the ingest-seedable key set "
        "excludes content_type on the premise that this path never writes it"
    )


async def _seed_corpus(kb: Khora, records: Sequence[SeedRecord]) -> _SeededCorpus:
    """Ingest one seed corpus into a fresh namespace and verify it round-tripped.

    One ``remember`` call per record, in declaration order. The reconciliation key is
    ``RememberResult.document_id`` — the write path's own answer to "which row did
    this record become" — so no field of the row and no key of the metadata blob is
    borrowed for bookkeeping, and every case's declared filter stays valid against
    the corpus it was authored for.
    """
    namespace_id = (await kb.create_namespace()).namespace_id
    doc_ids: dict[str, UUID] = {}
    for record in records:
        result = await kb.remember(namespace=namespace_id, **_remember_kwargs(record))
        doc_ids[record.id] = result.document_id
    assert len(set(doc_ids.values())) == len(records), (
        f"ingest collapsed {len(records)} records into {len(set(doc_ids.values()))} documents — "
        "checksum de-duplication would do this if two records shared content"
    )

    seeded = await _walk_documents(kb, namespace_id)
    by_id = {doc.id: doc for doc in seeded}
    assert set(by_id) == set(doc_ids.values()), (
        f"unfiltered walk returned {sorted(by_id)}, ingest reported {sorted(doc_ids.values())}"
    )
    for record in records:
        _assert_round_trip(record, by_id[doc_ids[record.id]])

    return _SeededCorpus(
        namespace_id=namespace_id,
        doc_ids=doc_ids,
        seed_ids={doc_id: seed_id for seed_id, doc_id in doc_ids.items()},
        updated_at={doc.id: doc.updated_at for doc in seeded},
    )


async def _seed_all(kb: Khora, cases: Sequence[ConformanceCase]) -> dict[str, _SeededCorpus]:
    """Seed every distinct corpus the given cases draw on; keyed by :func:`_corpus_key`."""
    corpora: dict[str, _SeededCorpus] = {}
    for case in cases:
        key = _corpus_key(case.seed_records)
        if key not in corpora:
            corpora[key] = await _seed_corpus(kb, case.seed_records)
    return corpora


# --------------------------------------------------------------------------- #
# Walking.
# --------------------------------------------------------------------------- #


async def _walk_pages(
    kb: Khora,
    namespace_id: UUID,
    *,
    filter: Any = None,
    status: DocumentStatus | None = None,
    updated_before: datetime | None = None,
) -> list[Any]:
    """Drive a full keyset walk and return every page, in order.

    Loops on ``exhausted`` (never on a short page, which is not an end signal) and
    resumes from ``next_after``. The page cap converts a cursor that fails to advance
    into a bounded failure instead of a hang.
    """
    pages: list[Any] = []
    after: Any = None
    while True:
        page = await kb.list_documents(
            namespace=namespace_id,
            filter=filter,
            status=status,
            updated_before=updated_before,
            limit=_WALK_LIMIT,
            after=after,
        )
        pages.append(page)
        if page.exhausted:
            return pages
        assert len(pages) < _MAX_PAGES, f"walk exceeded {_MAX_PAGES} pages without exhausting — cursor not advancing"
        after = page.next_after


async def _walk_documents(kb: Khora, namespace_id: UUID, **kwargs: Any) -> list[Any]:
    """Walk and return the flattened documents, with the page oracle applied."""
    pages = await _walk_pages(kb, namespace_id, **kwargs)
    return assert_walk_compliant(pages)


def _fail_with(failures: list[str], what: str) -> None:
    """Report an accumulated sweep failure list — the first few in full, then a roll-up.

    The sweep runs the whole generated corpus in one test, so failing on the first
    case would hide whether the defect is one shape or a whole family. Full detail on
    the first few plus a count of the rest keeps the report actionable without
    dumping hundreds of tracebacks.
    """
    if not failures:
        return
    head = "\n\n".join(failures[:5])
    tail = f"\n\n(+{len(failures) - 5} more)" if len(failures) > 5 else ""
    pytest.fail(f"{len(failures)} {what}:\n\n{head}{tail}", pytrace=False)


# --------------------------------------------------------------------------- #
# The sweep: every generated operator + composition case, walked.
# --------------------------------------------------------------------------- #


@both_legs
async def test_generated_filter_sweep_over_real_ingest(leg: _Leg) -> None:
    """Every ingest-seedable generated case walks to exactly its expected document set.

    Two assertions per case, and they fail for different reasons:

    * the WALK is compliant — ordered, exactly-once, correctly terminated, sound, and
      **complete** against the seeded ids the case expects (the only leg that can see
      a matching row the enumeration silently dropped); and
    * the surviving SEED ids equal both ``case.expected_ids`` and the
      implementation-blind oracle's recomputation of them from the seed records. That
      is the seed-fidelity cross-check: it says the real ingest transform and the
      write-API corpus the oracle describes are the same corpus, which is what makes
      the first assertion's expectation trustworthy in the first place.
    """
    cases = ingest_seedable_cases()
    _assert_sweep_is_not_vacuous(cases)
    corpora = await _seed_all(leg.kb, cases)

    failures: list[str] = []
    paged = 0
    for case in cases:
        corpus = corpora[_corpus_key(case.seed_records)]
        assert case.expected_ids is not None, f"{case.id}: documents corpus case without expected_ids"
        expected_docs = [corpus.doc_ids[seed_id] for seed_id in case.expected_ids]
        try:
            pages = await _walk_pages(leg.kb, corpus.namespace_id, filter=case.filter)
            paged += len(pages) > 1
            docs = assert_walk_compliant(pages, _case_ast(case), expected_ids=expected_docs)
            survivors = frozenset(corpus.seed_ids[doc.id] for doc in docs)
            assert survivors == case.expected_ids, f"survivors {sorted(survivors)} != {sorted(case.expected_ids)}"
            assert survivors == documents_oracle_survivors(case), (
                f"survivors {sorted(survivors)} != oracle {sorted(documents_oracle_survivors(case))}"
            )
        except AssertionError as exc:
            failures.append(f"[{case.id}] filter={case.filter!r}\n{exc}")

    _fail_with(failures, f"of {len(cases)} generated cases failed on the {leg.name} leg")
    assert paged, (
        f"no case's match set spanned more than one page at limit={_WALK_LIMIT} — every walk above collapsed "
        "to a single fetch, so the keyset cursor was never exercised"
    )


# --------------------------------------------------------------------------- #
# Implicit-AND ≡ explicit $and (and the corpus's other equivalence pairs).
# --------------------------------------------------------------------------- #

# Case-id pairs the generated corpus declares row-equivalent: two spellings of one
# predicate that must select byte-identical row sets. The implicit-AND / ``$and``
# pair is the one an enumeration caller writes by accident; the rest ride along
# because the same walk-and-compare proves them. Both members of a pair are seeded
# from the same records, so the comparison is over the same physical documents.
_EQUIVALENT_CASE_PAIRS: tuple[tuple[str, str], ...] = (
    ("F-LOGIC-implicit-and", "F-LOGIC-explicit-and"),
    ("F-LOGIC-or", "F-LOGIC-in-equiv"),
    ("F-LOGIC-nor", "F-LOGIC-not-or"),
    ("F-LOGIC-demorgan-not-and", "F-LOGIC-demorgan-or-ne"),
    ("F-LOGIC-distrib-and-or", "F-LOGIC-distrib-or-and"),
)


@both_legs
async def test_equivalent_filter_spellings_walk_identically(leg: _Leg) -> None:
    """Row-equivalent filter spellings return identical walks — implicit-AND ≡ ``$and``.

    Compared as sets of returned document ids, not against a declared expectation:
    the point is that the two spellings agree with EACH OTHER on the same physical
    corpus. Both members' agreement with the oracle is the sweep's job.
    """
    by_id = {case.id: case for case in ingest_seedable_cases()}
    missing = [cid for pair in _EQUIVALENT_CASE_PAIRS for cid in pair if cid not in by_id]
    assert not missing, f"equivalence pairs name cases absent from the ingest-seedable corpus: {missing}"

    cases = [by_id[cid] for pair in _EQUIVALENT_CASE_PAIRS for cid in pair]
    corpora = await _seed_all(leg.kb, cases)

    for left_id, right_id in _EQUIVALENT_CASE_PAIRS:
        left, right = by_id[left_id], by_id[right_id]
        assert _corpus_key(left.seed_records) == _corpus_key(right.seed_records), (
            f"{left_id} / {right_id} no longer share a seed corpus — the comparison would span two namespaces"
        )
        corpus = corpora[_corpus_key(left.seed_records)]
        left_docs = await _walk_documents(leg.kb, corpus.namespace_id, filter=left.filter)
        right_docs = await _walk_documents(leg.kb, corpus.namespace_id, filter=right.filter)
        assert {doc.id for doc in left_docs} == {doc.id for doc in right_docs}, (
            f"{left_id} ({left.filter!r}) and {right_id} ({right.filter!r}) are declared row-equivalent but "
            f"returned {sorted(corpus.seed_ids[d.id] for d in left_docs)} vs "
            f"{sorted(corpus.seed_ids[d.id] for d in right_docs)}"
        )


# --------------------------------------------------------------------------- #
# Filter × enumeration-kwarg interaction.
# --------------------------------------------------------------------------- #


def _representative_cases() -> list[ConformanceCase]:
    """One case per generated family — the widest shape spread at the smallest cost.

    The kwarg legs cost three more walks per case, so running them over the whole
    corpus would triple the sweep to re-prove the same conjunction. One case per
    family instead, preferring a case whose match set is a PROPER non-empty subset of
    its corpus: intersecting a kwarg with an all-or-nothing filter cannot distinguish
    "the kwarg was applied" from "the kwarg was ignored".
    """
    best: dict[str, ConformanceCase] = {}
    for case in ingest_seedable_cases():
        if case.expected_ids is None:
            continue
        family = "-".join(case.id.split("-")[:2])
        proper = 0 < len(case.expected_ids) < len(case.seed_records)
        incumbent = best.get(family)
        if incumbent is None:
            best[family] = case
        elif proper and not (0 < len(incumbent.expected_ids or ()) < len(incumbent.seed_records)):
            best[family] = case
    return [best[family] for family in sorted(best)]


@both_legs
async def test_filter_and_enumeration_kwargs_intersect(leg: _Leg) -> None:
    """``filter`` AND ``status`` AND ``updated_before`` — a document must satisfy all three.

    The three legs are chosen so that each can fail on its own. ``status`` is
    asserted in both directions (the status every ingested document carries must
    leave the match set untouched; a status none carries must empty it), because a
    kwarg that is silently dropped passes the first check alone. ``updated_before``
    is bound to a value read off the seeded rows themselves — the median
    ``updated_at`` — so the bound genuinely splits the corpus rather than admitting
    or rejecting all of it, and the expectation is the intersection of the filter's
    match set with the rows under the bound.

    The final assertion guards the guard: at least one representative case must have
    produced a proper, non-trivial intersection, so a corpus that drifted into
    all-or-nothing bounds cannot leave this test passing vacuously.
    """
    cases = _representative_cases()
    assert cases, "no representative cases selected"
    corpora = await _seed_all(leg.kb, cases)

    failures: list[str] = []
    informative = 0
    for case in cases:
        corpus = corpora[_corpus_key(case.seed_records)]
        assert case.expected_ids is not None
        matched = {corpus.doc_ids[seed_id] for seed_id in case.expected_ids}
        stamps = sorted(corpus.updated_at.values())
        bound = stamps[len(stamps) // 2]
        under_bound = {doc_id for doc_id, stamp in corpus.updated_at.items() if stamp < bound}
        try:
            # The status every ingested document carries: a pure no-op on the match set.
            pages = await _walk_pages(leg.kb, corpus.namespace_id, filter=case.filter, status=DocumentStatus.COMPLETED)
            assert_walk_compliant(pages, _case_ast(case), expected_ids=list(matched))

            # A status none carries: the conjunction must empty the match set.
            pages = await _walk_pages(leg.kb, corpus.namespace_id, filter=case.filter, status=DocumentStatus.PENDING)
            assert_walk_compliant(pages, _case_ast(case), expected_ids=[])

            # The real intersection: filter AND updated_at < bound.
            pages = await _walk_pages(leg.kb, corpus.namespace_id, filter=case.filter, updated_before=bound)
            assert_walk_compliant(pages, _case_ast(case), expected_ids=list(matched & under_bound))
        except AssertionError as exc:
            failures.append(f"[{case.id}] filter={case.filter!r} updated_before={bound!r}\n{exc}")
        if 0 < len(matched & under_bound) < len(matched):
            informative += 1

    _fail_with(failures, f"of {len(cases)} filter x kwarg intersections failed on the {leg.name} leg")
    assert informative, (
        "no representative case produced a partial filter x updated_before intersection — every assertion above "
        "was all-or-nothing, so a dropped updated_before kwarg would not have been caught"
    )


# --------------------------------------------------------------------------- #
# The metadata anchor: an arbitrary ingest-supplied metadata key is filterable.
# --------------------------------------------------------------------------- #

_ANCHOR_PATH = "vault/reports/q3-close.md"


@both_legs
async def test_metadata_key_supplied_at_ingest_is_filterable(leg: _Leg) -> None:
    """A free-form ``metadata`` key written by ``remember`` narrows an enumeration to its row.

    The end-to-end claim a caller actually makes: hand ``remember`` an arbitrary
    metadata key, then enumerate on ``metadata.<key>`` and get back that document and
    nothing else. The two decoys make the assertion falsifiable — one row carries a
    different value under the same key, one omits the key entirely, so a filter that
    was dropped on the way to SQL would return three rows, not one.
    """
    kb = leg.kb
    namespace_id = (await kb.create_namespace()).namespace_id
    common = {"entity_types": [], "relationship_types": []}
    target = await kb.remember(
        content="quarterly close narrative",
        namespace=namespace_id,
        metadata={"source_path": _ANCHOR_PATH, "tier": "gold"},
        **common,
    )
    await kb.remember(
        content="unrelated planning note",
        namespace=namespace_id,
        metadata={"source_path": "vault/notes/planning.md", "tier": "gold"},
        **common,
    )
    await kb.remember(content="note with no path at all", namespace=namespace_id, metadata={"tier": "gold"}, **common)

    filter_ = {"metadata.source_path": {"$eq": _ANCHOR_PATH}}
    pages = await _walk_pages(kb, namespace_id, filter=filter_)
    docs = assert_walk_compliant(
        pages, parse_to_ast(RecallFilter.model_validate(filter_)), expected_ids=[target.document_id]
    )
    assert [doc.metadata["source_path"] for doc in docs] == [_ANCHOR_PATH]


# --------------------------------------------------------------------------- #
# Enumerable key names.
# --------------------------------------------------------------------------- #

# The one system key a document row cannot back: the chunk event-time axis.
_NON_ENUMERABLE_KEY = "occurred_at"
_ENUMERABLE_KEYS: frozenset[str] = _SYSTEM_KEYS - {_NON_ENUMERABLE_KEY}

# The system keys backed by a date column. They need a date-shaped probe below: the
# validator rejects a presence test on a date key on TYPE grounds, which has nothing
# to do with whether the key is enumerable, and a probe that cannot tell the two
# rejections apart would report every date key as non-enumerable.
_DATE_SYSTEM_KEYS: frozenset[str] = frozenset({"occurred_at", "created_at", "source_timestamp"})


def _key_probe_filter(key: str) -> dict[str, Any]:
    """A filter that is type-valid for ``key``, so any rejection is about the KEY."""
    if key in _DATE_SYSTEM_KEYS:
        return {key: {"$gte": "2020-01-01T00:00:00Z"}}
    return {key: {"$exists": True}}


@both_legs
async def test_enumerable_system_keys_are_accepted(leg: _Leg) -> None:
    """Every system key except ``occurred_at`` is accepted by the enumeration surface.

    Behavioural, key by key, against the live leg — the set of names a caller may
    filter on is checked by filtering on them, not by reading a constant. An empty
    namespace suffices: acceptance is a validation outcome, and a rejected key raises
    before any row is read.

    A rejection is only counted as "not enumerable" when the store says so in the
    structured code; any other validation failure is re-raised rather than quietly
    shrinking the accepted set, which is how a type-shaped rejection would otherwise
    masquerade as a key-scope rule.
    """
    namespace_id = (await leg.kb.create_namespace()).namespace_id
    accepted = set()
    for key in sorted(_SYSTEM_KEYS):
        try:
            await _walk_pages(leg.kb, namespace_id, filter=_key_probe_filter(key))
        except RecallFilterValidationError as exc:
            codes = {error.code for error in exc.errors}
            assert codes == {"key_not_enumerable"}, f"{key} was rejected for an unrelated reason: {exc.errors}"
            continue
        accepted.add(key)
    assert accepted == _ENUMERABLE_KEYS, (
        f"enumerable keys drifted: accepted {sorted(accepted)}, expected {sorted(_ENUMERABLE_KEYS)}"
    )


@both_legs
@pytest.mark.parametrize(
    "filter_",
    [
        pytest.param({_NON_ENUMERABLE_KEY: {"$gte": "2020-01-01T00:00:00Z"}}, id="bare"),
        pytest.param(
            {"$and": [{"source_name": "linear"}, {_NON_ENUMERABLE_KEY: {"$gt": "2020-01-01T00:00:00Z"}}]},
            id="nested",
        ),
    ],
)
async def test_occurred_at_is_not_enumerable(leg: _Leg, filter_: dict[str, Any]) -> None:
    """``occurred_at`` is rejected with a structured code, at the top level and nested.

    Rejection, not a silent empty result: a document row carries no event-time
    column, so evaluating the key would return zero rows and look like "nothing
    matched". The structured ``allowed`` list is asserted too — it is what an SDK
    surfaces to a caller, so it has to name the real enumerable set.
    """
    namespace_id = (await leg.kb.create_namespace()).namespace_id
    with pytest.raises(RecallFilterValidationError) as excinfo:
        await leg.kb.list_documents(namespace=namespace_id, filter=filter_, limit=_WALK_LIMIT)
    error = excinfo.value.errors[0]
    assert error.code == "key_not_enumerable"
    assert error.path == _NON_ENUMERABLE_KEY
    assert error.allowed == sorted(_ENUMERABLE_KEYS)


@both_legs
def test_backend_documents_field_mapping_matches_enumerable_keys(leg: _Leg) -> None:
    """The leg's own documents compile context never maps a non-enumerable key.

    Binds the behavioural check above to the store-side declaration a filter is
    actually compiled against: the ``field_mapping`` is that store's pushdown
    whitelist, so a key appearing there is a key the compiler will emit SQL for.
    ``occurred_at`` must be absent (no such column exists), the ``metadata`` root
    must remap to the physical column, and every declared system key must
    identity-map.

    Deliberately a SUBSET check, not equality: a store legitimately withholds a
    backed key from pushdown when its stored format cannot be compared soundly
    against a compiled bind (the embedded tier withholds both date keys for exactly
    that reason and routes them to the post-filter). Withholding narrows what is
    pushed, never what is enumerable — which is why the enumerable set is asserted
    behaviourally above and only the "nothing extra, nothing renamed" half is
    asserted here.
    """
    context = importlib.import_module(leg.backend_module)._documents_compile_context()
    mapping = dict(context.field_mapping)
    assert context.backend_target == "documents"
    assert mapping.pop("metadata", None) == "metadata", "the metadata root must remap to the physical column"
    assert set(mapping) <= _ENUMERABLE_KEYS, f"documents pushdown declares non-enumerable keys: {set(mapping)}"
    assert _NON_ENUMERABLE_KEY not in mapping
    assert all(column == key for key, column in mapping.items()), f"system keys must identity-map, got {mapping}"


# --------------------------------------------------------------------------- #
# Cross-check against the write-API (create_document) corpus.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("leg", [_EMBEDDED], indirect=True, ids=[_EMBEDDED])
async def test_write_api_corpus_walks_to_the_same_survivors(leg: _Leg) -> None:
    """The same cases seeded via ``create_document`` select the same seed ids.

    The reconciliation between this module's real-ingest corpus and the write-API
    corpus every other enumeration leg uses. Both sides are asserted against
    ``case.expected_ids`` — the sweep above for the ingest side, this test for the
    write-API side — so agreeing with it is agreeing with each other, over the same
    filters and the same declared records.

    **Embedded only, and the boundary is not incidental.** ``seed_documents_case``
    pins each case's namespace to a deterministic id derived from the case id, so the
    rows it writes collide on ``memory_namespaces`` with any other process that
    seeded the same corpus into the same database — which the integration
    documents-conformance legs do, on the shared Postgres. The embedded leg's
    database is a private temporary file created fresh for this test, so the pinned
    ids are unowned there and the seeder's fail-loud duplicate detection stays a
    genuine signal rather than a scheduling accident.
    """
    cases = ingest_seedable_cases()
    coordinator = leg.kb.storage

    failures: list[str] = []
    for case in cases:
        assert case.expected_ids is not None
        try:
            id_map = await seed_documents_case(coordinator, case)
            seed_of = {doc_id: seed_id for seed_id, doc_id in id_map.items()}
            pages = await _walk_pages(leg.kb, _documents_case_namespace_id(case), filter=case.filter)
            docs = assert_walk_compliant(
                pages,
                _case_ast(case),
                expected_ids=[id_map[seed_id] for seed_id in case.expected_ids],
            )
            survivors = frozenset(seed_of[doc.id] for doc in docs)
            assert survivors == case.expected_ids, f"survivors {sorted(survivors)} != {sorted(case.expected_ids)}"
        except AssertionError as exc:
            failures.append(f"[{case.id}] filter={case.filter!r}\n{exc}")

    _fail_with(failures, f"of {len(cases)} write-API cases disagreed with the declared survivors")
