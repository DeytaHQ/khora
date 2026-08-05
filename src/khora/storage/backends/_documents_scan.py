"""Shared query construction for the bounded documents scan — ``@internal``.

The two SQLAlchemy-backed relational stores (PostgreSQL and the embedded
sqlite_lance adapter) run the *same* bounded keyset scan over the *same*
``DocumentModel``. Only one thing genuinely differs between them: how their
dialect compiler's pushdown fragment attaches to the statement (PostgreSQL's
compiler emits a SQLAlchemy expression; the SQLite compiler emits a string plus
positional binds). Everything else — the namespace scope, the optional ``status``
/ ``updated_before`` narrowing, the keyset predicate, the enumeration order, and
the row bound — is identical, so it is written once here rather than twice.

That is not only DRY. The keyset predicate is the one part of this scan whose
correctness depends on a *serialization* detail, and it is the part that cannot
be exercised without a live PostgreSQL. Building it once, in a form the embedded
store's own tests can execute, means the locally-runnable leg proves the shared
semantics and only the fragment attachment is left to a services-backed CI job.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy import Select

from khora.db.models import DocumentModel
from khora.storage.backends.base import DocumentScanKey

__all__ = ["build_documents_scan_query"]


def build_documents_scan_query(
    namespace_id: UUID,
    *,
    status: str | None = None,
    updated_before: datetime | None = None,
    after: DocumentScanKey | None = None,
    scan_limit: int = 100,
) -> Select[tuple[DocumentModel]]:
    """Build the bounded scan ``SELECT``, minus any compiled filter fragment.

    ``@internal``. The caller ``AND``-s its own dialect's pushdown fragment onto
    the result with a further ``.where(...)`` — appending a WHERE after
    ``order_by`` / ``limit`` is well-defined and leaves both in place — and then
    executes it in its own session.

    ``ORDER BY created_at DESC, id DESC`` is the total enumeration order all four
    relational stores adopted in khora #1576. ``created_at`` is ``NOT NULL`` as of
    khora #1583; the keyset predicate below depends on that — a NULL key would
    make the row-value comparison evaluate to NULL and silently drop the row from
    every page.

    **The explicit ``literal(value, <column>.type)`` on each cursor operand is
    load-bearing. Do not simplify it away.** A right-hand ``tuple_`` types each of
    its elements from the *value*, independently of the column it is compared
    against — nothing propagates leftwards. ``tuple_(a_datetime, a_uuid)``
    therefore renders correctly only because those two Python types happen to
    infer to this table's own column types; the agreement is a coincidence, not a
    mechanism. Hand it an operand one step off and the coincidence lapses
    silently: a ``date`` where a ``datetime`` belongs infers ``Date`` and binds
    ``'2026-01-31'``, which on the embedded store's TEXT column sorts below every
    ``'2026-01-31 …'`` row and skips the whole day. Naming the column's type
    forces the correct serialization instead, and turns the ``str``-instead-of-
    ``UUID`` slip from a silently mis-ordering dashed 36-char bind into an
    immediate ``AttributeError`` from the type's own processor.

    Formatting the cursor by hand is wrong in two directions and neither is loud:
    on the embedded store an ISO-8601 ``'T'`` separator sorts *above* the stored
    space-separated form, so the cursor's own row compares less than itself and a
    resumed walk returns it forever; while a space-separated form that omits the
    ``.000000`` microseconds sorts *below* its tie-mates, silently skipping them.
    Be precise about which construct reaches that second form, because it is not
    the one people reach for. ``text(...).bindparams(x=a_datetime)`` is **safe**:
    SQLAlchemy infers ``DateTime`` from the value, runs the processor, and emits
    the full ``'… 12:30:00.000000'``. Only a bind explicitly typed
    ``NullType()`` — or a value handed to the DBAPI outside SQLAlchemy's typing
    altogether — leaves the ``datetime`` raw for the driver's own deprecated
    adapter, and a walk built on that loses every tie-mate. Both forms measured.

    **Naming the type is a SQLite-side guard only — it is not what makes a cursor
    correct.** On PostgreSQL no bind processor runs, so a *well-typed* operand
    reaches the driver untouched under either spelling — though not a no-op even
    there, because the spelling picks the cast: measured on the asyncpg dialect, a
    ``date`` renders ``$1::DATE`` bare against ``$1::TIMESTAMP WITH TIME ZONE``
    typed, and a ``str`` id renders ``$2::VARCHAR`` bare against ``$2::UUID``.
    Neither spelling rescues the ``date`` — those two forms resolve midnight in
    the server's zone and the client host's respectively. A ``str`` id diverges
    the other way and is the sharpest asymmetry here: the typed spelling raises on
    the *embedded* store, whose column type's processor asks the operand for ``.hex``,
    while asyncpg's uuid encoder takes the ``PyUnicode`` branch and its parser
    skips ``-`` outright, so a dashed cursor decodes to the identical sixteen
    bytes and is silently *correct* on PostgreSQL. The same bad input is loud on
    one store and harmless on the other; neither store's behaviour predicts the
    other's, which is the whole reason this rule lives here rather than in either
    backend. PostgreSQL has the same hazard by a different route, on two
    different evidence standards. Its ``timestamptz`` encoder converts with
    ``obj.astimezone(utc)``, so a naive cursor resolves against whatever zone the
    process happens to run in — that is plain-Python arithmetic, run across four
    zones, and the shift is the host's UTC offset, hence zero and invisible on a
    UTC host. That the same encoder also wraps a bare ``date`` at midnight in the
    host's local zone is **read from its source only**: unlike the uuid parser
    above, it is a ``cdef`` with no Python entry point, so it cannot be exercised
    without a live server. The guarantee that holds on both stores is therefore
    the cheap one: **build a cursor from a row the store you are querying
    returned, never by hand.** What the typed bind buys is that the embedded
    path fails loudly, or not at all, instead of quietly reordering — worth
    having, but not a substitute.

    The cursor must not be *adjusted* on the way in either, and the two dialects
    make opposite adjustments look harmless. The embedded store holds wall clock
    with the writer's offset discarded, so attaching or stripping ``tzinfo``
    there is a no-op while converting to another zone moves the position; a
    ``timestamptz`` holds an instant, so converting is the no-op and stripping
    ``tzinfo`` is what moves it — by the host's UTC offset, hence exactly zero on
    a UTC host and invisible to a test that runs there. Neither store's rule
    generalizes to the other. The one that holds on both is narrower than either:
    bind the value you read, unmodified.

    Uniform descending on both keys is what makes a single row-value comparison
    legal; do not expand it to an ``OR`` form. Row values need SQLite >= 3.15,
    which this package cannot be installed below: ``requires-python >= 3.13`` and
    CPython 3.13's ``sqlite3`` requires 3.15.2 at build time. No runtime probe.

    Args:
        namespace_id: Row-level namespace scope. Always applied — a scan is never
            cross-namespace.
        status: Optional document-status narrowing, matching ``list_documents``.
        updated_before: Optional half-open ``updated_at <`` bound.
        after: Resume position — the ``(created_at, id)`` of the last row a
            previous step scanned. ``None`` starts from the newest row. Must have
            come from this same store (the key is store-local; see
            :data:`~khora.storage.backends.base.DocumentScanKey`).
        scan_limit: Maximum rows the window returns. Bounds rows *returned*, not
            rows *examined* — a selective pushdown fragment can still make one
            call read the whole namespace.

    Raises:
        ValueError: ``scan_limit`` below 1. A zero bound would return an empty
            window that reports neither a resume position nor exhaustion, which a
            walking caller cannot make progress past.
    """
    if scan_limit < 1:
        raise ValueError(f"scan_limit must be >= 1, got {scan_limit}")

    query = sa.select(DocumentModel).where(DocumentModel.namespace_id == namespace_id)
    if status:
        query = query.where(DocumentModel.status == status)
    if updated_before is not None:
        query = query.where(DocumentModel.updated_at < updated_before)
    if after is not None:
        cursor_created_at, cursor_id = after
        query = query.where(
            sa.tuple_(DocumentModel.created_at, DocumentModel.id)
            < sa.tuple_(
                sa.literal(cursor_created_at, DocumentModel.created_at.type),
                sa.literal(cursor_id, DocumentModel.id.type),
            )
        )
    return query.order_by(DocumentModel.created_at.desc(), DocumentModel.id.desc()).limit(scan_limit)
