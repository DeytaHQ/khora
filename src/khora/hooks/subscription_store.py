"""Durable storage for persistent hook subscriptions (#599).

Phase 3 of the semantic-hooks rollout (refs #577, #580). The
``HookDispatcher`` keeps its in-process callbacks in memory; a restart
loses them. A *persistent* subscription instead records a delivery target
(webhook URL / queue config) to the ``khora_hook_subscriptions`` table
(migration 049) so a worker can re-subscribe on startup.

This module is the persistence boundary only. It does NOT deliver events -
the webhook/queue worker is out of scope for #599. The dispatcher calls
:meth:`HookSubscriptionStore.persist` on register, :meth:`load_all` on
startup, and :meth:`delete` on unsubscribe.

The SQL mirrors :mod:`khora.dream.runstore` - one shared statement set,
dialect-aware UUID / timestamp / JSON binding so the embedded SQLite
stack stores text/JSON and Postgres stores native types. A
``session_factory`` is supplied by the caller (the storage coordinator on
PG, the relational adapter's factory on SQLite).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import text

from .models import SemanticFilter

# ---------------------------------------------------------------------------
# Persistent-subscription record
# ---------------------------------------------------------------------------


@dataclass
class PersistentSubscription:
    """A durable hook subscription with a delivery target.

    Cleanly separate from ``HookSubscription`` (the in-process callback
    shape): a persistent subscription carries no Python callable - it
    carries a ``delivery`` config (webhook URL / queue identifier) that a
    worker process resolves after a restart.
    """

    event_type: str
    delivery: dict[str, Any]
    id: UUID = field(default_factory=uuid4)
    namespace_id: UUID | None = None
    filter: SemanticFilter | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_delivered_at: datetime | None = None
    delivery_failure_count: int = 0
    paused_at: datetime | None = None


# ---------------------------------------------------------------------------
# SemanticFilter (de)serialization — only the persisted fields. Embeddings
# are recomputed from the description at load time (the dispatcher already
# drains pending filters via embed_pending_filters), so we do not store the
# vectors.
# ---------------------------------------------------------------------------


def serialize_filter(filt: SemanticFilter) -> dict[str, Any]:
    """Serialize a ``SemanticFilter`` to a JSON-safe dict.

    Embeddings are intentionally dropped - they are large and recomputed
    from the description on load. ``namespace_id`` is stored on the row,
    not the filter blob, so it is omitted here.
    """
    return {
        "id": str(filt.id),
        "name": filt.name,
        "description": filt.description,
        "entity_types": list(filt.entity_types),
        "relationship_types": list(filt.relationship_types),
        "dream_op_types": list(filt.dream_op_types),
        "dream_decisions": list(filt.dream_decisions),
        "match": filt.match,
        "examples": list(filt.examples),
        "anti_examples": list(filt.anti_examples),
        "similarity_threshold": filt.similarity_threshold,
        "llm_confidence_threshold": filt.llm_confidence_threshold,
        "filter_model": filt.filter_model,
    }


def deserialize_filter(raw: dict[str, Any], *, namespace_id: UUID | None) -> SemanticFilter:
    """Rebuild a ``SemanticFilter`` from a persisted blob."""
    fid = raw.get("id")
    return SemanticFilter(
        id=UUID(str(fid)) if fid else uuid4(),
        name=raw.get("name", ""),
        description=raw.get("description", ""),
        entity_types=list(raw.get("entity_types") or []),
        relationship_types=list(raw.get("relationship_types") or []),
        dream_op_types=list(raw.get("dream_op_types") or []),
        dream_decisions=list(raw.get("dream_decisions") or []),
        match=raw.get("match"),
        examples=list(raw.get("examples") or []),
        anti_examples=list(raw.get("anti_examples") or []),
        similarity_threshold=raw.get("similarity_threshold", 0.5),
        llm_confidence_threshold=raw.get("llm_confidence_threshold", 0.5),
        filter_model=raw.get("filter_model"),
        namespace_id=namespace_id,
    )


# ---------------------------------------------------------------------------
# Dialect helpers (mirror khora.dream.runstore)
# ---------------------------------------------------------------------------


def _is_postgres(session: Any) -> bool:
    bind = getattr(session, "bind", None)
    dialect = getattr(bind, "dialect", None)
    return dialect is not None and dialect.name == "postgresql"


def _uuid_param(session: Any, value: UUID | None) -> Any:
    if value is None:
        return None
    return value if _is_postgres(session) else str(value)


def _ts_param(session: Any, value: datetime | None) -> Any:
    if value is None:
        return None
    return value if _is_postgres(session) else value.isoformat()


def _json_param(value: dict[str, Any] | None) -> Any:
    # Always bind JSON as a string; the SQL CASTs to jsonb on Postgres and
    # stores text on SQLite's JSON column (mirrors khora.dream.runstore).
    if value is None:
        return None
    return json.dumps(value)


def _json_cast(session: Any, placeholder: str) -> str:
    return f"CAST({placeholder} AS jsonb)" if _is_postgres(session) else placeholder


def _coerce_uuid(value: Any) -> UUID | None:
    if value is None:
        return None
    return value if isinstance(value, UUID) else UUID(str(value))


def _coerce_dt(value: Any) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _coerce_json(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return json.loads(value) if value else None
    return dict(value)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class HookSubscriptionStore:
    """Raw-SQL persistence for ``khora_hook_subscriptions`` (#599).

    Constructed with a ``session_factory`` (an ``async_sessionmaker`` or
    any zero-arg callable returning an ``AsyncSession``). The store owns
    each session's transaction: commit on success, rollback on error.
    """

    def __init__(self, session_factory: Any) -> None:
        self._session_factory = session_factory

    def _open(self) -> _CommittingSession:
        return _CommittingSession(self._session_factory)

    async def persist(self, sub: PersistentSubscription) -> None:
        """Insert (or replace, keyed on ``id``) a persistent subscription."""
        async with self._open() as session:
            flt_cast = _json_cast(session, ":flt")
            dlv_cast = _json_cast(session, ":dlv")
            await session.execute(
                text(
                    "INSERT INTO khora_hook_subscriptions "  # noqa: S608 - casts are hardcoded literals
                    "(id, namespace_id, event_type, filter, delivery, created_at, "
                    " last_delivered_at, delivery_failure_count, paused_at) "
                    f"VALUES (:id, :ns, :et, {flt_cast}, {dlv_cast}, :ca, :lda, :dfc, :pa) "
                    "ON CONFLICT (id) DO UPDATE SET "
                    " namespace_id = EXCLUDED.namespace_id, "
                    " event_type = EXCLUDED.event_type, "
                    " filter = EXCLUDED.filter, "
                    " delivery = EXCLUDED.delivery"
                ),
                {
                    "id": _uuid_param(session, sub.id),
                    "ns": _uuid_param(session, sub.namespace_id),
                    "et": sub.event_type,
                    "flt": _json_param(serialize_filter(sub.filter) if sub.filter else None),
                    "dlv": _json_param(sub.delivery),
                    "ca": _ts_param(session, sub.created_at),
                    "lda": _ts_param(session, sub.last_delivered_at),
                    "dfc": sub.delivery_failure_count,
                    "pa": _ts_param(session, sub.paused_at),
                },
            )

    async def load_all(self) -> list[PersistentSubscription]:
        """Return every persistent subscription (paused ones included)."""
        async with self._open() as session:
            result = await session.execute(
                text(
                    "SELECT id, namespace_id, event_type, filter, delivery, "
                    " created_at, last_delivered_at, delivery_failure_count, paused_at "
                    "FROM khora_hook_subscriptions"
                )
            )
            rows = result.fetchall()
        return [_row_to_subscription(row) for row in rows]

    async def delete(self, subscription_id: UUID) -> bool:
        """Remove a persistent subscription. Returns True if a row was deleted."""
        async with self._open() as session:
            result = await session.execute(
                text("DELETE FROM khora_hook_subscriptions WHERE id = :id"),
                {"id": _uuid_param(session, subscription_id)},
            )
        return getattr(result, "rowcount", 0) > 0


def _row_to_subscription(row: Any) -> PersistentSubscription:
    namespace_id = _coerce_uuid(row.namespace_id)
    filter_raw = _coerce_json(row.filter)
    filt = deserialize_filter(filter_raw, namespace_id=namespace_id) if filter_raw else None
    return PersistentSubscription(
        id=_coerce_uuid(row.id),  # type: ignore[arg-type]
        namespace_id=namespace_id,
        event_type=row.event_type,
        filter=filt,
        delivery=_coerce_json(row.delivery) or {},
        created_at=_coerce_dt(row.created_at),  # type: ignore[arg-type]
        last_delivered_at=_coerce_dt(row.last_delivered_at),
        delivery_failure_count=row.delivery_failure_count or 0,
        paused_at=_coerce_dt(row.paused_at),
    )


class _CommittingSession:
    """Open a session, yield it, commit on success / roll back on error.

    Mirrors ``khora.dream.runstore._CommittingSession`` - the store owns
    the session lifecycle for both the standalone PG and SQLite paths.
    """

    def __init__(self, factory: Any) -> None:
        self._factory = factory
        self._session: Any = None

    async def __aenter__(self) -> Any:
        self._session = self._factory()
        return self._session

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        assert self._session is not None
        try:
            if exc_type is None:
                await self._session.commit()
            else:
                await self._session.rollback()
        finally:
            await self._session.close()
