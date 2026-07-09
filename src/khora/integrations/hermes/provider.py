"""``KhoraMemoryProvider`` — Hermes ``MemoryProvider`` backed by khora.

Hermes (``hermes-agent``) drives a long-running agent session and looks
up tools / context via a plugin-supplied ``MemoryProvider``. This module
ships the factory that builds one such provider, backed by khora's
vector + entity-graph recall and a background runtime that keeps the
sync Hermes call surface non-blocking.

Distribution model (per issue #628): this provider lives in khora's
optional ``[hermes]`` extra. Hermes itself sees it through a thin plugin
directory at ``examples/integrations/hermes/plugin/`` that calls
:func:`KhoraMemoryProvider` with ``kb=Khora.shared()`` (or a
user-supplied connected ``Khora``).

Module-load discipline: nothing from ``hermes_agent`` is imported at
module top level. The ``MemoryProvider`` ABC is resolved lazily on first
:func:`KhoraMemoryProvider` call (mirrors the LangGraph / Google ADK /
OpenAI Agents trick). The AST lint (``tools/check_optional_imports.py``)
does not catch ``import hermes_agent`` (note the underscore), so the
subprocess no-import test in
``tests/unit/integrations/test_no_eager_imports.py`` is the gate of
record.

Stability: experimental until the upstream ``hermes-agent`` SDK reaches
1.0 and ships one full minor without breaking ``MemoryProvider``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from loguru import logger

from khora.integrations.hermes._mapping import (
    derive_namespace_uuid,
    format_memory_context,
    message_pair_iter,
    turn_to_document,
)
from khora.integrations.hermes._tools import (
    dispatch_memory_recall,
    dispatch_memory_search,
    memory_recall_schema,
    memory_search_schema,
)
from khora.telemetry import bounded_text_hash, metric_counter, trace_span

if TYPE_CHECKING:  # pragma: no cover - typing only
    from khora.integrations.hermes._runtime import _KhoraRuntime
    from khora.khora import Khora


# Module-level counter instrument (created once, reused for every
# tool-call attribute combination). Per the CLAUDE.md cardinality rule,
# the ``tool`` label is bounded ("memory_search" | "memory_recall" |
# "unknown" | "uninitialized") — no namespace_id / session_id.
_TOOL_CALL_COUNTER = metric_counter(
    "khora.hermes.tool_call_total",
    description="Hermes tool calls handled by the khora memory provider, by tool name.",
)


# Cached pointer to the Hermes ``MemoryProvider`` ABC. Resolved on first
# ``KhoraMemoryProvider`` call; reused thereafter.
_MemoryProviderBase: type | None = None


def _get_memory_provider_base() -> type:
    """Lazy-resolve ``hermes_agent.agent.memory_provider.MemoryProvider``.

    Centralised so all method-body imports route through one helper —
    one place to maintain the error message when the extra isn't
    installed, and one place to pay the import cost. The AST lint at
    ``tools/check_optional_imports.py`` flags top-level ``import
    hermes_agent``; this lazy resolver is the only sanctioned import
    path.
    """
    global _MemoryProviderBase
    if _MemoryProviderBase is not None:
        return _MemoryProviderBase
    try:
        from hermes_agent.agent.memory_provider import (  # noqa: PLC0415 — lazy
            MemoryProvider,
        )
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ImportError(
            "KhoraMemoryProvider requires the optional `hermes` extra. Install with: pip install 'khora[hermes]'"
        ) from exc
    _MemoryProviderBase = MemoryProvider
    return MemoryProvider


def _get_runtime() -> type[_KhoraRuntime]:
    """Lazy import of the background runtime.

    Kept in a helper so the factory body stays linear and the runtime
    module isn't pulled in until a provider is actually instantiated.
    Python Engineer #2 owns ``_runtime.py``.
    """
    from khora.integrations.hermes._runtime import _KhoraRuntime  # noqa: PLC0415

    return _KhoraRuntime


def _build_namespace(kb: Khora, namespace_id: UUID, *, agent_identity: str, user_id: str | None) -> None:
    """Best-effort namespace creation — mirrors google_adk._ensure_namespace.

    Khora has no ``get_or_create_namespace`` primitive: ``_resolve_namespace``
    raises ``ValueError`` when the row is missing, and ``storage.create_namespace``
    is the actual create. We probe-then-create with a race-tolerant
    re-probe on failure, then swallow only the create error if the row
    really did materialise (concurrent provider).

    The namespace metadata records the *tenancy* key (agent_identity +
    user_id). ``session_id`` is intentionally NOT stored here — the
    namespace spans every session for this identity (issue #1466).
    """
    from khora.core.models.tenancy import MemoryNamespace  # noqa: PLC0415

    try:
        # Synchronous helper — we accept that we're called from
        # ``initialize`` which is sync per the Hermes ABC. The async
        # bridge lives in the runtime; namespace creation is a one-shot
        # bootstrap call that runs on the sync bridge loop.
        from khora.integrations._sync import run_sync  # noqa: PLC0415
    except ImportError:  # pragma: no cover - defensive only
        raise

    async def _ensure() -> None:
        try:
            await kb._resolve_namespace(namespace_id)
            return
        except ValueError:
            pass
        ns = MemoryNamespace(
            id=namespace_id,
            namespace_id=namespace_id,
            metadata={
                "source": "khora.integrations.hermes",
                "agent_identity": agent_identity,
                "user_id": user_id,
            },
        )
        try:
            await kb.storage.create_namespace(ns)
        except Exception as exc:  # pragma: no cover - race-safe creation
            try:
                await kb._resolve_namespace(namespace_id)
            except ValueError:
                raise exc from None
            logger.debug(
                "KhoraMemoryProvider namespace creation race resolved cleanly: {}",
                exc,
            )

    run_sync(_ensure())


def KhoraMemoryProvider(  # noqa: N802 — factory function masquerading as a class constructor
    *,
    kb: Khora,
    runtime: _KhoraRuntime | None = None,
    prefetch_timeout_s: float = 0.8,
    prefetch_cache_ttl_s: float = 30.0,
    queue_max_size: int = 256,
    drain_timeout_s: float = 5.0,
    failure_threshold_pct: float = 1.0,
) -> Any:
    """Build a Hermes ``MemoryProvider`` backed by Khora.

    Per (a)+example-dir distribution: this lives in khora's ``[hermes]``
    extra. The example plugin directory at
    ``examples/integrations/hermes/plugin/`` calls this factory with
    ``kb=Khora.shared()`` (or a user-supplied connected ``Khora``).

    The returned object is an instance of a dynamically-built subclass of
    the Hermes ``MemoryProvider`` ABC. The dynamic-subclass trick keeps
    ``import khora.integrations.hermes.provider`` working with the
    ``[hermes]`` extra absent — only the actual factory call requires
    Hermes to be installed.

    Args:
        kb: REQUIRED. A connected :class:`khora.Khora` instance. There is
            NO silent ``Khora.shared()`` fallback here — adapters MUST
            receive an explicit handle so their lifecycle is the
            caller's problem. The example plugin is the only place
            ``Khora.shared()`` is wired in.
        runtime: Optional injected ``_KhoraRuntime`` for tests. ``None``
            in production — the runtime is constructed lazily on first
            ``initialize``.
        prefetch_timeout_s: Upper bound on a synchronous
            ``prefetch`` call. On timeout the provider returns the
            abstention payload ("No prior memories found.").
        prefetch_cache_ttl_s: Lifetime of a cached prefetch result keyed
            on ``(namespace_id, session_id, query_hash)``.
        queue_max_size: Bounded background-queue capacity. Beyond this,
            ``enqueue_*`` drops the oldest pending work (runtime decides
            the eviction policy — provider just hands it the knob).
        drain_timeout_s: Wall clock budget for ``on_session_end`` to
            drain the runtime queue before returning.
        failure_threshold_pct: Background-task failure rate (percentage)
            above which ``on_session_end`` logs WARN with the last few
            exception summaries.

    Raises:
        ImportError: If ``hermes-agent`` is not installed.
    """
    base = _get_memory_provider_base()

    class _KhoraMemoryProviderImpl(base):  # type: ignore[misc, valid-type]
        """Concrete Hermes ``MemoryProvider`` backed by khora."""

        # Hermes inspects ``name`` as a property on the instance; we
        # implement it as a property so the contract matches whether
        # Hermes treats it as descriptor or attribute access.
        @property
        def name(self) -> str:
            return "khora"

        def __init__(self) -> None:
            self.kb = kb
            self._runtime_override = runtime
            self._runtime: _KhoraRuntime | None = runtime
            self._prefetch_timeout_s = prefetch_timeout_s
            self._prefetch_cache_ttl_s = prefetch_cache_ttl_s
            self._queue_max_size = queue_max_size
            self._drain_timeout_s = drain_timeout_s
            self._failure_threshold_pct = failure_threshold_pct

            # Populated by ``initialize``.
            self._namespace_id: UUID | None = None
            self._session_id: str = ""
            self._agent_identity: str = ""
            self._user_id: str | None = None
            self._turn_seq: int = 0

            # Try to call the upstream ABC's __init__ if it expects one.
            # Hermes's ``MemoryProvider`` may or may not define a
            # zero-arg ``__init__`` — be defensive.
            try:
                super().__init__()
            except TypeError:  # pragma: no cover - ABC has positional args
                pass

        # ------------------------------------------------------------------
        # KhoraIntegration marker Protocol attrs
        # ------------------------------------------------------------------

        @property
        def namespace_id(self) -> UUID:
            """Return the bound namespace id, or the zero UUID before init."""
            return self._namespace_id if self._namespace_id is not None else UUID(int=0)

        # ------------------------------------------------------------------
        # MemoryProvider required surface
        # ------------------------------------------------------------------

        def is_available(self) -> bool:
            """True when the bound Khora handle looks usable.

            Khora exposes no synchronous ``is_healthy`` today, so we
            check ``_connected`` (the private flag set by ``connect()``)
            with a defensive ``getattr`` so future renames degrade
            silently to "assume healthy". If a real ``is_healthy()``
            ships later, this picks it up automatically.
            """
            if self.kb is None:
                return False
            probe = getattr(self.kb, "is_healthy", None)
            if callable(probe):
                try:
                    return bool(probe())
                except Exception:  # noqa: BLE001 - any failure means "not healthy"
                    return False
            return bool(getattr(self.kb, "_connected", True))

        def initialize(self, session_id: str, **kwargs: Any) -> None:
            """Bind the provider to a Hermes agent session.

            Hermes always passes ``agent_identity``, ``hermes_home``, and
            ``platform`` in ``kwargs``; ``agent_context``, ``user_id``,
            etc. may also be present. The khora *namespace* is derived from
            the stable ``(agent_identity, user_id)`` identity ONLY — the
            ``session_id`` is the conversation scope and is threaded to
            khora's ``session_id`` column via the document mapping, not the
            namespace. Folding ``session_id`` into the namespace would give
            every session a fresh tenancy and void cross-session memory
            (issue #1466).
            """
            agent_identity = kwargs.get("agent_identity", "unknown")
            user_id = kwargs.get("user_id")
            with trace_span(
                "khora.integrations.hermes.initialize",
                **{
                    "hermes.agent_identity_hash": bounded_text_hash(agent_identity),
                    "hermes.user_id_hash": bounded_text_hash(str(user_id or "")),
                    "hermes.session_id_hash": bounded_text_hash(session_id),
                    "hermes.platform": str(kwargs.get("platform", "")),
                },
            ):
                self._agent_identity = agent_identity
                self._user_id = user_id
                self._session_id = session_id
                self._namespace_id = derive_namespace_uuid(agent_identity, user_id)
                self._turn_seq = 0

                _build_namespace(
                    self.kb,
                    self._namespace_id,
                    agent_identity=agent_identity,
                    user_id=user_id,
                )

                # Build runtime lazily so test injection wins over the
                # default daemon-loop runtime.
                if self._runtime is None:
                    runtime_cls = _get_runtime()
                    self._runtime = runtime_cls(
                        prefetch_cache_ttl_s=self._prefetch_cache_ttl_s,
                        queue_max_size=self._queue_max_size,
                    )

                logger.info(
                    "KhoraMemoryProvider initialized agent_identity={} user_id={} session_id={}",
                    agent_identity,
                    user_id,
                    session_id,
                )

        def system_prompt_block(self) -> str:
            """Return the system-prompt fragment Hermes injects into the LLM.

            Tells the model two things: (1) memories arrive automatically
            via ``<memory-context>``, (2) the two tools are available
            when an explicit lookup is needed (temporal questions,
            "what did we say about X", etc.).
            """
            return (
                "You have access to long-term memory of prior conversations with this user. "
                "Relevant context is auto-injected as a <memory-context> block before each turn. "
                "For explicit lookups, call `memory_search` (semantic) or `memory_recall` "
                "(semantic + time window)."
            )

        def get_tool_schemas(self) -> list[dict[str, Any]]:
            """Return the two tool schemas Hermes registers for this provider."""
            return [memory_search_schema(), memory_recall_schema()]

        # ------------------------------------------------------------------
        # Prefetch path (sync, latency-bounded)
        # ------------------------------------------------------------------

        def prefetch(self, query: str, *, session_id: str = "") -> str:
            """Synchronous recall, bounded by ``prefetch_timeout_s``.

            Hermes calls this on the hot path before each turn; we MUST
            return quickly. The runtime owns a small cache keyed on
            ``(namespace_id, session_id, query_hash)`` to absorb repeats
            (e.g. the same query firing for prefetch + tool-call), and a
            timeout that surfaces the abstention payload when khora is
            slow rather than blocking the LLM.
            """
            if self._namespace_id is None or self._runtime is None:
                return format_memory_context([])

            # session_id arg is accepted to match the Hermes signature but
            # the prefetch cache is namespace-scoped already, so we do
            # not need to thread it further today. Kept in the signature
            # so future variants can filter on it without an API break.
            _ = session_id

            effective_session = session_id or self._session_id
            with trace_span(
                "khora.integrations.hermes.prefetch",
                **{"hermes.query_hash": bounded_text_hash(query)},
            ) as span:
                cached = self._runtime.recall_sync(
                    self.kb,
                    self._namespace_id,
                    effective_session,
                    query,
                    timeout=self._prefetch_timeout_s,
                )
                span.set_attribute("cache_hit", cached is not None)
                if cached is None:
                    span.set_attribute("result_count", 0)
                    return format_memory_context([])
                chunks = list(getattr(cached, "chunks", []) or [])
                entities = getattr(cached, "entities", None)
                span.set_attribute("result_count", len(chunks))
                return format_memory_context(chunks, entities)

        def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
            """Fire-and-forget prefetch — warms the cache for the next turn."""
            if self._namespace_id is None or self._runtime is None:
                return
            effective_session = session_id or self._session_id
            self._runtime.enqueue_recall(
                self.kb,
                self._namespace_id,
                effective_session,
                query,
            )

        # ------------------------------------------------------------------
        # Write path (sync, non-blocking)
        # ------------------------------------------------------------------

        def sync_turn(
            self,
            user_content: str,
            assistant_content: str,
            *,
            session_id: str = "",
        ) -> None:
            """Persist one (user, assistant) turn through the background runtime.

            Returns immediately — the runtime owns the actual
            ``kb.remember`` call. Errors surface via the runtime's
            failure-rate counter (logged at ``on_session_end``).
            """
            if self._namespace_id is None or self._runtime is None:
                return

            with trace_span(
                "khora.integrations.hermes.sync_turn",
                **{
                    "hermes.user_content_hash": bounded_text_hash(user_content),
                    "hermes.assistant_content_hash": bounded_text_hash(assistant_content),
                },
            ):
                # Stamp a monotonically increasing seq per session. No
                # lock — Hermes calls ``sync_turn`` from one thread per
                # session by contract; if that contract weakens, swap to
                # itertools.count.
                self._turn_seq += 1
                turn_seq = self._turn_seq

                document = turn_to_document(
                    user_content,
                    assistant_content,
                    session_id=session_id or self._session_id,
                    turn_seq=turn_seq,
                    namespace_id=self._namespace_id,
                )
                self._runtime.enqueue_remember(self.kb, self._namespace_id, document)

        # ------------------------------------------------------------------
        # Tool-call dispatch (sync, bridges to async khora)
        # ------------------------------------------------------------------

        def handle_tool_call(self, tool_name: str, args: dict[str, Any]) -> str:
            """Dispatch a Hermes tool call synchronously and return text for the LLM."""
            if self._namespace_id is None or self._runtime is None:
                _TOOL_CALL_COUNTER.add(1, attributes={"tool": "uninitialized"})
                return format_memory_context([])

            if tool_name == "memory_search":
                _TOOL_CALL_COUNTER.add(1, attributes={"tool": "memory_search"})
                return self._runtime.dispatch_sync(
                    dispatch_memory_search,
                    self.kb,
                    self._namespace_id,
                    args,
                )
            if tool_name == "memory_recall":
                _TOOL_CALL_COUNTER.add(1, attributes={"tool": "memory_recall"})
                return self._runtime.dispatch_sync(
                    dispatch_memory_recall,
                    self.kb,
                    self._namespace_id,
                    args,
                )

            _TOOL_CALL_COUNTER.add(1, attributes={"tool": "unknown"})
            raise ValueError(f"unknown tool {tool_name!r}")

        # ------------------------------------------------------------------
        # Optional Hermes hooks
        # ------------------------------------------------------------------

        def on_pre_compress(self, messages: list[dict]) -> None:
            """Flush the tail of a conversation about to be compressed.

            Hermes calls this just before context-compression drops
            older messages; we use the chance to dump every (user,
            assistant) pair in the about-to-be-compressed window into
            khora so the long-term memory keeps them.
            """
            if self._namespace_id is None or self._runtime is None:
                return
            pairs = list(message_pair_iter(messages))
            if not pairs:
                return
            documents = []
            for i, (user, assistant) in enumerate(pairs):
                # Use a contiguous seq range starting after current
                # cursor so re-ingest dedupes via external_id.
                turn_seq = self._turn_seq + i + 1
                documents.append(
                    turn_to_document(
                        user,
                        assistant,
                        session_id=self._session_id,
                        turn_seq=turn_seq,
                        namespace_id=self._namespace_id,
                    )
                )
            self._turn_seq += len(pairs)
            self._runtime.enqueue_remember_batch(
                self.kb,
                self._namespace_id,
                documents,
            )

        def on_session_end(self, messages: list[dict] | None = None) -> None:
            """Drain the runtime queue and warn on elevated failure rate."""
            if self._runtime is None:
                return

            if messages:
                # Final flush of dropped tail — same path as pre_compress.
                self.on_pre_compress(messages)

            self._runtime.drain(timeout=self._drain_timeout_s)

            try:
                rate = self._runtime.failure_rate_pct()
            except Exception as exc:  # noqa: BLE001 - runtime may not expose yet
                logger.debug("hermes failure_rate_pct probe failed: {}", exc)
                return
            if rate > self._failure_threshold_pct:
                errors = []
                try:
                    errors = self._runtime.last_errors(5)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("hermes last_errors probe failed: {}", exc)
                logger.warning(
                    "KhoraMemoryProvider background failure rate {:.2f}% > threshold {:.2f}%; recent errors: {}",
                    rate,
                    self._failure_threshold_pct,
                    errors,
                )

        def on_session_switch(self, new_session_id: str, **kwargs: Any) -> None:
            """Re-bind the provider to a different session under the same agent.

            The namespace is identity-scoped now (issue #1466), so switching
            sessions keeps the SAME khora namespace — only the ``session_id``
            column on subsequently-ingested turns changes. We carry the
            remembered ``agent_identity`` / ``user_id`` forward so a switch
            that omits them doesn't silently re-tenant the memory.
            """
            kwargs.setdefault("agent_identity", self._agent_identity)
            kwargs.setdefault("user_id", self._user_id)
            self.initialize(new_session_id, **kwargs)

        def on_turn_start(self, *_args: Any, **_kwargs: Any) -> None:
            """No-op — we drive turn bookkeeping through ``sync_turn`` instead."""
            return None

        def on_delegation(self, *_args: Any, **_kwargs: Any) -> None:
            """No-op — delegation hand-off has no khora-side bookkeeping today."""
            return None

        def shutdown(self) -> None:
            """Idempotent runtime shutdown — safe to call multiple times."""
            if self._runtime is None:
                return
            self._runtime.shutdown()

    return _KhoraMemoryProviderImpl()


__all__ = ["KhoraMemoryProvider"]
