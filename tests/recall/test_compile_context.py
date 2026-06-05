"""Unit tests for ``CompileContext`` / ``SchemaCapabilities`` (Layer 4 seam).

Pins: the field set + defaults, frozen/slotted shape, the ``DEFAULTS``
capability instance, ``CompileError`` is a ``KhoraError``, the re-exported
``RecallFilterUnsupportedError`` is the model's (not a redefinition), and the
internal-only rule that these are absent from ``khora.__all__``.

A first cut — QA expands coverage after this.
"""

from __future__ import annotations

import dataclasses

import pytest

from khora.exceptions import KhoraError
from khora.filter.context import (
    CompileContext,
    CompileError,
    RecallFilterUnsupportedError,
    SchemaCapabilities,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# CompileContext field set + defaults.
# ---------------------------------------------------------------------------


def test_context_required_field_only_backend_target() -> None:
    ctx = CompileContext(backend_target="khora_chunks")
    assert ctx.backend_target == "khora_chunks"


def test_context_defaults() -> None:
    ctx = CompileContext(backend_target="documents")
    assert ctx.table_alias is None
    assert ctx.param_namespace == "f"
    assert ctx.field_mapping is None
    assert ctx.schema_capabilities is SchemaCapabilities.DEFAULTS
    assert ctx.on_unsupported == "raise"


def test_context_field_names_match_spec() -> None:
    names = {f.name for f in dataclasses.fields(CompileContext)}
    assert names == {
        "backend_target",
        "table_alias",
        "param_namespace",
        "field_mapping",
        "schema_capabilities",
        "on_unsupported",
    }


def test_context_accepts_overrides() -> None:
    caps = SchemaCapabilities(jsonb_path_query=True, full_text=True, native_map_metadata=True)
    ctx = CompileContext(
        backend_target="Chunk",
        table_alias="c",
        param_namespace="g",
        field_mapping={"source_name": "src_name"},
        schema_capabilities=caps,
        on_unsupported="split",
    )
    assert ctx.table_alias == "c"
    assert ctx.param_namespace == "g"
    assert ctx.field_mapping == {"source_name": "src_name"}
    assert ctx.schema_capabilities is caps
    assert ctx.on_unsupported == "split"


def test_context_on_unsupported_accepts_raise_and_split() -> None:
    # AC4: on_unsupported is the two-value policy literal — both arms construct.
    assert CompileContext(backend_target="t", on_unsupported="raise").on_unsupported == "raise"
    assert CompileContext(backend_target="t", on_unsupported="split").on_unsupported == "split"


def test_context_field_mapping_carries_system_key_to_column_map() -> None:
    # AC4: field_mapping lets one compiler serve a different schema. It is
    # carried verbatim (a Mapping), defaulting to None (identity mapping).
    mapping = {"source_name": "src_name", "title": "doc_title"}
    ctx = CompileContext(backend_target="chunks", field_mapping=mapping)
    assert ctx.field_mapping == mapping
    assert CompileContext(backend_target="chunks").field_mapping is None


def test_context_field_mapping_accepts_any_mapping_not_just_dict() -> None:
    # The field is typed Mapping[str, str] | None — an immutable MappingProxyType
    # (a natural choice for a read-only schema map) is carried verbatim.
    from types import MappingProxyType

    mp = MappingProxyType({"source_name": "src_name"})
    ctx = CompileContext(backend_target="chunks", field_mapping=mp)
    assert ctx.field_mapping is mp
    assert ctx.field_mapping["source_name"] == "src_name"


def test_context_supports_dataclasses_replace_for_derived_variants() -> None:
    # The context is frozen, but an engine can derive a variant via
    # dataclasses.replace (e.g. flip on_unsupported for a split pass) without
    # mutating the original.
    base = CompileContext(backend_target="khora_chunks", param_namespace="f")
    split = dataclasses.replace(base, on_unsupported="split")
    assert split.on_unsupported == "split"
    assert split.backend_target == "khora_chunks"
    assert split.param_namespace == "f"
    # Original is untouched.
    assert base.on_unsupported == "raise"


def test_context_with_dict_field_mapping_is_not_hashable() -> None:
    # CompileContext is frozen but carries a mutable dict field_mapping, so a
    # context built with a dict mapping is not hashable (dict is unhashable). This
    # documents that a CompileContext must not be used as a dict/set key — the
    # cache key is the canonical_hash of the AST, not the context.
    ctx = CompileContext(backend_target="khora_chunks", field_mapping={"a": "b"})
    with pytest.raises(TypeError):
        hash(ctx)


def test_context_is_frozen() -> None:
    ctx = CompileContext(backend_target="khora_chunks")
    with pytest.raises((AttributeError, TypeError)):
        ctx.backend_target = "documents"  # type: ignore[misc]


def test_context_is_slotted() -> None:
    ctx = CompileContext(backend_target="khora_chunks")
    assert not hasattr(ctx, "__dict__")


# ---------------------------------------------------------------------------
# SchemaCapabilities — 3 flags + DEFAULTS.
# ---------------------------------------------------------------------------


def test_schema_capabilities_flag_defaults_are_false() -> None:
    caps = SchemaCapabilities()
    assert caps.jsonb_path_query is False
    assert caps.full_text is False
    assert caps.native_map_metadata is False


def test_schema_capabilities_defaults_instance_is_all_false() -> None:
    assert isinstance(SchemaCapabilities.DEFAULTS, SchemaCapabilities)
    assert SchemaCapabilities.DEFAULTS == SchemaCapabilities()


def test_schema_capabilities_field_names() -> None:
    names = {f.name for f in dataclasses.fields(SchemaCapabilities)}
    assert names == {"jsonb_path_query", "full_text", "native_map_metadata"}


def test_schema_capabilities_is_frozen() -> None:
    caps = SchemaCapabilities()
    with pytest.raises((AttributeError, TypeError)):
        caps.full_text = True  # type: ignore[misc]


def test_schema_capabilities_is_slotted() -> None:
    assert not hasattr(SchemaCapabilities(), "__dict__")


def test_context_default_schema_capabilities_is_the_shared_defaults_instance() -> None:
    # AC4: the default is SchemaCapabilities.DEFAULTS — the same shared instance,
    # not a freshly-constructed equal one (callers need not construct one).
    a = CompileContext(backend_target="a")
    b = CompileContext(backend_target="b")
    assert a.schema_capabilities is SchemaCapabilities.DEFAULTS
    assert b.schema_capabilities is SchemaCapabilities.DEFAULTS
    assert a.schema_capabilities is b.schema_capabilities


def test_schema_capabilities_value_equality_and_hashing() -> None:
    # Frozen dataclass: equal-by-value instances compare equal AND hash equal, so
    # a richer-capability instance can be used as a dict/set key by an engine.
    assert SchemaCapabilities(jsonb_path_query=True) == SchemaCapabilities(jsonb_path_query=True)
    assert hash(SchemaCapabilities(jsonb_path_query=True)) == hash(SchemaCapabilities(jsonb_path_query=True))
    assert SchemaCapabilities(full_text=True) != SchemaCapabilities()
    # DEFAULTS equals a freshly-constructed all-False instance but the canonical
    # default is the shared singleton (asserted elsewhere).
    assert SchemaCapabilities.DEFAULTS == SchemaCapabilities()


def test_schema_capabilities_partial_flags() -> None:
    # A realistic backend declares a subset of capabilities (e.g. a JSONB-capable
    # store without native full-text); the other flags stay at their False default.
    caps = SchemaCapabilities(jsonb_path_query=True)
    assert caps.jsonb_path_query is True
    assert caps.full_text is False
    assert caps.native_map_metadata is False


# ---------------------------------------------------------------------------
# Error types.
# ---------------------------------------------------------------------------


def test_compile_error_is_khora_error() -> None:
    assert issubclass(CompileError, KhoraError)


def test_unsupported_error_is_the_models_not_a_redefinition() -> None:
    # context.py RE-EXPORTS the model's error; it must be the same class object,
    # so callers catch one type regardless of import path.
    from khora.filter.model import RecallFilterUnsupportedError as ModelErr

    assert RecallFilterUnsupportedError is ModelErr


def test_unsupported_error_is_distinct_from_compile_error() -> None:
    # A capability gap (RecallFilterUnsupportedError) is not an internal compiler
    # fault (CompileError); they are distinct types.
    assert RecallFilterUnsupportedError is not CompileError
    assert not issubclass(RecallFilterUnsupportedError, CompileError)


def test_compile_error_carries_message_and_catches_as_khora_error() -> None:
    # CompileError is a plain KhoraError subclass carrying a human message; it
    # must be catchable via the KhoraError base (callers may catch broadly).
    with pytest.raises(KhoraError) as exc_info:
        raise CompileError("unreachable AST branch")
    assert "unreachable AST branch" in str(exc_info.value)


def test_unsupported_error_carries_path_and_reason() -> None:
    # RecallFilterUnsupportedError carries structured path + reason (the backend
    # could not express the predicate at that path), surfaced in its message.
    err = RecallFilterUnsupportedError("/metadata.k/$regex", "backend lacks regex predicate")
    assert err.path == "/metadata.k/$regex"
    assert err.reason == "backend lacks regex predicate"
    assert "/metadata.k/$regex" in str(err)
    assert "backend lacks regex predicate" in str(err)


# ---------------------------------------------------------------------------
# Internal-only — absent from khora.__all__.
# ---------------------------------------------------------------------------


def test_context_absent_from_khora_top_level_all() -> None:
    import khora

    for name in ("CompileContext", "SchemaCapabilities", "CompileError"):
        assert name not in khora.__all__


def test_context_reachable_from_khora_filter() -> None:
    import khora.filter as f

    assert f.CompileContext is CompileContext
    assert f.SchemaCapabilities is SchemaCapabilities


def test_context_names_not_in_khora_filter_public_all() -> None:
    import khora.filter as f

    for name in ("CompileContext", "SchemaCapabilities", "CompileError"):
        assert name not in f.__all__
