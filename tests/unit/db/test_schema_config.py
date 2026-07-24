"""Unit tests for the migration schema-config helpers (#1260).

These size the pgvector columns / HNSW indexes from the injected embedding
dimension. They read ``alembic.context.config.attributes`` defensively, so a
missing/unconfigured context falls back to the historical 1536 default.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from khora.db.migrations import _schema_config as sc


def _patch_attributes(monkeypatch: pytest.MonkeyPatch, attributes: dict | None) -> None:
    """Point the helpers at a fake ``context.config.attributes`` mapping.

    ``attributes=None`` simulates an unconfigured context (attribute access
    raises), which must fall back to defaults.
    """
    if attributes is None:

        class _Unconfigured:
            @property
            def config(self):  # noqa: ANN202
                raise AttributeError("no config")

        monkeypatch.setattr(sc, "context", _Unconfigured())
    else:
        fake = SimpleNamespace(config=SimpleNamespace(attributes=attributes))
        monkeypatch.setattr(sc, "context", fake)


@pytest.mark.unit
def test_defaults_when_context_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_attributes(monkeypatch, None)
    assert sc.configured_embedding_dimension() == 1536
    assert sc.configured_use_halfvec() is True
    assert sc.full_precision_hnsw_supported() is True


@pytest.mark.unit
def test_defaults_when_attributes_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_attributes(monkeypatch, {})
    assert sc.configured_embedding_dimension() == 1536
    assert sc.configured_use_halfvec() is True


@pytest.mark.unit
def test_reads_injected_dimension(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_attributes(monkeypatch, {"embedding_dimension": 3072, "use_halfvec": True})
    assert sc.configured_embedding_dimension() == 3072
    assert sc.configured_use_halfvec() is True
    # 3072 > 2000 => full-precision vector HNSW is not buildable.
    assert sc.full_precision_hnsw_supported() is False


@pytest.mark.unit
def test_full_precision_supported_at_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_attributes(monkeypatch, {"embedding_dimension": sc.VECTOR_HNSW_MAX_DIM})
    assert sc.full_precision_hnsw_supported() is True
    _patch_attributes(monkeypatch, {"embedding_dimension": sc.VECTOR_HNSW_MAX_DIM + 1})
    assert sc.full_precision_hnsw_supported() is False


@pytest.mark.unit
def test_use_halfvec_false_is_read(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_attributes(monkeypatch, {"use_halfvec": False})
    assert sc.configured_use_halfvec() is False


@pytest.mark.unit
def test_none_valued_attributes_fall_back_to_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_attributes(monkeypatch, {"embedding_dimension": None, "use_halfvec": None})
    assert sc.configured_embedding_dimension() == 1536
    assert sc.configured_use_halfvec() is True
