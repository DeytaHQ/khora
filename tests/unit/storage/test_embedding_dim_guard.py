"""Unit tests for the process-wide single-embedding-dimension guard (#1260).

The pgvector embedding column type is a process-global; binding two different
dimensions in one process would silently corrupt writes, so a conflicting
concurrent bind must fail loudly (a single dimension per deployment is the
supported model). Sequential rebinding after a disconnect is allowed.
"""

from __future__ import annotations

import pytest

from khora.storage.backends import pgvector as backend_mod
from khora.storage.temporal import pgvector as temporal_mod


@pytest.fixture(autouse=True)
def _reset_slots():
    backend_mod._bound_embedding_dim = None
    temporal_mod._bound_embedding_dim = None
    yield
    backend_mod._bound_embedding_dim = None
    temporal_mod._bound_embedding_dim = None


@pytest.mark.unit
@pytest.mark.parametrize("mod", [backend_mod, temporal_mod])
def test_same_dimension_rebinds_are_idempotent(mod):
    mod._bind_process_embedding_dim(1536)
    mod._bind_process_embedding_dim(1536)  # no raise
    assert mod._bound_embedding_dim == 1536


@pytest.mark.unit
@pytest.mark.parametrize("mod", [backend_mod, temporal_mod])
def test_conflicting_dimension_raises(mod):
    mod._bind_process_embedding_dim(1536)
    with pytest.raises(RuntimeError, match="already bound to 1536"):
        mod._bind_process_embedding_dim(3072)


@pytest.mark.unit
@pytest.mark.parametrize("mod", [backend_mod, temporal_mod])
def test_sequential_rebind_after_release_is_allowed(mod):
    mod._bind_process_embedding_dim(1536)
    mod._bound_embedding_dim = None  # simulates disconnect() releasing the slot
    mod._bind_process_embedding_dim(3072)  # no raise
    assert mod._bound_embedding_dim == 3072


@pytest.mark.unit
def test_guards_are_independent_per_table_group():
    # The two module slots track different tables; binding one does not
    # constrain the other.
    backend_mod._bind_process_embedding_dim(1536)
    temporal_mod._bind_process_embedding_dim(3072)
    assert backend_mod._bound_embedding_dim == 1536
    assert temporal_mod._bound_embedding_dim == 3072
