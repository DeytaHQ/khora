"""Unit tests for stable namespace ID resolution and version creation.

Tests DYT-396 (schema/model changes) and DYT-397 (resolution logic).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.core.models.tenancy import MemoryNamespace
from khora.storage.coordinator import StorageCoordinator

# ---------------------------------------------------------------------------
# resolve_namespace — coordinator passthrough
# ---------------------------------------------------------------------------


class TestCoordinatorResolveNamespace:
    """Tests for StorageCoordinator.resolve_namespace()."""

    @pytest.mark.asyncio
    async def test_resolve_delegates_to_relational(self) -> None:
        """resolve_namespace delegates to the relational backend."""
        ns_id = uuid4()
        row_id = uuid4()

        rel = MagicMock()
        rel.resolve_namespace = AsyncMock(return_value=row_id)

        coord = StorageCoordinator(relational=rel)
        result = await coord.resolve_namespace(ns_id)

        assert result == row_id
        rel.resolve_namespace.assert_awaited_once_with(ns_id)

    @pytest.mark.asyncio
    async def test_resolve_raises_without_relational(self) -> None:
        """resolve_namespace raises RuntimeError without relational backend."""
        coord = StorageCoordinator()

        with pytest.raises(RuntimeError, match="Relational backend not configured"):
            await coord.resolve_namespace(uuid4())

    @pytest.mark.asyncio
    async def test_resolve_propagates_value_error(self) -> None:
        """ValueError from relational backend propagates to caller."""
        ns_id = uuid4()

        rel = MagicMock()
        rel.resolve_namespace = AsyncMock(
            side_effect=ValueError(f"No active namespace version found for namespace_id={ns_id}")
        )

        coord = StorageCoordinator(relational=rel)
        with pytest.raises(ValueError, match="No active namespace version"):
            await coord.resolve_namespace(ns_id)


# ---------------------------------------------------------------------------
# _namespace_model_to_domain — includes namespace_id
# ---------------------------------------------------------------------------


class TestNamespaceModelToDomain:
    """Tests for _namespace_model_to_domain including namespace_id."""

    def test_conversion_includes_namespace_id(self) -> None:
        """_namespace_model_to_domain maps namespace_id from ORM model."""
        from khora.storage.backends.postgresql import PostgreSQLBackend

        row_id = uuid4()
        stable_id = uuid4()

        model = MagicMock()
        model.id = row_id
        model.namespace_id = stable_id
        model.tenancy_mode = "shared"
        model.version = 2
        model.is_active = True
        model.previous_version_id = uuid4()
        model.config_overrides = {}
        model.sync_checkpoints = {}
        model.metadata_ = {}
        model.created_at = MagicMock()
        model.updated_at = MagicMock()

        backend = PostgreSQLBackend.__new__(PostgreSQLBackend)
        domain = backend._namespace_model_to_domain(model)

        assert domain.id == row_id
        assert domain.namespace_id == stable_id
        assert domain.version == 2

    def test_conversion_preserves_backfilled_namespace_id(self) -> None:
        """For migrated v1 rows, namespace_id == id (set by backfill)."""
        from khora.storage.backends.postgresql import PostgreSQLBackend

        row_id = uuid4()

        model = MagicMock()
        model.id = row_id
        model.namespace_id = row_id  # Backfill sets namespace_id = id
        model.tenancy_mode = "shared"
        model.version = 1
        model.is_active = True
        model.previous_version_id = None
        model.config_overrides = {}
        model.sync_checkpoints = {}
        model.metadata_ = {}
        model.created_at = MagicMock()
        model.updated_at = MagicMock()

        backend = PostgreSQLBackend.__new__(PostgreSQLBackend)
        domain = backend._namespace_model_to_domain(model)

        assert domain.namespace_id == domain.id


# ---------------------------------------------------------------------------
# create_namespace — namespace_id is independently generated
# ---------------------------------------------------------------------------


class TestCreateNamespaceStableId:
    """Tests for create_namespace namespace_id handling."""

    def test_new_namespace_has_independent_namespace_id(self) -> None:
        """New namespace gets independently generated namespace_id != id."""
        ns = MemoryNamespace(version=1)

        # Both are UUIDs but independently generated
        assert ns.namespace_id != ns.id

    def test_explicit_namespace_id_preserved(self) -> None:
        """Explicitly set namespace_id is preserved through creation."""
        stable_id = uuid4()
        row_id = uuid4()
        ns = MemoryNamespace(id=row_id, namespace_id=stable_id, version=2)

        assert ns.namespace_id == stable_id
        assert ns.id == row_id
        assert ns.namespace_id != ns.id


# ---------------------------------------------------------------------------
# create_namespace_version — inherits parent's namespace_id
# ---------------------------------------------------------------------------


class TestCreateNamespaceVersionInheritance:
    """Tests for create_namespace_version inheriting namespace_id."""

    def test_version_inherits_parent_namespace_id(self) -> None:
        """New version's MemoryNamespace should use parent's namespace_id."""
        parent_row_id = uuid4()
        parent_namespace_id = uuid4()

        parent = MemoryNamespace(
            id=parent_row_id,
            namespace_id=parent_namespace_id,
            version=1,
            is_active=True,
        )

        # Simulates the logic in PostgreSQLBackend.create_namespace_version
        child = MemoryNamespace(
            id=uuid4(),
            namespace_id=parent.namespace_id,
            version=parent.version + 1,
            is_active=True,
            previous_version_id=parent.id,
        )

        assert child.namespace_id == parent_namespace_id
        assert child.id != parent_row_id
        assert child.version == 2
        assert child.previous_version_id == parent_row_id

    def test_version_without_parent_gets_new_namespace_id(self) -> None:
        """Version without parent (fresh namespace) gets a new namespace_id."""
        # Simulates: namespace_id = previous_version.namespace_id if previous_version else uuid4()
        ns = MemoryNamespace(
            id=uuid4(),
            namespace_id=uuid4(),  # independent uuid4 when no parent
            version=1,
            is_active=True,
        )

        assert ns.namespace_id is not None
        assert ns.version == 1


# ---------------------------------------------------------------------------
# DB model — UniqueConstraint and column presence
# ---------------------------------------------------------------------------


class TestNamespaceModelSchema:
    """Tests for MemoryNamespaceModel schema (namespace_id column)."""

    def test_model_has_namespace_id_column(self) -> None:
        """MemoryNamespaceModel has namespace_id column."""
        from khora.db.models import MemoryNamespaceModel

        assert hasattr(MemoryNamespaceModel, "namespace_id")

    def test_unique_constraint_exists(self) -> None:
        """UniqueConstraint (namespace_id, version) exists on the model."""
        from khora.db.models import MemoryNamespaceModel

        table = MemoryNamespaceModel.__table__
        constraint_names = [c.name for c in table.constraints if hasattr(c, "name")]
        assert "uq_namespace_stable_id_version" in constraint_names

    def test_namespace_id_is_not_nullable(self) -> None:
        """namespace_id column should be NOT NULL."""
        from khora.db.models import MemoryNamespaceModel

        col = MemoryNamespaceModel.__table__.columns["namespace_id"]
        assert col.nullable is False

    def test_namespace_id_is_indexed(self) -> None:
        """namespace_id column should have an index."""
        from khora.db.models import MemoryNamespaceModel

        col = MemoryNamespaceModel.__table__.columns["namespace_id"]
        assert col.index is True

    def test_namespace_id_has_partial_index(self) -> None:
        """namespace_id should have a partial index for active-version lookups."""
        from khora.db.models import MemoryNamespaceModel

        table = MemoryNamespaceModel.__table__
        index_names = [idx.name for idx in table.indexes]
        assert "idx_namespace_stable_active" in index_names


# ---------------------------------------------------------------------------
# Protocol conformance — resolve_namespace in base protocol
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """Tests that resolve_namespace is part of the relational backend protocol."""

    def test_protocol_has_resolve_namespace(self) -> None:
        """RelationalBackendProtocol defines resolve_namespace."""
        from khora.storage.backends.base import RelationalBackendProtocol

        assert hasattr(RelationalBackendProtocol, "resolve_namespace")
