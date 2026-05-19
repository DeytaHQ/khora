"""Expertise definition storage.

Provides CRUD operations for storing and retrieving expertise configurations
in the database, allowing namespaces to have custom expertise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from loguru import logger
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

if TYPE_CHECKING:
    from khora.extraction.skills import ExpertiseConfig
    from khora.storage import StorageCoordinator


class ExpertiseStore:
    """Store for expertise definitions.

    Handles persistence of ExpertiseConfig objects to the database,
    allowing namespaces to have custom expertise definitions.

    Example usage:
        store = ExpertiseStore(storage_coordinator)

        # Save expertise for a namespace
        await store.save(namespace_id, expertise)

        # Get expertise by name
        expertise = await store.get(namespace_id, "saas_expert")

        # Get active expertise for namespace
        expertise = await store.get_active(namespace_id)

        # List all expertise for namespace
        all_expertise = await store.list(namespace_id)
    """

    def __init__(self, storage: StorageCoordinator) -> None:
        """Initialize the expertise store.

        Args:
            storage: StorageCoordinator for database access
        """
        self._storage = storage

    async def save(
        self,
        namespace_id: UUID,
        expertise: ExpertiseConfig,
        *,
        set_active: bool = True,
    ) -> UUID:
        """Save an expertise configuration for a namespace.

        Args:
            namespace_id: Namespace to save expertise for
            expertise: ExpertiseConfig to save
            set_active: Whether to set this as the active expertise

        Returns:
            ID of the saved expertise definition
        """
        from khora.db.models import ExpertiseDefinitionModel

        if not self._storage._relational:
            raise RuntimeError("Relational storage not configured")

        session: AsyncSession = await self._storage._relational._get_session().__aenter__()
        try:
            # Check if expertise with this name already exists
            result = await session.execute(
                select(ExpertiseDefinitionModel).where(
                    ExpertiseDefinitionModel.namespace_id == str(namespace_id),
                    ExpertiseDefinitionModel.name == expertise.name,
                )
            )
            existing = result.scalar_one_or_none()

            if existing:
                # Update existing
                existing.version = expertise.version
                existing.description = expertise.description
                existing.config = expertise.to_dict()
                existing.is_active = set_active
                await session.commit()
                logger.debug(f"Updated expertise '{expertise.name}' for namespace {namespace_id}")
                return UUID(existing.id)
            else:
                # Create new
                model = ExpertiseDefinitionModel(
                    namespace_id=str(namespace_id),
                    name=expertise.name,
                    version=expertise.version,
                    description=expertise.description,
                    config=expertise.to_dict(),
                    is_active=set_active,
                )
                session.add(model)
                await session.commit()
                await session.refresh(model)
                logger.debug(f"Created expertise '{expertise.name}' for namespace {namespace_id}")
                return UUID(model.id)

        finally:
            await session.close()

    async def get(
        self,
        namespace_id: UUID,
        name: str,
    ) -> ExpertiseConfig | None:
        """Get an expertise configuration by name.

        Args:
            namespace_id: Namespace to get expertise for
            name: Name of the expertise

        Returns:
            ExpertiseConfig or None if not found
        """
        from khora.db.models import ExpertiseDefinitionModel
        from khora.extraction.skills import ExpertiseConfig

        if not self._storage._relational:
            return None

        session: AsyncSession = await self._storage._relational._get_session().__aenter__()
        try:
            result = await session.execute(
                select(ExpertiseDefinitionModel).where(
                    ExpertiseDefinitionModel.namespace_id == str(namespace_id),
                    ExpertiseDefinitionModel.name == name,
                )
            )
            model = result.scalar_one_or_none()

            if model:
                return ExpertiseConfig.from_dict(model.config)
            return None

        finally:
            await session.close()

    async def get_by_id(self, expertise_id: UUID) -> ExpertiseConfig | None:
        """Get an expertise configuration by ID.

        Args:
            expertise_id: ID of the expertise definition

        Returns:
            ExpertiseConfig or None if not found
        """
        from khora.db.models import ExpertiseDefinitionModel
        from khora.extraction.skills import ExpertiseConfig

        if not self._storage._relational:
            return None

        session: AsyncSession = await self._storage._relational._get_session().__aenter__()
        try:
            result = await session.execute(
                select(ExpertiseDefinitionModel).where(
                    ExpertiseDefinitionModel.id == str(expertise_id),
                )
            )
            model = result.scalar_one_or_none()

            if model:
                return ExpertiseConfig.from_dict(model.config)
            return None

        finally:
            await session.close()

    async def get_active(self, namespace_id: UUID) -> ExpertiseConfig | None:
        """Get the active expertise configuration for a namespace.

        Args:
            namespace_id: Namespace to get expertise for

        Returns:
            Active ExpertiseConfig or None if none set
        """
        from khora.db.models import ExpertiseDefinitionModel
        from khora.extraction.skills import ExpertiseConfig

        if not self._storage._relational:
            return None

        session: AsyncSession = await self._storage._relational._get_session().__aenter__()
        try:
            result = await session.execute(
                select(ExpertiseDefinitionModel).where(
                    ExpertiseDefinitionModel.namespace_id == str(namespace_id),
                    ExpertiseDefinitionModel.is_active.is_(True),
                )
            )
            model = result.scalar_one_or_none()

            if model:
                return ExpertiseConfig.from_dict(model.config)
            return None

        finally:
            await session.close()

    async def get_by_namespace(self, namespace_id: UUID) -> ExpertiseConfig | None:
        """Get expertise for a namespace (alias for get_active).

        Args:
            namespace_id: Namespace to get expertise for

        Returns:
            Active ExpertiseConfig or None if none set
        """
        return await self.get_active(namespace_id)

    async def list(
        self,
        namespace_id: UUID,
        *,
        include_inactive: bool = False,
    ) -> list[dict[str, Any]]:
        """List all expertise definitions for a namespace.

        Args:
            namespace_id: Namespace to list expertise for
            include_inactive: Whether to include inactive expertise

        Returns:
            List of expertise info dicts (id, name, version, is_active)
        """
        from khora.db.models import ExpertiseDefinitionModel

        if not self._storage._relational:
            return []

        session: AsyncSession = await self._storage._relational._get_session().__aenter__()
        try:
            query = select(ExpertiseDefinitionModel).where(
                ExpertiseDefinitionModel.namespace_id == str(namespace_id),
            )
            if not include_inactive:
                query = query.where(ExpertiseDefinitionModel.is_active.is_(True))

            result = await session.execute(query)
            models = result.scalars().all()

            return [
                {
                    "id": m.id,
                    "name": m.name,
                    "version": m.version,
                    "description": m.description,
                    "is_active": m.is_active,
                    "created_at": m.created_at,
                    "updated_at": m.updated_at,
                }
                for m in models
            ]

        finally:
            await session.close()

    async def set_active(self, namespace_id: UUID, name: str) -> bool:
        """Set an expertise configuration as active.

        Deactivates any previously active expertise for the namespace.

        Args:
            namespace_id: Namespace
            name: Name of expertise to activate

        Returns:
            True if successful, False if expertise not found
        """
        from khora.db.models import ExpertiseDefinitionModel

        if not self._storage._relational:
            return False

        session: AsyncSession = await self._storage._relational._get_session().__aenter__()
        try:
            # Deactivate all
            await session.execute(
                update(ExpertiseDefinitionModel)
                .where(ExpertiseDefinitionModel.namespace_id == str(namespace_id))
                .values(is_active=False)
            )

            # Activate the specified one
            result = await session.execute(
                update(ExpertiseDefinitionModel)
                .where(
                    ExpertiseDefinitionModel.namespace_id == str(namespace_id),
                    ExpertiseDefinitionModel.name == name,
                )
                .values(is_active=True)
            )
            await session.commit()

            return result.rowcount > 0

        finally:
            await session.close()

    async def delete(self, namespace_id: UUID, name: str) -> bool:
        """Delete an expertise configuration.

        Args:
            namespace_id: Namespace
            name: Name of expertise to delete

        Returns:
            True if deleted, False if not found
        """
        from khora.db.models import ExpertiseDefinitionModel

        if not self._storage._relational:
            return False

        session: AsyncSession = await self._storage._relational._get_session().__aenter__()
        try:
            result = await session.execute(
                select(ExpertiseDefinitionModel).where(
                    ExpertiseDefinitionModel.namespace_id == str(namespace_id),
                    ExpertiseDefinitionModel.name == name,
                )
            )
            model = result.scalar_one_or_none()

            if model:
                await session.delete(model)
                await session.commit()
                logger.debug(f"Deleted expertise '{name}' from namespace {namespace_id}")
                return True

            return False

        finally:
            await session.close()
