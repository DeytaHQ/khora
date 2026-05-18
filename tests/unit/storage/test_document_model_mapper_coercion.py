"""Pin the NULL-to-empty-string DTO coercion in document-row mappers.

Migration 037 flips six ``documents`` columns — ``source``,
``content_type``, ``title``, ``author``, ``language``, ``checksum`` —
from ``NOT NULL DEFAULT ''`` to nullable. The ``DocumentMetadata`` DTO
surface is part of the stable public API and still types those fields
as ``str``. The DTO widening (allowing ``str | None``) lands in a
follow-up; until then, every backend's
row → ``Document`` mapper MUST coerce ``None`` to ``""`` at the DTO
boundary so downstream consumers (khora-cli, khora-explorer,
integrations) never see ``None`` where they expect ``str``.

These tests pin the coercion for every backend that shipped it in the
migration-037 diff. Removing the ``or ""`` from any mapper will break
a test here — that's the point. The accompanying DTO widening will
update both the mapper and these tests in one PR.

SurrealDB language divergence: the SurrealDB mapper defaults
``language`` to ``"en"`` (literal "en") rather than ``""``. This is a
deliberate historical divergence — the SurrealDB ``language`` column
schema defaults to ``"en"`` and downstream consumers of the Surreal
backend expect that. The test for surrealdb asserts ``"en"`` rather
than ``""`` for language, while still asserting ``""`` for the other
five nullable fields.

The mapper helpers are stateless instance methods — they don't touch
``self``. Tests instantiate the adapter classes via ``Klass.__new__``
to bypass real backend wiring (DB connections, engine creation, etc.).
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from khora.core.models.document import DocumentStatus
from khora.db.models import DocumentModel


def _make_document_model_with_null_nullable_fields() -> DocumentModel:
    """Build a ``DocumentModel`` with every newly-nullable column set to ``None``.

    Mirrors the post-migration-037 state where a row's ``source``,
    ``content_type``, ``title``, ``author``, ``language``, ``checksum``
    may be ``NULL`` (because empty-string rows were normalized away by
    the migration, or because callers passed ``None`` going forward).
    """
    return DocumentModel(
        id=uuid4(),
        namespace_id=uuid4(),
        content="document body",
        status="completed",
        source=None,
        source_type="library",
        content_type=None,
        title=None,
        author=None,
        language=None,
        checksum=None,
        size_bytes=0,
        metadata_={},
        chunk_count=0,
        entity_count=0,
        relationship_count=0,
    )


@pytest.mark.unit
class TestPostgreSQLBackendMapperCoercion:
    def test_null_columns_coerce_to_empty_string_on_read(self) -> None:
        from khora.storage.backends.postgresql import PostgreSQLBackend

        backend = PostgreSQLBackend.__new__(PostgreSQLBackend)
        model = _make_document_model_with_null_nullable_fields()

        doc = backend._document_model_to_domain(model)

        # All six nullable columns coerce to "" (the DocumentMetadata str contract).
        assert doc.metadata.source == ""
        assert doc.metadata.content_type == ""
        assert doc.metadata.title == ""
        assert doc.metadata.author == ""
        assert doc.metadata.language == ""
        assert doc.metadata.checksum == ""
        # source_type is NOT NULL post-migration (default 'library') — passes through unchanged.
        assert doc.metadata.source_type == "library"
        # Document body is plumbed through directly.
        assert doc.content == "document body"
        assert doc.status == DocumentStatus.COMPLETED

    def test_non_null_columns_pass_through_unchanged(self) -> None:
        """A row with all fields populated reads back with those exact values."""
        from khora.storage.backends.postgresql import PostgreSQLBackend

        backend = PostgreSQLBackend.__new__(PostgreSQLBackend)
        model = DocumentModel(
            id=uuid4(),
            namespace_id=uuid4(),
            content="body",
            status="completed",
            source="nango://linear/issues",
            source_type="external",
            content_type="text/plain",
            title="Some title",
            author="alice",
            language="fr",
            checksum="deadbeef",
            size_bytes=42,
            metadata_={"custom": "kv"},
            chunk_count=1,
            entity_count=2,
            relationship_count=3,
        )

        doc = backend._document_model_to_domain(model)

        assert doc.metadata.source == "nango://linear/issues"
        assert doc.metadata.source_type == "external"
        assert doc.metadata.content_type == "text/plain"
        assert doc.metadata.title == "Some title"
        assert doc.metadata.author == "alice"
        assert doc.metadata.language == "fr"
        assert doc.metadata.checksum == "deadbeef"


@pytest.mark.unit
class TestSqliteLanceBackendMapperCoercion:
    def test_null_columns_coerce_to_empty_string_on_read(self) -> None:
        from khora.storage.backends.sqlite_lance.relational import SQLiteLanceRelationalAdapter

        adapter = SQLiteLanceRelationalAdapter.__new__(SQLiteLanceRelationalAdapter)
        model = _make_document_model_with_null_nullable_fields()

        doc = adapter._document_model_to_domain(model)

        assert doc.metadata.source == ""
        assert doc.metadata.content_type == ""
        assert doc.metadata.title == ""
        assert doc.metadata.author == ""
        assert doc.metadata.language == ""
        assert doc.metadata.checksum == ""
        assert doc.metadata.source_type == "library"

    def test_metadata_none_coerces_to_empty_dict(self) -> None:
        """SQLite mapper coerces ``metadata_`` of ``None`` to ``{}`` to preserve dict contract."""
        from khora.storage.backends.sqlite_lance.relational import SQLiteLanceRelationalAdapter

        adapter = SQLiteLanceRelationalAdapter.__new__(SQLiteLanceRelationalAdapter)
        model = _make_document_model_with_null_nullable_fields()
        # ORM default is dict for metadata_, but a manually-built model can carry None.
        # Force the edge case the ``or {}`` guards against.
        model.metadata_ = None  # type: ignore[assignment]

        doc = adapter._document_model_to_domain(model)
        assert doc.metadata.custom == {}


@pytest.mark.unit
class TestSurrealDBBackendMapperCoercion:
    """SurrealDB mapper takes a row ``dict``, not a ``DocumentModel``.

    Language divergence: ``language`` defaults to ``"en"`` (literal "en"),
    matching the SurrealDB schema's ``DEFINE FIELD language ... DEFAULT 'en'``.
    Other nullable fields coerce to ``""`` like the SQL backends.
    """

    def _null_row(self) -> dict[str, Any]:
        return {
            "id": str(uuid4()),
            "namespace_id": str(uuid4()),
            "content": "body",
            "status": "completed",
            "source": None,
            "source_type": None,
            "content_type": None,
            "title": None,
            "author": None,
            "language": None,
            "checksum": None,
            "size_bytes": 0,
            "chunk_count": 0,
            "entity_count": 0,
            "relationship_count": 0,
            "metadata_": None,
        }

    def test_null_fields_coerce_per_backend_contract(self) -> None:
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        adapter = SurrealDBRelationalAdapter.__new__(SurrealDBRelationalAdapter)
        doc = adapter._row_to_document(self._null_row())

        # Five of six nullable fields coerce to "".
        assert doc.metadata.source == ""
        assert doc.metadata.source_type == ""
        assert doc.metadata.content_type == ""
        assert doc.metadata.title == ""
        assert doc.metadata.author == ""
        assert doc.metadata.checksum == ""
        # SurrealDB schema declares language with DEFAULT 'en' — the mapper
        # mirrors that with ``or "en"`` rather than ``or ""``. Documented in
        # the migration's "SurrealDB language divergence" carve-out.
        assert doc.metadata.language == "en"
        # custom defaults to {} when row's metadata_ is None.
        assert doc.metadata.custom == {}
