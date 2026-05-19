"""Pin the empty-string-to-None coercion in document-row mappers.

The storage row → ``Document`` mappers see legacy rows where the migrated
columns (``source``, ``content_type``, ``title``, ``author``, ``language``,
``checksum``) still carry ``""``. The flattened ``Document`` domain model
now types these as ``str | None``; the mappers MUST coerce the legacy
``""`` back to ``None`` so downstream consumers can rely on truthy checks
and don't confuse "explicitly unset" with "empty string".

``source_type`` keeps its NOT NULL contract and defaults to ``"library"``
post-migration — including when a legacy row has ``""``.

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


def _make_document_model_with_legacy_empty_strings() -> DocumentModel:
    """Build a ``DocumentModel`` where every nullable text column is ``""``.

    Mirrors the pre-migration row shape that still lives in long-lived
    databases. The mapper must coerce these legacy empty strings back to
    ``None`` at the DTO boundary.
    """
    return DocumentModel(
        id=uuid4(),
        namespace_id=uuid4(),
        content="document body",
        status="completed",
        source="",
        source_type="",
        content_type="",
        title="",
        author="",
        language="",
        checksum="",
        size_bytes=0,
        metadata_={},
        chunk_count=0,
        entity_count=0,
        relationship_count=0,
    )


def _make_document_model_with_null_nullable_fields() -> DocumentModel:
    """Build a ``DocumentModel`` with every newly-nullable column set to ``None``."""
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
    def test_legacy_empty_strings_coerce_to_none_on_read(self) -> None:
        from khora.storage.backends.postgresql import PostgreSQLBackend

        backend = PostgreSQLBackend.__new__(PostgreSQLBackend)
        model = _make_document_model_with_legacy_empty_strings()

        doc = backend._document_model_to_domain(model)

        # Legacy "" must come back as None on the flat fields.
        assert doc.source is None
        assert doc.content_type is None
        assert doc.title is None
        assert doc.author is None
        assert doc.language is None
        assert doc.checksum is None
        # source_type keeps NOT NULL contract: "" → "library".
        assert doc.source_type == "library"
        assert doc.content == "document body"
        assert doc.status == DocumentStatus.COMPLETED

    def test_null_columns_remain_none_on_read(self) -> None:
        from khora.storage.backends.postgresql import PostgreSQLBackend

        backend = PostgreSQLBackend.__new__(PostgreSQLBackend)
        model = _make_document_model_with_null_nullable_fields()

        doc = backend._document_model_to_domain(model)

        assert doc.source is None
        assert doc.content_type is None
        assert doc.title is None
        assert doc.author is None
        assert doc.language is None
        assert doc.checksum is None
        assert doc.source_type == "library"

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

        assert doc.source == "nango://linear/issues"
        assert doc.source_type == "external"
        assert doc.content_type == "text/plain"
        assert doc.title == "Some title"
        assert doc.author == "alice"
        assert doc.language == "fr"
        assert doc.checksum == "deadbeef"
        assert doc.metadata == {"custom": "kv"}


@pytest.mark.unit
class TestSqliteLanceBackendMapperCoercion:
    def test_legacy_empty_strings_coerce_to_none_on_read(self) -> None:
        from khora.storage.backends.sqlite_lance.relational import SQLiteLanceRelationalAdapter

        adapter = SQLiteLanceRelationalAdapter.__new__(SQLiteLanceRelationalAdapter)
        model = _make_document_model_with_legacy_empty_strings()

        doc = adapter._document_model_to_domain(model)

        assert doc.source is None
        assert doc.content_type is None
        assert doc.title is None
        assert doc.author is None
        assert doc.language is None
        assert doc.checksum is None
        assert doc.source_type == "library"

    def test_metadata_none_coerces_to_empty_dict(self) -> None:
        """SQLite mapper coerces ``metadata_`` of ``None`` to ``{}`` to preserve dict contract."""
        from khora.storage.backends.sqlite_lance.relational import SQLiteLanceRelationalAdapter

        adapter = SQLiteLanceRelationalAdapter.__new__(SQLiteLanceRelationalAdapter)
        model = _make_document_model_with_null_nullable_fields()
        # ORM default is dict for metadata_, but a manually-built model can carry None.
        model.metadata_ = None  # type: ignore[assignment]

        doc = adapter._document_model_to_domain(model)
        assert doc.metadata == {}


@pytest.mark.unit
class TestSurrealDBBackendMapperCoercion:
    """SurrealDB mapper takes a row ``dict``, not a ``DocumentModel``."""

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

        assert doc.source is None
        # source_type keeps NOT NULL contract — coerced to "library".
        assert doc.source_type == "library"
        assert doc.content_type is None
        assert doc.title is None
        assert doc.author is None
        assert doc.checksum is None
        assert doc.language is None
        # metadata defaults to {} when row's metadata_ is None.
        assert doc.metadata == {}


@pytest.mark.unit
class TestDocumentSourceTimestampRoundTrip:
    """``source_timestamp`` previously missed at least two converters — pin it."""

    def test_source_timestamp_roundtrips_postgresql(self) -> None:
        from datetime import UTC, datetime

        from khora.storage.backends.postgresql import PostgreSQLBackend

        backend = PostgreSQLBackend.__new__(PostgreSQLBackend)
        when = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        model = DocumentModel(
            id=uuid4(),
            namespace_id=uuid4(),
            content="body",
            status="completed",
            source_type="library",
            source_timestamp=when,
            size_bytes=0,
            metadata_={},
            chunk_count=0,
            entity_count=0,
            relationship_count=0,
        )
        doc = backend._document_model_to_domain(model)
        assert doc.source_timestamp == when

    def test_source_timestamp_roundtrips_sqlite_lance(self) -> None:
        from datetime import UTC, datetime

        from khora.storage.backends.sqlite_lance.relational import SQLiteLanceRelationalAdapter

        adapter = SQLiteLanceRelationalAdapter.__new__(SQLiteLanceRelationalAdapter)
        when = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        model = DocumentModel(
            id=uuid4(),
            namespace_id=uuid4(),
            content="body",
            status="completed",
            source_type="library",
            source_timestamp=when,
            size_bytes=0,
            metadata_={},
            chunk_count=0,
            entity_count=0,
            relationship_count=0,
        )
        doc = adapter._document_model_to_domain(model)
        assert doc.source_timestamp == when


@pytest.mark.unit
class TestDocumentSourceNameUrlRoundTrip:
    """``source_name`` and ``source_url`` were added in the same migration."""

    def test_source_name_and_url_roundtrip_postgresql(self) -> None:
        from khora.storage.backends.postgresql import PostgreSQLBackend

        backend = PostgreSQLBackend.__new__(PostgreSQLBackend)
        model = DocumentModel(
            id=uuid4(),
            namespace_id=uuid4(),
            content="body",
            status="completed",
            source_type="library",
            source_name="nango_gmail",
            source_url="https://example.com/x",
            size_bytes=0,
            metadata_={},
            chunk_count=0,
            entity_count=0,
            relationship_count=0,
        )
        doc = backend._document_model_to_domain(model)
        assert doc.source_name == "nango_gmail"
        assert doc.source_url == "https://example.com/x"

    def test_source_name_and_url_roundtrip_sqlite_lance(self) -> None:
        from khora.storage.backends.sqlite_lance.relational import SQLiteLanceRelationalAdapter

        adapter = SQLiteLanceRelationalAdapter.__new__(SQLiteLanceRelationalAdapter)
        model = DocumentModel(
            id=uuid4(),
            namespace_id=uuid4(),
            content="body",
            status="completed",
            source_type="library",
            source_name="nango_gmail",
            source_url="https://example.com/x",
            size_bytes=0,
            metadata_={},
            chunk_count=0,
            entity_count=0,
            relationship_count=0,
        )
        doc = adapter._document_model_to_domain(model)
        assert doc.source_name == "nango_gmail"
        assert doc.source_url == "https://example.com/x"
