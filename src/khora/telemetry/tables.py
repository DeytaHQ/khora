"""SQLAlchemy table definitions for telemetry events.

Uses Core tables (not ORM) for lightweight batch inserts.
Tables are auto-created via ``metadata.create_all`` -- no Alembic needed.
"""

from __future__ import annotations

import sqlalchemy as sa

metadata = sa.MetaData()

llm_events = sa.Table(
    "llm_events",
    metadata,
    sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
    sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
    sa.Column("service_name", sa.String(64), nullable=False),
    sa.Column("operation", sa.String(64), nullable=False, default=""),
    sa.Column("model", sa.String(128), nullable=False, default=""),
    sa.Column("prompt_tokens", sa.Integer, nullable=False, default=0),
    sa.Column("completion_tokens", sa.Integer, nullable=False, default=0),
    sa.Column("total_tokens", sa.Integer, nullable=False, default=0),
    sa.Column("cost_usd", sa.Float, nullable=False, default=0.0),
    sa.Column("latency_ms", sa.Float, nullable=False, default=0.0),
    sa.Column("status", sa.String(16), nullable=False, default="success"),
    sa.Column("error_message", sa.Text, nullable=True),
    sa.Column("namespace_id", sa.Uuid, nullable=True),
    sa.Column("metadata", sa.JSON, nullable=True),
)

storage_events = sa.Table(
    "storage_events",
    metadata,
    sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
    sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
    sa.Column("service_name", sa.String(64), nullable=False),
    sa.Column("backend", sa.String(32), nullable=False, default=""),
    sa.Column("operation", sa.String(64), nullable=False, default=""),
    sa.Column("latency_ms", sa.Float, nullable=False, default=0.0),
    sa.Column("record_count", sa.Integer, nullable=False, default=0),
    sa.Column("status", sa.String(16), nullable=False, default="success"),
    sa.Column("error_message", sa.Text, nullable=True),
    sa.Column("namespace_id", sa.Uuid, nullable=True),
    sa.Column("metadata", sa.JSON, nullable=True),
)

pipeline_events = sa.Table(
    "pipeline_events",
    metadata,
    sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
    sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
    sa.Column("service_name", sa.String(64), nullable=False),
    sa.Column("pipeline", sa.String(64), nullable=False, default=""),
    sa.Column("stage", sa.String(64), nullable=False, default=""),
    sa.Column("run_id", sa.Uuid, nullable=True),
    sa.Column("latency_ms", sa.Float, nullable=False, default=0.0),
    sa.Column("record_count", sa.Integer, nullable=False, default=0),
    sa.Column("status", sa.String(16), nullable=False, default="success"),
    sa.Column("error_message", sa.Text, nullable=True),
    sa.Column("namespace_id", sa.Uuid, nullable=True),
    sa.Column("metadata", sa.JSON, nullable=True),
)
