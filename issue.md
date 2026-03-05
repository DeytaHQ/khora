# Bug: SQLAlchemy sends uppercase enum name instead of lowercase value for tenancy_mode

## Problem

Inserting an organization into PostgreSQL fails with:

```
asyncpg.exceptions.InvalidTextRepresentationError: invalid input value for enum tenancy_mode: "SHARED"
```

The `tenancy_mode` PostgreSQL enum is defined in the initial migration (`alembic/versions/000_initial_schema.py`) with **lowercase** values:

```sql
CREATE TYPE tenancy_mode AS ENUM ('shared', 'isolated')
```

However, the SQLAlchemy `Enum` column in `OrganizationModel` (`src/khora/db/models.py:62-64`) uses the default behavior which sends the Python enum **name** (`SHARED`) instead of the enum **value** (`shared`):

```python
tenancy_mode: Mapped[str] = mapped_column(
    Enum(TenancyMode, name="tenancy_mode", create_constraint=True),
    default=TenancyMode.SHARED,
)
```

The Python enum in `src/khora/core/models/tenancy.py` defines:

```python
class TenancyMode(str, Enum):
    SHARED = "shared"
    ISOLATED = "isolated"
```

By default, SQLAlchemy's `Enum()` type uses `.name` (uppercase) for persistence. Since the DB enum only accepts lowercase values, the INSERT fails.

## Fix

Add `values_callable` to the `Enum()` column definition so SQLAlchemy uses the enum `.value` (lowercase) instead of `.name` (uppercase):

```python
tenancy_mode: Mapped[str] = mapped_column(
    Enum(TenancyMode, name="tenancy_mode", create_constraint=True, values_callable=lambda e: [m.value for m in e]),
    default=TenancyMode.SHARED,
)
```

### File changed

- `src/khora/db/models.py` line 63

## Reproduction

```bash
# From the genesis repo:
uv run genesis create \
    -c config/vectorcypher/genesis.yaml \
    -l config/vectorcypher/litellm.yaml \
    -o config/vectorcypher/ontology.yaml \
    -s config/vectorcypher/sources.yaml \
    -x config/vectorcypher/expertise.yaml \
    --batch-size 250 --rewrite
```

Fails at the `setup_tenancy` step when inserting into the `organizations` table.

## Verified

Fix tested and confirmed working against a live PostgreSQL instance with the existing `tenancy_mode` enum.
