#!/bin/sh
# Khora Docker entrypoint script
# Validates configuration and starts the FastAPI application

set -e

echo "=== Khora Startup ==="
echo "Port: ${KHORA_PORT:-8000}"
echo "Log Level: ${KHORA_LOG_LEVEL:-INFO}"
echo "Environment: ${KHORA_ENVIRONMENT:-development}"

# Display database configuration (redact sensitive info)
if [ -n "$KHORA_DATABASE_URL" ]; then
    echo "Database: configured"
else
    echo "Warning: KHORA_DATABASE_URL not set - database operations will fail"
fi

echo ""

# Run database migrations if database is configured
if [ -n "$KHORA_DATABASE_URL" ]; then
    echo "Running database migrations..."
    alembic upgrade head
    echo "Migrations completed."
fi

echo ""
echo "Starting Khora API server..."

# Determine config file path
CONFIG_PATH="${KHORA_CONFIG_PATH:-}"
echo "Config: ${CONFIG_PATH:-environment variables}"

# Build command arguments
CMD_ARGS="--host ${KHORA_HOST:-0.0.0.0} --port ${KHORA_PORT:-8000}"

# Add config if file exists
if [ -n "$CONFIG_PATH" ] && [ -f "$CONFIG_PATH" ]; then
    CMD_ARGS="--config $CONFIG_PATH $CMD_ARGS"
fi

# Start the Khora server using CLI
exec khora serve $CMD_ARGS
