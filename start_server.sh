#!/bin/bash
# Startup script for Chitra FastAPI application
# Uses uvicorn to run the FastAPI application

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default values
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5000}"
WORKERS="${WORKERS:-4}"
LOG_LEVEL="${LOG_LEVEL:-info}"
RELOAD="${RELOAD:-false}"

# Check if running in development mode
if [ "$RELOAD" = "true" ] || [ "$ENV" = "development" ]; then
    echo "Starting Chitra API in development mode..."
    uvicorn app_fastapi:app \
        --host "$HOST" \
        --port "$PORT" \
        --reload \
        --log-level "$LOG_LEVEL"
else
    echo "Starting Chitra API in production mode with $WORKERS workers..."
    uvicorn app_fastapi:app \
        --host "$HOST" \
        --port "$PORT" \
        --workers "$WORKERS" \
        --log-level "$LOG_LEVEL" \
        --access-log
fi

