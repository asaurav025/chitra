#!/bin/bash
# /home/saurav/services/chitra/start_production.sh
# Production startup script for Chitra FastAPI main service

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load environment variables.
# `set -a` marks everything the file defines for export, so sourcing it is
# enough. The previous `export $(cat ... | xargs)` form word-split values on
# spaces, mangled quotes, truncated any value containing a '#', and leaked the
# secrets into the process table where `ps` could read them.
if [ -f .env.production ]; then
    set -a; . ./.env.production; set +a
fi

# Cap BLAS/OpenMP threads. Sourced *after* .env.production so anything set
# there (or a systemd Environment=) wins. The API's only numeric work is a
# 1,694x512 GEMV over 3.31 MiB of embeddings — memory-bound, so 2 is plenty.
# It also bounds the damage if an ML import ever sneaks back in: 2 threads x 4
# uvicorn workers instead of 24 on a 6-core box.
. ./thread_limits.sh 2

# Default values
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5000}"
WORKERS="${WORKERS:-4}"
LOG_LEVEL="${LOG_LEVEL:-info}"

# Create logs directory if it doesn't exist
mkdir -p logs

# Determine Python interpreter (use venv if available)
if [ -f ".venv/bin/uvicorn" ]; then
    UVICORN_CMD=".venv/bin/uvicorn"
    PYTHON_CMD=".venv/bin/python3"
elif [ -f "venv/bin/uvicorn" ]; then
    UVICORN_CMD="venv/bin/uvicorn"
    PYTHON_CMD="venv/bin/python3"
else
    UVICORN_CMD="uvicorn"
    PYTHON_CMD="python3"
fi

echo "Starting Chitra API in production mode..."
echo "Host: $HOST"
echo "Port: $PORT"
echo "Workers: $WORKERS"
echo "Log Level: $LOG_LEVEL"

# Start uvicorn with multiple workers
exec $UVICORN_CMD app_fastapi:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers "$WORKERS" \
    --log-level "$LOG_LEVEL" \
    --timeout-keep-alive 120 \
    --timeout-graceful-shutdown 30

