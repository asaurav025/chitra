#!/bin/bash
# /home/saurav/services/chitra/start_workers.sh
# Production startup script for Chitra RQ workers (starts 4 workers)

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
# there (or a systemd Environment=) wins. 3 is the measured sweet spot for CLIP
# on this 6-core box: 68 ms/embed vs 153 ms at the 6-thread default. Tune with
# CHITRA_ML_THREADS.
. ./thread_limits.sh "${CHITRA_ML_THREADS:-3}"

# Number of workers
WORKER_COUNT="${WORKER_COUNT:-4}"

# Create logs directory if it doesn't exist
mkdir -p logs

# Determine Python interpreter (use venv if available)
if [ -f ".venv/bin/python3" ]; then
    PYTHON_CMD=".venv/bin/python3"
elif [ -f "venv/bin/python3" ]; then
    PYTHON_CMD="venv/bin/python3"
else
    PYTHON_CMD="python3"
fi

# ---------------------------------------------------------------------------
# CLIP embedding sidecar
#
# One CLIP for the whole box instead of one per uvicorn worker. The API used to
# import core.embedder in all four workers — ~450 MB each of torch import
# weight with no model loaded, ~1.14 GB each once search touched it — against a
# 4 GB cap that OOM-killed the service six times in two days.
#
# It lives here rather than in its own unit only because unit files need sudo.
# The target state is chitra-embed.service; set CHITRA_EMBED_SELF_START=0 in
# .env.production the moment that unit exists, and this block stands down.
#
# --workers 1 is not a tuning knob: a second uvicorn worker would load a second
# copy of the model, which is the exact bug this process exists to fix.
# ---------------------------------------------------------------------------
CHITRA_EMBED_SELF_START="${CHITRA_EMBED_SELF_START:-1}"
CHITRA_EMBED_HOST="${CHITRA_EMBED_HOST:-127.0.0.1}"
CHITRA_EMBED_PORT="${CHITRA_EMBED_PORT:-5101}"

if [ "$CHITRA_EMBED_SELF_START" = "1" ]; then
    if [ -f logs/embed.pid ] && ps -p "$(cat logs/embed.pid)" > /dev/null 2>&1; then
        echo "Embedding sidecar already running with PID $(cat logs/embed.pid)"
    else
        echo "Starting CLIP embedding sidecar on $CHITRA_EMBED_HOST:$CHITRA_EMBED_PORT..."
        $PYTHON_CMD -m uvicorn embed_service:app \
            --host "$CHITRA_EMBED_HOST" --port "$CHITRA_EMBED_PORT" \
            --workers 1 > logs/embed.log 2>&1 &
        echo $! > logs/embed.pid
        echo "Embedding sidecar started with PID $(cat logs/embed.pid)"
        echo "  (search 503s until CLIP finishes loading, ~10s)"
    fi
else
    echo "CHITRA_EMBED_SELF_START=0 — leaving the sidecar to chitra-embed.service"
fi

echo "Starting $WORKER_COUNT Chitra RQ workers (default queue)..."

# Start default-queue workers in background and save PIDs
for i in $(seq 1 $WORKER_COUNT); do
    echo "Starting worker $i..."
    $PYTHON_CMD worker.py default > "logs/worker_$i.log" 2>&1 &
    echo $! > "logs/worker_$i.pid"
    echo "Worker $i started with PID $(cat logs/worker_$i.pid)"
done

# Dedicated video-queue workers: caps concurrent CPU-heavy transcodes.
# They share the unit's CPUQuota=400%, so 2 workers ≈ 2 cores per transcode.
VIDEO_WORKER_COUNT="${VIDEO_WORKER_COUNT:-2}"
echo "Starting $VIDEO_WORKER_COUNT video transcode workers..."
for i in $(seq 1 $VIDEO_WORKER_COUNT); do
    $PYTHON_CMD worker.py video > "logs/worker_video_$i.log" 2>&1 &
    echo $! > "logs/worker_video_$i.pid"
    echo "Video worker $i started with PID $(cat logs/worker_video_$i.pid)"
done

echo "All $WORKER_COUNT default workers + $VIDEO_WORKER_COUNT video workers started"
echo "Worker PIDs:"
cat logs/worker_*.pid | xargs echo