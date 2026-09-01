#!/bin/bash
# /home/saurav/services/chitra/stop_workers.sh
# Stop all Chitra RQ workers

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Stopping Chitra RQ workers..."

# Kill workers by PID files
if [ -d logs ]; then
    for pidfile in logs/worker_*.pid; do
        if [ -f "$pidfile" ]; then
            pid=$(cat "$pidfile")
            if ps -p "$pid" > /dev/null 2>&1; then
                echo "Stopping worker with PID $pid..."
                kill "$pid" || true
            fi
            rm -f "$pidfile"
        fi
    done
fi

# CLIP embedding sidecar, if start_workers.sh started it. Stopped by pid file
# and *before* the pkill below, which matches "worker.py" and would never catch
# it: an orphaned sidecar would then hold the port and 1.14 GB, and the next
# start would silently fail to bind.
if [ -f logs/embed.pid ]; then
    embed_pid=$(cat logs/embed.pid)
    if ps -p "$embed_pid" > /dev/null 2>&1; then
        echo "Stopping embedding sidecar with PID $embed_pid..."
        kill "$embed_pid" || true
    fi
    rm -f logs/embed.pid
fi

# Also kill any remaining worker processes
pkill -f "worker.py" || true

echo "All workers stopped"