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

# Also kill any remaining worker processes
pkill -f "worker.py" || true

echo "All workers stopped"