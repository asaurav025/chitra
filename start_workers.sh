#!/bin/bash
# /home/saurav/services/chitra/start_workers.sh
# Production startup script for Chitra RQ workers (starts 4 workers)

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load environment variables
if [ -f .env.production ]; then
    export $(cat .env.production | grep -v '^#' | xargs)
fi

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

echo "Starting $WORKER_COUNT Chitra RQ workers..."

# Start workers in background and save PIDs
for i in $(seq 1 $WORKER_COUNT); do
    echo "Starting worker $i..."
    $PYTHON_CMD worker.py > "logs/worker_$i.log" 2>&1 &
    echo $! > "logs/worker_$i.pid"
    echo "Worker $i started with PID $(cat logs/worker_$i.pid)"
done

echo "All $WORKER_COUNT workers started"
echo "Worker PIDs:"
cat logs/worker_*.pid | xargs echo