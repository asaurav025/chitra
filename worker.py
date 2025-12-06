#!/usr/bin/env python3
"""
RQ Worker startup script.
Run this in a separate terminal/process to process background jobs.

Usage:
    python worker.py
"""
import os
import sys
from pathlib import Path

# Mark this as a worker process
os.environ["RQ_WORKER"] = "true"

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rq import Worker, Queue
from core.worker import get_redis_connection

if __name__ == "__main__":
    redis_conn = get_redis_connection()
    
    # Create queues
    queues = [Queue("default", connection=redis_conn)]
    
    print("Starting RQ worker...")
    print(f"Redis: {os.environ.get('REDIS_HOST', 'localhost')}:{os.environ.get('REDIS_PORT', '6379')}")
    print("Press Ctrl+C to stop")
    
    # In newer RQ versions, pass connection directly to Worker
    worker = Worker(queues, connection=redis_conn)
    worker.work()

