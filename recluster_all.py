#!/usr/bin/env python3
"""
Script to re-cluster all faces using HDBSCAN.

This script:
1. Unassigns all faces (sets person_id to NULL)
2. Triggers clustering using the background job with HDBSCAN
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.worker import get_queue
from core.jobs import cluster_faces_job
from core import db

# Database path
DB_PATH = os.environ.get("CHITRA_DB_PATH", db.DB_DEFAULT_PATH)

def main():
    """Re-cluster all faces."""
    print("=" * 60)
    print("Re-clustering all faces with HDBSCAN")
    print("=" * 60)
    print(f"Database: {DB_PATH}")
    print()
    
    # Check if database exists
    if not os.path.exists(DB_PATH):
        print(f"Error: Database not found at {DB_PATH}")
        sys.exit(1)
    
    # Get queue
    try:
        queue = get_queue()
        print("✓ Connected to Redis queue")
    except Exception as e:
        print(f"Error: Failed to connect to Redis: {e}")
        print("Make sure Redis is running and REDIS_URL is set correctly")
        sys.exit(1)
    
    # Queue clustering job with reset=True
    # Lower threshold (0.6) for less strict matching - groups more similar faces together
    threshold = 0.6
    print(f"\nQueuing re-clustering job (reset=True, threshold={threshold})...")
    try:
        job = queue.enqueue(
            cluster_faces_job,
            DB_PATH,
            threshold,  # Lower threshold for less strict matching
            None,  # photo_ids - None means all faces
            True,  # reset - unassign all faces first
            job_timeout='10m'  # Clustering can take time
        )
        print(f"✓ Job queued successfully!")
        print(f"  Job ID: {job.id}")
        print(f"  Status: {job.get_status()}")
        print()
        print("The clustering job is now running in the background.")
        print("Check your worker logs to see progress.")
        print()
        print("To check job status, you can use:")
        print(f"  from rq import Queue, Connection")
        print(f"  from redis import Redis")
        print(f"  queue = Queue(connection=Redis.from_url('redis://localhost:6379/0'))")
        print(f"  job = queue.fetch_job('{job.id}')")
        print(f"  print(job.get_status())")
    except Exception as e:
        print(f"Error: Failed to queue job: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

