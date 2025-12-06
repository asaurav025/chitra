"""
Redis Queue worker setup and connection utilities.
"""
import os
from redis import Redis
from rq import Queue

# Redis connection
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_DB = int(os.environ.get("REDIS_DB", "0"))

def get_redis_connection():
    """Get Redis connection."""
    return Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=False)

def get_queue(name="default"):
    """Get RQ queue by name."""
    return Queue(name, connection=get_redis_connection())

