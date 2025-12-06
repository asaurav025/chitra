"""
Local caching for frequently accessed thumbnails.
Uses in-memory LRU cache to reduce storage round-trips.
"""
from functools import lru_cache
from typing import Optional
import time

# In-memory cache for thumbnails (LRU with max 1000 items)
_thumbnail_cache = {}
_cache_timestamps = {}
_cache_max_size = 1000
_cache_ttl = 3600  # 1 hour TTL


def get_cached_thumbnail(thumb_path: str) -> Optional[bytes]:
    """
    Get thumbnail from cache if available and not expired.
    
    Args:
        thumb_path: storage path to thumbnail
        
    Returns:
        Thumbnail data bytes if cached and valid, None otherwise
    """
    if thumb_path not in _thumbnail_cache:
        return None
    
    # Check if cache entry has expired
    if thumb_path in _cache_timestamps:
        if time.time() - _cache_timestamps[thumb_path] > _cache_ttl:
            # Expired, remove from cache
            _thumbnail_cache.pop(thumb_path, None)
            _cache_timestamps.pop(thumb_path, None)
            return None
    
    return _thumbnail_cache.get(thumb_path)


def cache_thumbnail(thumb_path: str, thumb_data: bytes):
    """
    Cache thumbnail data.
    
    Args:
        thumb_path: storage path to thumbnail
        thumb_data: Thumbnail image data bytes
    """
    # If cache is full, remove oldest entry
    if len(_thumbnail_cache) >= _cache_max_size:
        # Remove oldest entry (simple FIFO, could be improved with LRU)
        if _cache_timestamps:
            oldest_path = min(_cache_timestamps.items(), key=lambda x: x[1])[0]
            _thumbnail_cache.pop(oldest_path, None)
            _cache_timestamps.pop(oldest_path, None)
    
    _thumbnail_cache[thumb_path] = thumb_data
    _cache_timestamps[thumb_path] = time.time()


def clear_cache():
    """Clear all cached thumbnails."""
    global _thumbnail_cache, _cache_timestamps
    _thumbnail_cache.clear()
    _cache_timestamps.clear()


def get_cache_stats():
    """Get cache statistics."""
    return {
        "size": len(_thumbnail_cache),
        "max_size": _cache_max_size,
        "ttl": _cache_ttl
    }

