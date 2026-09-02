"""In-process thumbnail cache for the API workers.

**Bounded by bytes, not by entry count.** The distinction is the whole point of
this module's shape. On 2026-09-01 the four uvicorn workers were holding
959/908/805/839 MB (3,352 MB of a 4,096 MB `MemoryMax`, 7,025 `memory.max`
reclaim events) a few hours after restarting at ~80 MB each, and this cache was
the growth: it capped at 1,000 *entries* and never looked at their size.
`core.gallery.ensure_thumb` writes 512x512 JPEG at `quality=100,
optimize=False, subsampling=0`, i.e. 150-400 KB per thumbnail, so the entry cap
permitted ~250 MB per worker and ~1 GB across four before it fired even once.

Policy, in order:

* **Byte budget** — `CHITRA_THUMB_CACHE_BYTES`, default 64 MiB *per worker*
  (4 workers => a 256 MiB ceiling for the tier). On insert, entries are evicted
  oldest-first until the new one fits.
* **Entry cap** — `CHITRA_THUMB_CACHE_ENTRIES`, default 1,000. A secondary
  bound, kept so that a flood of tiny objects cannot cost unbounded dict
  overhead while sitting well inside the byte budget.
* **TTL** — one hour, checked lazily on read.

**An item larger than the entire budget is not cached at all.** It cannot be:
it would put the cache over budget the instant it landed, and emptying the
cache to make room for something that still does not fit trades every useful
entry for one that is about to be evicted anyway. Such inserts are skipped and
counted as `oversize_rejections`; the caller simply re-reads that object from
storage each time. This is also what makes `CHITRA_THUMB_CACHE_BYTES=0` a valid
way to switch the cache off — every insert is oversized, and eviction
terminates rather than spinning on an empty cache.

Eviction is **FIFO** — oldest insertion first, not least-recently-used. The
module used to claim LRU in its docstring while implementing FIFO; it now says
what it does.

`get_cache_stats()` reports entries, bytes, both bounds and the counters, and
is surfaced on `/api/health` as `thumb_cache`. That is deliberate: the incident
above had to be diagnosed by reading cgroup numbers and doing arithmetic,
because the process could not say how much it was holding.
"""
import os
import threading
import time
from collections import OrderedDict
from typing import Optional

DEFAULT_MAX_BYTES = 64 * 1024 * 1024  # 64 MiB per uvicorn worker
DEFAULT_MAX_ENTRIES = 1000

# Insertion-ordered so eviction is popitem(last=False) — O(1). The previous
# implementation scanned every timestamp with min() to find one victim, which a
# byte budget makes O(n) per *evicted entry* rather than per insert.
_thumbnail_cache: "OrderedDict[str, bytes]" = OrderedDict()
_cache_timestamps: "OrderedDict[str, float]" = OrderedDict()
_cache_bytes = 0
_cache_ttl = 3600  # 1 hour TTL

# Uvicorn runs one event loop per worker, but thumbnails are also touched from
# `run_in_executor` threads. The byte counter is the kind of state where a lost
# update is silent and permanent: drift upward and the cache stops caching,
# drift downward and it stops bounding.
_lock = threading.RLock()

_stats = {
    "evictions": 0,
    "evicted_bytes": 0,
    "expirations": 0,
    "oversize_rejections": 0,
    "hits": 0,
    "misses": 0,
}


def _env_int(name: str, default: int) -> int:
    """Read a non-negative int from the environment, falling back on nonsense.

    A typo in `.env.production` must not leave the cache unbounded, so anything
    unparseable or negative becomes the default rather than an exception at
    import time or an infinite budget.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw.strip())
    except (TypeError, ValueError):
        return default
    return value if value >= 0 else default


_cache_max_bytes = _env_int("CHITRA_THUMB_CACHE_BYTES", DEFAULT_MAX_BYTES)
_cache_max_entries = _env_int("CHITRA_THUMB_CACHE_ENTRIES", DEFAULT_MAX_ENTRIES)


def configure_cache(max_bytes: Optional[int] = None, max_entries: Optional[int] = None):
    """Set the bounds, re-reading the environment for anything not passed.

    Called once at import. Passing an explicit value is how the tests pin a
    small budget; it also gives an operator a way to shrink the cache in a
    running interpreter. Lowering a bound evicts immediately rather than
    waiting for the next insert to notice.
    """
    global _cache_max_bytes, _cache_max_entries
    with _lock:
        _cache_max_bytes = (
            _env_int("CHITRA_THUMB_CACHE_BYTES", DEFAULT_MAX_BYTES)
            if max_bytes is None
            else max(0, int(max_bytes))
        )
        _cache_max_entries = (
            _env_int("CHITRA_THUMB_CACHE_ENTRIES", DEFAULT_MAX_ENTRIES)
            if max_entries is None
            else max(0, int(max_entries))
        )
        _evict_until_fits()


def _drop(thumb_path: str) -> int:
    """Remove one entry and return the bytes it was holding. Caller holds the lock."""
    global _cache_bytes
    data = _thumbnail_cache.pop(thumb_path, None)
    _cache_timestamps.pop(thumb_path, None)
    if data is None:
        return 0
    freed = len(data)
    _cache_bytes -= freed
    return freed


def _evict_until_fits(incoming_bytes: int = 0, reserve_slot: bool = False):
    """Evict oldest-first until `incoming_bytes` more would still be in budget.

    `reserve_slot` also makes room for one more *entry*, which is how the entry
    cap is enforced on insert. It is a separate argument from `incoming_bytes`
    because a legitimately zero-byte payload still needs a slot.

    Terminates on an empty cache regardless of `incoming_bytes` — the caller is
    responsible for having already refused an item that cannot ever fit.
    """
    while _thumbnail_cache and (
        _cache_bytes + incoming_bytes > _cache_max_bytes
        or len(_thumbnail_cache) + (1 if reserve_slot else 0) > _cache_max_entries
    ):
        oldest_path = next(iter(_thumbnail_cache))
        freed = _drop(oldest_path)
        _stats["evictions"] += 1
        _stats["evicted_bytes"] += freed


def get_cached_thumbnail(thumb_path: str) -> Optional[bytes]:
    """
    Get thumbnail from cache if available and not expired.

    Args:
        thumb_path: storage path to thumbnail

    Returns:
        Thumbnail data bytes if cached and valid, None otherwise
    """
    with _lock:
        if thumb_path not in _thumbnail_cache:
            _stats["misses"] += 1
            return None

        cached_at = _cache_timestamps.get(thumb_path)
        if cached_at is not None and time.time() - cached_at > _cache_ttl:
            # Expired. Dropping it here is what keeps the byte counter honest —
            # an expiry that only removed the entry would leak budget forever.
            _drop(thumb_path)
            _stats["expirations"] += 1
            _stats["misses"] += 1
            return None

        _stats["hits"] += 1
        return _thumbnail_cache[thumb_path]


def cache_thumbnail(thumb_path: str, thumb_data: bytes):
    """
    Cache thumbnail data, evicting oldest-first to stay inside the byte budget.

    An object larger than the whole budget is refused outright and no existing
    entry is evicted for it — see the module docstring.

    Args:
        thumb_path: storage path to thumbnail
        thumb_data: Thumbnail image data bytes
    """
    global _cache_bytes
    size = len(thumb_data)

    with _lock:
        if size > _cache_max_bytes or _cache_max_entries < 1:
            # Cannot ever hold it. Refuse without disturbing what we do hold.
            _stats["oversize_rejections"] += 1
            _drop(thumb_path)  # a stale smaller copy would now be wrong
            return

        # An overwrite must give its old bytes back before we measure the gap,
        # or a re-cached key double-counts and the cache slowly starves itself.
        _drop(thumb_path)
        _evict_until_fits(size, reserve_slot=True)

        _thumbnail_cache[thumb_path] = thumb_data
        _cache_timestamps[thumb_path] = time.time()
        _cache_bytes += size


def clear_cache():
    """Clear all cached thumbnails and reset the counters."""
    global _cache_bytes
    with _lock:
        _thumbnail_cache.clear()
        _cache_timestamps.clear()
        _cache_bytes = 0
        for key in _stats:
            _stats[key] = 0


def get_cache_stats():
    """Cache statistics, as served on `/api/health` under `thumb_cache`.

    Per *worker*: uvicorn runs four and health checks land on whichever one
    answers, so consecutive polls legitimately report different numbers.
    """
    with _lock:
        return {
            "entries": len(_thumbnail_cache),
            "bytes": _cache_bytes,
            "max_bytes": _cache_max_bytes,
            "max_entries": _cache_max_entries,
            "ttl": _cache_ttl,
            **_stats,
        }
