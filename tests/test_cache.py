"""The in-process thumbnail cache must be bounded by *bytes*, not entry count.

Why this file exists — measured on production 2026-09-01/02:

    uvicorn workers:   959 MB  908 MB  805 MB  839 MB   (4 workers, ~3.5 GB)
    chitra-api cgroup: 3,352 MB of a 4,096 MB MemoryMax
    memory.events:     max 7025      oom_kill 0

Those same four workers were ~80 MB each after the morning restart, and the
service was cgroup-OOM-killed six times in the two preceding days. The ML
models and ffmpeg were removed from the API tier in earlier phases of
`docs/plans/api-oom-fix.md`; this cache was the remaining unbounded growth.

`core/cache.py` bounded itself at 1,000 *entries* with no byte budget.
`core.gallery.ensure_thumb` writes a 512x512 JPEG at `quality=100,
optimize=False, subsampling=0` — typically 150-400 KB. 1,000 x ~250 KB is
~250 MB **per uvicorn worker**, ~1 GB across four, and nothing in the process
could report how much it was actually holding.

So: a byte budget (`CHITRA_THUMB_CACHE_BYTES`, default 64 MiB), eviction
oldest-first until under it, the entry cap kept as a secondary bound, and
`get_cache_stats()` reporting enough that the next investigation reads a
counter instead of inferring from cgroup numbers.
"""
import os
import unittest
import unittest.mock

os.environ.setdefault("CHITRA_DB_PATH", "/tmp/chitra_test.db")

from core import cache


MIB = 1024 * 1024


class CacheTestCase(unittest.TestCase):
    """Every test starts from an empty cache and restores env-derived config."""

    def setUp(self):
        cache.clear_cache()
        self.addCleanup(cache.configure_cache)  # re-read env
        self.addCleanup(cache.clear_cache)

    @staticmethod
    def payload(n_bytes: int, fill: bytes = b"x") -> bytes:
        return fill * n_bytes


class TestByteBudgetEviction(CacheTestCase):
    """Inserting past the byte budget evicts oldest-first until under it."""

    def test_insert_past_byte_budget_evicts_oldest(self):
        cache.configure_cache(max_bytes=1000, max_entries=1000)

        cache.cache_thumbnail("a", self.payload(400))
        cache.cache_thumbnail("b", self.payload(400))
        cache.cache_thumbnail("c", self.payload(400))

        self.assertIsNone(
            cache.get_cached_thumbnail("a"),
            "'a' was the oldest and should have been evicted to fit 'c'",
        )
        self.assertIsNotNone(cache.get_cached_thumbnail("b"))
        self.assertIsNotNone(cache.get_cached_thumbnail("c"))
        self.assertEqual(cache.get_cache_stats()["bytes"], 800)

    def test_eviction_continues_until_under_budget(self):
        """One big insert evicts as many old entries as it takes — not just one."""
        cache.configure_cache(max_bytes=1000, max_entries=1000)

        for key in ("a", "b", "c", "d", "e"):
            cache.cache_thumbnail(key, self.payload(100))
        self.assertEqual(cache.get_cache_stats()["bytes"], 500)

        cache.cache_thumbnail("big", self.payload(800))

        stats = cache.get_cache_stats()
        self.assertLessEqual(
            stats["bytes"], 1000, "cache exceeded its byte budget after insert"
        )
        # 800 + 200 of survivors == 1000; the two survivors are the newest.
        self.assertEqual(stats["bytes"], 1000)
        self.assertEqual(stats["entries"], 3)
        for evicted in ("a", "b", "c"):
            self.assertIsNone(
                cache.get_cached_thumbnail(evicted),
                f"'{evicted}' should have been evicted oldest-first",
            )
        self.assertIsNotNone(cache.get_cached_thumbnail("d"))
        self.assertIsNotNone(cache.get_cached_thumbnail("e"))
        self.assertIsNotNone(cache.get_cached_thumbnail("big"))

    def test_never_exceeds_budget_under_sustained_insert(self):
        """The production shape: 1,000 x 250 KB thumbnails against the default budget.

        This is the regression that mattered. Before the byte bound this loop
        left ~250 MB resident per uvicorn worker; the entry cap of 1,000 never
        fired because it never had to.
        """
        cache.configure_cache()  # defaults, i.e. 64 MiB
        thumb = self.payload(250 * 1024)  # one buffer, reused: the test is about accounting

        peak = 0
        for i in range(1000):
            cache.cache_thumbnail(f"thumbnails/photos/{i}.jpg", thumb)
            peak = max(peak, cache.get_cache_stats()["bytes"])

        self.assertLessEqual(
            peak,
            64 * MIB,
            "cache held more than the 64 MiB budget at some point during insert",
        )
        stats = cache.get_cache_stats()
        self.assertLessEqual(stats["bytes"], 64 * MIB)
        self.assertGreater(stats["evictions"], 0)
        # ~268 entries of 250 KB fit in 64 MiB, nowhere near the 1,000 entry cap.
        self.assertLess(stats["entries"], 300)


class TestEntryCap(CacheTestCase):
    """The entry cap survives as a secondary bound."""

    def test_entry_cap_still_applies(self):
        cache.configure_cache(max_bytes=64 * MIB, max_entries=3)

        for key in ("a", "b", "c", "d"):
            cache.cache_thumbnail(key, self.payload(10))

        stats = cache.get_cache_stats()
        self.assertEqual(stats["entries"], 3)
        self.assertEqual(stats["max_entries"], 3)
        self.assertIsNone(cache.get_cached_thumbnail("a"))
        self.assertIsNotNone(cache.get_cached_thumbnail("d"))

    def test_entry_cap_evicts_oldest_first(self):
        cache.configure_cache(max_bytes=64 * MIB, max_entries=2)

        cache.cache_thumbnail("a", self.payload(10))
        cache.cache_thumbnail("b", self.payload(10))
        cache.cache_thumbnail("c", self.payload(10))

        self.assertIsNone(cache.get_cached_thumbnail("a"))
        self.assertIsNotNone(cache.get_cached_thumbnail("b"))
        self.assertIsNotNone(cache.get_cached_thumbnail("c"))


class TestOversizedItem(CacheTestCase):
    """An item larger than the whole budget is refused, not cached, not looped over.

    Documented decision: caching it is impossible by definition — it would put
    the cache over budget the moment it landed — and evicting the entire cache
    to make room for something that still will not fit is strictly worse than
    not caching it. So it is skipped, counted, and the caller re-reads it from
    storage every time.
    """

    def test_item_larger_than_budget_is_not_cached(self):
        cache.configure_cache(max_bytes=1000, max_entries=1000)

        cache.cache_thumbnail("huge", self.payload(2000))

        self.assertIsNone(cache.get_cached_thumbnail("huge"))
        stats = cache.get_cache_stats()
        self.assertEqual(stats["entries"], 0)
        self.assertEqual(stats["bytes"], 0)
        self.assertEqual(stats["oversize_rejections"], 1)

    def test_oversized_insert_does_not_evict_existing_entries(self):
        """Refusing the giant must not cost us the entries we can actually hold."""
        cache.configure_cache(max_bytes=1000, max_entries=1000)
        cache.cache_thumbnail("a", self.payload(300))
        cache.cache_thumbnail("b", self.payload(300))

        cache.cache_thumbnail("huge", self.payload(5000))

        self.assertIsNotNone(cache.get_cached_thumbnail("a"))
        self.assertIsNotNone(cache.get_cached_thumbnail("b"))
        self.assertEqual(cache.get_cache_stats()["bytes"], 600)
        self.assertEqual(cache.get_cache_stats()["evictions"], 0)

    def test_item_exactly_at_budget_is_cached(self):
        """Exactly-at-budget fits; only strictly-larger is refused."""
        cache.configure_cache(max_bytes=1000, max_entries=1000)
        cache.cache_thumbnail("a", self.payload(400))

        cache.cache_thumbnail("exact", self.payload(1000))

        self.assertIsNotNone(cache.get_cached_thumbnail("exact"))
        self.assertIsNone(cache.get_cached_thumbnail("a"))
        self.assertEqual(cache.get_cache_stats()["bytes"], 1000)
        self.assertEqual(cache.get_cache_stats()["oversize_rejections"], 0)

    def test_zero_budget_disables_caching_without_looping(self):
        """A budget of 0 is a valid way to turn the cache off. It must terminate."""
        cache.configure_cache(max_bytes=0, max_entries=1000)

        cache.cache_thumbnail("a", self.payload(1))

        self.assertIsNone(cache.get_cached_thumbnail("a"))
        self.assertEqual(cache.get_cache_stats()["entries"], 0)
        self.assertEqual(cache.get_cache_stats()["oversize_rejections"], 1)


class TestCacheStats(CacheTestCase):
    """`get_cache_stats()` is the whole point of the observability half."""

    def test_reports_entries_bytes_and_budget(self):
        cache.configure_cache(max_bytes=4096, max_entries=7)
        cache.cache_thumbnail("a", self.payload(100))
        cache.cache_thumbnail("b", self.payload(250))

        stats = cache.get_cache_stats()
        self.assertEqual(stats["entries"], 2)
        self.assertEqual(stats["bytes"], 350)
        self.assertEqual(stats["max_bytes"], 4096)
        self.assertEqual(stats["max_entries"], 7)

    def test_reports_eviction_counters(self):
        cache.configure_cache(max_bytes=1000, max_entries=1000)
        self.assertEqual(cache.get_cache_stats()["evictions"], 0)
        self.assertEqual(cache.get_cache_stats()["evicted_bytes"], 0)

        for key in ("a", "b", "c"):
            cache.cache_thumbnail(key, self.payload(400))

        stats = cache.get_cache_stats()
        self.assertEqual(stats["evictions"], 1)
        self.assertEqual(stats["evicted_bytes"], 400)

    def test_reports_hits_and_misses(self):
        cache.configure_cache(max_bytes=4096, max_entries=10)
        cache.cache_thumbnail("a", self.payload(10))

        cache.get_cached_thumbnail("a")
        cache.get_cached_thumbnail("a")
        cache.get_cached_thumbnail("nope")

        stats = cache.get_cache_stats()
        self.assertEqual(stats["hits"], 2)
        self.assertEqual(stats["misses"], 1)

    def test_stats_are_json_safe(self):
        """These get served over HTTP, so every value must survive json.dumps."""
        import json

        cache.cache_thumbnail("a", self.payload(10))
        json.dumps(cache.get_cache_stats())

    def test_clear_cache_resets_contents_and_counters(self):
        cache.configure_cache(max_bytes=1000, max_entries=1000)
        for key in ("a", "b", "c"):
            cache.cache_thumbnail(key, self.payload(400))
        self.assertGreater(cache.get_cache_stats()["evictions"], 0)

        cache.clear_cache()

        stats = cache.get_cache_stats()
        self.assertEqual(stats["entries"], 0)
        self.assertEqual(stats["bytes"], 0)
        self.assertEqual(stats["evictions"], 0)
        self.assertEqual(stats["hits"], 0)
        self.assertEqual(stats["misses"], 0)


class TestConfiguration(CacheTestCase):
    """`CHITRA_THUMB_CACHE_BYTES`, default 64 MiB per worker."""

    def test_default_budget_is_64_mib(self):
        env = dict(os.environ)
        env.pop("CHITRA_THUMB_CACHE_BYTES", None)
        with unittest.mock.patch.dict(os.environ, env, clear=True):
            cache.configure_cache()
            self.assertEqual(cache.get_cache_stats()["max_bytes"], 64 * MIB)

    def test_env_var_sets_the_budget(self):
        with unittest.mock.patch.dict(
            os.environ, {"CHITRA_THUMB_CACHE_BYTES": str(8 * MIB)}
        ):
            cache.configure_cache()
            self.assertEqual(cache.get_cache_stats()["max_bytes"], 8 * MIB)

    def test_unparseable_env_var_falls_back_to_the_default(self):
        """A typo in .env.production must not leave the cache unbounded."""
        with unittest.mock.patch.dict(
            os.environ, {"CHITRA_THUMB_CACHE_BYTES": "64MB"}
        ):
            cache.configure_cache()
            self.assertEqual(cache.get_cache_stats()["max_bytes"], 64 * MIB)

    def test_negative_env_var_falls_back_to_the_default(self):
        with unittest.mock.patch.dict(
            os.environ, {"CHITRA_THUMB_CACHE_BYTES": "-1"}
        ):
            cache.configure_cache()
            self.assertEqual(cache.get_cache_stats()["max_bytes"], 64 * MIB)

    def test_entry_cap_default_is_1000(self):
        env = dict(os.environ)
        env.pop("CHITRA_THUMB_CACHE_ENTRIES", None)
        with unittest.mock.patch.dict(os.environ, env, clear=True):
            cache.configure_cache()
            self.assertEqual(cache.get_cache_stats()["max_entries"], 1000)

    def test_shrinking_the_budget_evicts_immediately(self):
        """Otherwise a lowered budget only takes effect on the next insert."""
        cache.configure_cache(max_bytes=4096, max_entries=1000)
        for key in ("a", "b", "c", "d"):
            cache.cache_thumbnail(key, self.payload(1000))
        self.assertEqual(cache.get_cache_stats()["bytes"], 4000)

        cache.configure_cache(max_bytes=2000, max_entries=1000)

        self.assertLessEqual(cache.get_cache_stats()["bytes"], 2000)
        self.assertIsNone(cache.get_cached_thumbnail("a"))
        self.assertIsNotNone(cache.get_cached_thumbnail("d"))


class TestRoundTripAndAccounting(CacheTestCase):
    """A hit returns exactly what was stored, and the byte counter stays honest."""

    def test_hit_returns_the_exact_bytes_stored(self):
        cache.configure_cache(max_bytes=4096, max_entries=10)
        jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00" + bytes(range(256)) + b"\xff\xd9"

        cache.cache_thumbnail("thumbnails/photos/1.jpg", jpeg)
        got = cache.get_cached_thumbnail("thumbnails/photos/1.jpg")

        self.assertEqual(got, jpeg)
        self.assertIs(got, jpeg, "the cache must hand back the object, not a copy")

    def test_miss_returns_none(self):
        self.assertIsNone(cache.get_cached_thumbnail("thumbnails/photos/nope.jpg"))

    def test_overwriting_a_key_does_not_double_count_bytes(self):
        cache.configure_cache(max_bytes=4096, max_entries=10)

        cache.cache_thumbnail("a", self.payload(100))
        cache.cache_thumbnail("a", self.payload(300))

        stats = cache.get_cache_stats()
        self.assertEqual(stats["entries"], 1)
        self.assertEqual(stats["bytes"], 300)
        self.assertEqual(len(cache.get_cached_thumbnail("a")), 300)

    def test_expired_entry_frees_its_bytes(self):
        cache.configure_cache(max_bytes=4096, max_entries=10)
        cache.cache_thumbnail("a", self.payload(100))
        cache.cache_thumbnail("b", self.payload(100))
        # Age 'a' past the TTL without sleeping.
        cache._cache_timestamps["a"] -= cache._cache_ttl + 1

        self.assertIsNone(cache.get_cached_thumbnail("a"))

        stats = cache.get_cache_stats()
        self.assertEqual(stats["entries"], 1)
        self.assertEqual(
            stats["bytes"], 100, "TTL expiry must decrement the byte counter"
        )

    def test_byte_counter_tracks_a_long_mixed_workload(self):
        """Insert, overwrite, expire, evict — the counter must still equal reality."""
        cache.configure_cache(max_bytes=10_000, max_entries=50)

        for i in range(200):
            cache.cache_thumbnail(f"k{i % 60}", self.payload(100 + (i % 7) * 10))
            if i % 13 == 0 and cache._cache_timestamps:
                stale = next(iter(cache._cache_timestamps))
                cache._cache_timestamps[stale] -= cache._cache_ttl + 1
                cache.get_cached_thumbnail(stale)

        actual = sum(len(v) for v in cache._thumbnail_cache.values())
        self.assertEqual(
            cache.get_cache_stats()["bytes"],
            actual,
            "the reported byte total drifted from the real contents",
        )
        self.assertEqual(cache.get_cache_stats()["entries"], len(cache._thumbnail_cache))


class TestHealthExposesCacheStats(CacheTestCase):
    """The counters have to reach an operator, not just a unit test.

    The 2026-09-01 incident was diagnosed from cgroup totals and arithmetic
    because the process could not report what it was holding. `/api/health`
    already carries `embed_status` for the same reason; `thumb_cache` joins it.

    `app_fastapi` is imported inside the test so the rest of this module stays
    a pure unit test of `core.cache`.
    """

    def _health_body(self):
        from unittest.mock import AsyncMock

        from fastapi.testclient import TestClient

        import app_fastapi

        fake = AsyncMock(health=AsyncMock(return_value="ok"))
        fake.base_url = "http://127.0.0.1:5101"
        app = app_fastapi.app
        app.dependency_overrides[app_fastapi.get_embedding_client] = lambda: fake
        self.addCleanup(app.dependency_overrides.clear)
        return TestClient(app).get("/api/health").json()

    def test_health_reports_the_thumbnail_cache(self):
        cache.configure_cache(max_bytes=4096, max_entries=9)
        cache.cache_thumbnail("thumbnails/photos/1.jpg", self.payload(120))

        body = self._health_body()

        self.assertIn("thumb_cache", body)
        self.assertEqual(body["thumb_cache"]["entries"], 1)
        self.assertEqual(body["thumb_cache"]["bytes"], 120)
        self.assertEqual(body["thumb_cache"]["max_bytes"], 4096)
        self.assertEqual(body["thumb_cache"]["max_entries"], 9)
        self.assertIn("evictions", body["thumb_cache"])


if __name__ == "__main__":
    unittest.main()
