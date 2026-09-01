"""HTTP semantics of the thumbnail endpoints: caching, storage round-trips, error bodies.

Every test here injects a fake storage client and overrides auth and the DB
through `app.dependency_overrides`, so nothing touches MinIO, the live
database, or the failing disk.

Why these exist:

* **Face crops were served `no-cache, no-store, must-revalidate` with no
  ETag** while photo thumbnails were served `public, max-age=604800` with one.
  A face crop is an immutable 160x160 JPEG at `thumbnails/faces/face_{id}.jpg`
  and `faces.id` is `AUTOINCREMENT`, so the key is never reused — `no-store`
  made the browser re-download all ~2,000 crops on every visit to the People
  route, and face thumbnails failed at 51.2% against 3.5% for photo thumbnails.
* **The photo thumbnail handler probed `file_exists_async` before serving**,
  doubling storage operations on a grid render to catch a rare missing object.
  Serve first, regenerate on the miss.
* **`storage_error: {str(e)}` put MinIO/urllib3 internals in response bodies.**
"""
import asyncio
import os
import tempfile
import unittest

from fastapi.testclient import TestClient

os.environ.setdefault("CHITRA_DB_PATH", "/tmp/chitra_test.db")

import app_fastapi
from core import cache as thumb_cache
from core import db_async


PHOTO_THUMB = "thumbnails/photos/1.jpg"
FACE_THUMB = "thumbnails/faces/face_1.jpg"
JPEG = b"\xff\xd8\xff\xe0fake-jpeg-bytes"


class FakeUser(dict):
    """Stands in for the aiosqlite.Row the auth dependency returns."""


class FakeStorage:
    """Records every call so a test can assert on the round-trips made."""

    def __init__(self, objects=None, download_error=None):
        self.objects = dict(objects or {})
        self.download_error = download_error
        self.downloads = []
        self.exists_calls = []
        self.uploads = []

    async def download_file_async(self, remote_path):
        self.downloads.append(remote_path)
        if self.download_error is not None:
            raise self.download_error
        if remote_path not in self.objects:
            raise FileNotFoundError(remote_path)
        return self.objects[remote_path]

    async def file_exists_async(self, remote_path):
        self.exists_calls.append(remote_path)
        return remote_path in self.objects

    async def upload_file_async(self, file_data, remote_path):
        self.uploads.append(remote_path)
        self.objects[remote_path] = file_data
        return remote_path

    def generate_thumbnail_path(self, item_id, item_type="photo"):
        if item_type == "photo":
            return f"thumbnails/photos/{item_id}.jpg"
        return f"thumbnails/faces/face_{item_id}.jpg"


class ThumbnailEndpointCase(unittest.TestCase):
    """Shared fixture: one photo, one face with a crop, one video."""

    def setUp(self):
        # The in-process thumbnail cache is module-global and would otherwise
        # leak served bytes between tests, hiding the round-trips under test.
        thumb_cache.clear_cache()
        self.addCleanup(thumb_cache.clear_cache)
        self.db_path = tempfile.mktemp(suffix=".db")
        self.addCleanup(self._unlink_db)
        asyncio.run(self._seed())

    def _unlink_db(self):
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    async def _seed(self):
        await db_async.init_db_async(self.db_path)
        async with db_async.connect_async(self.db_path) as conn:
            await conn.execute(
                "INSERT INTO photos (id, file_path, size, created_at, checksum, thumb_path, media_type)"
                " VALUES (1, 'photos/2026/09/a.jpg', 1, '2026-01-01T00:00:00', 'c1', ?, 'photo')",
                (PHOTO_THUMB,),
            )
            await conn.execute(
                "INSERT INTO photos (id, file_path, size, created_at, checksum, thumb_path, media_type)"
                " VALUES (2, 'photos/2026/09/b.jpg', 1, '2026-01-01T00:00:00', 'c2', NULL, 'photo')"
            )
            await conn.execute(
                "INSERT INTO photos (id, file_path, size, created_at, checksum, thumb_path, media_type)"
                " VALUES (3, 'videos/2026/09/c.mp4', 1, '2026-01-01T00:00:00', 'c3', ?, 'video')",
                ("thumbnails/photos/3.jpg",),
            )
            await conn.execute(
                "INSERT INTO faces (id, photo_id, face_index, embedding) VALUES (1, 1, 0, X'00')"
            )
            await conn.execute(
                "INSERT INTO face_thumbs (face_id, thumb_path) VALUES (1, ?)", (FACE_THUMB,)
            )
            await conn.commit()

    def client_with(self, storage, raise_server_exceptions=True):
        app = app_fastapi.app

        async def fake_db():
            async with db_async.connect_async(self.db_path) as conn:
                yield conn

        app.dependency_overrides[app_fastapi.get_db_async] = fake_db
        app.dependency_overrides[app_fastapi.get_current_active_user] = lambda: FakeUser(
            id=1, username="tester", role="user", is_active=1, is_whitelisted=1
        )
        app.dependency_overrides[app_fastapi.get_storage_client] = lambda: storage
        self.addCleanup(app.dependency_overrides.clear)
        return TestClient(app, raise_server_exceptions=raise_server_exceptions)


class TestFaceThumbnailCaching(ThumbnailEndpointCase):
    """A face crop is immutable; it must be cacheable exactly like a photo thumb."""

    def test_face_thumbnail_is_publicly_cacheable(self):
        client = self.client_with(FakeStorage({FACE_THUMB: JPEG}))

        resp = client.get("/api/faces/1/thumbnail")

        self.assertEqual(200, resp.status_code, resp.text)
        cache_control = resp.headers.get("Cache-Control", "")
        self.assertIn("public", cache_control)
        self.assertIn("max-age=", cache_control)
        self.assertNotIn("no-store", cache_control)
        self.assertNotIn("no-cache", cache_control)

    def test_face_thumbnail_sends_no_cache_busting_legacy_headers(self):
        client = self.client_with(FakeStorage({FACE_THUMB: JPEG}))

        resp = client.get("/api/faces/1/thumbnail")

        self.assertNotIn("Pragma", resp.headers)
        self.assertNotIn("Expires", resp.headers)

    def test_face_thumbnail_carries_an_etag(self):
        client = self.client_with(FakeStorage({FACE_THUMB: JPEG}))

        resp = client.get("/api/faces/1/thumbnail")

        etag = resp.headers.get("ETag")
        self.assertIsNotNone(etag, "face thumbnails must carry an ETag")
        self.assertTrue(etag.startswith('"') and etag.endswith('"'), etag)

    def test_face_thumbnail_etag_is_stable_across_requests(self):
        storage = FakeStorage({FACE_THUMB: JPEG})
        client = self.client_with(storage)

        first = client.get("/api/faces/1/thumbnail").headers["ETag"]
        thumb_cache.clear_cache()
        second = client.get("/api/faces/1/thumbnail").headers["ETag"]

        self.assertEqual(first, second)

    def test_conditional_request_gets_304(self):
        storage = FakeStorage({FACE_THUMB: JPEG})
        client = self.client_with(storage)
        etag = client.get("/api/faces/1/thumbnail").headers["ETag"]

        resp = client.get("/api/faces/1/thumbnail", headers={"If-None-Match": etag})

        self.assertEqual(304, resp.status_code, resp.text)
        self.assertEqual(etag, resp.headers.get("ETag"))
        self.assertIn("public", resp.headers.get("Cache-Control", ""))

    def test_304_costs_no_storage_round_trip(self):
        """The whole point: a revalidated crop must not be re-downloaded."""
        storage = FakeStorage({FACE_THUMB: JPEG})
        client = self.client_with(storage)
        etag = client.get("/api/faces/1/thumbnail").headers["ETag"]
        thumb_cache.clear_cache()
        storage.downloads.clear()

        resp = client.get("/api/faces/1/thumbnail", headers={"If-None-Match": etag})

        self.assertEqual(304, resp.status_code)
        self.assertEqual([], storage.downloads)

    def test_a_different_etag_still_serves_the_body(self):
        storage = FakeStorage({FACE_THUMB: JPEG})
        client = self.client_with(storage)

        resp = client.get("/api/faces/1/thumbnail", headers={"If-None-Match": '"stale"'})

        self.assertEqual(200, resp.status_code)
        self.assertEqual(JPEG, resp.content)

    def test_missing_face_thumb_row_still_404s(self):
        client = self.client_with(FakeStorage({FACE_THUMB: JPEG}))

        resp = client.get("/api/faces/999/thumbnail")

        self.assertEqual(404, resp.status_code)
        self.assertEqual("face_thumb_not_found", resp.json()["detail"])

    def test_missing_object_still_404s(self):
        client = self.client_with(FakeStorage({}))

        resp = client.get("/api/faces/1/thumbnail")

        self.assertEqual(404, resp.status_code)
        self.assertEqual("thumb_not_found_on_storage", resp.json()["detail"])


class TestThumbnailStorageRoundTrips(ThumbnailEndpointCase):
    """A thumbnail GET must cost one storage operation, not two.

    The handler used to call `file_exists_async` before serving, purely to catch
    the rare object that is recorded in the DB but gone from MinIO. On a grid
    render of 50 tiles that doubles storage operations. Serve first; regenerate
    on the FileNotFoundError.
    """

    def stub_regeneration(self, storage, thumb_path=PHOTO_THUMB, data=b"regenerated", error=None):
        """Replace the lazy PIL regeneration with a recorder.

        The real helper downloads the original, shells into PIL and writes temp
        files; none of that is what these tests are about, and the disk on this
        box is failing.
        """
        calls = []

        async def fake_ensure(file_path, photo_id, storage_arg, conn):
            calls.append((file_path, photo_id))
            if error is not None:
                raise error
            storage_arg.objects[thumb_path] = data
            return thumb_path

        original = app_fastapi.ensure_photo_thumb_async
        app_fastapi.ensure_photo_thumb_async = fake_ensure
        self.addCleanup(setattr, app_fastapi, "ensure_photo_thumb_async", original)
        return calls

    def no_poster_enqueue(self):
        """enqueue_video_poster talks to Redis; record instead."""
        calls = []
        original = app_fastapi.enqueue_video_poster
        app_fastapi.enqueue_video_poster = lambda photo_id, file_path: calls.append(photo_id) or True
        self.addCleanup(setattr, app_fastapi, "enqueue_video_poster", original)
        return calls

    # -- photos ---------------------------------------------------------------
    def test_serving_a_present_thumbnail_costs_one_storage_call(self):
        storage = FakeStorage({PHOTO_THUMB: JPEG})
        client = self.client_with(storage)

        resp = client.get("/api/photos/1/thumbnail")

        self.assertEqual(200, resp.status_code, resp.text)
        self.assertEqual(JPEG, resp.content)
        self.assertEqual([PHOTO_THUMB], storage.downloads)
        self.assertEqual(
            [], storage.exists_calls,
            "the happy path must not probe storage before serving",
        )

    def test_a_cache_hit_costs_no_storage_call_at_all(self):
        storage = FakeStorage({PHOTO_THUMB: JPEG})
        client = self.client_with(storage)
        client.get("/api/photos/1/thumbnail")
        storage.downloads.clear()
        storage.exists_calls.clear()

        resp = client.get("/api/photos/1/thumbnail")

        self.assertEqual(200, resp.status_code)
        self.assertEqual([], storage.downloads)
        self.assertEqual([], storage.exists_calls)

    def test_an_object_missing_from_storage_is_regenerated_and_served(self):
        """Same outcome as the old pre-check, one round-trip later."""
        storage = FakeStorage({})
        regenerated = self.stub_regeneration(storage, data=b"fresh-thumb")
        client = self.client_with(storage)

        resp = client.get("/api/photos/1/thumbnail")

        self.assertEqual(200, resp.status_code, resp.text)
        self.assertEqual(b"fresh-thumb", resp.content)
        self.assertEqual([("photos/2026/09/a.jpg", 1)], regenerated)

    def test_regeneration_records_the_new_path_in_the_database(self):
        storage = FakeStorage({})
        self.stub_regeneration(storage, thumb_path="thumbnails/photos/1.jpg")
        client = self.client_with(storage)

        client.get("/api/photos/1/thumbnail")

        async def read_back():
            async with db_async.connect_async(self.db_path) as conn:
                cur = await conn.execute("SELECT thumb_path FROM photos WHERE id=1")
                return (await cur.fetchone())["thumb_path"]

        self.assertEqual("thumbnails/photos/1.jpg", asyncio.run(read_back()))

    def test_a_photo_with_no_thumb_path_is_still_generated_lazily(self):
        """The 766 legacy NULL-media_type rows depend on this path."""
        storage = FakeStorage({})
        regenerated = self.stub_regeneration(storage, data=b"lazy")
        client = self.client_with(storage)

        resp = client.get("/api/photos/2/thumbnail")

        self.assertEqual(200, resp.status_code, resp.text)
        self.assertEqual(b"lazy", resp.content)
        self.assertEqual([("photos/2026/09/b.jpg", 2)], regenerated)

    def test_an_unregenerable_thumbnail_still_404s(self):
        from fastapi import HTTPException

        storage = FakeStorage({})
        self.stub_regeneration(
            storage,
            error=HTTPException(status_code=404, detail="Original photo file not found in storage"),
        )
        client = self.client_with(storage)

        resp = client.get("/api/photos/1/thumbnail")

        self.assertEqual(404, resp.status_code)

    def test_an_unknown_photo_still_404s_without_touching_storage(self):
        storage = FakeStorage({PHOTO_THUMB: JPEG})
        client = self.client_with(storage)

        resp = client.get("/api/photos/999/thumbnail")

        self.assertEqual(404, resp.status_code)
        self.assertEqual("photo_not_found", resp.json()["detail"])
        self.assertEqual([], storage.downloads)
        self.assertEqual([], storage.exists_calls)

    # -- videos ---------------------------------------------------------------
    def test_serving_a_present_video_poster_costs_one_storage_call(self):
        storage = FakeStorage({"thumbnails/photos/3.jpg": JPEG})
        enqueued = self.no_poster_enqueue()
        client = self.client_with(storage)

        resp = client.get("/api/photos/3/thumbnail")

        self.assertEqual(200, resp.status_code, resp.text)
        self.assertEqual([], storage.exists_calls)
        self.assertEqual([], enqueued)

    def test_a_missing_video_poster_is_enqueued_and_404s(self):
        """ffmpeg never runs in the API — the miss asks a worker and 404s."""
        storage = FakeStorage({})
        enqueued = self.no_poster_enqueue()
        client = self.client_with(storage)

        resp = client.get("/api/photos/3/thumbnail")

        self.assertEqual(404, resp.status_code)
        self.assertEqual("poster_pending", resp.json()["detail"])
        self.assertEqual("3", resp.headers.get("Retry-After"))
        self.assertEqual([3], enqueued)

    def test_a_video_with_no_poster_path_never_reaches_storage(self):
        storage = FakeStorage({})
        enqueued = self.no_poster_enqueue()
        client = self.client_with(storage)

        async def clear_poster():
            async with db_async.connect_async(self.db_path) as conn:
                await conn.execute("UPDATE photos SET thumb_path=NULL WHERE id=3")
                await conn.commit()

        asyncio.run(clear_poster())

        resp = client.get("/api/photos/3/thumbnail")

        self.assertEqual(404, resp.status_code)
        self.assertEqual("poster_pending", resp.json()["detail"])
        self.assertEqual([3], enqueued)
        self.assertEqual([], storage.downloads)
        self.assertEqual([], storage.exists_calls)

    def test_a_video_poster_never_falls_through_to_pil(self):
        """The old code downloaded the whole mp4 and 500'd opening it as an image."""
        storage = FakeStorage({})
        self.no_poster_enqueue()
        regenerated = self.stub_regeneration(storage)
        client = self.client_with(storage)

        client.get("/api/photos/3/thumbnail")

        self.assertEqual([], regenerated)


# What a real MinIO/urllib3 failure looks like when it reaches a handler.
MINIO_INTERNALS = (
    "S3 operation failed; code: InternalError, message: We encountered an "
    "internal error, host_id: minio-0.minio.default.svc:9000, "
    "HTTPSConnectionPool(host='10.0.0.4', port=9000): Max retries exceeded"
)


class TestStorageErrorsDoNotLeakInternals(ThumbnailEndpointCase):
    """`storage_error: {str(e)}` put MinIO hostnames and urllib3 tracebacks in
    response bodies. The client gets a stable token; the detail goes to the log."""

    def assert_generic_500(self, resp):
        self.assertEqual(500, resp.status_code, resp.text)
        self.assertEqual("storage_error", resp.json()["detail"])
        self.assertNotIn("InternalError", resp.text)
        self.assertNotIn("minio", resp.text.lower())
        self.assertNotIn("HTTPSConnectionPool", resp.text)

    def broken_storage(self):
        return FakeStorage(
            {PHOTO_THUMB: JPEG, FACE_THUMB: JPEG},
            download_error=RuntimeError(MINIO_INTERNALS),
        )

    def test_photo_thumbnail_returns_a_generic_detail(self):
        client = self.client_with(self.broken_storage())

        with self.assertLogs("app_fastapi", level="ERROR"):
            resp = client.get("/api/photos/1/thumbnail")

        self.assert_generic_500(resp)

    def test_photo_thumbnail_logs_the_real_cause(self):
        client = self.client_with(self.broken_storage())

        with self.assertLogs("app_fastapi", level="ERROR") as logs:
            client.get("/api/photos/1/thumbnail")

        self.assertTrue(
            any("InternalError" in record.getMessage() or "InternalError" in str(record.exc_info)
                for record in logs.records),
            "the cause has to survive somewhere — the log is where it belongs",
        )

    def test_face_thumbnail_returns_a_generic_detail(self):
        client = self.client_with(self.broken_storage())

        with self.assertLogs("app_fastapi", level="ERROR"):
            resp = client.get("/api/faces/1/thumbnail")

        self.assert_generic_500(resp)

    def test_photo_image_returns_a_generic_detail(self):
        client = self.client_with(self.broken_storage())

        with self.assertLogs("app_fastapi", level="ERROR"):
            resp = client.get("/api/photos/1/image")

        self.assert_generic_500(resp)

    def test_storage_passthrough_returns_a_generic_detail(self):
        client = self.client_with(self.broken_storage())

        with self.assertLogs("app_fastapi", level="ERROR"):
            resp = client.get("/api/storage/photos/2026/09/a.jpg")

        self.assert_generic_500(resp)

    def test_the_module_no_longer_interpolates_exceptions_into_storage_errors(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(repo_root, "app_fastapi.py")) as fh:
            source = fh.read()
        self.assertNotIn("storage_error: {str(e)}", source)


class FakeRedis:
    """Just enough of redis-py for the unavailable-thumbnail marker."""

    def __init__(self, broken=False):
        self.store = {}
        self.broken = broken
        self.sets = []

    def _check(self):
        if self.broken:
            raise ConnectionError("Error 111 connecting to 127.0.0.1:6379")

    def get(self, key):
        self._check()
        return self.store.get(key)

    def set(self, key, value, nx=False, ex=None):
        self._check()
        self.sets.append((key, value, ex))
        if nx and key in self.store:
            return None
        self.store[key] = value
        return True


class TestUndecodableSourceIsNot500(ThumbnailEndpointCase):
    """63 photos are `.arw`-named files that are actually JPEGs.

    Generating their thumbnail raised out of core.gallery and propagated
    uncaught out of the handler as a bare 500 — and every single request
    re-downloaded the multi-MB original first. core.gallery now raises the typed
    `ThumbnailUnavailable` carrying a machine-readable reason; the HTTP layer's
    job is to turn that into an honest status with that reason as the detail,
    and to stop paying for the download over and over.
    """

    def raise_from_regeneration(self, exc):
        async def fake_ensure(file_path, photo_id, storage_arg, conn):
            raise exc

        original = app_fastapi.ensure_photo_thumb_async
        app_fastapi.ensure_photo_thumb_async = fake_ensure
        self.addCleanup(setattr, app_fastapi, "ensure_photo_thumb_async", original)

    def count_regenerations(self, exc):
        calls = []

        async def fake_ensure(file_path, photo_id, storage_arg, conn):
            calls.append(photo_id)
            raise exc

        original = app_fastapi.ensure_photo_thumb_async
        app_fastapi.ensure_photo_thumb_async = fake_ensure
        self.addCleanup(setattr, app_fastapi, "ensure_photo_thumb_async", original)
        return calls

    def fake_redis(self, broken=False):
        redis = FakeRedis(broken=broken)
        original = app_fastapi.get_redis_connection
        app_fastapi.get_redis_connection = lambda: redis
        self.addCleanup(setattr, app_fastapi, "get_redis_connection", original)
        return redis

    def undecodable(self, reason="decode_failed"):
        from core.gallery import ThumbnailUnavailable

        return ThumbnailUnavailable(
            reason, "photos/2026/09/b.arw", "cannot identify image file <_io.BytesIO>"
        )

    # -- status and detail ----------------------------------------------------
    def test_an_undecodable_source_is_not_a_500(self):
        self.fake_redis()
        self.raise_from_regeneration(self.undecodable())
        client = self.client_with(FakeStorage({}), raise_server_exceptions=False)

        resp = client.get("/api/photos/2/thumbnail")

        self.assertNotEqual(500, resp.status_code, "still a bare 500")
        self.assertIn(resp.status_code, (404, 422))

    def test_decode_failed_reports_the_reason_as_the_detail(self):
        self.fake_redis()
        self.raise_from_regeneration(self.undecodable("decode_failed"))
        client = self.client_with(FakeStorage({}), raise_server_exceptions=False)

        resp = client.get("/api/photos/2/thumbnail")

        self.assertEqual(422, resp.status_code, resp.text)
        self.assertEqual("decode_failed", resp.json()["detail"])

    def test_source_unreadable_reports_the_reason_as_the_detail(self):
        self.fake_redis()
        self.raise_from_regeneration(self.undecodable("source_unreadable"))
        client = self.client_with(FakeStorage({}), raise_server_exceptions=False)

        resp = client.get("/api/photos/2/thumbnail")

        self.assertEqual(404, resp.status_code, resp.text)
        self.assertEqual("source_unreadable", resp.json()["detail"])

    def test_the_underlying_message_stays_out_of_the_body(self):
        self.fake_redis()
        self.raise_from_regeneration(self.undecodable())
        client = self.client_with(FakeStorage({}), raise_server_exceptions=False)

        resp = client.get("/api/photos/2/thumbnail")

        self.assertNotIn("BytesIO", resp.text)
        self.assertNotIn("photos/2026/09", resp.text)

    def test_the_failure_is_logged_with_the_source(self):
        self.fake_redis()
        self.raise_from_regeneration(self.undecodable())
        client = self.client_with(FakeStorage({}), raise_server_exceptions=False)

        with self.assertLogs("app_fastapi", level="WARNING") as logs:
            client.get("/api/photos/2/thumbnail")

        joined = " ".join(record.getMessage() for record in logs.records)
        self.assertIn("decode_failed", joined)

    def test_a_thumbnail_recorded_but_gone_and_undecodable_also_422s(self):
        """The other regeneration site: object missing from storage."""
        self.fake_redis()
        self.raise_from_regeneration(self.undecodable())
        client = self.client_with(FakeStorage({}), raise_server_exceptions=False)

        resp = client.get("/api/photos/1/thumbnail")

        self.assertEqual(422, resp.status_code, resp.text)
        self.assertEqual("decode_failed", resp.json()["detail"])

    # -- stop paying for it on every request ----------------------------------
    def test_a_deterministic_failure_is_not_retried_on_the_next_request(self):
        """Each retry re-downloads the multi-MB original before failing again."""
        self.fake_redis()
        calls = self.count_regenerations(self.undecodable("decode_failed"))
        client = self.client_with(FakeStorage({}), raise_server_exceptions=False)

        first = client.get("/api/photos/2/thumbnail")
        second = client.get("/api/photos/2/thumbnail")

        self.assertEqual(422, first.status_code)
        self.assertEqual(422, second.status_code)
        self.assertEqual("decode_failed", second.json()["detail"])
        self.assertEqual([2], calls, "the second request paid for the download again")

    def test_the_marker_expires(self):
        redis = self.fake_redis()
        self.raise_from_regeneration(self.undecodable("decode_failed"))
        client = self.client_with(FakeStorage({}), raise_server_exceptions=False)

        client.get("/api/photos/2/thumbnail")

        self.assertEqual(1, len(redis.sets))
        key, _value, ex = redis.sets[0]
        self.assertIn("2", key)
        self.assertIsNotNone(ex, "a permanent marker would outlive a reprocess")
        self.assertGreater(ex, 0)

    def test_a_transient_failure_is_not_marked(self):
        """source_unreadable may be the disk, not the bytes — let it retry."""
        redis = self.fake_redis()
        calls = self.count_regenerations(self.undecodable("source_unreadable"))
        client = self.client_with(FakeStorage({}), raise_server_exceptions=False)

        client.get("/api/photos/2/thumbnail")
        client.get("/api/photos/2/thumbnail")

        self.assertEqual([], redis.sets)
        self.assertEqual([2, 2], calls)

    def test_a_dead_redis_does_not_turn_the_422_into_a_500(self):
        self.fake_redis(broken=True)
        self.raise_from_regeneration(self.undecodable("decode_failed"))
        client = self.client_with(FakeStorage({}), raise_server_exceptions=False)

        resp = client.get("/api/photos/2/thumbnail")

        self.assertEqual(422, resp.status_code, resp.text)

    def test_the_marker_never_touches_the_happy_path(self):
        """A served thumbnail must not pay a Redis round-trip."""
        redis = self.fake_redis()
        client = self.client_with(FakeStorage({PHOTO_THUMB: JPEG}))

        resp = client.get("/api/photos/1/thumbnail")

        self.assertEqual(200, resp.status_code)
        self.assertEqual([], redis.sets)

    # -- a write failure is still a server fault ------------------------------
    def test_an_unwritable_thumbnail_is_still_a_500(self):
        """core.gallery raises ThumbnailUnavailable only for the *source*; a
        thumbnail it cannot write is a genuine server fault."""
        self.fake_redis()
        self.raise_from_regeneration(RuntimeError("Thumbnail write failed for /tmp/x.jpg"))
        client = self.client_with(FakeStorage({}), raise_server_exceptions=False)

        resp = client.get("/api/photos/2/thumbnail")

        self.assertEqual(500, resp.status_code)


if __name__ == "__main__":
    unittest.main()
