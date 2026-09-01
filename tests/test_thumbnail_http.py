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

    def client_with(self, storage):
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
        return TestClient(app)


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


if __name__ == "__main__":
    unittest.main()
