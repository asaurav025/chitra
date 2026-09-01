"""Video posters move out of the API and into a job.

Two of the six OOM kills of `chitra-api.service` named `av:hevc:df0` /
`av:hevc:df6` — ffmpeg HEVC frame-decode threads, running *inside the API
cgroup*, each allocating its own frame buffers. The API extracted a poster
keyframe inline on upload; a 4K HEVC decode there is unbounded memory in the
one process that must not be.

So: `extract_poster` gets an explicit thread cap, and the API enqueues rather
than decoding.

**Storage is stubbed throughout.** These tests never touch MinIO — the disk
holding it is failing, and a poster test has no business reading a multi-GB
original anyway. That constraint is also the point of Task 4.3: the job must
use `download_to_path`, which streams to a file, and never `download_file`,
which returns the whole video as `bytes`.
"""
import asyncio
import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from core import db, db_async, video


class RecordingRun:
    """Captures the argv ffmpeg would have been invoked with."""

    def __init__(self, returncode=0, touch_output=True):
        self.calls = []
        self.returncode = returncode
        self.touch_output = touch_output

    def __call__(self, cmd, *args, **kwargs):
        self.calls.append(cmd)
        if self.touch_output:
            # extract_poster checks the output exists before returning.
            with open(cmd[-1], "wb") as fh:
                fh.write(b"\xff\xd8\xff\xe0jpeg")

        class Proc:
            returncode = self.returncode
            stderr = "" if self.returncode == 0 else "boom"
            stdout = ""

        return Proc()


class TestExtractPosterFlags(unittest.TestCase):
    def run_extract(self, **kwargs):
        recorder = RecordingRun()
        out = tempfile.mktemp(suffix=".jpg")
        try:
            with patch.object(video.subprocess, "run", recorder):
                video.extract_poster("/nonexistent/source.mov", out, **kwargs)
        finally:
            if os.path.exists(out):
                os.unlink(out)
        self.assertEqual(1, len(recorder.calls))
        return recorder.calls[0]

    def test_caps_decode_threads_at_two(self):
        """The av:hevc:dfN OOM victims were frame-decode threads."""
        cmd = self.run_extract()
        self.assertIn("-threads", cmd)
        self.assertEqual("2", cmd[cmd.index("-threads") + 1])

    def test_thread_cap_precedes_the_input_so_it_applies_to_decode(self):
        """`-threads` after `-i` configures the *encoder* only. The decode is
        the expensive half and the half that OOM'd."""
        cmd = self.run_extract()
        self.assertLess(
            cmd.index("-threads"), cmd.index("-i"),
            f"-threads appears after -i in {cmd}; it would not cap the decoder",
        )

    def test_drops_audio_and_subtitle_streams(self):
        """A single keyframe needs neither; decoding them is pure waste."""
        cmd = self.run_extract()
        self.assertIn("-an", cmd)
        self.assertIn("-sn", cmd)

    def test_still_extracts_one_scaled_frame(self):
        cmd = self.run_extract()
        self.assertEqual("1", cmd[cmd.index("-frames:v") + 1])
        self.assertIn("-vf", cmd)

    def test_short_clips_still_seek_to_zero(self):
        cmd = self.run_extract(duration_seconds=0.5)
        self.assertEqual("0", cmd[cmd.index("-ss") + 1])


class FakeStorage:
    """A MinIO stand-in. `download_file` is a trap, not an implementation."""

    def __init__(self, existing=()):
        self.existing = set(existing)
        self.downloaded_to_path = []
        self.downloaded_bytes = []
        self.uploaded = []

    def generate_thumbnail_path(self, item_id, item_type="photo"):
        return f"thumbnails/photos/{item_id}.jpg"

    def file_exists(self, remote_path):
        return remote_path in self.existing

    def download_to_path(self, remote_path, local_path):
        self.downloaded_to_path.append((remote_path, local_path))
        with open(local_path, "wb") as fh:
            fh.write(b"fake video bytes")
        return local_path

    def download_file(self, remote_path):
        raise AssertionError(
            "download_file pulls the entire video into RAM — a multi-GB "
            "original would blow the worker. Use download_to_path."
        )

    def upload_file_from_path(self, local_path, remote_path):
        with open(local_path, "rb") as fh:
            self.uploaded.append((remote_path, len(fh.read())))
        self.existing.add(remote_path)
        return remote_path

    def upload_file(self, data, remote_path):
        self.uploaded.append((remote_path, len(data)))
        self.existing.add(remote_path)
        return remote_path

    # --- the async surface the HTTP layer uses ---
    async def file_exists_async(self, remote_path):
        return remote_path in self.existing

    async def download_file_async(self, remote_path):
        self.downloaded_bytes.append(remote_path)
        if remote_path not in self.existing:
            raise FileNotFoundError(remote_path)
        return b"\xff\xd8\xff\xe0poster-bytes"

    async def upload_file_async(self, data, remote_path):
        return self.upload_file(data, remote_path)


class TestGenerateVideoPosterJob(unittest.TestCase):
    def setUp(self):
        self.db_path = tempfile.mktemp(suffix=".db")
        asyncio.run(db_async.init_db_async(self.db_path))
        conn = db.connect(self.db_path)
        conn.execute(
            "INSERT INTO photos (file_path, size, created_at, checksum, media_type) "
            "VALUES (?, 1, '2026-01-01', 'abc', 'video')",
            ("videos/clip.mov",),
        )
        conn.commit()
        cur = conn.execute("SELECT id FROM photos WHERE file_path=?", ("videos/clip.mov",))
        self.photo_id = cur.fetchone()[0]
        conn.close()
        self.addCleanup(lambda: os.path.exists(self.db_path) and os.unlink(self.db_path))

    def row(self):
        conn = db.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT * FROM photos WHERE id=?", (self.photo_id,))
        row = dict(cur.fetchone())
        conn.close()
        return row

    def run_job(self, storage, extract=None):
        from core import jobs

        recorder = extract or (lambda *a, **kw: open(a[1], "wb").write(b"jpeg"))
        with patch.object(jobs, "_get_storage_client", lambda: storage), \
             patch.object(video, "extract_poster", recorder), \
             patch.object(video, "ensure_ffmpeg", lambda: True):
            return jobs.generate_video_poster_job(
                self.photo_id, "videos/clip.mov", self.db_path
            )

    def test_streams_the_source_to_a_file_never_into_memory(self):
        storage = FakeStorage()
        self.run_job(storage)
        self.assertEqual(1, len(storage.downloaded_to_path))
        self.assertEqual("videos/clip.mov", storage.downloaded_to_path[0][0])

    def test_uploads_the_poster_and_records_thumb_path(self):
        storage = FakeStorage()
        self.run_job(storage)
        expected = f"thumbnails/photos/{self.photo_id}.jpg"
        self.assertIn(expected, [key for key, _ in storage.uploaded])
        self.assertEqual(expected, self.row()["thumb_path"])

    def test_is_a_no_op_when_the_poster_already_exists(self):
        """A grid of 50 videos can fire 50 enqueues; the 50th must be cheap."""
        storage = FakeStorage(existing=[f"thumbnails/photos/{self.photo_id}.jpg"])
        self.run_job(storage)
        self.assertEqual([], storage.downloaded_to_path, "downloaded a video it did not need")
        self.assertEqual([], storage.uploaded)

    def test_never_touches_transcode_status(self):
        """A concurrent transcode owns that column; colliding strands the video."""
        conn = db.connect(self.db_path)
        conn.execute(
            "UPDATE photos SET transcode_status='processing' WHERE id=?", (self.photo_id,)
        )
        conn.commit()
        conn.close()

        self.run_job(FakeStorage())

        self.assertEqual("processing", self.row()["transcode_status"])

    def test_reraises_on_failure_rather_than_returning_false(self):
        """AGENTS.md: swallowing makes RQ mark a failed job successful."""
        def boom(*args, **kwargs):
            raise RuntimeError("ffmpeg poster extraction failed")

        with self.assertRaises(RuntimeError):
            self.run_job(FakeStorage(), extract=boom)

    def test_cleans_up_its_temp_files(self):
        storage = FakeStorage()
        self.run_job(storage)
        for _, local_path in storage.downloaded_to_path:
            self.assertFalse(os.path.exists(local_path), f"{local_path} left behind")


class TestApiHasNoFfmpeg(unittest.TestCase):
    """Task 4.6: no ffmpeg process may ever be a descendant of the API."""

    @classmethod
    def setUpClass(cls):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(repo_root, "app_fastapi.py")) as fh:
            cls.source = fh.read()

    def test_does_not_call_extract_poster(self):
        self.assertNotIn("extract_poster", self.source)

    def test_ensure_video_poster_async_is_gone(self):
        self.assertNotIn("ensure_video_poster_async", self.source)

    def test_the_only_video_module_use_left_is_the_ffmpeg_probe(self):
        """`video.ensure_ffmpeg()` is a `shutil.which` — it spawns nothing."""
        uses = {
            line.strip()
            for line in self.source.splitlines()
            if "video." in line and not line.strip().startswith("#")
        }
        offenders = [u for u in uses if "ensure_ffmpeg" not in u]
        self.assertEqual(
            [], offenders,
            f"app_fastapi still reaches into core.video for more than the "
            f"startup probe: {offenders}",
        )


if __name__ == "__main__":
    unittest.main()


class FakeRedis:
    """Just enough Redis for the SETNX dedupe."""

    def __init__(self):
        self.store = {}

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self.store:
            return None
        self.store[key] = value
        return True


class FakeQueue:
    def __init__(self):
        self.enqueued = []

    def enqueue(self, fn, *args, **kwargs):
        self.enqueued.append((fn, args, kwargs))
        return type("Job", (), {"id": "fake-job-id"})()


class TestPosterEnqueueDedupe(unittest.TestCase):
    """One grid render of 50 videos must not fire 50 identical jobs.

    The thumbnail GET enqueues on a miss, and every tile in a gallery misses at
    the same moment. Redis SETNX is the cheapest possible guard.
    """

    def setUp(self):
        import app_fastapi

        self.app_fastapi = app_fastapi
        self.redis = FakeRedis()
        self.queue = FakeQueue()
        patches = [
            patch.object(app_fastapi, "get_redis_connection", lambda: self.redis),
            patch.object(app_fastapi, "get_queue", lambda name="default": self.queue),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_enqueues_the_poster_job_once(self):
        self.assertTrue(self.app_fastapi.enqueue_video_poster(7, "videos/clip.mov"))
        self.assertEqual(1, len(self.queue.enqueued))
        fn, args, _ = self.queue.enqueued[0]
        self.assertEqual("generate_video_poster_job", fn.__name__)
        self.assertEqual(7, args[0])
        self.assertEqual("videos/clip.mov", args[1])

    def test_second_request_in_the_window_does_not_re_enqueue(self):
        self.app_fastapi.enqueue_video_poster(7, "videos/clip.mov")
        self.assertFalse(self.app_fastapi.enqueue_video_poster(7, "videos/clip.mov"))
        self.assertEqual(1, len(self.queue.enqueued))

    def test_different_photos_are_independent(self):
        self.app_fastapi.enqueue_video_poster(7, "a.mov")
        self.app_fastapi.enqueue_video_poster(8, "b.mov")
        self.assertEqual(2, len(self.queue.enqueued))

    def test_the_dedupe_key_expires(self):
        """A permanently-set key would mean a failed job is never retried."""
        self.app_fastapi.enqueue_video_poster(7, "a.mov")
        self.assertTrue(
            all(v is not None for v in self.redis.store.values()),
            "dedupe key was written without a TTL",
        )


class TestThumbnailGetForVideos(unittest.TestCase):
    """Today PIL opens an mp4 and 500s *after* downloading the whole video."""

    @classmethod
    def setUpClass(cls):
        import app_fastapi  # noqa: F401

    def setUp(self):
        from fastapi.testclient import TestClient
        import app_fastapi

        self.app_fastapi = app_fastapi
        self.db_path = tempfile.mktemp(suffix=".db")
        asyncio.run(db_async.init_db_async(self.db_path))
        self.queue = FakeQueue()
        self.redis = FakeRedis()

        for target, repl in (
            ("get_redis_connection", lambda: self.redis),
            ("get_queue", lambda name="default": self.queue),
        ):
            p = patch.object(app_fastapi, target, repl)
            p.start()
            self.addCleanup(p.stop)

        app = app_fastapi.app

        async def fake_db():
            async with db_async.connect_async(self.db_path) as conn:
                yield conn

        app.dependency_overrides[app_fastapi.get_db_async] = fake_db
        app.dependency_overrides[app_fastapi.get_current_active_user] = lambda: {"id": 1}
        self.storage = FakeStorage()
        app.dependency_overrides[app_fastapi.get_storage_client] = lambda: self.storage
        self.addCleanup(app.dependency_overrides.clear)
        self.addCleanup(lambda: os.path.exists(self.db_path) and os.unlink(self.db_path))
        self.client = TestClient(app)

    def add_photo(self, media_type, thumb_path=None, file_path="videos/clip.mov"):
        conn = db.connect(self.db_path)
        conn.execute(
            "INSERT INTO photos (file_path, size, created_at, checksum, media_type, thumb_path) "
            "VALUES (?, 1, '2026-01-01', ?, ?, ?)",
            (file_path, file_path, media_type, thumb_path),
        )
        conn.commit()
        cur = conn.execute("SELECT id FROM photos WHERE file_path=?", (file_path,))
        pid = cur.fetchone()[0]
        conn.close()
        return pid

    def test_video_without_a_poster_404s_with_retry_after(self):
        pid = self.add_photo("video")

        resp = self.client.get(f"/api/photos/{pid}/thumbnail")

        self.assertEqual(404, resp.status_code)
        self.assertEqual("3", resp.headers.get("Retry-After"))

    def test_video_without_a_poster_enqueues_one(self):
        pid = self.add_photo("video")

        self.client.get(f"/api/photos/{pid}/thumbnail")

        self.assertEqual(1, len(self.queue.enqueued))
        self.assertEqual("generate_video_poster_job", self.queue.enqueued[0][0].__name__)

    def test_a_grid_of_repeated_requests_enqueues_once(self):
        pid = self.add_photo("video")

        for _ in range(5):
            self.client.get(f"/api/photos/{pid}/thumbnail")

        self.assertEqual(1, len(self.queue.enqueued))

    def test_video_never_downloads_the_original_to_thumbnail_it(self):
        """The old path downloaded the whole video, then 500'd when PIL tried
        to open an mp4. Nothing may be fetched on this path at all."""
        pid = self.add_photo("video")

        self.client.get(f"/api/photos/{pid}/thumbnail")

        self.assertEqual([], self.storage.downloaded_bytes)
        self.assertEqual([], self.storage.downloaded_to_path)

    def test_null_media_type_is_treated_as_a_photo(self):
        """766 legacy rows have media_type NULL and must keep working."""
        pid = self.add_photo(None, file_path="legacy/photo.jpg")

        resp = self.client.get(f"/api/photos/{pid}/thumbnail")

        # It must go down the photo path, not the video 404-and-enqueue path.
        self.assertEqual([], self.queue.enqueued)
        self.assertNotEqual(
            "3", resp.headers.get("Retry-After"),
            "a legacy NULL media_type row was treated as a video",
        )

    def test_a_video_with_a_poster_is_served_normally(self):
        """The 404 is only for a poster that does not exist yet."""
        pid = self.add_photo("video", thumb_path="thumbnails/photos/1.jpg")
        self.storage.existing.add("thumbnails/photos/1.jpg")

        resp = self.client.get(f"/api/photos/{pid}/thumbnail")

        self.assertEqual(200, resp.status_code, resp.text)
        self.assertEqual(b"\xff\xd8\xff\xe0poster-bytes", resp.content)
        self.assertEqual([], self.queue.enqueued)
