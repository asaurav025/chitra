"""
Decision logic for `scripts/requeue_transcodes.py`.

A transcode job that is SIGTERM'd mid-flight (the `stop_workers.sh` double-TERM
hazard) never reaches its own `except` block, so the row keeps the
`transcode_status="processing"` the job set on entry. Nothing retries it and
nothing times it out: the video is unplayable forever.

Rows in that state are not all the same, and the difference matters because
re-transcoding 4K HEVC on this box is expensive:

    (a) the playback derivative is already in MinIO — the ffmpeg work finished
        and only the final status write was lost. Correct the status; do NOT
        re-transcode.
    (b) the original is gone from MinIO — no amount of retrying will help.
        Mark it failed with a reason so it stops being a mystery.
    (c) everything else — genuinely needs the work. Reset to pending, enqueue.

These tests pin that classification, the safety interlock that refuses to touch
a row with a live RQ job behind it, and the guarantee that a dry run mutates
nothing. The storage boundary is stubbed: the classifier must not require a
live MinIO, and the suite must not depend on one.
"""
import os
import sqlite3
import tempfile
import unittest

from scripts import requeue_transcodes as rq


class FakeStorage:
    """Stands in for MinIOStorageClient — only the two methods the script uses.

    `objects` maps object key -> size in bytes.
    """

    def __init__(self, objects=None):
        self.objects = dict(objects or {})

    def generate_playback_path(self, photo_id):
        return f"videos/playback/{photo_id}.mp4"

    def stat_object_size(self, key):
        if key not in self.objects:
            raise FileNotFoundError(f"File not found in MinIO: {key}")
        return self.objects[key]


class FakeQueue:
    """Records enqueue calls instead of talking to Redis."""

    def __init__(self):
        self.calls = []

    def enqueue(self, func, *args, **kwargs):
        self.calls.append((func, args, kwargs))
        return object()


def make_row(photo_id=1, file_path="photos/2026/09/a.mov", media_type="video",
             transcode_status="processing", playback_path=None,
             created_at="2026-01-01T00:00:00"):
    return {
        "id": photo_id,
        "file_path": file_path,
        "media_type": media_type,
        "transcode_status": transcode_status,
        "playback_path": playback_path,
        "created_at": created_at,
    }


class TestClassify(unittest.TestCase):
    """Which population does a row fall into?"""

    def test_playback_object_present_means_status_correction(self):
        """(a) The ffmpeg work finished; only the status write was lost."""
        row = make_row(photo_id=7)
        storage = FakeStorage({"videos/playback/7.mp4": 5_000_000})

        decision = rq.classify_row(row, storage, active_ids=set())

        self.assertEqual(decision.action, rq.ACTION_READY)
        self.assertEqual(decision.playback_key, "videos/playback/7.mp4")

    def test_missing_original_means_failed_not_retry(self):
        """(b) Retrying a video whose source is gone loops forever."""
        row = make_row(photo_id=8, file_path="photos/gone.mov")
        storage = FakeStorage({})  # neither playback nor original

        decision = rq.classify_row(row, storage, active_ids=set())

        self.assertEqual(decision.action, rq.ACTION_FAILED)
        self.assertIn("original", decision.reason.lower())

    def test_original_present_and_no_playback_means_requeue(self):
        """(c) The common case: genuinely needs the work."""
        row = make_row(photo_id=9, file_path="photos/2026/09/b.mov")
        storage = FakeStorage({"photos/2026/09/b.mov": 94_000_000})

        decision = rq.classify_row(row, storage, active_ids=set())

        self.assertEqual(decision.action, rq.ACTION_REQUEUE)

    def test_live_job_is_never_touched(self):
        """The safety interlock: a running transcode must not be stolen."""
        row = make_row(photo_id=10, file_path="photos/c.mov")
        storage = FakeStorage({"photos/c.mov": 1000})

        decision = rq.classify_row(row, storage, active_ids={10})

        self.assertEqual(decision.action, rq.ACTION_SKIP)
        self.assertIn("active", decision.reason.lower())

    def test_zero_byte_playback_is_not_ready(self):
        """A SIGTERM mid-upload can leave an empty object; that is not done."""
        row = make_row(photo_id=11, file_path="photos/d.mov")
        storage = FakeStorage(
            {"videos/playback/11.mp4": 0, "photos/d.mov": 1000}
        )

        decision = rq.classify_row(row, storage, active_ids=set())

        self.assertEqual(decision.action, rq.ACTION_REQUEUE)

    def test_existing_playback_path_column_is_preferred(self):
        """If the row names its own derivative, check that key, not the default."""
        row = make_row(photo_id=12, playback_path="videos/playback/custom.mp4")
        storage = FakeStorage({"videos/playback/custom.mp4": 2_000_000})

        decision = rq.classify_row(row, storage, active_ids=set())

        self.assertEqual(decision.action, rq.ACTION_READY)
        self.assertEqual(decision.playback_key, "videos/playback/custom.mp4")

    def test_non_video_row_is_skipped(self):
        row = make_row(photo_id=13, media_type="photo")
        storage = FakeStorage({})

        decision = rq.classify_row(row, storage, active_ids=set())

        self.assertEqual(decision.action, rq.ACTION_SKIP)


class _TempDBMixin:
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(prefix="chitra_requeue_", suffix=".db")
        os.close(fd)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(
            """CREATE TABLE photos (
                id INTEGER PRIMARY KEY,
                file_path TEXT,
                created_at TEXT,
                media_type TEXT,
                playback_path TEXT,
                transcode_status TEXT,
                video_codec TEXT
            )"""
        )
        self.conn.commit()

    def tearDown(self):
        self.conn.close()
        for suffix in ("", "-wal", "-shm"):
            path = self.db_path + suffix
            if os.path.exists(path):
                os.unlink(path)

    def insert(self, photo_id, **kw):
        row = make_row(photo_id=photo_id, **kw)
        self.conn.execute(
            "INSERT INTO photos (id, file_path, created_at, media_type,"
            " playback_path, transcode_status) VALUES (?,?,?,?,?,?)",
            (row["id"], row["file_path"], row["created_at"], row["media_type"],
             row["playback_path"], row["transcode_status"]),
        )
        self.conn.commit()

    def status_of(self, photo_id):
        return self.conn.execute(
            "SELECT transcode_status FROM photos WHERE id=?", (photo_id,)
        ).fetchone()[0]


class TestSelectRows(_TempDBMixin, unittest.TestCase):
    def test_stuck_selects_only_processing_videos(self):
        self.insert(1, transcode_status="processing")
        self.insert(2, transcode_status="ready")
        self.insert(3, transcode_status="processing", media_type="photo")

        rows = rq.select_rows(self.conn, stuck=True)

        self.assertEqual([r["id"] for r in rows], [1])

    def test_limit_keeps_batches_small(self):
        for pid in (1, 2, 3, 4):
            self.insert(pid, transcode_status="processing")

        rows = rq.select_rows(self.conn, stuck=True, limit=2)

        self.assertEqual([r["id"] for r in rows], [1, 2])

    def test_explicit_ids_ignore_status_filter(self):
        self.insert(1, transcode_status="failed")

        rows = rq.select_rows(self.conn, ids=[1])

        self.assertEqual([r["id"] for r in rows], [1])


class TestApply(_TempDBMixin, unittest.TestCase):
    def test_dry_run_mutates_nothing(self):
        self.insert(1, file_path="photos/a.mov", transcode_status="processing")
        storage = FakeStorage({"photos/a.mov": 1000})
        queue = FakeQueue()

        rows = rq.select_rows(self.conn, stuck=True)
        decisions = [rq.classify_row(r, storage, set()) for r in rows]
        rq.apply_decisions(self.conn, decisions, queue, "photo.db", dry_run=True)

        self.assertEqual(self.status_of(1), "processing")
        self.assertEqual(queue.calls, [], "dry run must not enqueue")

    def test_apply_requeue_sets_pending_and_enqueues(self):
        self.insert(1, file_path="photos/a.mov", transcode_status="processing")
        storage = FakeStorage({"photos/a.mov": 1000})
        queue = FakeQueue()

        rows = rq.select_rows(self.conn, stuck=True)
        decisions = [rq.classify_row(r, storage, set()) for r in rows]
        rq.apply_decisions(self.conn, decisions, queue, "photo.db", dry_run=False)

        self.assertEqual(self.status_of(1), "pending")
        self.assertEqual(len(queue.calls), 1)
        _func, args, kwargs = queue.calls[0]
        self.assertEqual(args, (1, "photos/a.mov", "photo.db"))
        self.assertEqual(kwargs.get("job_timeout"), "2h")

    def test_apply_ready_corrects_status_without_enqueueing(self):
        """Population (a) must never cost a re-transcode."""
        self.insert(1, file_path="photos/a.mov", transcode_status="processing")
        storage = FakeStorage({"videos/playback/1.mp4": 5000})
        queue = FakeQueue()

        rows = rq.select_rows(self.conn, stuck=True)
        decisions = [rq.classify_row(r, storage, set()) for r in rows]
        rq.apply_decisions(self.conn, decisions, queue, "photo.db", dry_run=False)

        self.assertEqual(self.status_of(1), "ready")
        self.assertEqual(queue.calls, [], "a finished video must not be redone")
        playback = self.conn.execute(
            "SELECT playback_path FROM photos WHERE id=1"
        ).fetchone()[0]
        self.assertEqual(playback, "videos/playback/1.mp4")

    def test_apply_failed_marks_missing_original(self):
        self.insert(1, file_path="photos/gone.mov", transcode_status="processing")
        storage = FakeStorage({})
        queue = FakeQueue()

        rows = rq.select_rows(self.conn, stuck=True)
        decisions = [rq.classify_row(r, storage, set()) for r in rows]
        rq.apply_decisions(self.conn, decisions, queue, "photo.db", dry_run=False)

        self.assertEqual(self.status_of(1), "failed")
        self.assertEqual(queue.calls, [])

    def test_rerun_is_idempotent(self):
        """After a requeue the row is pending, so --stuck finds nothing again."""
        self.insert(1, file_path="photos/a.mov", transcode_status="processing")
        storage = FakeStorage({"photos/a.mov": 1000})
        queue = FakeQueue()

        rows = rq.select_rows(self.conn, stuck=True)
        decisions = [rq.classify_row(r, storage, set()) for r in rows]
        rq.apply_decisions(self.conn, decisions, queue, "photo.db", dry_run=False)

        second = rq.select_rows(self.conn, stuck=True)

        self.assertEqual(second, [])
        self.assertEqual(len(queue.calls), 1, "second run must not re-enqueue")


if __name__ == "__main__":
    unittest.main()
