"""
Decision logic for `scripts/reembed.py`.

Re-embedding is a bulk read campaign against a disk that is actively failing
(1,950+ unrecovered read errors and climbing), so almost every test here is
about *not doing work*:

  * a photo that already carries a current-model vector is not re-read;
  * a 512 px thumbnail is preferred over a multi-MB HEIC/ARW original, which is
    ~20x less data off the bad disk and skips the RAW decode entirely;
  * a photo already recorded done in the journal is not re-read after a Ctrl-C;
  * a photo that failed to read once is not retried by default — the bad
    sectors are not going to heal;
  * the whole pass aborts if the kernel's medium-error counter climbs during
    the run.

The storage boundary and the embedder are stubbed throughout: the suite must
not need MinIO, a live sidecar, or a 1.1 GB CLIP load.
"""
import json
import os
import sqlite3
import tempfile
import unittest

from core import db
from scripts import reembed


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------
class FakeStorage:
    """Only the two methods the script uses. `objects` maps key -> bytes."""

    def __init__(self, objects=None):
        self.objects = dict(objects or {})
        self.reads = []

    def download_file(self, key):
        self.reads.append(key)
        if key not in self.objects:
            raise FileNotFoundError(f"File not found in MinIO: {key}")
        data = self.objects[key]
        if isinstance(data, Exception):
            raise data
        return data

    def stat_object_size(self, key):
        if key not in self.objects:
            raise FileNotFoundError(f"File not found in MinIO: {key}")
        return len(self.objects[key])


class FakeEmbedder:
    """Deterministic stand-in for CLIP. Records what it was asked to embed."""

    name = "fake/model-v1"
    dim = 4

    def __init__(self, fail_on=()):
        self.calls = []
        self.fail_on = set(fail_on)

    def image_embedding(self, filename, data):
        self.calls.append(filename)
        if filename in self.fail_on:
            raise RuntimeError("decode failed")
        import numpy as np
        v = np.arange(self.dim, dtype="float32") + float(len(data) % 7)
        return v / (np.linalg.norm(v) + 1e-9)

    def rank_labels(self, filename, data, labels, top_k):
        return [(lab, 0.5 - i * 0.01) for i, lab in enumerate(labels[:top_k])]


def make_db(rows, embeddings=(), tags=(), migrate=True):
    """Build a throwaway DB with just the columns the script reads."""
    fd, path = tempfile.mkstemp(suffix=".db", prefix="reembed-test-")
    os.close(fd)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE photos (
            id INTEGER PRIMARY KEY,
            file_path TEXT UNIQUE,
            size INTEGER,
            thumb_path TEXT,
            media_type TEXT
        );
        CREATE TABLE embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            photo_id INTEGER NOT NULL,
            dim INTEGER NOT NULL,
            vector BLOB NOT NULL
        );
        CREATE TABLE tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            photo_id INTEGER NOT NULL,
            tag TEXT NOT NULL,
            score REAL NOT NULL
        );
        """
    )
    # The writers upsert on (photo_id, model) and (photo_id, tag), so the test
    # DB has to carry the same migration a real one gets at startup.
    if migrate:
        db.migrate_embeddings_and_tags(conn)
    for r in rows:
        conn.execute(
            "INSERT INTO photos (id, file_path, size, thumb_path, media_type) VALUES (?,?,?,?,?)",
            (r["id"], r.get("file_path", f"photos/{r['id']}.heic"), r.get("size", 4_000_000),
             r.get("thumb_path"), r.get("media_type")),
        )
    for photo_id, dim in embeddings:
        conn.execute(
            "INSERT INTO embeddings (photo_id, dim, vector, model) VALUES (?,?,?,?)",
            (photo_id, dim, b"\x00" * (dim * 4), FakeEmbedder.name),
        )
    for photo_id, tag in tags:
        conn.execute("INSERT INTO tags (photo_id, tag, score) VALUES (?,?,?)", (photo_id, tag, 0.5))
    conn.commit()
    return conn, path


def photo(pid, thumb=True, media_type=None, ext="heic"):
    return {
        "id": pid,
        "file_path": f"photos/2026/01/p{pid}.{ext}",
        "thumb_path": f"thumbnails/photos/{pid}.jpg" if thumb else None,
        "media_type": media_type,
    }


# ----------------------------------------------------------------------
# source selection
# ----------------------------------------------------------------------
class TestSourceSelection(unittest.TestCase):
    """Which object does a photo get read from?"""

    def _decide(self, row, source_mode):
        conn, path = make_db([row])
        try:
            r = conn.execute("SELECT * FROM photos WHERE id=?", (row["id"],)).fetchone()
            return reembed.classify_row(r, source_mode=source_mode, force=False,
                                        done_ids=set(), failed_ids=set(),
                                        model=FakeEmbedder.name, has_embedding=False)
        finally:
            conn.close()
            os.unlink(path)

    def test_auto_prefers_the_thumbnail(self):
        """~237 KB of JPEG instead of ~4.6 MB of HEIC off a failing disk."""
        d = self._decide(photo(1, thumb=True), reembed.SOURCE_AUTO)
        self.assertEqual(d.action, reembed.ACTION_EMBED)
        self.assertEqual(d.source, reembed.SOURCE_THUMB)
        self.assertEqual(d.key, "thumbnails/photos/1.jpg")

    def test_auto_falls_back_to_the_original_when_there_is_no_thumbnail(self):
        d = self._decide(photo(2, thumb=False), reembed.SOURCE_AUTO)
        self.assertEqual(d.action, reembed.ACTION_EMBED)
        self.assertEqual(d.source, reembed.SOURCE_ORIGINAL)
        self.assertEqual(d.key, "photos/2026/01/p2.heic")

    def test_thumb_mode_skips_a_photo_with_no_thumbnail(self):
        """--source thumb is a hard cap on read volume, not a preference."""
        d = self._decide(photo(3, thumb=False), reembed.SOURCE_THUMB)
        self.assertEqual(d.action, reembed.ACTION_SKIP)
        self.assertIn("thumbnail", d.reason)

    def test_original_mode_ignores_an_available_thumbnail(self):
        d = self._decide(photo(4, thumb=True), reembed.SOURCE_ORIGINAL)
        self.assertEqual(d.source, reembed.SOURCE_ORIGINAL)
        self.assertEqual(d.key, "photos/2026/01/p4.heic")


# ----------------------------------------------------------------------
# skip / force
# ----------------------------------------------------------------------
class TestSkipAndForce(unittest.TestCase):
    def _decide(self, row, *, force=False, has_embedding=False, done_ids=(), failed_ids=()):
        conn, path = make_db([row])
        try:
            r = conn.execute("SELECT * FROM photos WHERE id=?", (row["id"],)).fetchone()
            return reembed.classify_row(r, source_mode=reembed.SOURCE_AUTO, force=force,
                                        done_ids=set(done_ids), failed_ids=set(failed_ids),
                                        model=FakeEmbedder.name, has_embedding=has_embedding)
        finally:
            conn.close()
            os.unlink(path)

    def test_video_is_never_embedded(self):
        """Jobs bail on videos; a bulk pass must not quietly disagree."""
        d = self._decide(photo(1, media_type="video"))
        self.assertEqual(d.action, reembed.ACTION_SKIP)
        self.assertIn("video", d.reason)

    def test_existing_current_model_embedding_is_skipped(self):
        d = self._decide(photo(1), has_embedding=True)
        self.assertEqual(d.action, reembed.ACTION_SKIP)
        self.assertIn("embedding", d.reason)

    def test_force_re_embeds_a_photo_that_already_has_one(self):
        d = self._decide(photo(1), has_embedding=True, force=True)
        self.assertEqual(d.action, reembed.ACTION_EMBED)

    def test_force_does_not_resurrect_a_video(self):
        """--force means 'redo every image', not 'embed things that cannot be'."""
        d = self._decide(photo(1, media_type="video"), has_embedding=True, force=True)
        self.assertEqual(d.action, reembed.ACTION_SKIP)

    def test_journal_done_is_skipped_even_under_force(self):
        """Resumability: a Ctrl-C'd --force run must not restart from zero."""
        d = self._decide(photo(9), has_embedding=True, force=True, done_ids=[9])
        self.assertEqual(d.action, reembed.ACTION_SKIP)
        self.assertIn("already", d.reason)

    def test_journal_failure_is_not_retried_by_default(self):
        """A bad sector does not heal; re-reading it only grows the error count."""
        d = self._decide(photo(9), failed_ids=[9])
        self.assertEqual(d.action, reembed.ACTION_SKIP)
        self.assertIn("failed", d.reason)


class TestPlanSelection(unittest.TestCase):
    """`plan()` against a whole table, including the has-embedding lookup."""

    def setUp(self):
        rows = [photo(1), photo(2), photo(3, thumb=False), photo(4, media_type="video")]
        self.conn, self.path = make_db(rows, embeddings=[(1, 4)])

    def tearDown(self):
        self.conn.close()
        os.unlink(self.path)

    def _plan(self, **kw):
        kw.setdefault("source_mode", reembed.SOURCE_AUTO)
        kw.setdefault("force", False)
        kw.setdefault("done_ids", set())
        kw.setdefault("failed_ids", set())
        kw.setdefault("model", FakeEmbedder.name)
        return reembed.plan(self.conn, **kw)

    def test_only_the_gaps_are_embedded(self):
        decisions = self._plan()
        embed_ids = [d.photo_id for d in decisions if d.action == reembed.ACTION_EMBED]
        self.assertEqual(embed_ids, [2, 3])

    def test_force_covers_every_image_but_not_the_video(self):
        decisions = self._plan(force=True)
        embed_ids = [d.photo_id for d in decisions if d.action == reembed.ACTION_EMBED]
        self.assertEqual(embed_ids, [1, 2, 3])

    def test_ids_filter_narrows_the_pass(self):
        decisions = self._plan(ids=[3])
        self.assertEqual([d.photo_id for d in decisions], [3])

    def test_limit_caps_the_work_not_the_scan(self):
        """--limit must cap photos actually read, so a cautious first batch is
        a batch of that many reads and not a batch of mostly-skips."""
        decisions = self._plan(force=True, limit=2)
        embed_ids = [d.photo_id for d in decisions if d.action == reembed.ACTION_EMBED]
        self.assertEqual(embed_ids, [1, 2])

    def test_a_vector_from_another_model_is_not_a_reason_to_skip(self):
        """The skip check keys on model, not on dimension: two models can have
        the same width, and the CLIP row must not make the SigLIP pass think
        photo 1 is already done."""
        self.conn.execute("UPDATE embeddings SET model='other/model-v9' WHERE photo_id=1")
        self.conn.commit()
        embed_ids = [d.photo_id for d in self._plan() if d.action == reembed.ACTION_EMBED]
        self.assertIn(1, embed_ids)

    def test_a_row_for_this_model_is_a_reason_to_skip_whatever_its_dim(self):
        self.conn.execute("UPDATE embeddings SET dim=768 WHERE photo_id=1")
        self.conn.commit()
        embed_ids = [d.photo_id for d in self._plan() if d.action == reembed.ACTION_EMBED]
        self.assertNotIn(1, embed_ids)


# ----------------------------------------------------------------------
# journal / resumability
# ----------------------------------------------------------------------
class TestJournal(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".jsonl", prefix="reembed-journal-")
        os.close(fd)
        os.unlink(self.path)

    def tearDown(self):
        if os.path.exists(self.path):
            os.unlink(self.path)

    def test_missing_journal_loads_as_empty(self):
        done, failed = reembed.load_journal(self.path, "m1")
        self.assertEqual((done, failed), (set(), {}))

    def test_done_records_survive_a_restart(self):
        j = reembed.Journal(self.path, model="m1", source_mode="auto", force=False)
        j.record_run(planned=2)
        j.record_done(7, source="thumb", dim=4)
        j.close()
        done, failed = reembed.load_journal(self.path, "m1")
        self.assertEqual(done, {7})
        self.assertEqual(failed, {})

    def test_records_are_scoped_to_the_model_that_wrote_them(self):
        """A model change is exactly when --force must redo everything."""
        j = reembed.Journal(self.path, model="m1", source_mode="auto", force=False)
        j.record_done(7, source="thumb", dim=4)
        j.close()
        done, _ = reembed.load_journal(self.path, "m2")
        self.assertEqual(done, set())

    def test_failures_are_recorded_with_their_reason(self):
        j = reembed.Journal(self.path, model="m1", source_mode="auto", force=False)
        j.record_fail(9, source="original", reason="IncompleteRead")
        j.close()
        done, failed = reembed.load_journal(self.path, "m1")
        self.assertEqual(done, set())
        self.assertIn(9, failed)
        self.assertIn("IncompleteRead", failed[9])

    def test_a_later_success_clears_an_earlier_failure(self):
        j = reembed.Journal(self.path, model="m1", source_mode="auto", force=False)
        j.record_fail(9, source="thumb", reason="boom")
        j.record_done(9, source="original", dim=4)
        j.close()
        done, failed = reembed.load_journal(self.path, "m1")
        self.assertEqual(done, {9})
        self.assertNotIn(9, failed)

    def test_a_truncated_final_line_does_not_lose_the_rest(self):
        """Ctrl-C during a write must not make the journal unreadable."""
        j = reembed.Journal(self.path, model="m1", source_mode="auto", force=False)
        j.record_done(1, source="thumb", dim=4)
        j.close()
        with open(self.path, "a") as fh:
            fh.write('{"type": "done", "photo_id": 2, "mod')
        done, _ = reembed.load_journal(self.path, "m1")
        self.assertEqual(done, {1})

    def test_the_journal_names_the_model_it_used(self):
        """There is no `model` column on `embeddings`; the run record is where
        'which model produced these vectors' is written down."""
        j = reembed.Journal(self.path, model="openai/clip-vit-base-patch32",
                            source_mode="thumb", force=True)
        j.record_run(planned=5)
        j.close()
        header = json.loads(open(self.path).readline())
        self.assertEqual(header["type"], "run")
        self.assertEqual(header["model"], "openai/clip-vit-base-patch32")
        self.assertEqual(header["source_mode"], "thumb")
        self.assertTrue(header["force"])


# ----------------------------------------------------------------------
# disk-health guard
# ----------------------------------------------------------------------
class TestDiskGuard(unittest.TestCase):
    def test_a_flat_error_count_never_trips(self):
        counts = iter([{"kern": 100, "ioerr": 50}] * 3)
        g = reembed.DiskGuard(threshold=10, reader=lambda: next(counts))
        g.start()
        self.assertIsNone(g.check())
        self.assertIsNone(g.check())

    def test_a_climb_inside_the_threshold_is_tolerated(self):
        counts = iter([{"kern": 100, "ioerr": 50}, {"kern": 105, "ioerr": 50}])
        g = reembed.DiskGuard(threshold=10, reader=lambda: next(counts))
        g.start()
        self.assertIsNone(g.check())

    def test_a_climb_past_the_threshold_aborts(self):
        counts = iter([{"kern": 100, "ioerr": 50}, {"kern": 100, "ioerr": 75}])
        g = reembed.DiskGuard(threshold=10, reader=lambda: next(counts))
        g.start()
        reason = g.check()
        self.assertIsNotNone(reason)
        self.assertIn("ioerr", reason)

    def test_an_unreadable_counter_is_ignored_rather_than_fatal(self):
        """kern.log needs group `adm`; a run as another user must still work."""
        counts = iter([{"kern": None, "ioerr": 50}, {"kern": None, "ioerr": 55}])
        g = reembed.DiskGuard(threshold=10, reader=lambda: next(counts))
        g.start()
        self.assertIsNone(g.check())

    def test_reader_parses_hex_ioerr_and_counts_kern_lines(self):
        d = tempfile.mkdtemp()
        kern = os.path.join(d, "kern.log")
        ioerr = os.path.join(d, "ioerr_cnt")
        with open(kern, "w") as fh:
            fh.write("boring\ncritical medium error, dev sda\ncritical medium error, dev sda\n")
        with open(ioerr, "w") as fh:
            fh.write("0x8fe\n")
        counts = reembed.read_disk_errors(kern_log=kern, ioerr_path=ioerr)
        self.assertEqual(counts["kern"], 2)
        self.assertEqual(counts["ioerr"], 0x8fe)

    def test_missing_sources_read_as_none_not_zero(self):
        """Zero would look like a healthy disk and silently disable the guard."""
        counts = reembed.read_disk_errors(kern_log="/nope/kern.log", ioerr_path="/nope/ioerr")
        self.assertIsNone(counts["kern"])
        self.assertIsNone(counts["ioerr"])


# ----------------------------------------------------------------------
# execution
# ----------------------------------------------------------------------
class TestRun(unittest.TestCase):
    def setUp(self):
        rows = [photo(1), photo(2), photo(3, thumb=False)]
        self.conn, self.path = make_db(rows)
        self.storage = FakeStorage({
            "thumbnails/photos/1.jpg": b"thumb-one",
            "thumbnails/photos/2.jpg": b"thumb-two",
            "photos/2026/01/p3.heic": b"original-three",
        })
        self.embedder = FakeEmbedder()
        fd, self.journal_path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        os.unlink(self.journal_path)

    def tearDown(self):
        self.conn.close()
        os.unlink(self.path)
        if os.path.exists(self.journal_path):
            os.unlink(self.journal_path)

    def _run(self, **kw):
        embedder = kw.pop("embedder", self.embedder)
        kw.setdefault("apply", False)
        kw.setdefault("source_mode", reembed.SOURCE_AUTO)
        kw.setdefault("force", False)
        kw.setdefault("delay", 0.0)
        kw.setdefault("tags", False)
        kw.setdefault("journal_path", self.journal_path)
        kw.setdefault("guard", reembed.DiskGuard(threshold=10, reader=lambda: {"kern": 0, "ioerr": 0}))
        return reembed.run(self.conn, self.storage, embedder, **kw)

    def _embedding_rows(self):
        return self.conn.execute(
            "SELECT photo_id, dim FROM embeddings ORDER BY photo_id").fetchall()

    # -- dry run --------------------------------------------------------
    def test_dry_run_writes_nothing_anywhere(self):
        result = self._run()
        self.assertEqual(self._embedding_rows(), [])
        self.assertEqual(self.storage.reads, [])
        self.assertEqual(self.embedder.calls, [])
        self.assertFalse(os.path.exists(self.journal_path))
        self.assertEqual(result.planned, 3)
        self.assertEqual(result.embedded, 0)

    # -- apply ----------------------------------------------------------
    def test_apply_embeds_every_gap(self):
        result = self._run(apply=True)
        self.assertEqual([r["photo_id"] for r in self._embedding_rows()], [1, 2, 3])
        self.assertEqual([r["dim"] for r in self._embedding_rows()], [4, 4, 4])
        self.assertEqual(result.embedded, 3)
        self.assertEqual(result.failed, [])

    def test_apply_reads_thumbnails_when_they_exist(self):
        self._run(apply=True)
        self.assertEqual(self.storage.reads,
                         ["thumbnails/photos/1.jpg", "thumbnails/photos/2.jpg",
                          "photos/2026/01/p3.heic"])

    def test_re_embedding_replaces_rather_than_duplicates(self):
        """`put_embedding` was a plain INSERT: a second pass would double every
        vector and silently double its weight in search."""
        self._run(apply=True)
        self._run(apply=True, force=True, reset_state=True)
        rows = self._embedding_rows()
        self.assertEqual(len(rows), 3)
        self.assertEqual([r["photo_id"] for r in rows], [1, 2, 3])

    def test_tags_are_replaced_not_appended(self):
        self._run(apply=True, tags=True)
        first = self.conn.execute("SELECT COUNT(*) FROM tags").fetchone()[0]
        self._run(apply=True, tags=True, force=True, reset_state=True)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM tags").fetchone()[0], first)

    def test_apply_migrates_an_unmigrated_schema(self):
        """The API and workers were launched before `embeddings.model` existed
        and cannot be restarted from here, so the first writer that needs the
        column is the one that has to add it."""
        conn, path = make_db([photo(1)], migrate=False)
        try:
            with self.assertRaises(sqlite3.OperationalError):
                conn.execute("SELECT model FROM embeddings")
            reembed.run(conn, self.storage, self.embedder, apply=True, delay=0.0,
                        tags=False, journal_path=self.journal_path,
                        guard=reembed.DiskGuard(threshold=10,
                                                reader=lambda: {"kern": 0, "ioerr": 0}))
            self.assertEqual(
                conn.execute("SELECT model FROM embeddings WHERE photo_id=1").fetchone()[0],
                FakeEmbedder.name)
        finally:
            conn.close()
            os.unlink(path)

    def test_a_dry_run_does_not_migrate_the_schema(self):
        conn, path = make_db([photo(1)], migrate=False)
        try:
            reembed.run(conn, self.storage, self.embedder, apply=False,
                        journal_path=self.journal_path)
            with self.assertRaises(sqlite3.OperationalError):
                conn.execute("SELECT model FROM embeddings")
        finally:
            conn.close()
            os.unlink(path)

    def test_the_model_is_written_into_the_embeddings_row(self):
        """`embeddings.model` is half the unique key. Without it Phase 6's
        cutover and rollback have nothing to key on."""
        self._run(apply=True)
        models = {r[0] for r in self.conn.execute("SELECT DISTINCT model FROM embeddings")}
        self.assertEqual(models, {FakeEmbedder.name})

    def test_a_second_model_lands_alongside_the_first(self):
        """The reason the key is (photo_id, model): writing SigLIP must not
        evict the CLIP row search is still answering from."""
        self._run(apply=True)
        self._run(apply=True, model="google/siglip2-base", reset_state=True)
        rows = self.conn.execute(
            "SELECT photo_id, model FROM embeddings ORDER BY photo_id, model").fetchall()
        self.assertEqual(len(rows), 6)
        self.assertEqual({r["model"] for r in rows},
                         {FakeEmbedder.name, "google/siglip2-base"})

    # -- failure handling ------------------------------------------------
    def test_an_unreadable_photo_does_not_stop_the_pass(self):
        self.storage.objects["thumbnails/photos/1.jpg"] = OSError("critical medium error")
        result = self._run(apply=True)
        self.assertEqual([r["photo_id"] for r in self._embedding_rows()], [2, 3])
        self.assertEqual([f[0] for f in result.failed], [1])
        self.assertIn("critical medium error", result.failed[0][1])

    def test_a_failed_photo_is_not_retried_on_the_next_run(self):
        self.storage.objects["thumbnails/photos/1.jpg"] = OSError("critical medium error")
        self._run(apply=True)
        self.storage.reads.clear()
        self._run(apply=True)
        self.assertNotIn("thumbnails/photos/1.jpg", self.storage.reads)

    def test_retry_failed_re_reads_it_deliberately(self):
        self.storage.objects["thumbnails/photos/1.jpg"] = OSError("critical medium error")
        self._run(apply=True)
        self.storage.objects["thumbnails/photos/1.jpg"] = b"thumb-one"
        self.storage.reads.clear()
        self._run(apply=True, retry_failed=True)
        self.assertIn("thumbnails/photos/1.jpg", self.storage.reads)
        self.assertEqual([r["photo_id"] for r in self._embedding_rows()], [1, 2, 3])

    def test_an_embedder_failure_is_also_non_fatal(self):
        embedder = FakeEmbedder(fail_on=["p3.heic"])
        result = self._run(apply=True, embedder=embedder)
        self.assertEqual([f[0] for f in result.failed], [3])
        self.assertEqual(result.embedded, 2)

    # -- resumability ----------------------------------------------------
    def test_a_second_run_re_reads_nothing(self):
        self._run(apply=True)
        self.storage.reads.clear()
        result = self._run(apply=True)
        self.assertEqual(self.storage.reads, [])
        self.assertEqual(result.embedded, 0)

    def test_force_resumes_from_the_journal_after_an_interrupt(self):
        """Half a --force pass then Ctrl-C: the restart must not redo the half
        that already cost real reads off the failing disk."""
        self._run(apply=True, force=True, limit=2)
        self.storage.reads.clear()
        self._run(apply=True, force=True)
        self.assertEqual(self.storage.reads, ["photos/2026/01/p3.heic"])

    def test_reset_state_deliberately_starts_over(self):
        self._run(apply=True, force=True)
        self.storage.reads.clear()
        self._run(apply=True, force=True, reset_state=True)
        self.assertEqual(len(self.storage.reads), 3)

    # -- the guard -------------------------------------------------------
    def test_a_climbing_error_count_aborts_the_pass(self):
        counts = iter([
            {"kern": 100, "ioerr": 0},
            {"kern": 100, "ioerr": 0},
            {"kern": 500, "ioerr": 0},
            {"kern": 500, "ioerr": 0},
        ])
        guard = reembed.DiskGuard(threshold=10, reader=lambda: next(counts))
        result = self._run(apply=True, guard=guard, check_every=1)
        self.assertTrue(result.aborted)
        self.assertIn("kern", result.abort_reason)
        self.assertLess(result.embedded, 3)

    def test_an_abort_keeps_the_work_already_done(self):
        counts = iter([{"kern": 0, "ioerr": 0}, {"kern": 0, "ioerr": 0}, {"kern": 900, "ioerr": 0}])
        guard = reembed.DiskGuard(threshold=10, reader=lambda: next(counts))
        self._run(apply=True, guard=guard, check_every=1)
        self.storage.reads.clear()
        self._run(apply=True)
        self.assertNotIn("thumbnails/photos/1.jpg", self.storage.reads)


class TestContentSniffing(unittest.TestCase):
    """The extension on the object key is not always the truth.

    63 of the 65 photos with no thumbnail are named `.arw`/`.ARW` but their
    bytes begin `FF D8 FF E0 JFIF` — they are JPEGs. `core.extractor.load_image`
    dispatches on the *extension*, so rawpy gets handed a JPEG and raises
    `LibRawFileUnsupportedError`. That is why those rows have no thumbnail and
    no embedding, and it is not a disk fault: the reads succeed.

    The bytes are already in hand by the time the filename is chosen, so the
    script sends the sidecar a name matching the real content and the PIL path
    handles it. Fixing `load_image` itself belongs to whoever owns
    `core/extractor.py`.
    """

    def test_a_jpeg_wearing_an_arw_extension_is_renamed(self):
        name = reembed.effective_filename("photos/2026/04/Snapseed.arw",
                                          b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01" + b"\x00" * 40)
        self.assertTrue(name.endswith(".jpg"), name)

    def test_a_real_raw_file_keeps_its_extension(self):
        """ARW is TIFF-based; sniffing must not rename a genuine RAW to .tif
        and send it down the PIL path, which would decode it wrongly or not
        at all."""
        name = reembed.effective_filename("photos/a.arw", b"II*\x00" + b"\x00" * 40)
        self.assertTrue(name.endswith(".arw"), name)

    def test_a_correctly_named_jpeg_is_left_alone(self):
        name = reembed.effective_filename("thumbnails/photos/1.jpg", b"\xff\xd8\xff\xe0" + b"\x00" * 40)
        self.assertEqual(name, "1.jpg")

    def test_a_png_wearing_a_jpg_extension_is_renamed(self):
        name = reembed.effective_filename("photos/x.jpg", b"\x89PNG\r\n\x1a\n" + b"\x00" * 40)
        self.assertTrue(name.endswith(".png"), name)

    def test_unrecognised_bytes_keep_the_declared_name(self):
        """Sniffing only ever corrects a mismatch it is sure about."""
        name = reembed.effective_filename("photos/x.heic", b"garbage" * 8)
        self.assertEqual(name, "x.heic")

    def test_heic_bytes_under_a_raw_extension_are_renamed(self):
        data = b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00" + b"\x00" * 40
        name = reembed.effective_filename("photos/x.arw", data)
        self.assertTrue(name.endswith(".heic"), name)

    def test_the_run_sends_the_corrected_name_to_the_embedder(self):
        conn, path = make_db([photo(1, thumb=False, ext="arw")])
        storage = FakeStorage({
            "photos/2026/01/p1.arw": b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01" + b"\x00" * 40,
        })
        embedder = FakeEmbedder()
        fd, journal = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        os.unlink(journal)
        try:
            reembed.run(conn, storage, embedder, apply=True, delay=0.0, tags=False,
                        journal_path=journal,
                        guard=reembed.DiskGuard(threshold=10,
                                                reader=lambda: {"kern": 0, "ioerr": 0}))
            self.assertEqual(embedder.calls, ["p1.jpg"])
        finally:
            conn.close()
            os.unlink(path)
            if os.path.exists(journal):
                os.unlink(journal)


# ----------------------------------------------------------------------
# videos  (Phase 7 — poster embedding, behind --include-videos)
# ----------------------------------------------------------------------
class TestVideoDecisions(unittest.TestCase):
    """A video is embedded from its poster or not at all.

    `core/jobs.py:_embed_source_key` settled the semantics for the job path in
    2b1b052: `thumb_path` for a video, `file_path` for everything else, and
    `None` — read nothing — for a video with no poster. This mirrors it. There
    is no second convention: the original is a multi-gigabyte MOV on a disk
    with 3,000+ unrecovered read errors, and CLIP consumes 224x224 of one frame
    either way.
    """

    def _decide(self, row, *, include_videos=False, source_mode=reembed.SOURCE_AUTO,
                force=False, has_embedding=False, done_ids=(), failed_ids=()):
        conn, path = make_db([row])
        try:
            r = conn.execute("SELECT * FROM photos WHERE id=?", (row["id"],)).fetchone()
            return reembed.classify_row(
                r, source_mode=source_mode, force=force, done_ids=set(done_ids),
                failed_ids=set(failed_ids), model=FakeEmbedder.name,
                has_embedding=has_embedding, include_videos=include_videos,
            )
        finally:
            conn.close()
            os.unlink(path)

    def test_a_video_is_skipped_without_the_flag(self):
        """Default off, like `scripts/retag.py --include-videos`: the cutover
        stays an explicit decision rather than a side effect of a normal pass."""
        d = self._decide(photo(1, media_type="video", ext="mov"))
        self.assertEqual(d.action, reembed.ACTION_SKIP)
        self.assertIn("--include-videos", d.reason)

    def test_a_video_with_a_poster_is_embedded_from_thumb_path(self):
        d = self._decide(photo(1, media_type="video", ext="mov"), include_videos=True)
        self.assertEqual(d.action, reembed.ACTION_EMBED)
        self.assertEqual(d.key, "thumbnails/photos/1.jpg")
        self.assertEqual(d.source, reembed.SOURCE_THUMB)
        self.assertEqual(d.media_type, "video")

    def test_a_video_with_no_poster_is_skipped_with_its_own_reason(self):
        """Distinct from every other skip, so a dry run says how many videos
        need `generate_video_poster_job` before a run would cover them."""
        d = self._decide(photo(2, thumb=False, media_type="video", ext="mov"),
                         include_videos=True)
        self.assertEqual(d.action, reembed.ACTION_SKIP)
        self.assertIsNone(d.key)
        self.assertEqual(d.reason, reembed.REASON_VIDEO_NO_POSTER)
        self.assertNotEqual(d.reason, reembed.REASON_VIDEO_EXCLUDED)

    def test_source_original_does_not_make_a_video_read_file_path(self):
        """The whole point. `--source original` governs photos; a video read
        from `file_path` is gigabytes instead of ~250 KB, for a worse vector."""
        d = self._decide(photo(3, media_type="video", ext="mov"),
                         include_videos=True, source_mode=reembed.SOURCE_ORIGINAL)
        self.assertEqual(d.action, reembed.ACTION_EMBED)
        self.assertEqual(d.source, reembed.SOURCE_THUMB)
        self.assertEqual(d.key, "thumbnails/photos/3.jpg")
        self.assertNotEqual(d.key, "photos/2026/01/p3.mov")

    def test_source_original_does_not_resurrect_a_posterless_video(self):
        d = self._decide(photo(4, thumb=False, media_type="video", ext="mov"),
                         include_videos=True, source_mode=reembed.SOURCE_ORIGINAL)
        self.assertEqual(d.action, reembed.ACTION_SKIP)
        self.assertEqual(d.reason, reembed.REASON_VIDEO_NO_POSTER)

    def test_source_thumb_leaves_a_video_on_its_poster(self):
        d = self._decide(photo(5, media_type="video", ext="mov"),
                         include_videos=True, source_mode=reembed.SOURCE_THUMB)
        self.assertEqual(d.source, reembed.SOURCE_THUMB)
        self.assertEqual(d.key, "thumbnails/photos/5.jpg")

    def test_a_video_already_carrying_this_model_is_skipped(self):
        d = self._decide(photo(6, media_type="video", ext="mov"),
                         include_videos=True, has_embedding=True)
        self.assertEqual(d.action, reembed.ACTION_SKIP)
        self.assertIn("embedding", d.reason)

    def test_the_journal_still_governs_a_video(self):
        """Resumability is not a photo-only property."""
        d = self._decide(photo(7, media_type="video", ext="mov"),
                         include_videos=True, force=True, done_ids=[7])
        self.assertEqual(d.action, reembed.ACTION_SKIP)
        self.assertIn("already", d.reason)

    def test_a_photo_is_unaffected_by_the_flag(self):
        for source_mode, expect_source, expect_key in (
            (reembed.SOURCE_AUTO, reembed.SOURCE_THUMB, "thumbnails/photos/8.jpg"),
            (reembed.SOURCE_ORIGINAL, reembed.SOURCE_ORIGINAL, "photos/2026/01/p8.heic"),
        ):
            for include_videos in (False, True):
                with self.subTest(source_mode=source_mode, include_videos=include_videos):
                    d = self._decide(photo(8), include_videos=include_videos,
                                     source_mode=source_mode)
                    self.assertEqual(d.action, reembed.ACTION_EMBED)
                    self.assertEqual(d.source, expect_source)
                    self.assertEqual(d.key, expect_key)
                    self.assertEqual(d.media_type, "photo")

    def test_a_null_media_type_is_a_photo(self):
        """766 of the 2,040 production rows have media_type NULL."""
        d = self._decide(photo(9, media_type=None), include_videos=True,
                         source_mode=reembed.SOURCE_ORIGINAL)
        self.assertEqual(d.source, reembed.SOURCE_ORIGINAL)
        self.assertEqual(d.key, "photos/2026/01/p9.heic")


class TestVideoPlan(unittest.TestCase):
    def setUp(self):
        rows = [
            photo(1),                                                    # photo, thumb
            photo(2, thumb=False),                                       # photo, no thumb
            photo(3, media_type="video", ext="mov"),                     # video + poster
            photo(4, thumb=False, media_type="video", ext="mov"),        # video, no poster
        ]
        self.conn, self.path = make_db(rows)

    def tearDown(self):
        self.conn.close()
        os.unlink(self.path)

    def _plan(self, **kw):
        kw.setdefault("source_mode", reembed.SOURCE_AUTO)
        kw.setdefault("force", False)
        kw.setdefault("done_ids", set())
        kw.setdefault("failed_ids", set())
        kw.setdefault("model", FakeEmbedder.name)
        return reembed.plan(self.conn, **kw)

    def test_videos_stay_out_by_default(self):
        ids = [d.photo_id for d in self._plan() if d.action == reembed.ACTION_EMBED]
        self.assertEqual(ids, [1, 2])

    def test_the_flag_adds_only_the_video_that_has_a_poster(self):
        decisions = self._plan(include_videos=True)
        ids = [d.photo_id for d in decisions if d.action == reembed.ACTION_EMBED]
        self.assertEqual(ids, [1, 2, 3])
        posterless = [d for d in decisions if d.reason == reembed.REASON_VIDEO_NO_POSTER]
        self.assertEqual([d.photo_id for d in posterless], [4])

    def test_limit_counts_a_video_read_like_any_other(self):
        decisions = self._plan(include_videos=True, limit=3)
        ids = [d.photo_id for d in decisions if d.action == reembed.ACTION_EMBED]
        self.assertEqual(ids, [1, 2, 3])


class TestVideoRun(unittest.TestCase):
    """End to end through `run()`, with the storage boundary stubbed."""

    def setUp(self):
        rows = [
            photo(1),
            photo(3, media_type="video", ext="mov"),
            photo(4, thumb=False, media_type="video", ext="mov"),
        ]
        self.conn, self.path = make_db(rows)
        self.storage = FakeStorage({
            "thumbnails/photos/1.jpg": b"thumb-one",
            "thumbnails/photos/3.jpg": b"\xff\xd8\xff\xe0" + b"poster-three",
            # The originals are present on purpose: if the script ever reads a
            # video's `file_path` the test must fail on the assertion below,
            # not incidentally on a missing object.
            "photos/2026/01/p1.heic": b"original-one",
            "photos/2026/01/p3.mov": b"a-multi-gigabyte-mov",
            "photos/2026/01/p4.mov": b"another-multi-gigabyte-mov",
        })
        self.embedder = FakeEmbedder()
        fd, self.journal_path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        os.unlink(self.journal_path)

    def tearDown(self):
        self.conn.close()
        os.unlink(self.path)
        if os.path.exists(self.journal_path):
            os.unlink(self.journal_path)

    def _run(self, **kw):
        kw.setdefault("apply", True)
        kw.setdefault("source_mode", reembed.SOURCE_AUTO)
        kw.setdefault("force", False)
        kw.setdefault("delay", 0.0)
        kw.setdefault("tags", False)
        kw.setdefault("journal_path", self.journal_path)
        kw.setdefault("guard", reembed.DiskGuard(threshold=10, reader=lambda: {"kern": 0, "ioerr": 0}))
        return reembed.run(self.conn, self.storage, self.embedder, **kw)

    def test_a_run_without_the_flag_touches_no_video_object(self):
        result = self._run()
        self.assertEqual(self.storage.reads, ["thumbnails/photos/1.jpg"])
        self.assertEqual(result.embedded, 1)
        self.assertEqual(
            self.conn.execute(
                "SELECT COUNT(*) FROM embeddings e JOIN photos p ON p.id = e.photo_id "
                "WHERE p.media_type = 'video'").fetchone()[0],
            0,
        )

    def test_the_flag_embeds_a_video_from_its_poster_and_never_the_original(self):
        result = self._run(include_videos=True)
        self.assertEqual(self.storage.reads,
                         ["thumbnails/photos/1.jpg", "thumbnails/photos/3.jpg"])
        self.assertNotIn("photos/2026/01/p3.mov", self.storage.reads)
        self.assertNotIn("photos/2026/01/p4.mov", self.storage.reads)
        self.assertEqual(result.embedded, 2)
        self.assertEqual(result.failed, [])
        row = self.conn.execute(
            "SELECT model, dim FROM embeddings WHERE photo_id = 3").fetchone()
        self.assertEqual(row["model"], FakeEmbedder.name)
        self.assertEqual(row["dim"], FakeEmbedder.dim)

    def test_source_original_still_reads_only_the_poster_for_a_video(self):
        self._run(include_videos=True, source_mode=reembed.SOURCE_ORIGINAL)
        self.assertEqual(self.storage.reads,
                         ["photos/2026/01/p1.heic", "thumbnails/photos/3.jpg"])

    def test_a_posterless_video_is_recorded_as_a_skip_not_a_failure(self):
        result = self._run(include_videos=True)
        self.assertEqual(result.failed, [])
        reasons = [d.reason for d in result.decisions if d.photo_id == 4]
        self.assertEqual(reasons, [reembed.REASON_VIDEO_NO_POSTER])

    def test_a_video_is_journalled_and_not_re_read_on_the_next_pass(self):
        self._run(include_videos=True)
        self.storage.reads.clear()
        second = self._run(include_videos=True, force=True)
        self.assertEqual(self.storage.reads, [])
        self.assertEqual(second.embedded, 0)

    def test_a_dry_run_with_the_flag_reads_nothing(self):
        result = self._run(apply=False, include_videos=True)
        self.assertEqual(self.storage.reads, [])
        self.assertEqual(self.embedder.calls, [])
        self.assertEqual(result.planned, 2)


class TestVideoSummary(unittest.TestCase):
    """The dry-run summary has to tell the operator what a run would cover."""

    def _summary(self, decisions):
        import contextlib
        import io as _io

        buf = _io.StringIO()
        with contextlib.redirect_stdout(buf):
            reembed._summarise_plan(decisions)
        return buf.getvalue()

    def test_it_separates_would_embed_from_no_poster_from_excluded(self):
        rows = [
            photo(1),
            photo(3, media_type="video", ext="mov"),
            photo(4, thumb=False, media_type="video", ext="mov"),
        ]
        conn, path = make_db(rows)
        try:
            decisions = reembed.plan(
                conn, source_mode=reembed.SOURCE_AUTO, force=False, done_ids=set(),
                failed_ids=set(), model=FakeEmbedder.name, include_videos=True,
            )
            out = self._summary(decisions)
        finally:
            conn.close()
            os.unlink(path)
        self.assertIn("1 video poster", out)
        self.assertIn(reembed.REASON_VIDEO_NO_POSTER, out)

    def test_the_default_run_says_why_the_videos_are_missing(self):
        rows = [photo(1), photo(3, media_type="video", ext="mov")]
        conn, path = make_db(rows)
        try:
            decisions = reembed.plan(
                conn, source_mode=reembed.SOURCE_AUTO, force=False, done_ids=set(),
                failed_ids=set(), model=FakeEmbedder.name,
            )
            out = self._summary(decisions)
        finally:
            conn.close()
            os.unlink(path)
        self.assertIn(reembed.REASON_VIDEO_EXCLUDED, out)


class TestArgParsing(unittest.TestCase):
    def test_dry_run_is_the_default(self):
        args = reembed.build_parser().parse_args([])
        self.assertFalse(args.apply)
        self.assertFalse(args.force)
        self.assertEqual(args.source, reembed.SOURCE_AUTO)
        self.assertEqual(args.concurrency, 1)

    def test_concurrency_must_be_positive(self):
        with self.assertRaises(SystemExit):
            reembed.build_parser().parse_args(["--concurrency", "0"])

    def test_model_defaults_to_the_backend_not_a_hardcoded_string(self):
        """Phase 6's cutover keys on `embeddings.model`; a wrong default there
        would label SigLIP rows as CLIP and make the table unqueryable."""
        self.assertIsNone(reembed.build_parser().parse_args([]).model)

    def test_model_is_accepted_and_carried(self):
        args = reembed.build_parser().parse_args(["--model", "google/siglip2-base"])
        self.assertEqual(args.model, "google/siglip2-base")

    def test_source_is_restricted_to_the_three_modes(self):
        with self.assertRaises(SystemExit):
            reembed.build_parser().parse_args(["--source", "raw"])


    def test_include_videos_is_off_by_default(self):
        self.assertFalse(reembed.build_parser().parse_args([]).include_videos)

    def test_include_videos_is_accepted(self):
        args = reembed.build_parser().parse_args(["--include-videos"])
        self.assertTrue(args.include_videos)

    def test_the_source_help_documents_the_video_interaction(self):
        """`--source original` on a video would read gigabytes; the help text
        is where an operator finds out that it does not."""
        help_text = reembed.build_parser().format_help()
        self.assertIn("--include-videos", help_text)
        source_help = [a for a in reembed.build_parser()._actions
                       if a.dest == "source"][0].help
        self.assertIn("video", source_help)
        self.assertIn("poster", source_help)


# ----------------------------------------------------------------------
# BACKEND SELECTION
# ----------------------------------------------------------------------
class TestBackendMatchesTheModelItWritesUnder(unittest.TestCase):
    """The backend must actually be the model whose name gets written down.

    `embeddings.model` is half the table's unique key and the value every
    ranking read filters on, so a row is only meaningful if the vector in it
    was produced by the model the row claims. The sidecar is a *separate
    process with its own resident model*: during the SigLIP migration it is
    still holding CLIP while this script is asked to write 768-d SigLIP rows.
    Handing back that sidecar anyway writes 512-d CLIP vectors under
    `google/siglip2-base-patch16-224` -- rows that lie about their own
    provenance, pass a coverage check by count, and take search down at the
    cutover.
    """

    def setUp(self):
        self._real_sidecar = reembed.SidecarEmbedder
        self._real_inprocess = reembed.InProcessEmbedder
        self.built = []

        outer = self

        class FakeSidecar:
            def __init__(self, *a, **kw):
                self.name = outer.sidecar_model
                self.dim = 512
                self.closed = False

            def close(self):
                self.closed = True

        class FakeInProcess:
            def __init__(self, model_name=reembed.DEFAULT_MODEL,
                         threads=reembed.DEFAULT_THREADS):
                self.name = model_name
                self.threads = threads
                self.dim = 768 if "siglip" in model_name else 512
                outer.built.append(model_name)

        self.sidecar_model = "openai/clip-vit-base-patch32"
        reembed.SidecarEmbedder = FakeSidecar
        reembed.InProcessEmbedder = FakeInProcess

    def tearDown(self):
        reembed.SidecarEmbedder = self._real_sidecar
        reembed.InProcessEmbedder = self._real_inprocess

    def test_auto_uses_the_sidecar_when_it_holds_the_requested_model(self):
        em = reembed.build_embedder(reembed.BACKEND_AUTO,
                                    "openai/clip-vit-base-patch32")
        self.assertEqual(em.name, "openai/clip-vit-base-patch32")
        self.assertEqual(self.built, [])

    def test_auto_still_adopts_the_sidecars_model_when_none_is_requested(self):
        """`--model` defaults to whatever the backend reports; that contract
        must survive the mismatch check."""
        self.sidecar_model = "google/siglip2-base-patch16-224"
        em = reembed.build_embedder(reembed.BACKEND_AUTO, None)
        self.assertEqual(em.name, "google/siglip2-base-patch16-224")
        self.assertEqual(self.built, [])

    def test_auto_refuses_a_sidecar_loaded_with_a_different_model(self):
        em = reembed.build_embedder(reembed.BACKEND_AUTO,
                                    "google/siglip2-base-patch16-224")
        self.assertEqual(em.name, "google/siglip2-base-patch16-224")
        self.assertEqual(em.dim, 768)
        self.assertEqual(self.built, ["google/siglip2-base-patch16-224"])

    def test_an_explicit_sidecar_request_raises_rather_than_lying(self):
        """`--backend sidecar` asked for that process specifically. Silently
        loading 1.5 GB of weights in-process instead would be a surprise; so
        would writing CLIP vectors under a SigLIP name."""
        with self.assertRaises(RuntimeError) as ctx:
            reembed.build_embedder(reembed.BACKEND_SIDECAR,
                                   "google/siglip2-base-patch16-224")
        self.assertIn("google/siglip2-base-patch16-224", str(ctx.exception))
        self.assertIn("openai/clip-vit-base-patch32", str(ctx.exception))


class TestInProcessBackendHonoursTheModelName(unittest.TestCase):
    """`--backend inprocess --model <siglip>` must build a SigLIP embedder.

    It hardcoded `ClipEmbedder(model_name)`, so a SigLIP identifier meant
    `CLIPModel.from_pretrained` on a SigLIP checkpoint: transformers warns
    "using a model of type siglip to instantiate a model of type clip" and
    then dies on a `[64, 768]` vs `[77, 512]` state-dict mismatch. The
    registry in `core.embedder.build_embedder` is the one place that knows
    which class an identifier means, and it raises on an unknown one rather
    than falling back to CLIP.
    """

    def test_it_dispatches_through_the_core_registry(self):
        import core.embedder as core_embedder

        seen = []

        class FakeModelEmbedder:
            def text_embedding(self, text):
                import numpy as np

                return np.zeros(768, dtype="float32")

        real = core_embedder.build_embedder

        def fake_build(model_name=None):
            seen.append(model_name)
            return FakeModelEmbedder()

        core_embedder.build_embedder = fake_build
        try:
            em = reembed.InProcessEmbedder("google/siglip2-base-patch16-224",
                                           threads=1)
        finally:
            core_embedder.build_embedder = real

        self.assertEqual(seen, ["google/siglip2-base-patch16-224"])
        self.assertEqual(em.name, "google/siglip2-base-patch16-224")
        self.assertEqual(em.dim, 768)

    def test_an_unknown_identifier_raises_instead_of_becoming_clip(self):
        with self.assertRaises(ValueError):
            reembed.InProcessEmbedder("nobody/nothing-v9", threads=1)


if __name__ == "__main__":
    unittest.main()
