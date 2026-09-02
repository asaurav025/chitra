"""The embedding jobs go through the sidecar, and hold no model themselves.

What this is protecting
-----------------------
RQ forks a work-horse per job, so `core/jobs.py`'s module-level `_EMBEDDER` is
built in the child and dies with it. Every embedding job reloaded CLIP from
scratch: ~90% of a measured 58.5 s job, and 1.67 GB of peak RSS in a cgroup
that has been OOM-killed six times. `embed_service.py` already holds one
resident CLIP and has exposed `POST /embed/image` since it was written, with no
caller.

Four properties are load-bearing, and each is one revert away from being lost
silently rather than loudly:

1. **No `ClipEmbedder` is ever constructed.** Pinned with a stub that raises on
   instantiation, and with a subprocess run that asserts torch never even
   enters `sys.modules`.
2. **Exactly one image embed per photo.** `core.tagger.auto_tags` calls
   `rank_labels`, which calls `image_embedding` *again* — so the old job ran two
   CLIP forward passes over the same bytes, threw the second vector away, and
   recomputed the 17-label text batch from scratch every time. With the vector
   in hand the tags cost a dot product.
3. **A sidecar failure re-raises.** The job used to swallow every exception and
   return `False`, which makes RQ mark it *successful*. And an in-process
   fallback would restore the 58 s path and the 1.67 GB invisibly — the
   pipeline would just get slow again with nothing saying why.
4. **MinIO is downloaded exactly once.** The disk under it has 3,000+
   unrecovered read errors; a second read per photo is a second chance to hit
   one, for nothing.
"""
import os
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from core import db, jobs
from core.embed_client_sync import EmbeddingUnavailable

REPO_ROOT = Path(__file__).resolve().parent.parent


class ExplodingClipEmbedder:
    """A `ClipEmbedder` that fails the test if anything constructs it.

    Constructing the real one costs ~1.1 GB and ~50 s in a forked work-horse.
    The assertion has to be on construction rather than on elapsed time,
    because a regression here looks like nothing at all until someone times a
    job.
    """

    def __init__(self, *args, **kwargs):
        raise AssertionError(
            "the embedding job constructed a ClipEmbedder — the whole point of "
            "routing through the sidecar is that the worker holds no model"
        )


class CountingStorage:
    """MinIO stub that records every object key it was asked for."""

    def __init__(self, objects=None):
        self.downloads = []
        self.objects = objects or {}

    def download_file(self, key):
        self.downloads.append(key)
        return self.objects.get(key, b"\xff\xd8\xff\xe0 fake jpeg bytes")


class CountingSidecar:
    """Sidecar stub that records image embeds and label lookups."""

    def __init__(self, dim=4, fail_with=None, model="google/siglip2-base-patch16-224"):
        self.image_calls = []
        self.label_calls = []
        self.dim = dim
        self.fail_with = fail_with
        self.model = model

    def served_model(self):
        if self.model is None:
            raise EmbeddingUnavailable("sidecar did not name its model")
        return self.model

    def image_embedding(self, filename, data):
        self.image_calls.append((filename, len(data)))
        if self.fail_with is not None:
            raise self.fail_with
        vec = np.zeros(self.dim, dtype="float32")
        vec[0] = 1.0
        return vec

    def rank_labels_for_vector(self, image_vec, labels, top_k=6):
        self.label_calls.append(tuple(labels))
        return [(label, 0.2) for label in list(labels)[:top_k]]


class EmbeddingJobTestCase(unittest.TestCase):
    """A real SQLite DB with one photo row, and both dependencies stubbed."""

    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        db.init_db(self.db_path)
        self.conn = db.connect(self.db_path)
        self.conn.execute(
            "INSERT INTO photos (id, file_path, media_type, thumb_path) VALUES (?,?,?,?)",
            (1, "photos/1.jpg", "photo", "thumbnails/photos/1.jpg"),
        )
        self.conn.commit()

        self.storage = CountingStorage()
        self.sidecar = CountingSidecar()
        self._patches = [
            patch.object(jobs, "_get_storage_client", lambda: self.storage),
            patch.object(jobs, "_get_embed_client", lambda: self.sidecar),
            patch.dict(sys.modules, {"core.embedder": self._exploding_module()}),
        ]
        for p in self._patches:
            p.start()
            self.addCleanup(p.stop)

    def tearDown(self):
        self.conn.close()
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    @staticmethod
    def _exploding_module():
        module = types.ModuleType("core.embedder")
        module.ClipEmbedder = ExplodingClipEmbedder
        return module

    def rows(self, sql, *params):
        cur = self.conn.cursor()
        cur.execute(sql, params)
        return cur.fetchall()


class TestProcessPhotoEmbeddingJob(EmbeddingJobTestCase):
    def test_it_never_constructs_a_clip_embedder(self):
        self.assertTrue(jobs.process_photo_embedding_job(1, "photos/1.jpg", self.db_path))

    def test_exactly_one_image_embed_per_photo(self):
        jobs.process_photo_embedding_job(1, "photos/1.jpg", self.db_path)
        self.assertEqual(1, len(self.sidecar.image_calls), self.sidecar.image_calls)

    def test_minio_is_downloaded_exactly_once(self):
        jobs.process_photo_embedding_job(1, "photos/1.jpg", self.db_path)
        self.assertEqual(["photos/1.jpg"], self.storage.downloads)

    def test_it_stores_the_vector_the_sidecar_returned(self):
        jobs.process_photo_embedding_job(1, "photos/1.jpg", self.db_path)

        rows = self.rows("SELECT dim, vector FROM embeddings WHERE photo_id=1")
        self.assertEqual(1, len(rows))
        self.assertEqual(4, rows[0]["dim"])
        stored = np.frombuffer(rows[0]["vector"], dtype="float32")
        np.testing.assert_allclose([1.0, 0.0, 0.0, 0.0], stored)

    def test_it_still_writes_tags(self):
        jobs.process_photo_embedding_job(1, "photos/1.jpg", self.db_path)

        tags = [r["tag"] for r in self.rows("SELECT tag FROM tags WHERE photo_id=1")]
        self.assertEqual(6, len(tags))
        self.assertEqual(1, len(self.sidecar.label_calls))

    def test_tags_are_ranked_from_the_vector_not_a_second_image_pass(self):
        jobs.process_photo_embedding_job(1, "photos/1.jpg", self.db_path)
        # One image embed total, and the label ranking happened anyway.
        self.assertEqual(1, len(self.sidecar.image_calls))
        self.assertEqual(1, len(self.sidecar.label_calls))

    def test_a_sidecar_failure_re_raises(self):
        self.sidecar.fail_with = EmbeddingUnavailable("sidecar is down")

        with self.assertRaises(EmbeddingUnavailable):
            jobs.process_photo_embedding_job(1, "photos/1.jpg", self.db_path)

    def test_a_sidecar_failure_leaves_no_embedding_behind(self):
        self.sidecar.fail_with = EmbeddingUnavailable("sidecar is down")
        with self.assertRaises(EmbeddingUnavailable):
            jobs.process_photo_embedding_job(1, "photos/1.jpg", self.db_path)

        self.assertEqual([], self.rows("SELECT id FROM embeddings WHERE photo_id=1"))

    def test_a_storage_failure_re_raises_too(self):
        """Jobs that return False are marked *successful* by RQ. AGENTS.md:
        new jobs re-raise after logging."""

        def boom(key):
            raise OSError("MinIO said no")

        self.storage.download_file = boom
        with self.assertRaises(OSError):
            jobs.process_photo_embedding_job(1, "photos/1.jpg", self.db_path)


class TestProcessSingleEmbedding(EmbeddingJobTestCase):
    """The batch path — `index_embeddings_batch_job`'s per-photo worker."""

    def test_it_never_constructs_a_clip_embedder(self):
        self.assertTrue(jobs._process_single_embedding(1, "photos/1.jpg", self.db_path))

    def test_exactly_one_image_embed_and_one_download(self):
        jobs._process_single_embedding(1, "photos/1.jpg", self.db_path)
        self.assertEqual(1, len(self.sidecar.image_calls))
        self.assertEqual(["photos/1.jpg"], self.storage.downloads)

    def test_a_sidecar_failure_re_raises(self):
        self.sidecar.fail_with = EmbeddingUnavailable("sidecar is down")
        with self.assertRaises(EmbeddingUnavailable):
            jobs._process_single_embedding(1, "photos/1.jpg", self.db_path)

    def test_the_batch_job_survives_one_photo_failing(self):
        """`index_embeddings_batch_job` counts successes; a failure inside one
        photo must not take the batch down with it."""
        self.conn.execute(
            "INSERT INTO photos (id, file_path, media_type) VALUES (2, 'photos/2.jpg', 'photo')"
        )
        self.conn.commit()

        calls = {"n": 0}
        real = self.sidecar.image_embedding

        def flaky(filename, data):
            calls["n"] += 1
            if filename == "2.jpg":
                raise EmbeddingUnavailable("sidecar blinked")
            return real(filename, data)

        self.sidecar.image_embedding = flaky
        indexed = jobs.index_embeddings_batch_job(
            [(1, "photos/1.jpg"), (2, "photos/2.jpg")], self.db_path, True
        )
        self.assertEqual(1, indexed)


class TestTheRowsNameTheModelThatMadeThem(EmbeddingJobTestCase):
    """A row is only worth anything if it names the model that produced it.

    `embeddings.model` is half that table's unique key and the value every
    ranking read filters on. `_embed_and_tag` stored the sidecar's vector
    without saying which model computed it, so `db.put_embedding` fell back to
    `DEFAULT_EMBED_MODEL` — the CLIP identifier — and after the SigLIP cutover
    the very first upload wrote a **768-d vector under
    `openai/clip-vit-base-patch32`**. Measured on production: photo 2775, one
    row, `dim=768, model='openai/clip-vit-base-patch32'`, and no row at all
    under the SigLIP name that `CHITRA_ACTIVE_EMBED_MODEL` filters to. Two
    consequences, neither of which raises anything:

    * the photo is invisible to search, because search reads the active model
      and there is no row there;
    * a rollback to CLIP takes `search_photos` down with
      ``ValueError: all input arrays must have the same shape`` the moment
      `np.stack` meets a 768 beside 2,721 512s.

    `tags.source` has the same disease from the same cause — `db.add_tag`'s
    `DEFAULT_TAG_SOURCE` is the literal `clip-vitb32/vocab-v1`, so SigLIP's
    tags have been filing themselves under CLIP's name since the 11:48
    restart.
    """

    def test_the_embedding_row_names_the_model_the_sidecar_served(self):
        jobs.process_photo_embedding_job(1, "photos/1.jpg", self.db_path)

        rows = self.rows("SELECT model FROM embeddings WHERE photo_id=1")
        self.assertEqual(["google/siglip2-base-patch16-224"],
                         [r["model"] for r in rows])

    def test_it_does_not_fall_back_to_the_clip_default(self):
        from core.db import DEFAULT_EMBED_MODEL

        self.sidecar.model = "google/siglip2-base-patch16-224"
        jobs.process_photo_embedding_job(1, "photos/1.jpg", self.db_path)

        rows = self.rows("SELECT model FROM embeddings WHERE photo_id=1")
        self.assertNotIn(DEFAULT_EMBED_MODEL, [r["model"] for r in rows])

    def test_tag_provenance_names_the_model_the_sidecar_served(self):
        jobs.process_photo_embedding_job(1, "photos/1.jpg", self.db_path)

        sources = {r["source"] for r in self.rows("SELECT source FROM tags WHERE photo_id=1")}
        self.assertEqual(1, len(sources), sources)
        source = sources.pop()
        self.assertTrue(source.startswith("siglip2-base-patch16-224/"), source)

    def test_a_sidecar_that_will_not_name_its_model_fails_the_job(self):
        """Guessing the name is the bug; there is nothing safe to guess."""
        self.sidecar.model = None

        with self.assertRaises(EmbeddingUnavailable):
            jobs.process_photo_embedding_job(1, "photos/1.jpg", self.db_path)

        self.assertEqual([], self.rows("SELECT 1 FROM embeddings WHERE photo_id=1"))


class TestTheSharedClient(unittest.TestCase):
    def setUp(self):
        self._saved = jobs._EMBED_CLIENT
        jobs._EMBED_CLIENT = None
        self.addCleanup(lambda: setattr(jobs, "_EMBED_CLIENT", self._saved))

    def test_it_is_a_sync_client_and_is_reused(self):
        from core.embed_client_sync import SyncEmbeddingClient

        first = jobs._get_embed_client()
        self.assertIsInstance(first, SyncEmbeddingClient)
        self.assertIs(first, jobs._get_embed_client())


class TestTheWorkerLoadsNoModel(unittest.TestCase):
    """The definitive check, in a fresh interpreter.

    Everything above runs in a process where `tests/test_search.py` may already
    have imported torch, so `sys.modules` proves nothing there. Here the job
    runs end-to-end against stubs in a clean subprocess, and torch must never
    appear.
    """

    def test_running_the_job_never_imports_torch(self):
        script = textwrap.dedent(
            """
            import sys, tempfile, os
            import numpy as np
            from core import db, jobs

            path = tempfile.mktemp(suffix=".db")
            db.init_db(path)
            conn = db.connect(path)
            conn.execute("INSERT INTO photos (id, file_path, media_type) "
                         "VALUES (1, 'photos/1.jpg', 'photo')")
            conn.commit()
            conn.close()

            class Storage:
                def download_file(self, key):
                    return b"bytes"

            class Sidecar:
                def served_model(self):
                    return "google/siglip2-base-patch16-224"
                def image_embedding(self, filename, data):
                    return np.array([1, 0, 0, 0], dtype="float32")
                def rank_labels_for_vector(self, vec, labels, top_k=6):
                    return [(l, 0.2) for l in list(labels)[:top_k]]

            jobs._STORAGE_CLIENT = Storage()
            jobs._EMBED_CLIENT = Sidecar()
            jobs.process_photo_embedding_job(1, "photos/1.jpg", path)
            os.unlink(path)

            print(sorted(k for k in sys.modules
                         if k in ("torch", "transformers", "core.embedder",
                                  "insightface") or k.startswith("torch.")))
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=180,
            env={**os.environ, "CHITRA_DB_PATH": "/tmp/chitra_test.db"},
        )
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual("[]", result.stdout.strip().splitlines()[-1], result.stdout)


if __name__ == "__main__":
    unittest.main()
