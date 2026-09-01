"""Videos become searchable through their poster, and only their poster.

258 of the 262 videos already have a 512x512 JPEG poster in `thumb_path` —
~250 KB each, ~64 MB for the lot. The originals are gigabytes. So the whole
cost of making videos searchable is choosing which object key to download; the
poster is already generated, already on disk, and already read by
`scripts/reembed.py`.

The one thing that must not happen is embedding the *original*. A video's
`file_path` is a multi-gigabyte MOV on a disk with 3,000+ unrecovered read
errors, CLIP would look at 224x224 of the first decodable frame, and the read
would cost four orders of magnitude more than the poster. `_is_video` still
blocks ML on the original — face detection still skips videos entirely — and
these tests pin the distinction, because "videos are embedded now" is exactly
the kind of change that quietly generalises into "so point it at `file_path`".
"""
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from core import db, jobs


class ExplodingClipEmbedder:
    def __init__(self, *args, **kwargs):
        raise AssertionError("the embedding job must not construct a ClipEmbedder")


class CountingStorage:
    def __init__(self):
        self.downloads = []

    def download_file(self, key):
        self.downloads.append(key)
        return b"\xff\xd8\xff\xe0 fake jpeg bytes"


class CountingSidecar:
    def __init__(self):
        self.image_calls = []

    def image_embedding(self, filename, data):
        self.image_calls.append(filename)
        vec = np.zeros(4, dtype="float32")
        vec[0] = 1.0
        return vec

    def rank_labels_for_vector(self, image_vec, labels, top_k=6):
        return [(label, 0.2) for label in list(labels)[:top_k]]


class VideoEmbeddingTestCase(unittest.TestCase):
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        db.init_db(self.db_path)
        self.conn = db.connect(self.db_path)

        self.storage = CountingStorage()
        self.sidecar = CountingSidecar()
        module = types.ModuleType("core.embedder")
        module.ClipEmbedder = ExplodingClipEmbedder
        for p in (
            patch.object(jobs, "_get_storage_client", lambda: self.storage),
            patch.object(jobs, "_get_embed_client", lambda: self.sidecar),
            patch.dict(sys.modules, {"core.embedder": module}),
        ):
            p.start()
            self.addCleanup(p.stop)

    def tearDown(self):
        self.conn.close()
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def add_photo(self, photo_id, file_path, media_type, thumb_path):
        self.conn.execute(
            "INSERT INTO photos (id, file_path, media_type, thumb_path) VALUES (?,?,?,?)",
            (photo_id, file_path, media_type, thumb_path),
        )
        self.conn.commit()

    def rows(self, sql, *params):
        cur = self.conn.cursor()
        cur.execute(sql, params)
        return cur.fetchall()


class TestVideoPosterIsEmbedded(VideoEmbeddingTestCase):
    def setUp(self):
        super().setUp()
        self.add_photo(9, "videos/9.mov", "video", "thumbnails/photos/9.jpg")

    def test_it_reads_the_poster_and_never_the_original(self):
        jobs.process_photo_embedding_job(9, "videos/9.mov", self.db_path)

        self.assertEqual(["thumbnails/photos/9.jpg"], self.storage.downloads)
        self.assertNotIn("videos/9.mov", self.storage.downloads)

    def test_the_poster_is_what_reaches_the_sidecar(self):
        jobs.process_photo_embedding_job(9, "videos/9.mov", self.db_path)
        self.assertEqual(["9.jpg"], self.sidecar.image_calls)

    def test_the_video_gets_an_embedding(self):
        jobs.process_photo_embedding_job(9, "videos/9.mov", self.db_path)

        rows = self.rows("SELECT dim FROM embeddings WHERE photo_id=9")
        self.assertEqual(1, len(rows))
        self.assertEqual(4, rows[0]["dim"])

    def test_the_video_gets_tags(self):
        jobs.process_photo_embedding_job(9, "videos/9.mov", self.db_path)
        self.assertEqual(6, len(self.rows("SELECT tag FROM tags WHERE photo_id=9")))

    def test_the_batch_path_uses_the_poster_too(self):
        self.assertTrue(jobs._process_single_embedding(9, "videos/9.mov", self.db_path))
        self.assertEqual(["thumbnails/photos/9.jpg"], self.storage.downloads)


class TestVideoWithoutAPoster(VideoEmbeddingTestCase):
    """4 of the 262 videos have no poster. They read nothing at all.

    Falling back to `file_path` here would be worse than doing nothing: it is
    the multi-gigabyte original, and it is on the failing disk.
    """

    def test_a_null_poster_reads_nothing(self):
        self.add_photo(9, "videos/9.mov", "video", None)

        self.assertTrue(jobs.process_photo_embedding_job(9, "videos/9.mov", self.db_path))
        self.assertEqual([], self.storage.downloads)
        self.assertEqual([], self.rows("SELECT id FROM embeddings WHERE photo_id=9"))

    def test_an_empty_poster_path_reads_nothing(self):
        self.add_photo(9, "videos/9.mov", "video", "")

        self.assertTrue(jobs.process_photo_embedding_job(9, "videos/9.mov", self.db_path))
        self.assertEqual([], self.storage.downloads)

    def test_the_batch_path_reports_it_as_not_indexed(self):
        self.add_photo(9, "videos/9.mov", "video", None)

        self.assertFalse(jobs._process_single_embedding(9, "videos/9.mov", self.db_path))
        self.assertEqual([], self.storage.downloads)


class TestPhotosAreUnchanged(VideoEmbeddingTestCase):
    """A photo still embeds its original, thumbnail or not.

    The thumbnail path is `scripts/reembed.py`'s deliberate trade for a bulk
    pass over a failing disk. A single upload has already paid for the original
    and should embed what the user actually uploaded.
    """

    def test_a_photo_with_a_thumbnail_still_reads_the_original(self):
        self.add_photo(1, "photos/1.jpg", "photo", "thumbnails/photos/1.jpg")

        jobs.process_photo_embedding_job(1, "photos/1.jpg", self.db_path)
        self.assertEqual(["photos/1.jpg"], self.storage.downloads)

    def test_a_null_media_type_is_treated_as_a_photo(self):
        """766 of the 2,040 rows have `media_type` NULL."""
        self.add_photo(2, "photos/2.jpg", None, "thumbnails/photos/2.jpg")

        jobs.process_photo_embedding_job(2, "photos/2.jpg", self.db_path)
        self.assertEqual(["photos/2.jpg"], self.storage.downloads)


class TestFaceDetectionStillSkipsVideos(VideoEmbeddingTestCase):
    """`_is_video` keeps blocking ML on the original. Only embedding gained a
    poster path; face detection has none and must not grow one by accident."""

    def test_the_face_job_reads_nothing_for_a_video(self):
        self.add_photo(9, "videos/9.mov", "video", "thumbnails/photos/9.jpg")

        self.assertEqual(0, jobs.process_photo_faces_job(9, "videos/9.mov", self.db_path))
        self.assertEqual([], self.storage.downloads)

    def test_the_batch_face_path_reads_nothing_for_a_video(self):
        self.add_photo(9, "videos/9.mov", "video", "thumbnails/photos/9.jpg")

        self.assertFalse(jobs._process_single_face(9, "videos/9.mov", self.db_path, 0.5, 160))
        self.assertEqual([], self.storage.downloads)


if __name__ == "__main__":
    unittest.main()
