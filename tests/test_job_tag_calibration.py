"""New uploads get the calibrated 345-label vocabulary — when one exists.

The problem
-----------
`scripts/retag.py` gave the library `vocab-v2`: 345 labels scored against
per-label percentiles learned from the whole corpus. The upload path never
moved. `core/jobs._embed_and_tag` ranked `core.tagger.DEFAULT_LABELS` — the
legacy 17 — so every photo uploaded after the re-tag re-opened the same drift
the re-tag had just closed. Production showed it directly: 12,002 rows under
`clip-vitb32/vocab-v2` beside 24 already back under `vocab-v1`.

Why the job cannot just switch its label list
---------------------------------------------
`vocab-v2`'s quality is not in the label count, it is in the calibration, and
the calibration is *corpus-relative*: a label is kept when this photo sits in
the upper tail of that label's distribution **over the whole library**. A job
holds one photo. It has no corpus and cannot make one.

Plain top-k over the 345 is the tempting shortcut and it is measurably wrong.
Measured on the 2,721 stored SigLIP vectors: top-6 puts `diwali celebration`
on 22.6% of the library and `candid portrait` on 20.2%, and every photo gets
exactly 6 tags whether or not it resembles anything. The calibrated pass over
the same matrix gives 4.72 tags per photo (0 to 8), uses all 345 labels, and
tops out at 3.8% coverage for any one label.

Nor does SigLIP's sigmoid rescue it, which is what the plan assumed. With the
checkpoint's own trained scalars — `logit_scale` 4.7244534, `logit_bias`
-16.771725, so `p = sigmoid(112.7 * cos - 16.77)` — the principled p >= 0.5 cut
tags 28 of 2,721 photos. At p >= 0.05, 1,838 photos (68%) get nothing. The
absolute threshold is real arithmetic and a useless operating point here, so
per-label calibration stays.

So the job is a **consumer**, never a producer
----------------------------------------------
`scripts/retag.py` already computes both artifacts a job would need — the label
matrix and the calibration — and already caches the first on the NVMe. It now
writes the second beside it. The job loads both, keyed on
`(model, vocab fingerprint)`, and:

* both present and matching -> calibrated 345-label tags, stamped `vocab-v2`;
* either absent -> the legacy 17-label top-k it does today, stamped `vocab-v1`.

Staleness is handled by the key, not by a timestamp. A model change or a
vocabulary change moves the *filename*, so the artifacts are simply absent and
the job falls back rather than applying thresholds measured against a different
space. An artifact that exists under a name it does not describe is corruption,
not staleness, and raises — the failure mode this whole design exists to avoid
is a run that completes with every tag silently wrong.

What the job must never do is compute either artifact itself. Embedding 345
prompts through the sidecar costs ~30 s (measured) in a forked work-horse that
lives for one photo.
"""
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from core import db, jobs, tagger, vocabulary

MODEL = "google/siglip2-base-patch16-224"
DIM = 8


def orthonormal_matrix(n_labels, dim=DIM, seed=7):
    """Deterministic unit-norm label vectors — no model, no sidecar."""
    rng = np.random.default_rng(seed)
    mat = rng.standard_normal((n_labels, dim)).astype("float32")
    return mat / np.linalg.norm(mat, axis=1, keepdims=True)


class Sidecar:
    """Sidecar stub that records whether the legacy ranking path was used."""

    def __init__(self, vec):
        self.vec = np.asarray(vec, dtype="float32")
        self.label_calls = []

    def served_model(self):
        return MODEL

    def image_embedding(self, filename, data):
        return self.vec

    def rank_labels_for_vector(self, image_vec, labels, top_k=6):
        self.label_calls.append(tuple(labels))
        return [(label, 0.2) for label in list(labels)[:top_k]]


class Storage:
    def download_file(self, key):
        return b"\xff\xd8\xff\xe0 bytes"


class CalibrationTestCase(unittest.TestCase):
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        db.init_db(self.db_path)
        self.conn = db.connect(self.db_path)
        self.conn.execute(
            "INSERT INTO photos (id, file_path, media_type) VALUES (1, 'photos/1.jpg', 'photo')"
        )
        self.conn.commit()

        self.cache = tempfile.mkdtemp(prefix="tagcache-")
        self.labels = list(vocabulary.LABELS)
        self.matrix = orthonormal_matrix(len(self.labels))

        # A photo that sits exactly on top of label 0 and nowhere else.
        self.vec = self.matrix[0].copy()
        self.sidecar = Sidecar(self.vec)

        self._patches = [
            patch.object(jobs, "_get_storage_client", lambda: Storage()),
            patch.object(jobs, "_get_embed_client", lambda: self.sidecar),
            patch.dict(os.environ, {tagger.CACHE_DIR_ENV: self.cache}),
        ]
        for p in self._patches:
            p.start()
            self.addCleanup(p.stop)
        jobs._TAG_CORPUS = None
        self.addCleanup(lambda: setattr(jobs, "_TAG_CORPUS", None))

    def tearDown(self):
        self.conn.close()
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def write_artifacts(self, model=MODEL, labels=None, corpus=None):
        """Write the pair `scripts/retag.py` leaves behind after an apply run."""
        labels = list(self.labels if labels is None else labels)
        fingerprint = vocabulary.vocab_fingerprint(labels=labels)
        tagger.save_cached_matrix(
            tagger.label_matrix_path(self.cache, model, fingerprint),
            self.matrix, model=model, fingerprint=fingerprint, labels=labels,
        )
        if corpus is None:
            # A corpus where label 0 is rare and everything else is flat-ish,
            # so only label 0 clears its own HIGH percentile for self.vec.
            corpus = orthonormal_matrix(200, seed=99) @ self.matrix.T
        calibration = tagger.calibrate(corpus, labels)
        tagger.save_calibration(
            tagger.calibration_path(self.cache, model, fingerprint),
            calibration, model=model, fingerprint=fingerprint,
        )
        return fingerprint

    def tags(self):
        cur = self.conn.cursor()
        cur.execute("SELECT tag, source FROM tags WHERE photo_id=1")
        return cur.fetchall()


class TestTheCalibratedPath(CalibrationTestCase):
    def test_it_tags_from_the_full_vocabulary_when_the_artifacts_are_there(self):
        self.write_artifacts()
        jobs.process_photo_embedding_job(1, "photos/1.jpg", self.db_path)

        tags = {r["tag"] for r in self.tags()}
        self.assertTrue(tags, "the calibrated path wrote no tags at all")
        self.assertIn(self.labels[0], tags)
        # A label outside the legacy 17 must be reachable, or nothing changed.
        self.assertTrue(
            tags - set(vocabulary.LEGACY_LABELS),
            f"only legacy labels were used: {sorted(tags)}",
        )

    def test_it_stamps_the_vocabulary_it_actually_used(self):
        self.write_artifacts()
        jobs.process_photo_embedding_job(1, "photos/1.jpg", self.db_path)

        sources = {r["source"] for r in self.tags()}
        self.assertEqual({vocabulary.tag_source(MODEL)}, sources)
        self.assertNotIn(db.DEFAULT_TAG_SOURCE, sources)

    def test_it_costs_the_sidecar_no_label_round_trips(self):
        """345 text embeds is ~30 s per photo in a work-horse that lives once."""
        self.write_artifacts()
        jobs.process_photo_embedding_job(1, "photos/1.jpg", self.db_path)

        self.assertEqual([], self.sidecar.label_calls)

    def test_the_tag_count_varies_instead_of_being_a_constant(self):
        """The legacy path's defining symptom was exactly 6 tags, always."""
        self.write_artifacts()
        jobs.process_photo_embedding_job(1, "photos/1.jpg", self.db_path)

        n = len(self.tags())
        self.assertGreaterEqual(n, tagger.MIN_TAGS_PER_PHOTO)
        self.assertLessEqual(n, tagger.MAX_TAGS_PER_PHOTO)


class TestTheFallback(CalibrationTestCase):
    def test_no_artifacts_means_the_legacy_seventeen(self):
        jobs.process_photo_embedding_job(1, "photos/1.jpg", self.db_path)

        self.assertEqual(1, len(self.sidecar.label_calls))
        self.assertEqual(tuple(vocabulary.LEGACY_LABELS), self.sidecar.label_calls[0])
        self.assertEqual(
            {vocabulary.tag_source(MODEL, vocabulary.LEGACY_VERSION)},
            {r["source"] for r in self.tags()},
        )

    def test_artifacts_for_another_model_are_not_used(self):
        """Thresholds are raw cosines; they mean nothing in another space."""
        self.write_artifacts(model="openai/clip-vit-base-patch32")
        jobs.process_photo_embedding_job(1, "photos/1.jpg", self.db_path)

        self.assertEqual(1, len(self.sidecar.label_calls))
        self.assertEqual(tuple(vocabulary.LEGACY_LABELS), self.sidecar.label_calls[0])

    def test_a_calibration_without_its_matrix_is_not_half_used(self):
        fingerprint = vocabulary.vocab_fingerprint(labels=self.labels)
        corpus = orthonormal_matrix(200, seed=99) @ self.matrix.T
        tagger.save_calibration(
            tagger.calibration_path(self.cache, MODEL, fingerprint),
            tagger.calibrate(corpus, self.labels),
            model=MODEL, fingerprint=fingerprint,
        )
        jobs.process_photo_embedding_job(1, "photos/1.jpg", self.db_path)

        self.assertEqual(1, len(self.sidecar.label_calls))


class TestCorruptionIsLoudNotSilent(CalibrationTestCase):
    def test_a_calibration_that_contradicts_its_own_name_raises(self):
        fingerprint = self.write_artifacts()
        path = tagger.calibration_path(self.cache, MODEL, fingerprint)
        meta = json.loads(Path(path).read_text())
        meta["model"] = "openai/clip-vit-base-patch32"
        Path(path).write_text(json.dumps(meta))

        with self.assertRaises(tagger.CacheMismatch):
            jobs.process_photo_embedding_job(1, "photos/1.jpg", self.db_path)

    def test_a_calibration_for_a_different_label_list_raises(self):
        fingerprint = self.write_artifacts()
        path = tagger.calibration_path(self.cache, MODEL, fingerprint)
        meta = json.loads(Path(path).read_text())
        meta["labels"] = list(reversed(meta["labels"]))
        Path(path).write_text(json.dumps(meta))

        with self.assertRaises(tagger.CacheMismatch):
            jobs.process_photo_embedding_job(1, "photos/1.jpg", self.db_path)


class TestTheCalibrationArtifact(unittest.TestCase):
    """Round trip, and the three ways it is allowed to refuse."""

    def setUp(self):
        self.cache = tempfile.mkdtemp(prefix="tagcache-")
        self.labels = ["a", "b", "c"]
        self.fp = "deadbeefdeadbeef"
        self.cal = tagger.calibrate(
            np.array([[0.1, 0.2, 0.3], [0.2, 0.4, 0.6], [0.3, 0.6, 0.9]]),
            self.labels,
        )
        self.path = tagger.calibration_path(self.cache, MODEL, self.fp)

    def test_absent_is_none_not_an_error(self):
        self.assertIsNone(tagger.load_calibration(
            self.path, model=MODEL, fingerprint=self.fp, labels=self.labels))

    def test_it_round_trips(self):
        tagger.save_calibration(self.path, self.cal, model=MODEL, fingerprint=self.fp)
        back = tagger.load_calibration(
            self.path, model=MODEL, fingerprint=self.fp, labels=self.labels)

        self.assertEqual(self.cal.labels, back.labels)
        np.testing.assert_allclose(self.cal.low, back.low)
        np.testing.assert_allclose(self.cal.high, back.high)
        self.assertEqual(self.cal.low_percentile, back.low_percentile)
        self.assertEqual(self.cal.high_percentile, back.high_percentile)
        self.assertEqual(self.cal.n_photos, back.n_photos)

    def test_it_records_how_stale_it_may_be(self):
        """Percentiles drift as the library grows; the operator needs the size."""
        tagger.save_calibration(self.path, self.cal, model=MODEL, fingerprint=self.fp)
        meta = json.loads(Path(self.path).read_text())

        self.assertEqual(3, meta["n_photos"])
        self.assertIn("written_at", meta)

    def test_a_wrong_model_raises(self):
        tagger.save_calibration(self.path, self.cal, model=MODEL, fingerprint=self.fp)
        with self.assertRaises(tagger.CacheMismatch):
            tagger.load_calibration(self.path, model="other",
                                    fingerprint=self.fp, labels=self.labels)

    def test_a_wrong_fingerprint_raises(self):
        tagger.save_calibration(self.path, self.cal, model=MODEL, fingerprint=self.fp)
        with self.assertRaises(tagger.CacheMismatch):
            tagger.load_calibration(self.path, model=MODEL,
                                    fingerprint="0000", labels=self.labels)

    def test_a_wrong_label_list_raises(self):
        tagger.save_calibration(self.path, self.cal, model=MODEL, fingerprint=self.fp)
        with self.assertRaises(tagger.CacheMismatch):
            tagger.load_calibration(self.path, model=MODEL, fingerprint=self.fp,
                                    labels=["a", "b", "z"])


class TestTheCacheDirectoryIsNotCwdSensitive(unittest.TestCase):
    """`faiss_indexes/` taught this lesson once already.

    A relative `models/` resolved against the process CWD means a worker
    started from anywhere but the repo root finds no artifacts, falls back to
    the legacy 17 forever, and says nothing about it.
    """

    def test_the_default_is_absolute_and_under_the_repo(self):
        self.assertTrue(tagger.resolve_cache_dir().is_absolute())
        self.assertEqual(
            Path(__file__).resolve().parent.parent / "models",
            tagger.resolve_cache_dir(),
        )

    def test_the_environment_overrides_it(self):
        with patch.dict(os.environ, {tagger.CACHE_DIR_ENV: "/tmp/elsewhere"}):
            self.assertEqual(Path("/tmp/elsewhere"), tagger.resolve_cache_dir())

    def test_an_explicit_argument_wins_over_the_environment(self):
        with patch.dict(os.environ, {tagger.CACHE_DIR_ENV: "/tmp/elsewhere"}):
            self.assertEqual(Path("/tmp/explicit"),
                             tagger.resolve_cache_dir("/tmp/explicit"))


if __name__ == "__main__":
    unittest.main()
