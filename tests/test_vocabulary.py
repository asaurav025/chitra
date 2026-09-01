"""Vocabulary and the pure tag scorer.

The thing under test here is arithmetic, not a model. `tag_from_vector` takes a
precomputed image vector and a matrix of label vectors and returns labels — no
CLIP, no file, no socket — so every assertion below is checkable by hand.
"""
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import tagger, vocabulary  # noqa: E402


LEGACY_17 = [
    "portrait", "selfie", "group photo", "family", "friends", "landscape",
    "city", "night", "sunset", "food", "pets", "indoors", "outdoors",
    "travel", "wedding", "party", "sports",
]


class TestVocabulary(unittest.TestCase):
    def test_a_few_hundred_labels(self):
        self.assertGreaterEqual(len(vocabulary.LABELS), 200)
        self.assertLessEqual(len(vocabulary.LABELS), 600)

    def test_labels_are_unique(self):
        self.assertEqual(len(vocabulary.LABELS), len(set(vocabulary.LABELS)))

    def test_every_legacy_label_survives(self):
        """No stored tag may be orphaned by the new vocabulary.

        All 11,456 rows in production carry one of these 17 strings. A label
        dropped here is a tag nothing can ever re-score or re-rank.
        """
        missing = [lab for lab in LEGACY_17 if lab not in vocabulary.LABELS]
        self.assertEqual(missing, [], f"legacy labels dropped: {missing}")

    def test_every_label_has_a_facet(self):
        for label in vocabulary.LABELS:
            self.assertIn(vocabulary.facet_of(label), vocabulary.FACETS,
                          f"{label!r} has no facet")

    def test_facets_partition_the_vocabulary(self):
        flat = [lab for labs in vocabulary.FACETS.values() for lab in labs]
        self.assertEqual(sorted(flat), sorted(vocabulary.LABELS))

    def test_several_facets_present(self):
        for expected in ("scene", "place", "activity", "object", "people",
                         "time_of_day", "season", "occasion", "photo_style"):
            self.assertIn(expected, vocabulary.FACETS)

    def test_prompt_template(self):
        self.assertEqual(vocabulary.PROMPT_TEMPLATE, "a photo of {label}")
        self.assertEqual(vocabulary.prompt_for("beach"), "a photo of beach")

    def test_prompts_align_with_labels(self):
        prompts = vocabulary.prompts()
        self.assertEqual(len(prompts), len(vocabulary.LABELS))
        self.assertEqual(prompts[0], vocabulary.prompt_for(vocabulary.LABELS[0]))

    def test_fingerprint_is_stable(self):
        self.assertEqual(vocabulary.vocab_fingerprint(),
                         vocabulary.vocab_fingerprint())

    def test_fingerprint_covers_the_labels(self):
        base = vocabulary.vocab_fingerprint()
        other = vocabulary.vocab_fingerprint(labels=vocabulary.LABELS + ("zzz",))
        self.assertNotEqual(base, other)

    def test_fingerprint_covers_the_version(self):
        base = vocabulary.vocab_fingerprint()
        other = vocabulary.vocab_fingerprint(version="not-the-version")
        self.assertNotEqual(base, other)

    def test_fingerprint_covers_the_template(self):
        """Changing the template changes every score, so it is part of identity.

        A cached label matrix built under "a photo of {label}" is worthless
        under "{label}" — the vectors move — and nothing else in the filename
        would say so.
        """
        base = vocabulary.vocab_fingerprint()
        other = vocabulary.vocab_fingerprint(template="{label}")
        self.assertNotEqual(base, other)

    def test_fingerprint_is_order_sensitive(self):
        """The matrix rows are positional; a reordered vocabulary is a different one."""
        reordered = tuple(reversed(vocabulary.LABELS))
        self.assertNotEqual(vocabulary.vocab_fingerprint(),
                            vocabulary.vocab_fingerprint(labels=reordered))

    def test_tag_source_names_model_and_version(self):
        src = vocabulary.tag_source("openai/clip-vit-base-patch32")
        self.assertIn(vocabulary.VOCAB_VERSION, src)
        self.assertTrue(src.startswith("clip-vitb32/"), src)

    def test_tag_source_differs_from_the_legacy_source(self):
        """Rollback is `DELETE FROM tags WHERE source = ...`; it needs a new value."""
        from core import db
        self.assertNotEqual(vocabulary.tag_source("openai/clip-vit-base-patch32"),
                            db.DEFAULT_TAG_SOURCE)


class TestTagFromVectorIsPure(unittest.TestCase):
    """Hand-built orthogonal matrix: the right answer is arithmetic."""

    def setUp(self):
        # Four mutually orthogonal unit label vectors in R^4. cosine with e_j
        # is exactly the j-th component of the image vector.
        self.labels = ("alpha", "beta", "gamma", "delta")
        self.matrix = np.eye(4, dtype="float32")

    def test_scores_are_the_components(self):
        vec = np.array([0.8, 0.1, 0.5, 0.3], dtype="float32")
        out = tagger.tag_from_vector(vec, self.matrix, self.labels, max_tags=4)
        self.assertEqual([lab for lab, _ in out], ["alpha", "gamma", "delta", "beta"])
        self.assertAlmostEqual(dict(out)["alpha"], 0.8, places=6)
        self.assertAlmostEqual(dict(out)["beta"], 0.1, places=6)

    def test_max_tags_truncates(self):
        vec = np.array([0.8, 0.1, 0.5, 0.3], dtype="float32")
        out = tagger.tag_from_vector(vec, self.matrix, self.labels, max_tags=2)
        self.assertEqual([lab for lab, _ in out], ["alpha", "gamma"])

    def test_no_network_no_files_no_model(self):
        """Pure means pure: no open(), no socket, no torch import."""
        import builtins
        import socket

        def boom_open(*a, **k):
            raise AssertionError(f"tag_from_vector opened a file: {a!r}")

        def boom_socket(*a, **k):
            raise AssertionError("tag_from_vector opened a socket")

        real_open, real_socket = builtins.open, socket.socket
        builtins.open, socket.socket = boom_open, boom_socket
        try:
            vec = np.array([0.8, 0.1, 0.5, 0.3], dtype="float32")
            tagger.tag_from_vector(vec, self.matrix, self.labels)
        finally:
            builtins.open, socket.socket = real_open, real_socket

    def test_rejects_a_mismatched_matrix(self):
        vec = np.zeros(4, dtype="float32")
        with self.assertRaises(ValueError):
            tagger.tag_from_vector(vec, np.eye(3, dtype="float32"),
                                   ("a", "b", "c"))

    def test_rejects_label_count_mismatch(self):
        vec = np.zeros(4, dtype="float32")
        with self.assertRaises(ValueError):
            tagger.tag_from_vector(vec, self.matrix, ("a", "b"))


class TestCalibration(unittest.TestCase):
    """Corpus-relative per-label calibration.

    Built so the corpus is known exactly: label 0 scores high for everyone,
    label 1 scores high for nobody but photo 0. Absolute score says label 0 is
    the better tag for photo 0; corpus-relative says label 1 is, because photo 0
    is the only photo that is at all label-1-ish.
    """

    def setUp(self):
        n = 200
        rng = np.random.default_rng(0)
        # label 0: everyone ~0.30.  label 1: everyone ~0.10 except photo 0 at 0.25.
        col0 = 0.30 + rng.normal(0, 0.005, n)
        col1 = 0.10 + rng.normal(0, 0.005, n)
        col1[0] = 0.25
        col0[0] = 0.30
        self.scores = np.stack([col0, col1], axis=1).astype("float32")
        self.labels = ("common", "rare")

    def test_calibration_has_one_pair_of_thresholds_per_label(self):
        cal = tagger.calibrate(self.scores, self.labels)
        self.assertEqual(len(cal.low), 2)
        self.assertEqual(len(cal.high), 2)
        self.assertTrue(np.all(cal.high >= cal.low))

    def test_calibration_rejects_a_shape_mismatch(self):
        with self.assertRaises(ValueError):
            tagger.calibrate(self.scores, ("only-one",))

    def test_corpus_relative_beats_absolute(self):
        cal = tagger.calibrate(self.scores, self.labels,
                               low_percentile=50.0, high_percentile=95.0)
        matrix = np.eye(2, dtype="float32")
        vec = self.scores[0].astype("float32")  # identity matrix -> scores back
        out = tagger.tag_from_vector(vec, matrix, self.labels, calibration=cal,
                                     max_tags=2, min_tags=0)
        self.assertEqual(out[0][0], "rare",
                         "absolute score prefers 'common'; calibration must not")

    def test_uncalibrated_ranking_is_the_absolute_one(self):
        matrix = np.eye(2, dtype="float32")
        vec = self.scores[0].astype("float32")
        out = tagger.tag_from_vector(vec, matrix, self.labels, max_tags=2)
        self.assertEqual(out[0][0], "common")

    def test_a_photo_below_every_low_threshold_gets_nothing(self):
        cal = tagger.calibrate(self.scores, self.labels,
                               low_percentile=50.0, high_percentile=95.0)
        matrix = np.eye(2, dtype="float32")
        vec = np.array([-1.0, -1.0], dtype="float32")
        out = tagger.tag_from_vector(vec, matrix, self.labels, calibration=cal,
                                     min_tags=0, max_tags=8)
        self.assertEqual(out, [])

    def test_min_tags_backfills_from_the_weak_tier_only(self):
        cal = tagger.calibrate(self.scores, self.labels,
                               low_percentile=50.0, high_percentile=95.0)
        matrix = np.eye(2, dtype="float32")
        vec = np.array([-1.0, -1.0], dtype="float32")
        out = tagger.tag_from_vector(vec, matrix, self.labels, calibration=cal,
                                     min_tags=2, max_tags=8)
        self.assertEqual(out, [], "min_tags must not invent tags below the low threshold")


class TestTagCountVaries(unittest.TestCase):
    """The regression that matters.

    Today every one of the 1,909 tagged photos has exactly 6.0 tags, because
    top-k returns k whether anything fits or not. If the new scorer also
    returns a constant, nothing has been fixed.
    """

    def _corpus(self, n=400, m=60, seed=7):
        rng = np.random.default_rng(seed)
        # Correlated corpus: a few latent factors, like real CLIP scores.
        latent = rng.normal(0, 1, (n, 6))
        loadings = rng.normal(0, 1, (6, m))
        raw = latent @ loadings
        # Squash into CLIP's measured 0.16-0.28 band so the test exercises the
        # regime the calibration actually has to work in.
        raw = 0.22 + 0.02 * (raw - raw.mean()) / (raw.std() + 1e-9)
        return raw.astype("float32")

    def test_counts_are_not_constant(self):
        scores = self._corpus()
        labels = tuple(f"l{i}" for i in range(scores.shape[1]))
        cal = tagger.calibrate(scores, labels)
        matrix = np.eye(scores.shape[1], dtype="float32")
        counts = [
            len(tagger.tag_from_vector(scores[i], matrix, labels, calibration=cal))
            for i in range(scores.shape[0])
        ]
        self.assertGreater(len(set(counts)), 1,
                           f"every photo got {counts[0]} tags — that is the old bug")

    def test_counts_stay_in_the_intended_band(self):
        scores = self._corpus()
        labels = tuple(f"l{i}" for i in range(scores.shape[1]))
        cal = tagger.calibrate(scores, labels)
        matrix = np.eye(scores.shape[1], dtype="float32")
        counts = [
            len(tagger.tag_from_vector(scores[i], matrix, labels, calibration=cal))
            for i in range(scores.shape[0])
        ]
        self.assertLessEqual(max(counts), tagger.MAX_TAGS_PER_PHOTO)
        self.assertGreater(float(np.mean(counts)), 1.0)
        self.assertLess(float(np.mean(counts)), float(tagger.MAX_TAGS_PER_PHOTO))

    def test_no_label_lands_on_nearly_everything(self):
        """`travel` is on 81.4% of the library. Calibration must bound that."""
        scores = self._corpus()
        labels = tuple(f"l{i}" for i in range(scores.shape[1]))
        cal = tagger.calibrate(scores, labels)
        matrix = np.eye(scores.shape[1], dtype="float32")
        hits = {}
        for i in range(scores.shape[0]):
            for lab, _ in tagger.tag_from_vector(scores[i], matrix, labels,
                                                 calibration=cal):
                hits[lab] = hits.get(lab, 0) + 1
        worst = max(hits.values()) / scores.shape[0]
        self.assertLess(worst, 0.40, f"one label covers {worst:.1%} of the corpus")


class TestTaggerImportsNoTorch(unittest.TestCase):
    def test_fresh_subprocess_import_leaves_torch_out(self):
        """`core/tagger.py:4` used to be `from core.embedder import ClipEmbedder`.

        The name was used in one annotation and cost every importer 1.1 GB of
        torch. The API tier already pays a memory-budget test for this; the
        tagger must not be the hole in it.
        """
        repo = Path(__file__).resolve().parent.parent
        code = (
            "import sys; import core.tagger as t;"
            "bad=[m for m in ('torch','transformers','core.embedder')"
            " if m in sys.modules];"
            "assert t.tag_from_vector is not None;"
            "print('LEAKED:'+','.join(bad) if bad else 'CLEAN')"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], cwd=str(repo),
            capture_output=True, text=True, timeout=180,
        )
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        self.assertIn("CLEAN", out.stdout, out.stdout + out.stderr[-2000:])

    def test_auto_tags_shim_still_exists(self):
        """cli/main.py and core/jobs.py both call it; the signature is a contract."""
        import inspect
        params = list(inspect.signature(tagger.auto_tags).parameters)
        self.assertEqual(params[:3], ["embedder", "image_path", "k"])

    def test_default_labels_still_exported(self):
        """scripts/reembed.py:784 imports it by name."""
        self.assertEqual(sorted(tagger.DEFAULT_LABELS), sorted(LEGACY_17))


if __name__ == "__main__":
    unittest.main()
