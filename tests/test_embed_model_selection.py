"""Which model the sidecar serves, and why that is a *separate* question from
which model search ranks from.

The migration this exists for
-----------------------------
Moving the library from CLIP 512-d to SigLIP 2 768-d cannot be atomic: ~1,800
photos have to be re-embedded, and until every one of them has a 768-d row,
`/api/search/photos` must keep answering from the 512-d rows or it silently
loses every photo not yet converted.

That needs two independent switches:

* `CHITRA_EMBED_MODEL`        — what the sidecar *loads and computes with*.
                                Flipped first, at the start of the re-embed.
* `CHITRA_ACTIVE_EMBED_MODEL` — what `search_photos` *ranks from*
                                (`core.db_async.active_embed_model`).
                                Flipped last, only once coverage is complete.

If one variable drove both, there would be no window in which new SigLIP rows
are being written while search still answers from CLIP — which is the entire
migration. Hence the tests below assert the two are readable and settable
independently, not merely that each works.

`/health` reporting the dimension matters for the same reason: it is how an
operator confirms which model a running sidecar actually holds before flipping
the read side. A sidecar serving 768-d vectors while search expects 512-d
produces no error, just wrong answers.
"""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

import embed_service
from tests.test_embed_service import StubEmbedder, decode

CLIP_MODEL = "openai/clip-vit-base-patch32"
SIGLIP_MODEL = "google/siglip2-base-patch16-224"

LOAD_MODELS = os.environ.get("CHITRA_TEST_LOAD_MODELS") == "1"


class TestEmbedderFactory(unittest.TestCase):
    """`core.embedder.build_embedder` picks the class from the model name."""

    def test_builds_clip_for_a_clip_identifier(self):
        import core.embedder as em

        with patch.object(em, "ClipEmbedder") as clip, patch.object(em, "SiglipEmbedder") as sig:
            em.build_embedder(CLIP_MODEL)
        clip.assert_called_once_with(CLIP_MODEL)
        sig.assert_not_called()

    def test_builds_siglip_for_a_siglip_identifier(self):
        import core.embedder as em

        with patch.object(em, "ClipEmbedder") as clip, patch.object(em, "SiglipEmbedder") as sig:
            em.build_embedder(SIGLIP_MODEL)
        sig.assert_called_once_with(SIGLIP_MODEL)
        clip.assert_not_called()

    def test_reads_chitra_embed_model_when_given_nothing(self):
        import core.embedder as em

        with patch.dict(os.environ, {"CHITRA_EMBED_MODEL": SIGLIP_MODEL}):
            with patch.object(em, "SiglipEmbedder") as sig:
                em.build_embedder()
        sig.assert_called_once_with(SIGLIP_MODEL)

    def test_defaults_to_clip_so_an_unset_variable_keeps_todays_behaviour(self):
        import core.embedder as em

        env = {k: v for k, v in os.environ.items() if k != "CHITRA_EMBED_MODEL"}
        with patch.dict(os.environ, env, clear=True):
            with patch.object(em, "ClipEmbedder") as clip:
                em.build_embedder()
        clip.assert_called_once_with(em.DEFAULT_EMBED_MODEL)
        self.assertEqual(CLIP_MODEL, em.DEFAULT_EMBED_MODEL)

    def test_refuses_an_unknown_identifier_rather_than_guessing(self):
        """A typo must not silently fall back to CLIP and write 512-d rows
        under a name that says otherwise."""
        import core.embedder as em

        with self.assertRaises(ValueError):
            em.build_embedder("nobody/nothing-v9")


class TestInterfaceParity(unittest.TestCase):
    """SigLIP must be a drop-in, so the sidecar, sync client and tagger are
    untouched by the migration."""

    def test_siglip_exposes_the_same_surface_as_clip(self):
        from core.embedder import ClipEmbedder, SiglipEmbedder

        for name in ("image_embedding", "text_embedding", "rank_labels"):
            self.assertTrue(callable(getattr(SiglipEmbedder, name, None)), name)
            self.assertTrue(callable(getattr(ClipEmbedder, name, None)), name)

    def test_both_declare_their_dimension_without_being_loaded(self):
        """`/health` needs the dim, and asking the model for it would mean
        loading it just to answer a health check."""
        from core.embedder import ClipEmbedder, SiglipEmbedder

        self.assertEqual(512, ClipEmbedder.DIM)
        self.assertEqual(768, SiglipEmbedder.DIM)


class TestHealthReportsTheActiveModel(unittest.TestCase):
    def test_health_reports_model_and_dim(self):
        stub = StubEmbedder(dim=768)
        app = embed_service.create_app(embedder_factory=lambda: stub)
        with patch.dict(os.environ, {"CHITRA_EMBED_MODEL": SIGLIP_MODEL}):
            with TestClient(app) as client:
                body = client.get("/health").json()
        self.assertEqual(SIGLIP_MODEL, body["model"])
        self.assertEqual(768, body["dim"])

    def test_dim_follows_the_embedder_not_the_name(self):
        """If the loaded model disagrees with the configured name, the number
        that matters is what it actually produces."""
        stub = StubEmbedder(dim=512)
        app = embed_service.create_app(embedder_factory=lambda: stub)
        with patch.dict(os.environ, {"CHITRA_EMBED_MODEL": CLIP_MODEL}):
            with TestClient(app) as client:
                body = client.get("/health").json()
        self.assertEqual(512, body["dim"])

    def test_still_honours_the_legacy_chitra_clip_model(self):
        """Production sets CHITRA_CLIP_MODEL today; renaming the variable must
        not silently change which model a deployed box loads."""
        env = {k: v for k, v in os.environ.items() if k != "CHITRA_EMBED_MODEL"}
        env["CHITRA_CLIP_MODEL"] = CLIP_MODEL
        stub = StubEmbedder(dim=512)
        app = embed_service.create_app(embedder_factory=lambda: stub)
        with patch.dict(os.environ, env, clear=True):
            with TestClient(app) as client:
                body = client.get("/health").json()
        self.assertEqual(CLIP_MODEL, body["model"])


class TestTheTwoSwitchesAreSeparable(unittest.TestCase):
    """The load-bearing property of the whole migration."""

    def test_sidecar_can_serve_siglip_while_search_still_ranks_clip(self):
        from core.db_async import active_embed_model

        stub = StubEmbedder(dim=768)
        app = embed_service.create_app(embedder_factory=lambda: stub)
        with patch.dict(
            os.environ,
            {
                "CHITRA_EMBED_MODEL": SIGLIP_MODEL,
                "CHITRA_ACTIVE_EMBED_MODEL": CLIP_MODEL,
            },
        ):
            with TestClient(app) as client:
                body = client.get("/health").json()
            ranked_from = active_embed_model()

        self.assertEqual(SIGLIP_MODEL, body["model"], "sidecar should compute SigLIP")
        self.assertEqual(CLIP_MODEL, ranked_from, "search should still rank CLIP")
        self.assertNotEqual(body["model"], ranked_from)

    def test_flipping_the_read_side_does_not_disturb_the_sidecar(self):
        stub = StubEmbedder(dim=768)
        app = embed_service.create_app(embedder_factory=lambda: stub)
        seen = []
        for active in (CLIP_MODEL, SIGLIP_MODEL):
            with patch.dict(
                os.environ,
                {"CHITRA_EMBED_MODEL": SIGLIP_MODEL, "CHITRA_ACTIVE_EMBED_MODEL": active},
            ):
                with TestClient(app) as client:
                    seen.append(client.get("/health").json()["model"])
        self.assertEqual([SIGLIP_MODEL, SIGLIP_MODEL], seen)

    def test_sidecar_ignores_the_read_side_variable_entirely(self):
        """CHITRA_ACTIVE_EMBED_MODEL alone must never change what is loaded —
        otherwise the cutover flips both halves at once and the window closes."""
        env = {k: v for k, v in os.environ.items() if k != "CHITRA_EMBED_MODEL"}
        env["CHITRA_ACTIVE_EMBED_MODEL"] = SIGLIP_MODEL
        env.pop("CHITRA_CLIP_MODEL", None)
        stub = StubEmbedder(dim=512)
        app = embed_service.create_app(embedder_factory=lambda: stub)
        with patch.dict(os.environ, env, clear=True):
            with TestClient(app) as client:
                body = client.get("/health").json()
        self.assertEqual(embed_service.DEFAULT_MODEL, body["model"])


@unittest.skipUnless(
    LOAD_MODELS, "set CHITRA_TEST_LOAD_MODELS=1 to load SigLIP 2 (~1.8 GB)"
)
class TestSiglipEmbedderAgainstTheRealModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from core.embedder import SiglipEmbedder

        cls.emb = SiglipEmbedder(SIGLIP_MODEL)

    def _synthetic(self):
        import tempfile

        from tests.test_embedder_stability import write_synthetic_png

        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        write_synthetic_png(path)
        return path

    def test_image_embedding_is_768d_and_unit_norm(self):
        path = self._synthetic()
        try:
            v = self.emb.image_embedding(path)
        finally:
            os.unlink(path)
        self.assertEqual((768,), v.shape)
        self.assertEqual(np.dtype("float32"), v.dtype)
        self.assertAlmostEqual(1.0, float(np.linalg.norm(v)), places=5)

    def test_text_embedding_is_768d_and_unit_norm(self):
        v = self.emb.text_embedding("a photo of a beach at sunset")
        self.assertEqual((768,), v.shape)
        self.assertAlmostEqual(1.0, float(np.linalg.norm(v)), places=5)

    def test_text_tower_actually_discriminates(self):
        """Guards against a text tower that loads but returns garbage — the
        failure mode that would make free-text search quietly useless."""
        a = self.emb.text_embedding("a photo of a dog")
        b = self.emb.text_embedding("a photo of a puppy")
        c = self.emb.text_embedding("a screenshot of a spreadsheet")
        self.assertGreater(float(a @ b), float(a @ c))

    def test_rank_labels_returns_sorted_pairs(self):
        path = self._synthetic()
        try:
            got = self.emb.rank_labels(path, ["a beach", "a dog", "a diagram"], top_k=2)
        finally:
            os.unlink(path)
        self.assertEqual(2, len(got))
        self.assertTrue(all(isinstance(lbl, str) for lbl, _ in got))
        self.assertGreaterEqual(got[0][1], got[1][1])

    def test_rank_labels_on_no_labels_is_empty(self):
        self.assertEqual([], self.emb.rank_labels("unused.png", []))


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(
    LOAD_MODELS, "set CHITRA_TEST_LOAD_MODELS=1 to load SigLIP 2 (~1.8 GB)"
)
class TestSiglipTextPadding(unittest.TestCase):
    """SigLIP pads every sequence to its full 64-token context.

    Padding to the longest item in the batch instead would make a label's score
    depend on which other labels were ranked with it — the same tag would score
    differently in a 3-label call and a 345-label call, with nothing to show
    for it in the output.
    """

    @classmethod
    def setUpClass(cls):
        from core.embedder import SiglipEmbedder

        cls.emb = SiglipEmbedder(SIGLIP_MODEL)

    def test_every_sequence_is_padded_to_the_full_context(self):
        self.assertEqual(64, self.emb.text_max_length)
        for texts in (["hi"], ["hi", "a substantially longer descriptive caption"]):
            ids = self.emb.processor(
                text=texts,
                padding=self.emb.TEXT_PADDING,
                truncation=True,
                max_length=self.emb.text_max_length,
                return_tensors="pt",
            )["input_ids"]
            self.assertEqual((len(texts), 64), tuple(ids.shape))

    def test_a_string_embeds_identically_alone_and_in_a_batch(self):
        alone = self.emb._text_matrix(["a beach"])[0]
        batched = self.emb._text_matrix(
            ["a beach", "a photograph of an extremely long descriptive caption here"]
        )[0]
        np.testing.assert_allclose(alone, batched, atol=1e-5)
