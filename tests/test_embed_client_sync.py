"""The *sync* client for the embedding sidecar — the one the RQ jobs use.

Why a second client
-------------------
`core/embed_client.py` is `async` because its caller is a uvicorn handler. The
RQ jobs are plain synchronous functions running in a forked work-horse, and
`asyncio.run()` per call would spin up and tear down an event loop for a single
HTTP round trip. So the jobs get their own thin client over `httpx.Client`,
sharing this module's wire format and — literally — its decode helper, so the
two can never disagree about what a payload means.

Two things here are load-bearing and are what these tests pin:

* **The timeout is not the API's 5 s.** A cold sidecar takes ~10 s to load CLIP,
  and a worker that gives up at 5 s would fail every job following a restart
  while the sidecar was perfectly healthy. `CHITRA_EMBED_JOB_TIMEOUT` defaults
  to 60 s.
* **No in-process fallback, ever.** A sidecar failure raises. Quietly building a
  `ClipEmbedder` here would restore the 58 s per-job model load and 1.67 GB of
  residency in the worker, invisibly — the jobs would just get slow again and
  nothing would say why.

`httpx.MockTransport` lets all of it be tested with no sidecar running.
"""
import base64
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np

from core.embed_client_sync import (
    DEFAULT_JOB_TIMEOUT,
    EmbeddingUnavailable,
    SyncEmbeddingClient,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def encode_vector(vec):
    """Encode a float32 vector exactly the way `embed_service._encode` does."""
    arr = np.asarray(vec, dtype="float32")
    return {
        "dim": int(arr.shape[0]),
        "dtype": "float32",
        "vector_b64": base64.b64encode(arr.tobytes()).decode("ascii"),
    }


def client_for(handler, **kwargs):
    """A `SyncEmbeddingClient` whose HTTP calls are served by `handler`."""
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return SyncEmbeddingClient(http=http, base_url="http://embed.test", **kwargs)


class TestImageEmbedding(unittest.TestCase):
    def test_posts_multipart_to_embed_image(self):
        seen = {}

        def handler(request):
            seen["method"] = request.method
            seen["url"] = str(request.url)
            seen["content_type"] = request.headers.get("content-type", "")
            seen["body"] = request.content
            return httpx.Response(200, json=encode_vector([1.0, 0.0]))

        client_for(handler).image_embedding("holiday.jpg", b"\xff\xd8\xff\xe0 jpeg bytes")

        self.assertEqual("POST", seen["method"])
        self.assertEqual("http://embed.test/embed/image", seen["url"])
        self.assertIn("multipart/form-data", seen["content_type"])
        # The sidecar names its temp file from the upload's filename.
        self.assertIn(b"holiday.jpg", seen["body"])
        self.assertIn(b"jpeg bytes", seen["body"])

    def test_decodes_base64_float32_into_a_unit_vector(self):
        original = np.array([3.0, 4.0, 0.0, 0.0], dtype="float32")

        def handler(request):
            return httpx.Response(200, json=encode_vector(original))

        vec = client_for(handler).image_embedding("a.jpg", b"x")

        self.assertEqual((4,), vec.shape)
        self.assertEqual("float32", vec.dtype.name)
        np.testing.assert_allclose(vec, [0.6, 0.8, 0.0, 0.0], atol=1e-6)

    def test_a_truncated_payload_is_an_error_not_a_short_vector(self):
        def handler(request):
            payload = encode_vector([1.0, 0.0])
            payload["dim"] = 512  # claims 512, carries 2
            return httpx.Response(200, json=payload)

        with self.assertRaises(EmbeddingUnavailable):
            client_for(handler).image_embedding("a.jpg", b"x")


class TestFailuresRaise(unittest.TestCase):
    """Every failure mode raises. None of them falls back to a local model."""

    def test_transport_error_raises(self):
        def handler(request):
            raise httpx.ConnectError("connection refused")

        with self.assertRaises(EmbeddingUnavailable) as ctx:
            client_for(handler).image_embedding("a.jpg", b"x")
        self.assertIn("http://embed.test", str(ctx.exception))

    def test_timeout_raises_and_names_the_timeout(self):
        def handler(request):
            raise httpx.ReadTimeout("too slow")

        with self.assertRaises(EmbeddingUnavailable) as ctx:
            client_for(handler, timeout=60.0).image_embedding("a.jpg", b"x")
        self.assertIn("60", str(ctx.exception))

    def test_non_200_raises_and_carries_the_status(self):
        def handler(request):
            return httpx.Response(503, text="model still loading")

        with self.assertRaises(EmbeddingUnavailable) as ctx:
            client_for(handler).image_embedding("a.jpg", b"x")
        self.assertIn("503", str(ctx.exception))

    def test_the_module_never_imports_the_in_process_embedder(self):
        """Checked over the parsed imports, not the text — the docstring
        explains at length why the fallback is absent, and a substring search
        would be satisfied by deleting the explanation."""
        import ast

        tree = ast.parse((REPO_ROOT / "core" / "embed_client_sync.py").read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
                imported.update(f"{node.module}.{a.name}" for a in node.names)

        for forbidden in ("core.embedder", "core.embedder.ClipEmbedder", "torch", "transformers"):
            self.assertNotIn(forbidden, imported)


class TestTimeoutConfiguration(unittest.TestCase):
    def test_default_is_60_seconds(self):
        self.assertEqual(60.0, DEFAULT_JOB_TIMEOUT)

    def test_default_exceeds_the_api_clients_timeout(self):
        """A cold sidecar takes ~10 s to load CLIP. The API's 5 s is right for a
        user waiting on a search; it is wrong for a job that can afford to
        wait, and would fail every job after a restart."""
        from core.embed_client import DEFAULT_TIMEOUT as API_TIMEOUT

        self.assertGreater(DEFAULT_JOB_TIMEOUT, API_TIMEOUT)
        self.assertGreaterEqual(DEFAULT_JOB_TIMEOUT, 10.0)

    def test_reads_CHITRA_EMBED_JOB_TIMEOUT(self):
        with patch.dict(os.environ, {"CHITRA_EMBED_JOB_TIMEOUT": "12.5"}):
            self.assertEqual(12.5, SyncEmbeddingClient().timeout)

    def test_a_nonsense_timeout_falls_back_to_the_default(self):
        with patch.dict(os.environ, {"CHITRA_EMBED_JOB_TIMEOUT": "soon"}):
            self.assertEqual(DEFAULT_JOB_TIMEOUT, SyncEmbeddingClient().timeout)

    def test_reads_CHITRA_EMBED_URL(self):
        with patch.dict(os.environ, {"CHITRA_EMBED_URL": "http://elsewhere:9/"}):
            self.assertEqual("http://elsewhere:9", SyncEmbeddingClient().base_url)


class TestToken(unittest.TestCase):
    def test_sends_a_bearer_token_when_configured(self):
        seen = {}

        def handler(request):
            seen["auth"] = request.headers.get("authorization")
            return httpx.Response(200, json=encode_vector([1.0, 0.0]))

        client_for(handler, token="s3cret").image_embedding("a.jpg", b"x")
        self.assertEqual("Bearer s3cret", seen["auth"])

    def test_sends_no_header_without_a_token(self):
        seen = {}

        def handler(request):
            seen["auth"] = request.headers.get("authorization")
            return httpx.Response(200, json=encode_vector([1.0, 0.0]))

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CHITRA_EMBED_TOKEN", None)
            client_for(handler).image_embedding("a.jpg", b"x")
        self.assertIsNone(seen["auth"])


class TestLabelRanking(unittest.TestCase):
    """Ranking labels against a vector we already have.

    `ClipEmbedder.rank_labels` embeds the image a *second* time to do this. The
    maths is a dot product between two normalised vectors, so with the image
    vector in hand it needs no image pass at all — and the label vectors are
    the same for every photo, so they are embedded once per process.
    """

    def test_ranks_by_cosine_against_the_given_vector(self):
        vectors = {"beach": [1.0, 0.0], "kitchen": [0.0, 1.0]}

        def handler(request):
            import json

            label = json.loads(request.content)["text"]
            return httpx.Response(200, json=encode_vector(vectors[label]))

        client = client_for(handler)
        ranked = client.rank_labels_for_vector(
            np.array([0.8, 0.6], dtype="float32"), ["beach", "kitchen"], top_k=2
        )

        self.assertEqual(["beach", "kitchen"], [t for t, _ in ranked])
        self.assertAlmostEqual(0.8, ranked[0][1], places=5)
        self.assertAlmostEqual(0.6, ranked[1][1], places=5)

    def test_top_k_truncates(self):
        def handler(request):
            return httpx.Response(200, json=encode_vector([1.0, 0.0]))

        ranked = client_for(handler).rank_labels_for_vector(
            np.array([1.0, 0.0], dtype="float32"), ["a", "b", "c"], top_k=2
        )
        self.assertEqual(2, len(ranked))

    def test_label_vectors_are_embedded_once_per_client(self):
        calls = []

        def handler(request):
            import json

            calls.append(json.loads(request.content)["text"])
            return httpx.Response(200, json=encode_vector([1.0, 0.0]))

        client = client_for(handler)
        vec = np.array([1.0, 0.0], dtype="float32")
        client.rank_labels_for_vector(vec, ["a", "b"], top_k=2)
        client.rank_labels_for_vector(vec, ["a", "b"], top_k=2)
        client.rank_labels_for_vector(vec, ["a", "b"], top_k=2)

        self.assertEqual(["a", "b"], calls)

    def test_no_labels_is_no_requests_and_no_tags(self):
        def handler(request):
            raise AssertionError("the sidecar must not be called for zero labels")

        self.assertEqual(
            [], client_for(handler).rank_labels_for_vector(np.zeros(2, "float32"), [])
        )


class TestServedModel(unittest.TestCase):
    """What the *resident* process is holding, not what this one's env names.

    `embeddings.model` is half that table's unique key and the value every
    ranking read filters on. Reading `CHITRA_EMBED_MODEL` in the worker would
    describe the worker's environment, which says nothing about the separate
    process that actually computed the vector — the exact mistake
    `scripts/reembed.py` was carrying, where a sidecar still serving CLIP
    filled `google/siglip2-...` rows with 512-d CLIP vectors.
    """

    def test_it_reports_the_model_from_health(self):
        def handler(request):
            self.assertEqual("/health", request.url.path)
            return httpx.Response(200, json={"status": "ok",
                                             "model": "google/siglip2-base-patch16-224",
                                             "dim": 768})

        self.assertEqual("google/siglip2-base-patch16-224",
                         client_for(handler).served_model())

    def test_it_asks_once_per_client(self):
        calls = []

        def handler(request):
            calls.append(request.url.path)
            return httpx.Response(200, json={"status": "ok", "model": "m", "dim": 4})

        client = client_for(handler)
        client.served_model()
        client.served_model()
        self.assertEqual(1, len(calls), calls)

    def test_a_health_payload_with_no_model_raises(self):
        """A default here writes a SigLIP vector under CLIP's name."""
        def handler(request):
            return httpx.Response(200, json={"status": "ok"})

        with self.assertRaises(EmbeddingUnavailable):
            client_for(handler).served_model()

    def test_an_unreachable_sidecar_raises(self):
        def handler(request):
            raise httpx.ConnectError("connection refused")

        with self.assertRaises(EmbeddingUnavailable):
            client_for(handler).served_model()

    def test_a_non_200_raises(self):
        def handler(request):
            return httpx.Response(503, text="model still loading")

        with self.assertRaises(EmbeddingUnavailable):
            client_for(handler).served_model()


class TestImportWeight(unittest.TestCase):
    """The point of the sidecar is that nothing else holds a model.

    Measured: `import core.tagger` costs 7.16 s and 509 MB because of its
    module-scope `from core.embedder import ClipEmbedder`. If this client ever
    picked up the same weight, every forked work-horse would pay it again and
    Phase 2 would have bought nothing.
    """

    def test_importing_it_leaves_torch_out_of_sys_modules(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import core.embed_client_sync as m; "
                "print(sorted(k for k in sys.modules "
                "if k in ('torch', 'transformers', 'core.embedder') "
                "or k.startswith('torch.')))",
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual("[]", result.stdout.strip(), result.stdout)


if __name__ == "__main__":
    unittest.main()
