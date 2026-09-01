"""The API's client for the embedding sidecar.

Search used to build a `ClipEmbedder` inside every uvicorn worker: ~1.14 GB
resident per worker to do 12.7 ms of arithmetic per query. The model now lives
in one long-lived sidecar and the API talks to it over loopback HTTP.

Two properties are load-bearing and are what these tests pin:

* **No in-process fallback.** If the sidecar is down, search returns 503. A
  silent fallback to an in-process CLIP would re-create the exact OOM under
  load and would be invisible in the logs.
* **The call is genuinely awaited.** The old code called a blocking model
  straight from an async handler and stalled the event loop for every other
  request; an `httpx.AsyncClient` is only an improvement if it is awaited.

`httpx.MockTransport` lets all of this be tested with no sidecar running.
"""
import base64
import os
import sys
import textwrap
import unittest
from unittest.mock import patch

import httpx
import numpy as np
from fastapi import HTTPException

from core.embed_client import EmbeddingClient


def encode_vector(vec):
    """Encode a float32 vector the way the sidecar does."""
    arr = np.asarray(vec, dtype="float32")
    return {
        "dim": int(arr.shape[0]),
        "dtype": "float32",
        "vector_b64": base64.b64encode(arr.tobytes()).decode("ascii"),
    }


def client_for(handler, **kwargs):
    """An EmbeddingClient whose HTTP calls are served by `handler`."""
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return EmbeddingClient(http=http, base_url="http://embed.test", **kwargs)


class TestTextEmbedding(unittest.IsolatedAsyncioTestCase):
    async def test_posts_the_documented_wire_shape(self):
        seen = {}

        def handler(request):
            seen["method"] = request.method
            seen["url"] = str(request.url)
            seen["body"] = request.content
            return httpx.Response(200, json=encode_vector([1.0, 0.0]))

        await client_for(handler).text_embedding("a dog on a beach")

        self.assertEqual("POST", seen["method"])
        self.assertEqual("http://embed.test/embed/text", seen["url"])
        import json as _json
        self.assertEqual({"text": "a dog on a beach"}, _json.loads(seen["body"]))

    async def test_decodes_base64_float32_into_a_vector(self):
        original = np.array([0.6, 0.8, 0.0, 0.0], dtype="float32")

        def handler(request):
            return httpx.Response(200, json=encode_vector(original))

        got = await client_for(handler).text_embedding("q")

        self.assertEqual("float32", got.dtype.name)
        self.assertEqual((4,), got.shape)
        np.testing.assert_allclose(original, got, atol=1e-6)

    async def test_l2_normalises_the_result(self):
        """The handler's ranking is a cosine similarity; the vector must be unit."""
        def handler(request):
            return httpx.Response(200, json=encode_vector([3.0, 4.0]))

        got = await client_for(handler).text_embedding("q")

        self.assertAlmostEqual(1.0, float(np.linalg.norm(got)), places=5)
        np.testing.assert_allclose([0.6, 0.8], got, atol=1e-6)

    async def test_rejects_a_dim_that_disagrees_with_the_payload(self):
        """A truncated body must not silently become a shorter vector."""
        payload = encode_vector([1.0, 2.0, 3.0])
        payload["dim"] = 512

        def handler(request):
            return httpx.Response(200, json=payload)

        with self.assertRaises(HTTPException) as ctx:
            await client_for(handler).text_embedding("q")
        self.assertEqual(503, ctx.exception.status_code)

    async def test_sends_the_bearer_token_when_configured(self):
        seen = {}

        def handler(request):
            seen["auth"] = request.headers.get("authorization")
            return httpx.Response(200, json=encode_vector([1.0, 0.0]))

        await client_for(handler, token="s3cret").text_embedding("q")
        self.assertEqual("Bearer s3cret", seen["auth"])

    async def test_sends_no_auth_header_when_no_token(self):
        seen = {}

        def handler(request):
            seen["auth"] = request.headers.get("authorization")
            return httpx.Response(200, json=encode_vector([1.0, 0.0]))

        await client_for(handler).text_embedding("q")
        self.assertIsNone(seen["auth"])


class TestFailureIsLoud(unittest.IsolatedAsyncioTestCase):
    """Every failure mode must surface as 503, never as a silent local model."""

    async def assert_503(self, handler, needle="search_unavailable"):
        with self.assertRaises(HTTPException) as ctx:
            await client_for(handler).text_embedding("q")
        self.assertEqual(503, ctx.exception.status_code)
        self.assertIn(needle, str(ctx.exception.detail))
        return ctx.exception

    async def test_connect_error_becomes_503(self):
        def handler(request):
            raise httpx.ConnectError("connection refused", request=request)

        await self.assert_503(handler)

    async def test_read_timeout_becomes_503(self):
        def handler(request):
            raise httpx.ReadTimeout("timed out", request=request)

        await self.assert_503(handler)

    async def test_server_error_becomes_503(self):
        def handler(request):
            return httpx.Response(500, text="boom")

        await self.assert_503(handler)

    async def test_unparseable_body_becomes_503(self):
        def handler(request):
            return httpx.Response(200, text="not json")

        await self.assert_503(handler)


class TestNoInProcessFallback(unittest.TestCase):
    """The one thing this client must never do is load CLIP itself.

    This has to run in a **fresh subprocess**. `tests/run_tests.py` discovers
    every module into one interpreter, and `test_search` constructs a real
    `ClipEmbedder`, so an in-process `sys.modules` check would be polluted by
    whichever module ran first — it would fail here for a reason that has
    nothing to do with this client, and would pass in isolation for no reason
    at all.
    """

    def test_a_failed_call_loads_no_ml_runtime(self):
        import subprocess

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Port 1 is privileged and never listening: a real ConnectError, no stub.
        probe = textwrap.dedent(
            """
            import asyncio, sys
            from fastapi import HTTPException
            from core.embed_client import EmbeddingClient

            async def main():
                client = EmbeddingClient(base_url="http://127.0.0.1:1")
                try:
                    await client.text_embedding("q")
                except HTTPException as exc:
                    assert exc.status_code == 503, exc.status_code
                else:
                    raise AssertionError("a dead sidecar did not raise")
                finally:
                    await client.aclose()

            asyncio.run(main())
            print([m for m in ("torch", "transformers", "core.embedder")
                   if m in sys.modules])
            """
        )
        proc = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=repo_root, capture_output=True, text=True, timeout=300,
        )
        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertEqual(
            "[]", proc.stdout.strip().splitlines()[-1],
            "a failed embed call pulled in an ML runtime — a fallback would "
            "silently re-create the OOM this whole change exists to fix",
        )


class TestHealth(unittest.IsolatedAsyncioTestCase):
    async def test_reports_ok_when_the_sidecar_answers(self):
        def handler(request):
            self.assertEqual("http://embed.test/health", str(request.url))
            return httpx.Response(200, json={"status": "ok", "model": "clip"})

        self.assertEqual("ok", await client_for(handler).health())

    async def test_reports_unavailable_rather_than_raising(self):
        """/api/health must still answer when the sidecar is down."""
        def handler(request):
            raise httpx.ConnectError("connection refused", request=request)

        status = await client_for(handler).health()
        self.assertNotEqual("ok", status)
        self.assertIn("unavailable", status)


class TestConfiguration(unittest.TestCase):
    def test_defaults_to_loopback_port_5101(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual("http://127.0.0.1:5101", EmbeddingClient().base_url)

    def test_reads_url_timeout_and_token_from_the_environment(self):
        env = {
            "CHITRA_EMBED_URL": "http://127.0.0.1:9999/",
            "CHITRA_EMBED_TIMEOUT": "1.5",
            "CHITRA_EMBED_TOKEN": "abc",
        }
        with patch.dict(os.environ, env, clear=True):
            client = EmbeddingClient()
            self.assertEqual("http://127.0.0.1:9999", client.base_url)
            self.assertEqual(1.5, client.timeout)
            self.assertEqual("abc", client.token)


if __name__ == "__main__":
    unittest.main()
