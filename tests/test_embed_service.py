"""The CLIP embedding sidecar.

One long-lived process holding the model the four uvicorn workers used to hold
four copies of. Everything here runs against a **stubbed** embedder — the point
of the sidecar is that CLIP is loaded once, in production, not once per test.

`import embed_service` must therefore stay cheap: torch and `core.embedder` are
imported inside the startup path, not at module scope, so this module can be
imported and exercised without dragging 450 MB into the test interpreter.
"""
import base64
import os
import sys
import threading
import types
import unittest
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

import embed_service


class StubEmbedder:
    """Stands in for `ClipEmbedder`, recording how it was called."""

    def __init__(self, dim=8):
        self.dim = dim
        self.text_calls = []
        self.image_calls = []
        self.threads_seen = []

    def _vec(self, seed):
        rng = np.random.default_rng(abs(hash(seed)) % (2**32))
        v = rng.standard_normal(self.dim).astype("float32")
        return v / np.linalg.norm(v)

    def text_embedding(self, text):
        self.text_calls.append((text, threading.current_thread().name))
        return self._vec(text)

    def image_embedding(self, path):
        with open(path, "rb") as fh:
            data = fh.read()
        self.image_calls.append((path, len(data), threading.current_thread().name))
        return self._vec(str(len(data)))


def make_client(embedder=None, **env):
    """A TestClient over the sidecar app with a stub embedder.

    Used as a context manager so the app's lifespan actually runs — that is
    where the model is constructed and the thread count is set.
    """
    embedder = embedder or StubEmbedder()
    app = embed_service.create_app(embedder_factory=lambda: embedder)
    return TestClient(app), embedder


def decode(payload):
    raw = base64.b64decode(payload["vector_b64"])
    return np.frombuffer(raw, dtype="float32")


class TestHealth(unittest.TestCase):
    def test_reports_ok_with_the_model_and_thread_count(self):
        client, _ = make_client()
        with client:
            resp = client.get("/health")
        self.assertEqual(200, resp.status_code)
        body = resp.json()
        self.assertEqual("ok", body["status"])
        self.assertIn("model", body)
        self.assertIsInstance(body["threads"], int)
        self.assertGreater(body["threads"], 0)

    def test_reports_resident_size(self):
        """The whole justification for this process is its memory; surface it."""
        client, _ = make_client()
        with client:
            body = client.get("/health").json()
        self.assertIn("rss_mb", body)
        self.assertGreater(body["rss_mb"], 0)


class TestTextEndpoint(unittest.TestCase):
    def test_round_trips_a_float32_vector_as_base64(self):
        client, stub = make_client()
        with client:
            resp = client.post("/embed/text", json={"text": "a dog on a beach"})

        self.assertEqual(200, resp.status_code)
        body = resp.json()
        self.assertEqual("float32", body["dtype"])
        self.assertEqual(stub.dim, body["dim"])
        vec = decode(body)
        self.assertEqual((stub.dim,), vec.shape)
        np.testing.assert_allclose(stub._vec("a dog on a beach"), vec, atol=1e-6)
        self.assertEqual("a dog on a beach", stub.text_calls[0][0])

    def test_returns_a_unit_vector(self):
        client, _ = make_client()
        with client:
            body = client.post("/embed/text", json={"text": "q"}).json()
        self.assertAlmostEqual(1.0, float(np.linalg.norm(decode(body))), places=5)

    def test_rejects_empty_text(self):
        client, _ = make_client()
        with client:
            self.assertEqual(422, client.post("/embed/text", json={"text": ""}).status_code)
            self.assertEqual(422, client.post("/embed/text", json={"text": "   "}).status_code)

    def test_rejects_a_missing_field(self):
        client, _ = make_client()
        with client:
            self.assertEqual(422, client.post("/embed/text", json={}).status_code)

    def test_embeds_off_the_event_loop(self):
        """A blocking forward pass on the loop thread would serialise the box.

        The API's search handler awaits this call; if the sidecar ran the model
        on its own event-loop thread it could not answer /health while busy.
        """
        client, stub = make_client()
        with client:
            client.post("/embed/text", json={"text": "q"})
        worker_thread = stub.text_calls[0][1]
        self.assertNotIn(
            "MainThread", worker_thread,
            f"the embed ran on {worker_thread!r}; it must go through run_in_executor",
        )


class TestImageEndpoint(unittest.TestCase):
    """Exposed from day one so routing the RQ workers here later is trivial."""

    def test_embeds_an_uploaded_image(self):
        client, stub = make_client()
        payload = b"\xff\xd8\xff\xe0not-really-a-jpeg"
        with client:
            resp = client.post("/embed/image", files={"file": ("x.jpg", payload, "image/jpeg")})

        self.assertEqual(200, resp.status_code)
        body = resp.json()
        self.assertEqual(stub.dim, body["dim"])
        self.assertEqual(len(payload), stub.image_calls[0][1])

    def test_does_not_leave_the_temp_file_behind(self):
        client, stub = make_client()
        with client:
            client.post("/embed/image", files={"file": ("x.jpg", b"abc", "image/jpeg")})
        path = stub.image_calls[0][0]
        self.assertFalse(os.path.exists(path), f"{path} was left on disk")


class TestThreadConfiguration(unittest.TestCase):
    """Task 6.2: the sidecar sets its own thread count explicitly.

    torch is stubbed rather than imported: 3 threads is the measured CLIP
    optimum on this 6-core box (113 ms/image embed vs 194-220 ms at the
    6-thread default), and proving we *call* set_num_threads does not require
    paying 450 MB to import the real thing.
    """

    def run_configure(self, env):
        fake_torch = types.ModuleType("torch")
        fake_torch.calls = []
        fake_torch.set_num_threads = lambda n: fake_torch.calls.append(n)
        fake_torch.get_num_threads = lambda: (fake_torch.calls[-1] if fake_torch.calls else 0)
        with patch.dict(sys.modules, {"torch": fake_torch}):
            with patch.dict(os.environ, env, clear=False):
                returned = embed_service.configure_threads()
        return returned, fake_torch.calls

    def test_defaults_to_three_threads(self):
        with patch.dict(os.environ, {}, clear=True):
            returned, calls = self.run_configure({})
        self.assertEqual(3, returned)
        self.assertEqual([3], calls)

    def test_honours_chitra_ml_threads(self):
        returned, calls = self.run_configure({"CHITRA_ML_THREADS": "2"})
        self.assertEqual(2, returned)
        self.assertEqual([2], calls)

    def test_ignores_a_nonsense_value_rather_than_crashing_at_startup(self):
        returned, calls = self.run_configure({"CHITRA_ML_THREADS": "banana"})
        self.assertEqual(3, returned)
        self.assertEqual([3], calls)

    def test_startup_sets_the_thread_count(self):
        fake_torch = types.ModuleType("torch")
        fake_torch.calls = []
        fake_torch.set_num_threads = lambda n: fake_torch.calls.append(n)
        fake_torch.get_num_threads = lambda: (fake_torch.calls[-1] if fake_torch.calls else 0)
        client, _ = make_client()
        with patch.dict(sys.modules, {"torch": fake_torch}):
            with client:
                client.get("/health")
        self.assertTrue(fake_torch.calls, "lifespan never called torch.set_num_threads")


class TestTokenAuth(unittest.TestCase):
    def test_rejects_a_missing_token_when_one_is_configured(self):
        with patch.dict(os.environ, {"CHITRA_EMBED_TOKEN": "s3cret"}, clear=False):
            client, _ = make_client()
            with client:
                self.assertEqual(401, client.post("/embed/text", json={"text": "q"}).status_code)
                ok = client.post(
                    "/embed/text",
                    json={"text": "q"},
                    headers={"Authorization": "Bearer s3cret"},
                )
                self.assertEqual(200, ok.status_code)

    def test_open_when_no_token_is_configured(self):
        env = {k: v for k, v in os.environ.items() if k != "CHITRA_EMBED_TOKEN"}
        with patch.dict(os.environ, env, clear=True):
            client, _ = make_client()
            with client:
                self.assertEqual(200, client.post("/embed/text", json={"text": "q"}).status_code)


class TestImportIsCheap(unittest.TestCase):
    """The module must be importable without loading the ML stack."""

    def test_module_scope_does_not_import_torch(self):
        import subprocess

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        code = (
            "import sys; import embed_service; "
            "print([m for m in ('torch','transformers') if m in sys.modules])"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=repo_root, capture_output=True, text=True, timeout=300,
        )
        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertEqual("[]", proc.stdout.strip().splitlines()[-1])


if __name__ == "__main__":
    unittest.main()


class TestLauncherWiring(unittest.TestCase):
    """The sidecar has to actually get started, and stopped, by the launchers.

    Until `chitra-embed.service` exists (owner action, needs sudo) the sidecar
    rides along with the RQ workers — the only unit whose ExecStart is a script
    in this repo. `CHITRA_EMBED_SELF_START=0` is the switch that hands it over
    to systemd later without a code change.
    """

    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def script(self, name):
        with open(os.path.join(self.REPO_ROOT, name)) as fh:
            return fh.read()

    def test_start_workers_launches_the_sidecar(self):
        body = self.script("start_workers.sh")
        self.assertIn("embed_service", body, "start_workers.sh never starts the sidecar")

    def test_sidecar_runs_exactly_one_uvicorn_worker(self):
        """Each worker would load its own 1.14 GB copy — the original bug."""
        body = self.script("start_workers.sh")
        self.assertIn("--workers 1", body)

    def test_sidecar_binds_loopback_only(self):
        body = self.script("start_workers.sh")
        self.assertIn("127.0.0.1", body)
        self.assertNotIn("--host 0.0.0.0", body)

    def test_self_start_is_switchable_and_defaults_on(self):
        body = self.script("start_workers.sh")
        self.assertIn("CHITRA_EMBED_SELF_START", body)
        self.assertIn("CHITRA_EMBED_SELF_START:-1", body)

    def test_sidecar_writes_a_pid_file(self):
        body = self.script("start_workers.sh")
        self.assertIn("embed.pid", body)

    def test_stop_workers_stops_the_sidecar(self):
        body = self.script("stop_workers.sh")
        self.assertIn("embed", body, "stop_workers.sh leaves the sidecar running")

    def test_stop_workers_pkill_does_not_sweep_up_the_sidecar_blindly(self):
        """`pkill -f worker.py` must not be the thing that kills the sidecar —
        it would leave the pid file lying and log nothing."""
        body = self.script("stop_workers.sh")
        self.assertIn("embed.pid", body)

    def test_env_example_documents_the_sidecar(self):
        body = self.script(".env.example")
        for var in ("CHITRA_EMBED_URL", "CHITRA_EMBED_SELF_START"):
            self.assertIn(var, body, f"{var} is undocumented")
