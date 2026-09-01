"""`/api/search/photos` against the embedding sidecar.

The handler used to build a `ClipEmbedder` in-process — 1.14 GB resident per
uvicorn worker — and then call it *synchronously from an async handler*, which
blocked the event loop for every other request for the duration of the forward
pass. It now awaits `EmbeddingClient`.

These tests inject a fake client through `app.dependency_overrides`, so nothing
here loads a model or needs a sidecar running. They pin three things:

1. the ranking still works and still respects `min_score` and `limit`;
2. a sidecar failure propagates as 503 rather than being swallowed;
3. the client call is genuinely **awaited** — a coroutine left unawaited would
   raise a `TypeError` deep in numpy instead of failing here, so this is worth
   asserting explicitly.
"""
import asyncio
import os
import tempfile
import unittest
import unittest.mock
from unittest.mock import AsyncMock

import numpy as np
from fastapi import HTTPException
from fastapi.testclient import TestClient

os.environ.setdefault("CHITRA_DB_PATH", "/tmp/chitra_test.db")

import app_fastapi
from core import db_async


def unit(vec):
    v = np.asarray(vec, dtype="float32")
    return v / (np.linalg.norm(v) + 1e-9)


class FakeUser(dict):
    """Stands in for the aiosqlite.Row the auth dependency returns."""


class TestSearchEndpoint(unittest.TestCase):
    DIM = 4

    @classmethod
    def setUpClass(cls):
        cls.db_path = tempfile.mktemp(suffix=".db")
        asyncio.run(cls._seed())

    @classmethod
    async def _seed(cls):
        """Three photos whose embeddings are ordered against the query vector."""
        await db_async.init_db_async(cls.db_path)
        async with db_async.connect_async(cls.db_path) as conn:
            cls.vectors = {
                "near.jpg": unit([1.0, 0.0, 0.0, 0.0]),      # cosine 1.00
                "mid.jpg": unit([0.6, 0.8, 0.0, 0.0]),       # cosine 0.60
                "far.jpg": unit([0.0, 0.0, 1.0, 0.0]),       # cosine 0.00
            }
            for name, vec in cls.vectors.items():
                await db_async.upsert_photo_async(
                    conn, file_path=name, size=1, created_at="2026-01-01T00:00:00",
                    checksum=name, phash=None, exif_datetime=None,
                    latitude=None, longitude=None, media_type="photo",
                )
                cur = await conn.execute("SELECT id FROM photos WHERE file_path=?", (name,))
                photo_id = (await cur.fetchone())[0]
                await db_async.put_embedding_async(conn, photo_id, vec.tobytes(), cls.DIM)
            await conn.commit()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.db_path):
            os.unlink(cls.db_path)

    def client_with(self, embed_client):
        """A TestClient with auth, the DB and the embedding client overridden."""
        app = app_fastapi.app

        async def fake_db():
            async with db_async.connect_async(self.db_path) as conn:
                yield conn

        app.dependency_overrides[app_fastapi.get_db_async] = fake_db
        app.dependency_overrides[app_fastapi.get_current_active_user] = lambda: FakeUser(
            id=1, username="tester", role="user", is_active=1, is_whitelisted=1
        )
        app.dependency_overrides[app_fastapi.get_embedding_client] = lambda: embed_client
        self.addCleanup(app.dependency_overrides.clear)
        return TestClient(app)

    # ------------------------------------------------------------------
    def test_ranks_results_by_similarity(self):
        client = self.client_with(AsyncMock(text_embedding=AsyncMock(
            return_value=unit([1.0, 0.0, 0.0, 0.0]))))

        resp = client.get("/api/search/photos", params={"query": "a thing", "min_score": 0.0})

        self.assertEqual(200, resp.status_code, resp.text)
        body = resp.json()
        self.assertEqual(
            ["near.jpg", "mid.jpg", "far.jpg"],
            [os.path.basename(r["file_path"]) for r in body["results"]],
        )
        self.assertAlmostEqual(1.0, body["results"][0]["score"], places=4)
        self.assertAlmostEqual(0.6, body["results"][1]["score"], places=4)

    def test_passes_the_query_through_to_the_sidecar(self):
        fake = AsyncMock(text_embedding=AsyncMock(return_value=unit([1.0, 0.0, 0.0, 0.0])))
        client = self.client_with(fake)

        client.get("/api/search/photos", params={"query": "a dog on a beach"})

        fake.text_embedding.assert_awaited_once_with("a dog on a beach")

    def test_awaits_the_client_rather_than_calling_it_synchronously(self):
        """A blocking call here stalls the event loop for every other request."""
        fake = AsyncMock(text_embedding=AsyncMock(return_value=unit([1.0, 0.0, 0.0, 0.0])))
        client = self.client_with(fake)

        resp = client.get("/api/search/photos", params={"query": "q"})

        self.assertEqual(200, resp.status_code, resp.text)
        self.assertTrue(
            fake.text_embedding.await_count == 1,
            "the handler called the client but never awaited it",
        )

    def test_min_score_filters(self):
        client = self.client_with(AsyncMock(text_embedding=AsyncMock(
            return_value=unit([1.0, 0.0, 0.0, 0.0]))))

        body = client.get(
            "/api/search/photos", params={"query": "q", "min_score": 0.5}
        ).json()

        self.assertEqual(
            ["near.jpg", "mid.jpg"],
            [os.path.basename(r["file_path"]) for r in body["results"]],
        )

    def test_limit_caps_the_result_count(self):
        client = self.client_with(AsyncMock(text_embedding=AsyncMock(
            return_value=unit([1.0, 0.0, 0.0, 0.0]))))

        body = client.get(
            "/api/search/photos", params={"query": "q", "min_score": 0.0, "limit": 1}
        ).json()

        self.assertEqual(1, len(body["results"]))

    def test_a_dead_sidecar_surfaces_as_503(self):
        """Fail loud. A silent in-process fallback would re-create the OOM."""
        fake = AsyncMock(text_embedding=AsyncMock(
            side_effect=HTTPException(status_code=503, detail="search_unavailable: down")))
        client = self.client_with(fake)

        resp = client.get("/api/search/photos", params={"query": "q"})

        self.assertEqual(503, resp.status_code)
        self.assertIn("search_unavailable", resp.json()["detail"])

    def test_missing_query_still_400s_without_touching_the_sidecar(self):
        fake = AsyncMock(text_embedding=AsyncMock(return_value=unit([1.0, 0.0, 0.0, 0.0])))
        client = self.client_with(fake)

        resp = client.get("/api/search/photos")

        self.assertEqual(400, resp.status_code)
        fake.text_embedding.assert_not_awaited()


class TestHealthReportsEmbedStatus(unittest.TestCase):
    """`/api/health` has to say whether search actually works."""

    def test_health_includes_embed_status(self):
        app = app_fastapi.app
        fake = AsyncMock(health=AsyncMock(return_value="ok"))
        fake.base_url = "http://127.0.0.1:5101"
        app.dependency_overrides[app_fastapi.get_embedding_client] = lambda: fake
        self.addCleanup(app.dependency_overrides.clear)

        body = TestClient(app).get("/api/health").json()

        self.assertIn("embed_status", body)

    def test_health_reports_a_down_sidecar(self):
        app = app_fastapi.app
        fake = AsyncMock(health=AsyncMock(return_value="unavailable: connection refused"))
        fake.base_url = "http://127.0.0.1:5101"
        app.dependency_overrides[app_fastapi.get_embedding_client] = lambda: fake
        self.addCleanup(app.dependency_overrides.clear)

        body = TestClient(app).get("/api/health").json()

        self.assertIn("unavailable", body["embed_status"])


class TestNoEmbedderDependencyRemains(unittest.TestCase):
    """The old in-process seam must be gone, not merely unused."""

    def test_get_embedder_is_removed(self):
        self.assertFalse(
            hasattr(app_fastapi, "get_embedder"),
            "app_fastapi.get_embedder still exists — something can still "
            "construct a ClipEmbedder inside a uvicorn worker",
        )

    def test_module_neither_imports_nor_constructs_a_clip_embedder(self):
        """No import, no construction. A *mention* in a docstring is fine and
        wanted — the next reader should know what this replaced and why."""
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(repo_root, "app_fastapi.py")) as fh:
            source = fh.read()
        self.assertNotIn("from core.embedder", source)
        self.assertNotIn("import core.embedder", source)
        self.assertNotIn("ClipEmbedder(", source)


if __name__ == "__main__":
    unittest.main()


class TestSearchIsModelAware(unittest.TestCase):
    """`embeddings` is keyed on `(photo_id, model)`, so two models can coexist.

    The query handler could not survive that. `search_photos` selected *every*
    row and called `np.stack` on the lot; the `v.shape[0] != dim` guard passes a
    768-d row whose `dim` column says 768, so a single SigLIP row made numpy
    raise `ValueError: all input arrays must have the same shape` — a 500 on
    every search request, for every user, from one row.

    Scores from two models are not comparable either (CLIP cosine lives in
    0.16-0.28; SigLIP's sigmoid does not), so merging would be silently wrong
    even if it did not crash. The handler filters to one active model instead,
    named by `CHITRA_ACTIVE_EMBED_MODEL`.
    """

    CLIP_MODEL = "openai/clip-vit-base-patch32"
    SIGLIP_MODEL = "google/siglip2-base-patch16-224"
    CLIP_DIM = 512
    SIGLIP_DIM = 768

    @classmethod
    def setUpClass(cls):
        cls.db_path = tempfile.mktemp(suffix=".db")
        asyncio.run(cls._seed())

    @classmethod
    async def _seed(cls):
        """Two 512-d CLIP rows and one 768-d SigLIP row, exactly as a partially
        re-embedded library looks mid-migration."""
        await db_async.init_db_async(cls.db_path)

        def basis(dim, i, weight=1.0):
            v = np.zeros(dim, dtype="float32")
            v[i] = weight
            return unit(v)

        cls.clip_query = basis(cls.CLIP_DIM, 0)
        cls.siglip_query = basis(cls.SIGLIP_DIM, 0)

        rows = [
            # name,          model,            dim,             vector
            ("clip_near.jpg", cls.CLIP_MODEL, cls.CLIP_DIM,
             unit(np.eye(cls.CLIP_DIM, dtype="float32")[0])),
            ("clip_far.jpg", cls.CLIP_MODEL, cls.CLIP_DIM,
             unit(0.6 * np.eye(cls.CLIP_DIM, dtype="float32")[0]
                  + 0.8 * np.eye(cls.CLIP_DIM, dtype="float32")[1])),
            ("siglip_only.jpg", cls.SIGLIP_MODEL, cls.SIGLIP_DIM,
             unit(np.eye(cls.SIGLIP_DIM, dtype="float32")[0])),
        ]
        async with db_async.connect_async(cls.db_path) as conn:
            for name, model, dim, vec in rows:
                await db_async.upsert_photo_async(
                    conn, file_path=name, size=1, created_at="2026-01-01T00:00:00",
                    checksum=name, phash=None, exif_datetime=None,
                    latitude=None, longitude=None, media_type="photo",
                )
                cur = await conn.execute("SELECT id FROM photos WHERE file_path=?", (name,))
                photo_id = (await cur.fetchone())[0]
                await db_async.put_embedding_async(
                    conn, photo_id, vec.tobytes(), dim, model=model
                )
            await conn.commit()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.db_path):
            os.unlink(cls.db_path)

    def client_with(self, query_vec):
        app = app_fastapi.app

        async def fake_db():
            async with db_async.connect_async(self.db_path) as conn:
                yield conn

        app.dependency_overrides[app_fastapi.get_db_async] = fake_db
        app.dependency_overrides[app_fastapi.get_current_active_user] = lambda: FakeUser(
            id=1, username="tester", role="user", is_active=1, is_whitelisted=1
        )
        app.dependency_overrides[app_fastapi.get_embedding_client] = lambda: AsyncMock(
            text_embedding=AsyncMock(return_value=query_vec)
        )
        self.addCleanup(app.dependency_overrides.clear)
        return TestClient(app)

    # ------------------------------------------------------------------
    def test_a_foreign_dimension_row_does_not_500_the_endpoint(self):
        """The production bug: one 768-d row, every search request 500s."""
        with unittest.mock.patch.dict(
            os.environ, {"CHITRA_ACTIVE_EMBED_MODEL": self.CLIP_MODEL}
        ):
            client = self.client_with(self.clip_query)
            resp = client.get(
                "/api/search/photos", params={"query": "q", "min_score": 0.0}
            )

        self.assertEqual(200, resp.status_code, resp.text)

    def test_ranks_only_the_active_models_rows(self):
        with unittest.mock.patch.dict(
            os.environ, {"CHITRA_ACTIVE_EMBED_MODEL": self.CLIP_MODEL}
        ):
            client = self.client_with(self.clip_query)
            body = client.get(
                "/api/search/photos", params={"query": "q", "min_score": 0.0}
            ).json()

        self.assertEqual(
            ["clip_near.jpg", "clip_far.jpg"],
            [os.path.basename(r["file_path"]) for r in body["results"]],
        )

    def test_the_default_active_model_is_clip(self):
        """Unset means the model every stored row was written under, not 'all'."""
        env = {k: v for k, v in os.environ.items() if k != "CHITRA_ACTIVE_EMBED_MODEL"}
        with unittest.mock.patch.dict(os.environ, env, clear=True):
            client = self.client_with(self.clip_query)
            resp = client.get(
                "/api/search/photos", params={"query": "q", "min_score": 0.0}
            )

        self.assertEqual(200, resp.status_code, resp.text)
        self.assertEqual(
            ["clip_near.jpg", "clip_far.jpg"],
            [os.path.basename(r["file_path"]) for r in resp.json()["results"]],
        )

    def test_the_env_var_actually_selects_the_model(self):
        """Cutover is one config change — so flipping it must change the answer."""
        with unittest.mock.patch.dict(
            os.environ, {"CHITRA_ACTIVE_EMBED_MODEL": self.SIGLIP_MODEL}
        ):
            client = self.client_with(self.siglip_query)
            body = client.get(
                "/api/search/photos", params={"query": "q", "min_score": 0.0}
            ).json()

        self.assertEqual(
            ["siglip_only.jpg"],
            [os.path.basename(r["file_path"]) for r in body["results"]],
        )

    def test_no_rows_for_the_active_model_is_empty_not_a_500(self):
        """Mid-cutover with zero rows written yet: empty results, not an error."""
        with unittest.mock.patch.dict(
            os.environ, {"CHITRA_ACTIVE_EMBED_MODEL": "nobody/nothing"}
        ):
            client = self.client_with(self.clip_query)
            resp = client.get(
                "/api/search/photos", params={"query": "q", "min_score": 0.0}
            )

        self.assertEqual(200, resp.status_code, resp.text)
        self.assertEqual([], resp.json()["results"])


class TestGetEmbeddingsAsyncModelFilter(unittest.TestCase):
    """The filter belongs in the query, not in a post-hoc Python loop — the
    handler must never hold two models' vectors in the same list."""

    def test_filters_rows_to_one_model(self):
        db_path = tempfile.mktemp(suffix=".db")
        self.addCleanup(lambda: os.path.exists(db_path) and os.unlink(db_path))

        async def run():
            await db_async.init_db_async(db_path)
            async with db_async.connect_async(db_path) as conn:
                for name, model, dim in (
                    ("a.jpg", "model-a", 4),
                    ("b.jpg", "model-b", 6),
                ):
                    await db_async.upsert_photo_async(
                        conn, file_path=name, size=1,
                        created_at="2026-01-01T00:00:00", checksum=name,
                        phash=None, exif_datetime=None, latitude=None,
                        longitude=None, media_type="photo",
                    )
                    cur = await conn.execute(
                        "SELECT id FROM photos WHERE file_path=?", (name,))
                    pid = (await cur.fetchone())[0]
                    await db_async.put_embedding_async(
                        conn, pid, np.zeros(dim, dtype="float32").tobytes(),
                        dim, model=model,
                    )
                await conn.commit()
                only_a = await db_async.get_embeddings_async(conn, model="model-a")
                everything = await db_async.get_embeddings_async(conn)
                return only_a, everything

        only_a, everything = asyncio.run(run())

        self.assertEqual([4], [dim for _, dim, _ in only_a])
        self.assertEqual({4, 6}, {dim for _, dim, _ in everything},
                         "model=None must stay an unfiltered read")
