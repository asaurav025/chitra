"""The four read-only discovery endpoints in `core/routes_discovery.py`.

Everything here is computable from data already in SQLite — tag counts, a tag
join, a dot product against stored vectors, and Hamming distance over stored
pHashes. Nothing reads MinIO and nothing loads a model, which is why these
could ship without waiting on the re-embed.

The tests override `get_current_active_user` and `get_db_async` on the real
app, so they exercise the real routing and the real dependency graph without
joining the seven known `test_endpoints` auth failures. One test deliberately
does *not* override auth: a missing auth dependency is silent — the route just
serves anyone who reaches the tunnel — so it has to be asserted, not assumed.
"""
import asyncio
import os
import tempfile
import unittest

import numpy as np
from fastapi.testclient import TestClient

os.environ.setdefault("CHITRA_DB_PATH", "/tmp/chitra_test.db")

import app_fastapi
from core import db_async, routes_discovery


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIP_MODEL = "openai/clip-vit-base-patch32"
SIGLIP_MODEL = "google/siglip2-base-patch16-224"


class FakeUser(dict):
    """Stands in for the aiosqlite.Row the auth dependency returns."""


def unit(vec):
    v = np.asarray(vec, dtype="float32")
    return v / (np.linalg.norm(v) + 1e-9)


class DiscoveryFixture(unittest.TestCase):
    """A small library: five photos, tags, embeddings and pHashes."""

    @classmethod
    def setUpClass(cls):
        cls.db_path = tempfile.mktemp(suffix=".db")
        asyncio.run(cls._seed())

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.db_path):
            os.unlink(cls.db_path)

    # pHashes chosen so the grouping is a transitive closure and not a
    # pairwise listing: dup_a~dup_b (5 bits) and dup_b~dup_c (5 bits), but
    # dup_a and dup_c are 10 bits apart and never directly compared under
    # max_distance=8. A pairwise implementation reports two groups; the right
    # answer is one group of three.
    PHASHES = {
        "dup_a.jpg": "0000000000000000",   # 0
        "dup_b.jpg": "000000000000001f",   # 5 bits from a
        "dup_c.jpg": "0000000000001f1f",   # 5 bits from b, 10 from a
        "lonely.jpg": "ffffffffff000000",  # 40 bits from everything
        "novec.jpg": None,                 # no pHash at all
    }

    @classmethod
    async def _seed(cls):
        await db_async.init_db_async(cls.db_path)
        async with db_async.connect_async(cls.db_path) as conn:
            for name, phash in cls.PHASHES.items():
                await db_async.upsert_photo_async(
                    conn, file_path=name, size=1,
                    created_at="2026-01-01T00:00:00", checksum=name,
                    phash=phash, exif_datetime=None, latitude=None,
                    longitude=None, media_type="photo",
                )
            cls.ids = {}
            async with conn.execute("SELECT id, file_path FROM photos") as cur:
                for row in await cur.fetchall():
                    cls.ids[row["file_path"]] = row["id"]

            # Tags: `beach` on three photos with distinct scores, `sunset` on
            # one, so GROUP BY counts and min_score filtering are separable.
            tags = [
                ("dup_a.jpg", "beach", 0.30),
                ("dup_b.jpg", "beach", 0.20),
                ("dup_c.jpg", "beach", 0.10),
                ("lonely.jpg", "sunset", 0.25),
            ]
            for name, tag, score in tags:
                await db_async.add_tag_async(conn, cls.ids[name], tag, score)

            # Embeddings: four CLIP rows whose similarity to dup_a is known,
            # plus one 768-d SigLIP row that must never enter the matrix.
            dim = 8
            vecs = {
                "dup_a.jpg": unit([1, 0, 0, 0, 0, 0, 0, 0]),      # the probe
                "dup_b.jpg": unit([1, 0, 0, 0, 0, 0, 0, 0]),      # cosine 1.00
                "dup_c.jpg": unit([0.6, 0.8, 0, 0, 0, 0, 0, 0]),  # cosine 0.60
                "lonely.jpg": unit([0, 1, 0, 0, 0, 0, 0, 0]),     # cosine 0.00
            }
            for name, vec in vecs.items():
                await db_async.put_embedding_async(
                    conn, cls.ids[name], vec.tobytes(), dim, model=CLIP_MODEL
                )
            await db_async.put_embedding_async(
                conn, cls.ids["dup_a.jpg"],
                np.zeros(768, dtype="float32").tobytes(), 768,
                model=SIGLIP_MODEL,
            )
            await conn.commit()

    def client(self, *, with_auth=True):
        app = app_fastapi.app

        async def fake_db():
            async with db_async.connect_async(self.db_path) as conn:
                yield conn

        app.dependency_overrides[app_fastapi.get_db_async] = fake_db
        if with_auth:
            app.dependency_overrides[app_fastapi.get_current_active_user] = (
                lambda: FakeUser(id=1, username="tester", role="user",
                                 is_active=1, is_whitelisted=1)
            )
        self.addCleanup(app.dependency_overrides.clear)
        return TestClient(app)


# ----------------------------------------------------------------------
# GET /api/tags
# ----------------------------------------------------------------------
class TestTagsEndpoint(DiscoveryFixture):

    def test_returns_the_vocabulary_with_counts(self):
        body = self.client().get("/api/tags").json()

        by_tag = {t["tag"]: t for t in body["tags"]}
        self.assertEqual({"beach", "sunset"}, set(by_tag))
        self.assertEqual(3, by_tag["beach"]["count"])
        self.assertEqual(1, by_tag["sunset"]["count"])
        self.assertEqual(2, body["total"])

    def test_orders_by_count_descending(self):
        body = self.client().get("/api/tags").json()

        self.assertEqual(["beach", "sunset"], [t["tag"] for t in body["tags"]])

    def test_reports_the_score_range_per_tag(self):
        """`travel` sits on 80.6% of the library at a score indistinguishable
        from noise. Counts alone hide that; the score range is the tell."""
        body = self.client().get("/api/tags").json()
        beach = next(t for t in body["tags"] if t["tag"] == "beach")

        self.assertAlmostEqual(0.30, beach["max_score"], places=4)
        self.assertAlmostEqual(0.10, beach["min_score"], places=4)
        self.assertAlmostEqual(0.20, beach["avg_score"], places=4)

    def test_requires_authentication(self):
        resp = self.client(with_auth=False).get("/api/tags")

        self.assertEqual(401, resp.status_code, resp.text)


# ----------------------------------------------------------------------
# GET /api/search/by-tag
# ----------------------------------------------------------------------
class TestSearchByTag(DiscoveryFixture):

    def test_mirrors_the_by_person_response_shape(self):
        body = self.client().get(
            "/api/search/by-tag", params={"tag": "beach"}).json()

        self.assertEqual("beach", body["query"])
        self.assertEqual(
            {"dup_a.jpg", "dup_b.jpg", "dup_c.jpg"},
            {r["file_path"] for r in body["results"]},
        )

    def test_ranks_by_tag_score(self):
        body = self.client().get(
            "/api/search/by-tag", params={"tag": "beach"}).json()

        self.assertEqual(
            ["dup_a.jpg", "dup_b.jpg", "dup_c.jpg"],
            [r["file_path"] for r in body["results"]],
        )
        self.assertAlmostEqual(0.30, body["results"][0]["score"], places=4)

    def test_min_score_filters(self):
        body = self.client().get(
            "/api/search/by-tag",
            params={"tag": "beach", "min_score": 0.15},
        ).json()

        self.assertEqual(
            ["dup_a.jpg", "dup_b.jpg"],
            [r["file_path"] for r in body["results"]],
        )

    def test_limit_caps_the_result_count(self):
        body = self.client().get(
            "/api/search/by-tag", params={"tag": "beach", "limit": 1}).json()

        self.assertEqual(1, len(body["results"]))

    def test_a_missing_tag_is_a_400(self):
        resp = self.client().get("/api/search/by-tag")

        self.assertEqual(400, resp.status_code)
        self.assertEqual("missing_tag", resp.json()["detail"])

    def test_an_unknown_tag_is_empty_not_a_404(self):
        resp = self.client().get(
            "/api/search/by-tag", params={"tag": "nothing-has-this"})

        self.assertEqual(200, resp.status_code, resp.text)
        self.assertEqual([], resp.json()["results"])

    def test_requires_authentication(self):
        resp = self.client(with_auth=False).get(
            "/api/search/by-tag", params={"tag": "beach"})

        self.assertEqual(401, resp.status_code, resp.text)


# ----------------------------------------------------------------------
# GET /api/photos/{id}/similar
# ----------------------------------------------------------------------
class TestSimilarPhotos(DiscoveryFixture):

    def test_ranks_neighbours_and_drops_the_probe_itself(self):
        pid = self.ids["dup_a.jpg"]

        body = self.client().get(f"/api/photos/{pid}/similar").json()

        self.assertEqual(
            ["dup_b.jpg", "dup_c.jpg", "lonely.jpg"],
            [r["file_path"] for r in body["results"]],
        )
        self.assertNotIn(pid, [r["id"] for r in body["results"]])
        self.assertAlmostEqual(1.0, body["results"][0]["score"], places=4)
        self.assertAlmostEqual(0.6, body["results"][1]["score"], places=4)

    def test_a_foreign_model_row_does_not_break_the_matrix(self):
        """dup_a also carries a 768-d SigLIP row. Unfiltered, `np.stack`
        raises and this endpoint 500s the moment a re-embed starts."""
        pid = self.ids["dup_a.jpg"]

        resp = self.client().get(f"/api/photos/{pid}/similar")

        self.assertEqual(200, resp.status_code, resp.text)

    def test_limit_caps_the_result_count(self):
        pid = self.ids["dup_a.jpg"]

        body = self.client().get(
            f"/api/photos/{pid}/similar", params={"limit": 1}).json()

        self.assertEqual(1, len(body["results"]))

    def test_a_photo_with_no_embedding_is_a_404(self):
        pid = self.ids["novec.jpg"]

        resp = self.client().get(f"/api/photos/{pid}/similar")

        self.assertEqual(404, resp.status_code, resp.text)
        self.assertEqual("no_embedding", resp.json()["detail"])

    def test_an_unknown_photo_is_a_404(self):
        resp = self.client().get("/api/photos/999999/similar")

        self.assertEqual(404, resp.status_code, resp.text)

    def test_requires_authentication(self):
        pid = self.ids["dup_a.jpg"]

        resp = self.client(with_auth=False).get(f"/api/photos/{pid}/similar")

        self.assertEqual(401, resp.status_code, resp.text)


# ----------------------------------------------------------------------
# GET /api/duplicates
# ----------------------------------------------------------------------
class TestDuplicates(DiscoveryFixture):

    def test_groups_by_transitive_closure_not_by_pair(self):
        """dup_a~dup_b and dup_b~dup_c, but dup_a and dup_c are 10 bits
        apart. One group of three, not two groups of two."""
        body = self.client().get(
            "/api/duplicates", params={"max_distance": 8}).json()

        self.assertEqual(1, body["group_count"])
        group = body["groups"][0]
        self.assertEqual(3, group["size"])
        self.assertEqual(
            {"dup_a.jpg", "dup_b.jpg", "dup_c.jpg"},
            {p["file_path"] for p in group["photos"]},
        )

    def test_a_tighter_distance_splits_the_closure(self):
        """At max_distance=4 no pair is within range, so nothing groups."""
        body = self.client().get(
            "/api/duplicates", params={"max_distance": 4}).json()

        self.assertEqual(0, body["group_count"])

    def test_a_looser_distance_still_excludes_the_outlier(self):
        body = self.client().get(
            "/api/duplicates", params={"max_distance": 12}).json()

        joined = {p["file_path"] for g in body["groups"] for p in g["photos"]}
        self.assertNotIn("lonely.jpg", joined)
        self.assertNotIn("novec.jpg", joined)

    def test_echoes_the_distance_it_used(self):
        body = self.client().get(
            "/api/duplicates", params={"max_distance": 8}).json()

        self.assertEqual(8, body["max_distance"])

    def test_requires_authentication(self):
        resp = self.client(with_auth=False).get("/api/duplicates")

        self.assertEqual(401, resp.status_code, resp.text)


class TestPhashGrouping(unittest.TestCase):
    """The grouping is a pure function so the closure can be tested without
    a database, and so the handler can hand it straight to an executor."""

    def test_transitive_closure(self):
        groups = routes_discovery.group_by_phash(
            [(1, "0000000000000000"), (2, "000000000000001f"),
             (3, "0000000000001f1f")],
            max_distance=8,
        )

        self.assertEqual([[1, 2, 3]], groups)

    def test_singletons_are_not_groups(self):
        groups = routes_discovery.group_by_phash(
            [(1, "0000000000000000"), (2, "ffffffffffffffff")],
            max_distance=8,
        )

        self.assertEqual([], groups)

    def test_identical_hashes_group_at_distance_zero(self):
        groups = routes_discovery.group_by_phash(
            [(1, "0000000000000000"), (2, "0000000000000000"),
             (3, "000000000000001f")],
            max_distance=0,
        )

        self.assertEqual([[1, 2]], groups)

    def test_unparseable_hashes_are_skipped_not_fatal(self):
        groups = routes_discovery.group_by_phash(
            [(1, "0000000000000000"), (2, "0000000000000000"),
             (3, "not-a-hash"), (4, "")],
            max_distance=0,
        )

        self.assertEqual([[1, 2]], groups)

    def test_groups_are_ordered_largest_first(self):
        groups = routes_discovery.group_by_phash(
            [(1, "0000000000000000"), (2, "0000000000000000"),
             (3, "0000000000000000"),
             (4, "ffffffffffffffff"), (5, "ffffffffffffffff")],
            max_distance=0,
        )

        self.assertEqual([[1, 2, 3], [4, 5]], groups)

    def test_chunking_does_not_change_the_answer(self):
        """The all-pairs matrix is chunked so memory stays bounded as the
        library grows. A chunk boundary must not hide a pair."""
        rows = [(i, f"{i:016x}") for i in range(200)]

        whole = routes_discovery.group_by_phash(rows, max_distance=1)
        chunked = routes_discovery.group_by_phash(rows, max_distance=1, chunk=7)

        self.assertEqual(whole, chunked)
        self.assertNotEqual([], whole)


# ----------------------------------------------------------------------
# The router must not drag ML into the API tier
# ----------------------------------------------------------------------
class TestRouterStaysLight(unittest.TestCase):
    """`import app_fastapi` has a 200 MB budget with no torch resident
    (`tests/test_api_memory_budget.py`). This router is imported by it."""

    SOURCE = os.path.join(REPO_ROOT, "core", "routes_discovery.py")

    #: Roots that drag the ML stack in. `torch` alone is ~1.1 GB resident.
    FORBIDDEN_ROOTS = {"torch", "transformers", "insightface", "onnxruntime"}
    FORBIDDEN_MODULES = {"core.embedder", "core.tagger", "core.face"}

    def _imported_modules(self):
        """Every module named by a real `import` statement in the source.

        Parsed rather than grepped: a substring search both misses
        `import torch.nn as nn` and trips over the module docstring saying
        the module must not import torch, which is prose worth keeping.
        """
        import ast

        with open(self.SOURCE) as fh:
            tree = ast.parse(fh.read())

        named = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                named.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                named.add(node.module)
                named.update(f"{node.module}.{a.name}" for a in node.names)
        return named

    def test_imports_neither_torch_nor_the_embedder(self):
        named = self._imported_modules()

        roots = {name.split(".")[0] for name in named}
        self.assertEqual(
            set(), roots & self.FORBIDDEN_ROOTS,
            f"the discovery router imports the ML stack: {roots & self.FORBIDDEN_ROOTS}",
        )
        self.assertEqual(
            set(), named & self.FORBIDDEN_MODULES,
            f"the discovery router imports a model-loading module: "
            f"{named & self.FORBIDDEN_MODULES}",
        )

    def test_no_heavy_runtime_is_resident_after_importing_the_router(self):
        """The AST check above covers direct imports; this covers transitive
        ones — a light-looking helper that itself imports the ML stack.

        Run in a **fresh subprocess** on purpose. `tests/run_tests.py`
        discovers every module into one interpreter and `test_search` builds a
        real `ClipEmbedder`, so an in-process `sys.modules` assertion would
        pass or fail on test ordering rather than on this module.
        """
        import json
        import subprocess
        import sys
        import textwrap

        probe = textwrap.dedent(
            """
            import json, sys
            import core.routes_discovery  # noqa: F401
            print(json.dumps([m for m in ("torch", "transformers",
                                          "onnxruntime", "insightface")
                              if m in sys.modules]))
            """
        )
        proc = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
        )

        self.assertEqual(0, proc.returncode, proc.stderr)
        resident = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertEqual(
            [], resident,
            f"importing the discovery router pulled {resident} into the API tier",
        )

    def test_the_all_pairs_scan_runs_off_the_event_loop(self):
        """0.18 s over 1,715 hashes. Inline, that is 180 ms during which the
        single uvicorn event loop serves nobody."""
        with open(self.SOURCE) as fh:
            source = fh.read()

        self.assertIn("run_in_executor", source)

    def test_every_route_carries_an_auth_dependency(self):
        """A missing dependency is silent — the route serves anyone who can
        reach the tunnel. Checked structurally, not just per-endpoint."""
        paths = {"/api/tags", "/api/search/by-tag", "/api/duplicates",
                 "/api/photos/{photo_id}/similar"}
        found = set()
        for route in app_fastapi.app.routes:
            if getattr(route, "path", None) not in paths:
                continue
            found.add(route.path)
            names = [d.call.__name__ for d in route.dependant.dependencies]
            self.assertIn(
                "get_current_active_user", names,
                f"{route.path} has no auth dependency",
            )
        self.assertEqual(paths, found, "a discovery route is not registered")


if __name__ == "__main__":
    unittest.main()
