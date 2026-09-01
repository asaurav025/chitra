"""Re-tagging the whole library from vectors that are already in SQLite.

The load-bearing claim of this phase is that **re-tagging reads no media**. A
tag score is `cosine(image_vec, text_vec)`, and the blobs in `embeddings.vector`
already *are* the L2-normalised CLIP image vectors (measured: norm 1.000000 on
every row, all dim=512). So the whole pass is `N x 512 @ 512 x M` over 3.7 MiB
of SQLite — and `/dev/sda`, which holds every original, thumbnail and poster and
has 3,000+ unrecovered read errors, is never touched.

That is not a comment, it is an assertion: `TestRetagReadsNoStorage` runs the
real pass with a `StorageClient` whose every attribute raises.
"""
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import db, vocabulary  # noqa: E402
from scripts import retag  # noqa: E402


MODEL = "openai/clip-vit-base-patch32"


def _unit(rng, n, d=32):
    v = rng.normal(0, 1, (n, d)).astype("float32")
    return v / np.linalg.norm(v, axis=1, keepdims=True)


class _StubTextEmbedder:
    """Deterministic stand-in for the sidecar's POST /embed/text.

    Records every call so a test can prove the cache saved the round trips.
    """

    def __init__(self, dim=32, name=MODEL):
        self.dim = dim
        self.name = name
        self.calls = []

    def embed_texts(self, prompts):
        self.calls.append(list(prompts))
        rng = np.random.default_rng(abs(hash(tuple(prompts))) % (2 ** 32))
        return _unit(rng, len(prompts), self.dim)


class _ExplodingStorage:
    """Any attribute access at all is a failed test."""

    def __init__(self, *a, **k):
        raise AssertionError("retag constructed a MinIOStorageClient")

    def __getattr__(self, name):
        raise AssertionError(f"retag touched storage: {name}")


def _make_db(n_photos=40, dim=32, seed=3, legacy_tags=True):
    fd, path = tempfile.mkstemp(suffix=".db", prefix="chitra-retag-")
    os.close(fd)
    os.unlink(path)
    db.init_db(path)
    conn = db.connect(path)
    rng = np.random.default_rng(seed)
    vecs = _unit(rng, n_photos, dim)
    for i in range(n_photos):
        conn.execute(
            "INSERT INTO photos (file_path, size, created_at, thumb_path) VALUES (?,?,?,?)",
            (f"orig/{i}.jpg", 100, "2026-01-01T00:00:00", f"thumbs/{i}.jpg"),
        )
        pid = conn.execute("SELECT last_insert_rowid() AS i").fetchone()["i"]
        db.put_embedding(conn, pid, vecs[i].tobytes(), dim, model=MODEL)
        if legacy_tags:
            for lab in ("travel", "outdoors", "portrait", "food", "city", "night"):
                db.add_tag(conn, pid, lab, 0.20, source=db.DEFAULT_TAG_SOURCE)
    conn.commit()
    return path, conn


class _DbCase(unittest.TestCase):
    def setUp(self):
        self.path, self.conn = _make_db()
        self.cache = tempfile.mkdtemp(prefix="chitra-retag-cache-")
        self.embedder = _StubTextEmbedder()

    def tearDown(self):
        try:
            self.conn.close()
        except Exception:
            pass
        for p in (self.path,):
            if os.path.exists(p):
                os.unlink(p)

    def _tags(self):
        return [tuple(r) for r in self.conn.execute(
            "SELECT photo_id, tag, source FROM tags ORDER BY photo_id, tag")]

    def _run(self, **kw):
        kw.setdefault("embedder", self.embedder)
        kw.setdefault("model", MODEL)
        kw.setdefault("cache_dir", self.cache)
        return retag.retag(self.conn, **kw)


class TestDryRunWritesNothing(_DbCase):
    def test_dry_run_leaves_the_tags_table_alone(self):
        before = self._tags()
        result = self._run(apply=False)
        self.assertFalse(result.applied)
        self.assertEqual(self._tags(), before)

    def test_dry_run_still_reports_a_distribution(self):
        result = self._run(apply=False)
        self.assertGreater(result.photos, 0)
        self.assertGreater(sum(result.distribution.values()), 0)
        self.assertEqual(sum(result.distribution.values()),
                         sum(result.tag_counts))


class TestApplyWrites(_DbCase):
    def test_apply_stamps_the_new_source(self):
        result = self._run(apply=True)
        self.assertTrue(result.applied)
        sources = {r[2] for r in self._tags()}
        self.assertEqual(sources, {vocabulary.tag_source(MODEL)})

    def test_apply_drops_labels_the_new_run_does_not_predict(self):
        """`add_tag` in a loop cannot orphan a stale label — but it must.

        Every photo currently carries `travel`. If re-tagging only upserts,
        `travel` stays on 100% of the library forever and the whole phase is
        cosmetic.
        """
        self._run(apply=True)
        n = self.conn.execute("SELECT COUNT(*) c FROM tags WHERE tag='travel'").fetchone()["c"]
        total = self.conn.execute("SELECT COUNT(*) c FROM photos").fetchone()["c"]
        self.assertLess(n, total)

    def test_apply_is_idempotent(self):
        self._run(apply=True)
        first = self._tags()
        self._run(apply=True)
        self.assertEqual(self._tags(), first)

    def test_apply_leaves_foreign_sources_alone(self):
        """A hand-added tag is not this script's to delete."""
        db.add_tag(self.conn, 1, "grandma's house", 1.0, source="manual")
        self._run(apply=True)
        rows = self.conn.execute(
            "SELECT tag FROM tags WHERE photo_id=1 AND source='manual'").fetchall()
        self.assertEqual([r["tag"] for r in rows], ["grandma's house"])

    def test_tag_counts_vary_across_photos(self):
        result = self._run(apply=True)
        self.assertGreater(len(set(result.tag_counts)), 1,
                           f"every photo got {result.tag_counts[0]} tags")

    def test_no_tag_lands_on_the_whole_library(self):
        result = self._run(apply=True)
        worst = max(result.distribution.values()) / result.photos
        self.assertLess(worst, 0.60, f"worst label covers {worst:.1%}")

    def test_limit_bounds_the_pass(self):
        result = self._run(apply=False, limit=5)
        self.assertEqual(result.photos, 5)


class TestModelScoping(_DbCase):
    def test_only_the_active_model_is_read(self):
        """A 768-d SigLIP row must not be stacked next to a 512-d CLIP one.

        `np.stack` raises `ValueError: all input arrays must have the same
        shape` — the same failure that takes `/api/search/photos` down.
        """
        other = np.ones(64, dtype="float32") / 8.0
        db.put_embedding(self.conn, 1, other.tobytes(), 64, model="google/siglip2-base")
        result = self._run(apply=False)
        self.assertEqual(result.dim, 32)
        self.assertEqual(result.photos, 40)

    def test_unknown_model_yields_nothing_rather_than_crashing(self):
        result = self._run(apply=False, model="nope/not-a-model")
        self.assertEqual(result.photos, 0)


class TestLabelMatrixCache(_DbCase):
    def _cache_files(self):
        return sorted(p.name for p in Path(self.cache).iterdir())

    def test_matrix_is_cached_under_model_and_fingerprint(self):
        self._run(apply=False)
        names = self._cache_files()
        stem = f"tag_vectors_clip-vitb32_{vocabulary.vocab_fingerprint()}"
        self.assertIn(stem + ".npy", names)
        self.assertIn(stem + ".json", names)

    def test_cache_is_reused_on_a_second_run(self):
        self._run(apply=False)
        n = len(self.embedder.calls)
        self.assertGreater(n, 0)
        self._run(apply=False)
        self.assertEqual(len(self.embedder.calls), n, "cache was not reused")

    def test_cache_refuses_a_fingerprint_mismatch(self):
        path = retag.label_matrix_path(self.cache, MODEL, "deadbeef")
        np.save(path, np.eye(4, dtype="float32"))
        path.with_suffix(".json").write_text(json.dumps({
            "model": MODEL, "fingerprint": "cafef00d",
            "labels": ["a", "b", "c", "d"],
            "template": vocabulary.PROMPT_TEMPLATE,
            "version": vocabulary.VOCAB_VERSION,
        }))
        with self.assertRaises(retag.CacheMismatch):
            retag.load_cached_matrix(path, model=MODEL, fingerprint="deadbeef",
                                     labels=("a", "b", "c", "d"))

    def test_cache_refuses_a_model_mismatch(self):
        """A SigLIP matrix under a CLIP run silently produces garbage tags."""
        fp = vocabulary.vocab_fingerprint()
        path = retag.label_matrix_path(self.cache, MODEL, fp)
        np.save(path, np.eye(4, dtype="float32"))
        path.with_suffix(".json").write_text(json.dumps({
            "model": "google/siglip2-base", "fingerprint": fp,
            "labels": ["a", "b", "c", "d"],
            "template": vocabulary.PROMPT_TEMPLATE,
            "version": vocabulary.VOCAB_VERSION,
        }))
        with self.assertRaises(retag.CacheMismatch):
            retag.load_cached_matrix(path, model=MODEL, fingerprint=fp,
                                     labels=("a", "b", "c", "d"))

    def test_cache_refuses_a_row_count_mismatch(self):
        fp = vocabulary.vocab_fingerprint()
        path = retag.label_matrix_path(self.cache, MODEL, fp)
        np.save(path, np.eye(4, dtype="float32"))
        path.with_suffix(".json").write_text(json.dumps({
            "model": MODEL, "fingerprint": fp,
            "labels": list(vocabulary.LABELS),
            "template": vocabulary.PROMPT_TEMPLATE,
            "version": vocabulary.VOCAB_VERSION,
        }))
        with self.assertRaises(retag.CacheMismatch):
            retag.load_cached_matrix(path, model=MODEL, fingerprint=fp,
                                     labels=vocabulary.LABELS)

    def test_missing_cache_returns_none(self):
        path = retag.label_matrix_path(self.cache, MODEL, "nothing-here")
        self.assertIsNone(retag.load_cached_matrix(
            path, model=MODEL, fingerprint="nothing-here", labels=("a",)))

    def test_cached_matrix_rows_are_normalised(self):
        self._run(apply=False)
        stem = f"tag_vectors_clip-vitb32_{vocabulary.vocab_fingerprint()}.npy"
        m = np.load(Path(self.cache) / stem)
        norms = np.linalg.norm(m, axis=1)
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-5))


class TestRetagReadsNoStorage(_DbCase):
    """The load-bearing assertion of this phase."""

    def test_full_apply_pass_never_touches_storage(self):
        import core.storage_client as sc

        real = sc.MinIOStorageClient
        sc.MinIOStorageClient = _ExplodingStorage
        try:
            result = self._run(apply=True)
        finally:
            sc.MinIOStorageClient = real
        self.assertTrue(result.applied)
        self.assertGreater(result.tags_written, 0)

    def test_no_file_is_opened_outside_the_cache(self):
        """SQLite does not use builtins.open; anything that does is media."""
        import builtins

        real_open = builtins.open
        allowed = str(Path(self.cache).resolve())
        seen = []

        def guarded(file, *a, **k):
            p = str(file)
            if not p.startswith(allowed):
                seen.append(p)
            return real_open(file, *a, **k)

        builtins.open = guarded
        try:
            self._run(apply=True)
        finally:
            builtins.open = real_open
        offenders = [p for p in seen if "/thumb" in p or "/orig" in p or "minio" in p]
        self.assertEqual(offenders, [], f"opened media: {offenders}")

    def test_importing_retag_pulls_in_no_storage_and_no_torch(self):
        repo = Path(__file__).resolve().parent.parent
        code = (
            "import sys; import scripts.retag as r;"
            "bad=[m for m in ('torch','transformers','minio','core.storage_client',"
            "'core.embedder') if m in sys.modules];"
            "assert r.retag is not None;"
            "print('LEAKED:'+','.join(bad) if bad else 'CLEAN')"
        )
        out = subprocess.run([sys.executable, "-c", code], cwd=str(repo),
                             capture_output=True, text=True, timeout=180)
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        self.assertIn("CLEAN", out.stdout, out.stdout + out.stderr[-2000:])


class TestCli(_DbCase):
    def test_dry_run_is_the_default(self):
        parser = retag.build_parser()
        args = parser.parse_args([])
        self.assertFalse(args.apply)

    def test_apply_flag_parses(self):
        args = retag.build_parser().parse_args(["--apply"])
        self.assertTrue(args.apply)

    def test_main_dry_run_writes_nothing(self):
        before = self._tags()
        rc = retag.main(["--db", self.path, "--cache-dir", self.cache],
                        embedder=self.embedder)
        self.assertEqual(rc, 0)
        conn = db.connect(self.path)
        after = [tuple(r) for r in conn.execute(
            "SELECT photo_id, tag, source FROM tags ORDER BY photo_id, tag")]
        conn.close()
        self.assertEqual(after, before)


class TestVectorLoading(unittest.TestCase):
    def test_rows_with_a_wrong_length_blob_are_rejected_loudly(self):
        """A truncated blob would silently reshape the whole matrix."""
        path, conn = _make_db(n_photos=3, dim=32)
        try:
            conn.execute("UPDATE embeddings SET vector = ? WHERE photo_id = 1",
                         (b"\x00" * 8,))
            conn.commit()
            with self.assertRaises(ValueError):
                retag.load_vectors(conn, MODEL)
        finally:
            conn.close()
            os.unlink(path)

    def test_vectors_come_back_normalised(self):
        path, conn = _make_db(n_photos=5, dim=32)
        try:
            ids, mat = retag.load_vectors(conn, MODEL)
            self.assertEqual(len(ids), 5)
            self.assertTrue(np.allclose(np.linalg.norm(mat, axis=1), 1.0, atol=1e-5))
        finally:
            conn.close()
            os.unlink(path)

    def test_videos_are_excluded_by_default(self):
        path, conn = _make_db(n_photos=5, dim=32)
        try:
            conn.execute("UPDATE photos SET media_type='video' WHERE id=1")
            conn.commit()
            ids, _ = retag.load_vectors(conn, MODEL)
            self.assertNotIn(1, ids)
        finally:
            conn.close()
            os.unlink(path)

    def test_videos_can_be_included_explicitly(self):
        """Poster embedding landed in 2b1b052, so videos will have vectors.

        Skipping them silently would leave every video embedded and untagged
        with nothing saying why. The exclusion stays the default because Phase 7
        owns that cutover, but it is one flag, not a buried WHERE clause.
        """
        path, conn = _make_db(n_photos=5, dim=32)
        try:
            conn.execute("UPDATE photos SET media_type='video' WHERE id=1")
            conn.commit()
            ids, _ = retag.load_vectors(conn, MODEL, include_videos=True)
            self.assertIn(1, ids)
        finally:
            conn.close()
            os.unlink(path)

    def test_include_videos_reaches_the_cli(self):
        self.assertFalse(retag.build_parser().parse_args([]).include_videos)
        self.assertTrue(retag.build_parser().parse_args(["--include-videos"]).include_videos)


if __name__ == "__main__":
    unittest.main()


class TestSidecarTextEmbedderRetries(unittest.TestCase):
    """345 sequential requests to a single-worker sidecar will hit a reset.

    Observed on the real box: the label-matrix build died 10 minutes in with
    `httpx.ReadError: [Errno 104] Connection reset by peer` while other agents
    were exercising the same sidecar. Losing 300 completed embeds to one dropped
    connection is not acceptable when the whole point of the cache is that this
    happens once.
    """

    def _embedder(self, handler):
        import httpx
        transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=transport, base_url="http://sidecar")
        return retag.SidecarTextEmbedder(base_url="http://sidecar", client=client,
                                         retries=3, backoff=0.0)

    @staticmethod
    def _ok_vector():
        import base64
        vec = np.ones(4, dtype="float32")
        return {"dim": 4, "dtype": "float32",
                "vector_b64": base64.b64encode(vec.tobytes()).decode()}

    def test_a_dropped_connection_is_retried(self):
        import httpx

        state = {"calls": 0}

        def handler(request):
            if request.url.path == "/health":
                return httpx.Response(200, json={"status": "ok", "model": MODEL})
            state["calls"] += 1
            if state["calls"] == 1:
                raise httpx.ReadError("Connection reset by peer")
            return httpx.Response(200, json=self._ok_vector())

        out = self._embedder(handler).embed_texts(["one"])
        self.assertEqual(out.shape, (1, 4))
        self.assertEqual(state["calls"], 2)

    def test_retries_are_bounded_and_the_failure_surfaces(self):
        import httpx

        def handler(request):
            if request.url.path == "/health":
                return httpx.Response(200, json={"status": "ok", "model": MODEL})
            raise httpx.ReadError("Connection reset by peer")

        with self.assertRaises(httpx.ReadError):
            self._embedder(handler).embed_texts(["one"])

    def test_an_http_error_is_not_silently_swallowed(self):
        import httpx

        def handler(request):
            if request.url.path == "/health":
                return httpx.Response(200, json={"status": "ok", "model": MODEL})
            return httpx.Response(422, json={"detail": "empty_text"})

        with self.assertRaises(httpx.HTTPStatusError):
            self._embedder(handler).embed_texts(["one"])

    def test_vectors_come_back_normalised(self):
        import httpx

        def handler(request):
            if request.url.path == "/health":
                return httpx.Response(200, json={"status": "ok", "model": MODEL})
            return httpx.Response(200, json=self._ok_vector())

        out = self._embedder(handler).embed_texts(["one", "two"])
        self.assertTrue(np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5))
