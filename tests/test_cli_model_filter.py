"""The sync reader's model filter, and the two CLI paths that stack vectors.

Why this file exists
--------------------
`core.db_async.get_embeddings_async` already filters on `model` in SQL, because
`/api/search/photos` stacks whatever it returns and `np.stack` over a mixed list
raises::

    ValueError: all input arrays must have the same shape

The **sync** reader had no such parameter, and two CLI commands read it
unfiltered:

* `cli/main.py` `cluster` -> `core.cluster.threshold_clusters` -> `np.stack`
* `cli/main.py` `search`  -> `np.stack(vectors)`

Every stored row is 512-d CLIP today, so both are harmless. The moment the
SigLIP re-embed writes a single 768-d row they hit the identical failure that
was already fixed on the API side — a `ValueError` from the first CLI command
an operator runs mid-migration, which is exactly when the CLI is most likely to
be reached for.

The migration needs the two generations resident at once (`embeddings` is
unique on `(photo_id, model)` precisely so a SigLIP row lands *alongside* the
CLIP row search still answers from), so "just don't have mixed rows" is not
available. The filter has to go in the SQL, so the mixed list never exists.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from typer.testing import CliRunner

from core import db, db_async

CLIP_MODEL = "openai/clip-vit-base-patch32"
SIGLIP_MODEL = "google/siglip2-base-patch16-224"
CLIP_DIM = 512
SIGLIP_DIM = 768

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def unit(v: np.ndarray) -> np.ndarray:
    return (v / (np.linalg.norm(v) + 1e-9)).astype("float32")


def basis(dim: int, i: int) -> np.ndarray:
    v = np.zeros(dim, dtype="float32")
    v[i] = 1.0
    return unit(v)


class TestGetEmbeddingsModelFilter(unittest.TestCase):
    """The sync twin of `TestGetEmbeddingsAsyncModelFilter`."""

    def setUp(self):
        self.db_path = tempfile.mktemp(suffix=".db")
        self.addCleanup(
            lambda: os.path.exists(self.db_path) and os.unlink(self.db_path)
        )
        db.init_db(self.db_path)
        self.conn = db.connect(self.db_path)
        self.addCleanup(self.conn.close)
        for name, model, dim in (
            ("a.jpg", "model-a", 4),
            ("b.jpg", "model-b", 6),
        ):
            db.upsert_photo(
                self.conn, file_path=name, size=1,
                created_at="2026-01-01T00:00:00", checksum=name, phash=None,
                exif_datetime=None, latitude=None, longitude=None,
                media_type="photo",
            )
            cur = self.conn.execute("SELECT id FROM photos WHERE file_path=?", (name,))
            pid = cur.fetchone()[0]
            db.put_embedding(
                self.conn, pid, np.zeros(dim, dtype="float32").tobytes(), dim,
                model=model,
            )
        self.conn.commit()

    def test_filters_rows_to_one_model(self):
        only_a = db.get_embeddings(self.conn, model="model-a")
        self.assertEqual([4], [dim for _, dim, _ in only_a])

    def test_model_none_stays_an_unfiltered_read(self):
        """Migration and coverage-counting callers genuinely want every row."""
        everything = db.get_embeddings(self.conn)
        self.assertEqual({4, 6}, {dim for _, dim, _ in everything})
        self.assertEqual(everything, db.get_embeddings(self.conn, model=None))

    def test_an_unknown_model_returns_nothing_rather_than_everything(self):
        self.assertEqual([], db.get_embeddings(self.conn, model="model-c"))


class TestActiveEmbedModelHasOneDefinition(unittest.TestCase):
    """`active_embed_model` lives in `core.db`; `core.db_async` re-exports it.

    Same precedent as `DEFAULT_EMBED_MODEL` and the `(photo_id, model)`
    migration: the sync module owns the definition and the async module imports
    it. A second copy would be a second default model name, and two spellings of
    the CLIP identifier is the failure mode the guard test in
    `test_embed_model_selection.py` exists to catch — rows written under one
    name, search filtering on another, every query silently empty.
    """

    def test_the_async_side_reuses_the_sync_function(self):
        self.assertIs(db.active_embed_model, db_async.active_embed_model)

    def test_defaults_to_the_shared_clip_identifier(self):
        env = {k: v for k, v in os.environ.items()
               if k != "CHITRA_ACTIVE_EMBED_MODEL"}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(db.DEFAULT_EMBED_MODEL, db.active_embed_model())
        self.assertEqual(CLIP_MODEL, db.DEFAULT_EMBED_MODEL)

    def test_reads_the_environment_at_call_time_not_import_time(self):
        """The cutover is a config flip and the rollback is flipping it back.

        Asserted in a fresh subprocess that imports the module with the
        variable *unset* and only then sets it: a module-level constant would
        keep answering CLIP and the rollback would need a code change.
        """
        prog = (
            "import os, sys; sys.path.insert(0, %r)\n"
            "os.environ.pop('CHITRA_ACTIVE_EMBED_MODEL', None)\n"
            "from core import db\n"
            "before = db.active_embed_model()\n"
            "os.environ['CHITRA_ACTIVE_EMBED_MODEL'] = %r\n"
            "print(before, db.active_embed_model())\n" % (REPO_ROOT, SIGLIP_MODEL)
        )
        out = subprocess.run(
            [sys.executable, "-c", prog], capture_output=True, text=True, check=True,
        ).stdout.split()
        self.assertEqual([db.DEFAULT_EMBED_MODEL, SIGLIP_MODEL], out)


class _MixedDimensionLibrary(unittest.TestCase):
    """A partially re-embedded library: two CLIP rows and one SigLIP row."""

    def setUp(self):
        self.db_path = tempfile.mktemp(suffix=".db")
        self.addCleanup(
            lambda: os.path.exists(self.db_path) and os.unlink(self.db_path)
        )
        db.init_db(self.db_path)
        conn = db.connect(self.db_path)
        self.ids = {}
        rows = (
            ("clip_near.jpg", CLIP_MODEL, CLIP_DIM, basis(CLIP_DIM, 0)),
            ("clip_far.jpg", CLIP_MODEL, CLIP_DIM,
             unit(0.6 * basis(CLIP_DIM, 0) + 0.8 * basis(CLIP_DIM, 1))),
            ("siglip_only.jpg", SIGLIP_MODEL, SIGLIP_DIM, basis(SIGLIP_DIM, 0)),
        )
        for name, model, dim, vec in rows:
            db.upsert_photo(
                conn, file_path=name, size=1,
                created_at="2026-01-01T00:00:00", checksum=name, phash=None,
                exif_datetime=None, latitude=None, longitude=None,
                media_type="photo",
            )
            cur = conn.execute("SELECT id FROM photos WHERE file_path=?", (name,))
            self.ids[name] = cur.fetchone()[0]
            db.put_embedding(conn, self.ids[name], vec.tobytes(), dim, model=model)
        conn.commit()
        conn.close()
        self.runner = CliRunner()

    def clustered(self):
        conn = db.connect(self.db_path)
        try:
            rows = conn.execute(
                "SELECT photo_id FROM clusters ORDER BY photo_id"
            ).fetchall()
        finally:
            conn.close()
        return [r["photo_id"] for r in rows]


class TestClusterCommandWithMixedDimensions(_MixedDimensionLibrary):
    """`cluster` fed both generations raised ValueError out of np.stack."""

    def invoke(self):
        import cli.main as cli

        return self.runner.invoke(
            cli.app, ["cluster", "--db", self.db_path, "--threshold", "0.78"],
            catch_exceptions=False,
        )

    def test_clusters_only_the_active_model(self):
        with patch.dict(os.environ, {"CHITRA_ACTIVE_EMBED_MODEL": CLIP_MODEL}):
            result = self.invoke()
        self.assertEqual(0, result.exit_code, result.output)
        self.assertEqual(
            sorted([self.ids["clip_near.jpg"], self.ids["clip_far.jpg"]]),
            self.clustered(),
            "the 768-d row must not be clustered while CLIP is the active model",
        )

    def test_follows_the_cutover_to_the_other_model(self):
        with patch.dict(os.environ, {"CHITRA_ACTIVE_EMBED_MODEL": SIGLIP_MODEL}):
            result = self.invoke()
        self.assertEqual(0, result.exit_code, result.output)
        self.assertEqual([self.ids["siglip_only.jpg"]], self.clustered())


class TestSearchCommandWithMixedDimensions(_MixedDimensionLibrary):
    """`search` fed both generations raised ValueError out of np.stack."""

    def invoke(self, query_dim):
        import cli.main as cli

        class StubEmbedder:
            def text_embedding(self, text):
                return basis(query_dim, 0)

        with patch.object(cli, "ClipEmbedder", StubEmbedder):
            return self.runner.invoke(
                cli.app,
                ["search", "a beach", "--db", self.db_path, "--top-k", "5"],
                catch_exceptions=False,
            )

    def test_ranks_only_the_active_model(self):
        with patch.dict(os.environ, {"CHITRA_ACTIVE_EMBED_MODEL": CLIP_MODEL}):
            result = self.invoke(CLIP_DIM)
        self.assertEqual(0, result.exit_code, result.output)
        self.assertIn("clip_near.jpg", result.output)
        self.assertIn("clip_far.jpg", result.output)
        self.assertNotIn(
            "siglip_only.jpg", result.output,
            "a 768-d row must not be ranked against a 512-d query",
        )

    def test_follows_the_cutover_to_the_other_model(self):
        with patch.dict(os.environ, {"CHITRA_ACTIVE_EMBED_MODEL": SIGLIP_MODEL}):
            result = self.invoke(SIGLIP_DIM)
        self.assertEqual(0, result.exit_code, result.output)
        self.assertIn("siglip_only.jpg", result.output)
        self.assertNotIn("clip_near.jpg", result.output)


if __name__ == "__main__":
    unittest.main()
