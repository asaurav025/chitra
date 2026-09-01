"""
`embeddings` and `tags` must survive being written twice.

All four writers — `db.put_embedding`, `db_async.put_embedding_async`,
`db.add_tag`, `db_async.add_tag_async` — were plain INSERTs over tables with no
unique constraint. The production DB is clean only because nothing has ever
re-indexed. Everything now planned *is* a re-index, and a duplicated vector is
not a harmless extra row: `search_photos` stacks every row it gets back, so one
photo would occupy two result slots and score twice.

The key is `(photo_id, model)`, deliberately **not** `(photo_id)`. The CLIP
512-d -> SigLIP 768-d migration needs both generations resident in the table at
once: search keeps answering from CLIP while SigLIP rows are still being
written, and rollback is then a config flip rather than a full re-embed. A bare
`(photo_id)` key makes the first SigLIP row evict the CLIP row it is supposed to
run alongside — on a disk where an interrupted run is the expected case, that
turns a safe migration into an unrecoverable one.

`tags` is unique on `(photo_id, tag)`. `tags.source` records which model and
vocabulary produced a label — provenance, not identity — so it stays out of the
key.
"""
import asyncio
import os
import sqlite3
import tempfile
import unittest

from core import db, db_async


def _tmp_db():
    fd, path = tempfile.mkstemp(suffix=".db", prefix="chitra-uniq-")
    os.close(fd)
    os.unlink(path)
    return path


def _old_schema(path):
    """The schema as it shipped: no `model`, no `source`, no unique indexes."""
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE photos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT UNIQUE,
            size INTEGER,
            created_at TEXT,
            checksum TEXT,
            phash TEXT,
            exif_datetime TEXT,
            latitude REAL,
            longitude REAL,
            thumb_path TEXT
        );
        CREATE TABLE embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            photo_id INTEGER NOT NULL,
            dim INTEGER NOT NULL,
            vector BLOB NOT NULL
        );
        CREATE TABLE tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            photo_id INTEGER NOT NULL,
            tag TEXT NOT NULL,
            score REAL NOT NULL
        );
        INSERT INTO photos (id, file_path) VALUES (1, 'photos/a.heic');
        INSERT INTO embeddings (photo_id, dim, vector) VALUES (1, 512, X'00');
        INSERT INTO tags (photo_id, tag, score) VALUES (1, 'travel', 0.22);
        """
    )
    conn.commit()
    conn.close()
    return path


def _indexes(conn, table):
    return {
        row[1]: bool(row[2])  # name -> unique?
        for row in conn.execute(f"PRAGMA index_list({table})")
    }


def _index_columns(conn, name):
    return [row[2] for row in conn.execute(f"PRAGMA index_info({name})")]


class TestMigration(unittest.TestCase):
    """An old-schema database must come forward without losing a row."""

    def setUp(self):
        self.path = _old_schema(_tmp_db())

    def tearDown(self):
        for suffix in ("", "-wal", "-shm"):
            if os.path.exists(self.path + suffix):
                os.unlink(self.path + suffix)

    def test_embeddings_gains_a_backfilled_model_column(self):
        db.init_db(self.path)
        conn = db.connect(self.path)
        try:
            row = conn.execute("SELECT model FROM embeddings WHERE photo_id=1").fetchone()
            self.assertEqual(row["model"], db.DEFAULT_EMBED_MODEL)
        finally:
            conn.close()

    def test_tags_gains_a_backfilled_source_column(self):
        db.init_db(self.path)
        conn = db.connect(self.path)
        try:
            row = conn.execute("SELECT source FROM tags WHERE photo_id=1").fetchone()
            self.assertEqual(row["source"], db.DEFAULT_TAG_SOURCE)
        finally:
            conn.close()

    def test_the_embeddings_key_is_photo_id_and_model(self):
        """A bare (photo_id) key would evict CLIP the moment SigLIP writes."""
        db.init_db(self.path)
        conn = db.connect(self.path)
        try:
            uniques = {n for n, is_unique in _indexes(conn, "embeddings").items() if is_unique}
            self.assertTrue(uniques, "embeddings has no unique index at all")
            cols = {tuple(_index_columns(conn, n)) for n in uniques}
            self.assertIn(("photo_id", "model"), cols)
            self.assertNotIn(("photo_id",), cols)
        finally:
            conn.close()

    def test_the_tags_key_is_photo_id_and_tag(self):
        db.init_db(self.path)
        conn = db.connect(self.path)
        try:
            uniques = {n for n, is_unique in _indexes(conn, "tags").items() if is_unique}
            cols = {tuple(_index_columns(conn, n)) for n in uniques}
            self.assertIn(("photo_id", "tag"), cols)
            self.assertNotIn(("photo_id", "tag", "source"), cols)
        finally:
            conn.close()

    def test_running_it_twice_is_a_no_op(self):
        db.init_db(self.path)
        db.init_db(self.path)
        conn = db.connect(self.path)
        try:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0], 1)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM tags").fetchone()[0], 1)
        finally:
            conn.close()

    def test_the_async_schema_migrates_identically(self):
        """`db.py` and `db_async.py` carry duplicated DDL that has already
        diverged once. Both copies must grow this migration."""
        asyncio.run(db_async.init_db_async(self.path))
        conn = sqlite3.connect(self.path)
        try:
            model = conn.execute("SELECT model FROM embeddings WHERE photo_id=1").fetchone()[0]
            source = conn.execute("SELECT source FROM tags WHERE photo_id=1").fetchone()[0]
            self.assertEqual(model, db.DEFAULT_EMBED_MODEL)
            self.assertEqual(source, db.DEFAULT_TAG_SOURCE)
            uniques = {n for n, is_unique in _indexes(conn, "embeddings").items() if is_unique}
            cols = {tuple(_index_columns(conn, n)) for n in uniques}
            self.assertIn(("photo_id", "model"), cols)
        finally:
            conn.close()


class TestSyncWriters(unittest.TestCase):
    def setUp(self):
        self.path = _old_schema(_tmp_db())
        db.init_db(self.path)
        self.conn = db.connect(self.path)
        self.conn.execute("DELETE FROM embeddings")
        self.conn.execute("DELETE FROM tags")
        self.conn.commit()

    def tearDown(self):
        self.conn.close()
        for suffix in ("", "-wal", "-shm"):
            if os.path.exists(self.path + suffix):
                os.unlink(self.path + suffix)

    def _rows(self, table="embeddings"):
        return self.conn.execute(f"SELECT * FROM {table} ORDER BY id").fetchall()

    def test_writing_the_same_photo_and_model_twice_leaves_one_row(self):
        db.put_embedding(self.conn, 1, b"\x01\x02", 2)
        db.put_embedding(self.conn, 1, b"\x03\x04", 2)
        rows = self._rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["vector"], b"\x03\x04")

    def test_two_models_coexist_on_one_photo(self):
        """The whole point of the key: CLIP keeps answering search while
        SigLIP rows are still being written, and rollback is a config flip."""
        db.put_embedding(self.conn, 1, b"\x01" * 8, 2, model="openai/clip-vit-base-patch32")
        db.put_embedding(self.conn, 1, b"\x02" * 12, 3, model="google/siglip2-base")
        rows = self._rows()
        self.assertEqual(len(rows), 2)
        self.assertEqual({r["model"] for r in rows},
                         {"openai/clip-vit-base-patch32", "google/siglip2-base"})
        self.assertEqual({r["dim"] for r in rows}, {2, 3})

    def test_the_model_defaults_to_the_current_clip_identifier(self):
        db.put_embedding(self.conn, 1, b"\x01\x02", 2)
        self.assertEqual(self._rows()[0]["model"], db.DEFAULT_EMBED_MODEL)

    def test_the_unique_index_is_enforced_by_the_database(self):
        """Not just by the writer: a raw INSERT must be refused too."""
        db.put_embedding(self.conn, 1, b"\x01\x02", 2)
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute(
                "INSERT INTO embeddings (photo_id, dim, vector, model) VALUES (?,?,?,?)",
                (1, 2, b"\x09\x09", db.DEFAULT_EMBED_MODEL),
            )

    def test_add_tag_is_idempotent_and_updates_the_score(self):
        db.add_tag(self.conn, 1, "travel", 0.20)
        db.add_tag(self.conn, 1, "travel", 0.31)
        rows = self._rows("tags")
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["score"], 0.31)

    def test_different_tags_on_one_photo_all_survive(self):
        db.add_tag(self.conn, 1, "travel", 0.2)
        db.add_tag(self.conn, 1, "sunset", 0.3)
        self.assertEqual(len(self._rows("tags")), 2)

    def test_replace_tags_swaps_the_whole_set(self):
        """`add_tag` in a loop cannot drop a label the new vocabulary no longer
        predicts, so a re-tagged photo would accumulate the union of every list
        it was ever scored against."""
        db.replace_tags(self.conn, 1, [("travel", 0.2), ("sunset", 0.3)])
        db.replace_tags(self.conn, 1, [("food", 0.4)])
        rows = self._rows("tags")
        self.assertEqual([r["tag"] for r in rows], ["food"])

    def test_get_embeddings_still_returns_photo_dim_vector(self):
        """The search handler unpacks exactly three values."""
        db.put_embedding(self.conn, 1, b"\x01\x02", 2)
        photo_id, dim, vector = db.get_embeddings(self.conn)[0]
        self.assertEqual((photo_id, dim, vector), (1, 2, b"\x01\x02"))


class TestAsyncWriters(unittest.TestCase):
    def setUp(self):
        self.path = _old_schema(_tmp_db())
        asyncio.run(db_async.init_db_async(self.path))
        conn = sqlite3.connect(self.path)
        conn.executescript("DELETE FROM embeddings; DELETE FROM tags;")
        conn.commit()
        conn.close()

    def tearDown(self):
        for suffix in ("", "-wal", "-shm"):
            if os.path.exists(self.path + suffix):
                os.unlink(self.path + suffix)

    def _count(self, table):
        conn = sqlite3.connect(self.path)
        try:
            return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        finally:
            conn.close()

    def test_put_embedding_async_replaces_rather_than_appends(self):
        async def go():
            async with db_async.connect_async(self.path) as conn:
                await db_async.put_embedding_async(conn, 1, b"\x01\x02", 2)
                await db_async.put_embedding_async(conn, 1, b"\x03\x04", 2)
        asyncio.run(go())
        self.assertEqual(self._count("embeddings"), 1)

    def test_put_embedding_async_keeps_two_models_apart(self):
        async def go():
            async with db_async.connect_async(self.path) as conn:
                await db_async.put_embedding_async(conn, 1, b"\x01\x02", 2, model="a")
                await db_async.put_embedding_async(conn, 1, b"\x03\x04", 2, model="b")
        asyncio.run(go())
        self.assertEqual(self._count("embeddings"), 2)

    def test_add_tag_async_is_idempotent(self):
        async def go():
            async with db_async.connect_async(self.path) as conn:
                await db_async.add_tag_async(conn, 1, "travel", 0.2)
                await db_async.add_tag_async(conn, 1, "travel", 0.4)
        asyncio.run(go())
        self.assertEqual(self._count("tags"), 1)


if __name__ == "__main__":
    unittest.main()
