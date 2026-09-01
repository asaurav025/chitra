"""
Deleting a photo must delete everything that hangs off it.

The schema declares `ON DELETE CASCADE` on embeddings, tags, clusters, faces
and (via faces) face_thumbs. SQLite enforces foreign keys only when
`PRAGMA foreign_keys=ON` is issued, and that pragma is per-connection and
defaults to OFF. If the connection helpers do not set it, every CASCADE in the
schema is decorative: deleting a photo leaves its rows behind forever, and the
orphans accumulate silently because nothing ever reads them back.

These tests pin the pragma on both connection helpers — sync (`core.db.connect`,
used by the CLI and RQ jobs) and async (`core.db_async.connect_async`, used by
the API) — and then assert the cascade actually fires.
"""
import asyncio
import os
import tempfile
import unittest

from core import db as sync_db
from core import db_async as async_db


class _TempDBMixin:
    """Each test gets its own throwaway database file."""

    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(prefix="chitra_cascade_", suffix=".db")
        os.close(fd)
        os.unlink(self.db_path)  # let the driver create it

    def tearDown(self):
        for suffix in ("", "-wal", "-shm"):
            path = self.db_path + suffix
            if os.path.exists(path):
                os.unlink(path)


class TestSyncCascadeDelete(_TempDBMixin, unittest.TestCase):
    """core.db.connect must enforce foreign keys."""

    def setUp(self):
        super().setUp()
        sync_db.init_db(self.db_path)

    def test_foreign_keys_pragma_is_on(self):
        conn = sync_db.connect(self.db_path)
        try:
            enabled = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(
            enabled,
            1,
            "PRAGMA foreign_keys is OFF on a sync connection; every ON DELETE "
            "CASCADE in the schema is a no-op.",
        )

    def test_deleting_photo_cascades_to_children(self):
        conn = sync_db.connect(self.db_path)
        try:
            cur = conn.cursor()
            cur.execute("INSERT INTO photos (file_path) VALUES (?)", ("/cascade/sync.jpg",))
            photo_id = cur.lastrowid

            cur.execute(
                "INSERT INTO tags (photo_id, tag, score) VALUES (?, ?, ?)",
                (photo_id, "beach", 0.9),
            )
            cur.execute(
                "INSERT INTO embeddings (photo_id, dim, vector) VALUES (?, ?, ?)",
                (photo_id, 4, b"\x00\x01\x02\x03"),
            )
            cur.execute(
                "INSERT INTO clusters (photo_id, cluster_id, score) VALUES (?, ?, ?)",
                (photo_id, 7, 0.5),
            )
            cur.execute(
                """INSERT INTO faces (photo_id, face_index, embedding)
                   VALUES (?, ?, ?)""",
                (photo_id, 0, b"\x04\x05"),
            )
            face_id = cur.lastrowid
            cur.execute(
                "INSERT INTO face_thumbs (face_id, thumb_path) VALUES (?, ?)",
                (face_id, "/crops/0.jpg"),
            )
            conn.commit()

            cur.execute("DELETE FROM photos WHERE id=?", (photo_id,))
            conn.commit()

            for table, column, value in (
                ("tags", "photo_id", photo_id),
                ("embeddings", "photo_id", photo_id),
                ("clusters", "photo_id", photo_id),
                ("faces", "photo_id", photo_id),
                ("face_thumbs", "face_id", face_id),
            ):
                cur.execute(f"SELECT COUNT(*) FROM {table} WHERE {column}=?", (value,))
                remaining = cur.fetchone()[0]
                self.assertEqual(
                    remaining,
                    0,
                    f"{remaining} orphan row(s) left in {table} after the parent "
                    f"photo was deleted; ON DELETE CASCADE did not fire.",
                )
        finally:
            conn.close()


class TestAsyncCascadeDelete(_TempDBMixin, unittest.TestCase):
    """core.db_async.connect_async must enforce foreign keys."""

    def setUp(self):
        super().setUp()
        asyncio.run(async_db.init_db_async(self.db_path))

    def test_foreign_keys_pragma_is_on(self):
        async def check():
            async with async_db.connect_async(self.db_path) as conn:
                cur = await conn.execute("PRAGMA foreign_keys")
                row = await cur.fetchone()
                return row[0]

        self.assertEqual(
            asyncio.run(check()),
            1,
            "PRAGMA foreign_keys is OFF on an async connection; every ON DELETE "
            "CASCADE in the schema is a no-op.",
        )

    def test_deleting_photo_cascades_to_children(self):
        async def scenario():
            async with async_db.connect_async(self.db_path) as conn:
                cur = await conn.cursor()
                await cur.execute(
                    "INSERT INTO photos (file_path) VALUES (?)", ("/cascade/async.jpg",)
                )
                photo_id = cur.lastrowid

                await cur.execute(
                    "INSERT INTO tags (photo_id, tag, score) VALUES (?, ?, ?)",
                    (photo_id, "sunset", 0.8),
                )
                await cur.execute(
                    "INSERT INTO embeddings (photo_id, dim, vector) VALUES (?, ?, ?)",
                    (photo_id, 4, b"\x00\x01\x02\x03"),
                )
                await cur.execute(
                    "INSERT INTO clusters (photo_id, cluster_id, score) VALUES (?, ?, ?)",
                    (photo_id, 9, 0.5),
                )
                await cur.execute(
                    """INSERT INTO faces (photo_id, face_index, embedding)
                       VALUES (?, ?, ?)""",
                    (photo_id, 0, b"\x06\x07"),
                )
                face_id = cur.lastrowid
                await cur.execute(
                    "INSERT INTO face_thumbs (face_id, thumb_path) VALUES (?, ?)",
                    (face_id, "/crops/1.jpg"),
                )
                await conn.commit()

                await cur.execute("DELETE FROM photos WHERE id=?", (photo_id,))
                await conn.commit()

                counts = {}
                for table, column, value in (
                    ("tags", "photo_id", photo_id),
                    ("embeddings", "photo_id", photo_id),
                    ("clusters", "photo_id", photo_id),
                    ("faces", "photo_id", photo_id),
                    ("face_thumbs", "face_id", face_id),
                ):
                    await cur.execute(
                        f"SELECT COUNT(*) FROM {table} WHERE {column}=?", (value,)
                    )
                    counts[table] = (await cur.fetchone())[0]
                return counts

        counts = asyncio.run(scenario())
        for table, remaining in counts.items():
            self.assertEqual(
                remaining,
                0,
                f"{remaining} orphan row(s) left in {table} after the parent photo "
                f"was deleted; ON DELETE CASCADE did not fire.",
            )


if __name__ == "__main__":
    unittest.main()
