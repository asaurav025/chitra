"""
Unit tests for async database operations.
"""
import unittest
import os
import tempfile
import asyncio
import aiosqlite

from core import db_async


class TestAsyncDatabase(unittest.TestCase):
    """Test async database operations."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class."""
        cls.test_db = tempfile.mktemp(suffix='.db')
        # Initialize test database
        asyncio.run(db_async.init_db_async(cls.test_db))
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test database."""
        if os.path.exists(cls.test_db):
            os.unlink(cls.test_db)
    
    def setUp(self):
        """Set up test fixtures."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        """Clean up after tests."""
        self.loop.close()
    
    def run_async(self, coro):
        """Helper to run async functions."""
        return self.loop.run_until_complete(coro)
    
    def test_connect_async(self):
        """Test async database connection."""
        async def test():
            async with db_async.connect_async(self.test_db) as conn:
                self.assertIsNotNone(conn)
                await conn.execute("SELECT 1")
        
        self.run_async(test())
    
    def test_upsert_photo_async(self):
        """Test photo upsert operation."""
        async def test():
            async with db_async.connect_async(self.test_db) as conn:
                await db_async.upsert_photo_async(
                    conn,
                    file_path="test/photo.jpg",
                    size=1024,
                    checksum="test_checksum",
                    phash="test_phash"
                )
                await conn.commit()
                
                # Verify photo was inserted
                cur = await conn.cursor()
                await cur.execute("SELECT id FROM photos WHERE file_path=?", ("test/photo.jpg",))
                row = await cur.fetchone()
                self.assertIsNotNone(row)
        
        self.run_async(test())
    
    def test_get_embeddings_async(self):
        """Test getting embeddings."""
        async def test():
            async with db_async.connect_async(self.test_db) as conn:
                # First insert a photo
                await db_async.upsert_photo_async(
                    conn,
                    file_path="test/photo.jpg",
                    size=1024
                )
                await conn.commit()
                
                # Get photo ID
                cur = await conn.cursor()
                await cur.execute("SELECT id FROM photos WHERE file_path=?", ("test/photo.jpg",))
                row = await cur.fetchone()
                photo_id = row["id"]
                
                # Add embedding
                test_embedding = b"test_embedding_data"
                await db_async.put_embedding_async(conn, photo_id, test_embedding, 512)
                await conn.commit()
                
                # Get embeddings
                embeddings = await db_async.get_embeddings_async(conn)
                self.assertGreater(len(embeddings), 0)
        
        self.run_async(test())
    
    def test_add_tag_async(self):
        """Test adding tags."""
        async def test():
            async with db_async.connect_async(self.test_db) as conn:
                # Insert photo
                await db_async.upsert_photo_async(
                    conn,
                    file_path="test/photo.jpg",
                    size=1024
                )
                await conn.commit()
                
                # Get photo ID
                cur = await conn.cursor()
                await cur.execute("SELECT id FROM photos WHERE file_path=?", ("test/photo.jpg",))
                row = await cur.fetchone()
                photo_id = row["id"]
                
                # Add tag
                await db_async.add_tag_async(conn, photo_id, "test_tag", 0.9)
                await conn.commit()
                
                # Verify tag
                await cur.execute(
                    "SELECT tag, score FROM tags WHERE photo_id=?", (photo_id,)
                )
                tag_row = await cur.fetchone()
                self.assertIsNotNone(tag_row)
                self.assertEqual(tag_row["tag"], "test_tag")
        
        self.run_async(test())


if __name__ == '__main__':
    unittest.main()

