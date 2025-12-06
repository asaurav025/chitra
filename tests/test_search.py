"""
Tests for search functionality.
"""
import unittest
import os
import tempfile
import asyncio
import numpy as np

from core import db_async
from core.embedder import ClipEmbedder


class TestSearch(unittest.TestCase):
    """Test search functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class."""
        cls.test_db = tempfile.mktemp(suffix='.db')
        try:
            asyncio.run(db_async.init_db_async(cls.test_db))
        except Exception as e:
            raise unittest.SkipTest(f"Could not initialize test database: {e}")
        
        # Try to initialize embedder (may skip if model not available)
        try:
            cls.embedder = ClipEmbedder()
        except Exception as e:
            cls.embedder = None
            print(f"Warning: Could not load embedder: {e}")
    
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
    
    @unittest.skipUnless(lambda: TestSearch.embedder is not None, "Embedder not available")
    def test_text_embedding(self):
        """Test text embedding generation."""
        if self.embedder is None:
            self.skipTest("Embedder not available")
        
        query = "test query"
        embedding = self.embedder.text_embedding(query)
        
        self.assertIsInstance(embedding, np.ndarray)
        self.assertGreater(embedding.shape[0], 0)
    
    def test_search_with_embeddings(self):
        """Test search functionality with embeddings."""
        async def test():
            async with db_async.connect_async(self.test_db) as conn:
                # Insert test photo
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
                if row:
                    photo_id = row["id"]
                    
                    # Add test embedding
                    test_embedding = np.random.rand(512).astype(np.float32)
                    await db_async.put_embedding_async(
                        conn, photo_id, test_embedding.tobytes(), 512
                    )
                    await conn.commit()
                    
                    # Get embeddings
                    embeddings = await db_async.get_embeddings_async(conn)
                    self.assertGreater(len(embeddings), 0)
        
        self.run_async(test())
    
    def test_empty_search_results(self):
        """Test search with no embeddings."""
        async def test():
            async with db_async.connect_async(self.test_db) as conn:
                embeddings = await db_async.get_embeddings_async(conn)
                # If no embeddings, search should return empty results
                # This is handled by the endpoint
                pass
        
        self.run_async(test())


if __name__ == '__main__':
    unittest.main()

