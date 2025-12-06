"""
Tests for background job processing.
"""
import unittest
import os
import tempfile
from pathlib import Path

from core.jobs import (
    process_photo_embedding_job,
    process_photo_faces_job,
    index_embeddings_batch_job,
    index_faces_batch_job,
)
from core.storage_client import MinIOStorageClient


class TestBackgroundJobs(unittest.TestCase):
    """Test background job functions."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class."""
        cls.test_db = tempfile.mktemp(suffix='.db')
        # Initialize test database
        try:
            import asyncio
            from core import db_async
            asyncio.run(db_async.init_db_async(cls.test_db))
        except Exception as e:
            raise unittest.SkipTest(f"Could not initialize test database: {e}")
        
        # Check if MinIO is available
        try:
            cls.storage = MinIOStorageClient()
        except Exception as e:
            raise unittest.SkipTest(f"MinIO not available: {e}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test database."""
        if os.path.exists(cls.test_db):
            os.unlink(cls.test_db)
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_photo_path = "test/test_photo.jpg"
        # Create a minimal test image file in MinIO
        # (In real tests, you'd upload an actual image)
    
    @unittest.skip("Requires actual image file and embedder model")
    def test_process_photo_embedding_job(self):
        """Test photo embedding job processing."""
        # This would require:
        # 1. A real photo file in MinIO
        # 2. Photo ID in database
        # 3. CLIP embedder model loaded
        pass
    
    @unittest.skip("Requires actual image file and face detection model")
    def test_process_photo_faces_job(self):
        """Test photo faces job processing."""
        # This would require:
        # 1. A real photo file in MinIO
        # 2. Photo ID in database
        # 3. Face detection model loaded
        pass
    
    def test_index_embeddings_batch_job_structure(self):
        """Test that index_embeddings_batch_job function exists and has correct signature."""
        # Just verify the function exists and is callable
        self.assertTrue(callable(index_embeddings_batch_job))
        
        # Test with empty list
        result = index_embeddings_batch_job([], self.test_db, True)
        self.assertEqual(result, 0)
    
    def test_index_faces_batch_job_structure(self):
        """Test that index_faces_batch_job function exists and has correct signature."""
        # Just verify the function exists and is callable
        self.assertTrue(callable(index_faces_batch_job))
        
        # Test with empty list
        result = index_faces_batch_job([], self.test_db, 0.5, 160)
        self.assertEqual(result, 0)


if __name__ == '__main__':
    unittest.main()

