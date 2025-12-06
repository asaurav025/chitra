"""
Unit tests for MinIO storage client operations.
"""
import unittest
import os
import tempfile
from pathlib import Path

from core.storage_client import MinIOStorageClient


class TestMinIOStorageClient(unittest.TestCase):
    """Test MinIO storage client operations."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class - initialize storage client."""
        try:
            cls.storage = MinIOStorageClient()
            cls.test_bucket = cls.storage.bucket_name
        except Exception as e:
            raise unittest.SkipTest(f"MinIO not available: {e}")
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_file_path = "test/test_file.txt"
        self.test_data = b"Hello, MinIO! This is test data."
    
    def tearDown(self):
        """Clean up after tests."""
        # Try to delete test file if it exists
        try:
            if self.storage.file_exists(self.test_file_path):
                self.storage.delete_file(self.test_file_path)
        except:
            pass
    
    def test_connection(self):
        """Test MinIO connection."""
        self.assertIsNotNone(self.storage.client)
        self.assertEqual(self.storage.bucket_name, self.test_bucket)
    
    def test_upload_file(self):
        """Test file upload."""
        result = self.storage.upload_file(self.test_data, self.test_file_path)
        self.assertEqual(result, self.test_file_path)
        self.assertTrue(self.storage.file_exists(self.test_file_path))
    
    def test_download_file(self):
        """Test file download."""
        # Upload first
        self.storage.upload_file(self.test_data, self.test_file_path)
        
        # Download
        downloaded = self.storage.download_file(self.test_file_path)
        self.assertEqual(downloaded, self.test_data)
    
    def test_file_exists(self):
        """Test file existence check."""
        # File doesn't exist yet
        self.assertFalse(self.storage.file_exists(self.test_file_path))
        
        # Upload file
        self.storage.upload_file(self.test_data, self.test_file_path)
        
        # File exists now
        self.assertTrue(self.storage.file_exists(self.test_file_path))
    
    def test_delete_file(self):
        """Test file deletion."""
        # Upload file
        self.storage.upload_file(self.test_data, self.test_file_path)
        self.assertTrue(self.storage.file_exists(self.test_file_path))
        
        # Delete file
        result = self.storage.delete_file(self.test_file_path)
        self.assertTrue(result)
        self.assertFalse(self.storage.file_exists(self.test_file_path))
    
    def test_generate_photo_path(self):
        """Test photo path generation."""
        filename = "test_photo.jpg"
        path = self.storage.generate_photo_path(filename)
        self.assertIn("photos", path)
        self.assertIn(filename, path)
    
    def test_generate_thumbnail_path(self):
        """Test thumbnail path generation."""
        path = self.storage.generate_thumbnail_path(123, "photo")
        self.assertIn("thumbnails", path)
        self.assertIn("123", path)
    
    @unittest.skip("Requires async context")
    def test_async_operations(self):
        """Test async operations (requires async context)."""
        # These would be tested in async test framework
        pass


if __name__ == '__main__':
    unittest.main()

