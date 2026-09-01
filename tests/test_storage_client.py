"""
Unit tests for MinIO storage client operations.
"""
import unittest
import logging
import os
import tempfile
from pathlib import Path

from minio.error import S3Error, ServerError
from urllib3.exceptions import MaxRetryError, ResponseError

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


class TestFileExistsSwallowsTransportFailures(unittest.TestCase):
    """`file_exists` must not leak transport errors as bare 500s.

    When the underlying disk is failing, `stat_object` does not raise `S3Error`
    — MinIO answers 500, urllib3 exhausts its retries and raises
    `MaxRetryError(... Caused by ResponseError('too many 500 error responses'))`,
    a `urllib3.exceptions.HTTPError`. The old `except S3Error` missed it
    entirely, so it escaped uncaught out of the HTTP handler: 41 tracebacks
    through `file_exists`/`stat_object` in a ~2 hour window, each an unhandled
    500 with no detail.

    Storage never has to be up for these: the MinIO client is stubbed.
    """

    def _client(self, stat_effect):
        """A MinIOStorageClient with a stubbed `client`, built without running
        __init__ (which does a network round-trip to create the bucket)."""
        storage = object.__new__(MinIOStorageClient)
        storage.bucket_name = "chitra-photos"

        class _Stub:
            def stat_object(self, bucket, path):
                if isinstance(stat_effect, BaseException):
                    raise stat_effect
                return stat_effect

        storage.client = _Stub()
        return storage

    @staticmethod
    def _s3_error(code):
        return S3Error(None, code, "message", "/chitra-photos/x", "req-id", None)

    @staticmethod
    def _max_retry_error():
        return MaxRetryError(
            None,
            "/chitra-photos/photos/2026/04/x.jpg",
            ResponseError("too many 500 error responses"),
        )

    def test_true_when_the_object_is_there(self):
        self.assertTrue(self._client(object()).file_exists("photos/x.jpg"))

    def test_false_when_the_object_is_absent(self):
        storage = self._client(self._s3_error("NoSuchKey"))
        self.assertFalse(storage.file_exists("photos/x.jpg"))

    def test_false_on_transport_failure_instead_of_raising(self):
        storage = self._client(self._max_retry_error())
        with self.assertLogs("core.storage_client", level="WARNING"):
            self.assertFalse(storage.file_exists("photos/2026/04/x.jpg"))

    def test_transport_failure_is_logged_with_the_path_and_the_cause(self):
        """A transport error means the storage is unhealthy, which is worth
        knowing — returning False silently would hide a failing disk."""
        storage = self._client(self._max_retry_error())
        with self.assertLogs("core.storage_client", level="WARNING") as captured:
            storage.file_exists("photos/2026/04/x.jpg")
        joined = "\n".join(captured.output)
        self.assertIn("photos/2026/04/x.jpg", joined)
        self.assertIn("too many 500 error responses", joined)

    def test_false_on_minio_server_error(self):
        """A 500 that minio itself turns into ServerError rather than letting
        urllib3 retry — same meaning, same handling."""
        storage = self._client(ServerError("server failed with HTTP status code 500", 500))
        with self.assertLogs("core.storage_client", level="WARNING"):
            self.assertFalse(storage.file_exists("photos/x.jpg"))

    def test_absence_is_not_logged_as_a_storage_problem(self):
        """A missing object is a normal answer, not an incident."""
        storage = self._client(self._s3_error("NoSuchKey"))
        logger = logging.getLogger("core.storage_client")
        records = []

        class _Collect(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = _Collect()
        logger.addHandler(handler)
        self.addCleanup(logger.removeHandler, handler)
        self.assertFalse(storage.file_exists("photos/x.jpg"))
        self.assertEqual(records, [])


if __name__ == '__main__':
    unittest.main()

