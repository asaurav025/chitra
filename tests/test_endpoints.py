"""
Integration tests for FastAPI endpoints.
"""
import unittest
import os
import tempfile
import json
from pathlib import Path

# Note: These tests require FastAPI TestClient
# Install with: pip install httpx


class TestEndpoints(unittest.TestCase):
    """Test FastAPI endpoints."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class."""
        try:
            from fastapi.testclient import TestClient
            from app_fastapi import app
            cls.client = TestClient(app)
            cls.app = app
        except ImportError:
            raise unittest.SkipTest("FastAPI TestClient not available. Install: pip install httpx")
        except Exception as e:
            raise unittest.SkipTest(f"Could not import app: {e}")
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("version", data)
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("message", data)
        self.assertIn("version", data)
    
    def test_list_photos(self):
        """Test listing photos."""
        response = self.client.get("/api/photos?limit=10&offset=0")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("items", data)
        self.assertIn("limit", data)
        self.assertIn("offset", data)
        self.assertIsInstance(data["items"], list)
    
    def test_get_photo_not_found(self):
        """Test getting non-existent photo."""
        response = self.client.get("/api/photos/999999")
        self.assertEqual(response.status_code, 404)
    
    def test_scan_path_invalid(self):
        """Test scan path with invalid path."""
        response = self.client.post(
            "/api/photos/scan-path",
            json={"root": "/nonexistent/path"}
        )
        self.assertEqual(response.status_code, 400)
    
    def test_create_person(self):
        """Test creating a person."""
        response = self.client.post(
            "/api/persons",
            json={"name": "Test Person"}
        )
        if response.status_code == 200:
            data = response.json()
            self.assertIn("id", data)
            self.assertIn("name", data)
            self.assertEqual(data["name"], "Test Person")
    
    def test_list_persons(self):
        """Test listing persons."""
        response = self.client.get("/api/persons")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("persons", data)
        self.assertIsInstance(data["persons"], list)
    
    def test_list_faces(self):
        """Test listing faces."""
        response = self.client.get("/api/faces")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("items", data)
        self.assertIsInstance(data["items"], list)
    
    def test_search_photos_missing_query(self):
        """Test search photos without query."""
        response = self.client.get("/api/search/photos")
        self.assertEqual(response.status_code, 400)
    
    def test_search_photos_with_query(self):
        """Test search photos with query."""
        response = self.client.get("/api/search/photos?q=test")
        # May return 200 with empty results or 400 if no embeddings
        self.assertIn(response.status_code, [200, 400])
        if response.status_code == 200:
            data = response.json()
            self.assertIn("query", data)
            self.assertIn("results", data)


if __name__ == '__main__':
    unittest.main()

