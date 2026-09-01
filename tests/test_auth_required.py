"""
Every route that reads or mutates library data must reject an unauthenticated
caller. This suite is the regression guard for the class of bug where a route
is added (or edited) without an auth dependency and silently serves anyone who
can reach the API.

The API is exposed to the internet through a cloudflared tunnel, so a missing
dependency is not a theoretical problem: DELETE /api/photos/{id} removes the
database row *and* the original, thumbnail, transcode and face crops from
object storage.
"""
import os
import tempfile
import unittest

# app_fastapi binds DB_PATH at import time, and the default is the production
# photo.db sitting in the repo root. Point it somewhere disposable *before* the
# import below.
_TEST_DB = os.path.join(tempfile.gettempdir(), "chitra_auth_required_test.db")
os.environ.setdefault("CHITRA_DB_PATH", _TEST_DB)


class TestUnauthenticatedAccessIsRejected(unittest.TestCase):
    """Protected routes must answer 401 when no credentials are supplied."""

    @classmethod
    def setUpClass(cls):
        try:
            from fastapi.testclient import TestClient
            from app_fastapi import app
            cls.client = TestClient(app)
        except ImportError:
            raise unittest.SkipTest("FastAPI TestClient not available. Install: pip install httpx")
        except Exception as e:
            raise unittest.SkipTest(f"Could not import app: {e}")

    def assertRequiresAuth(self, method: str, path: str):
        response = self.client.request(method, path)
        self.assertEqual(
            response.status_code,
            401,
            f"{method} {path} answered {response.status_code} without credentials; "
            f"expected 401. A route that is not 401 here is reachable by anyone.",
        )

    # The destructive one. This is the regression test for the endpoint that
    # shipped with no auth dependency at all.
    def test_delete_photo_requires_auth(self):
        self.assertRequiresAuth("DELETE", "/api/photos/1")

    def test_list_photos_requires_auth(self):
        self.assertRequiresAuth("GET", "/api/photos")

    def test_get_photo_requires_auth(self):
        self.assertRequiresAuth("GET", "/api/photos/1")

    def test_search_requires_auth(self):
        self.assertRequiresAuth("GET", "/api/search/photos?query=beach")

    def test_list_faces_requires_auth(self):
        self.assertRequiresAuth("GET", "/api/faces")

    def test_list_persons_requires_auth(self):
        self.assertRequiresAuth("GET", "/api/persons")

    def test_storage_read_requires_auth(self):
        self.assertRequiresAuth("GET", "/api/storage/photos/2026/01/anything.jpg")


class TestPublicRoutesStayPublic(unittest.TestCase):
    """The deliberately public surface must not regress into requiring auth."""

    @classmethod
    def setUpClass(cls):
        try:
            from fastapi.testclient import TestClient
            from app_fastapi import app
            cls.client = TestClient(app)
        except ImportError:
            raise unittest.SkipTest("FastAPI TestClient not available. Install: pip install httpx")
        except Exception as e:
            raise unittest.SkipTest(f"Could not import app: {e}")

    def test_health_is_public(self):
        self.assertNotEqual(self.client.get("/api/health").status_code, 401)

    def test_root_is_public(self):
        self.assertNotEqual(self.client.get("/").status_code, 401)


if __name__ == "__main__":
    unittest.main()
