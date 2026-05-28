"""Unit tests for media-scoped tokens in core.auth."""
import unittest
from datetime import timedelta

from core import auth


class TestMediaToken(unittest.TestCase):
    def test_roundtrip_valid(self):
        tok = auth.create_media_token(42)
        payload = auth.verify_media_token(tok)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["sub"], "42")
        self.assertEqual(payload["scope"], auth.MEDIA_TOKEN_SCOPE)

    def test_regular_access_token_rejected_as_media(self):
        # A normal access token (no media scope) must not pass media verification.
        access = auth.create_access_token({"sub": "42", "username": "x", "role": "user"})
        self.assertIsNone(auth.verify_media_token(access))

    def test_media_token_is_detectable(self):
        tok = auth.create_media_token(7)
        payload = auth.verify_token(tok)
        self.assertTrue(auth.is_media_token(payload))
        access = auth.create_access_token({"sub": "7"})
        self.assertFalse(auth.is_media_token(auth.verify_token(access)))

    def test_expired_media_token_rejected(self):
        tok = auth.create_media_token(1, expires_minutes=-1)
        self.assertIsNone(auth.verify_media_token(tok))

    def test_garbage_rejected(self):
        self.assertIsNone(auth.verify_media_token("not-a-jwt"))


if __name__ == "__main__":
    unittest.main()
