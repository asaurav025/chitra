"""Tests for JWT secret configuration in core.auth."""
import importlib
import os
import unittest


KNOWN_DEFAULT = "your-secret-key-change-in-production"


def reload_auth(env_value):
    """Reload core.auth with JWT_SECRET_KEY set to env_value (None = unset)."""
    saved = os.environ.pop("JWT_SECRET_KEY", None)
    try:
        if env_value is not None:
            os.environ["JWT_SECRET_KEY"] = env_value
        import core.auth
        return importlib.reload(core.auth)
    finally:
        if saved is None:
            os.environ.pop("JWT_SECRET_KEY", None)
        else:
            os.environ["JWT_SECRET_KEY"] = saved


class JWTSecretConfigTest(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        # Leave the module in a state matching the real environment
        import core.auth
        importlib.reload(core.auth)

    def test_uses_env_value_when_set(self):
        auth = reload_auth("test-secret-from-env")
        self.assertEqual(auth.JWT_SECRET_KEY, "test-secret-from-env")

    def test_never_uses_known_default_when_unset(self):
        auth = reload_auth(None)
        self.assertNotEqual(auth.JWT_SECRET_KEY, KNOWN_DEFAULT)

    def test_unset_fallback_is_a_strong_random_secret(self):
        # reload() returns the same module object, so capture the string
        # values immediately — two fresh loads must not agree
        secret1 = reload_auth(None).JWT_SECRET_KEY
        secret2 = reload_auth(None).JWT_SECRET_KEY
        self.assertGreaterEqual(len(secret1), 32)
        self.assertNotEqual(secret1, secret2)


if __name__ == "__main__":
    unittest.main()
