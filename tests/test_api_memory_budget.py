"""Guard the API tier's import-time memory budget.

`chitra-api.service` was cgroup-OOM-killed 6x in two days. The measured cause was
~450 MB per uvicorn worker of pure torch/transformers *import weight with no model
loaded*, times 4 workers, out of a 4 GB cap. These tests fail if that weight — or
any other heavy ML runtime — comes back into the API's import chain.

IMPORTANT: the probe must run in a **fresh subprocess**. `tests/run_tests.py`
discovers every module into a single interpreter, and other test modules import
`core.jobs` and construct a real `ClipEmbedder`. An in-process `sys.modules`
assertion would be polluted by whatever ran first and would silently pass (or
fail) for reasons that have nothing to do with the module under test.
"""
import json
import os
import subprocess
import sys
import textwrap
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Runtimes that must never be resident after importing the API. Each drags in
# hundreds of MB of interpreter state before a single model is loaded.
HEAVY_MODULES = ("torch", "transformers", "onnxruntime", "insightface")

# Measured import weight of the API with the ML chain removed was ~97 MB; the
# budget leaves headroom without leaving room for a torch import (~450 MB).
RSS_BUDGET_MB = 200

_PROBE = textwrap.dedent(
    """
    import json, resource, sys
    __import__({target!r})
    print(json.dumps({{
        "rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "heavy": [m for m in {heavy!r} if m in sys.modules],
    }}))
    """
)


def probe_import(target):
    """Import `target` in a fresh interpreter; return its rss_mb and heavy modules.

    Runs with CHITRA_DB_PATH pointed at a scratch file: importing `app_fastapi`
    reads that variable at module scope, and the default is the *live production
    database*.
    """
    env = os.environ.copy()
    env["CHITRA_DB_PATH"] = env.get("CHITRA_DB_PATH") or "/tmp/chitra_test.db"
    code = _PROBE.format(target=target, heavy=list(HEAVY_MODULES))
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"import probe for {target!r} failed (exit {proc.returncode}):\n{proc.stderr}"
        )
    # The app logs warnings on import; the JSON payload is the last stdout line.
    return json.loads(proc.stdout.strip().splitlines()[-1])


class TestApiImportIsMLFree(unittest.TestCase):
    """`import app_fastapi` must not drag the ML runtimes into every uvicorn worker."""

    @classmethod
    def setUpClass(cls):
        cls.result = probe_import("app_fastapi")

    def test_no_heavy_ml_modules_resident(self):
        self.assertEqual(
            [],
            self.result["heavy"],
            f"importing app_fastapi loaded {self.result['heavy']} — "
            f"each uvicorn worker pays this with no model loaded",
        )

    def test_import_rss_within_budget(self):
        self.assertLess(
            self.result["rss_mb"],
            RSS_BUDGET_MB,
            f"import of app_fastapi resident at {self.result['rss_mb']:.0f} MB, "
            f"budget is {RSS_BUDGET_MB} MB (x4 uvicorn workers)",
        )


class TestJobsImportIsMLFree(unittest.TestCase):
    """`core.jobs` is imported by the API purely for its job *names*.

    The workers load the models they need at call time; importing the module must
    not itself cost a torch import in the API process.
    """

    @classmethod
    def setUpClass(cls):
        cls.result = probe_import("core.jobs")

    def test_no_heavy_ml_modules_resident(self):
        self.assertEqual(
            [],
            self.result["heavy"],
            f"importing core.jobs loaded {self.result['heavy']} — "
            f"the API imports this module only to enqueue by reference",
        )


if __name__ == "__main__":
    unittest.main()
