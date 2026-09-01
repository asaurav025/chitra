"""Thread caps for the BLAS/OpenMP stacks, and their override precedence.

Measured on this box (6 cores): torch and ONNX Runtime both default to 6
threads, and a CLIP image embed costs 153 ms at 6 threads but 68 ms at 3 — the
default is 2.25x slower. Nothing in the codebase set a thread count.

`.env.production` is git-ignored and systemd unit files need sudo, so the caps
live in `thread_limits.sh`, sourced by the launcher scripts *after* they source
`.env.production`. The ordering plus `: "${VAR:=n}"` is what makes an operator
override win over the built-in default; these tests pin both.
"""
import os
import subprocess
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAGMENT = os.path.join(REPO_ROOT, "thread_limits.sh")

THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def source_fragment(argv_line, env_overrides=None):
    """Source `thread_limits.sh` in a fresh bash and report the resulting env.

    `argv_line` is the literal shell line that sources it, so tests can feed in
    the exact line a launcher script uses rather than a paraphrase of it.
    """
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    env.update(env_overrides or {})
    script = f'set -e\ncd "{REPO_ROOT}"\n{argv_line}\n'
    script += "".join(f'echo "{v}=${{{v}-<unset>}}"\n' for v in THREAD_VARS)
    proc = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, env=env, timeout=30
    )
    if proc.returncode != 0:
        raise AssertionError(f"sourcing failed (exit {proc.returncode}):\n{proc.stderr}")
    return dict(line.split("=", 1) for line in proc.stdout.strip().splitlines())


def launcher_lines(name):
    with open(os.path.join(REPO_ROOT, name)) as fh:
        return fh.read().splitlines()


def source_line_in(name):
    """The line in launcher `name` that sources the thread-limits fragment."""
    hits = [ln.strip() for ln in launcher_lines(name) if "thread_limits.sh" in ln
            and not ln.strip().startswith("#")]
    if not hits:
        raise AssertionError(f"{name} does not source thread_limits.sh")
    return hits[0]


class TestThreadLimitsFragment(unittest.TestCase):
    def test_fragment_exists(self):
        self.assertTrue(
            os.path.exists(FRAGMENT), f"{FRAGMENT} is missing — nothing caps thread counts"
        )

    def test_sets_every_thread_var_to_the_requested_default(self):
        env = source_fragment('. ./thread_limits.sh 2')
        for var in THREAD_VARS:
            self.assertEqual("2", env[var], f"{var} was not capped")

    def test_exports_and_does_not_merely_set(self):
        # A bare assignment would not reach torch, which reads these at import
        # in a child process.
        script = 'set -e\ncd "%s"\n. ./thread_limits.sh 3\nenv | grep -c "^OMP_NUM_THREADS=3$"' % REPO_ROOT
        proc = subprocess.run(
            ["bash", "-c", script], capture_output=True, text=True,
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")}, timeout=30,
        )
        self.assertEqual("1", proc.stdout.strip(), "OMP_NUM_THREADS was set but not exported")

    def test_preexisting_value_wins_over_the_default(self):
        """The whole point: .env.production / systemd Environment= must win."""
        env = source_fragment('. ./thread_limits.sh 2', {"OMP_NUM_THREADS": "8"})
        self.assertEqual("8", env["OMP_NUM_THREADS"], "the built-in default clobbered an override")
        # The vars the operator did not override still take the default.
        self.assertEqual("2", env["MKL_NUM_THREADS"])

    def test_empty_value_is_treated_as_unset(self):
        env = source_fragment('. ./thread_limits.sh 2', {"OMP_NUM_THREADS": ""})
        self.assertEqual("2", env["OMP_NUM_THREADS"])


class TestLauncherWiring(unittest.TestCase):
    def test_api_launcher_caps_at_two_threads(self):
        env = source_fragment(source_line_in("start_production.sh"))
        for var in THREAD_VARS:
            self.assertEqual(
                "2", env[var],
                f"start_production.sh leaves {var} at {env[var]}; the API's only "
                f"numeric work is a 1,694x512 GEMV over 3.31 MiB",
            )

    def test_worker_launcher_defaults_to_three_threads(self):
        env = source_fragment(source_line_in("start_workers.sh"))
        for var in THREAD_VARS:
            self.assertEqual("3", env[var], f"start_workers.sh leaves {var} at {env[var]}")

    def test_worker_launcher_honours_chitra_ml_threads(self):
        env = source_fragment(source_line_in("start_workers.sh"), {"CHITRA_ML_THREADS": "5"})
        self.assertEqual("5", env["OMP_NUM_THREADS"], "CHITRA_ML_THREADS is not respected")

    def test_caps_are_sourced_after_env_production(self):
        """Ordering is load-bearing: sourcing first would let the default lose."""
        for name in ("start_production.sh", "start_workers.sh"):
            with self.subTest(script=name):
                lines = launcher_lines(name)
                env_idx = next(
                    i for i, ln in enumerate(lines) if ".env.production" in ln and "." in ln
                )
                cap_idx = next(
                    i for i, ln in enumerate(lines)
                    if "thread_limits.sh" in ln and not ln.strip().startswith("#")
                )
                self.assertGreater(
                    cap_idx, env_idx,
                    f"{name} sources thread_limits.sh at line {cap_idx + 1}, before "
                    f".env.production at line {env_idx + 1} — overrides would be ignored",
                )


if __name__ == "__main__":
    unittest.main()
