"""
Tests for core/faiss_index.py — FAISSIndexManager.

Covers three defects:

* index_dir was a relative path resolved against the process CWD, so a
  systemd unit without a matching WorkingDirectory silently created a second,
  empty index directory and face matching stopped working with no error.
* save_index wrote with faiss.write_index straight onto the final path, so a
  concurrent reader could read a truncated file.
* update_index is a read-modify-write with no lock, reachable from 4
  concurrent default-queue workers — a lost update.
"""
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

import faiss
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.faiss_index as faiss_index_mod
from core.faiss_index import FAISSIndexManager

ENV_VAR = "CHITRA_FAISS_INDEX_DIR"


def _vecs(n, dim=32, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.random((n, dim), dtype=np.float32)
    faiss.normalize_L2(x)
    return x


class IndexDirResolutionTests(unittest.TestCase):
    """Fix 4: index_dir must be absolute and must not follow the CWD."""

    def setUp(self):
        self._env = os.environ.get(ENV_VAR)
        os.environ.pop(ENV_VAR, None)
        self._cwd = os.getcwd()

    def tearDown(self):
        os.chdir(self._cwd)
        if self._env is None:
            os.environ.pop(ENV_VAR, None)
        else:
            os.environ[ENV_VAR] = self._env

    def test_default_index_dir_is_absolute(self):
        self.assertTrue(Path(FAISSIndexManager().index_dir).is_absolute())

    def test_default_index_dir_does_not_move_with_cwd(self):
        """The whole point: two different CWDs must resolve to one directory."""
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            os.chdir(a)
            first = Path(FAISSIndexManager().index_dir)
            os.chdir(b)
            second = Path(FAISSIndexManager().index_dir)

            self.assertEqual(first, second)
            # And it must not have created a stray directory beside the CWD —
            # that empty second directory is exactly the silent failure.
            self.assertFalse((Path(a) / "faiss_indexes").exists())
            self.assertFalse((Path(b) / "faiss_indexes").exists())

    def test_default_index_dir_is_derived_from_the_module_location(self):
        repo_root = Path(faiss_index_mod.__file__).resolve().parent.parent
        self.assertEqual(
            Path(FAISSIndexManager().index_dir),
            repo_root / "faiss_indexes",
        )

    def test_env_var_overrides_the_default(self):
        with tempfile.TemporaryDirectory() as t:
            target = Path(t) / "custom_indexes"
            os.environ[ENV_VAR] = str(target)
            manager = FAISSIndexManager()
            self.assertEqual(Path(manager.index_dir), target.resolve())
            self.assertTrue(target.is_dir())

    def test_env_var_relative_value_is_made_absolute(self):
        with tempfile.TemporaryDirectory() as t:
            os.chdir(t)
            os.environ[ENV_VAR] = "rel_indexes"
            path = Path(FAISSIndexManager().index_dir)
            self.assertTrue(path.is_absolute())
            self.assertEqual(path, (Path(t) / "rel_indexes").resolve())

    def test_explicit_argument_still_wins_and_is_absolute(self):
        with tempfile.TemporaryDirectory() as t:
            os.environ[ENV_VAR] = str(Path(t) / "from_env")
            explicit = Path(t) / "explicit"
            manager = FAISSIndexManager(index_dir=str(explicit))
            self.assertEqual(Path(manager.index_dir), explicit.resolve())

    def test_resolved_path_is_logged_once_per_process(self):
        """A misconfiguration must be visible, but not once per detected face."""
        with tempfile.TemporaryDirectory() as t:
            target = Path(t) / "logged_indexes"
            os.environ[ENV_VAR] = str(target)
            faiss_index_mod._reset_logged_dirs_for_tests()

            with self.assertLogs(faiss_index_mod.logger, level="INFO") as captured:
                FAISSIndexManager()
            self.assertTrue(
                any(str(target) in line for line in captured.output),
                f"resolved path not logged: {captured.output}",
            )

            # Second construction for the same directory must stay quiet.
            with self.assertNoLogs(faiss_index_mod.logger, level="INFO"):
                FAISSIndexManager()


class AtomicSaveTests(unittest.TestCase):
    """Fix 3a: save_index must go via a temp file plus os.replace."""

    def setUp(self):
        self._env = os.environ.get(ENV_VAR)
        self._tmp = tempfile.TemporaryDirectory()
        os.environ[ENV_VAR] = self._tmp.name
        self.manager = FAISSIndexManager()

    def tearDown(self):
        self._tmp.cleanup()
        if self._env is None:
            os.environ.pop(ENV_VAR, None)
        else:
            os.environ[ENV_VAR] = self._env

    def _index(self, n=64):
        index = faiss.IndexFlatIP(32)
        index.add(_vecs(n))
        return index

    def test_save_index_uses_os_replace_onto_the_final_path(self):
        index = self._index()
        with mock.patch.object(
            faiss_index_mod.os, "replace", wraps=os.replace
        ) as spy:
            self.assertTrue(self.manager.save_index(index, "atomic"))

        self.assertTrue(spy.called, "save_index did not use os.replace")
        src, dst = spy.call_args[0][:2]
        final = self.manager.get_index_path("atomic")
        self.assertEqual(Path(dst), Path(final))
        self.assertNotEqual(Path(src), Path(final))
        # os.replace is only atomic within one filesystem.
        self.assertEqual(Path(src).parent, Path(final).parent)

    def test_faiss_never_writes_directly_to_the_final_path(self):
        index = self._index()
        final = str(self.manager.get_index_path("direct"))
        seen = []

        real_write = faiss_index_mod.faiss.write_index

        def spy_write(idx, path, *a, **kw):
            seen.append(path)
            return real_write(idx, path, *a, **kw)

        with mock.patch.object(faiss_index_mod.faiss, "write_index", spy_write):
            self.manager.save_index(index, "direct")

        self.assertTrue(seen)
        for path in seen:
            self.assertNotEqual(
                os.path.abspath(path),
                os.path.abspath(final),
                "faiss.write_index targeted the final path — truncation window",
            )

    def test_a_failed_write_leaves_the_previous_index_intact(self):
        good = self._index(64)
        self.manager.save_index(good, "keep")
        before = self.manager.get_index_path("keep").read_bytes()

        def boom(idx, path, *a, **kw):
            with open(path, "wb") as fh:  # partial file, then fail
                fh.write(b"\x00" * 128)
            raise RuntimeError("disk went away")

        with mock.patch.object(faiss_index_mod.faiss, "write_index", boom):
            self.assertFalse(self.manager.save_index(self._index(8), "keep"))

        self.assertEqual(self.manager.get_index_path("keep").read_bytes(), before)
        self.assertEqual(self.manager.load_index("keep").ntotal, 64)

    def test_no_temp_files_are_left_behind(self):
        self.manager.save_index(self._index(), "tidy")
        leftovers = [
            p.name
            for p in Path(self.manager.index_dir).iterdir()
            if p.name != "tidy.index" and not p.name.endswith(".lock")
        ]
        self.assertEqual(leftovers, [], f"temp files left behind: {leftovers}")


# A writer subprocess: rewrites one index repeatedly through the manager.
_WRITER = textwrap.dedent(
    """
    import os, sys
    sys.path.insert(0, {repo!r})
    os.environ["CHITRA_FAISS_INDEX_DIR"] = sys.argv[1]
    import faiss, numpy as np
    from core.faiss_index import FAISSIndexManager

    n, dim, rounds = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    rng = np.random.default_rng(7)
    x = rng.random((n, dim), dtype=np.float32)
    faiss.normalize_L2(x)
    index = faiss.IndexFlatIP(dim)
    index.add(x)

    manager = FAISSIndexManager()
    for _ in range(rounds):
        manager.save_index(index, "concurrent")
    """
)

# An updater subprocess: read-modify-write of a shared index via update_index.
_UPDATER = textwrap.dedent(
    """
    import os, sys
    sys.path.insert(0, {repo!r})
    os.environ["CHITRA_FAISS_INDEX_DIR"] = sys.argv[1]
    import faiss, numpy as np
    from core.faiss_index import FAISSIndexManager

    seed, per_proc, dim = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    rng = np.random.default_rng(seed)
    x = rng.random((per_proc, dim), dtype=np.float32)
    faiss.normalize_L2(x)

    FAISSIndexManager().update_index("shared", x)
    """
)

REPO_ROOT = str(Path(__file__).resolve().parent.parent)


class ConcurrentIndexWriteTests(unittest.TestCase):
    """Fix 3b/3c: readers must never see truncation; updates must not be lost."""

    def setUp(self):
        self._env = os.environ.get(ENV_VAR)
        self._tmp = tempfile.TemporaryDirectory()
        os.environ[ENV_VAR] = self._tmp.name
        self.manager = FAISSIndexManager()

    def tearDown(self):
        self._tmp.cleanup()
        if self._env is None:
            os.environ.pop(ENV_VAR, None)
        else:
            os.environ[ENV_VAR] = self._env

    def _run(self, source, args):
        script = Path(self._tmp.name) / "script.py"
        script.write_text(source.format(repo=REPO_ROOT))
        return subprocess.Popen(
            [sys.executable, str(script)] + [str(a) for a in args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def test_reader_never_sees_a_truncated_index(self):
        n, dim, rounds = 40000, 128, 12  # ~20 MB per write
        path = str(self.manager.get_index_path("concurrent"))

        # Seed the file so the reader has something valid from the start.
        seed = faiss.IndexFlatIP(dim)
        seed.add(_vecs(n, dim, seed=7))
        self.manager.save_index(seed, "concurrent")

        proc = self._run(_WRITER, [self._tmp.name, n, dim, rounds])
        reads, bad = 0, []
        try:
            while proc.poll() is None:
                try:
                    index = faiss.read_index(path)
                    reads += 1
                    if index.ntotal != n:
                        bad.append(f"ntotal={index.ntotal}")
                except Exception as exc:  # truncated / partial file
                    bad.append(f"{type(exc).__name__}: {str(exc)[:120]}")
        finally:
            out, err = proc.communicate(timeout=120)

        self.assertEqual(proc.returncode, 0, f"writer failed: {err}")
        self.assertGreater(reads, 0, "reader never managed a read")
        self.assertEqual(bad, [], f"reader saw {len(bad)} bad states, e.g. {bad[:3]}")

    def test_concurrent_update_index_does_not_lose_vectors(self):
        dim, per_proc, procs = 32, 40, 4  # 4 default-queue workers
        procs_running = [
            self._run(_UPDATER, [self._tmp.name, 100 + i, per_proc, dim])
            for i in range(procs)
        ]
        for proc in procs_running:
            out, err = proc.communicate(timeout=120)
            self.assertEqual(proc.returncode, 0, f"updater failed: {err}")

        index = self.manager.load_index("shared")
        self.assertIsNotNone(index)
        self.assertEqual(
            index.ntotal,
            per_proc * procs,
            "lost update: concurrent update_index calls clobbered each other",
        )


if __name__ == "__main__":
    unittest.main()
