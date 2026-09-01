"""
Tests for the face-matching half of core/jobs.py.

Two defects:

**cluster_faces_job reported a number that was not true.**
``total_clustered = matched_count + len(unmatched_indices)`` counted every face
that went *into* HDBSCAN, including the ones it left as noise and never
assigned. The catch-up run reported ``{'clustered': 1532}`` while assigning
495. That is the same class of metric that hid the original dead-scheduler
failure — a broken pipeline reporting success — so it gets a test.

**_auto_match_face_to_person rebuilt a full index per face.** For every
detected face it selected every person-assigned embedding from SQLite and built
a brand-new IndexFlatIP: O(N) embedding reads plus a full index build per face,
O(N*M) for a batch import. The persistent index exists to avoid exactly this.

Both are guarded here against a fixed, hand-built database — no MinIO, no
models, no production data.
"""
import os
import re
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import faiss
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import db
import core.faiss_index as faiss_index_mod
from core.faiss_index import FAISSIndexManager, is_id_mapped, index_ids

DIM = 128
PERSON_INDEX = "existing_person_faces"
ENV_INDEX_DIR = "CHITRA_FAISS_INDEX_DIR"


def unit(vec) -> np.ndarray:
    vec = np.asarray(vec, dtype="float32")
    return vec / (np.linalg.norm(vec) + 1e-12)


def direction(i: int) -> np.ndarray:
    """A distinct, well-separated unit direction (a basis vector)."""
    v = np.zeros(DIM, dtype="float32")
    v[i % DIM] = 1.0
    return v


def near(base: np.ndarray, rng, jitter=0.03) -> np.ndarray:
    """A vector tightly clustered around `base` (cosine ~0.999)."""
    return unit(base + jitter * rng.standard_normal(DIM).astype("float32"))


def mix(a: np.ndarray, b: np.ndarray, cos_to_a: float) -> np.ndarray:
    """Unit vector whose cosine with orthonormal `a` is exactly `cos_to_a`."""
    return unit(cos_to_a * a + np.sqrt(max(0.0, 1.0 - cos_to_a ** 2)) * b)


class FaceDBFixture(unittest.TestCase):
    """A throwaway SQLite database plus a throwaway index directory."""

    def setUp(self):
        self._env = os.environ.get(ENV_INDEX_DIR)
        self._tmp = tempfile.TemporaryDirectory()
        self.index_dir = Path(self._tmp.name) / "faiss_indexes"
        os.environ[ENV_INDEX_DIR] = str(self.index_dir)
        faiss_index_mod._reset_logged_dirs_for_tests()

        self.db_path = str(Path(self._tmp.name) / "faces.db")
        db.init_db(self.db_path)
        self.conn = db.connect(self.db_path)
        self._next_photo = 0

    def tearDown(self):
        self.conn.close()
        self._tmp.cleanup()
        if self._env is None:
            os.environ.pop(ENV_INDEX_DIR, None)
        else:
            os.environ[ENV_INDEX_DIR] = self._env

    # -- fixture helpers ------------------------------------------------

    def add_person(self, name: str) -> int:
        cur = self.conn.cursor()
        cur.execute("INSERT INTO persons (name) VALUES (?)", (name,))
        self.conn.commit()
        return cur.lastrowid

    def add_face(self, vec: np.ndarray, person_id=None) -> int:
        """One face in its own photo, so photo_id never collides."""
        self._next_photo += 1
        pid = self._next_photo
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO photos (id, file_path, media_type) VALUES (?, ?, 'image')",
            (pid, f"obj/{pid}.jpg"),
        )
        cur.execute(
            "INSERT INTO faces (photo_id, face_index, embedding, person_id) "
            "VALUES (?, 0, ?, ?)",
            (pid, unit(vec).tobytes(), person_id),
        )
        self.conn.commit()
        return cur.lastrowid

    def assigned_count(self) -> int:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM faces WHERE person_id IS NOT NULL")
        return cur.fetchone()[0]

    def person_of(self, face_id: int):
        cur = self.conn.cursor()
        cur.execute("SELECT person_id FROM faces WHERE id=?", (face_id,))
        return cur.fetchone()[0]

    def manager(self) -> FAISSIndexManager:
        return FAISSIndexManager()


class ClusteringResultTests(FaceDBFixture):
    """Fix 1: the job must report what it actually assigned."""

    def test_reported_count_matches_faces_actually_assigned(self):
        from core.jobs import cluster_faces_job

        rng = np.random.default_rng(11)
        # One tight group of 10 that must cluster...
        for _ in range(10):
            self.add_face(near(direction(0), rng))
        # ...and 20 mutually distant faces HDBSCAN must leave as noise.
        for i in range(20):
            self.add_face(direction(5 + i))

        result = cluster_faces_job(self.db_path)

        assigned = self.assigned_count()
        self.assertGreater(assigned, 0, "nothing clustered — fixture is wrong")
        self.assertGreater(
            30 - assigned, 0, "everything clustered — fixture does not test noise"
        )
        self.assertEqual(
            result["clustered"], assigned,
            "'clustered' must be the number of faces actually assigned",
        )

    def test_the_three_outcomes_are_distinguishable(self):
        from core.jobs import cluster_faces_job

        rng = np.random.default_rng(12)
        # An existing person, so Phase 1 has something to match against.
        alice = self.add_person("alice")
        for _ in range(4):
            self.add_face(near(direction(0), rng), person_id=alice)
        # Unassigned faces that should match alice.
        expect_matched = [self.add_face(near(direction(0), rng)) for _ in range(3)]
        # A fresh tight group that should become a new person.
        for _ in range(6):
            self.add_face(near(direction(1), rng))
        # Noise.
        for i in range(15):
            self.add_face(direction(10 + i))

        result = cluster_faces_job(self.db_path)

        for key in ("clustered", "persons_created", "matched_to_existing",
                    "newly_clustered", "left_as_noise"):
            self.assertIn(key, result, f"missing result key {key!r}")

        self.assertEqual(
            result["matched_to_existing"] + result["newly_clustered"],
            result["clustered"],
            "the outcome breakdown must add up to the total",
        )
        self.assertGreater(result["matched_to_existing"], 0)
        self.assertGreater(result["newly_clustered"], 0)
        self.assertGreater(result["left_as_noise"], 0)

        # left_as_noise is the faces that went into HDBSCAN and came out
        # unassigned — the number the old total silently folded into success.
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM faces WHERE person_id IS NULL")
        self.assertEqual(result["left_as_noise"], cur.fetchone()[0])

        for face_id in expect_matched:
            self.assertEqual(self.person_of(face_id), alice)

    def test_a_batch_too_small_for_hdbscan_is_reported_as_noise_not_a_crash(self):
        """
        Found in the live worker logs, not in this brief: HDBSCAN asks its
        kd-tree for ``min_samples + 1`` neighbours, so a batch of one unmatched
        face raises ``ValueError: k must be less than or equal to the number of
        training points`` and the whole job fails. Six such failures were
        logged in half an hour — every upload that produced a single
        unrecognised face. One face cannot be a cluster; it is noise, and the
        job must say so.
        """
        from core.jobs import cluster_faces_job

        self.add_face(direction(3))

        result = cluster_faces_job(self.db_path)

        self.assertEqual(result["clustered"], 0)
        self.assertEqual(result["left_as_noise"], 1)
        self.assertEqual(self.assigned_count(), 0)

    def test_nothing_to_do_reports_zero_for_every_outcome(self):
        from core.jobs import cluster_faces_job

        result = cluster_faces_job(self.db_path)
        self.assertEqual(result["clustered"], 0)
        self.assertEqual(result["persons_created"], 0)


class AutoMatchUsesPersistentIndexTests(FaceDBFixture):
    """Fix 2: per-face matching must go through the persistent index."""

    def _seed_person(self, name, base, n, rng):
        person_id = self.add_person(name)
        for _ in range(n):
            self.add_face(near(base, rng), person_id=person_id)
        return person_id

    def _sql_trace(self):
        seen = []
        self.conn.set_trace_callback(seen.append)
        return seen

    def test_warm_path_cost_does_not_grow_with_the_number_of_assigned_faces(self):
        """The O(N)-per-face read is the defect; measure N-independence."""
        from core.jobs import _auto_match_face_to_person

        rng = np.random.default_rng(21)
        self._seed_person("alice", direction(0), 40, rng)

        # Warm the persistent index.
        _auto_match_face_to_person(
            self.conn, self.add_face(near(direction(0), rng)),
            near(direction(0), rng)
        )

        def statements_for_one_match():
            face_id = self.add_face(near(direction(0), rng))
            seen = self._sql_trace()
            try:
                _auto_match_face_to_person(
                    self.conn, face_id, near(direction(0), rng)
                )
            finally:
                self.conn.set_trace_callback(None)
            return seen

        small = statements_for_one_match()

        # Grow the assigned set by 4x, then measure again.
        self._seed_person("bob", direction(1), 160, rng)
        _auto_match_face_to_person(
            self.conn, self.add_face(near(direction(0), rng)),
            near(direction(0), rng)
        )  # let the index catch up with the new rows
        large = statements_for_one_match()

        self.assertEqual(
            len(small), len(large),
            f"per-face SQL grew with N: {len(small)} -> {len(large)}\n"
            f"small={small}\nlarge={large}",
        )

    def test_warm_path_does_not_read_every_assigned_embedding(self):
        from core.jobs import _auto_match_face_to_person

        rng = np.random.default_rng(22)
        self._seed_person("alice", direction(0), 60, rng)
        _auto_match_face_to_person(
            self.conn, self.add_face(near(direction(0), rng)),
            near(direction(0), rng)
        )  # warm

        face_id = self.add_face(near(direction(0), rng))
        seen = self._sql_trace()
        try:
            _auto_match_face_to_person(self.conn, face_id, near(direction(0), rng))
        finally:
            self.conn.set_trace_callback(None)

        full_scan = re.compile(
            r"FROM\s+faces\s+f\s+JOIN\s+persons", re.IGNORECASE | re.DOTALL
        )
        offenders = [s for s in seen if full_scan.search(s)]
        self.assertEqual(
            offenders, [],
            f"still scanning every assigned embedding per face: {offenders}",
        )

    def test_warm_path_does_not_rebuild_the_index_per_face(self):
        from core.jobs import _auto_match_face_to_person

        rng = np.random.default_rng(23)
        self._seed_person("alice", direction(0), 40, rng)
        _auto_match_face_to_person(
            self.conn, self.add_face(near(direction(0), rng)),
            near(direction(0), rng)
        )  # warm

        real_build = FAISSIndexManager.build_hnsw_index
        with mock.patch.object(
            FAISSIndexManager, "build_hnsw_index", autospec=True,
            side_effect=real_build,
        ) as build_spy:
            for _ in range(5):
                face_id = self.add_face(near(direction(0), rng))
                _auto_match_face_to_person(
                    self.conn, face_id, near(direction(0), rng)
                )

        self.assertEqual(
            build_spy.call_count, 0,
            "rebuilt the whole index on a warm path — that is the defect",
        )

    def test_it_still_matches_the_obvious_case(self):
        from core.jobs import _auto_match_face_to_person

        rng = np.random.default_rng(24)
        alice = self._seed_person("alice", direction(0), 5, rng)
        self._seed_person("bob", direction(1), 5, rng)

        vec = near(direction(0), rng)
        face_id = self.add_face(vec)
        _auto_match_face_to_person(self.conn, face_id, vec)
        self.assertEqual(self.person_of(face_id), alice)

    def test_a_stranger_is_left_unassigned(self):
        from core.jobs import _auto_match_face_to_person

        rng = np.random.default_rng(25)
        self._seed_person("alice", direction(0), 5, rng)

        vec = direction(50)
        face_id = self.add_face(vec)
        _auto_match_face_to_person(self.conn, face_id, vec)
        self.assertIsNone(self.person_of(face_id))

    def test_a_face_assigned_earlier_in_the_same_batch_is_visible(self):
        """
        The staleness trap. face_b matches alice only *through* face_a, which
        was assigned moments earlier in the same batch. If the persistent index
        is not kept in step, matching silently degrades within a run.
        """
        from core.jobs import _auto_match_face_to_person

        a0, a1 = direction(0), direction(1)
        alice = self.add_person("alice")
        self.add_face(a0, person_id=alice)

        # cos(v_a, a0) = 0.65 -> matches alice directly.
        v_a = mix(a0, a1, 0.65)
        # cos(v_b, a0) = 0.40 -> below threshold, cannot match alice directly.
        v_b = mix(a0, a1, 0.40)
        self.assertGreater(float(v_a @ v_b), 0.60)   # but is close to v_a
        self.assertLess(float(v_b @ a0), 0.60)

        face_a = self.add_face(v_a)
        _auto_match_face_to_person(self.conn, face_a, v_a)
        self.assertEqual(self.person_of(face_a), alice, "setup: face_a must match")

        face_b = self.add_face(v_b)
        _auto_match_face_to_person(self.conn, face_b, v_b)
        self.assertEqual(
            self.person_of(face_b), alice,
            "face assigned earlier in the same batch was invisible to the next "
            "face — same-batch staleness",
        )

    def test_the_persistent_index_is_id_mapped_and_tracks_the_database(self):
        from core.jobs import _auto_match_face_to_person

        rng = np.random.default_rng(26)
        self._seed_person("alice", direction(0), 6, rng)
        vec = near(direction(0), rng)
        face_id = self.add_face(vec)
        _auto_match_face_to_person(self.conn, face_id, vec)

        index = self.manager().load_index(PERSON_INDEX)
        self.assertIsNotNone(index, "persistent index was never written")
        self.assertTrue(
            is_id_mapped(index),
            "index carries no id mapping — position i is not row i of anything",
        )
        cur = self.conn.cursor()
        cur.execute("SELECT id FROM faces WHERE person_id IS NOT NULL ORDER BY id")
        self.assertEqual(sorted(index_ids(index)), [r[0] for r in cur.fetchall()])


class IndexMappingHonestyTests(FaceDBFixture):
    """
    Regression for the bug the live index was already carrying.

    ``existing_person_face_ids[position]`` assumes the index was built from the
    same unordered query, in the same order, and never appended to. The live
    index had 715 of 1,044 positions (68.5%) pointing at the wrong person while
    still passing the ntotal count check.
    """

    def _two_people(self):
        rng = np.random.default_rng(31)
        alice = self.add_person("alice")
        bob = self.add_person("bob")
        alice_faces = [self.add_face(near(direction(0), rng), person_id=alice)
                       for _ in range(5)]
        bob_faces = [self.add_face(near(direction(1), rng), person_id=bob)
                     for _ in range(5)]
        return alice, bob, alice_faces, bob_faces, rng

    def _write_index(self, face_ids, ids=None):
        """Persist an index over `face_ids` in exactly that order."""
        cur = self.conn.cursor()
        vecs = []
        for fid in face_ids:
            cur.execute("SELECT embedding FROM faces WHERE id=?", (fid,))
            vecs.append(np.frombuffer(cur.fetchone()[0], dtype=np.float32))
        xb = np.stack(vecs).astype("float32")
        if ids is None:
            faiss.normalize_L2(xb)
            index = faiss.IndexHNSWFlat(DIM, 32)  # legacy: no id map, L2
            index.add(xb)
            faiss.write_index(index, str(self.manager().get_index_path(PERSON_INDEX)))
        else:
            self.manager().build_hnsw_index(xb, PERSON_INDEX, ids=ids)

    def test_scrambled_id_mapped_index_still_attributes_correctly(self):
        from core.jobs import _auto_match_face_to_person

        alice, bob, alice_faces, bob_faces, rng = self._two_people()
        # Index order deliberately unlike any ORDER BY: bob first, interleaved.
        order = [bob_faces[0], alice_faces[3], bob_faces[2], alice_faces[0],
                 bob_faces[4], alice_faces[1], bob_faces[1], alice_faces[4],
                 bob_faces[3], alice_faces[2]]
        self._write_index(order, ids=order)

        vec = near(direction(0), rng)
        face_id = self.add_face(vec)
        _auto_match_face_to_person(self.conn, face_id, vec)
        self.assertEqual(
            self.person_of(face_id), alice,
            "attributed to the wrong person: index positions were trusted as "
            "query row numbers",
        )

    def test_legacy_index_without_an_id_map_is_rebuilt_not_trusted(self):
        from core.jobs import _auto_match_face_to_person

        alice, bob, alice_faces, bob_faces, rng = self._two_people()
        # A legacy index whose count matches the DB exactly, so the
        # count-differs check says "fresh", but whose order is scrambled.
        self._write_index(bob_faces + alice_faces)
        legacy = self.manager().load_index(PERSON_INDEX)
        self.assertFalse(is_id_mapped(legacy), "setup: must be a legacy index")
        self.assertEqual(legacy.ntotal, 10, "setup: count must match the DB")

        vec = near(direction(0), rng)
        face_id = self.add_face(vec)
        _auto_match_face_to_person(self.conn, face_id, vec)

        self.assertEqual(self.person_of(face_id), alice)
        self.assertTrue(
            is_id_mapped(self.manager().load_index(PERSON_INDEX)),
            "a legacy index must be rebuilt with an id map, not trusted",
        )

    def test_cluster_job_attributes_correctly_with_a_scrambled_index(self):
        from core.jobs import cluster_faces_job

        alice, bob, alice_faces, bob_faces, rng = self._two_people()
        self._write_index(bob_faces + alice_faces)  # legacy, count matches

        new_alice = [self.add_face(near(direction(0), rng)) for _ in range(3)]
        new_bob = [self.add_face(near(direction(1), rng)) for _ in range(3)]

        result = cluster_faces_job(self.db_path)

        self.assertEqual(result["clustered"], self.assigned_count() - 10)
        for face_id in new_alice:
            self.assertEqual(self.person_of(face_id), alice, f"face {face_id}")
        for face_id in new_bob:
            self.assertEqual(self.person_of(face_id), bob, f"face {face_id}")

    def test_a_close_match_is_not_dropped_by_the_metric_conversion(self):
        """
        A single-query search over a squared-L2 index returns ~0.3 for a good
        match. The old batch-wide "if max > 1.0 it must be L2" guess then did
        no conversion and compared 0.3 against a 0.60 cosine threshold — the
        good match was rejected. Nothing may depend on batch composition.
        """
        from core.jobs import _auto_match_face_to_person

        rng = np.random.default_rng(33)
        alice = self.add_person("alice")
        for _ in range(4):
            self.add_face(near(direction(0), rng, jitter=0.01), person_id=alice)

        vec = near(direction(0), rng, jitter=0.01)  # cosine ~0.999
        face_id = self.add_face(vec)
        _auto_match_face_to_person(self.conn, face_id, vec)
        self.assertEqual(self.person_of(face_id), alice)


if __name__ == "__main__":
    unittest.main()
