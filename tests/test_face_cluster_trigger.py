"""
Tests for the post-upload face-clustering trigger.

Regression cover for the frozen-clustering bug: the upload path used
`queue.enqueue_in(timedelta(seconds=30), ...)`, but no `rq scheduler` daemon
runs in this deployment and `worker.py` builds its Worker without
`with_scheduler=True`. Nothing ever dequeued `rq:scheduled:default`, so 1,485
clustering jobs accumulated there over nine months and 1,532 of 2,081 faces
were never assigned to a person.

The fix gates clustering on the batch's own face-detection jobs with RQ's
`depends_on`, which needs no scheduler and — unlike a fixed delay — cannot
race a batch that takes longer than the delay to detect.
"""
import unittest

from rq.job import Dependency, Job

import app_fastapi
from core.jobs import FACE_MATCH_THRESHOLD, cluster_faces_job


class FakeRedis:
    """Placeholder connection. Nothing here reaches Redis."""


def FakeJob(job_id):
    """A real rq.job.Job with no live connection.

    Deliberately a genuine Job rather than a duck-typed stub: RQ's Dependency
    validates its members and only accepts Job instances or job-id strings, so
    a stub would let the production code pass a test it would fail in Redis.
    """
    return Job(id=job_id, connection=FakeRedis())


class RecordingQueue:
    """Queue double that records enqueue calls instead of touching Redis."""

    def __init__(self):
        self.enqueue_calls = []
        self.enqueue_in_calls = []

    def enqueue(self, func, *args, **kwargs):
        self.enqueue_calls.append({"func": func, "args": args, "kwargs": kwargs})
        return FakeJob(f"job-{len(self.enqueue_calls)}")

    def enqueue_in(self, *args, **kwargs):
        # The bug. Recorded rather than raised so the assertion message is
        # about the scheduler dependency, not an opaque AttributeError.
        self.enqueue_in_calls.append({"args": args, "kwargs": kwargs})
        return FakeJob("scheduled")


class TestClusterTriggerDependsOnFaceJobs(unittest.TestCase):
    """The clustering job must be gated on face detection, not on a clock."""

    def setUp(self):
        self.queue = RecordingQueue()
        self.face_jobs = [FakeJob("face-1"), FakeJob("face-2")]

    def test_enqueues_clustering_job(self):
        """A batch with photos queues exactly one clustering job."""
        app_fastapi._enqueue_cluster_after_faces(
            self.queue, [1, 2], self.face_jobs
        )

        self.assertEqual(len(self.queue.enqueue_calls), 1)
        self.assertIs(self.queue.enqueue_calls[0]["func"], cluster_faces_job)

    def test_never_uses_delayed_scheduling(self):
        """`enqueue_in` needs a scheduler daemon this deployment does not run."""
        app_fastapi._enqueue_cluster_after_faces(
            self.queue, [1, 2], self.face_jobs
        )

        self.assertEqual(
            self.queue.enqueue_in_calls,
            [],
            "clustering must not be scheduled for later — nothing dequeues "
            "rq:scheduled:default here",
        )

    def test_depends_on_every_face_job_in_the_batch(self):
        """Clustering waits for all of this batch's face-detection jobs."""
        app_fastapi._enqueue_cluster_after_faces(
            self.queue, [1, 2], self.face_jobs
        )

        depends_on = self.queue.enqueue_calls[0]["kwargs"].get("depends_on")
        self.assertIsInstance(depends_on, Dependency)
        self.assertEqual(list(depends_on.dependencies), self.face_jobs)

    def test_dependency_allows_failure(self):
        """One crashed face job must not strand clustering as deferred forever."""
        app_fastapi._enqueue_cluster_after_faces(
            self.queue, [1, 2], self.face_jobs
        )

        depends_on = self.queue.enqueue_calls[0]["kwargs"].get("depends_on")
        self.assertTrue(depends_on.allow_failure)

    def test_passes_photo_ids_to_scope_the_pass(self):
        """Clustering is scoped to the uploaded photos, not the whole table."""
        app_fastapi._enqueue_cluster_after_faces(
            self.queue, [7, 8, 9], self.face_jobs
        )

        args = self.queue.enqueue_calls[0]["args"]
        self.assertIn([7, 8, 9], args)

    def test_never_resets_existing_assignments(self):
        """The upload path must never pass reset=True — it wipes manual labels."""
        app_fastapi._enqueue_cluster_after_faces(
            self.queue, [1, 2], self.face_jobs
        )

        call = self.queue.enqueue_calls[0]
        self.assertNotIn(True, call["args"][1:], "reset must stay False")
        self.assertFalse(call["kwargs"].get("reset", False))

    def test_no_photos_enqueues_nothing(self):
        """An upload batch of only videos or duplicates queues no clustering."""
        app_fastapi._enqueue_cluster_after_faces(self.queue, [], self.face_jobs)

        self.assertEqual(self.queue.enqueue_calls, [])
        self.assertEqual(self.queue.enqueue_in_calls, [])

    def test_no_face_jobs_still_clusters_without_dependency(self):
        """If face jobs failed to enqueue, clustering still runs, ungated."""
        app_fastapi._enqueue_cluster_after_faces(self.queue, [1, 2], [])

        self.assertEqual(len(self.queue.enqueue_calls), 1)
        self.assertIsNone(self.queue.enqueue_calls[0]["kwargs"].get("depends_on"))


class TestUnifiedFaceMatchThreshold(unittest.TestCase):
    """One threshold, one definition — see .claude/rules/ml-pipeline.md."""

    def test_trigger_uses_the_shared_constant(self):
        """The upload path must not carry its own hardcoded 0.6."""
        queue = RecordingQueue()
        app_fastapi._enqueue_cluster_after_faces(queue, [1], [FakeJob("f")])

        args = queue.enqueue_calls[0]["args"]
        self.assertIn(FACE_MATCH_THRESHOLD, args)

    def test_auto_match_defaults_to_the_shared_constant(self):
        """`_auto_match_face_to_person` must not keep its own 0.75."""
        import inspect

        from core.jobs import _auto_match_face_to_person

        default = inspect.signature(
            _auto_match_face_to_person
        ).parameters["threshold"].default
        self.assertEqual(default, FACE_MATCH_THRESHOLD)

    def test_cluster_job_defaults_to_the_shared_constant(self):
        """`cluster_faces_job` must not keep its own 0.75."""
        import inspect

        default = inspect.signature(
            cluster_faces_job
        ).parameters["threshold"].default
        self.assertEqual(default, FACE_MATCH_THRESHOLD)

    def test_face_detection_jobs_use_the_shared_constant(self):
        """The two in-job `_auto_match_face_to_person` calls must not hardcode."""
        import re
        from pathlib import Path

        source = Path(app_fastapi.__file__).with_name("core").joinpath("jobs.py")
        hardcoded = re.findall(
            r"_auto_match_face_to_person\([^)]*threshold\s*=\s*([0-9.]+)",
            source.read_text(),
        )
        self.assertEqual(
            hardcoded, [], f"hardcoded thresholds at call sites: {hardcoded}"
        )

    def test_recluster_script_uses_the_shared_constant(self):
        """`recluster_all.py` must not keep a fourth copy of the value."""
        import re
        from pathlib import Path

        source = Path(app_fastapi.__file__).with_name("recluster_all.py")
        self.assertNotRegex(
            source.read_text(),
            r"^\s*threshold\s*=\s*0\.\d+\s*$",
            "recluster_all.py should reference FACE_MATCH_THRESHOLD",
        )

    def test_threshold_separates_the_labelled_population(self):
        """
        The chosen value must sit between the two observed distributions.

        Measured over the 549 faces assigned to the 8 named persons in the
        live DB (see the FACE_MATCH_THRESHOLD comment for the derivation):

          MAX_IMPOSTOR   the highest similarity ever seen between a pair of
                         faces belonging to *different* people. Below this,
                         Phase 1 starts merging people.
          MEDIAN_COHESION  the median per-person average pairwise similarity.
                         Above this, the Phase-2 acceptance gate
                         (avg_similarity >= threshold) rejects more than half
                         of the people who actually exist in this library —
                         which is what 0.75 and 0.78 were doing.
        """
        MAX_IMPOSTOR = 0.366
        MEDIAN_COHESION = 0.732

        self.assertGreater(FACE_MATCH_THRESHOLD, MAX_IMPOSTOR)
        self.assertLess(FACE_MATCH_THRESHOLD, MEDIAN_COHESION)


if __name__ == "__main__":
    unittest.main()
