#!/usr/bin/env python3
"""
Reclaim video transcodes that died and left their row stuck in `processing`.

Background: `process_video_transcode_job` sets `transcode_status="processing"`
on entry and only ever leaves that state from inside its own try/except. When
the RQ work-horse is SIGTERM'd mid-ffmpeg — the `stop_workers.sh` double-TERM
hazard in AGENTS.md — that handler never runs. RQ marks the *job* failed, but
the *row* keeps saying `processing` forever. Nothing retries it, nothing times
it out, and the video is permanently unplayable: `/api/photos/{id}/video`
returns 409 `transcode_in_progress` for a transcode that stopped existing
months ago. Before this script there was no requeue path at all.

Not every stuck row needs the same repair, and the difference is expensive to
get wrong — re-encoding 4K HEVC on this box is minutes of CPU:

    (a) the playback derivative is already in MinIO. The ffmpeg work finished
        and only the final status write was lost to the signal. Correct the
        status to `ready`; do NOT re-transcode.
    (b) the original object is gone from MinIO. Retrying can never succeed, so
        mark it `failed` with a reason rather than looping forever.
    (c) everything else. Reset to `pending` and enqueue the real job on the
        `video` queue.

Safety interlock: a row whose photo id has a live RQ job behind it (queued or
started) is never touched, whatever the flags say. That check — not the age
threshold — is what actually distinguishes a dead transcode from a running one.
`--min-age-hours` is a secondary guard and deliberately conservative: the only
timestamp on the row is `created_at`, which is the *capture* date, not the row
insert time, so a freshly uploaded old video looks ancient to it. Trust the
queue check; treat the age threshold as belt and braces.

Dry run by default. Nothing is written and nothing is enqueued without
`--apply`. Idempotent: a requeued row becomes `pending`, which `--stuck` no
longer selects, so a second run finds nothing.

Usage:
    # what would happen to every stuck row
    python scripts/requeue_transcodes.py --stuck

    # actually repair two of them (batch size matters: 2 video workers,
    # ~2 cores each, under a 400% CPUQuota)
    python scripts/requeue_transcodes.py --stuck --limit 2 --apply

    # target specific ids regardless of their current status
    python scripts/requeue_transcodes.py --ids 1226,1623 --apply

Run it from the repo root with the environment the workers use, so that the
MinIO credentials and DB path match production:

    set -a; . ./.env.production; set +a
    .venv/bin/python scripts/requeue_transcodes.py --stuck
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from datetime import datetime, timedelta
from typing import Any, Iterable, List, NamedTuple, Optional, Sequence, Set

# Allow running as a plain script from the repo root as well as being imported
# as `scripts.requeue_transcodes` by the test suite.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DB_DEFAULT_PATH = os.environ.get("CHITRA_DB_PATH", "photo.db")

# The status a killed job leaves behind.
STATUS_STUCK = "processing"

ACTION_READY = "ready"
ACTION_FAILED = "failed"
ACTION_REQUEUE = "requeue"
ACTION_SKIP = "skip"

# Matches the enqueue in app_fastapi.py's upload path.
JOB_TIMEOUT = "2h"

_SELECT_COLUMNS = (
    "id, file_path, created_at, media_type, playback_path, transcode_status"
)


class Decision(NamedTuple):
    """What to do with one row, and why."""

    photo_id: int
    action: str
    reason: str
    file_path: Optional[str] = None
    playback_key: Optional[str] = None


def _playback_size(storage: Any, key: str) -> Optional[int]:
    """Size of an object in bytes, or None if it is not there."""
    try:
        return storage.stat_object_size(key)
    except FileNotFoundError:
        return None


def classify_row(row: Any, storage: Any, active_ids: Set[int]) -> Decision:
    """Decide which population a row belongs to.

    Pure apart from two storage lookups, so it is testable against a stub. The
    order of the checks is deliberate: the playback object is checked before
    the original, because a video whose derivative exists is playable even if
    the source has since been deleted.
    """
    photo_id = row["id"]
    file_path = row["file_path"]

    if photo_id in active_ids:
        return Decision(
            photo_id, ACTION_SKIP,
            "an active RQ job is already working on it", file_path,
        )

    if (row["media_type"] or "photo") != "video":
        return Decision(
            photo_id, ACTION_SKIP,
            f"not a video (media_type={row['media_type']!r})", file_path,
        )

    # (a) Did the ffmpeg work actually finish?
    playback_key = row["playback_path"] or storage.generate_playback_path(photo_id)
    size = _playback_size(storage, playback_key)
    if size:
        # Non-zero: a SIGTERM mid-upload can leave a 0-byte object behind, and
        # that is not a finished transcode.
        return Decision(
            photo_id, ACTION_READY,
            f"playback object already exists ({size} bytes)",
            file_path, playback_key,
        )

    # (b) Is the source still there to work from?
    if not file_path or _playback_size(storage, file_path) is None:
        return Decision(
            photo_id, ACTION_FAILED,
            "original object is missing from MinIO; retrying cannot help",
            file_path,
        )

    # (c) Genuinely needs the work.
    return Decision(
        photo_id, ACTION_REQUEUE, "original present, no playback derivative",
        file_path,
    )


def _older_than(created_at: Optional[str], cutoff: datetime) -> bool:
    """True if `created_at` parses and is before the cutoff.

    An unparseable or absent timestamp counts as old: these rows predate the
    problem being noticed and holding them back helps nobody.
    """
    if not created_at:
        return True
    try:
        return datetime.fromisoformat(created_at) < cutoff
    except ValueError:
        return True


def select_rows(
    conn: sqlite3.Connection,
    ids: Optional[Sequence[int]] = None,
    stuck: bool = False,
    limit: Optional[int] = None,
    min_age_hours: float = 0.0,
    now: Optional[datetime] = None,
) -> List[Any]:
    """Candidate rows, oldest id first.

    Explicit `ids` win over the status filter — targeting a row by id is a
    deliberate act, and the whole point may be to repair a `failed` one.
    """
    if ids:
        placeholders = ",".join("?" for _ in ids)
        sql = (
            f"SELECT {_SELECT_COLUMNS} FROM photos "
            f"WHERE id IN ({placeholders}) ORDER BY id"
        )
        params: List[Any] = list(ids)
    elif stuck:
        sql = (
            f"SELECT {_SELECT_COLUMNS} FROM photos "
            f"WHERE transcode_status=? AND media_type='video' ORDER BY id"
        )
        params = [STATUS_STUCK]
    else:
        return []

    rows = conn.execute(sql, params).fetchall()

    if stuck and min_age_hours > 0:
        cutoff = (now or datetime.now()) - timedelta(hours=min_age_hours)
        rows = [r for r in rows if _older_than(r["created_at"], cutoff)]

    if limit is not None:
        rows = rows[:limit]
    return rows


def find_active_photo_ids(queue: Any) -> Set[int]:
    """Photo ids with a job queued or currently running on the video queue.

    This is the real liveness signal. A row in `processing` with nothing here
    behind it is dead by definition.
    """
    from rq.job import Job
    from rq.registry import StartedJobRegistry

    active: Set[int] = set()
    job_ids: List[str] = list(queue.get_job_ids())
    job_ids += StartedJobRegistry(queue.name, connection=queue.connection).get_job_ids()

    for job_id in job_ids:
        try:
            job = Job.fetch(job_id, connection=queue.connection)
        except Exception:
            continue
        if job.args:
            try:
                active.add(int(job.args[0]))
            except (TypeError, ValueError):
                pass
    return active


def apply_decisions(
    conn: sqlite3.Connection,
    decisions: Iterable[Decision],
    queue: Any,
    db_path: str,
    dry_run: bool = True,
) -> dict:
    """Carry out the decisions. A dry run writes nothing and enqueues nothing."""
    from core.jobs import process_video_transcode_job

    counts = {ACTION_READY: 0, ACTION_FAILED: 0, ACTION_REQUEUE: 0, ACTION_SKIP: 0}

    for decision in decisions:
        counts[decision.action] += 1
        verb = "would" if dry_run else ""

        if decision.action == ACTION_SKIP:
            print(f"  [{decision.photo_id}] skip — {decision.reason}")
            continue

        if decision.action == ACTION_READY:
            print(
                f"  [{decision.photo_id}] {verb} correct status -> ready "
                f"— {decision.reason}"
            )
            if not dry_run:
                conn.execute(
                    "UPDATE photos SET transcode_status='ready', playback_path=? "
                    "WHERE id=?",
                    (decision.playback_key, decision.photo_id),
                )
                conn.commit()

        elif decision.action == ACTION_FAILED:
            print(f"  [{decision.photo_id}] {verb} mark failed — {decision.reason}")
            if not dry_run:
                conn.execute(
                    "UPDATE photos SET transcode_status='failed' WHERE id=?",
                    (decision.photo_id,),
                )
                conn.commit()

        elif decision.action == ACTION_REQUEUE:
            print(
                f"  [{decision.photo_id}] {verb} requeue -> pending "
                f"— {decision.reason}"
            )
            if not dry_run:
                # Status first: if the enqueue throws, the row is left in
                # `pending`, which is honest (nothing is working on it) and
                # selectable by a later --ids run. Leaving it in `processing`
                # would recreate exactly the bug this script exists to fix.
                conn.execute(
                    "UPDATE photos SET transcode_status='pending' WHERE id=?",
                    (decision.photo_id,),
                )
                conn.commit()
                queue.enqueue(
                    process_video_transcode_job,
                    decision.photo_id,
                    decision.file_path,
                    db_path,
                    job_timeout=JOB_TIMEOUT,
                )

    return counts


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reclaim video transcodes stuck in 'processing'.",
    )
    parser.add_argument("--db", default=DB_DEFAULT_PATH,
                        help=f"database to operate on (default: {DB_DEFAULT_PATH})")
    parser.add_argument("--ids", help="comma-separated photo ids to target")
    parser.add_argument("--stuck", action="store_true",
                        help=f"target every video row in '{STATUS_STUCK}'")
    parser.add_argument("--limit", type=int,
                        help="process at most N rows (keep transcode batches small)")
    parser.add_argument("--min-age-hours", type=float, default=0.0,
                        help="with --stuck, only rows whose created_at is older "
                             "than this. Secondary guard; the live-job check is "
                             "the authoritative one (default: 0, off)")
    parser.add_argument("--apply", action="store_true",
                        help="actually mutate. Without this the script only reports.")
    args = parser.parse_args(argv)

    ids: Optional[List[int]] = None
    if args.ids:
        try:
            ids = [int(x) for x in args.ids.split(",") if x.strip()]
        except ValueError:
            print("error: --ids must be comma-separated integers", file=sys.stderr)
            return 2

    if not ids and not args.stuck:
        print("error: pass --ids or --stuck", file=sys.stderr)
        return 2

    if not os.path.exists(args.db):
        print(f"error: no such database: {args.db}", file=sys.stderr)
        return 2

    mode = "APPLY (rows will change)" if args.apply else "DRY RUN (no changes)"
    print(f"Database: {args.db}")
    print(f"Mode:     {mode}\n")

    from core.storage_client import MinIOStorageClient
    from core.worker import get_queue

    storage = MinIOStorageClient()
    queue = get_queue("video")
    active_ids = find_active_photo_ids(queue)
    if active_ids:
        print(f"Live video jobs for photo ids: {sorted(active_ids)} — leaving alone\n")

    conn = sqlite3.connect(args.db, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        rows = select_rows(
            conn, ids=ids, stuck=args.stuck, limit=args.limit,
            min_age_hours=args.min_age_hours,
        )
        if not rows:
            print("Nothing to do — no matching rows.")
            return 0

        print(f"{len(rows)} row(s) selected:")
        decisions = [classify_row(r, storage, active_ids) for r in rows]
        counts = apply_decisions(conn, decisions, queue, args.db, dry_run=not args.apply)

        print()
        print(f"  status corrected to ready: {counts[ACTION_READY]}")
        print(f"  marked failed:             {counts[ACTION_FAILED]}")
        print(f"  requeued for transcode:    {counts[ACTION_REQUEUE]}")
        print(f"  skipped:                   {counts[ACTION_SKIP]}")
        if not args.apply:
            print("\nDry run: nothing changed. Re-run with --apply.")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
