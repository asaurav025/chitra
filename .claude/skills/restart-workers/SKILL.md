---
name: restart-workers
description: Safely restart Chitra's RQ workers without cold-killing an in-flight video transcode. Use when worker code changed, workers are wedged, models need reloading, or a queue is backed up.
---

# Restart the RQ workers safely

`stop_workers.sh` sends two SIGTERMs in quick succession. A transcode killed
that way leaves the video permanently `transcode_status="failed"` with no
requeue endpoint — recovery is manual SQL. So the whole procedure is: **prove
the video queue is idle, then stop, then let systemd respawn.**

## 1. Check both queues are idle

```bash
redis-cli llen rq:queue:video
redis-cli llen rq:queue:default
redis-cli keys 'rq:job:*' | head
```

Also confirm no ffmpeg is running:

```bash
pgrep -af ffmpeg
```

If either returns work in progress, **stop here and wait.** A 4K transcode can
run for many minutes; the job timeout is 2 hours. Report the queue depth to the
owner rather than forcing it.

## 2. Stop

Only when the video queue is empty and no ffmpeg is running:

```bash
cd /home/saurav/services/chitra && ./stop_workers.sh
```

This runs as `saurav`, no sudo needed.

## 3. Let systemd respawn

`chitra-workers.service` has `Restart=always`, so workers come back on their own
about 10 seconds later. Do not run `start_workers.sh` by hand — you will end up
with two sets of workers competing for the same queues.

## 4. Verify

```bash
pgrep -af 'worker.py' | wc -l     # expect 6: 4 default + 2 video
tail -20 logs/worker_1.log
tail -20 logs/worker_video_1.log
```

Worker logs are truncated on every restart, so anything you needed from the
previous run must be captured before step 2.

## If a transcode was killed anyway

The video is stuck at `transcode_status="failed"`. There is no requeue endpoint;
it needs a manual DB update back to `pending` plus a re-enqueued job. Flag it to
the owner with the photo id rather than improvising a fix.
