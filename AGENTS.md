# Chitra — API & worker backend

Self-hosted photo/video backup and search. FastAPI serves the API; RQ workers do
ML and video transcoding; MinIO stores originals and derivatives; SQLite holds
metadata; Redis backs the queues. Runs on a single home server (CPU-only
i5-8500), fronted by nginx and a cloudflared tunnel.

Clients live in sibling repos: `../chitra_ui_next` (web) and `../chitra_ios` (iOS).

## Commands

All Python runs through the virtualenv at `.venv` — git-ignored, created by
`make install`. Never use the system interpreter; the ML stack (torch, CLIP,
InsightFace, FAISS) is installed only in the venv.

```bash
# Full test suite (~3s)
CHITRA_DB_PATH=/tmp/chitra_test.db .venv/bin/python tests/run_tests.py

# One module — the fast loop while working
CHITRA_DB_PATH=/tmp/chitra_test.db .venv/bin/python tests/run_tests.py test_video

# Dependencies
.venv/bin/pip install -r requirements.txt
```

**IMPORTANT: always set `CHITRA_DB_PATH` when running tests.** It defaults to
`photo.db` in this directory, which is the *live production database* — the
suite contains writes, so an unset variable corrupts real data.

There is no linter, formatter, or type checker configured yet. `pyproject.toml`
carries a ruff config for when it is installed; nothing runs it automatically.

## Test baseline

The suite is **not green**. As of 2026-09-01 a clean run is
`168 tests, 7 failures, 4 errors, 2 skipped`. These are pre-existing:

- `test_endpoints.py` — 7 failures, all `401 != 200/400/404`. The tests build a
  `TestClient` with no auth token and predate the auth layer.
- `test_db_async.py` (3 errors), `test_search.py` (1 error).
- 2 skips in `test_storage_client` / `test_background_jobs` when MinIO is down.

Do not "fix" these as a side effect of unrelated work, and do not treat them as
your own breakage. Compare against this baseline; the only number that matters
is whether *your* change added a failure.

## Gotchas

- **`ffmpeg` and `ffprobe` are apt system packages, not pip dependencies.**
  `requirements.txt` only mentions them in a comment. Poster generation and
  transcoding shell out to them — **from the workers only**. The API must
  never spawn ffmpeg: it did (inline poster extraction on upload) and the
  `av:hevc:dfN` decode threads were named in 2 of 6 cgroup OOM kills.
  `tests/test_video_poster.py` guards that.
- **The API holds no ML model.** Text embeddings for search come from the
  `embed_service.py` sidecar over loopback; `core/embed_client.py` is the
  client. There is deliberately no in-process fallback — search 503s when the
  sidecar is down, and `/api/health` reports `embed_status`. Importing
  `app_fastapi` must stay under 200 MB with no torch/transformers resident;
  `tests/test_api_memory_budget.py` enforces it in a fresh subprocess.
- **Thread counts come from `thread_limits.sh`**, sourced by the launchers
  after `.env.production` so operator overrides win. 3 is the measured CLIP
  optimum on this 6-core box; the curve is sharply non-monotonic (4 and 6 are
  ~1.7x worse than 3). Note `OMP_NUM_THREADS` does **not** reach ONNX Runtime —
  measured, see `docs/plans/api-oom-fix.md`.
- **Never run `stop_workers.sh` while a transcode is in flight.** It sends two
  SIGTERMs and cold-kills the job. `process_video_transcode_job` sets
  `transcode_status="processing"` on entry and only leaves that state from
  inside its own try/except, which a SIGTERM never reaches — so the row is
  stranded in `processing` (not `failed`), nothing retries it, and
  `/api/photos/{id}/video` answers 409 `transcode_in_progress` forever. Check
  the video queue is idle first. systemd `Restart=always` respawns workers ~10s
  after a stop. To clean up afterwards:
  `python scripts/requeue_transcodes.py --stuck` (dry run; `--apply` to fix).
  Keep `--limit` at 1 for 4K sources — see the transcode cost note below.

- **Transcoding is VA-API hardware-accelerated but decode stays on CPU.**
  `core/video.py` encodes via `h264_vaapi` on `/dev/dri/renderD128` (the service
  user must be in the `render` group) and falls back to libx264. Because the
  HEVC decode half is still software, two concurrent 4K transcodes push the
  1-minute load average past 12 on this box. Requeue 4K work one at a time.
- **Schema migrations are additive `ALTER TABLE ADD COLUMN` calls run at
  startup** (`core/db_async.py`), wrapped in try/except. There is no migration
  framework and no schema-version table — renames, backfills and type changes
  have nowhere to live, so plan additive-only changes.
- **`core/db.py` and `core/db_async.py` carry duplicated schema DDL that has
  already diverged** — the sync copy has no `users` table, so a DB created by
  `cli/main.py init` cannot authenticate against the API.
- **`app_fastapi.py:1524-1913` is unreachable dead code** — an abandoned
  pre-refactor clustering pass that would `NameError` if it ever ran. Ignore it
  when reading the file; don't extend it.
- **The `api` and `ftp` Makefile targets are broken** — they point at `app.py`
  and `ftp_server.py`, neither of which exists. Production starts via
  `start_production.sh` (uvicorn) and `start_workers.sh` (RQ).
- Two queues: `default` (4 workers, ML) and `video` (2 workers, transcode).
  Workers fork per job, so module-level model caches do not survive between jobs.

## Conventions

- Raw SQL throughout, no ORM. Async paths use `aiosqlite` via `core/db_async.py`;
  sync paths (CLI, jobs) use `core/db.py`.
- Secrets come from `.env.production` (git-ignored). Never hardcode a fallback
  for `JWT_SECRET_KEY` — `core/auth.py` deliberately refuses a known default and
  generates an ephemeral secret instead. That refusal is intentional; keep it.
- Jobs currently swallow exceptions and return `False`, which makes RQ mark them
  successful. New jobs should re-raise after logging so failures are visible.

## Git

`main` is the only branch and the default. It was renamed from `flask` on
2026-09-01; the stale `master` pointer (27 commits behind, fully contained in
`main`) was deleted at the same time. Branch from `main`.

## Safety

Do not run `sudo`, `systemctl`, or deployment scripts. Restarting services,
deploying, and touching the production database are human decisions — surface
what needs doing and let the owner run it.
