@AGENTS.md

<!-- Claude-specific additions only below this line -->

- Scoped rules in `.claude/rules/` load automatically when you touch the files
  they cover (ML pipeline, HTTP layer).
- `.claude/settings.json` pre-approves the test/typecheck loop and denies
  `sudo`, `systemctl`, force-push, and `stop_workers.sh`. A denied command is a
  deliberate boundary, not an obstacle to work around.
- The `restart-workers` project skill documents the only safe way to restart the
  RQ workers without cold-killing an in-flight transcode.

## Repo structure

```
app_fastapi.py        All 40 HTTP routes + middleware + DI + DTO mappers.
                      2.5k lines; :1524-1913 is unreachable dead code.
worker.py             RQ worker entrypoint. Forks per job — see core/jobs.py.
run.py                Dev server launcher.
recluster_all.py      One-off op: reset person_id and re-run face clustering.
backfill_video_dates.py  One-off op: fill video timestamps via ffprobe.
test_fastapi.py       NOT a test — a manual smoke script needing a live server.

core/
  auth.py             JWT issue/verify, bcrypt. Refuses a default secret.
  db.py               Sync SQLite (CLI, jobs). Schema DDL — diverged from
                      db_async.py; has no users table.
  db_async.py         Async SQLite (API). The authoritative schema.
  schemas.py          Pydantic request/response models.
  storage_client.py   MinIO wrapper. __init__ does a network round-trip.
  jobs.py             All RQ jobs: embedding, faces, transcode, clustering.
  embedder.py         CLIP (transformers, ViT-B/32). CPU.
  face.py             InsightFace buffalo_l. CPU, forced ctx_id=-1.
  tagger.py           Zero-shot tagging over 17 hardcoded labels.
  faiss_index.py      Persistent HNSW index for person faces.
  cluster.py          Photo-level clustering — reachable only from the CLI.
  extractor.py        EXIF, GPS, SHA-1, pHash, ffprobe metadata.
  video.py            ffmpeg transcode + poster frames. Best-written module;
                      the only one using the logging framework.
  gallery.py          Thumbnail generation (PIL).
  raw_loader.py       RAW format decoding.
  cache.py            In-process thumbnail cache. FIFO despite the name.
  worker.py           Redis connection + queue definitions.

cli/                  Typer CLI + Textual TUI. Separate entrypoint from the
                      API; create-admin lives here.
tests/                unittest suite; run via tests/run_tests.py.
faiss_indexes/        Persisted FAISS indexes. Relative path — CWD-sensitive.
logs/                 Worker logs. Truncated on every restart, unrotated.
photos/               Legacy local-storage era. Effectively empty.
```
