# API OOM fix — get ML models and ffmpeg out of the API tier

Status: ready to execute. Branch from `main`.
Owner actions (sudo) are collected in one section — agents must not run them.

## What we measured

Numbers measured 2026-09-01 on the box. Do not re-derive these.

### The incident
- `chitra-api.service` cgroup-OOM-killed 6x in 2 days (Aug 31 20:01, Aug 31
  21:50, Sep 01 08:08, Sep 01 09:10 x2, Sep 01 09:26).
  `oom-kill:constraint=CONSTRAINT_MEMCG, oom_memcg=/system.slice/chitra-api.service`.
- Victim anon-RSS 1.05 -> 2.18 GB. `memory.peak` reached 4,296 MB — exactly the
  `MemoryMax=4G` cap — with ~3,600 `memory.max` reclaim events.
- 2 of the 6 kills named `av:hevc:df0` / `av:hevc:df6` — ffmpeg HEVC decode
  threads, running *inside the API cgroup*.
- Box: i5-8500, 6 cores / 6 threads, 15.8 GB RAM, 1.5 GB already in swap. It also
  runs open-webui and a `/opt/rag` service — Chitra does not own the whole box.

### Model footprints (measured)

| | peak | steady |
|---|---|---|
| CLIP (ViT-B/32) alone | 1,670 MB | 1,139 MB |
| InsightFace buffalo_l alone | 1,462 MB | 1,381 MB |
| both together | 2,054 MB | 1,927 MB |

Only ~300 MB shares between processes; ~1.66 GB per process is private dirty.
uvicorn runs 4 workers and each imports the app independently (no
fork-after-import), so nothing meaningful is shared.

### Import weight — measured fresh, 2026-09-01

```
baseline_MB=9  after_import_MB=557  delta_MB=547
heavy modules resident: ['torch', 'transformers', 'faiss', 'rawpy']
```

The same import set with the ML chain removed:

```
web+np 75 MB -> +core-light 88 MB -> +extractor 88 MB -> +faiss 97 MB -> +heif 97 MB
torch resident: False
```

**~450 MB per uvicorn worker is pure torch/transformers import weight with no
model loaded. x4 workers ~= 1.8 GB of the 4 GB cap spent on nothing.**

Confirmed live (service restarted since the last kill, counters reset):

```
memory.current = 1,287 MiB
memory.peak    = 1,289 MiB
anon           = 1,275 MiB
memory.events  = all zero
```

1,275 MiB anon / 4 workers ~= 319 MiB each — torch imported, no model loaded.
That is the *floor* today, before anyone searches.

### CPU
- `torch.get_num_threads()` = 6, ONNX Runtime also defaults to 6, on 6 cores.
  Nothing in the codebase sets `OMP_NUM_THREADS` or calls `set_num_threads`.
- CLIP image embed: **153 ms @ 6 threads, 68 ms @ 3 threads** — the default is
  2.25x slower.
- `chitra-workers` cpu.stat: `nr_throttled 8762 / nr_periods 10823` = **81% of
  periods throttled**, 1,089 s of throttling, at `CPUQuota=400%`.
- `chitra-api` cpu.stat: `135 / 25049` = 0.5% throttled at `CPUQuota=200%`.

### Data scale (production `photo.db`, read-only query)
- 2,040 photos: 1,012 `media_type='photo'`, 262 `'video'`, **766 NULL** (legacy
  rows — any `media_type` branch must treat NULL as photo).
- 1,694 embeddings totalling **3.31 MiB**. 2,081 faces.

That last number matters: `/api/search/photos` loads *every* embedding and does a
brute-force numpy GEMV. At 3.31 MiB that costs single-digit milliseconds and
negligible memory. **The entire cost of the search endpoint is the CLIP text
embedding — 12.7 ms of compute sitting on top of 1.1 GB of resident model.**

### The import chain that drags torch into the API

| `app_fastapi.py` | pulls |
|---|---|
| `:46 from core.embedder import ClipEmbedder` | torch, transformers (direct) |
| `:48 from core.face import face_encodings` | never called in this file; dead weight |
| `:49 from core.tagger import auto_tags` | never called in this file; `core/tagger.py:4` imports `ClipEmbedder` |
| `:54-63 from core.jobs import (...)` | `core/jobs.py:14,16,19` import all three at module scope |

All four must go before the API is torch-free.

### An additional finding
`core/cache.py` is an in-process thumbnail cache with `_cache_max_size = 1000`
**entries and no byte budget**. `core/gallery.ensure_thumb` writes 512x512 JPEG
at `quality=100, optimize=False, subsampling=0` — typically 150-400 KB each.
1,000 entries x ~250 KB ~= 250 MB **per uvicorn worker**, ~1 GB across four.
Arithmetic, not measured RSS, but plausibly a large slice of the cap and
trivially bounded. Task 5.1 fixes it.

## Goal (end state)

1. `import app_fastapi` in a fresh interpreter resides under 200 MB and has
   `torch`, `transformers`, `onnxruntime`, `insightface` absent from
   `sys.modules`.
2. No `ffmpeg` process is ever a descendant of `chitra-api.service`.
3. `chitra-api.service` steady-state `memory.current` under ~900 MB with four
   workers; `MemoryMax` lowered to 2G so a regression trips loudly.
4. Search still answers synchronously in well under 100 ms.
5. Thread counts are explicit everywhere, set from files in the repo (no sudo).
6. Test suite no worse than the baseline.

## Design decisions

### D1 — Search text embedding: a dedicated long-lived embedding sidecar

| option | search latency | API memory | verdict |
|---|---|---|---|
| **A. Long-lived sidecar, API calls over loopback HTTP** | 12.7 ms + ~2 ms ~= **15 ms** | **0** | **chosen** |
| B. Small text-only CLIP tower in the API | ~13 ms | still x4; the 1.1 GB is mostly torch runtime, not weights | rejected |
| C. Compute via an RQ job and poll | RQ forks per job -> CLIP reloads every time -> **tens of seconds** | 0 | rejected |
| D. Cache query->vector | 0 ms hit, unbounded miss | 0 | adjunct only (Task 3.8) |

Reasoning for A:

- Converts a per-uvicorn-worker cost into a per-box cost. One CLIP instead of
  four: ~1.14 GB resident total instead of up to ~4.5 GB of ceiling exposure.
- Preserves synchronous search semantics exactly. No client change in either
  repo, no polling protocol, no `202` state machine.
- 15 ms is not a regression worth discussing next to a 12.7 ms baseline. The
  rest of the endpoint already costs more than the HTTP hop.
- Fixes the event-loop blocking called out in `.claude/rules/http-layer.md` for
  free: with `httpx.AsyncClient` the call is genuinely awaited.
- `get_embedder()` is already the seam. An `EmbeddingClient` with the same
  `text_embedding(text) -> np.ndarray` shape keeps the handler nearly unchanged.

Option B was rejected on a measurement: the 1,139 MB steady figure is dominated
by the torch runtime, not the ~250 MB of fp32 weights. Splitting the tower keeps
four copies of the expensive part.

Option C was rejected because RQ's fork-per-job model makes every text embedding
pay a full cold model load — the one architecture *worse* than today.

**Explicit sub-decision: no in-process CLIP fallback when the sidecar is down.**
A silent fallback would re-create the exact OOM under load and be invisible in
logs. Search returns `503 search_unavailable`; `/api/health` reports
`embed_status`. Fail loud.

**Sub-decision: sidecar launches from `start_workers.sh` (no sudo) but the target
state is its own `chitra-embed.service`.** `CHITRA_EMBED_SELF_START` (default
`1`) switches between them, making this landable today without sudo.

**Sub-decision: the sidecar exposes `POST /embed/image` from day one, but the RQ
workers are NOT switched over in this plan.** That is a throughput redesign
needing its own measurements. The endpoint costs ~10 lines and makes the
follow-up trivial.

### D2 — Thumbnails and posters: split by mechanism

Verified in both client repos:

- **Web** (`chitra_ui_next`): `client.ts:189` types `saved[].thumbnail`, but **no
  component reads it** — `UploadDialog.tsx:43-45` only invalidates the `photos`
  query. Thumbnails render via `AuthImage` -> `GET /api/photos/{id}/thumbnail`.
  `useAuthImage` has `staleTime: Infinity`, so **a failed thumbnail fetch renders
  the fallback and will not refetch while mounted** — the one real regression
  surface.
- **iOS** (`chitra_ios`): `UploadResult.swift:17` decodes `thumbnail` as
  `String?`; `SyncEngine.swift:189-196` uses only `saved.first.id`. Nothing reads
  the thumbnail fields.

**Returning `thumbnail: null` breaks neither client's decoding.** Only timing
changes.

- **Video posters (ffmpeg) leave the API unconditionally** — 2 of the 6 kills.
  `ensure_video_poster_async` is deleted; `generate_video_poster_job` runs on the
  **`default` queue** (4 workers, so it starts immediately rather than queueing
  behind a multi-minute transcode on the 2-worker `video` queue).
- **Photo thumbnails stop being generated inline on upload, but the lazy
  generate-on-GET path stays** — PIL thumbnailing is neither ffmpeg nor ML, and
  the lazy path is the only thing that makes thumbnails work for the 766 legacy
  rows. Note the current upload path passes the **remote** key to
  `ensure_photo_thumb_async`, re-downloading from MinIO the file it just
  uploaded; enqueuing removes that waste entirely.
- **Accepted, bounded regression:** a freshly uploaded video may be fetched
  before its poster exists (expected 1-3 s). Mitigation: that GET returns `404` +
  `Retry-After: 3` and enqueues the poster job if not already queued (Redis
  SETNX dedupe). A client-side `staleTime` follow-up is out of scope here.

### D3 — Thread limits go in the repo's launcher scripts

`.env.production` is git-ignored and agent-denied; unit files need sudo.
`start_production.sh` and `start_workers.sh` are in the repo, are the actual
`ExecStart` of both units, and already source `.env.production` — so a
`: "${VAR:=n}"; export VAR` block placed *after* that source gives defaults while
letting `.env.production` or systemd `Environment=` win. Env vars are also the
only lever that works, because torch and BLAS read them at import.

- **API**: `OMP/MKL/OPENBLAS/NUMEXPR/VECLIB_NUM_THREADS=2`. After this change the
  API's only numeric work is a 1,694x512 GEMV over 3.31 MiB — memory-bound.
  Setting 2 also means a sneaked-back torch import spawns 2 threads x 4 workers,
  not 24.
- **Workers and sidecar**: `${CHITRA_ML_THREADS:-3}` — the measured sweet spot.
- **ffmpeg poster**: `-threads 2`, plus `-an -sn`. Frame-threaded HEVC decode
  allocating per-thread buffers is what the `av:hevc:dfN` victims were.
- **ONNX Runtime is an open question and gets an investigation task, not an
  assertion** (Task 6.3).

**Do not raise `CPUQuota` before the thread caps land and are re-measured.**

## Sequencing rationale

Ordered by (value / risk), everything verifiable-without-restart first.

**Phase 1 is the safest thing to land first and also the biggest single win** —
mechanical, no HTTP contract change, no new process, fully verifiable in-repo.
Phase 6 is equally zero-risk and rides the same restart. Phase 2 must precede
Phase 3's final step.

**Requires an owner restart:** every change only takes effect after
`systemctl restart chitra-api` and a safe worker restart.

## Tasks

Conventions: TDD — write the test, **run it and paste the failing output into the
task's evidence line**, then implement. After each task confirm the suite is at
or better than baseline (counts may rise as tests are added; failures and errors
must not).

```
CHITRA_DB_PATH=/tmp/chitra_test.db .venv/bin/python tests/run_tests.py 2>&1 | tail -5
```

### Phase 0 — Baseline

**Task 0.1 — Record the pre-change baseline.** Run the suite, the import probe,
and both cgroup dumps; append raw output to Evidence with a timestamp.

**Task 0.2 — Add the memory-budget guard test. Watch it fail.**
New `tests/test_api_memory_budget.py`.

Critical: **the probe must run in a fresh subprocess.** `tests/run_tests.py`
discovers all modules into one interpreter, and `test_background_jobs` imports
`core.jobs` while `test_search` constructs `ClipEmbedder` — an in-process
`sys.modules` assertion would be polluted and silently useless.

Assertions: after `import app_fastapi`, none of `torch`, `transformers`,
`onnxruntime`, `insightface` in `sys.modules`; `ru_maxrss` < 200 MB.

**Done when:** it fails showing `heavy=['torch','transformers']`, `rss_mb=557`.
Errors go 4 -> 5 temporarily; note it in the commit message.

### Phase 1 — Break the ML import chain

**Task 1.1 — Make `core/jobs.py`'s ML imports lazy.** Move `:14` `ClipEmbedder`,
`:16` `face_encodings`, `:19` `auto_tags` into the functions that use them. Leave
`core.db`, `core.video`, `core.extractor`, `core.storage_client`, `core.gallery`
at module scope — all measured cheap.

**Conflict note:** touch only the import block (lines 12-19) and the four ML
functions. Rebase on `main` immediately before committing.

**Done when:** `python -c "import core.jobs,sys; assert 'torch' not in sys.modules"`
passes, ruff reports no F821, suite at baseline.

**Task 1.2 — Delete the two unused ML imports from `app_fastapi.py`** (`:48`,
`:49`). Both are verifiably unreferenced elsewhere in the file.

**Task 1.3 — Re-measure.** `core.jobs` alone should be clean; `app_fastapi` still
shows torch via `:46` until Task 3.2.

### Phase 2 — Embedding sidecar

**Task 2.1 — Red test for the client** (`tests/test_embed_client.py`), using
`httpx.MockTransport`: POST shape, base64 float32 decode, L2 normalisation,
`HTTPException(503)` on `ConnectError` and `ReadTimeout`.

**Task 2.2 — Implement `core/embed_client.py`.** Wire format
`POST /embed/text {"text": ...}` -> `{"dim":512,"dtype":"float32","vector_b64":...}`.
Base64 float32 rather than a JSON float array: lossless, compact, no float-repr
round-trip. `httpx==0.28.1` is already pinned — **no new dependency**. Config:
`CHITRA_EMBED_URL` (default `http://127.0.0.1:5101`), `CHITRA_EMBED_TIMEOUT`
(5.0), optional `CHITRA_EMBED_TOKEN`. No fallback path.

**Task 2.3 — Red test for the service** (`tests/test_embed_service.py`) with
`ClipEmbedder` stubbed: `/health` shape, base64 round-trip, 422 on empty text,
`torch.set_num_threads` called at startup.

**Task 2.4 — Implement `embed_service.py`** at the repo root. `lifespan`
constructs one `ClipEmbedder` after calling `torch.set_num_threads`. Both
endpoints run the embed via `run_in_executor`. Loopback only, `--workers 1`
mandatory. Logs resident size at startup.

**Task 2.5 — Launcher wiring.** `start_workers.sh` starts it when
`CHITRA_EMBED_SELF_START` is 1, writing a pid file; `stop_workers.sh` stops it.
Update `.claude/skills/restart-workers/SKILL.md` — now 7 processes, and a worker
restart briefly 503s search. Update `.env.example`.

### Phase 3 — Cut the API over

**Task 3.1 — Red test for the search handler**, injecting a fake client via
`app.dependency_overrides`: ranked results, 503 propagation, and — with an
`AsyncMock` — that the handler **awaits** the client. Supply a token or override
`get_current_active_user`, or these add to the 7 known auth failures.

**Task 3.2 — Replace `get_embedder()` with `get_embedding_client()`.** Delete
`:46`, `:77 _EMBEDDER`, `:197-202`. Construct one shared `httpx.AsyncClient` in
`lifespan`, close on shutdown.

**Task 3.3 — Rewrite the search handler's embedding call** (`:1958-1966`) to
`await client.text_embedding(query)`. Keep the defensive re-normalise.

**Task 3.4 — Add `embed_status` to `/api/health`.** Leave the "returns 200 while
degraded" behaviour alone — separate known issue, changing it here is scope creep.

**Task 3.5 — Confirm the memory-budget test goes green.** The probe should print
`heavy= []` and `rss_mb` under 200. **The single most important done-check.**

**Task 3.6 — Confirm the suite is back to baseline.**

**Task 3.7 — Live sidecar integration check** on a scratch port with a scratch
DB: `/health`, one `/embed/text`, compare against a direct `ClipEmbedder` call
(max abs diff < 1e-5), time 20 sequential calls, then kill and confirm gone.

**Task 3.8 (optional) — Redis cache for text embeddings.** Key
`chitra:embed:text:v1:{sha256(model + query)}`, raw float32 value, 30-day TTL.

### Phase 4 — Get ffmpeg out of the API

**Task 4.1 — Red tests** for `extract_poster` flags, the poster job's use of
`download_to_path` (**not** `download_file` — the latter pulls a multi-GB video
into RAM), no-op when already present, re-raise on failure, and a grep assertion
that `app_fastapi.py` no longer references `extract_poster`.

**Task 4.2 — Add `-threads 2 -an -sn` to `core.video.extract_poster`.** Place
`-threads 2` right after `-y` so it applies to decode too. **Scope guard:** do
not touch the transcode command builders.

**Task 4.3 — Add `generate_video_poster_job` to `core/jobs.py`.** A new function
that touches no `transcode_status` value, so it cannot collide with concurrent
transcode work. Accepts one duplicated MinIO download in exchange for not
queueing behind a multi-minute transcode.

**Task 4.4 — Upload path enqueues instead of extracting.** Delete the
`ensure_video_poster_async` call and the function itself; return
`thumbnail: None` (both clients tolerate it — verified in D2).

**Task 4.5 — Guard the thumbnail GET for videos.** Today PIL tries to open an mp4
and 500s after downloading the whole video. Instead enqueue (Redis SETNX dedupe,
or one grid render of 50 videos fires 50 enqueues) and return `404` +
`Retry-After: 3`. NULL `media_type` counts as photo — 766 legacy rows.

**Task 4.6 — Prove ffmpeg is gone.** The grep should return only the
`video.ensure_ffmpeg()` startup probe (a `shutil.which`, harmless); decide
whether to move that warning to the workers and record the choice.

### Phase 5 — Thumbnails off the request path, bound the cache

**Task 5.1 — Bound the thumbnail cache by bytes.** Red test in
`tests/test_cache.py`: exceeding `CHITRA_THUMB_CACHE_BYTES` (default 64 MiB)
evicts oldest-first; `get_cache_stats()` reports `bytes` and `max_bytes`.

**Task 5.2 — Red test:** upload returns `thumbnail: null` and enqueues a job;
the lazy GET still generates when `thumb_path` is unset.

**Task 5.3 — Add `generate_photo_thumb_job` and enqueue it from upload.** Keep
`ensure_photo_thumb_async` — it is still the lazy GET fallback.

### Phase 6 — Thread limits

**Task 6.1 — Export thread caps from the launcher scripts**, placed after the
`.env.production` source so overrides still win. API gets 2; workers get
`${CHITRA_ML_THREADS:-3}`. Demonstrate override precedence.

**Task 6.2 — Sidecar sets `torch.set_num_threads` explicitly** (covered by 2.4).

**Task 6.3 — Investigate whether ONNX Runtime honours `OMP_NUM_THREADS`.** The
PyPI `onnxruntime==1.19.0` CPU wheel is likely not OpenMP-linked, in which case
it sizes its own pool to core count and ignores the var;
`insightface.app.FaceAnalysis` does not expose `SessionOptions`. Measure thread
count during a face job. If it stays at 6, add a task to construct sessions with
`intra_op_num_threads`. **Record the evidence either way.**

**Task 6.4 — Re-measure the CLIP embed at 1/2/3/4/6 threads.** We have only two
points; record a full curve so `CHITRA_ML_THREADS` can be tuned from evidence.

## Owner actions (sudo — agents must not run these)

Unit files live at `/etc/systemd/system/chitra-{api,workers}.service` and **are
not in the repo** — a known gap. An agent can create a `deploy/` copy as
documentation; only installing it needs sudo.

### Immediate stop-gap, zero code
Set `WORKERS=2` in `.env.production` and restart the API. Halves the per-worker
torch import weight (~450 MB x 2 saved) and halves worst-case model residency, at
the cost of concurrency. Fully revertible. Worth doing now if kills recur before
this plan lands.

### After Phase 1 + Phase 6
`sudo systemctl restart chitra-api`, then a safe worker restart per
`.claude/skills/restart-workers` — prove the video queue is idle and no ffmpeg is
running first.

### After Phase 3 — `chitra-api.service`
1. **First** add only `MemoryHigh=1G` (keep `MemoryMax=4G`) and observe for a day;
   watch the `high` counter in `memory.events`.
2. **Then** lower `MemoryMax` 4G -> 2G. Lowering it is the point: it turns "a
   model or an ffmpeg silently came back" from invisible creep into loud failure.
3. Leave `CPUQuota=200%` — only 0.5% throttled.

### `chitra-workers.service`
Keep `MemoryMax=8G`; add `MemoryHigh=6G`. **Do not raise `CPUQuota=400%` yet** —
81% throttled, but raising it before thread caps land just lets 6 processes x 6
threads fight harder. Re-measure after Phase 6.

### Optional — `chitra-embed.service`
Promotes the sidecar out of the worker cgroup; then set
`CHITRA_EMBED_SELF_START=0`. `WorkingDirectory` is load-bearing —
`faiss_indexes/` is a relative path and a unit without it silently creates a
second, empty index directory (see `.claude/rules/ml-pipeline.md`). Suggested
limits `MemoryMax=2G`, `MemoryHigh=1600M`, `CPUQuota=150%`. Add
`Wants=chitra-embed.service` to the API unit.

**Capacity note:** 2G + 8G + 2G = 12 GB of caps on a 15.8 GB box that also runs
open-webui and `/opt/rag`. Caps are ceilings, not reservations, but there is no
headroom for all three to peak together — which is why `MemoryHigh` (reclaim) is
the primary control and `MemoryMax` the backstop.

## Post-restart verification (agent-runnable, no sudo)

```
cat /sys/fs/cgroup/system.slice/chitra-api.service/memory.current   # expect < ~900 MB
cat /sys/fs/cgroup/system.slice/chitra-api.service/memory.events    # max 0, oom_kill 0
cat /sys/fs/cgroup/system.slice/chitra-workers.service/cpu.stat
pgrep -af 'uvicorn app_fastapi' | wc -l
pgrep -af embed_service
curl -s localhost:5101/health
pgrep -af ffmpeg   # must never show a child of the API
```

Success after a week: `oom_kill` still 0, `memory.peak` under 1.5 GB, no
`av:hevc` process ever parented to the API.

## Risks

| risk | mitigation |
|---|---|
| Sidecar down -> search 503s | `Restart=always`; `embed_status` in `/api/health`; optional Redis cache. Deliberately **no** in-process fallback — it would silently re-create the OOM. |
| Worker restart now also restarts search | Documented in the restart-workers skill; removed entirely by `chitra-embed.service`. |
| Video poster arrives seconds after upload; `staleTime: Infinity` makes a broken tile sticky | Poster on the 4-worker `default` queue (1-3 s); `404 + Retry-After` + enqueue-if-missing; client follow-up documented. |
| Lazy imports could `NameError` at a module-scope use site | ruff F821, full suite, existing job tests. Keep the diff to the import block. |
| Merge conflict with concurrent transcode work | Only the import block and four ML functions are touched. Rebase before committing. |
| Lowering `MemoryMax` to 2G trips on something unforeseen | Land `MemoryHigh=1G` first and observe for a day. |

## Rollback

One commit per phase on a branch off `main`, each independently revertable.
Phase 3 is the only one with a runtime dependency; reverting restores in-process
CLIP — an emergency escape hatch, not a configuration option.

## Explicitly out of scope

- `scripts/requeue_transcodes.py`, `process_video_transcode_job`, and
  `transcode_status` semantics.
- The pre-existing baseline failures. Do not "fix" them here.
- The `app_fastapi.py:1524-1913` dead code block.
- The health endpoint returning 200 while degraded.

## Follow-ups (not this change)

1. **Route worker image embeddings through the sidecar.** Would cut embedding
   jobs from ~58 s to ~1 s *and* stop four workers each loading 1.1 GB. A
   throughput redesign needing its own measurements.
2. **`chitra_ui_next`:** lower `AuthImage`'s `staleTime` for video thumbnails
   whose `playback_status` is `pending`/`processing`.
3. **Thread caps on `_sw_transcode_cmd`** in `core/video.py`.
4. **`tests/test_search.py:setUpClass` constructs a real `ClipEmbedder`**,
   loading 1.1 GB into the test process on every full run. Gate behind
   `CHITRA_TEST_LOAD_MODELS=1`. May change the documented baseline — update
   `AGENTS.md` in the same commit if so.
5. **Copy the two unit files into `deploy/`** so they stop being an undocumented,
   sudo-gated part of the system.

## Evidence

_(Appended by the executing agent as tasks complete.)_

### Phase 0 — Baseline

**Task 0.1 — pre-change baseline, 2026-09-01T16:54:25+05:30.**

Suite (`CHITRA_DB_PATH=/tmp/chitra_test.db .venv/bin/python tests/run_tests.py`):

```
Ran 81 tests in 3.127s
FAILED (failures=7, errors=4, skipped=2)
```

Note: 81 tests, not the 66 recorded in `AGENTS.md` — 15 passing tests were added
by concurrent work since the plan was written. Failures and errors are unchanged
at 7/4, so the baseline that matters is intact.

Import probe (fresh interpreter, `cwd=repo root`, `ru_maxrss`):

```
target=app_fastapi baseline_MB=9 after_import_MB=557 delta_MB=548
heavy modules resident: ['torch', 'transformers', 'faiss', 'rawpy']

target=core.jobs   baseline_MB=9 after_import_MB=519 delta_MB=509
heavy modules resident: ['torch', 'transformers', 'rawpy']
```

Reproduces the plan's measured `after_import_MB=557` exactly. `core.jobs` alone
accounts for 519 of those 557 MB.

`chitra-api.service` cgroup (service restarted since the last kill):

```
memory.current  1,349,849,088  (1,287 MiB)
memory.peak     1,352,994,816  (1,290 MiB)
memory.events   low 0  high 0  max 0  oom 0  oom_kill 0  oom_group_kill 0
cpu.stat        nr_periods 30696  nr_throttled 135  throttled_usec 6,875,617   (0.4% throttled)
```

`chitra-workers.service` cgroup:

```
memory.current    135,684,096  (129 MiB — idle between jobs)
memory.peak     3,471,708,160  (3,311 MiB)
memory.events   all zero
cpu.stat        nr_periods 10853  nr_throttled 8762  throttled_usec 1,089,240,159   (80.7% throttled)
```

Worker CPU throttling confirmed at 80.7% of periods, matching the plan's 81%.

**Task 0.2 — memory-budget guard test added and watched fail.**

New `tests/test_api_memory_budget.py`. The probe runs in a fresh subprocess via
`subprocess.run([sys.executable, "-c", ...])` with `cwd` at the repo root and
`CHITRA_DB_PATH` forced to a scratch path. This is load-bearing: `run_tests.py`
discovers every module into one interpreter, where `test_background_jobs`
imports `core.jobs` and `test_search` constructs a real `ClipEmbedder`, so an
in-process `sys.modules` assertion would be pre-polluted and silently useless.

Two guard classes: `TestApiImportIsMLFree` (`app_fastapi`, plus the < 200 MB RSS
budget) and `TestJobsImportIsMLFree` (`core.jobs`) — the latter is the red test
for Task 1.1, whose done-check is exactly this assertion.

Red, for the right reasons:

```
FAIL: test_no_heavy_ml_modules_resident (TestApiImportIsMLFree)
AssertionError: Lists differ: [] != ['torch', 'transformers']
 : importing app_fastapi loaded ['torch', 'transformers'] — each uvicorn worker
   pays this with no model loaded

FAIL: test_import_rss_within_budget (TestApiImportIsMLFree)
AssertionError: 556.73828125 not less than 200 : import of app_fastapi resident
   at 557 MB, budget is 200 MB (x4 uvicorn workers)

FAIL: test_no_heavy_ml_modules_resident (TestJobsImportIsMLFree)
AssertionError: Lists differ: [] != ['torch', 'transformers']
 : importing core.jobs loaded ['torch', 'transformers'] — the API imports this
   module only to enqueue by reference
```

`rss_mb=557` and `heavy=['torch','transformers']` are exactly the predicted
done-when values.

Suite after Task 0.2:

```
Ran 84 tests in 7.456s
FAILED (failures=10, errors=4, skipped=2)
```

**Divergence from the plan:** the plan predicted "errors go 4 -> 5". In fact the
guard lands as **3 failures, not 1 error** (7 -> 10 failures; errors unchanged at
4), because the probe subprocess succeeds and it is the assertions that fail —
an error would mean the probe itself crashed. Intended and temporary.

### Phase 1 — Break the ML import chain

**Task 1.1 — `core/jobs.py` ML imports made lazy.**

Removed from module scope: `core.embedder.ClipEmbedder`, `core.face.face_encodings`,
`core.tagger.auto_tags`. Left at module scope as measured-cheap: `core.db`,
`core.video`, `core.extractor`, `core.storage_client`, `core.gallery`.

Moved into the five call sites (`core/jobs.py`):

| line | function | import |
|---|---|---|
| 132 | `_get_embedder` | `from core.embedder import ClipEmbedder` |
| 149 | `process_photo_embedding_job` | `from core.tagger import auto_tags` |
| 196 | `process_photo_faces_job` | `from core.face import face_encodings` |
| 287 | `_process_single_embedding` | `from core.tagger import auto_tags` |
| 355 | `_process_single_face` | `from core.face import face_encodings` |

A comment block at `core/jobs.py:17-22` records *why* they are lazy, so the next
reader does not tidy them back to module scope.

Diff scope respected — `git diff -U0` hunk headers touch only the import block
and those five functions. `process_video_transcode_job` and everything
`transcode_status`-related are untouched (concurrent transcode work was in
flight).

Done-check:

```
$ .venv/bin/python -c "import core.jobs, sys; assert 'torch' not in sys.modules"
PASS: core.jobs imported, torch not in sys.modules
```

**F821 check — note: `ruff` is not installed** in the venv or on PATH (AGENTS.md
already records that no linter is configured), so the plan's "ruff reports no
F821" done-check could not be run as written. Substituted an AST check that
proves every reference to the three lazily-imported symbols is bound in the
scope that uses it — precisely the NameError risk the plan flags:

```
  ok: ClipEmbedder   @ line 136 bound locally in _get_embedder()
  ok: auto_tags      @ line 172 bound locally in process_photo_embedding_job()
  ok: face_encodings @ line 212 bound locally in process_photo_faces_job()
  ok: auto_tags      @ line 310 bound locally in _process_single_embedding()
  ok: face_encodings @ line 371 bound locally in _process_single_face()

PASS: all 5 references resolve; 0 F821-equivalent problems
```

**Task 1.2 — deleted the two unused ML imports from `app_fastapi.py`.** Verified
unreferenced first: `grep -n` returned exactly one hit each, the import line
itself (`:48 from core.face import face_encodings`, `:49 from core.tagger import
auto_tags`).

**Task 1.3 — re-measure.**

```
                 before                                    after
core.jobs        519 MB  ['torch','transformers','rawpy']   53 MB  ['rawpy']
app_fastapi      557 MB  ['torch','transformers','faiss','rawpy']
                                                           557 MB  (unchanged)
```

**`core.jobs` drops 519 -> 53 MB, a 466 MB reduction**, and no longer resides
torch/transformers. `app_fastapi` is unchanged at 557 MB, exactly as the plan
predicts — `:46 from core.embedder import ClipEmbedder` is still a direct
import and only Task 3.2 removes it.

Per-module probe confirming `core.embedder` is now the **sole** remaining torch
source in the API's chain:

```
  core.auth              rss=  31 MB  heavy=[]
  core.cache             rss=  11 MB  heavy=[]
  core.db                rss=  13 MB  heavy=[]
  core.db_async          rss=  21 MB  heavy=[]
  core.embedder          rss= 511 MB  heavy=['torch', 'transformers']
  core.extractor         rss=  38 MB  heavy=[]
  core.faiss_index       rss=  35 MB  heavy=[]
  core.gallery           rss=  39 MB  heavy=[]
  core.jobs              rss=  53 MB  heavy=[]
  core.schemas           rss=  31 MB  heavy=[]
  core.storage_client    rss=  32 MB  heavy=[]
  core.video             rss=  14 MB  heavy=[]
  core.worker            rss=  29 MB  heavy=[]
```

Every module except `core.embedder` is at or under 53 MB, which supports the
plan's projection that the API lands near ~97 MB once Task 3.2 lands — well
inside the 200 MB budget.

Suite after Phase 1:

```
Ran 84 tests in 5.723s
FAILED (failures=9, errors=4, skipped=2)
```

That is the 7 pre-existing `test_endpoints` auth failures + 4 pre-existing
errors (3 `test_db_async`, 1 `test_search`) + **2 intentionally-red guard
assertions**. `TestJobsImportIsMLFree` went green. No regression.

**Phase 1 ends with the guard test still red by design** — the two
`TestApiImportIsMLFree` assertions (`heavy=['torch','transformers']`,
`rss_mb=557`) stay red until Phase 3 Task 3.2 removes `app_fastapi.py:46`. That
is the plan's expected state, not a defect.
