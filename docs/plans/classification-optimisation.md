# Classification / ML pipeline optimisation — throughput, tags, and the storage gate

Status: ready to execute, **with a hard gate at Phase 5**. Branch from `main`.
Owner actions (sudo) are collected in one section — agents must not run them.

Successor to `docs/plans/api-oom-fix.md`, which removed ML from the API tier and
built the embedding sidecar. This plan makes the *worker* side fast, makes the
tag vocabulary useful, and exposes the search surfaces that are already
computable from data on disk.

## Storage risk (accepted by the owner — do not re-litigate)

`/dev/sda` is failing — 1,950+ unrecovered read errors, still climbing, 11 bad
sectors across four regions.

```
sda      931.5G  "Expansion"     -> /dev/sda1  /mnt/minio-data   xfs   69G used
nvme0n1  238.5G  "NFORCE 256NV"  -> nvme0n1p2  /                 ext4  150G free
```

All originals, thumbnails, posters and face crops are on the failing disk.
`photo.db`, Redis, `faiss_indexes/`, the HF model cache and this repo are on the
healthy NVMe. Anything reading only SQLite/Redis/model-cache touches `/dev/sda`
zero times.

**The owner has accepted this risk and chosen to proceed.** Consequences folded
into the plan, not used to block it:

* The re-embed reads **thumbnails, not originals** (`scripts/reembed.py`'s
  design) — ~100x less I/O. Coverage measured: **1,971 of 2,040 rows (96.6%)**,
  and **every one of the 1,694 currently-embedded photos has a thumbnail**, so
  the thumbnail path covers the whole existing corpus with no fallback.
* `scripts/reembed.py` carries a disk-health guard that aborts if kernel read
  errors climb mid-run. **Do not add overrides, and do not re-run past an abort
  without reporting it.** An abort is data, not an obstacle.
* Phase ordering puts every zero-media-read task first anyway — that is what
  value / risk gives on its own, not a gate.

The evacuation remains the highest-value owner action and is listed there.

## Concurrent work — read this first

**Agent A — face-clustering trigger (committed `4383ff0`).** Owns
`app_fastapi.py` `:63-65`, `:803+`, `:1167`, `:1302-1311`, `:1387-1395`; and in
`core/jobs.py` the new `FACE_MATCH_THRESHOLD = 0.60`, the
`_auto_match_face_to_person` signature and its call sites at `:303` / `:461`.
**It does not touch `process_photo_embedding_job` or
`_process_single_embedding`**, so Phase 2 — the largest change here — is
conflict-free.

**Agent B — `scripts/reembed.py` and the duplicate-insert fix. Nothing landed
yet** (verified: no `scripts/reembed.py` committed; `embeddings` still
`(id, photo_id, dim, vector)`; `put_embedding` still a plain INSERT at
`core/db.py:279`). **This plan treats Agent B's work as a dependency with a
contract, not as existing code** — Task 0.3 verifies what actually landed.
**Do not implement the uniqueness fix; Agent B owns it.**

Re-check `git log --oneline -5` at the start of every phase.

## What we measured

### Hardware
i5-8500, **6 cores / 6 threads, no hyperthreading**, AVX2, no AVX-512, 15.8 GB
RAM, no GPU. The box also runs open-webui and `/opt/rag`.

### Model footprints (peak RSS, isolated processes)

| | peak | steady |
|---|---|---|
| CLIP ViT-B/32 alone | 1,670 MB | 1,139 MB |
| InsightFace buffalo_l alone | 1,462 MB | 1,381 MB |
| **both in one process** | **2,054 MB** | 1,927 MB |

buffalo_l's 1,462 MB is 4.5x its 326 MB on-disk size (ONNX arena allocation).
Only ~300 MB shares between processes; ~1.66 GB per process is private dirty.
**Marginal cost of another resident ML process: ~1,700 MB. Marginal cost of
adding buffalo_l to the process already holding CLIP: 384 MB.** That 4.4x ratio
drives D3.

### Throughput (idle box, median of 15)

| threads | CLIP image | CLIP text |
|---|---|---|
| 1 | 164 ms | 21.6 ms |
| 2 | 120 ms | 14.9 ms |
| **3** | **113 ms** | **13.0 ms** |
| 4 | 187 ms | 12.3 ms |
| 6 (default) | 194-220 ms | 12.8 ms |

Sharply non-monotonic — 3 optimal, 4 and 6 ~1.7x worse. Text flat from 3 up.

InsightFace buffalo_l detect, `ctx_id=-1`, `det_size=(640,640)`:

| config | ORT threads added | detect median |
|---|---|---|
| uncapped | +25 | **249 ms** |
| `intra_op_num_threads=2` | +5 | 143 ms |
| **`intra_op_num_threads=3`** | **+10** | **104 ms (2.4x)** |
| `intra_op_num_threads=4` | +15 | 302 ms |

**`OMP_NUM_THREADS` does not reach ONNX Runtime** — the CPU wheel links no
OpenMP. `FaceAnalysis` forwards only `providers`/`provider_options`; there is no
`sess_options` anywhere in the chain, so this needs a `core/face.py` change.

### The per-job model reload
RQ forks per job, so module-level `_EMBEDDER` and `_FACE_APP` die with the child.
**~90% of the 58.5 s embedding job and 37.1 s face job is model reload.**
`chitra-workers` was throttled in 78-81% of periods at `CPUQuota=400%`;
`memory.peak` 3,311 MB against `MemoryMax=8G`.

### Data scale (production, read-only)
- 2,040 photos: 1,012 photo, 262 video, **766 NULL — treat NULL as photo**.
  Non-video: **1,778**.
- **1,694 embeddings, 3.31 MiB.** All `dim=512`. **All exactly L2-normalised**
  (norm min/max/mean 1.000000). One row per photo, zero duplicates today.
- **84 non-video photos have no embedding.** All 1,694 embedded photos have a
  `thumb_path`.
- 2,081 faces. 257 of 262 videos already have a poster.
- 10,158 tags over a **17-label** vocabulary, at exactly **6.0 tags/photo**.
- 1,715 non-video photos carry a 64-bit pHash.

### Two things the tag data proves
Raw CLIP cosine across all 10,158 stored tags:

```
min 0.1584  p1 0.1705  median 0.2037  p99 0.2514  max 0.2775
top-1 per photo: median 0.2249    6th (worst kept): median 0.1934
```

1. The whole range is 0.16-0.28 — an absolute `min_score` carries no meaning, and
   one tuned for 17 labels will not transfer to 300.
2. Top-k always returns exactly k. `travel` is on **1,365 of 1,694 photos
   (80.6%)**; best-to-worst-kept gap is 0.03. Tagging is near-random below rank 1-2.

### pHash near-duplicates are nearly free
All-pairs Hamming over 1,715 hashes: `<=0`: 15 pairs, `<=4`: 112, `<=8`: 242,
`<=12`: 388. **0.18 s wall, 23.5 MB peak, SQLite only.**

### A mixed-model embeddings table 500s
`app_fastapi.py:2039-2054` filters `if v.shape[0] != dim: continue` — which
*passes* a 768-d row whose `dim` says 768 — then `np.stack` raises
`ValueError: all input arrays must have the same shape`.
**One 768-d row makes every search request 500 for every user.**

### There is NO photo-embedding FAISS index to rebuild
This corrects a natural assumption, because "768-d means rebuild the FAISS
indexes" is exactly what a reader expects — and acting on it would discard 549
assigned faces for nothing. Verified:

* `core/faiss_index.py:47` reads `dim = vectors.shape[1]` — dimension-agnostic.
* The only two persisted indexes (`existing_person_faces.index` 406 KB,
  `unmatched_faces_cluster.index` 25 KB) are built from **`faces.embedding`** —
  `core/jobs.py:726` is literally `SELECT f.id, f.embedding ... FROM faces f`.
  These are **buffalo_l 512-d face vectors, not CLIP photo vectors.**
* `core/cluster.py` builds a CLIP index but in memory, per invocation, CLI-only.
* **Photo search uses no FAISS at all** — brute-force numpy GEMV over 3.31 MiB.

**So the 512->768 transition requires no FAISS rebuild.** The dimension coupling
is entirely `search_photos`'s `np.stack`. Task 1.4 pins this with a test.

### Dependency state
`transformers==4.35.0` installed and **absent from `requirements.txt`** — a clean
install cannot embed. `huggingface_hub>=0.16.4,<0.18` **blocks** `transformers
>=4.50` (needs `>=0.30`); `tokenizers==0.14.1` also blocks it (needs `>=0.21`).
`open-clip-torch==2.24.0` installed, unused by any code path.

## Goal

**Today:** no duplicate embeddings/tags; `model` column; search filtered to one
model and unbreakable by foreign dimensions; worker embedding does zero model
loads (~58 s -> ~113 ms); face detection ~104 ms and no per-job load; vocabulary
of a few hundred labels with **every embedded photo re-tagged without one MinIO
read**; four new read-only endpoints; `transformers` pinned and SigLIP's real
footprint measured. Suite no worse than `168 tests, 7 failures, 4 errors, 2 skipped`.

**Blocked:** library re-embedded under SigLIP 768-d; 262 video posters embedded;
the 84 stragglers filled.

## Design decisions

### D1 — Uniqueness and the `model` column come first
All four writers (`db.put_embedding`, `db_async.put_embedding_async`,
`db.add_tag`, `db_async.add_tag_async`) are plain INSERTs. The DB is clean today
only because nobody has run `POST /api/index/embeddings` with
`incremental: false`. Everything else in this plan is a re-index.

```sql
ALTER TABLE embeddings ADD COLUMN model TEXT;
ALTER TABLE tags       ADD COLUMN source TEXT;
UPDATE embeddings SET model = 'openai/clip-vit-base-patch32' WHERE model IS NULL;
UPDATE tags       SET source = 'clip-vitb32/vocab-v1'        WHERE source IS NULL;
CREATE UNIQUE INDEX idx_embeddings_photo_model ON embeddings(photo_id, model);
CREATE UNIQUE INDEX idx_tags_photo_tag         ON tags(photo_id, tag);
CREATE INDEX        idx_tags_tag               ON tags(tag);
```

Both backfills are idempotent single statements over <11k rows.

**`embeddings` unique on `(photo_id, model)`, not `(photo_id)`** — that is what
lets old and new coexist during migration. **`tags` unique on `(photo_id, tag)`
— `source` is provenance, not identity.**

**Is a mixed-model index queryable? No, and the failure is total.** Options:
group-by-dim and merge was rejected (scores from two models are not comparable —
CLIP lives in 0.16-0.28, SigLIP does not; merging silently produces a
meaningless ranking). **Chosen: filter to one active model at query time.**
Writes incremental, reads all-or-nothing. Cutover is one config change; rollback
is changing it back.

### D2 — Route worker image embeddings through the existing sidecar
`embed_service.py` already exposes `POST /embed/image`, built and unused.
~58.5 s -> ~113 ms + download. **Adds no media reads** — the job already
downloads to a temp file; it now uploads that same file over loopback.

**A second win rides along:** `auto_tags` calls `rank_labels`, which calls
`image_embedding` *again* — every job runs **two** CLIP forward passes plus a
17-label text batch recomputed from scratch. With the vector already in hand,
tagging becomes 1 pass, 0 text passes.

**No fallback.** A silent in-process fallback would restore the 58 s path and
1.67 GB residency invisibly.

RQ jobs are sync, `core/embed_client.py` is async: **add a small sync client**
rather than `asyncio.run` per call.

### D3 — Face detection: YES, co-resident in the same process
```
second sidecar process holding buffalo_l:  +1,700 MB
buffalo_l added to the CLIP process:       2,054 - 1,670 = +384 MB
```
**4.4x cheaper co-resident**, measured on this exact model pair.
Face jobs go ~37.1 s -> ~104 ms. **`intra_op_num_threads=3` alone cannot deliver
this** — it fixes ~4 s inside a ~37 s job; only residency removes the reload.

Costs: the sidecar becomes a single point of failure for search, embedding and
detection (mitigated by `Restart=always` + `embed_status`); the single-worker
executor serialises everything (**worst case ~113 ms added search latency** on a
14 ms baseline — accepted, and deliberate: it keeps the 3-thread cap the real
degree of parallelism). Ceiling ~4.6 photos/s.

Lands behind `CHITRA_EMBED_FACES` (default `0`) after D4, so a bad outcome is one
env var from reverted. **Rejected:** having the sidecar read MinIO itself — that
puts reads off the failing disk into the process search depends on.

### D4 — `intra_op_num_threads=3` for InsightFace in `core/face.py`
2.4x (249 -> 104 ms). Wrap `ort.InferenceSession` inside `_lazy_init_insightface`
before constructing `FaceAnalysis`. **Must be scoped and restored** via
`try/finally` — a leaked monkeypatch would silently reconfigure every other ONNX
session in the process. Required whether or not D3 lands.

### D5 — Expand the vocabulary and re-tag TODAY from stored vectors
**The one large user-visible win available before the disk is fixed.**
A tag score is `cosine(image_vec, text_vec)`; the stored blobs **are** the
normalised image vectors (verified: norm 1.000000 on all 1,694). So re-tagging is
`1694x512 @ 512xN` — 260 MFLOP at N=300, single-digit ms over 3.31 MiB. Label
vectors cost 300 x 13 ms = 3.9 s once, cached to `.npy` on the NVMe.
**MinIO reads: zero.**

The 84 unembedded photos and all 262 videos have no stored vector; they stay
behind the gate.

**Calibration:** CLIP cosine spans 0.158-0.278 for everything, and fixed top-6 is
why `travel` sits on 80.6% of the library. With 300 labels top-6 would be worse.
**Chosen: corpus-relative per-label calibration** — keep tag ℓ on photo p only
when `score(p,ℓ)` is above a per-label percentile, plus a per-photo cap of 3-8.
That converts "which of 300 strings is least unlike this photo" into "is this
photo unusually *beach* for this library".

**Honest limit:** a calibration hack around a contrastive objective. Scores stay
comparable within an image, not across images — which is exactly what SigLIP's
sigmoid loss fixes. Say so in the comment; do not oversell.

### D6 — The four endpoints go in a router, not the god-file

| endpoint | reads |
|---|---|
| `GET /api/tags` | `tags` GROUP BY (needs `idx_tags_tag`) |
| `GET /api/search/by-tag` | `tags` join `photos` |
| `GET /api/photos/{id}/similar` | `embeddings` |
| `GET /api/duplicates` | `photos.phash` |

New `core/routes_discovery.py`; the only `app_fastapi.py` edit is one
`include_router` line — which also minimises conflict with the concurrent
clustering work. Every endpoint takes `Depends(get_current_active_user)`.
`/api/duplicates` at 0.18 s must run under `run_in_executor`.

### D7 — SigLIP 2: split into measurable-today and blocked
Gate is `SiglipProcessor.tokenizer_class` -> `AutoTokenizer` in **4.50.0**, not
the model class. Full checkpoint 1.50 GB fp32, **52.4% is a [256000,768]
multilingual token embedding**; vision tower 371 MB fp32 / 186 MB fp16.
768-d output. Encoder 2.84x -> ~321 ms/image -> ~7.3 min compute for a full
re-embed. Compute is not the problem; the reads are.

**Tension the brief did not resolve: `SiglipVisionModel` alone cannot serve
free-text search.** `/api/search/photos` takes arbitrary text and the text tower
holds the 786 MB embedding.

| option | consequence |
|---|---|
| vision only | tagging works; **free-text search dies**. Unacceptable |
| full fp32 | +~1.1 GB over vision-only |
| text tower fp16 | ~564 MB text side; needs a quality check |
| text tower loaded only to precompute vocab | search still dead; adjunct only |

**Decision: do not choose in advance.** Task 4.3 measures all of it, reading no
media.

**Second risk: bumping transformers may change CLIP's own preprocessing.**
4.35 -> 4.50+ crosses the slow->fast image-processor default switch. Any drift
means every new embedding diverges from the 1,694 stored ones with no error
anywhere. Testable with a synthetic image, reading nothing.

### D8 — Video posters and the 84 stragglers are cheaply gated
257 videos already have a poster: 512x512 JPEG, ~250 KB, **~64 MB total** vs
69 GB for originals. Still a read off `/dev/sda`, so still gated — but by far the
cheapest thing behind the gate.

Same arithmetic for the re-embed: siglip2-base-patch16-**224** consumes 224x224,
so a 512x512 thumbnail is a strict superset. **Re-embedding from thumbnails is
~424 MB instead of ~69 GB — 160x less.** Caveats making it an owner decision:
thumbnails are re-compressed and LANCZOS-downscaled (difference unmeasured), and
it must be **all-or-nothing per model generation** or the vector space is
inconsistent and ranking goes subtly wrong with no error.

## Sequencing

Phase 1 first despite no user-visible effect: every other phase is a re-index,
and without it a re-index doubles the data; the `np.stack` failure means the
`model` column is what stops the first SigLIP row taking search down.
Phase 2 second: largest throughput win per line, no contract change, no extra
media reads. Phase 3 third: biggest user-visible win available today.
Phase 4 fourth. SigLIP dependency work last on the green side — riskiest, and the
only green task safely abandonable.

## Read cost, for reference

Phases 0-5 read **zero** MinIO objects (one single-object exception: Task 0.2's
timing baseline). Phase 6 is the only bulk-read phase, and the thumbnail path
keeps it to ~424 MB rather than ~69 GB. Phase 7's videos ride the same pass at
**zero additional I/O** if sequenced together — which is the recommendation.

## Tasks

TDD throughout: write the test, **run it and paste the failing output into
Evidence**, then implement. Baseline `168 tests, 7 failures, 4 errors, 2 skipped`
— counts may rise, **failures and errors must not**.

```bash
CHITRA_DB_PATH=/tmp/chitra_test.db .venv/bin/python tests/run_tests.py 2>&1 | tail -5
```

**Concurrency note.** Another agent owns `app_fastapi.py` ~1240-1260 and
~1345-1372, and in `core/jobs.py` the module-level `FACE_MATCH_THRESHOLD`, the
`_auto_match_face_to_person` signature and its call sites at `:301`/`:459`. Keep
every hunk inside job-function bodies; rebase immediately before committing;
re-check `git log --oneline -5` at the start of each phase.

### Phase 0 — Baseline
- **0.1** Record suite counts, cgroup `memory.peak`/`cpu.stat`, sidecar `/health`,
  and four data counts from a read-only connection
  (`sqlite3 'file:photo.db?mode=ro&immutable=1'`).
- **0.2** Time one real embedding job and one real face job on an
  already-uploaded photo. **The only MinIO read before Phase 6** — a single
  object, and the baseline every later "Nx faster" claim measures against.
- **0.3 — Verify what Agent B landed, and check the contract.** THE HIGHEST-VALUE
  CHECK IN THIS PLAN. Record, from schema and code rather than expectation:
  1. **The unique key on `embeddings`.** If `(photo_id, model)` — proceed. **If
     `(photo_id)` alone, STOP AND RAISE IT.** A bare `photo_id` key means the
     first SigLIP row *evicts* the CLIP row it is meant to run alongside —
     coexistence and rollback both vanish, on a disk where an interrupted run is
     the expected case. Do not drop and recreate their index underneath them.
  2. Whether `embeddings.model` already exists (if so, Phase 1 shrinks to the
     search-handler half).
  3. Whether `scripts/reembed.py` accepts **`--model`** and populates that
     column. If not, request it — Phase 6's cutover and rollback have nothing to
     key on without it.
  4. Its flags: `--apply`, `--force`, `--limit`, resumability, guard behaviour.
  5. Whether it loads the model in-process or calls the sidecar. In-process
     alongside the sidecar is 1,670 + 2,054 = **~3.7 GB concurrently** in a
     cgroup capped at 8G — fits, but worth knowing before the run.

### Phase 1 — Make re-indexing safe (SQLite only)
- **1.1** Red tests: double `put_embedding` for same `(photo_id, model)` leaves
  one row; two different models leave two; `add_tag` idempotent; old-schema DB
  gains both columns backfilled non-NULL plus the indexes; same for `_async`.
- **1.2** `scripts/dedupe_embeddings.py`, dry-run by default. **The unique index
  in 1.3 will raise on any DB with duplicates** — this is the escape hatch.
  Production measured clean (0 groups both tables).
- **1.3** Add the ALTERs, backfills and indexes to **both** DDL copies. The two
  `CREATE UNIQUE INDEX` must log loudly on failure — a swallowed failure means
  the constraint is absent and nobody knows. Do not attempt to reconcile the
  wider `db.py`/`db_async.py` divergence.
- **1.4** Convert the four writers to UPSERTs with optional `model=`/`source=`
  defaulting to the current CLIP identifier.
- **1.5** Red test: with 512-d **and** 768-d rows present the endpoint returns 200
  and ranks only the active model. **Watch it fail with the real
  `ValueError`** — that traceback is the production bug. Then add a `model`
  filter driven by `CHITRA_ACTIVE_EMBED_MODEL`.
- **1.4 — Pin that the face FAISS indexes are untouched by a CLIP model change.**
  Assert `existing_person_faces.index` is built from `faces.embedding`
  (buffalo_l 512-d), not `embeddings.vector`, and that `build_hnsw_index` derives
  `dim` from its input. Write the rationale into the docstring: "768-d means
  rebuild the indexes" is the natural assumption and it is **wrong here** —
  acting on it would discard 549 assigned faces for nothing. Record the finding
  in `.claude/rules/ml-pipeline.md`.
- **1.6** Guard `POST /api/index/embeddings`: prove a double non-incremental run
  leaves counts unchanged, and skip videos (`COALESCE(media_type,'photo')`).

### Phase 2 — Kill the per-job model load (no new media reads)
- **2.1/2.2** `SyncEmbeddingClient` in `core/embed_client.py`, sharing the wire
  format. `httpx==0.28.1` already pinned — **no new dependency**. Timeout must
  exceed the async client's 5 s (cold sidecar takes ~10 s):
  `CHITRA_EMBED_JOB_TIMEOUT`, default 60 s. Assert **in a fresh subprocess** that
  importing it leaves torch out of `sys.modules`.
- **2.3/2.4** Reroute `process_photo_embedding_job` and
  `_process_single_embedding`. Assert on a `ClipEmbedder` stub **that raises if
  instantiated**; exactly one image embed per photo, not two; sidecar failure
  re-raises; **MinIO downloaded exactly once** (counting stub). Do not delete
  `_get_embedder` yet.
- **2.5/2.6** InsightFace thread cap, with the restore assertion.
- **2.7** Re-time both jobs against 0.2.
- **2.8/2.9** `POST /detect/faces` behind `CHITRA_EMBED_FACES` (default `0`).
  bbox must be in **original-image pixel coordinates** — the job crops
  full-resolution thumbnails from those numbers. Touch only the
  `faces = face_encodings(...)` line in each job.
- **2.10** Owner decision point: adopt `CHITRA_EMBED_FACES=1` **only** if sidecar
  RSS lands within ~10% of 2,054 MB. Near 3,100 MB means two arenas and the
  co-residency premise was wrong — stop and report.

### Phase 3 — A vocabulary worth searching (SQLite only)
- **3.1/3.2** `core/vocabulary.py`: a few hundred labels grouped by facet, all 17
  legacy labels retained, `PROMPT_TEMPLATE = "a photo of {label}"`, and a
  `vocab_fingerprint()` over (labels, version, template).
- **3.3/3.4** `tag_from_vector(...)` **pure** — no model, no file I/O, no network;
  assert against a hand-built orthogonal matrix. Per-label calibration; per-photo
  count varies (**not always exactly 6** — that is the regression that matters).
  Cache refuses to load on fingerprint or model mismatch.
  **Remove `core/tagger.py:4`'s module-scope `ClipEmbedder` import** — it drags
  torch into anything importing the tagger.
- **3.5/3.6** `retag_from_embeddings_job`, with **a storage stub that raises on
  any call** — that is the load-bearing assertion of this phase. Dry-run against
  a **copy** of production and confirm `travel`'s share falls far below 80.6%.
- **3.7/3.8** `POST /api/index/retag` (admin-only), then run it: ~4 s of text
  embedding, ~20 ms of GEMM, ~11k upserts, **zero MinIO reads**.

### Phase 4 — Endpoints and SigLIP groundwork
- **4.1/4.2** `core/routes_discovery.py` + one `include_router` line. Override
  auth in tests so they do not join the 7 known failures. Duplicates test must
  include a transitive-closure case (A~B, B~C, A!~C). `/api/duplicates` under
  `run_in_executor`.
- **4.3** Measure SigLIP's real footprint — vision/text/both x fp32/fp16, plus
  co-resident with buffalo_l. Use **`VmHWM`, not `ru_maxrss`** (a subprocess
  child inherits the parent's peak and over-reports). Re-measure the thread curve:
  **the 3-thread optimum is a CLIP result and may not transfer** — patch16-224
  has 4x the tokens. Reads no media. If the co-resident row exceeds what the box
  tolerates, **say so and stop** — that is a finding, not a failure.
- **4.4** Red test for CLIP stability across the bump: embed a **fixed-seed
  synthetic PIL gradient** (no photo, no MinIO) against a checked-in reference,
  tolerance 1e-4, behind `CHITRA_TEST_LOAD_MODELS=1`. **This is the entire safety
  net for 4.5.**
- **4.5** Fix the dependency chain: **add `transformers>=4.50,<5`** (currently
  absent), relax `huggingface_hub` to `>=0.30`, add `tokenizers>=0.21`, remove
  `open-clip-torch`. Install into a **scratch venv first**. **If 4.4 fails,
  stop** — drift converts a dependency bump into a gated full re-embed.
- **4.6** Optional: gate `tests/test_search.py:setUpClass`'s real `ClipEmbedder`
  behind the same flag. **Changes the documented baseline** — update `AGENTS.md`
  in the same commit or skip.

### Phase 5 — transformers upgrade and SigLIP staging (no media reads)

- **5.1 — Red test for CLIP stability across the bump.** Embed a **fixed-seed
  synthetic PIL gradient generated in the test** (no photo, no MinIO) against a
  checked-in reference vector, tolerance 1e-4, behind `CHITRA_TEST_LOAD_MODELS=1`.
  **Done when it PASSES on 4.35.0**, establishing the reference. This is the
  entire safety net for 5.2.
- **5.2 — Fix the dependency chain.** **Add `transformers>=4.50,<5`** (currently
  absent — a clean install cannot embed), relax `huggingface_hub` `<0.18` ->
  `>=0.30`, add `tokenizers>=0.21`, remove `open-clip-torch` (unimported).
  **Scratch venv first.** Re-run 5.1. **If it fails, STOP** — drift converts a
  dependency bump into a forced full re-embed on a failing disk.
- **5.3 — Measure SigLIP's footprint.** Isolated processes, **`VmHWM`, not
  `ru_maxrss`** (a subprocess child inherits the parent's peak). Table: vision-only
  fp32/fp16, full fp32, text-tower fp16, and **co-resident with CLIP + buffalo_l**.
  Re-measure the thread curve — **the 3-thread optimum is a CLIP result and may
  not transfer**; patch16-224 has 4x the tokens. **If the text tower cannot fit
  alongside buffalo_l, say so and STOP** — vision-only breaks free-text search,
  and that is a finding, not a failure.
- **5.4 — `SiglipEmbedder` behind `CHITRA_EMBED_MODEL`**, same interface as
  `ClipEmbedder` so sidecar, sync client and tagger need no changes. 768-d.
- **5.5 — Red test for the dual-model sidecar.** `CHITRA_EMBED_MODEL` and
  `CHITRA_ACTIVE_EMBED_MODEL` must be **separable** — that is what lets the
  sidecar serve SigLIP while search still answers from CLIP.
- **5.6 — Optional:** gate `tests/test_search.py:setUpClass`'s real
  `ClipEmbedder` behind the same flag. **Changes the documented baseline** —
  update `AGENTS.md` in the same commit or skip.

### Phase 6 — Re-embed and cut over (uses `scripts/reembed.py`)

This plan does **not** specify a competing pipeline. Agent B owns the mechanism.

- **6.1 — Pilot on 50 photos.** `--limit 50 --apply`. Confirm rows land with
  `model` populated and `dim=768`, **the 512-d rows are still present** (if they
  were evicted, the unique key is wrong and Task 0.3 missed it), and search is
  unaffected throughout.
- **6.2 — Full pass.** `--force --apply`. ~7.3 min of compute; the reads are the
  cost, mitigated by the thumbnail path. **If the disk guard aborts, STOP AND
  REPORT** — resumability means an abort costs progress, not correctness.
- **6.3 — Verify coverage before cutting over.** SigLIP row count must match
  CLIP's. **Do not flip on partial coverage** — search would silently lose every
  photo not yet re-embedded, with no error.
- **6.4 — Cut over.** Flip `CHITRA_ACTIVE_EMBED_MODEL` and `CHITRA_EMBED_MODEL`.
  One config change, because Phase 1 did the work.
- **6.5 — Soak, then retire.** **Keep the 512-d rows a week** — 3.31 MiB, and they
  are the rollback. Then one `DELETE ... WHERE model = 'openai/clip-vit-base-patch32'`.
- **6.6 — Re-tag under sigmoid scoring.** Where D5's calibration hack goes away:
  SigLIP's sigmoid loss makes scores comparable *across* images, so an absolute
  threshold finally means something. Mechanically 3.6 re-run. **Zero MinIO reads.**

**Rollback in full:** flip `CHITRA_ACTIVE_EMBED_MODEL` back. Search returns to the
512-d rows instantly, the 768-d rows sit inert, tags revert with
`DELETE FROM tags WHERE source LIKE 'siglip%'` plus a re-run of 3.6 under CLIP.
Nothing needs re-embedding and no FAISS index is involved. This survives an
interrupted or abandoned migration — the case worth designing for here.

### Phase 7 — Videos and the tails (mostly free)

**258 videos already have a poster**, and `scripts/reembed.py` reads `thumb_path`
— so they are **already in its input set**. The only thing excluding them is the
`_is_video` skip.

- **7.1/7.2 — Lift the skip for the poster path only.** `_is_video` must keep
  blocking ML on the **original**; assert a video embedding job reads
  `thumb_path`, not `file_path` (the latter pulls gigabytes instead of ~250 KB).
  **Zero additional disk I/O if this rides Phase 6's pass** — the recommendation.
- **7.3 — Update `.claude/rules/ml-pipeline.md`**: "videos get no ML" becomes
  "videos get no ML from the original; posters are embedded", same commit. A
  stale scoped rule is worse than none.
- **7.4 — The tails.** 19 photos with a thumb but no embedding (free, same path);
  65 with neither (generate thumbnails first); 5 videos with no poster
  (`generate_video_poster_job`, reads the original).

## Owner actions (sudo)

1. **Evacuate MinIO off `/dev/sda`. This is the gate and the priority.** The NVMe
   has 150 GB free against a 69 GB library — the destination already exists. Do
   not delete the `/dev/sda1` copy until a second verified copy exists.
2. **Restart workers** after Phase 2 via `scripts/safe_restart.sh` — prove the
   video queue is idle first. A worker restart also restarts the sidecar, so
   search 503s for ~10 s (~20 s once buffalo_l co-loads).
3. **After 2.10**, if `CHITRA_EMBED_FACES=1`: promoting the sidecar to
   `chitra-embed.service` becomes clearly worth it. **`WorkingDirectory` is
   load-bearing** — `faiss_indexes/` is a relative path.
4. **Re-measure worker CPU throttling** after Phase 2 (was 78-81%). **Do not
   raise `CPUQuota` before re-measuring.**
5. **Run the re-tag (3.8).** Zero MinIO reads; back up `photo.db` first anyway.

## Risks

| risk | mitigation |
|---|---|
| **Agent B's unique key is `(photo_id)`, not `(photo_id, model)`** | Task 0.3 checks it BEFORE anything builds on it. A bare key makes coexistence and rollback impossible. Raise it; do not work around it. |
| **`scripts/reembed.py` does not populate `embeddings.model`** | Task 0.3 checks; `--model` is the one addition Phase 6 depends on. Without it the cutover and rollback have nothing to key on. |
| Disk-health guard aborts mid-run | Stop and report. Resumability means an abort costs progress, not correctness. Do not add an override. |
| Cutover on partial coverage silently hides photos | Task 6.3 gates the flip on a row-count match. |
| Sidecar down -> embedding and face jobs fail | Deliberate. Jobs re-raise, RQ shows failed, `/api/health` reports it. A silent fallback would restore the 58 s path invisibly. |
| Sidecar = SPOF for three subsystems | `Restart=always`; own unit; 113 ms worst-case queueing on a 14 ms baseline. |
| Unique index fails on pre-existing duplicates | `scripts/dedupe_embeddings.py` first; production measured clean. |
| A 768-d row reaches an unfiltered search -> **every query 500s** | Task 1.5 lands before any SigLIP row can be written; its red test *is* that `ValueError`. |
| transformers bump silently changes CLIP preprocessing | Task 4.4's synthetic reference vector, established before and re-run after. Failure stops the bump. |
| Merge conflict with the clustering agent | Only `include_router` touches the god-file; `core/jobs.py` hunks stay inside job bodies; rebase per commit. |
| Expanded vocabulary tags worse than 17 labels | Per-label calibration; dry-run distribution against a DB **copy** before any write; `source` makes revert one DELETE. |
| SigLIP text tower does not fit alongside buffalo_l | 4.3 measures before commitment. A negative result is valid and costs only the measurement. |
| Thumbnail re-embed degrades quality | Owner decision with numbers; all-or-nothing per generation; 512-d rows stay as rollback. |

## Rollback

One commit per phase. Phase 1 additive-only. Phase 2 reverts to fork-and-reload
(slow, not broken); `CHITRA_EMBED_FACES=0` reverts D3 with no code change.
Phase 3 reverts via `DELETE FROM tags WHERE source = ...` — what the `source`
column is for. Phase 4 endpoints additive. Phase 7 reverts by flipping
`CHITRA_ACTIVE_EMBED_MODEL` back, because 7.7 deliberately waits.

## Explicitly out of scope

**The `put_embedding`/`add_tag` uniqueness fix and `scripts/reembed.py` — Agent B owns both.** VLM captioning via Ollama (ruled out on measurement: 7.7-10 tok/s at 534% CPU,
10-24 h backlog, 7.6x API latency degradation — **do not reintroduce**);
pre-existing baseline failures; `app_fastapi.py:1524-1913` dead code; the
face-clustering `depends_on` change and threshold unification (another agent);
the health endpoint returning 200 while degraded; client changes.

## Follow-ups

1. `_auto_match_face_to_person` rebuilds a FAISS index over every assigned face,
   once per detected face. With D3 making detection ~104 ms this becomes the
   dominant per-face cost.
2. FAISS index writes are non-atomic and unlocked across four workers.
3. A batch endpoint on the sidecar — only worth it for 7.4.
4. `core/cluster.py` is CLI-only; with `/similar` shipping, an albums feature is close.
5. Drop the `nvidia-*` wheels — dead weight on a GPU-less box.
6. `logs/` truncated on every restart and unrotated.

## Evidence

_(Appended by the executing agent as tasks complete.)_

### Phase 3 — vocabulary and re-tag (complete, not applied)

Commits `c8856d4`, `a1857b7`, `808d7f9`, `8351d55`, `94f0317`. Zero MinIO reads.

**Vocabulary.** 345 labels over nine facets — scene 52, place 57, activity 53,
object 71, people 27, occasion 30, photo_style 32, time_of_day 12, season 11.
All 17 legacy labels retained; `core/vocabulary.py` refuses to import if one is
ever dropped. `vocab_fingerprint()` = `dd82e956e98f8e77` over labels, order,
version and `PROMPT_TEMPLATE`.

**Calibration.** Per-label percentiles learned from the whole corpus: p90 floor,
p98.5 keep, per-photo 3-8. Per-label coverage is then structurally bounded by
`100 - low_percentile`, which is the mechanism — not a tuning accident — that
prevents another `travel`.

**Dry run, copy of production (`/tmp/retag_test.db`), 1,930 vectors, dim 512:**

```
tags/photo   min 0  median 3.0  mean 4.39  max 8   (was: exactly 6.0, always)
             0:12  1:12  2:33  3:1103  4:131  5:109  6:79  7:60  8:391
worst label  wedding 3.6%                          (was: travel 80.9%)
```

| legacy label | before | after | | legacy label | before | after |
|---|---|---|---|---|---|---|
| travel | **80.9%** | **1.8%** | | wedding | 32.5% | 3.6% |
| outdoors | 46.8% | 1.0% | | family | 31.2% | 2.0% |
| portrait | 45.3% | 1.5% | | sports | 30.9% | 1.1% |
| food | 39.7% | 1.9% | | indoors | 30.6% | 1.2% |
| landscape | 38.8% | 1.2% | | party | 29.9% | 1.1% |
| selfie | 38.3% | 2.1% | | night | 22.1% | 1.3% |
| group photo | 36.9% | 2.4% | | pets | 13.6% | 1.1% |
| friends | 36.3% | 1.7% | | sunset | 10.9% | 1.0% |
| city | 35.1% | 1.0% | | | | |

Distribution alone does not prove the tags are *right*, so a random sample was
read back (filenames only, still no media). Old vs new on the same photo:

```
IMG_20260822_154139  old: food, indoors, night, portrait, selfie, travel
                     new: studying, reading, library, working on a laptop, book
IMG_5344             old: family, food, friends, outdoors, portrait, travel
                     new: bird, monkey, wildlife
IMG_5303             old: city, indoors, party, sports, travel, wedding
                     new: nightclub, concert, concert hall
```

**Sensitivity**, so tuning needs no code change (`--high-percentile`):

| p | mean tags | median | at floor (3) | at cap (8) | travel |
|---|---|---|---|---|---|
| 99.0 | 3.93 | 3.0 | 1,313 | 245 | 1.5% |
| **98.5 (default)** | **4.39** | **3.0** | **1,103** | **391** | **1.8%** |
| 97.0 | 5.50 | 5.5 | 653 | 783 | 1.9% |
| 96.0 | 6.07 | 8.0 | 447 | 994 | 2.2% |

The count distribution is bimodal at the floor and the cap, which is real
structure rather than an artefact: label scores are strongly correlated, so a
photo is either generically unlike everything or distinctively like a cluster of
related labels. 57 photos (3%) fall below p90 on all 345 labels and get 0-2
tags. Top-k could not express that, and it is the honest answer for them.

**Not applied.** Production `photo.db` is untouched — still 17 distinct tags,
one source, `travel` on 1,562 photos. Task 3.8 is the owner's to run:

```bash
cp photo.db photo.db.bak-$(date +%F)
CHITRA_DB_PATH=/tmp/chitra_test.db .venv/bin/python scripts/retag.py --db photo.db --apply
```

Rollback: `DELETE FROM tags WHERE source = 'clip-vitb32/vocab-v2';` then re-run
the legacy path. Label matrix is cached at
`models/tag_vectors_clip-vitb32_dd82e956e98f8e77.npy` (706 KB, NVMe), so the
apply run skips the ~21 min of text embedding this dry run paid — that time was
sidecar contention from concurrent agents, not the ~4.5 s the work costs on an
idle box. The GEMM itself is milliseconds.

**Not done, deliberately:** `POST /api/index/retag` (3.7). `app_fastapi.py` is
owned by another agent this session, and a standalone script is the safer entry
point for an 11k-row rewrite anyway — it cannot be triggered by a stray request.
