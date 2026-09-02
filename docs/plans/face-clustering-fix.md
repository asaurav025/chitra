# Face clustering — make discovery work again, and prove it

Status: ready to execute. Branch from `main`.
Owner actions (backups, the production run, any restart) are in one section —
agents must not run them.

Successor to the 2026-09-01 trigger fix (`enqueue_in` -> `depends_on`). **That
fix worked and this plan is not about it**: verified 2026-09-02, photo 2775's
face job ended 06:38:55.398Z and the dependent cluster job was enqueued
06:38:55.403Z — 5 ms. Clustering runs. It just cannot discover a person.

## What we measured

Measured 2026-09-02 against `file:photo.db?mode=ro`. No MinIO reads.
**Do not re-derive.**

| | |
|---|---|
| `faces` with an embedding | **3,289** |
| assigned | **1,410** |
| unassigned | **1,879** |
| `persons` | 19 (**18 with faces**; `Person 22` has zero) |
| dim / norm | 512, L2 norm 0.9999999–1.0000001 |

`.claude/rules/ml-pipeline.md` and `api-oom-fix.md` both say **2,081 faces** —
stale by 1,208 rows. Task 6.1 fixes that.

Indexes: `existing_person_faces.index` ntotal **1,413**, id-mapped, 3 ahead of
the DB (within `_INDEX_CATCHUP_LIMIT`=64, and `_resolve_person` reads ownership
from SQLite, so drift costs recall not correctness). `unmatched_faces_cluster.index`
is an 11-vector orphan referenced only from the dead block at `app_fastapi.py:1943`.
`faiss_indexes/` also holds six test artefacts from runs without
`CHITRA_FAISS_INDEX_DIR`.

### Similarity, refreshed on 1,410 labelled faces

```
within-person:   mean 0.583  median 0.590  1st pct 0.249
between-person:  mean 0.046  99th 0.239  99.99th 0.337  max 0.383
```

Separation is intact — worst impostor 0.383, and 0.60 clears it by 0.217.
**`FACE_MATCH_THRESHOLD` does not change.**

Per-person mean pairwise cohesion **did** change:

| person | n | cohesion | | person | n | cohesion |
|---|---|---|---|---|---|---|
| swati | 516 | **0.554** | | Person 3 | 15 | 0.699 |
| saurav | 436 | 0.621 | | Person 5 | 14 | 0.709 |
| mummy | 97 | 0.601 | | Shekhar | 14 | 0.702 |
| papa | 69 | 0.624 | | Person 2 | 36 | 0.660 |
| badi mummy | 69 | **0.529** | | Person 1 | 20 | 0.648 |
| dada ji | 4 | 0.743 | | Person 4 | 3 | 0.691 |

The rules file records 0.590–0.809. It is now **0.529–0.743**, and cohesion
falls as a person grows — the two below 0.60 have 516 and 69 faces; everyone
above 0.69 has <= 15.

### The failure, reproduced against ground truth

Current Phase-2 config over the 1,410 labelled faces, labels hidden:

```
min_cluster_size = max(2, int(1410*0.01)) = 14 ; min_samples = 7
cluster_selection_epsilon = sqrt(2*(1-0.60)) = 0.894 ; allow_single_cluster = True

-> 1 cluster holding 88.2% of faces, mixing 12 of 18 persons
   B-cubed precision 0.365 / recall 0.831
-> mean pairwise 0.144, rejected by the acceptance gate
-> 0 clusters kept, 0 faces assigned
```

The gate is right to reject it. The clusterer had already lost.

### Two defects, not one

**1. `min_cluster_size` scales with `n`.** 14 at n=1,410, 18 at n=1,879. A person
in 5 photos is a person whether the library holds 500 faces or 50,000. With
`epsilon = 0.894` flattening every distance below cosine 0.60 into one blob and
`allow_single_cluster=True`, the only cluster findable is "everybody".

**2. The acceptance gate is now empirically wrong.** A cluster is kept only if
mean pairwise >= 0.60. **swati (0.554) and badi mummy (0.529) would both be
rejected** — 585 of 1,410 labelled faces, 41.5%. The rules file's "real people
average 0.59–0.81" was true of 549 faces, not of 1,410. The bias is
size-dependent in the wrong direction: the gate tightens exactly as a cluster
becomes the person you most want. **Fixing only defect 1 leaves the pipeline
able to find swati and then throw her away.**

### Algorithm bake-off

Labels hidden, all 3,289 faces clustered, scored on the 1,410 labelled.
"merges" = clusters holding >= 2 faces of each of two known persons.

| algorithm | clusters >=2 | assigned | B3P | B3R | merges | wall |
|---|---|---|---|---|---|---|
| **production HDBSCAN + gate** | **0** | **0** | — | **0.000** | 0 | 3.1 s |
| HDBSCAN mcs=3 ms=1 eom | 362 | 2,200 | 1.000 | 0.102 | 0 | 8.8 s |
| HDBSCAN mcs=5 ms=1 eom | 83 | 2,541 | 1.000 | 0.967 | 0 | 8.7 s |
| Chinese Whispers kNN20 >=0.50 | 133 | 2,472 | 1.000 | 0.579 | 0 | 1.0 s |
| single-link kNN10 >=0.60 | 105 | 2,291 | 1.000 | 0.946 | 0 | 0.1 s |
| single-link >=0.40 | 106 | 2,683 | 0.812 | 0.988 | **1** | 0.1 s |
| agglomerative avg >=0.45 | 142 | 2,407 | 1.000 | 0.926 | 0 | 1.8 s |
| **block >=0.55 + split >=0.45** | **123** | **2,337** | **1.000** | **0.926** | **0** | **0.2 s** |

The merge at single-link 0.40 is `{saurav:435, badi mummy:69, mummy:97,
swati:1, Amar:1}` — three real people in one row. At 0.35 it is ten; at 0.30,
sixteen.

### Two results that decided it

**Single-link has a cliff at 0.40–0.45; the split step removes it.** The hybrid
produces **zero merges at every blocking threshold 0.40–0.60**, B3R varying only
0.879–0.882. The blocking threshold stops being a risk parameter.

**Leave-one-person-out discovery.** Unassign one whole person, run Phase 1
against the other 17, run discovery on the residue:

```
proposed hybrid:    1,354 of 1,410 recovered (96.0%); 16 of 18 as one cluster
                    purity 1.000 for 13 of 18; lowest 0.625 (Arnab, 15 faces)
production HDBSCAN: 0 of 1,410. Zero, for all 18.
```

### Phase 1 is fine

30% holdout per person (seed 7, 422 faces), matching at 0.60:

```
415 of 422 matched correctly, 0 wrong. recall 0.983, precision 1.000
the 7 that fell through reached discovery and formed no mixed cluster
```

**Phase 1 needs no change.**

### Predicted effect on today's DB

```
Phase 1 (>=0.60):        28 of 1,879 matched
Discovery pool:          1,851
  711 no neighbour >=0.55   -> genuine one-offs
  200 in pairs              -> declined by min_size=3
  940 in a block of >=3
Discovery (0.55/0.45) -> 99 new persons, 918 assigned, 933 noise
     largest: 108, 87, 68, 43, 33, 28, 24, 22, 21, 17
```

Against today's **0 new persons, 28 assigned**.

### Cost

```
read 3,289 embeddings   0.02 s
FAISS kNN k=31          0.13 s
union-find 48,313 edges 0.05 s
average linkage         0.12 s
                        0.32 s wall, 157 MB peak
```

vs HDBSCAN's 8.7 s. **Import weight nearly bit us:** baseline 9 MB, +numpy+faiss
34 MB, **+sklearn.cluster 122 MB**, +scipy.cluster.hierarchy 67 MB.
`app_fastapi` imports `core.jobs` at module scope against a 200 MB budget, so a
module-scope sklearn spends 88 MB on a function the API never calls — the
`core/tagger.py -> core.embedder` mistake again. **scipy's
`linkage(method='average')` is the same algorithm for a third of the import.**

### Four findings that change the picture

**`cli faces-cluster` is more destructive than `recluster_all.py`, and nothing
says so.** `cli/main.py:322` calls `db.get_faces_embeddings` = `SELECT id,
embedding FROM faces` with **no `person_id` filter**, then `db.set_face_person`
on *every* face. It does not reset; it overwrites. It names clusters via
`db.get_or_create_person`, which **returns the existing row** when the name is
taken — and `Person 1/2/3/5` exist today — so it dumps strangers into persons the
owner curated. It creates a person per singleton (710 today). It never updates
the FAISS index. No prompt, no dry run, no `--yes`.

**`POST /api/index/faces-cluster` forwards `reset` from the request body**
(`app_fastapi.py:1745`). One admin POST nulls 1,410 assignments.

**"Two faces in one photo are different people" is FALSE here — do not
implement it.** 74 of 1,015 same-photo labelled pairs share a person (7.29%),
and **all 74 have bbox IoU exactly 0** — collages, photos-of-photos, mirrors.

**The ground truth is contaminated.** Most of the 1,410 came from Phase 1
(nearest-assigned-neighbour >= 0.60), so the labelled set is by construction
single-link-connected at 0.60, and any single-link-flavoured algorithm scores
optimistically on **recall**. Precision is not contaminated the same way — a
merge event is real evidence. **Trust precision, discount recall**, and add
provenance (D7) so the next evaluation is clean.

## Goal

1. A bulk pass creates ~90–100 persons and assigns ~900 faces, vs 0 and 28.
2. No parameter is a function of corpus size.
3. Zero merges on the labelled set, with >= 0.10 margin to the measured cliff.
4. Per-upload matches; discovery is a separate deliberate bulk job.
5. `scripts/evaluate_face_clustering.py` reproduces every number here in under a
   minute from a DB copy, reading no media.
6. No path destroys a manual assignment without an explicit human-typed flag.
7. Suite no worse than baseline **649 / 7F / 5E / 13S** (measure per AGENTS.md
   with a detached worktree; failures and errors are the signal, not totals).

## Design decisions

### D1 — Block-and-split, not HDBSCAN

FAISS kNN (k=30) -> keep edges >= `link` (0.55) -> union-find blocks ->
average-linkage split inside each block at `1 - split` (0.45) -> keep >= 3.
Blocks of 2 pass through; blocks of 1 are noise. The structure `core/cluster.py`
already uses for photos.

| option | verdict |
|---|---|
| **Block-and-split** | **chosen** — 0 merges, B3R 0.926, 96.0% LOPO, 0.2 s, insensitive to its own blocking threshold 0.40–0.60 |
| HDBSCAN absolute (mcs=5 ms=1) | rejected — *works* (B3R 0.967) but mcs=3 gives 0.102. A 9.5x swing between adjacent values of an underivable parameter is not shippable on a growing library. 40x slower. |
| single-link alone | rejected — best raw (0.946) but safety is a property of the threshold: 0.40 merges three people, 0.30 merges sixteen |
| Chinese Whispers (dlib's) | rejected on measurement — B3R 0.579, fragments the large persons, and this library's value is two people with 436 and 516 faces. Also stochastic. |
| agglomerative alone | rejected on scaling — same quality, but O(n^2): 3,289 is 43 MB, 50,000 is **10 GB** |
| same-photo must-not-link | rejected on measurement — 7.29% violation |

**Falsifiers:** any merge at the operating point (hard stop); the cliff rising
within 0.10 of `link`; LOPO below 0.90 while HDBSCAN-absolute stays above; a
block over ~20,000 faces.

### D2 — Absolute parameters, derived from the distributions

```python
FACE_CLUSTER_LINK     = 0.55   # blocking
FACE_CLUSTER_SPLIT    = 0.45   # average-linkage acceptance
FACE_CLUSTER_MIN_SIZE = 3
FACE_CLUSTER_KNN      = 30
FACE_MATCH_THRESHOLD  = 0.60   # unchanged, Phase 1 only
```

* **link 0.55** — above the 99.99th between-person percentile (0.337) by 0.21
  and the observed max (0.383) by 0.17; below the within-person median (0.590).
  k=10/20/30 identical at 0.60; k=5 loses recall.
* **split 0.45** — average linkage compares group means and the lowest real
  person means 0.529. 0.50 fragments swati and badi mummy (B3R 0.879); 0.45 does
  not (0.926); 0.40/0.35 gain nothing and approach bridging.
* **min_size 3** — the smallest person the owner has named has 3 faces. 2 costs
  no precision but proposes 100 extra pair-persons: a People page nobody can
  use. Reversible — pairs are picked up when a third photo arrives.
* **`min_cluster_size` as a function of `n` is deleted and must never return.**
  Task 1.1 pins it with a test at n=100 and n=5,000.

**The mean-pairwise gate is deleted, not retuned.** It backstopped
`allow_single_cluster`; with the blob gone it is only a size-dependent bias that
rejects the two largest real people. `min_size` and the split are the acceptance
criteria. If a sanity check is wanted later, use *median within-cluster nearest
neighbour*, which does not decay with size — but measure before gating.

### D3 — Matching per upload, discovery as a bulk job

Discovery over one batch is meaningless (a stranger appearing once per upload
never accumulates) and harmful (a two-face batch cluster becomes a `Person N`
the next identical stranger does not join).

* `cluster_faces_job(db_path, threshold, photo_ids)` becomes **matching-only**;
  Phase 2 and the hdbscan import leave. Return shape kept.
* `discover_persons_job(db_path, link, split, min_size, dry_run=True)` is new —
  Phase 1 over all unassigned first, then block-and-split over the residue.
* **Not RQ-scheduled.** `worker.py` has no `with_scheduler=True` and
  `start_workers.sh` starts no scheduler — that is exactly why 1,485 jobs rotted
  in `rq:scheduled:default`. **Do not use `enqueue_in`.** It runs from
  `scripts/discover_persons.py`, by hand or an owner-installed timer.
* The two paths meet through the persistent index: a discovered person is
  indexed before the job returns, so the next upload's Phase 1 matches into it.
* **A deleted person must not be re-proposed forever.** Deleting sets
  `person_id = NULL` (FK `ON DELETE SET NULL`), so discovery re-creates it. D7's
  provenance column is the fix: skip faces where `assigned_by='discovery' AND
  person_id IS NULL`.

### D4 — Validation: precision first, and say so

**More clusters is not evidence of improvement.** A wrongly merged person is a
wrong answer the owner must notice and unpick by hand, with no undo in the UI;
an unassigned face is merely invisible.

`scripts/evaluate_face_clustering.py --db <copy>` — never writes, reads no media:

**(a) Hidden-label re-cluster** — B-cubed P/R, cluster count, faces assigned,
and the gate: **merge events**. Plus a `link` sweep to locate the **merge cliff**.
**(b) Leave-one-person-out discovery** — the only test that measures *discovery*,
and the one the current code scores 0/18 on.
**(c) Phase-1 holdout** — 30% per person, fixed seed.

| measurement | gate |
|---|---|
| merge events, (a) and (b) | **0 — hard stop** |
| margin from `link` to the cliff | **>= 0.10** |
| LOPO persons recovered as one cluster | **>= 16 of 18** |
| LOPO faces recovered | **>= 90%** |
| Phase-1 precision / recall | **1.000 / >= 0.95** |
| B3 recall | reported, **not gated** — contaminated |
| cluster count | reported, **never** a success criterion |

Report `dada ji` (4) and `Person 4` (3) separately in (b) — at `min_size=3` they
sit on the boundary, which is information about `min_size`, not the algorithm.

**Visual verification is out of reach**: face thumbnails live on `/dev/sda` with
3,000+ read errors. The People page is the owner's check, afterwards.

### D5 — Nothing here can destroy a manual assignment

| path | today | after |
|---|---|---|
| `cluster_faces_job(reset=False)` | NULLs only | unchanged, pinned by a test |
| `discover_persons_job` | — | NULLs only, pinned by a test |
| `cluster_faces_job(reset=True)` | NULLs 1,410 | reachable only from the guarded script |
| `recluster_all.py` | a comment is the only guard | requires `--yes-destroy-manual-labels` |
| `cli faces-cluster` | **overwrites every face, reuses `Person N`** | unassigned-only, `min_size`, non-colliding names |
| `POST /api/index/faces-cluster` | forwards `reset` from the body | `reset` removed from the model |

**The production run uses `discover_persons_job`, never `reset`.** Its exact
inverse exists: `DELETE FROM persons WHERE id IN (<created>)`, which FK
`ON DELETE SET NULL` turns back into `person_id = NULL` on precisely the faces it
touched. The job returns those ids; the script writes a manifest.

### D6 — `core/face_cluster.py`, out of the API's import chain

```python
def cluster_faces(vectors, *, link=0.55, split=0.45, min_size=3, knn=30) -> np.ndarray
```

`-1` for noise. Deterministic; clusters numbered by first member index so a diff
of two runs is readable. **Must not import sklearn or scipy at module scope, and
`core/jobs.py` must import it lazily.** Use scipy (`linkage` + `fcluster`),
imported inside the split function, and add `sklearn` to the guard test's heavy
list so the familiar API cannot be swapped back in.

### D7 — `faces.assigned_by`

```sql
ALTER TABLE faces ADD COLUMN assigned_by TEXT;  -- manual | match | discovery | NULL
```

Three reasons: today's labels cannot be separated into "the owner said so" and
"Phase 1 said so", which is why recall is uninterpretable; it is how a rejected
proposal stays rejected; and it makes "how many proposals did the owner keep?" a
query. Both `core/db.py` and `core/db_async.py` need it — they have diverged once
already and must not do so again.

## Sequencing

Goal-backward: the deliverable is *evidence*, so the instrument is built first
and must reproduce the failure before anything changes. Phase 1 is a pure
function with no callers (zero blast radius). Phase 2 rewires the jobs. Phase 3
is independent and could land first but is placed after so risky edits do not
compete with a live investigation. Phase 5 is the only phase that writes to
production and is entirely owner-gated.

**Nothing needs a restart to be verified** — the job and script run in a fresh
process against a copy. Workers restart only to pick up new code for future
uploads.

## Tasks

TDD throughout: write the test, **run it, paste the failing output into
Evidence**, then implement. Every face test must set `CHITRA_FAISS_INDEX_DIR` to
a temp dir — six stray `.index` files are what happens otherwise;
`tests/test_face_match_index.py:FaceDBFixture` is the pattern.

### Phase 0 — Baseline and instrument
- **0.1** Record baseline: suite in a detached worktree; corpus and per-person
  counts read-only; `ntotal`/`d`/id-mapped for both indexes.
- **0.2** Working copy to `/tmp` via `.backup` from a read-only handle. **Never
  open the live file for writing.** Done when the copy has 3,289 faces and the
  live mtime is unchanged.
- **0.3** Red tests for the harness — B-cubed on a known fixture; a merge is
  detected when two seeded persons land together and **not** when one person
  splits; LOPO returns one row per person; the harness writes nothing; a
  fresh-subprocess check that importing it leaves `minio`, `core.storage_client`,
  `torch`, `insightface` out of `sys.modules`.
- **0.4** Implement `scripts/evaluate_face_clustering.py`. Flags: `--db --algo
  {current,blocksplit,hdbscan-absolute} --link --split --min-size --knn
  --holdout-frac --seed --sweep-link --json`. **No `--apply`; it cannot write.**
- **0.5** Reproduce the failure: `--algo current` must report 0 clusters, 0
  assigned, **LOPO 0 of 18**. **If it does not, stop** — the harness is measuring
  something else and everything downstream is worthless.

### Phase 1 — The clusterer
- **1.1** Red tests (synthetic vectors only): **the scaling regression** — a
  tight 5-face group found identically at n=100 and n=5,000; two tight groups
  bridged by one face at ~0.56 do **not** merge; `min_size-1` is noise and
  `min_size` is a cluster; noise stays `-1`; labels deterministic and stable
  under row permutation; empty and single-row inputs do not raise; **nothing
  derives any parameter from `n`** (grep-style assertion).
- **1.2** Implement. Blocks of 1 and 2 short-circuit without touching scipy.
  Raise on a block over `MAX_SPLIT_BLOCK` (20,000) rather than allocating 1.6 GB.
  scipy imported **inside** the split function.
- **1.3** Add `sklearn` to `HEAVY_MODULES` in `tests/test_api_memory_budget.py`.
- **1.4** Wire `blocksplit` and `hdbscan-absolute` into the harness; reproduce
  the bake-off table from this code. **A different number is a finding — record
  it and stop.**
- **1.5** `--sweep-link 0.30..0.60`; record the cliff and confirm >= 0.10 margin.

### Phase 2 — Split matching from discovery
- **2.1** Red: with `photo_ids` set, `cluster_faces_job` creates no persons even
  when the batch holds a tight group of 5 strangers.
- **2.2** Make it matching-only. Delete Phase 2, the hdbscan import, the epsilon
  arithmetic, `allow_single_cluster`, and the mean-pairwise gate. Keep `reset`
  and the return shape — `ClusteringResultTests` depends on it and must not be
  weakened. **Touches `core/jobs.py:1069-1250`; rebase before committing.**
- **2.3** Red for `discover_persons_job`: never writes to a non-NULL
  `person_id` (snapshot every assigned row); returns created ids; a second run
  creates nothing; `dry_run=True` writes nothing; new persons appear in the index
  **with their ids**; faces with `assigned_by='discovery' AND person_id IS NULL`
  are skipped.
- **2.4** Implement. Phase 1 through the existing `_person_face_index` /
  `_resolve_person` — **do not write a second matcher**. Name persons so they
  cannot collide (read `SELECT name FROM persons` first; `get_or_create_person`
  is wrong). Stamp `assigned_by`. Re-raise on failure.
- **2.5** `scripts/discover_persons.py` — dry-run default, `--apply`, prints
  proposed clusters with sizes and cohesion, writes `logs/discover-<ts>.json`
  with created person ids and their face ids.
- **2.6** End-to-end on the copy: ~99 persons, ~918 faces, the 1,410
  pre-existing assignments **byte-identical**, manifest replays as a rollback.

### Phase 3 — Close the destructive paths
- **3.1/3.2** `recluster_all.py` requires `--yes-destroy-manual-labels`; without
  it, print the count that would be destroyed and exit 2.
- **3.3/3.4** `cli faces-cluster`: unassigned faces only (add
  `db.get_unassigned_faces_embeddings`, mirroring the async one that already
  exists), cluster through `core.face_cluster`, apply `min_size`, non-colliding
  names.
- **3.5** Remove `reset` from `ClusterFacesRequest` and the enqueue at
  `app_fastapi.py:1754`. **Check `../chitra_ui_next` and `../chitra_ios` for
  senders first** and record what you find.
- **3.6** Grep guard: `UPDATE faces SET person_id = NULL` appears in exactly one
  place, so a fourth destructive path cannot appear quietly.

### Phase 4 — Provenance and hygiene
- **4.1** `faces.assigned_by` in both schema modules, additive, backfilled NULL.
  Stamp `match` / `discovery` / `manual` at the three routes. Assert both modules
  agree via `PRAGMA table_info`.
- **4.2** Delete `unmatched_faces_cluster.index` and the six test artefacts; add
  the `CHITRA_FAISS_INDEX_DIR` guard to whichever tests created them. **Do not
  touch the dead block** — out of scope, and it is the only reference.
- **4.3** Report the count of persons with zero faces. **Report, do not delete.**

### Phase 5 — Production run (owner-gated)
- **5.1** Owner: back up `photo.db` and the index. Workers idle.
- **5.2** Dry run against live, read-only. Gates from D4 must pass.
- **5.3** **Owner: `--apply`. Agents must not run this.**
- **5.4** Post-run verification (read-only): counts against prediction; persons =
  19 + created; index ntotal vs assigned (drift <= 64); **every pre-run
  assignment unchanged**; manifest ids all exist with >= min_size faces.
- **5.5** Rollback drill on the **copy** — before 5.3, not after.

### Phase 6 — Correct the record
- **6.1** Rewrite the clustering paragraph in `.claude/rules/ml-pipeline.md`.
  Three statements are wrong: the face count (2,081 -> 3,289); "a catch-up pass
  measurably makes none" (it is the answer once parameters are absolute); and
  the description of Phase 2 as HDBSCAN. Add the refreshed cohesion range, the
  gate rejecting the two largest persons, the contamination caveat, the
  sklearn/scipy import weights, and that `cli faces-cluster` was destructive.
- **6.2** `AGENTS.md`: add the three new files; note `recluster_all.py` now
  needs a flag.

## Owner actions

**Before.** Back up `photo.db` and `existing_person_faces.index` with the
existing `.bak-<tag>-<ts>` convention; confirm the `default` queue is idle.

**The run.** `.venv/bin/python scripts/discover_persons.py --db photo.db --apply`
— ~0.5 s, ~99 new `Person N` rows. Review in the People page over following days;
rename what is real, delete what is not. **Deleting is safe** — FK returns the
faces to unassigned, and `assigned_by` stops them being re-proposed.

**Worker restart** — only for future uploads to use the matching-only job.
Follow `.claude/skills/restart-workers`; prove the video queue is idle first.

**Optional weekly timer.** RQ's scheduler is not running and must not be relied
on. `OnCalendar=Sun 04:00` is the right shape if the manual run proves out.
**`WorkingDirectory` is load-bearing** — `faiss_indexes/` is a relative path.

## Risks

| risk | mitigation |
|---|---|
| The fix merges two real people | Zero merges is a hard gate on (a) and (b); the split stage removes the blocking threshold from the risk surface; >= 0.10 margin to the cliff recorded |
| 99 unnamed persons overwhelm the People page | `min_size=3` already declines 100 pair-clusters; sizes printed; deletion is a clean inverse |
| Recall flatters because labels came from Phase 1 | Stated; recall reported but **not gated**; `assigned_by` cleans the next evaluation |
| Module-scope sklearn blows the 200 MB API budget | scipy instead (33 vs 88 MB), lazily imported, `sklearn` added to the guard list |
| A block outgrows the O(m^2) split | `MAX_SPLIT_BLOCK` raises rather than allocating; largest today is 530 |
| Someone runs `recluster_all.py` or `cli faces-cluster` mid-plan | Phase 3 closes both; until then they are production hazards |
| Conflict with concurrent `core/jobs.py` work | Only `:1069-1250` and the new function; rebase immediately before committing |

## Rollback

One commit per phase. Phases 0, 1, 4 add code and a column and change no
behaviour. Phase 2 is the behavioural change; reverting restores HDBSCAN
discovery, which is inert, so the revert is safe but pointless. **Reverting
Phase 3 restores three destructive paths — do not.**

Data rollback is exact: `DELETE FROM persons WHERE id IN (<manifest>)`. Drill it
on the copy (5.5) before the live run.

## Out of scope

`FACE_MATCH_THRESHOLD` and Phase 1 (precision 1.000 / recall 0.983 on 1,410 —
leave it alone); the `app_fastapi.py:1524-1913` dead block; `core/cluster.py`'s
0.78 photo threshold; face detection and the `_is_video` gate; person merging in
the UI and any automatic merge of a discovered person into an existing one —
**proposing is safe, merging is not**; the pre-existing baseline failures.

## Follow-ups

1. **Re-run the harness after the next bulk detection pass** and record the cliff
   — the early warning for single-link blocking becoming unsafe. 30 seconds.
2. **Median within-cluster nearest-neighbour as a reported statistic.** Unlike
   mean pairwise it does not decay with size, so it may be the gate the deleted
   one should have been. Measure before gating.
3. **Drop the `hdbscan` pin** once `--algo hdbscan-absolute` has served as the
   falsification path. A compiled dependency carried for one comparison.
4. **`_auto_match_face_to_person` and the matching job now do the same thing
   twice.** Unify after Phase 2, when the second has shrunk to just Phase 1.
5. **A "not this person" signal in the UI.** `assigned_by` gives it schema room,
   and it is the only thing that would let the clusterer improve from the
   owner's corrections rather than merely avoid repeating them.
6. **The 711 faces with no neighbour at all** are the real ceiling. Some are
   strangers; some are the same person at a distance, in profile, or as a child.
   Nothing here helps them and no threshold will.

## Evidence

_(Appended by the executing agent as tasks complete.)_
