# SigLIP 2 footprint on this box — measured, Phase 5 Task 5.3

`google/siglip2-base-patch16-224`, 1,539.5 MB on disk in `~/.cache/huggingface`
(NVMe). Measured under `transformers==4.57.6` in a scratch venv.

**Verdict: SigLIP 2 can land, in full fp32 with the text tower.** Free-text
search survives. The vision-only compromise is unnecessary and should not be
built.

**Reads: zero MinIO objects, zero photos.** The probe image is generated
arithmetically, the same deterministic pattern as
`tests/test_embedder_stability.py`. `/dev/sda` was not touched.

## Method

`VmHWM` and `VmRSS` from `/proc/self/status`, read **inside** each child, one
isolated `subprocess` per configuration. Not `ru_maxrss`: a `subprocess.run`
child inherits the parent's peak and over-reports — the trap already documented
at the top of `tests/test_api_memory_budget.py`.

* `peak` = `VmHWM`, the process's own high-water mark, reset at exec.
* `steady` = `VmRSS` after load **plus a fixed 5 forward passes** plus
  `gc.collect()`, no `malloc_trim`. The fixed pass count is what makes rows
  comparable; an earlier run varied it and made `steady` meaningless.
* `torch.set_num_threads(3)` throughout.

Every process starts at ~492 MB of bare torch import weight; that is included
in all figures below, because it is real and unavoidable in any process holding
these models.

## The table

| configuration | peak | steady |
|---|---|---|
| `SiglipVisionModel` only, fp32 | **1,460 MB** | 1,460 MB |
| `SiglipVisionModel` only, fp16 | **1,338 MB** | 1,325 MB |
| full `SiglipModel`, fp32 | **1,790 MB** | 1,790 MB |
| full `SiglipModel`, text tower fp16 | **2,851 MB** | 2,049 MB |
| SigLIP + CLIP + buffalo_l co-resident | **2,493 MB** | 2,404 MB |

Rows measured alongside, for calibration and for the decision:

| configuration | peak | steady |
|---|---|---|
| CLIP ViT-B/32 alone | 1,257 MB | 1,257 MB |
| buffalo_l alone | 1,434 MB | 1,351 MB |
| **CLIP + buffalo_l — what production runs today** | **2,018 MB** | 1,952 MB |
| **SigLIP full fp32 + buffalo_l — the post-migration state** | **2,137 MB** | 1,985 MB |
| **SigLIP + buffalo_l, token embedding fully resident — worst case** | **3,240 MB** | 3,071 MB |
| full `SiglipModel`, text fp16 via post-hoc `.half()` | 3,125 MB | 3,120 MB |
| full `SiglipModel`, loaded fp16 throughout | 2,848 MB | 1,817 MB |

Calibration against the numbers already on record (`docs/plans/api-oom-fix.md`):
buffalo_l 1,434 vs 1,462 MB and the CLIP+buffalo_l pair 2,018 vs 2,054 MB both
agree within 2%. CLIP alone reads 1,257 against 1,670 MB — 413 MB lower. The two
agreeing rows say the method is sound; the CLIP-alone gap is unexplained and is
recorded rather than chased, since no conclusion here rests on it.

## The decision this settles

`/api/search/photos` takes arbitrary user text, so a vision-only deployment
would kill the product's primary feature. The question was whether the text
tower — which holds a `[256000, 768]` multilingual token embedding, 786.4 MB in
fp32, 52.4% of the checkpoint — fits alongside buffalo_l.

It does, with room:

```
CLIP + buffalo_l          (today)   2,018 MB peak
SigLIP fp32 + buffalo_l   (after)   2,137 MB peak     +119 MB   (+5.9%)
```

**Switching the whole vision-language model costs 119 MB of peak RSS.** Against
a cgroup `MemoryMax=8G` whose `memory.peak` is already 3,311 MB, that is noise.

## Why it is so cheap — and the number you must actually plan for

`from_pretrained` maps the safetensors file rather than copying it, so the
786 MB token embedding is **file-backed and only the rows you touch become
resident**:

```
torch imported                                    694 MB
SiglipModel fp32 "loaded"                         822 MB   <- 786 MB table not resident
after one image forward                         1,467 MB
after one text forward (a few token rows)       1,789 MB
after touching EVERY token-embedding row        2,537 MB   <- +748 MB
+ buffalo_l                                     3,240 MB
```

A day-one measurement therefore reads ~2,137 MB, and that number quietly grows
toward ~3,240 MB as real queries touch more of the vocabulary. **Size for
3,240 MB, not 2,137 MB.** Anyone who measures this on the first day and sizes
the cgroup to it will be surprised weeks later, with no code change to blame.

Still fits: 3,240 MB peak against an 8G cap.

The pages are clean and file-backed, so under pressure the kernel evicts them
and re-reads from `~/.cache/huggingface` — **on the NVMe, not the failing
`/dev/sda`.** Eviction costs latency, never correctness or a bad read.

## fp16 is strictly worse here. Do not use it

The plan offered "text tower fp16" as the way to fit the text side. Measured, it
fails on both axes:

| | peak | image forward |
|---|---|---|
| full fp32 | **1,790 MB** | **1,395 ms** |
| loaded fp16, vision promoted to fp32 | 2,851 MB | — |
| post-hoc `.half()` on the text tower | 3,125 MB | — |
| vision-only fp16 | 1,338 MB | 7,809 ms (5.6x slower) |

Two independent reasons:

1. **Peak is higher, not lower.** Converting dtype materialises both
   representations before the old one is freed, and glibc does not return the
   arena, so `VmHWM` keeps the sum. Post-hoc `.half()` is the worst of the lot
   at 3,125 MB — *1.7x the fp32 model it was supposed to shrink*.
2. **fp16 forward is 5.6x slower.** This CPU has no fp16 compute path, so torch
   converts to fp32 per operation and pays for it. (Both timings taken under the
   same load, so the ratio holds even though the absolute values do not — see
   the caveat below.)

fp32 is simultaneously smaller at peak and faster. There is no configuration in
which fp16 is the right answer on this box.

## Latency — NOT USABLE, must be re-measured on an idle box

Every timing here was taken at **load average 21-25 on a 6-core box**, with two
ffmpeg transcodes and three other agents' test suites running. CLIP's image
embed measured **364 ms against the 113 ms on record** at the same 3 threads, so
absolute figures are roughly 3x inflated.

| model | image embed | text embed |
|---|---|---|
| CLIP ViT-B/32 (113 ms on an idle box) | 364 ms | 39 ms |
| SigLIP 2 vision fp32 | 1,395 ms | — |
| SigLIP 2 full fp32 | 1,779 ms | 501 ms |
| SigLIP 2 vision fp16 | 7,809 ms | — |

Taking the CLIP ratio as the contention factor puts SigLIP's real image embed
near ~430-550 ms, against the ~321 ms the plan predicted from the 2.84x encoder
ratio — same order, and compute was never the constraint.

**The thread curve is deliberately not reported.** Contention does not merely
scale a thread curve, it reshapes it: an already-oversubscribed box makes low
thread counts look artificially good, so a 3-vs-4-vs-6 comparison taken now
would be actively misleading about the real optimum. The open question stands —
**the 3-thread optimum is a CLIP result and patch16-224 has 4x the tokens** —
and it needs a quiet box. It gates only tuning, not whether SigLIP lands.

## Recommended configuration

**Full `SiglipModel`, fp32, text tower included.** Not vision-only (kills
free-text search), not fp16 (bigger and slower).

Budget **3,240 MB peak** for the sidecar once it holds SigLIP and buffalo_l, and
**2,493 MB** for the migration window where CLIP is still resident to answer
search from the 512-d rows.

## Follow-up, not acted on

The `[256000, 768]` token embedding is multilingual and is 52.4% of the
checkpoint. Chitra's search is English-only. Pruning the vocabulary to the
tokens actually reachable would remove most of the 786 MB — and it is the entire
difference between the 2,137 MB day-one figure and the 3,240 MB worst case. Out
of scope here; worth it only if the sidecar ever becomes memory-constrained.
