# SigLIP 2 text latency — measured on an idle box, and the 200 ms correction

Companion to `docs/plans/siglip2-footprint.md`, which measured SigLIP 2's
**memory**. This measures its **text latency**, because the footprint doc's
latency section is the one part of it that was estimated rather than measured,
and the estimate was wrong by 3x.

**Reads: zero MinIO objects, zero photos.** `photo.db` was opened once as
`file:photo.db?mode=ro&immutable=1` — SQLite takes no locks and writes nothing —
to count rows. `/dev/sda` was not touched. All model I/O is the HF cache on the
NVMe. Measured in a scratch venv (`transformers==4.57.6`); `.venv` was not
modified.

## Headline

| | on record (`siglip2-footprint.md`) | measured here | |
|---|---|---|---|
| SigLIP 2 text embed | ~200 ms | **66.7 ms** | 3.0x lower |
| CLIP text embed | 13.0 ms | **12.4 ms** | agrees |
| **ratio** | **15.4x** | **5.4x** | |

The 15x regression on the search path does not exist. The real figure is 5.4x
on the embed step and **3.1x end to end**, +56 ms on a request that today takes
~27 ms.

**The page-fault hypothesis is refuted.** Not "partly" — the token embedding
table contributes 0.06% of the forward pass, and forcing all 256,000 rows
resident changes latency by 0.0%. The 5.4x is genuine steady-state compute, and
it is fully explained by the two structural causes the plan already named.

## Method

`CHITRA_ML_THREADS=3` and the full `thread_limits.sh` env set throughout, one
isolated process per configuration, box idle (load average 0.15 at start; the
only load during the runs was the measurement itself). Every SigLIP number is
the public `SiglipEmbedder.text_embedding()` — the exact call
`embed_service.py:199` makes — not a hand-rolled forward pass.

Page faults from `/proc/self/stat` fields 10-13, cross-checked against
`resource.getrusage()` and confirmed identical. Note the off-by-one trap: after
stripping `comm`, `parts[0]` is field 3, so `minflt` is `parts[7]`, not
`parts[6]`. `parts[6]` is the `flags` word, which is a constant 4194304 and
looks plausible enough to publish.

Every configuration was run at least twice. Spread across independent runs at
different load: SigLIP 66.58 / 66.81 ms, CLIP 12.40 / 12.43 ms — ±0.3%.

## 1. Cold vs warm, identical phrase

Same phrase 21 times, first call separated.

| | first call | calls 2-21 median | cold delta | minflt on call 1 | **majflt** |
|---|---|---|---|---|---|
| SigLIP, warm page cache | 80.5 ms | **66.8 ms** | +13.6 ms (1.20x) | 9,034 | **0** |
| SigLIP, page cache dropped | **439.9 ms** | **66.9 ms** | +373.0 ms (6.58x) | 9,197 | **1,263** |
| CLIP, warm page cache | 16.1 ms | **12.4 ms** | +3.7 ms (1.29x) | 1,580 | **0** |

The "page cache dropped" row is `posix_fadvise(POSIX_FADV_DONTNEED)` over the
whole 1.5 GB safetensors blob before loading — a genuinely cold mmap, which is
harsher than anything production will ever see.

So the page-fault signature **is real and is measurable**: 1,263 major faults
and 373 ms of them. But it is a **one-off, once per process**, it is the
transformer weights faulting in rather than the vocabulary table, and by call 2
it is gone. It cannot be what a per-request cost of 200 ms was made of. A
long-lived sidecar pays it once at startup and never again.

## 2. Novel vs repeated vocabulary — the direct test

30 common-English queries, then 30 deliberately rare/multilingual ones
(Devanagari, Cyrillic, CJK, Hangul, Arabic, Greek, Thai, Hebrew, Bengali,
Kannada, Tamil, Malayalam, Georgian, Armenian, Burmese, Amharic), then both
repeated. The rare batch uses **255 token ids that the common batch never
touches** — genuinely untouched rows of the table.

| batch | SigLIP median | minflt | majflt | CLIP median |
|---|---|---|---|---|
| common, first pass | 67.54 ms | 79 | 0 | 11.88 ms |
| rare, first pass | 67.34 ms | 223 | 0 | 16.55 ms |
| common, repeated | 67.48 ms | 0 | 0 | 11.73 ms |
| rare, repeated | 67.20 ms | 0 | 0 | 16.59 ms |
| second, disjoint rare batch (novel) | 67.06 ms | 141 | 0 | 16.32 ms |

**SigLIP novel-vs-repeat gap: +0.14 ms. That is 0.2%, and it is noise.**
Rare is if anything *faster* than common on the first pass.

The CLIP column is what makes this conclusive rather than merely negative.
CLIP shows a real +4.67 ms rare-vs-common gap — so the experiment can detect a
gap when one exists. But CLIP's gap **persists exactly on repeat** (16.55 →
16.59 ms), so it is not page faults either. It is sequence length: CLIP pads to
the longest item in the batch, so multilingual text tokenizes into more tokens
and costs more compute. SigLIP pads every sequence to a fixed 64 and is
therefore structurally immune to the vocabulary of the query.

## 3. Attribution — where the 66.7 ms actually goes

Per call, phrase `"a photo of a beach at sunset"`, median of 30:

| | SigLIP (seq 64) | share | CLIP (seq 9) | share |
|---|---|---|---|---|
| tokenise | 0.19 ms | 0.28% | 0.24 ms | 1.98% |
| **transformer forward** | **66.72 ms** | **99.7%** | **11.90 ms** | **98.0%** |
| — of which token-table lookup | 0.04 ms | **0.06%** | 0.02 ms | 0.17% |

Tokenisation is not the problem, in either model. The `[256000, 768]` table —
the only place the 750 MiB is ever read — is **0.06% of the forward pass**.

Warm steady state takes **0 minor and 0 major faults per call**. Across all 120
queries of experiment 2 the whole process took 302 minor faults and zero major.

**The decisive experiment.** Force every one of the 256,000 rows resident
(`float(token_embedding.weight.sum())`), then re-run:

```
whole-table pre-touch:      0.03 s, RSS +686 MB
common queries before:      67.48 ms
common queries after:       67.25 ms      <- 0.3% apart, i.e. unchanged
```

Run the other way round — pre-touch immediately after load, before any query —
and the warm median is 67.22 ms against 66.81 ms without. If the table were the
cost, removing it entirely would have shown up. It does not, in either
direction.

The arithmetic agrees and always did: 255 novel rows x 3,072 B = **0.78 MB**,
at most 510 pages. Even charging every one of them a cold NVMe major fault,
that is a couple of milliseconds spread across 30 queries.

## 4. Does the warm cost match the structural prediction? Yes, exactly

The plan named two structural causes — a 768-wide text tower against CLIP's
512, and a fixed 64-token padding — and guessed they were worth ~3x. Measured
separately, by sweeping sequence length inside each architecture:

| seq | SigLIP | CLIP | ratio | analytic FLOP ratio |
|---|---|---|---|---|
| 1 | 14.46 ms | 7.84 ms | 1.84x | 2.24x |
| 4 | 22.14 ms | 10.88 ms | 2.03x | 2.25x |
| 8 | 23.44 ms | 11.95 ms | 1.96x | 2.25x |
| 9 | 23.44 ms | 11.79 ms | 1.99x | 2.25x |
| 16 | 25.88 ms | 13.42 ms | 1.93x | 2.25x |
| 24 | 29.14 ms | 15.13 ms | 1.93x | 2.24x |
| 32 | 33.51 ms | 17.12 ms | 1.96x | 2.24x |
| 48 | 41.26 ms | 20.84 ms | 1.98x | 2.24x |
| 64 | 66.56 ms | 29.21 ms | 2.28x | 2.23x |

* **Width, isolated:** at matched sequence length SigLIP is a flat **~1.95x**
  CLIP across the whole range, against an analytic 2.25x.
* **Padding, isolated:** within SigLIP alone, seq 64 costs **2.84x** seq 8.
* **Together: 1.95 x 2.84 = 5.5x**, against 5.4x measured end to end.

Both structural causes are real and together they account for all of it. The
plan's "~3x" undercounted; the "15x" never existed.

### Why the naive FLOP ratio (16x) overpredicts

SigLIP's production path does 16.2x CLIP's FLOPs (11.02 vs 0.68 GFLOP) but
takes only 5.4x the time, because short sequences are overhead-bound and long
ones are not:

| | effective GFLOP/s |
|---|---|
| CLIP at seq 9 (its production path) | 57.8 |
| SigLIP at seq 64 (its production path) | 165.6 |

SigLIP does 16x the arithmetic at 2.9x the efficiency. **This is the trap in
reasoning about the gap from FLOPs alone, in either direction** — and it is why
the honest answer had to be measured.

(Both curves peak near seq 48 — 199.7 GFLOP/s for SigLIP, 176.6 for CLIP — and
lose ~17% at 64. Noted, not chased; nothing here depends on it.)

## 5. Resident memory — the mmap drift claim is confirmed

Base run, `VmRSS` from `/proc/self/status`:

| point | VmRSS | RssFile | note |
|---|---|---|---|
| before importing torch | 12 MB | 7 MB | |
| torch imported | 485 MB | 227 MB | the fixed import weight |
| `SiglipModel` "loaded" | **929 MB** | 431 MB | the 750 MiB table is *not* resident |
| after 21 text embeds | 1,255 MB | 756 MB | +326 MB — transformer weights fault in |
| after 141 more queries | 1,317 MB | 818 MB | +62 MB of vocabulary drift |
| whole table forced resident | **2,003 MB** | 1,505 MB | the ceiling |

The drift is real and is **larger than the row arithmetic implies**: 30 novel
rare queries added **31 MB** of `RssFile` while needing only 0.78 MB of rows.
That factor of 40 is mmap readahead — the kernel pulls a ~128 KB window around
each faulted page. This is the mechanism behind the footprint doc's
2,137 MB → 3,240 MB drift, and **that guidance stands: size for the ceiling.**

What changes is only the *cost* of the drift. It is RSS, not latency. Those
pages are clean and file-backed on the NVMe; touching all of them takes 0.03 s.

## 6. Why the 200 ms figure was wrong

It was never measured — it was extrapolated. `siglip2-footprint.md` measured
SigLIP text at 298 ms under load average 13, measured CLIP text at 19.3 ms in
the same window against 13.0 ms on record, and scaled by that 1.48x contention
factor to get ~200 ms.

The scaling assumes contention penalises both towers equally. It does not. On a
6-core box at 3 threads, SigLIP's 64-token pass saturates its threads far harder
than CLIP's 9-token pass, so it degrades much further under load: its true
contention factor in that window was ~4.5x, not 1.48x.

**Cross-contamination check — the same error inflated the image estimate:**

| | footprint doc (extrapolated) | measured idle | |
|---|---|---|---|
| SigLIP image embed | ~346 ms | **170-174 ms** | 2.0x lower |
| CLIP image embed | 113 ms | **62.1 ms** | (see below) |
| **ratio** | 3.06x | **2.79x** | matches the 2.84x encoder prediction |

CLIP's 62.1 ms here versus 113 ms on record is not a contradiction: this
measurement embeds a 256x256 synthetic PNG, while the on-record figure embeds a
real photo and includes ~50 ms of JPEG decode and downscale. That preprocessing
is identical for both models, so **the ratio is the transferable number** and
real-photo figures are roughly 112 ms (CLIP) and 224 ms (SigLIP).

Consequences for the plan:

* A full re-embed of 2,276 photos is **~8.5 min** of compute, not ~13 min.
* **D3's accepted worst case needs updating.** It budgeted "~113 ms of executor
  queueing" behind a co-resident image embed. Under SigLIP that becomes ~224 ms,
  so worst-case search is ~82 + 224 ≈ **306 ms**, not the ~200 ms implied. Still
  acceptable, but it should be written down rather than discovered.

## 7. End-to-end search budget

`/api/search/photos`, measured stage by stage against the live row count
(2,276 embeddings, all `openai/clip-vit-base-patch32`, 4.66 MB):

| stage | CLIP today | SigLIP |
|---|---|---|
| text embed in the sidecar | 12.4 ms | 66.7 ms |
| loopback hop (on record) | ~2 ms | ~2 ms |
| `get_embeddings_async` fetch | 5.4 ms | 5.4 ms |
| `np.stack` + normalise + GEMV | 6.9 ms (4.66 MB) | 8.4 ms (6.99 MB) |
| **total** | **~26.7 ms** | **~82.5 ms** |

**+56 ms, 3.1x end to end.** Note the ranking half grows with the library while
the embed half is constant, so the multiple shrinks over time.

## 8. The obvious optimisation, tested and rejected

Shortening `max_length` from 64 is the visible 2.6x lever — seq 16 is 25.9 ms
against 66.6 ms. **It destroys the vector.** Cosine of the shortened embedding
against the pad-64 embedding *of the same query*:

| padding | cos to pad-64, mean | min |
|---|---|---|
| 8 | 0.645 | 0.582 |
| 16 | 0.657 | 0.597 |
| 24 | 0.755 | 0.673 |
| 32 | 0.729 | 0.634 |
| 48 | 0.852 | 0.782 |

For scale, **two entirely different queries at pad-64 sit at cosine 0.783 on
average** (min 0.691). So a pad-16 "beach at sunset" is *further* from a pad-64
"beach at sunset" than a pad-64 "beach at sunset" is from a pad-64 "birthday
cake". The image vectors are unaffected by text padding, so this would compare
off-distribution text against in-distribution images.

`SiglipEmbedder.TEXT_PADDING = "max_length"` is correct and load-bearing. Do not
"optimise" it.

## 9. Does a query-vector cache change the answer?

**No, and it should not be built now.**

A hit would cost ~16 ms (hop + DB + rank) against ~82 ms for a miss — a ~66 ms
saving, which puts a cache hit *below* today's CLIP latency. That is a real win
per hit. Three things make it not worth building yet:

1. **The hit rate cannot be measured, and nobody should quote one.** Nothing
   logs search queries — `logs/` holds only worker and sidecar output, and
   uvicorn's access log goes to journald, which is not readable without sudo.
   Any hit-rate number for this library today is a guess.
2. **The web client already caches the repeats.** `Search.tsx` uses TanStack
   Query keyed `["search", kind, q]` with `staleTime: 30_000` and
   `gcTime: 5 * 60_000`. The highest-frequency repeats — re-mount, back/forward,
   an immediate re-search — never reach the server at all. A server-side cache
   would therefore see traffic that is *already* skewed toward first-time
   queries, which are its misses. iOS (`TimelineModel.search`) keeps no result
   cache and would benefit; that is one of two clients.
3. **Neither client searches as you type.** Web is `onSubmit`, iOS is
   `.onSubmit` — and `TimelineModel.swift:245` says so explicitly. This is the
   one thing that would have made 66 ms genuinely painful: at one request per
   keystroke, 66 ms per character would be a real product problem. It is not the
   case, so the cost is paid once per deliberate search.

If it is ever built, build the simple version: an **in-process LRU inside the
sidecar**, keyed `sha256(model + normalized query)`. A 768-d fp32 vector is
3 KB, so 10,000 entries is 30 MB against a sidecar already sized for 3,240 MB.
Redis is not wrong, but it adds a hop, a serialisation format and a dependency
to save the same 66 ms, and this is a single-box deployment. Redis only earns
its place if the sidecar restarts often enough that cache warmth matters across
restarts — which, with `Restart=always` and a long-lived process, it does not.

## Verdict

**The text-latency finding does not block SigLIP 2 adoption.**

* Steady-state text embed is **66.7 ms, not 200 ms**; the multiple over CLIP is
  **5.4x, not 15.4x**.
* It is **genuine compute**, not page faults: tokenisation 0.28%, forward pass
  99.7%, token-table lookup 0.06%, and forcing the entire 750 MiB table resident
  changes latency by 0.0%.
* It is **fully explained structurally** — 1.95x from the 768-wide tower, 2.84x
  from the fixed 64-token padding, 1.95 x 2.84 = 5.5x against 5.4x measured. No
  unexplained residual, so there is nothing left to go hunting for.
* End-to-end search goes **~27 ms → ~82 ms**. For a personal photo library that
  is the difference between instant and instant.
* A query cache is a genuine ~66 ms saving per hit but is **not needed to make
  this decision**, and the web client already absorbs the repeats it would
  serve. Ship SigLIP first; add the cache later only if search latency is
  actually complained about.

What *should* change in the plan is smaller and unrelated to the go/no-go: D3's
worst-case queueing budget (~113 ms → ~224 ms) and the re-embed estimate
(~13 min → ~8.5 min), both of which move because the image figure was
extrapolated the same way the text figure was.

The memory half of `siglip2-footprint.md` is confirmed by this run and needs no
change — including the 3,240 MB sizing guidance, whose mechanism (mmap readahead
amplifying row-level touches ~40x) is measured directly above.
