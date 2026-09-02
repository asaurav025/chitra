# Deploying the 2026-09-01 work

57 commits landed on 2026-09-01. The services were last restarted at **22:44**, and
**15 commits landed after that** — so roughly the last hour of work is committed
but not running.

This plan sequences the rollout by risk, cheapest and most reversible first. Each
stage is independently valuable and independently revertable; stop at any point.

## Current delta between repo and runtime

| | committed | live |
|---|---|---|
| code (15 commits) | yes | **no** — needs restart |
| tag vocabulary | 345 labels | **17** (`clip-vitb32/vocab-v1`) |
| video embeddings | code landed | **0 of 262 videos searchable** |
| `transformers` | `>=4.50,<5` | **4.35.0** — venv does not satisfy requirements.txt |
| photos with no thumbnail | 66 | 66 |

---

## Stage 1 — Restart (low risk, reverts by `git revert` + restart)

Picks up all 15 commits: sidecar job routing, the InsightFace thread cap, video
poster embedding, model-aware search, the four discovery endpoints, the
magic-number image dispatch, typed 422/404 thumbnail errors, and the `use_fast`
pin.

```bash
~/services/chitra/scripts/safe_restart.sh
```

The script refuses to proceed if a transcode is in flight. **Search returns
`503 search_unavailable` for ~20 s** while the sidecar reloads CLIP — deliberate,
there is no in-process fallback by design.

**Verify:**
```bash
curl -s localhost:5000/api/health          # expect embed_status: ok
curl -s localhost:5101/health              # sidecar, model + threads
pgrep -fc 'worker.py'                      # expect 6
cat /sys/fs/cgroup/system.slice/chitra-api.service/memory.current
```

**Watch for, in the first hour:** the 66 photos with no thumbnail should now
return 422/404 with a reason instead of a bare 500; a newly uploaded photo should
embed in seconds rather than ~58 s. Check `logs/worker_1.log` for the first job
after restart.

**Rollback:** `git revert` the offending commit, restart. No data changes yet.

---

## Stage 2 — Re-tag the library (reverts with one SQL statement)

The largest user-visible change. **Reads zero MinIO objects** — it works from
embedding vectors already in SQLite.

```bash
cd ~/services/chitra
cp photo.db photo.db.bak-$(date +%F)
.venv/bin/python scripts/retag.py --db photo.db --apply
```

Expected from the dry run on a copy: `travel` **80.9% -> 1.8%**, tags per photo
from *always exactly 6* to a 0-8 spread (median 3).

**Verify:** spot-check a handful of photos in the UI. The dry run's samples were
`food, indoors, night, portrait, selfie, travel` -> `studying, reading, library,
working on a laptop, book`, and a second -> `bird, monkey, wildlife`.

**Rollback:** `DELETE FROM tags WHERE source = 'clip-vitb32/vocab-v2';` then
re-run the v1 tagger, or restore the backup.

**Note:** this only re-tags existing photos. New uploads still get the old
17-label tagging at job time — wiring corpus calibration into upload-time tagging
is a deliberate follow-up.

---

## Stage 3 — Embed video posters (additive, reverts by deleting rows)

**0 of 262 videos are currently searchable.** 258 already have a poster; the code
to embed them landed in Stage 1 but no bulk run has happened.

Run it after Stage 1 is confirmed healthy. Roughly 258 objects at ~250 KB.

**Rollback:** `DELETE FROM embeddings WHERE photo_id IN (SELECT id FROM photos WHERE media_type='video');`

---

## Stage 4 — Upgrade the venv (do this on its own, not folded into Stage 1)

`requirements.txt` now declares `transformers>=4.50,<5`; the venv is on **4.35.0**.
A restart alone will not change this — `start_workers.sh` installs nothing.

This is a 15-minor-version jump under a live sidecar, and it is the one change
that could silently alter every future embedding. That is exactly what
`tests/test_embedder_stability.py` exists to catch.

```bash
cd ~/services/chitra
.venv/bin/pip install -r requirements.txt
CHITRA_TEST_LOAD_MODELS=1 .venv/bin/python tests/run_tests.py test_embedder_stability
~/services/chitra/scripts/safe_restart.sh
```

**The stability test must pass before restarting.** It compares a fixed synthetic
image against a reference vector at `1e-4` tolerance. Measured: the correct
(slow) image processor gives 1.27e-07; the fast one gives 1.08e-03, ten times over
tolerance. `core/embedder.py` pins `use_fast=False`, and the test asserts the
kwarg is *passed* rather than that today's default happens to be right.

**If it fails, stop and do not restart** — the venv can be rolled back with
`pip install transformers==4.35.0 huggingface_hub==0.17.3 tokenizers==0.14.1`.

---

## Stage 5 — SigLIP 2 migration (a decision, not a task)

Measured verdict: **it fits.** SigLIP + buffalo_l is 2,137 MB against 2,018 MB for
CLIP + buffalo_l today — **+119 MB (+5.9%)**. Text latency is **66.7 ms, a 5.4x
regression, not the 15x first reported** (that figure was extrapolated under load;
see `docs/plans/siglip2-text-latency.md`). End-to-end search goes ~27 ms -> ~82 ms.

**Pre-flight, required before any re-embed:**

- **`cli/main.py:198` and `:468` call the sync `db.get_embeddings(conn)` with no
  model filter.** Harmless today because every row is CLIP, but the moment mixed
  dimensions exist they hit the same `np.stack` ValueError that was fixed on the
  API side. Fix before starting.
- Size for **3,240 MB**, not the 822 MB it shows on day one — the 786 MB token
  table is mmap-backed and drifts resident as query vocabulary widens.
- Stage 4 must be complete; `SiglipModel` does not exist in transformers 4.35.

Then Phase 6 of `docs/plans/classification-optimisation.md`: pilot 50 photos,
confirm 512-d and 768-d rows coexist, full pass (~8.5 min of compute), verify
coverage matches before flipping `CHITRA_ACTIVE_EMBED_MODEL`, soak a week, then
retire the 512-d rows.

**Rollback at any point is a config flip back** — which is the whole reason the
`(photo_id, model)` key exists.

---

## Not in scope here

- **Client-side thumbnail fixes** in `chitra_ui_next` — the Faces route renders
  ~2,000 `<AuthImage>` at once with no virtualization, and `staleTime: Infinity`
  turns one transient failure into a permanently broken tile. Untouched.
- **Re-timing the jobs on an idle box.** The "58 s -> ~1 s" claim is structurally
  sound but was never measured cleanly; last night's box was at load average 22.
- `test_endpoints.test_health_check` — order-dependent, documented in AGENTS.md,
  ours to fix.

## Deferred by the owner

- Disk-level checks on `/dev/sda`. Errors stopped after the 21:16 reboot; the
  standing instruction is to observe passively and report only if they resurface.
