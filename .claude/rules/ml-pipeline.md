---
paths:
  - "core/jobs.py"
  - "core/embedder.py"
  - "core/face.py"
  - "core/tagger.py"
  - "core/faiss_index.py"
  - "core/cluster.py"
  - "recluster_all.py"
---

# ML pipeline rules

**Models load per job, not per worker.** RQ forks a work-horse for every job, so
any module-level model cache is populated in the child and thrown away when it
exits. Assume any per-job model load is the dominant cost: that was ~90% of the
measured 58 s embedding job and 37 s face job.

**Embedding no longer pays it — face detection still does.** The embedding jobs
go through the resident CLIP in `embed_service.py` over loopback
(`core/embed_client_sync.py`), so they load nothing and hold no torch at all;
`tests/test_embedding_job_sidecar.py` runs one in a fresh subprocess and asserts
torch never enters `sys.modules`. **Do not add an in-process fallback for when
the sidecar is down.** It would restore the 58 s path and 1.67 GB of residency
invisibly — the pipeline would simply become slow again with nothing in the logs
to say why. Jobs re-raise instead, so RQ marks them failed. `_get_embedder` is
kept unreferenced for one release purely so the change is a clean revert.
`core/face.py` still loads the five buffalo_l ONNX files (326 MB) per job; only
residency removes that, and `intra_op_num_threads=3` addresses ~4 s of it.

**One image embed per photo, not two.** `auto_tags` calls `rank_labels`, which
calls `image_embedding` again. Anything holding a vector already should score
labels with `SyncEmbeddingClient.rank_labels_for_vector` — both vectors are
normalised, so the score is a dot product and needs no forward pass.

**CLIP's image preprocessing is pinned, and the pin is load-bearing.**
`core/embedder.py` passes `use_fast=False` to `CLIPProcessor.from_pretrained`
rather than inheriting the default. transformers is moving that default to the
torchvision-backed fast processor; it still returns the slow one for
`openai/clip-vit-base-patch32` only because a slow config ships with the
checkpoint, and it warns on every load that this will stop. Measured on
4.57.6 against the synthetic reference: slow 1.27e-07 from the stored vectors,
**fast 1.08e-03** — 10x the tolerance in
`tests/test_embedder_stability.py` and ~8,500x the thread-count noise floor.
Every one of the ~1,900 stored vectors came from the slow path, so inheriting a
default that flips would compute new ones in a different space with nothing
raised anywhere; search would just get gradually worse. **Do not drop the
kwarg, and do not widen that test's tolerance to get past a failure** — a
failure there means the corpus and new embeddings have diverged, which turns a
version bump into a forced full re-embed on a failing disk. Switching to the
fast processor is a re-embed decision, not a performance tweak.

**`transformers>=4.50,<5` — and it was missing from `requirements.txt`
entirely** while `core/embedder.py` imported it, so a clean install produced a
venv that could not embed at all. The blockers on raising it were
`huggingface_hub<0.18` and `tokenizers==0.14.1`, not the model class. Validated
4.35.0 -> 4.57.6 with no vector movement. **The deployed `.venv` is still on
4.35.0**, which cannot construct `SiglipEmbedder` at all (`SiglipModel` does
not exist there) — that import is inside `__init__` for exactly this reason.
Upgrading `.venv` requires a worker and sidecar restart to be consistent.

**Two model switches, deliberately separate.** `CHITRA_EMBED_MODEL` selects
what the sidecar loads and computes with (`core.embedder.build_embedder`, via
`embed_service.configured_model`); `CHITRA_ACTIVE_EMBED_MODEL` selects what
`search_photos` ranks from (`core.db_async.active_embed_model`). The re-embed
cannot be atomic, so the migration needs a window where the sidecar already
writes 768-d SigLIP rows while search still answers from the complete set of
512-d CLIP rows: flip the first, re-embed, verify coverage, then flip the
second. **The sidecar must never read `CHITRA_ACTIVE_EMBED_MODEL`** — if one
variable drove both, the cutover would flip the read side before a single photo
was converted and search would silently lose everything not yet re-embedded.
`build_embedder` raises on an unknown identifier rather than falling back to
CLIP, which would write 512-d rows under a `model` name claiming otherwise.

**SigLIP 2 fits, in full fp32 with the text tower** — measured, see
`docs/plans/siglip2-footprint.md`. With buffalo_l co-resident it peaks at
2,137 MB against 2,018 MB for the CLIP it replaces (+119 MB). **Do not build a
vision-only deployment**: `/api/search/photos` embeds arbitrary user text, and
dropping the text tower deletes the primary feature to save nothing. **Do not
use fp16**: it loses on both axes here — peak RSS goes *up*, because a dtype
conversion materialises both representations and glibc keeps the arena
(post-hoc `.half()` on the text tower hits 3,125 MB, 1.7x the fp32 model it was
meant to shrink), and the forward pass is 5.6x slower since this CPU has no
fp16 compute path. One non-obvious operational number: the `[256000, 768]`
multilingual token embedding is 786 MB and is *mmap-backed*, so only the token
rows real queries touch go resident. Day one reads ~2,137 MB and drifts toward
**3,240 MB** as the query vocabulary widens, with no code change to blame.
**Size for 3,240 MB.** Those pages are clean and file-backed, so eviction
re-reads from `~/.cache/huggingface` on the NVMe, never from `/dev/sda`.

**SigLIP text is padded to the full 64-token context**, not to the longest item
in a batch, or a label's score would depend on which other labels were ranked
beside it. `SiglipProcessor` already supplies the 64, but the tokenizer logs
"Asking to pad to max_length but no maximum length is provided ... Default to
no padding" — that message is **wrong** and `SiglipEmbedder` passes
`max_length` explicitly so nobody "fixes" correct code on the strength of it.

**Everything is CPU.** `torch.cuda.is_available()` is False on this box and
`onnxruntime` is the CPU build. The pinned `nvidia-*` wheels are dead weight.
Never add a model or code path that assumes a GPU.

**Face matching thresholds are inconsistent and that is a known bug**: 0.75 in
`_auto_match_face_to_person`, 0.6 from the upload path and `recluster_all.py`,
0.78 in the photo-level CLI clusterer. If you touch matching, unify rather than
add a fourth value.

**`_auto_match_face_to_person` rebuilds a FAISS index over every assigned face,
once per detected face.** The persistent HNSW index at
`faiss_indexes/existing_person_faces.index` exists precisely to avoid this and
is bypassed on that path. Don't copy the pattern.

**FAISS index writes are non-atomic and unlocked** across four concurrent
workers — `save_index` writes straight onto the final path. Any new index write
should go to a temp file in the same directory followed by `os.replace()`.

**`faiss_indexes/` is a relative path** resolved against the process CWD. It
works because `start_workers.sh` cds first; a systemd unit without a matching
`WorkingDirectory` silently creates an empty second index directory and face
matching stops working with no error.

**Videos get no ML from the original; posters are embedded.** The embedding
jobs resolve their object key through `_embed_source_key`, which returns
`photos.thumb_path` for a video and `file_path` for everything else — so a
video is embedded from its 512x512, ~250 KB poster and its multi-gigabyte
original is never opened. 258 of the 262 videos already have a poster (~64 MB
for the set); the 4 without one read nothing at all rather than falling back to
the original, and `generate_video_poster_job` is the fix for those. CLIP
consumes 224x224 of one frame regardless, so the poster is not a compromise —
reading the original would cost four orders of magnitude more for a worse
vector.

**Face detection still skips videos entirely.** `_is_video()` remains the gate
there, and it has no poster path. Do not generalise the embedding change into
it without measuring; `tests/test_video_embedding.py` pins the distinction.

**`embeddings` is unique on `(photo_id, model)` — not on `photo_id`.** All four
writers (`db.put_embedding`, `db_async.put_embedding_async`, `db.add_tag`,
`db_async.add_tag_async`) were plain INSERTs into tables with no constraint, so
any bulk re-index duplicated every row; `search_photos` stacks whatever
`get_embeddings` returns, so a duplicate gave one photo two result slots. They
are upserts now. The key deliberately includes `model` so a SigLIP row lands
*alongside* the CLIP row search is still answering from — a bare `photo_id` key
would evict it and take rollback-by-config with it. `tags` is unique on
`(photo_id, tag)`; `tags.source` is provenance and stays out of the key. The
migration lives once, in `core/db.py`, and `db_async.py` imports it.

**`load_image` dispatches on the file extension, and the extension lies.** 63
photos are named `.arw`/`.ARW` whose bytes begin `FF D8 FF E0 JFIF` — they are
JPEGs, so rawpy raises `LibRawFileUnsupportedError` and both thumbnail
generation and embedding fail. This looks like a storage fault and is not: the
reads succeed. `scripts/reembed.py` works around it by sniffing the magic
number and renaming before it calls the sidecar; `core/extractor.py` itself
still trusts `path.suffix`.

**Embed from `photos.thumb_path`, not the original, for anything bulk.**
Measured: 1,713 originals are 7,959 MB against 396 MB of thumbnails (mean
237 KB) — 20.1x less data off a disk with 3,000+ unrecovered read errors, and
no RAW/HEIC decode. Cosine against the original-derived vector over 60 photos:
median 0.99914, mean 0.99676, min 0.97494; HEIC (the bulk of the library)
averages 0.99930. The one weak class is PNG screenshots at 0.98288 mean.
`scripts/reembed.py --source thumb|original|auto` is the switch.

**Tagging is corpus-relative, and that changes what a tag asserts.** CLIP's
contrastive objective calibrates scores *within* an image, not across images:
measured over all 11,456 stored tag rows, every raw cosine — good match or bad
— lands between 0.158 and 0.278, median 0.204, with a ~0.03 gap between a
photo's best and worst kept label. There is no absolute threshold to pick, and
one tuned for 17 labels does not survive 345. So `core.tagger.calibrate` learns
two percentiles per label from the whole corpus and `tags_from_scores` keeps a
label only where the photo sits in that label's own upper tail. Consequences to
know before touching any of it:

* **Per-label coverage is structurally bounded by `100 - low_percentile`** (~10%
  with the defaults). That is the mechanism that stops another `travel` at
  81.4%, and it is also why a library that genuinely is 30% portraits will have
  portraits it does not tag. A tag means "unusually beach-like *for this
  library*", not "depicts a beach".
* **It needs the whole corpus at once**, which a per-photo RQ job does not have.
  `auto_tags` is therefore still the uncalibrated legacy-17 top-k path;
  `scripts/retag.py` is what gives a photo its real tags. Do not try to
  calibrate inside a job.
* **This is a workaround, not a fix.** SigLIP's sigmoid loss trains each
  image-text pair independently, so its scores are absolute and a fixed
  threshold finally means something. Plan 3's calibration is what Phase 6.6
  deletes. Do not describe it as more than it is.

**Re-tagging reads zero media, and that must stay true.** A tag score is
`cosine(image_vec, text_vec)` and `embeddings.vector` already holds the
L2-normalised image vectors (norm min/max/mean 1.000000 on all 1,910 rows), so
the whole library re-tags as one `N x 512 @ 512 x 345` GEMM over 3.7 MiB of
SQLite. `tests/test_retag.py::TestRetagReadsNoStorage` pins it: the pass runs
with `MinIOStorageClient` replaced by a stub that raises on every attribute,
`builtins.open` fails the test on any path outside the cache dir, and a fresh
subprocess asserts `import scripts.retag` leaves torch, transformers, minio and
`core.storage_client` out of `sys.modules`. Any storage read added to that path
is a regression, not an optimisation.

**`core/tagger.py` must not import `core.embedder` at module scope.** It did,
for a single type annotation, and measured cost of that one line was **509.3 MB
peak RSS and a torch + transformers import in every process that touched the
tagger** — against 28.8 MB now. The annotation is a string under
`TYPE_CHECKING`. `tests/test_vocabulary.py::TestTaggerImportsNoTorch` checks it
in a fresh subprocess; `core/vocabulary.py` must stay ML-free for the same
reason.

**The label matrix cache is keyed on model *and* vocabulary fingerprint**
(`models/tag_vectors_{model}_{fingerprint}.npy`, NVMe, never MinIO). The
fingerprint covers the labels, their order, `VOCAB_VERSION` and
`PROMPT_TEMPLATE` — changing the template moves every vector. A mismatched
cache is the worst outcome available: right shape, clean run, every tag
silently wrong, so `load_cached_matrix` raises rather than loads. Adding,
removing or reordering a label in `core/vocabulary.py` changes the fingerprint
and invalidates the cache automatically; that is deliberate, do not defeat it.
