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
the module-level `_EMBEDDER` and `_FACE_APP` caches are populated in the child
and thrown away when it exits. CLIP (1.2 GB) and the five buffalo_l ONNX files
(326 MB) reload every single time — that is ~90% of the measured 58s embedding
job and 37s face job. Assume any per-job model load is the dominant cost.

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

**Videos get no ML.** Jobs re-check `_is_video()` and bail. Poster frames exist
but are not embedded, so videos are invisible to search.

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
