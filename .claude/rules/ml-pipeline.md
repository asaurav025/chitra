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

**`put_embedding` and `add_tag` are plain INSERTs with no unique constraint.**
Re-running a non-incremental bulk index would duplicate every embedding and tag.
Fix the constraint before any re-embedding campaign.
