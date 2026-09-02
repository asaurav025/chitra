"""
Background job functions for RQ (Redis Queue).
These functions are executed by worker processes.
"""
import os
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from core import db
from core import video
from core.extractor import load_image
from core.storage_client import MinIOStorageClient
from core.gallery import ensure_thumb

# core.embedder, core.face and core.tagger are imported lazily inside the jobs
# that use them. app_fastapi imports this module only to enqueue jobs by
# reference, and a module-scope import here costs every uvicorn worker a full
# torch/transformers import (~450 MB) with no model loaded — which is what
# OOM-killed chitra-api. Keep these imports inside the functions.
# Guarded by tests/test_api_memory_budget.py.


# The one cosine-similarity threshold for "these two faces are the same person".
#
# Four copies of this number used to disagree: 0.75 in _auto_match_face_to_person
# and cluster_faces_job, 0.6 at the upload trigger and in recluster_all.py, 0.75
# in the `faces-cluster` CLI command. Everything that matches faces now reads
# this. (core/cluster.py's 0.78 is deliberately NOT this value — it groups whole
# photos by CLIP embedding, a different space and a different question.)
#
# 0.60 is measured, not guessed. Over the 549 faces already assigned to the 8
# named persons in the live DB (buffalo_l 512-d embeddings, L2-normalised):
#
#   between-person pairs: mean 0.038, 99.99th pct 0.326, absolute max 0.366
#   within-person pairs:  mean 0.629, median 0.644
#   per-person average pairwise similarity: 0.590 - 0.809 (median 0.732)
#
# Two independent uses, both satisfied at 0.60 and neither at 0.75:
#
#   Phase 1, nearest-assigned-neighbour matching. Simulated on the labelled
#   set, precision is 100% at every cut from 0.50 to 0.85 — the populations
#   barely overlap — so the choice is pure recall: 98.2% at 0.60, 94.7% at
#   0.75, 90.9% at 0.78. 0.60 still clears the worst observed impostor pair
#   (0.366) by 0.23.
#
#   Phase 2, the HDBSCAN acceptance gate in cluster_faces_job, which keeps a
#   cluster only if its *average* pairwise similarity >= this value. Real
#   people here average 0.59-0.81, so a 0.75 gate rejects 5 of the 8 known
#   persons (including both large ones: saurav 0.654, swati 0.590) and 0.78
#   rejects 6 of 8. At those values Phase 2 could not have formed a correct
#   cluster for this dataset at all — a second, independent reason the People
#   feature stayed empty.
#
# Re-derive with the same query before changing this; a higher value silently
# stops grouping people rather than failing loudly.
FACE_MATCH_THRESHOLD = 0.60

# Global instances (will be initialized per worker)
_STORAGE_CLIENT = None
_EMBEDDER = None
_EMBED_CLIENT = None


def _is_video(conn, photo_id: int) -> bool:
    """True if the photo row is a video (image-only ML jobs must skip these).

    This is still the gate for **face detection**, which has no poster path and
    must not grow one by accident. Embedding uses `_embed_source_key` instead:
    it runs on the poster, never on the original.
    """
    cur = conn.cursor()
    cur.execute("SELECT media_type FROM photos WHERE id=?", (photo_id,))
    row = cur.fetchone()
    return bool(row and row["media_type"] == "video")


def _embed_source_key(conn, photo_id: int, file_path: str):
    """The object key to embed for this photo, or None if there is nothing to.

    For a video that is the **poster** in `thumb_path`, and never `file_path`.
    258 of the 262 videos already have one: a 512x512 JPEG of ~250 KB, ~64 MB
    for the whole set, already generated and already sitting next to every
    photo thumbnail. The original is a multi-gigabyte MOV on a disk with 3,000+
    unrecovered read errors, and CLIP consumes 224x224 of one frame regardless
    — so reading it would cost four orders of magnitude more for a worse
    vector. That is why videos were excluded in the first place; the poster
    removes the reason rather than the exclusion.

    The 4 videos with no poster return None and read nothing. Falling back to
    `file_path` for them would be strictly worse than doing nothing:
    `generate_video_poster_job` is the way to fix those, and it is the only
    thing that should ever open a video original.

    `media_type` is NULL on 766 of the 2,040 rows, which means photo.
    """
    cur = conn.cursor()
    cur.execute(
        "SELECT COALESCE(media_type, 'photo') AS media_type, thumb_path "
        "FROM photos WHERE id=?",
        (photo_id,),
    )
    row = cur.fetchone()
    if row is None or row["media_type"] != "video":
        return file_path
    return row["thumb_path"] or None


#: The persistent index over every face already assigned to a person.
PERSON_INDEX_NAME = "existing_person_faces"

#: One predicate for "belongs in the person index", used by *both* the
#: freshness count and the rebuild, so the two can never disagree about what
#: the index is supposed to contain.
_INDEXABLE_FACES = "person_id IS NOT NULL AND embedding IS NOT NULL"

#: Neighbours to ask the index for. More than one because the nearest hit may
#: have been unassigned since it was indexed — the next hit is then still a
#: legitimate match — and small enough that resolving them stays one bounded
#: query regardless of how many faces are assigned.
_MATCH_NEIGHBOURS = 5

#: An index further behind the database than this is rebuilt rather than
#: caught up: a handful of missing vectors is cheaper to append than a full
#: HNSW rebuild, a thousand is not.
_INDEX_CATCHUP_LIMIT = 64


def _clustering_result(matched=0, newly_clustered=0, persons_created=0,
                       left_as_noise=0):
    """The shape every ``cluster_faces_job`` return takes."""
    return {
        "clustered": matched + newly_clustered,
        "persons_created": persons_created,
        "matched_to_existing": matched,
        "newly_clustered": newly_clustered,
        "left_as_noise": left_as_noise,
    }


def _indexable_face_count(conn) -> int:
    """How many faces the person index should contain right now."""
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT(*) FROM faces WHERE {_INDEXABLE_FACES}")
    return int(cur.fetchone()[0])


def _read_face_vectors(conn, face_ids=None):
    """
    ``(ids, matrix)`` for indexable faces, ids and rows in the same order.

    This is the O(N) read. Only cold paths — a rebuild, a catch-up, a batch
    job — may call it. Doing it once per *detected face* is the O(N*M) defect
    this module used to have.
    """
    cur = conn.cursor()
    if face_ids is None:
        cur.execute(
            f"SELECT id, embedding FROM faces WHERE {_INDEXABLE_FACES} ORDER BY id"
        )
    else:
        face_ids = list(face_ids)
        if not face_ids:
            return [], None
        placeholders = ",".join("?" * len(face_ids))
        cur.execute(
            f"SELECT id, embedding FROM faces "
            f"WHERE {_INDEXABLE_FACES} AND id IN ({placeholders}) ORDER BY id",
            face_ids,
        )

    ids, vecs, dim = [], [], None
    for row in cur.fetchall():
        fid, emb_bytes = row[0], row[1]
        if not emb_bytes:
            continue
        v = np.frombuffer(emb_bytes, dtype=np.float32)
        if v.size == 0:
            continue
        if dim is None:
            dim = v.size
        elif v.size != dim:
            # A stray odd-sized embedding would make np.stack raise and take
            # the whole index with it. Skip it and keep matching working.
            print(f"Warning: face {fid} embedding is {v.size}-d, expected {dim}; skipped")
            continue
        ids.append(int(fid))
        vecs.append(v)

    if not vecs:
        return [], None
    return ids, np.stack(vecs).astype("float32")


def _person_face_index(conn, index_manager):
    """
    The persistent index over every assigned face: ID-mapped, and in step with
    the database.

    Rebuilt — never trusted — when it is missing or carries no id map. Without
    an id map the caller has to assume index position *i* is row *i* of some
    later query; that assumption is what rotted, and the live index was found
    mis-attributing 715 of its 1,044 positions while still passing a count
    check. With ids, a search returns real ``faces.id`` values and the person
    is looked up from the database, so a stale index costs recall and can
    never cost correctness. That is what makes the cheap freshness rule below
    safe.

    Returns None when there is nothing to match against.
    """
    from core.faiss_index import index_ids, is_id_mapped

    db_count = _indexable_face_count(conn)
    if db_count == 0:
        return None

    index = index_manager.load_index(PERSON_INDEX_NAME)

    if index is not None and is_id_mapped(index):
        behind = db_count - index.ntotal
        if behind <= 0:
            # Level with the database, or holding ids it no longer agrees
            # with — those are dropped when the match is resolved.
            if behind >= -_INDEX_CATCHUP_LIMIT:
                return index
        elif behind <= _INDEX_CATCHUP_LIMIT:
            known = set(index_ids(index))
            cur = conn.cursor()
            cur.execute(f"SELECT id FROM faces WHERE {_INDEXABLE_FACES}")
            absent = [int(r[0]) for r in cur.fetchall() if int(r[0]) not in known]
            ids, xb = _read_face_vectors(conn, absent)
            if ids:
                return index_manager.update_index(PERSON_INDEX_NAME, xb, ids)
            return index

    ids, xb = _read_face_vectors(conn)
    if not ids:
        return None
    return index_manager.build_hnsw_index(xb, PERSON_INDEX_NAME, ids=ids)


def _resolve_person(conn, face_ids, cosines, threshold, exclude=None):
    """
    The best ``(person_id, person_name, similarity)`` among index hits.

    The index supplies candidate ``faces.id`` values and their similarity;
    *who* those faces belong to is read from the database every time. A hit
    whose face has since been unassigned, or whose person has been deleted, is
    skipped rather than trusted — so an index running slightly ahead of the
    database is a recall question, not a correctness one.
    """
    candidates = {}
    for fid, cos in zip(face_ids, cosines):
        fid, cos = int(fid), float(cos)
        if fid < 0 or fid == exclude or cos < threshold:
            continue
        candidates[fid] = max(cos, candidates.get(fid, -1.0))
    if not candidates:
        return None

    cur = conn.cursor()
    placeholders = ",".join("?" * len(candidates))
    cur.execute(
        f"SELECT id, person_id FROM faces WHERE id IN ({placeholders})",
        list(candidates),
    )
    owner = {int(r[0]): int(r[1]) for r in cur.fetchall() if r[1] is not None}
    if not owner:
        return None

    person_ids = sorted(set(owner.values()))
    placeholders = ",".join("?" * len(person_ids))
    cur.execute(
        f"SELECT id, name FROM persons WHERE id IN ({placeholders})", person_ids
    )
    names = {int(r[0]): r[1] for r in cur.fetchall()}

    for fid, cos in sorted(candidates.items(), key=lambda kv: -kv[1]):
        person_id = owner.get(fid)
        if person_id is None or person_id not in names:
            continue
        return person_id, names[person_id], cos
    return None


def _auto_match_face_to_person(conn, face_id: int, face_embedding: np.ndarray,
                               threshold: float = FACE_MATCH_THRESHOLD):
    """
    Assign a newly detected face to an existing person, if one is close enough.

    Goes through the persistent index. This used to select *every*
    person-assigned embedding out of SQLite and build a fresh ``IndexFlatIP``
    for every single detected face — O(N) reads plus a full index build per
    face, O(N*M) for a batch, against the 1,000+ assigned faces this library
    now has. The persistent index exists precisely to avoid that.

    The face it assigns is appended to the index immediately, so the *next*
    face in the same batch can match through it. Without that, matching
    quietly degrades over the course of a run.

    Args:
        conn: Database connection
        face_id: ID of the face to match
        face_embedding: Face embedding vector (numpy array)
        threshold: Cosine similarity threshold for matching.
                   Defaults to FACE_MATCH_THRESHOLD; see its derivation there.
    """
    try:
        from core.faiss_index import FAISSIndexManager, scores_to_cosine
    except ImportError:
        # FAISS not available, skip matching
        return

    try:
        index_manager = FAISSIndexManager()
        index = _person_face_index(conn, index_manager)
        if index is None or index.ntotal == 0:
            # No existing faces to match against
            return

        query = np.array(face_embedding, dtype="float32", copy=True).reshape(1, -1)
        if query.shape[1] != index.d:
            print(f"Warning: face {face_id} is {query.shape[1]}-d but the person "
                  f"index is {index.d}-d; skipping match")
            return

        scores, ids = index_manager.search(
            index, query, k=min(_MATCH_NEIGHBOURS, index.ntotal)
        )
        # Ask the index for its metric instead of guessing from the scores.
        cosines = scores_to_cosine(scores[0], index_manager.metric_of(index))

        match = _resolve_person(conn, ids[0], cosines, threshold, exclude=face_id)
        if match is None:
            return
        person_id, person_name, similarity = match

        cur = conn.cursor()
        cur.execute("UPDATE faces SET person_id=? WHERE id=?", (person_id, face_id))
        conn.commit()

        # Same-batch visibility: the face just assigned has to be in the index
        # before the next face is matched.
        try:
            index_manager.update_index(PERSON_INDEX_NAME, query.copy(), [face_id])
        except Exception as exc:
            print(f"Warning: face {face_id} assigned but not added to the index: {exc}")

        print(f"Auto-matched face {face_id} to existing person '{person_name}' "
              f"(similarity: {similarity:.3f})")
    except Exception as e:
        # If FAISS fails, skip matching
        print(f"Warning: Face matching failed: {e}")


def _get_storage_client():
    """Get or create MinIO storage client instance."""
    global _STORAGE_CLIENT
    if _STORAGE_CLIENT is None:
        _STORAGE_CLIENT = MinIOStorageClient()
    return _STORAGE_CLIENT


def _get_embedder():
    """Get or create an in-process embedder instance.

    **Nothing on the job paths calls this any more** — `process_photo_embedding_job`
    and `_process_single_embedding` both go through the sidecar now (see
    `_get_embed_client`). It is kept for one release so the change is a pure
    revert if the sidecar route turns out to have a problem in production, and
    should be deleted once it has not.

    Do not reintroduce a call to it as a fallback. RQ forks per job, so `_EMBEDDER`
    is built in the child and dies with it: every call here is a full 1.1 GB CLIP
    load, ~90% of the measured 58.5 s job. A fallback would restore that path and
    1.67 GB of residency invisibly — the pipeline would simply become slow again
    with nothing in the logs to say why.
    """
    from core.embedder import ClipEmbedder  # lazy: keeps torch out of the API

    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = ClipEmbedder()
    return _EMBEDDER


def _get_embed_client():
    """The shared sync client for the resident CLIP in `embed_service.py`.

    One per process. In a single-photo job that is one client for one photo; in
    `index_embeddings_batch_job` the five worker threads share it, so the label
    vectors it caches are embedded once for the whole batch rather than once per
    photo.
    """
    from core.embed_client_sync import SyncEmbeddingClient  # no torch, no fastapi model

    global _EMBED_CLIENT
    if _EMBED_CLIENT is None:
        _EMBED_CLIENT = SyncEmbeddingClient()
    return _EMBED_CLIENT


#: Tags kept per photo by the embedding job. Unchanged from the `auto_tags(k=6)`
#: it replaces — Phase 3's calibrated `core.tagger.tag_from_vector` is what
#: makes this count vary per photo, and it needs a corpus calibration this job
#: does not have.
EMBED_JOB_TAG_COUNT = 6


def _embed_and_tag(conn, photo_id: int, key: str, data: bytes) -> None:
    """Store one photo's vector and its tags from a **single** sidecar embed.

    The old path ran CLIP over the same bytes twice: once here for the stored
    vector, and once more inside `auto_tags` -> `rank_labels`, which embeds the
    image again and then recomputes the whole label text batch from scratch. The
    score is a dot product between two already-normalised vectors, so with the
    image vector in hand the tags cost no forward pass at all and the label
    vectors are embedded once per process rather than once per photo.

    `DEFAULT_LABELS` is imported here rather than at module scope because
    `core/jobs.py` is imported by `app_fastapi` purely to enqueue jobs by
    reference — see the note at the top of this file.
    """
    from core.tagger import DEFAULT_LABELS
    from core.vocabulary import LEGACY_VERSION, tag_source

    client = _get_embed_client()
    # Asked *before* the embed so a sidecar that cannot name itself costs one
    # round trip rather than a whole forward pass and a download.
    model = client.served_model()
    vec = np.asarray(client.image_embedding(os.path.basename(key), data), dtype="float32")
    db.put_embedding(conn, photo_id, vec.tobytes(), int(vec.shape[0]), model=model)

    # `DEFAULT_LABELS` is still the legacy 17, so the stamp says v1. Claiming
    # `vocab-v2` here would be the same class of lie as the model name was.
    source = tag_source(model, LEGACY_VERSION)
    for tag, score in client.rank_labels_for_vector(vec, DEFAULT_LABELS, EMBED_JOB_TAG_COUNT):
        db.add_tag(conn, photo_id, tag, float(score), source=source)


def process_photo_embedding_job(photo_id: int, file_path: str, db_path: str):
    """
    Background job to process photo embedding and auto-tags.

    The vector comes from the resident CLIP in `embed_service.py` over loopback,
    not from a model loaded here. RQ forks per job, so an in-process model is
    loaded and discarded every single time — ~90% of the measured 58.5 s this
    job used to take, and 1.67 GB of peak RSS in a cgroup with a history of OOM
    kills. The downloaded bytes go straight to the sidecar, so this also no
    longer writes the whole original out to a temp file.

    **It re-raises.** It used to swallow every exception and return `False`,
    which makes RQ mark the job *successful* — a failed embedding was invisible.
    There is deliberately no in-process fallback when the sidecar is down: see
    `_get_embedder`.

    Args:
        photo_id: Photo ID in database
        file_path: MinIO object key to the photo file
        db_path: Path to SQLite database
    """
    conn = db.connect(db_path)

    try:
        key = _embed_source_key(conn, photo_id, file_path)
        if key is None:
            print(f"Skipping embedding for video photo {photo_id}: no poster")
            return True
        # One download, straight to the sidecar. The disk under MinIO has 3,000+
        # unrecovered read errors; a second read per photo buys nothing.
        file_data = _get_storage_client().download_file(key)
        _embed_and_tag(conn, photo_id, key, file_data)
        return True
    except Exception as e:
        print(f"Error processing embedding for photo {photo_id}: {e}")
        raise
    finally:
        conn.close()


def process_photo_faces_job(photo_id: int, file_path: str, db_path: str, min_score=0.5, thumb_size=160):
    """
    Background job to process photo faces and generate thumbnails.
    
    Args:
        photo_id: Photo ID in database
        file_path: MinIO object key to the photo file
        db_path: Path to SQLite database
        min_score: Minimum face detection score
        thumb_size: Thumbnail size
    """
    from core.face import face_encodings  # lazy: keeps onnxruntime out of the API

    conn = db.connect(db_path)
    storage_client = _get_storage_client()

    try:
        if _is_video(conn, photo_id):
            print(f"Skipping face detection for video photo {photo_id}")
            return 0
        # Download from MinIO to temporary file
        file_data = storage_client.download_file(file_path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file_path).suffix) as tmp:
            tmp.write(file_data)
            tmp_path = tmp.name

        try:
            faces = face_encodings(tmp_path)
            if not faces:
                return 0
            
            img = load_image(Path(tmp_path))
            face_count = 0
            
            for idx_face, f in enumerate(faces):
                if f.get("score", 1.0) < min_score:
                    continue
                
                bbox = f["bbox"]
                x, y, w, h = bbox
                emb = f["embedding"]
                emb_bytes = emb.astype("float32").tobytes()
                
                db.add_face(
                    conn,
                    photo_id=photo_id,
                    face_index=idx_face,
                    embedding_bytes=emb_bytes,
                    bbox_x=float(x),
                    bbox_y=float(y),
                    bbox_w=float(w),
                    bbox_h=float(h),
                )
                
                # Get face_id
                cur = conn.cursor()
                cur.execute(
                    "SELECT id FROM faces WHERE photo_id=? AND face_index=?",
                    (photo_id, idx_face),
                )
                row = cur.fetchone()
                if row:
                    face_id = row["id"]
                    
                    # Generate thumbnail
                    crop = img.crop((x, y, x + w, y + h))
                    crop = crop.resize((thumb_size, thumb_size))
                    
                    # Save to temporary file, then upload to MinIO
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as thumb_tmp:
                        crop.save(thumb_tmp.name, "JPEG", quality=100)
                        with open(thumb_tmp.name, 'rb') as f:
                            thumb_data = f.read()
                    
                    # Upload to MinIO
                    face_thumb_path = storage_client.generate_thumbnail_path(face_id, "face")
                    storage_client.upload_file(thumb_data, face_thumb_path)
                    db.set_face_thumb(conn, face_id, face_thumb_path)
                    
                    # Auto-match face to existing persons
                    try:
                        _auto_match_face_to_person(conn, face_id, emb)
                    except Exception as e:
                        # Don't fail face processing if matching fails
                        print(f"Warning: Auto-matching face {face_id} failed: {e}")
                    
                    # Cleanup
                    os.unlink(thumb_tmp.name)
                    face_count += 1
            
            return face_count
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        print(f"Error processing faces for photo {photo_id}: {e}")
        return 0
    finally:
        conn.close()


def _process_single_embedding(pid: int, file_path: str, db_path: str) -> bool:
    """Process embedding for a single photo (used in parallel processing).

    Same sidecar route as `process_photo_embedding_job`, and it re-raises for
    the same reason. `index_embeddings_batch_job` catches per photo, so one
    failure costs one photo rather than the batch.
    """
    conn = db.connect(db_path)

    try:
        key = _embed_source_key(conn, pid, file_path)
        if key is None:
            print(f"Skipping embedding for video photo {pid}: no poster")
            return False
        file_data = _get_storage_client().download_file(key)
        _embed_and_tag(conn, pid, key, file_data)
        return True
    except Exception as e:
        print(f"Error processing embedding for photo {pid}: {e}")
        raise
    finally:
        conn.close()


def index_embeddings_batch_job(photo_ids_and_paths: list, db_path: str, incremental: bool):
    """
    Background job to process embeddings for multiple photos in parallel.
    
    Args:
        photo_ids_and_paths: List of tuples (photo_id, file_path)
        db_path: Path to SQLite database
        incremental: Whether to skip photos that already have embeddings
    """
    indexed = 0
    max_workers = min(5, len(photo_ids_and_paths))  # Max 5 concurrent processing
    
    # Process in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_photo = {
            executor.submit(_process_single_embedding, pid, file_path, db_path): (pid, file_path)
            for pid, file_path in photo_ids_and_paths
        }
        
        for future in as_completed(future_to_photo):
            pid, file_path = future_to_photo[future]
            try:
                if future.result():
                    indexed += 1
            except Exception as e:
                print(f"Error processing photo {pid}: {e}")
    
    return indexed


def _process_single_face(pid: int, file_path: str, db_path: str, min_score: float, thumb_size: int) -> bool:
    """Process faces for a single photo (used in parallel processing)."""
    from core.face import face_encodings  # lazy: keeps onnxruntime out of the API

    conn = db.connect(db_path)
    storage_client = _get_storage_client()

    try:
        if _is_video(conn, pid):
            print(f"Skipping face detection for video photo {pid}")
            return False
        # Download from MinIO to temporary file
        file_data = storage_client.download_file(file_path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file_path).suffix) as tmp:
            tmp.write(file_data)
            tmp_path = tmp.name

        try:
            faces = face_encodings(tmp_path)
            if not faces:
                return False
            
            img = load_image(Path(tmp_path))
            
            for idx_face, f in enumerate(faces):
                if f.get("score", 1.0) < min_score:
                    continue
                
                bbox = f["bbox"]
                x, y, w, h = bbox
                emb = f["embedding"]
                emb_bytes = emb.astype("float32").tobytes()
                
                db.add_face(
                    conn,
                    photo_id=pid,
                    face_index=idx_face,
                    embedding_bytes=emb_bytes,
                    bbox_x=float(x),
                    bbox_y=float(y),
                    bbox_w=float(w),
                    bbox_h=float(h),
                )
                
                # Get face_id
                cur = conn.cursor()
                cur.execute(
                    "SELECT id FROM faces WHERE photo_id=? AND face_index=?",
                    (pid, idx_face),
                )
                row = cur.fetchone()
                if row:
                    face_id = row["id"]
                    
                    # Generate thumbnail
                    crop = img.crop((x, y, x + w, y + h))
                    crop = crop.resize((thumb_size, thumb_size))
                    
                    # Save to temporary file, then upload to MinIO
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as thumb_tmp:
                        crop.save(thumb_tmp.name, "JPEG", quality=100)
                        with open(thumb_tmp.name, 'rb') as f:
                            thumb_data = f.read()
                    
                    # Upload to MinIO
                    face_thumb_path = storage_client.generate_thumbnail_path(face_id, "face")
                    storage_client.upload_file(thumb_data, face_thumb_path)
                    db.set_face_thumb(conn, face_id, face_thumb_path)
                    
                    # Auto-match face to existing persons
                    try:
                        _auto_match_face_to_person(conn, face_id, emb)
                    except Exception as e:
                        # Don't fail face processing if matching fails
                        print(f"Warning: Auto-matching face {face_id} failed: {e}")
                    
                    # Cleanup
                    os.unlink(thumb_tmp.name)
            
            return True
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        print(f"Error processing faces for photo {pid}: {e}")
        return False
    finally:
        conn.close()


def index_faces_batch_job(photo_ids_and_paths: list, db_path: str, min_score=0.5, thumb_size=160):
    """
    Background job to process faces for multiple photos in parallel.
    
    Args:
        photo_ids_and_paths: List of tuples (photo_id, file_path)
        db_path: Path to SQLite database
        min_score: Minimum face detection score
        thumb_size: Thumbnail size
    """
    processed = 0
    max_workers = min(5, len(photo_ids_and_paths))  # Max 5 concurrent processing
    
    # Process in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_photo = {
            executor.submit(_process_single_face, pid, file_path, db_path, min_score, thumb_size): (pid, file_path)
            for pid, file_path in photo_ids_and_paths
        }
        
        for future in as_completed(future_to_photo):
            pid, file_path = future_to_photo[future]
            try:
                if future.result():
                    processed += 1
            except Exception as e:
                print(f"Error processing photo {pid}: {e}")
    
    return processed



def process_video_transcode_job(photo_id: int, file_path: str, db_path: str):
    """Probe an uploaded video; if it isn't web-safe (e.g. HEVC .mov), transcode to an
    H.264/AAC MP4 derivative for browser playback. Writes dimensions/duration/codec and a
    `transcode_status` the UI polls. Heavy/CPU-bound — runs on the dedicated `video` queue.
    """
    conn = db.connect(db_path)
    storage_client = _get_storage_client()
    tmp_src = None
    tmp_out = None
    try:
        if not video.ensure_ffmpeg():
            print(f"ffmpeg/ffprobe not available; cannot transcode video {photo_id}")
            db.update_video_fields(conn, photo_id, transcode_status="failed")
            return False

        db.update_video_fields(conn, photo_id, transcode_status="processing")

        ext = Path(file_path).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp_src = tmp.name
        storage_client.download_to_path(file_path, tmp_src)

        info = video.ffprobe_info(tmp_src)
        db.update_video_fields(
            conn,
            photo_id,
            width=info.get("width"),
            height=info.get("height"),
            duration_seconds=info.get("duration_seconds"),
            video_codec=info.get("video_codec"),
        )

        if video.is_web_safe(info, ext):
            db.update_video_fields(
                conn, photo_id, playback_path=file_path, transcode_status="not_needed"
            )
            print(f"Video {photo_id} is already web-safe; serving original")
            return True

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp_out = tmp.name
        video.transcode_to_h264(tmp_src, tmp_out)

        playback_key = storage_client.generate_playback_path(photo_id)
        storage_client.upload_file_from_path(tmp_out, playback_key)
        db.update_video_fields(
            conn, photo_id, playback_path=playback_key, transcode_status="ready"
        )
        print(f"Video {photo_id} transcoded -> {playback_key}")
        return True
    except Exception as e:
        print(f"Error transcoding video {photo_id}: {e}")
        try:
            db.update_video_fields(conn, photo_id, transcode_status="failed")
        except Exception:
            pass
        return False
    finally:
        for p in (tmp_src, tmp_out):
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass
        conn.close()


def generate_video_poster_job(photo_id: int, file_path: str, db_path: str):
    """Extract a poster keyframe for a video and upload it as its thumbnail.

    This used to happen inline in the API's upload handler. Two of the six
    chitra-api OOM kills named `av:hevc:df0` / `av:hevc:df6` — ffmpeg HEVC
    decode threads running inside the API cgroup. Poster extraction is ffmpeg
    work and belongs in a worker.

    Runs on the **default** queue, not `video`: the video queue has 2 workers
    and a poster enqueued behind a multi-minute 4K transcode would arrive
    minutes late. The default queue has 4 and is otherwise idle-ish, so the
    poster lands in 1-3 s. The cost is one duplicated download of the original
    — deliberate, and cheaper than the wait.

    Deliberately touches **no** `transcode_status` value. A transcode for the
    same photo may be running concurrently on the other queue and owns that
    column; writing it from here could strand the video in a state nothing
    retries.

    Re-raises on failure so RQ marks the job failed (AGENTS.md: returning False
    makes a failed job look successful).
    """
    storage_client = _get_storage_client()
    thumb_path = storage_client.generate_thumbnail_path(photo_id, "photo")

    # A grid render of 50 videos can fire 50 enqueues; make the redundant ones
    # cheap, and above all do not re-download the original for them.
    if storage_client.file_exists(thumb_path):
        print(f"Poster for photo {photo_id} already exists at {thumb_path}")
        return thumb_path

    if not video.ensure_ffmpeg():
        raise RuntimeError(f"ffmpeg/ffprobe not available; cannot make a poster for {photo_id}")

    conn = None
    tmp_src = None
    poster_tmp = None
    try:
        ext = Path(file_path).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp_src = tmp.name
        # download_to_path streams to disk. download_file would return the
        # whole original as bytes — a multi-GB video straight into RAM.
        storage_client.download_to_path(file_path, tmp_src)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as pt:
            poster_tmp = pt.name
        video.extract_poster(tmp_src, poster_tmp)

        storage_client.upload_file_from_path(poster_tmp, thumb_path)

        conn = db.connect(db_path)
        conn.execute("UPDATE photos SET thumb_path = ? WHERE id = ?", (thumb_path, photo_id))
        conn.commit()
        print(f"Poster for photo {photo_id} -> {thumb_path}")
        return thumb_path
    except Exception as e:
        print(f"Error generating poster for photo {photo_id}: {e}")
        raise
    finally:
        if conn is not None:
            conn.close()
        for p in (tmp_src, poster_tmp):
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass


def cluster_faces_job(db_path: str, threshold: float = FACE_MATCH_THRESHOLD,
                      photo_ids: list = None, reset: bool = False):
    """
    Background job to cluster unassigned faces into persons using HDBSCAN.
    
    This job runs after face detection jobs complete to automatically
    group similar faces into persons.
    
    Args:
        db_path: Path to SQLite database
        threshold: Cosine similarity threshold for both matching against
                   existing persons and accepting a new HDBSCAN cluster.
                   Defaults to FACE_MATCH_THRESHOLD; see its derivation there.
        photo_ids: Optional list of photo IDs to limit clustering to faces from these photos only.
                   If None, clusters all unassigned faces (less efficient for large databases).
        reset: If True, unassign all faces before clustering (re-cluster everything).
    """
    import faiss
    from core.faiss_index import FAISSIndexManager, scores_to_cosine
    
    conn = db.connect(db_path)
    index_manager = FAISSIndexManager()
    
    try:
        # If reset=True, unassign all faces first (re-cluster everything)
        if reset:
            cur = conn.cursor()
            cur.execute("UPDATE faces SET person_id = NULL WHERE person_id IS NOT NULL")
            conn.commit()
            print(f"✓ Reset all face assignments - re-clustering all {cur.rowcount} faces")
            # Every id in the index now points at an unassigned face. Left in
            # place it would never be rebuilt (it only *grows* relative to the
            # database) and every search would hit dead ids.
            index_manager.delete_index(PERSON_INDEX_NAME)
        # Get unassigned faces with embeddings
        # If photo_ids provided, only cluster faces from those photos (more efficient)
        cur = conn.cursor()
        if photo_ids and len(photo_ids) > 0:
            # Only cluster faces from the specified photos
            placeholders = ','.join(['?'] * len(photo_ids))
            cur.execute(f"""
                SELECT id, embedding
                FROM faces
                WHERE person_id IS NULL 
                  AND embedding IS NOT NULL
                  AND photo_id IN ({placeholders})
            """, photo_ids)
            print(f"Clustering faces from {len(photo_ids)} specific photos (efficient mode)")
        else:
            # Cluster all unassigned faces (less efficient for large databases)
            cur.execute("""
                SELECT id, embedding
                FROM faces
                WHERE person_id IS NULL AND embedding IS NOT NULL
            """)
            print("Clustering all unassigned faces (full database scan)")
        unassigned_faces = cur.fetchall()
        
        if not unassigned_faces:
            print("No unassigned faces to cluster")
            return _clustering_result()
        
        # Process unassigned faces
        unassigned_face_ids = []
        unassigned_vecs = []
        
        for row in unassigned_faces:
            fid, emb_bytes = row
            if not emb_bytes:
                continue
            v = np.frombuffer(emb_bytes, dtype=np.float32)
            if v.size == 0:
                continue
            unassigned_face_ids.append(fid)
            unassigned_vecs.append(v)
        
        if not unassigned_vecs:
            print("No valid embeddings in unassigned faces")
            return _clustering_result()
        
        # Phase 1: match unassigned faces against every face already assigned.
        #
        # The index is ID-mapped, so a hit is a real ``faces.id`` and the
        # person it belongs to is read from the database. The old code took the
        # hit's *position* and used it to index its own query results — correct
        # only if the index had been built from that same unordered query, in
        # that same order, and never appended to since. It had not been.
        matched_assignments = []
        unmatched_indices = []
        existing_index = _person_face_index(conn, index_manager)

        if existing_index is not None and existing_index.ntotal > 0:
            batch_xb = np.stack(unassigned_vecs).astype("float32")
            k = min(_MATCH_NEIGHBOURS, existing_index.ntotal)
            D_match, I_match = index_manager.search(
                existing_index, batch_xb, k=k, ef_search=50
            )
            # Ask the index for its metric rather than guessing from the
            # batch's own maximum; that guess dropped good matches.
            D_match = scores_to_cosine(D_match, index_manager.metric_of(existing_index))

            for i, face_id in enumerate(unassigned_face_ids):
                # Check if face is still unassigned
                cur.execute("SELECT person_id FROM faces WHERE id=?", (face_id,))
                face_row = cur.fetchone()
                if face_row and face_row[0] is not None:
                    continue

                match = _resolve_person(
                    conn, I_match[i], D_match[i], threshold, exclude=face_id
                )
                if match is None:
                    unmatched_indices.append(i)
                else:
                    matched_assignments.append((face_id, match[0]))
        else:
            unmatched_indices = list(range(len(unassigned_face_ids)))
        
        # Batch update matched faces
        if matched_assignments:
            for face_id, person_id in matched_assignments:
                cur.execute("UPDATE faces SET person_id=? WHERE id=?", (person_id, face_id))
            conn.commit()
        matched_count = len(matched_assignments)
        
        # Keep the persistent index in step with what Phase 1 just assigned,
        # with ids, so the next run (and the per-face path) can see them.
        if matched_assignments:
            try:
                ids, matched_xb = _read_face_vectors(
                    conn, [fid for fid, _ in matched_assignments]
                )
                if ids:
                    existing_index = index_manager.update_index(
                        PERSON_INDEX_NAME, matched_xb, ids
                    )
                    print(f"✓ Updated FAISS index: added {len(ids)} matched faces")
            except Exception as e:
                print(f"Warning: Failed to update FAISS index with matched faces: {e}")
                # Continue - the index is rebuilt when it falls behind.
        
        # Phase 2: Cluster unmatched faces into new persons using HDBSCAN
        persons_created = 0
        newly_clustered = 0
        if unmatched_indices:
            unmatched_face_ids = [unassigned_face_ids[i] for i in unmatched_indices]
            unmatched_vecs = [unassigned_vecs[i] for i in unmatched_indices]
            
            print(f"Debug: Phase 2 - Starting HDBSCAN clustering for {len(unmatched_face_ids)} unmatched faces with threshold {threshold}")
            
            xb = np.stack(unmatched_vecs).astype("float32")
            n = xb.shape[0]
            
            # Normalize embeddings for cosine similarity
            # HDBSCAN with cosine metric expects normalized vectors
            xb_normalized = xb.copy()
            faiss.normalize_L2(xb_normalized)
            
            # Import HDBSCAN
            try:
                import hdbscan
            except ImportError:
                print("Error: hdbscan not installed. Please install it: pip install hdbscan")
                raise
            
            # Configure HDBSCAN parameters
            # min_cluster_size: minimum number of faces in a cluster (allow pairs)
            # min_samples: minimum neighbors (lower = more permissive, higher = more strict)
            # For threshold 0.75, use min_samples=2 to allow pairs with high similarity
            # Lower threshold = more permissive clustering
            min_cluster_size = max(2, int(n * 0.01))  # At least 2, or 1% of faces
            min_samples = max(1, int(min_cluster_size * 0.5))  # Half of min_cluster_size

            # HDBSCAN asks its kd-tree for min_samples + 1 neighbours, so a
            # batch smaller than that raises "k must be less than or equal to
            # the number of training points" and takes the whole job down with
            # it — six failed clustering jobs in half an hour of live logs,
            # one for every upload that produced a single unrecognised face.
            # Too few faces to form a cluster is not an error; it is noise,
            # and it is now reported as noise.
            too_small_to_cluster = n < max(min_cluster_size, min_samples + 1)
            
            # Use euclidean metric on normalized vectors (equivalent to cosine distance)
            # For L2-normalized vectors, euclidean distance = sqrt(2 * (1 - cosine_similarity))
            # So we adjust the epsilon accordingly
            # Cosine similarity threshold -> euclidean distance threshold
            # cosine_sim >= threshold -> euclidean_dist <= sqrt(2 * (1 - threshold))
            # Ensure epsilon is valid (>= 0 and reasonable)
            euclidean_threshold = np.sqrt(max(0.0, 2.0 * (1.0 - threshold)))
            # Cap epsilon at a reasonable maximum (e.g., 2.0 for normalized vectors)
            euclidean_threshold = min(euclidean_threshold, 2.0)
            
            # HDBSCAN parameters
            clusterer_kwargs = {
                'min_cluster_size': min_cluster_size,
                'min_samples': min_samples,
                'metric': 'euclidean',  # Use euclidean on normalized vectors (equivalent to cosine)
                'cluster_selection_method': 'eom',  # Excess of Mass
                'prediction_data': True,  # Enable approximate_predict for future matching
                # HDBSCAN refuses to select the root of its condensed tree
                # unless asked, and a batch of "one person plus strangers" has
                # nothing *but* a root: the strangers fall out one at a time,
                # which is never a binary split, so no child cluster is ever
                # formed. Measured on a 6-tight-faces + 15-strangers fixture:
                # the default returns 21 noise points and zero clusters; with
                # this it returns the 6 as one cluster and the 15 as noise. On
                # 10 + 20 it returns the 10 intact instead of splitting them
                # into 5 and 4. That degenerate shape is the *common* one for
                # an incremental run over a few photos of one family member,
                # and it silently produced no people at all.
                #
                # The risk this takes on — one sprawling cluster of everybody
                # — is what the average-pairwise-similarity gate below is for:
                # a cluster is only kept if it averages >= threshold, and a
                # cluster of unrelated faces averages far below it.
                'allow_single_cluster': True,
            }
            
            # Only add epsilon if it's positive and reasonable
            if euclidean_threshold > 0:
                clusterer_kwargs['cluster_selection_epsilon'] = float(euclidean_threshold)
            
            clusterer = hdbscan.HDBSCAN(**clusterer_kwargs)
            
            # Perform clustering
            if too_small_to_cluster:
                print(f"Debug: Phase 2 - {n} unmatched face(s) is fewer than HDBSCAN "
                      f"can cluster (needs {max(min_cluster_size, min_samples + 1)}); "
                      f"leaving them as noise")
                cluster_labels = np.full(n, -1, dtype=int)
            else:
                cluster_labels = clusterer.fit_predict(xb_normalized)
            
            # Filter clusters based on average similarity within cluster
            # Collect valid clusters (exclude noise points with label -1)
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label >= 0:  # Valid cluster (not noise)
                    clusters.setdefault(label, []).append(i)
            
            # Filter clusters by average similarity
            valid_clusters = {}
            for label, members in clusters.items():
                if len(members) < 2:
                    continue
                
                # Calculate average cosine similarity within cluster
                cluster_vecs = xb_normalized[members]
                # Compute pairwise cosine similarities
                similarities = np.dot(cluster_vecs, cluster_vecs.T)
                # Get upper triangle (excluding diagonal)
                triu_indices = np.triu_indices(len(members), k=1)
                avg_similarity = similarities[triu_indices].mean()
                
                # Only keep clusters with average similarity >= threshold
                if avg_similarity >= threshold:
                    valid_clusters[label] = members
                    print(f"Debug: Cluster {label} has {len(members)} faces with avg similarity {avg_similarity:.3f}")
                else:
                    print(f"Debug: Cluster {label} rejected (avg similarity {avg_similarity:.3f} < {threshold})")
            
            clusters = valid_clusters
            
            # Create persons and assign faces
            if clusters:
                # Get existing person names
                cur.execute("SELECT name FROM persons")
                existing_names = {row[0] for row in cur.fetchall()}
                
                person_idx = 1
                cluster_assignments = []
                
                for root, members in clusters.items():
                    if not members:
                        continue
                    
                    # Find next available person name
                    while True:
                        person_name = f"Person {person_idx}"
                        if person_name not in existing_names:
                            break
                        person_idx += 1
                    
                    # Create person
                    cur.execute("INSERT INTO persons (name) VALUES (?)", (person_name,))
                    person_id = cur.lastrowid
                    existing_names.add(person_name)
                    person_idx += 1
                    persons_created += 1
                    
                    # Assign faces
                    for m in members:
                        if m >= len(unmatched_face_ids):
                            continue
                        fid = unmatched_face_ids[m]
                        cluster_assignments.append((fid, person_id))
                    
                    print(f"Debug: Created person '{person_name}' (ID: {person_id}) with {len(members)} faces")
                
                # Batch update
                if cluster_assignments:
                    for face_id, person_id in cluster_assignments:
                        cur.execute("UPDATE faces SET person_id=? WHERE id=?", (person_id, face_id))
                    conn.commit()
                    print(f"✓ Successfully assigned {len(cluster_assignments)} faces to {persons_created} new persons")
                    
                    newly_clustered = len(cluster_assignments)

                    # Keep the persistent index in step, *with ids*. The old
                    # code re-selected the embeddings without their ids and
                    # appended them positionally, which is how position stopped
                    # meaning identity in the first place.
                    try:
                        ids, cluster_xb = _read_face_vectors(
                            conn, [fid for fid, _ in cluster_assignments]
                        )
                        if ids:
                            existing_index = index_manager.update_index(
                                PERSON_INDEX_NAME, cluster_xb, ids
                            )
                            print(f"✓ Updated FAISS index: added {len(ids)} newly clustered faces")
                    except Exception as e:
                        print(f"Warning: Failed to update FAISS index with clustered faces: {e}")
                        # Continue - the index is rebuilt when it falls behind.
        
        # Report what was actually assigned.
        #
        # The old total was ``matched_count + len(unmatched_indices)``, which
        # added every face that went *into* HDBSCAN — including the ones it
        # left as noise and never assigned to anybody. The catch-up run
        # reported 1532 clustered while assigning 495. That is the same class
        # of metric that hid the dead scheduler: a pipeline that fails and
        # reports success. The three outcomes are now separate numbers, and
        # ``clustered`` counts only faces that came out of this job with a
        # person on them.
        left_as_noise = len(unmatched_indices) - newly_clustered
        total_clustered = matched_count + newly_clustered
        print(
            f"✓ Clustering complete: {total_clustered} faces assigned "
            f"({matched_count} to existing persons, {newly_clustered} into "
            f"{persons_created} new persons), {left_as_noise} left as noise"
        )

        return {
            "clustered": total_clustered,
            "persons_created": persons_created,
            "matched_to_existing": matched_count,
            "newly_clustered": newly_clustered,
            "left_as_noise": left_as_noise,
        }
        
    except Exception as e:
        print(f"Error in cluster_faces_job: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        conn.close()


def update_faiss_index_after_merge_job(
    db_path: str,
    source_person_id: int,
    target_person_id: int
):
    """
    Background job to update FAISS index after merging persons.
    
    Args:
        db_path: Path to SQLite database
        source_person_id: Source person ID that was merged (for logging)
        target_person_id: Target person ID that faces were merged into
    """
    import faiss
    import numpy as np
    from core.faiss_index import FAISSIndexManager
    
    conn = db.connect(db_path)
    index_manager = FAISSIndexManager()
    
    try:
        cur = conn.cursor()
        
        # Get target person name for logging
        cur.execute("SELECT name FROM persons WHERE id=?", (target_person_id,))
        person_row = cur.fetchone()
        target_person_name = person_row["name"] if person_row else f"Person #{target_person_id}"
        
        # A merge does not change *which* faces are indexed — the ids are the
        # same faces — only which person they belong to, and that is read from
        # the database when a match is resolved. So the only work here is to
        # add any of the target person's faces the index does not already
        # hold, with their ids.
        from core.faiss_index import index_ids, is_id_mapped

        cur.execute(
            f"SELECT id FROM faces WHERE person_id = ? AND {_INDEXABLE_FACES}",
            (target_person_id,),
        )
        target_face_ids = [int(r[0]) for r in cur.fetchall()]

        if not target_face_ids:
            print(f"ℹ No faces with embeddings found for person {target_person_id} after merge")
            return {"updated": 0, "message": "No faces to update"}

        existing_index = index_manager.load_index(PERSON_INDEX_NAME)
        if existing_index is None or not is_id_mapped(existing_index):
            # A missing or legacy index is rebuilt from scratch the next time
            # anything matches; appending to it here would only entrench it.
            print(f"ℹ FAISS index absent or unmapped - it is rebuilt on next match. "
                  f"Merged {len(target_face_ids)} faces to person {target_person_id} ({target_person_name})")
            return {"updated": 0, "message": "Index will be rebuilt on next match"}

        known = set(index_ids(existing_index))
        absent = [fid for fid in target_face_ids if fid not in known]
        if not absent:
            print(f"✓ FAISS index already holds all {len(target_face_ids)} faces of "
                  f"person {target_person_id} ({target_person_name})")
            return {"updated": 0, "message": "Index already up to date"}

        ids, merged_xb = _read_face_vectors(conn, absent)
        if not ids:
            print("ℹ No valid embeddings found for merged faces")
            return {"updated": 0, "message": "No valid embeddings"}

        index_manager.update_index(PERSON_INDEX_NAME, merged_xb, ids)

        print(f"✓ Updated FAISS index: Added {len(ids)} merged faces from person "
              f"{source_person_id} to person {target_person_id} ({target_person_name})")

        return {
            "updated": len(ids),
            "message": f"Successfully updated FAISS index with {len(ids)} face embeddings"
        }
        
    except Exception as e:
        print(f"Error updating FAISS index after merge: {e}")
        import traceback
        traceback.print_exc()
        # Don't raise - merge already succeeded in database
        return {"updated": 0, "error": str(e), "message": "Index update failed but merge succeeded"}
    finally:
        conn.close()


def rebuild_faiss_index_job(db_path: str):
    """
    Background job to rebuild FAISS index from all persons with faces.
    This is useful after merging persons or when the index is out of sync.
    
    Args:
        db_path: Path to SQLite database
    """
    import faiss
    import numpy as np
    from core.faiss_index import FAISSIndexManager
    
    conn = db.connect(db_path)
    index_manager = FAISSIndexManager()
    
    try:
        cur = conn.cursor()
        
        # Rebuild from the database, *with* ids. An index built without them
        # is the legacy shape: position i means row i of whatever unordered
        # query happened to build it, which is the mis-attribution this whole
        # path exists to end. _person_face_index throws such an index away on
        # sight, so building one here would just guarantee another rebuild.
        ids, xb = _read_face_vectors(conn)
        if not ids:
            print("ℹ No faces with embeddings found for any person")
            index_manager.delete_index(PERSON_INDEX_NAME)
            return {"updated": 0, "message": "No faces to index"}

        cur.execute(
            f"SELECT COUNT(DISTINCT person_id) FROM faces WHERE {_INDEXABLE_FACES}"
        )
        person_count = int(cur.fetchone()[0])

        index_manager.build_hnsw_index(
            xb, PERSON_INDEX_NAME, m=32, ef_construction=200, ids=ids
        )
        print(f"✓ Rebuilt HNSW FAISS index with {len(ids)} faces from {person_count} persons")

        return {
            "updated": len(ids),
            "persons": person_count,
            "message": f"Successfully rebuilt FAISS index with {len(ids)} face embeddings from {person_count} persons"
        }
        
    except Exception as e:
        print(f"Error rebuilding FAISS index: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        conn.close()
