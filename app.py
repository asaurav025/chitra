
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from flask import Flask, jsonify, request, send_file

from core import db
from core.extractor import collect_metadata, load_image, iter_images
from core.embedder import ClipEmbedder
from core.gallery import ensure_thumb
from core.face import face_encodings
from core.tagger import auto_tags


# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
DB_PATH = os.environ.get("CHITRA_DB_PATH", db.DB_DEFAULT_PATH)
PHOTO_ROOT = Path(os.environ.get("CHITRA_PHOTO_ROOT", "photos")).resolve()
THUMB_ROOT = PHOTO_ROOT / ".thumbs"
FACE_THUMB_ROOT = PHOTO_ROOT / ".faces"

PHOTO_ROOT.mkdir(parents=True, exist_ok=True)
THUMB_ROOT.mkdir(parents=True, exist_ok=True)
FACE_THUMB_ROOT.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

_EMBEDDER: Optional[ClipEmbedder] = None


def get_conn():
    return db.connect(DB_PATH)


def get_embedder() -> ClipEmbedder:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = ClipEmbedder()
    return _EMBEDDER


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def row_to_photo_dto(row) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "file_path": row["file_path"],
        "size": row["size"],
        "created_at": row["created_at"],
        "checksum": row["checksum"],
        "phash": row["phash"],
        "exif_datetime": row["exif_datetime"],
        "latitude": row["latitude"],
        "longitude": row["longitude"],
    }


def ensure_photo_thumb(file_path: str, photo_id: int) -> Path:
    thumb_path = THUMB_ROOT / f"{photo_id}.jpg"
    if not thumb_path.exists():
        thumb_path.parent.mkdir(parents=True, exist_ok=True)
        ensure_thumb(file_path, str(thumb_path))
    return thumb_path


# -----------------------------------------------------------------------------
# LIFECYCLE
# -----------------------------------------------------------------------------
@app.before_first_request
def init_db():
    db.init_db(DB_PATH)


# -----------------------------------------------------------------------------
# BASIC
# -----------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "db_path": str(DB_PATH), "photo_root": str(PHOTO_ROOT)})


# -----------------------------------------------------------------------------
# PHOTOS: LIST + DETAIL
# -----------------------------------------------------------------------------
@app.get("/api/photos")
def list_photos():
    limit = int(request.args.get("limit", 100))
    offset = int(request.args.get("offset", 0))

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """SELECT * FROM photos ORDER BY id DESC LIMIT ? OFFSET ?""",
        (limit, offset),
    )
    rows = cur.fetchall()
    conn.close()

    items = [row_to_photo_dto(r) for r in rows]
    return jsonify({"items": items, "limit": limit, "offset": offset})


@app.get("/api/photos/<int:photo_id>")
def get_photo(photo_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM photos WHERE id=?", (photo_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return jsonify({"error": "photo_not_found"}), 404

    return jsonify(row_to_photo_dto(row))


# -----------------------------------------------------------------------------
# PHOTOS: IMAGE + THUMB
# -----------------------------------------------------------------------------
@app.get("/api/photos/<int:photo_id>/image")
def get_photo_image(photo_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT file_path FROM photos WHERE id=?", (photo_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return jsonify({"error": "photo_not_found"}), 404

    file_path = row["file_path"]
    if not os.path.isabs(file_path):
        file_path = str(PHOTO_ROOT / file_path)

    if not os.path.exists(file_path):
        return jsonify({"error": "file_missing_on_disk"}), 404

    return send_file(file_path)


@app.get("/api/photos/<int:photo_id>/thumbnail")
def get_photo_thumbnail(photo_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT file_path FROM photos WHERE id=?", (photo_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return jsonify({"error": "photo_not_found"}), 404

    file_path = row["file_path"]
    if not os.path.isabs(file_path):
        file_path = str(PHOTO_ROOT / file_path)

    if not os.path.exists(file_path):
        return jsonify({"error": "file_missing_on_disk"}), 404

    thumb_path = ensure_photo_thumb(file_path, photo_id)
    return send_file(str(thumb_path))


# -----------------------------------------------------------------------------
# PHOTOS: UPLOAD
# -----------------------------------------------------------------------------
@app.post("/api/photos/upload")
def upload_photos():
    if "file" not in request.files and "files" not in request.files:
        return jsonify({"error": "no_files"}), 400

    files = []
    if "file" in request.files:
        files.append(request.files["file"])
    files.extend(request.files.getlist("files"))

    saved = []

    conn = get_conn()

    for storage in files:
        filename = storage.filename
        if not filename:
            continue

        # Normalize path: store under PHOTO_ROOT using filename (or unique name)
        dest = PHOTO_ROOT / filename
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Avoid overwriting: add suffix if exists
        base = dest.stem
        ext = dest.suffix
        counter = 1
        while dest.exists():
            dest = dest.with_name(f"{base}_{counter}{ext}")
            counter += 1

        storage.save(dest)

        # Collect metadata + DB upsert
        meta = collect_metadata(dest)
        db.upsert_photo(conn, **meta)

        # Fetch photo id
        cur = conn.cursor()
        cur.execute("SELECT id FROM photos WHERE file_path=?", (str(dest),))
        row = cur.fetchone()
        if not row:
            continue
        photo_id = row["id"]

        # Thumbnail
        thumb_path = ensure_photo_thumb(str(dest), photo_id)

        saved.append(
            {
                "id": photo_id,
                "file_path": str(dest),
                "thumbnail": str(thumb_path),
            }
        )

    conn.close()
    return jsonify({"saved": saved})


# -----------------------------------------------------------------------------
# PHOTOS: SCAN EXISTING DIRECTORY
# -----------------------------------------------------------------------------
@app.post("/api/photos/scan-path")
def scan_path():
    data = request.get_json(force=True, silent=True) or {}
    root = data.get("root")
    if not root:
        return jsonify({"error": "missing_root"}), 400

    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        return jsonify({"error": "root_not_found"}), 400

    conn = get_conn()

    processed = 0
    for img_path in iter_images(root_path):
        meta = collect_metadata(img_path)
        db.upsert_photo(conn, **meta)
        processed += 1

    conn.close()
    return jsonify({"root": str(root_path), "processed": processed})


# -----------------------------------------------------------------------------
# TAGS
# -----------------------------------------------------------------------------
@app.get("/api/photos/<int:photo_id>/tags")
def get_photo_tags(photo_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT tag, score FROM tags WHERE photo_id=? ORDER BY score DESC", (photo_id,))
    rows = cur.fetchall()
    conn.close()

    tags = [{"tag": r["tag"], "score": r["score"]} for r in rows]
    return jsonify({"photo_id": photo_id, "tags": tags})


@app.post("/api/photos/<int:photo_id>/tags")
def add_photo_tags(photo_id: int):
    data = request.get_json(force=True, silent=True) or {}
    tags = data.get("tags")
    if not isinstance(tags, list) or not tags:
        return jsonify({"error": "invalid_tags"}), 400

    conn = get_conn()
    for item in tags:
        if isinstance(item, str):
            tag = item
            score = 1.0
        elif isinstance(item, dict):
            tag = item.get("tag")
            score = float(item.get("score", 1.0))
        else:
            continue

        if not tag:
            continue
        db.add_tag(conn, photo_id, tag, score)

    conn.close()
    return jsonify({"status": "ok"})


# -----------------------------------------------------------------------------
# AUTO TAG + EMBEDDINGS
# -----------------------------------------------------------------------------
@app.post("/api/index/embeddings")
def index_embeddings():
    data = request.get_json(force=True, silent=True) or {}
    incremental = bool(data.get("incremental", True))

    conn = get_conn()
    em = get_embedder()
    cur = conn.cursor()

    if incremental:
        cur.execute(
            """
            SELECT p.id, p.file_path
            FROM photos p
            LEFT JOIN embeddings e ON e.photo_id = p.id
            WHERE e.photo_id IS NULL
            ORDER BY p.id ASC
            """
        )
    else:
        cur.execute("SELECT id, file_path FROM photos ORDER BY id ASC")

    rows = cur.fetchall()
    if not rows:
        conn.close()
        return jsonify({"indexed": 0})

    indexed = 0

    for r in rows:
        pid = r["id"]
        file_path = r["file_path"]
        try:
            img_vec = em.image_embedding(file_path)
        except Exception:
            continue

        vec_bytes = img_vec.tobytes()
        dim = img_vec.shape[0]
        db.put_embedding(conn, pid, vec_bytes, dim)

        # auto tags
        atags = auto_tags(em, file_path, k=6)
        for tag, score in atags:
            db.add_tag(conn, pid, tag, float(score))

        indexed += 1

    conn.close()
    return jsonify({"indexed": indexed})


# -----------------------------------------------------------------------------
# SEARCH (TEXT → PHOTOS)
# -----------------------------------------------------------------------------
@app.get("/api/search/photos")
def search_photos():
    query = request.args.get("q") or request.args.get("query")
    if not query:
        return jsonify({"error": "missing_query"}), 400

    top_k = int(request.args.get("limit", 20))

    conn = get_conn()
    em = get_embedder()

    q = em.text_embedding(query)  # 1D float32
    q = q / (np.linalg.norm(q) + 1e-9)

    rows = db.get_embeddings(conn)
    if not rows:
        conn.close()
        return jsonify({"results": []})

    photo_ids: List[int] = []
    vecs: List[np.ndarray] = []

    for photo_id, dim, vec_bytes in rows:
        v = np.frombuffer(vec_bytes, dtype="float32")
        if v.shape[0] != dim:
            continue
        photo_ids.append(photo_id)
        vecs.append(v)

    if not vecs:
        conn.close()
        return jsonify({"results": []})

    mat = np.stack(vecs, axis=0)
    mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)

    sims = mat @ q
    idx = np.argsort(-sims)[:top_k]

    cur = conn.cursor()
    results = []
    for i in idx:
        pid = photo_ids[int(i)]
        score = float(sims[int(i)])
        cur.execute("SELECT * FROM photos WHERE id=?", (pid,))
        row = cur.fetchone()
        if not row:
            continue
        dto = row_to_photo_dto(row)
        dto["score"] = score
        results.append(dto)

    conn.close()
    return jsonify({"query": query, "results": results})


# -----------------------------------------------------------------------------
# FACES
# -----------------------------------------------------------------------------
@app.post("/api/index/faces")
def index_faces():
    data = request.get_json(force=True, silent=True) or {}
    limit = data.get("limit")
    min_score = float(data.get("min_score", 0.5))
    thumb_size = int(data.get("thumb_size", 160))

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, file_path FROM photos ORDER BY id ASC")
    rows = cur.fetchall()

    processed = 0
    for r in rows:
        pid = r["id"]
        file_path = r["file_path"]

        faces = face_encodings(file_path)
        if not faces:
            continue

        img = load_image(Path(file_path))

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
            cur2 = conn.cursor()
            cur2.execute(
                "SELECT id FROM faces WHERE photo_id=? AND face_index=?",
                (pid, idx_face),
            )
            row = cur2.fetchone()
            if not row:
                continue
            face_id = row["id"]

            # Thumbnail
            crop = img.crop((x, y, x + w, y + h))
            crop = crop.resize((thumb_size, thumb_size))
            FACE_THUMB_ROOT.mkdir(parents=True, exist_ok=True)
            face_thumb_path = FACE_THUMB_ROOT / f"face_{face_id}.jpg"
            crop.save(face_thumb_path, "JPEG", quality=90)
            db.set_face_thumb(conn, face_id, str(face_thumb_path))

        processed += 1
        if limit is not None and processed >= int(limit):
            break

    conn.close()
    return jsonify({"processed_photos": processed})


@app.get("/api/faces")
def list_faces():
    conn = get_conn()
    rows = db.get_face_thumbs(conn)
    conn.close()

    items = []
    for r in rows:
        items.append(
            {
                "face_id": r["face_id"],
                "photo_id": r["photo_id"],
                "photo_path": r["file_path"],
                "thumb_path": r["thumb_path"],
                "person_id": r["person_id"],
                "person_name": r["person_name"],
            }
        )

    return jsonify({"items": items})


# -----------------------------------------------------------------------------
# PERSONS
# -----------------------------------------------------------------------------
@app.get("/api/persons")
def list_persons():
    conn = get_conn()
    rows = db.list_persons(conn)
    conn.close()

    persons = [{"id": r["id"], "name": r["name"]} for r in rows]
    return jsonify({"persons": persons})


@app.post("/api/persons")
def create_person():
    data = request.get_json(force=True, silent=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "missing_name"}), 400

    conn = get_conn()
    person_id = db.create_person(conn, name)
    conn.close()
    return jsonify({"id": person_id, "name": name})


@app.post("/api/faces/<int:face_id>/assign-person")
def assign_face_person(face_id: int):
    data = request.get_json(force=True, silent=True) or {}
    person_id = data.get("person_id")
    if not isinstance(person_id, int):
        return jsonify({"error": "invalid_person_id"}), 400

    conn = get_conn()
    db.assign_person(conn, face_id, person_id)
    conn.close()
    return jsonify({"status": "ok"})


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Simple dev server
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
