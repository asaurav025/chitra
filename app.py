
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

from core import db
from core.extractor import collect_metadata, load_image, iter_images, RAW_EXTS
from core.ftp_client import FTPStorageClient
from PIL import Image
# Register HEIC/HEIF support
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass  # pillow-heif not installed, HEIC support will be limited
from core.embedder import ClipEmbedder
from core.gallery import ensure_thumb
from core.face import face_encodings
from core.tagger import auto_tags
from core.worker import get_queue
from core.jobs import (
    process_photo_embedding_job,
    process_photo_faces_job,
    index_embeddings_batch_job,
    index_faces_batch_job,
)
from core.cache import get_cached_thumbnail, cache_thumbnail
from rq import Queue
from rq.job import Job
import tempfile


# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
DB_PATH = os.environ.get("CHITRA_DB_PATH", db.DB_DEFAULT_PATH)

# FTP storage is mandatory - no local storage fallback
FTP_STORAGE_HOST = os.environ.get("FTP_STORAGE_HOST")
if not FTP_STORAGE_HOST:
    raise ValueError(
        "FTP_STORAGE_HOST environment variable is required. "
        "Please configure FTP storage: export FTP_STORAGE_HOST=ftp.example.com"
    )

# Initialize FTP storage client
FTP_CLIENT = FTPStorageClient()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

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


def ensure_photo_thumb(file_path: str, photo_id: int) -> str:
    """
    Ensure thumbnail exists on FTP server. Returns FTP path.
    """
    thumb_path = FTP_CLIENT.generate_thumbnail_path(photo_id, "photo")
    # Check if thumbnail exists on FTP
    if not FTP_CLIENT.file_exists(thumb_path):
        # Download original, generate thumb, upload
        tmp_path = None
        thumb_file_path = None
        try:
            file_data = FTP_CLIENT.download_file(file_path)
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file_path).suffix) as tmp:
                tmp.write(file_data)
                tmp_path = tmp.name
            
            # Generate thumbnail
            thumb_file_path = str(Path(tmp_path).with_suffix('.jpg'))
            try:
                ensure_thumb(tmp_path, thumb_file_path)
            except Exception as thumb_err:
                raise Exception(f"Thumbnail generation failed: {str(thumb_err)}")
            
            # Check if thumbnail was actually created
            if not os.path.exists(thumb_file_path):
                raise Exception(f"Thumbnail generation failed: {thumb_file_path} was not created (ensure_thumb returned without error but file missing)")
            
            # Read thumbnail and upload
            with open(thumb_file_path, 'rb') as f:
                thumb_data = f.read()
            FTP_CLIENT.upload_file(thumb_data, thumb_path)
        except Exception as e:
            raise Exception(f"Failed to create thumbnail: {str(e)}")
        finally:
            # Cleanup - only delete files that exist
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass
            if thumb_file_path and os.path.exists(thumb_file_path):
                try:
                    os.unlink(thumb_file_path)
                except:
                    pass
    return thumb_path


# -----------------------------------------------------------------------------
# LIFECYCLE
# -----------------------------------------------------------------------------
@app.before_request
def init_db_once():
    if not hasattr(app, "_db_initialized"):
        db.init_db(DB_PATH)
        app._db_initialized = True


# -----------------------------------------------------------------------------
# BASIC
# -----------------------------------------------------------------------------
@app.get("/api/health")
def health():
    conn = get_conn()
    try:
        db.init_db(DB_PATH)
    except:
        pass
    
    health_info = {
        "status": "ok",
        "db_path": str(DB_PATH),
        "storage_type": "ftp",
        "ftp_host": FTP_CLIENT.host,
        "ftp_base_path": FTP_CLIENT.base_path,
    }
    
    # Test FTP connection
    try:
        FTP_CLIENT._connect()
        health_info["ftp_status"] = "connected"
        FTP_CLIENT._disconnect()
    except Exception as e:
        health_info["ftp_status"] = f"error: {str(e)}"
    
    return jsonify(health_info)


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
    
    # Download from FTP
    try:
        file_data = FTP_CLIENT.download_file(file_path)
    except FileNotFoundError:
        return jsonify({"error": "file_not_found_on_ftp"}), 404
    except Exception as e:
        return jsonify({"error": f"ftp_error: {str(e)}"}), 500
    
    # Check if file needs conversion (RAW or HEIC/HEIF)
    file_ext = Path(file_path).suffix.lower()
    heic_exts = {".heic", ".heif"}
    
    if file_ext in RAW_EXTS or file_ext in heic_exts:
        # Process in temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(file_data)
            tmp_path = tmp.name
        
        try:
            img = load_image(Path(tmp_path))
            # Convert PIL Image to JPEG bytes
            img_io = io.BytesIO()
            img.save(img_io, format='JPEG', quality=95)
            img_io.seek(0)
            response = send_file(img_io, mimetype='image/jpeg')
            response.cache_control.public = True
            response.cache_control.max_age = 86400  # 24 hours
            return response
        except Exception as e:
            return jsonify({"error": f"failed_to_convert: {str(e)}"}), 500
        finally:
            os.unlink(tmp_path)
    
    # Serve directly
    ext = Path(file_path).suffix.lower()
    mimetype = {
        '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
        '.png': 'image/png', '.gif': 'image/gif',
        '.webp': 'image/webp',
    }.get(ext, 'application/octet-stream')
    
    response = send_file(io.BytesIO(file_data), mimetype=mimetype)
    response.cache_control.public = True
    response.cache_control.max_age = 86400  # 24 hours
    return response


@app.get("/api/photos/<int:photo_id>/thumbnail")
def get_photo_thumbnail(photo_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT file_path, thumb_path FROM photos WHERE id=?", (photo_id,))
    row = cur.fetchone()
    
    if not row:
        conn.close()
        return jsonify({"error": "photo_not_found"}), 404

    file_path = row["file_path"]
    thumb_path = row["thumb_path"]
    
    # If thumb_path not in DB, generate it and store
    if not thumb_path:
        conn.close()
        thumb_path = ensure_photo_thumb(file_path, photo_id)
        # Re-open connection to update
        conn = get_conn()
        conn.execute("UPDATE photos SET thumb_path = ? WHERE id = ?", (thumb_path, photo_id))
        conn.commit()
        conn.close()
    else:
        conn.close()
        # Check if thumbnail exists on FTP (only if we have path in DB)
        if not FTP_CLIENT.file_exists(thumb_path):
            # Thumbnail was deleted or doesn't exist, regenerate
            thumb_path = ensure_photo_thumb(file_path, photo_id)
            conn = get_conn()
            conn.execute("UPDATE photos SET thumb_path = ? WHERE id = ?", (thumb_path, photo_id))
            conn.commit()
            conn.close()
    
    # Check cache first
    thumb_data = get_cached_thumbnail(thumb_path)
    
    if thumb_data is None:
        # Download from FTP
        try:
            thumb_data = FTP_CLIENT.download_file(thumb_path)
            # Cache for future requests
            cache_thumbnail(thumb_path, thumb_data)
        except FileNotFoundError:
            return jsonify({"error": "thumbnail_not_found_on_ftp"}), 404
        except Exception as e:
            return jsonify({"error": f"ftp_error: {str(e)}"}), 500
    
    response = send_file(io.BytesIO(thumb_data), mimetype='image/jpeg')
    # Add HTTP caching headers
    response.cache_control.public = True
    response.cache_control.max_age = 86400  # 24 hours
    return response


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

    # Read all file data upfront (file objects may not be thread-safe)
    file_data_list = []
    for storage in files:
        filename = storage.filename
        if not filename:
            continue
        file_data = storage.read()
        file_data_list.append((filename, file_data))
    
    if not file_data_list:
        return jsonify({"error": "no_valid_files"}), 400

    # Check auto_process setting
    auto_process = request.args.get("auto_process", "true").lower() == "true"
    
    def upload_single_file(filename: str, file_data: bytes) -> Dict[str, Any]:
        """
        Upload a single file - runs in parallel with other uploads.
        Each thread gets its own database connection.
        """
        conn = None
        tmp_path = None
        try:
            conn = get_conn()
            
            # Upload to FTP storage
            remote_path = FTP_CLIENT.generate_photo_path(filename)
            
            # Ensure uniqueness - check database first (much faster than FTP)
            counter = 1
            base_path = remote_path
            cur = conn.cursor()
            while True:
                cur.execute("SELECT id FROM photos WHERE file_path=?", (remote_path,))
                if cur.fetchone() is None:
                    # Not in database, check FTP only if needed (for safety)
                    if not FTP_CLIENT.file_exists(remote_path):
                        break
                # Path exists, generate new one
                path_obj = Path(base_path)
                ext = path_obj.suffix
                name = path_obj.stem
                remote_path = f"{path_obj.parent}/{name}_{counter}{ext}"
                counter += 1
            
            # Upload to FTP (connection pooling handles concurrency)
            stored_path = FTP_CLIENT.upload_file(file_data, remote_path)
            
            # Process metadata using temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
                tmp.write(file_data)
                tmp_path = tmp.name
            
            # Collect metadata
            meta = collect_metadata(Path(tmp_path))
            # Store FTP path
            meta['file_path'] = stored_path
            db.upsert_photo(conn, **meta)
            
            # Get photo ID
            cur = conn.cursor()
            cur.execute("SELECT id FROM photos WHERE file_path=?", (stored_path,))
            row = cur.fetchone()
            if not row:
                return {"error": f"Failed to get photo ID for {filename}"}
            photo_id = row["id"]
            
            # Generate thumbnail
            thumb_path = ensure_photo_thumb(stored_path, photo_id)
            
            # Store thumb_path in database
            conn.execute("UPDATE photos SET thumb_path = ? WHERE id = ?", (thumb_path, photo_id))
            conn.commit()
            
            # Auto-process: embeddings and faces (if enabled)
            # Queue background jobs instead of processing synchronously
            if auto_process:
                try:
                    queue = get_queue()
                    # Queue embedding processing job
                    queue.enqueue(
                        process_photo_embedding_job,
                        photo_id,
                        stored_path,
                        DB_PATH,
                        job_timeout='10m'  # 10 minute timeout
                    )
                    # Queue face processing job
                    queue.enqueue(
                        process_photo_faces_job,
                        photo_id,
                        stored_path,
                        DB_PATH,
                        job_timeout='10m'  # 10 minute timeout
                    )
                except Exception as e:
                    # Don't fail upload if job queuing fails
                    print(f"Warning: Failed to queue processing jobs for {filename}: {e}")
            
            return {
                "id": photo_id,
                "file_path": stored_path,
                "storage_url": f"/api/storage/{stored_path}",
                "thumbnail": thumb_path,
                "thumbnail_url": f"/api/storage/{thumb_path}",
            }
        except Exception as e:
            import traceback
            error_msg = f"Upload failed for {filename}: {str(e)}"
            print(f"Error in upload_single_file: {error_msg}")
            print(traceback.format_exc())
            return {"error": error_msg}
        finally:
            # Clean up temporary file
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass
            # Close database connection
            if conn:
                try:
                    conn.close()
                except:
                    pass
    
    # Upload files in parallel (max 5 concurrent uploads)
    # This significantly speeds up multiple file uploads
    saved = []
    errors = []
    
    # For single file, process directly (no threading overhead)
    if len(file_data_list) == 1:
        filename, file_data = file_data_list[0]
        try:
            result = upload_single_file(filename, file_data)
            if "error" in result:
                return jsonify({"error": result["error"]}), 500
            saved.append(result)
        except Exception as e:
            return jsonify({"error": f"Exception uploading {filename}: {str(e)}"}), 500
    else:
        # Multiple files: use parallel processing
        max_workers = min(5, len(file_data_list))  # Max 5 concurrent uploads
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all upload tasks
            future_to_file = {
                executor.submit(upload_single_file, filename, file_data): filename
                for filename, file_data in file_data_list
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_file):
                filename = future_to_file[future]
                try:
                    result = future.result()
                    if "error" in result:
                        errors.append(result["error"])
                    else:
                        saved.append(result)
                except Exception as e:
                    errors.append(f"Exception uploading {filename}: {str(e)}")
    
    # Return results
    response = {"saved": saved}
    if errors:
        response["errors"] = errors
    
    return jsonify(response)


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
# -----------------------------------------------------------------------------
# HELPER FUNCTIONS FOR AUTO-PROCESSING
# -----------------------------------------------------------------------------

def process_photo_embedding(photo_id: int, file_path: str, conn):
    """Process a single photo: generate embedding and auto-tags."""
    try:
        # Download from FTP to temporary file
        file_data = FTP_CLIENT.download_file(file_path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file_path).suffix) as tmp:
            tmp.write(file_data)
            tmp_path = tmp.name
        try:
            em = get_embedder()
            img_vec = em.image_embedding(tmp_path)
            vec_bytes = img_vec.tobytes()
            dim = img_vec.shape[0]
            db.put_embedding(conn, photo_id, vec_bytes, dim)
            
            # Auto tags
            atags = auto_tags(em, tmp_path, k=6)
            for tag, score in atags:
                db.add_tag(conn, photo_id, tag, float(score))
            return True
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        print(f"Error processing embedding for photo {photo_id}: {e}")
        return False


def process_photo_faces(photo_id: int, file_path: str, conn, min_score=0.5, thumb_size=160):
    """Process a single photo: detect faces and generate thumbnails."""
    try:
        # Download from FTP to temporary file
        file_data = FTP_CLIENT.download_file(file_path)
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
                    
                    # Save to temporary file, then upload to FTP
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as thumb_tmp:
                        crop.save(thumb_tmp.name, "JPEG", quality=90)
                        with open(thumb_tmp.name, 'rb') as f:
                            thumb_data = f.read()
                    
                    # Upload to FTP
                    face_thumb_path = FTP_CLIENT.generate_thumbnail_path(face_id, "face")
                    FTP_CLIENT.upload_file(thumb_data, face_thumb_path)
                    db.set_face_thumb(conn, face_id, face_thumb_path)
                    
                    # Cleanup
                    os.unlink(thumb_tmp.name)
                    face_count += 1
            
            return face_count
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        print(f"Error processing faces for photo {photo_id}: {e}")
        return 0

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
    conn.close()
    
    if not rows:
        return jsonify({"indexed": 0, "job_id": None, "message": "No photos to process"})

    # Prepare photo list for batch job
    photo_ids_and_paths = [(r["id"], r["file_path"]) for r in rows]
    
    # Queue background job
    try:
        queue = get_queue()
        job = queue.enqueue(
            index_embeddings_batch_job,
            photo_ids_and_paths,
            DB_PATH,
            incremental,
            job_timeout='1h'  # 1 hour timeout for batch processing
        )
        return jsonify({
            "indexed": 0,  # Will be updated when job completes
            "job_id": job.id,
            "message": f"Queued {len(photo_ids_and_paths)} photos for processing",
            "status": "queued"
        })
    except Exception as e:
        return jsonify({"error": f"Failed to queue job: {str(e)}"}), 500


@app.get("/api/jobs/<job_id>")
def get_job_status(job_id: str):
    """Get status of a background job."""
    try:
        queue = get_queue()
        job = Job.fetch(job_id, connection=queue.connection)
        
        status_info = {
            "job_id": job.id,
            "status": job.get_status(),
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "ended_at": job.ended_at.isoformat() if job.ended_at else None,
        }
        
        if job.is_finished:
            status_info["result"] = job.result
        elif job.is_failed:
            status_info["error"] = str(job.exc_info) if job.exc_info else "Job failed"
        
        return jsonify(status_info)
    except Exception as e:
        return jsonify({"error": f"Job not found: {str(e)}"}), 404


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
    conn.close()
    
    if not rows:
        return jsonify({"processed_photos": 0, "job_id": None, "message": "No photos to process"})
    
    # Apply limit if specified
    photo_ids_and_paths = [(r["id"], r["file_path"]) for r in rows]
    if limit is not None:
        photo_ids_and_paths = photo_ids_and_paths[:int(limit)]
    
    # Queue background job
    try:
        queue = get_queue()
        job = queue.enqueue(
            index_faces_batch_job,
            photo_ids_and_paths,
            DB_PATH,
            min_score,
            thumb_size,
            job_timeout='1h'  # 1 hour timeout for batch processing
        )
        return jsonify({
            "processed_photos": 0,  # Will be updated when job completes
            "job_id": job.id,
            "message": f"Queued {len(photo_ids_and_paths)} photos for face processing",
            "status": "queued"
        })
    except Exception as e:
        return jsonify({"error": f"Failed to queue job: {str(e)}"}), 500


@app.post("/api/index/faces-cluster")
def cluster_faces():
    """
    Cluster faces into persons using FAISS on face embeddings.
    Creates automatic person names like 'Person 1', 'Person 2', etc.
    """
    import faiss

    data = request.get_json(force=True, silent=True) or {}
    threshold = float(data.get("threshold", 0.75))

    conn = get_conn()
    rows = db.get_faces_embeddings(conn)

    if not rows:
        conn.close()
        return jsonify({"error": "no_faces_found", "message": "No faces found. Run face indexing first."}), 400

    face_ids = []
    vecs = []

    for row in rows:
        fid = row["id"]
        emb_bytes = row["embedding"]
        v = np.frombuffer(emb_bytes, dtype=np.float32)
        if v.size == 0:
            continue
        face_ids.append(fid)
        vecs.append(v)

    if not vecs:
        conn.close()
        return jsonify({"error": "no_valid_embeddings"}), 400

    xb = np.stack(vecs).astype("float32")
    faiss.normalize_L2(xb)
    dim = xb.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(xb)

    # For each face, find neighbors and cluster via a simple union-find
    n = xb.shape[0]
    k = min(10, n)  # check top-10 nearest faces per face

    # Union-Find
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    D, I = index.search(xb, k)
    for i in range(n):
        for j in range(1, k):  # skip self at j=0
            if I[i, j] < 0:
                continue
            if D[i, j] >= threshold:
                union(i, I[i, j])

    # Collect clusters
    clusters: Dict[int, List[int]] = {}
    for i in range(n):
        root = find(i)
        clusters.setdefault(root, []).append(i)

    # Assign persons
    person_idx = 1
    for root, members in clusters.items():
        # create person name
        person_name = f"Person {person_idx}"
        person_id = db.get_or_create_person(conn, person_name)
        person_idx += 1

        for m in members:
            fid = face_ids[m]
            db.set_face_person(conn, fid, person_id)

    conn.close()

    return jsonify({
        "clustered_faces": len(face_ids),
        "persons_created": len(clusters),
        "threshold": threshold,
    })


@app.get("/api/faces/<int:face_id>/thumbnail")
def get_face_thumbnail(face_id: int):
    """Serve face thumbnail image."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT ft.thumb_path
        FROM face_thumbs ft
        WHERE ft.face_id = ?
        """,
        (face_id,),
    )
    row = cur.fetchone()
    conn.close()

    if not row or not row["thumb_path"]:
        return jsonify({"error": "face_thumb_not_found"}), 404

    thumb_path = row["thumb_path"]
    
    # Check cache first
    thumb_data = get_cached_thumbnail(thumb_path)
    
    if thumb_data is None:
        # Download from FTP
        try:
            thumb_data = FTP_CLIENT.download_file(thumb_path)
            # Cache for future requests
            cache_thumbnail(thumb_path, thumb_data)
        except FileNotFoundError:
            return jsonify({"error": "thumb_not_found_on_ftp"}), 404
        except Exception as e:
            return jsonify({"error": f"ftp_error: {str(e)}"}), 500
    
    response = send_file(io.BytesIO(thumb_data), mimetype='image/jpeg')
    response.cache_control.public = True
    response.cache_control.max_age = 86400  # 24 hours
    return response


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


@app.put("/api/persons/<int:person_id>")
def update_person(person_id: int):
    """Update a person's name."""
    data = request.get_json(force=True, silent=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "missing_name"}), 400

    conn = get_conn()
    db.rename_person(conn, person_id, name)
    conn.close()
    return jsonify({"id": person_id, "name": name})


# -----------------------------------------------------------------------------
# SEARCH BY PERSON NAME
# -----------------------------------------------------------------------------
@app.get("/api/search/by-person")
def search_by_person():
    """Search photos containing a specific person."""
    person_name = request.args.get("name") or request.args.get("q")
    if not person_name:
        return jsonify({"error": "missing_person_name"}), 400

    conn = get_conn()
    cur = conn.cursor()
    
    # Find photos that have faces tagged with this person (case-insensitive partial match)
    cur.execute(
        """
        SELECT DISTINCT p.*
        FROM photos p
        JOIN faces f ON f.photo_id = p.id
        JOIN persons per ON per.id = f.person_id
        WHERE per.name LIKE ?
        ORDER BY p.id DESC
        """,
        (f"%{person_name}%",),
    )
    rows = cur.fetchall()
    conn.close()

    results = [row_to_photo_dto(r) for r in rows]
    return jsonify({"query": person_name, "results": results})


@app.post("/api/faces/<int:face_id>/assign-person")
def assign_face_person(face_id: int):
    data = request.get_json(force=True, silent=True) or {}
    person_id = data.get("person_id")
    if not isinstance(person_id, int):
        return jsonify({"error": "invalid_person_id"}), 400

    conn = get_conn()
    db.set_face_person(conn, face_id, person_id)
    conn.close()
    return jsonify({"status": "ok"})


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
@app.get("/api/persons/<int:person_id>/faces")
def get_person_faces(person_id: int):
    """Get face thumbnails for a specific person."""
    conn = get_conn()
    rows = db.get_face_thumbs_for_person(conn, person_id)
    conn.close()
    
    items = []
    for r in rows:
        items.append(
            {
                "face_id": r["face_id"],
                "photo_id": r["photo_id"],
                "thumb_path": r["thumb_path"],
            }
        )
    return jsonify({"items": items})


@app.post("/api/persons/<int:target_person_id>/merge")
def merge_persons_endpoint(target_person_id: int):
    """Merge source person into target person."""
    data = request.get_json(force=True, silent=True) or {}
    source_person_id = data.get("source_person_id")
    
    if not isinstance(source_person_id, int):
        return jsonify({"error": "missing_source_person_id"}), 400
    
    if source_person_id == target_person_id:
        return jsonify({"error": "cannot_merge_same_person"}), 400
    
    conn = get_conn()
    try:
        db.merge_persons(conn, source_person_id, target_person_id)
        conn.close()
        return jsonify({"status": "merged", "source_person_id": source_person_id, "target_person_id": target_person_id})
    except ValueError as e:
        conn.close()
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        conn.close()
        return jsonify({"error": str(e)}), 500


# -----------------------------------------------------------------------------
# GENERIC STORAGE ENDPOINT
# -----------------------------------------------------------------------------
@app.get("/api/storage/<path:file_path>")
def get_storage_file(file_path: str):
    """
    Generic endpoint to serve any file from FTP storage.
    This allows direct URL access to files stored on FTP server.
    """
    try:
        file_data = FTP_CLIENT.download_file(file_path)
        ext = Path(file_path).suffix.lower()
        mimetype = {
            '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
            '.png': 'image/png', '.gif': 'image/gif',
            '.webp': 'image/webp', '.bmp': 'image/bmp',
            '.tif': 'image/tiff', '.tiff': 'image/tiff',
        }.get(ext, 'application/octet-stream')
        
        return send_file(io.BytesIO(file_data), mimetype=mimetype)
    except FileNotFoundError:
        return jsonify({"error": "file_not_found"}), 404
    except Exception as e:
        return jsonify({"error": f"ftp_error: {str(e)}"}), 500


# -----------------------------------------------------------------------------
# FRONTEND STATIC FILES
# -----------------------------------------------------------------------------
UI_DIST_PATH = Path(__file__).parent.parent / "chitra_ui" / "dist"

@app.route("/", defaults={"path": ""})

def serve_frontend(path):
    """Serve the frontend application. All non-API routes serve index.html for client-side routing."""
    # Don't serve frontend for API routes
    if path.startswith("api/"):
        return jsonify({"error": "not_found"}), 404
    
    # Serve static assets
    if path.startswith("assets/"):
        asset_path = UI_DIST_PATH / path
        if asset_path.exists() and asset_path.is_file():
            return send_file(asset_path)
        return jsonify({"error": "not_found"}), 404
    
    # Serve index.html for all other routes (SPA routing)
    index_path = UI_DIST_PATH / "index.html"
    if index_path.exists():
        return send_file(index_path)
    
    return jsonify({"error": "frontend_not_built", "message": "Please build the frontend first: cd chitra_ui && npm run build"}), 503
 

if __name__ == "__main__":
    # Simple dev server
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        debug=False,
        use_reloader=False
    )

