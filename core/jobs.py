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
from core.embedder import ClipEmbedder
from core.extractor import load_image
from core.face import face_encodings
from core.storage_client import MinIOStorageClient
from core.gallery import ensure_thumb
from core.tagger import auto_tags


# Global instances (will be initialized per worker)
_STORAGE_CLIENT = None
_EMBEDDER = None


def _auto_match_face_to_person(conn, face_id: int, face_embedding: np.ndarray, threshold: float = 0.75):
    """
    Automatically match a newly detected face to an existing person.
    
    Args:
        conn: Database connection
        face_id: ID of the face to match
        face_embedding: Face embedding vector (numpy array)
        threshold: Similarity threshold for matching (default 0.75)
    """
    try:
        import faiss
    except ImportError:
        # FAISS not available, skip matching
        return
    
    # Get all existing person faces with embeddings
    cur = conn.cursor()
    cur.execute("""
        SELECT f.id, f.embedding, f.person_id, p.name as person_name
        FROM faces f
        JOIN persons p ON f.person_id = p.id
        WHERE f.embedding IS NOT NULL AND f.id != ?
    """, (face_id,))
    
    existing_faces = cur.fetchall()
    
    if not existing_faces:
        # No existing faces to match against
        return
    
    # Build vectors for comparison
    existing_vecs = []
    existing_face_ids = []
    existing_person_map = {}  # face_id -> (person_id, person_name)
    
    for row in existing_faces:
        emb_bytes = row[1]  # embedding column
        if not emb_bytes:
            continue
        v = np.frombuffer(emb_bytes, dtype=np.float32)
        if v.size == 0:
            continue
        
        existing_vecs.append(v)
        existing_face_ids.append(row[0])  # face id
        existing_person_map[row[0]] = (row[2], row[3])  # (person_id, person_name)
    
    if not existing_vecs:
        return
    
    # Normalize and compare
    face_vec = face_embedding.astype("float32")
    existing_xb = np.stack(existing_vecs).astype("float32")
    
    # Normalize for cosine similarity
    faiss.normalize_L2(face_vec.reshape(1, -1))
    faiss.normalize_L2(existing_xb)
    
    # Build index and search
    try:
        index = faiss.IndexFlatIP(existing_xb.shape[1])
        index.add(existing_xb)
        
        # Search for best match
        distances, indices = index.search(face_vec.reshape(1, -1), 1)
        
        if distances[0, 0] >= threshold:
            # Match found! Assign to existing person
            matched_face_idx = indices[0, 0]
            matched_face_id = existing_face_ids[matched_face_idx]
            person_id, person_name = existing_person_map[matched_face_id]
            
            # Assign face to person
            cur.execute("UPDATE faces SET person_id=? WHERE id=?", (person_id, face_id))
            conn.commit()
            
            print(f"Auto-matched face {face_id} to existing person '{person_name}' (similarity: {distances[0, 0]:.3f})")
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
    """Get or create embedder instance."""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = ClipEmbedder()
    return _EMBEDDER


def process_photo_embedding_job(photo_id: int, file_path: str, db_path: str):
    """
    Background job to process photo embedding and auto-tags.
    
    Args:
        photo_id: Photo ID in database
        file_path: MinIO object key to the photo file
        db_path: Path to SQLite database
    """
    conn = db.connect(db_path)
    storage_client = _get_storage_client()
    
    try:
        # Download from MinIO to temporary file
        file_data = storage_client.download_file(file_path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file_path).suffix) as tmp:
            tmp.write(file_data)
            tmp_path = tmp.name
        
        try:
            em = _get_embedder()
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
    conn = db.connect(db_path)
    storage_client = _get_storage_client()
    
    try:
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
                        crop.save(thumb_tmp.name, "JPEG", quality=90)
                        with open(thumb_tmp.name, 'rb') as f:
                            thumb_data = f.read()
                    
                    # Upload to MinIO
                    face_thumb_path = storage_client.generate_thumbnail_path(face_id, "face")
                    storage_client.upload_file(thumb_data, face_thumb_path)
                    db.set_face_thumb(conn, face_id, face_thumb_path)
                    
                    # Auto-match face to existing persons
                    try:
                        _auto_match_face_to_person(conn, face_id, emb, threshold=0.75)
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
    """Process embedding for a single photo (used in parallel processing)."""
    conn = db.connect(db_path)
    storage_client = _get_storage_client()
    em = _get_embedder()
    
    try:
        # Download from MinIO to temporary file
        file_data = storage_client.download_file(file_path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file_path).suffix) as tmp:
            tmp.write(file_data)
            tmp_path = tmp.name
        
        try:
            img_vec = em.image_embedding(tmp_path)
            vec_bytes = img_vec.tobytes()
            dim = img_vec.shape[0]
            db.put_embedding(conn, pid, vec_bytes, dim)
            
            # Auto tags
            atags = auto_tags(em, tmp_path, k=6)
            for tag, score in atags:
                db.add_tag(conn, pid, tag, float(score))
            return True
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        print(f"Error processing embedding for photo {pid}: {e}")
        return False
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
    conn = db.connect(db_path)
    storage_client = _get_storage_client()
    
    try:
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
                        crop.save(thumb_tmp.name, "JPEG", quality=90)
                        with open(thumb_tmp.name, 'rb') as f:
                            thumb_data = f.read()
                    
                    # Upload to MinIO
                    face_thumb_path = storage_client.generate_thumbnail_path(face_id, "face")
                    storage_client.upload_file(thumb_data, face_thumb_path)
                    db.set_face_thumb(conn, face_id, face_thumb_path)
                    
                    # Auto-match face to existing persons
                    try:
                        _auto_match_face_to_person(conn, face_id, emb, threshold=0.75)
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

