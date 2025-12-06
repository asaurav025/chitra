"""
Background job functions for RQ (Redis Queue).
These functions are executed by worker processes.
"""
import os
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from core import db
from core.embedder import ClipEmbedder
from core.extractor import load_image
from core.face import face_encodings
from core.ftp_client import FTPStorageClient
from core.gallery import ensure_thumb
from core.tagger import auto_tags


# Global instances (will be initialized per worker)
_FTP_CLIENT = None
_EMBEDDER = None


def _get_ftp_client():
    """Get or create FTP client instance."""
    global _FTP_CLIENT
    if _FTP_CLIENT is None:
        _FTP_CLIENT = FTPStorageClient()
    return _FTP_CLIENT


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
        file_path: FTP path to the photo file
        db_path: Path to SQLite database
    """
    conn = db.connect(db_path)
    ftp_client = _get_ftp_client()
    
    try:
        # Download from FTP to temporary file
        file_data = ftp_client.download_file(file_path)
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
        file_path: FTP path to the photo file
        db_path: Path to SQLite database
        min_score: Minimum face detection score
        thumb_size: Thumbnail size
    """
    conn = db.connect(db_path)
    ftp_client = _get_ftp_client()
    
    try:
        # Download from FTP to temporary file
        file_data = ftp_client.download_file(file_path)
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
                    face_thumb_path = ftp_client.generate_thumbnail_path(face_id, "face")
                    ftp_client.upload_file(thumb_data, face_thumb_path)
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
    finally:
        conn.close()


def _process_single_embedding(pid: int, file_path: str, db_path: str) -> bool:
    """Process embedding for a single photo (used in parallel processing)."""
    conn = db.connect(db_path)
    ftp_client = _get_ftp_client()
    em = _get_embedder()
    
    try:
        # Download from FTP to temporary file
        file_data = ftp_client.download_file(file_path)
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
    ftp_client = _get_ftp_client()
    
    try:
        # Download from FTP to temporary file
        file_data = ftp_client.download_file(file_path)
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
                    
                    # Save to temporary file, then upload to FTP
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as thumb_tmp:
                        crop.save(thumb_tmp.name, "JPEG", quality=90)
                        with open(thumb_tmp.name, 'rb') as f:
                            thumb_data = f.read()
                    
                    # Upload to FTP
                    face_thumb_path = ftp_client.generate_thumbnail_path(face_id, "face")
                    ftp_client.upload_file(thumb_data, face_thumb_path)
                    db.set_face_thumb(conn, face_id, face_thumb_path)
                    
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

