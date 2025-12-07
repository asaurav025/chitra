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
                        crop.save(thumb_tmp.name, "JPEG", quality=100)
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
                        crop.save(thumb_tmp.name, "JPEG", quality=100)
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



def cluster_faces_job(db_path: str, threshold: float = 0.75, photo_ids: list = None, reset: bool = False):
    """
    Background job to cluster unassigned faces into persons using HDBSCAN.
    
    This job runs after face detection jobs complete to automatically
    group similar faces into persons.
    
    Args:
        db_path: Path to SQLite database
        threshold: Similarity threshold for clustering (default 0.75)
        photo_ids: Optional list of photo IDs to limit clustering to faces from these photos only.
                   If None, clusters all unassigned faces (less efficient for large databases).
        reset: If True, unassign all faces before clustering (re-cluster everything).
    """
    import faiss
    from core.faiss_index import FAISSIndexManager
    
    conn = db.connect(db_path)
    index_manager = FAISSIndexManager()
    
    try:
        # If reset=True, unassign all faces first (re-cluster everything)
        if reset:
            cur = conn.cursor()
            cur.execute("UPDATE faces SET person_id = NULL WHERE person_id IS NOT NULL")
            conn.commit()
            print(f"✓ Reset all face assignments - re-clustering all {cur.rowcount} faces")
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
            return {"clustered": 0, "persons_created": 0}
        
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
            return {"clustered": 0, "persons_created": 0}
        
        # Get existing person faces for matching
        cur.execute("""
            SELECT f.id, f.embedding, f.person_id, p.name as person_name
            FROM faces f
            JOIN persons p ON f.person_id = p.id
            WHERE f.embedding IS NOT NULL
        """)
        existing_person_faces = cur.fetchall()
        
        # Build index of existing person faces
        existing_person_vecs = []
        existing_person_face_ids = []
        existing_person_map = {}
        
        for row in existing_person_faces:
            fid, emb_bytes, person_id, person_name = row
            if not emb_bytes:
                continue
            v = np.frombuffer(emb_bytes, dtype=np.float32)
            if v.size == 0:
                continue
            existing_person_vecs.append(v)
            existing_person_face_ids.append(fid)
            existing_person_map[fid] = (person_id, person_name)
        
        # Phase 1: Match unassigned faces to existing persons
        matched_assignments = []
        unmatched_indices = []
        existing_index = None  # Initialize for Phase 2 access
        existing_index_name = "existing_person_faces"
        
        if existing_person_vecs:
            existing_xb = np.stack(existing_person_vecs).astype("float32")
            
            # Load or build index
            existing_index = index_manager.load_index(existing_index_name)
            index_needs_rebuild = False
            
            if existing_index is None:
                index_needs_rebuild = True
            else:
                try:
                    if hasattr(existing_index, 'ntotal') and existing_index.ntotal != len(existing_person_vecs):
                        index_needs_rebuild = True
                except Exception:
                    index_needs_rebuild = True
            
            if index_needs_rebuild:
                try:
                    existing_index = index_manager.build_hnsw_index(
                        existing_xb, existing_index_name, m=32, ef_construction=200
                    )
                except Exception as e:
                    print(f"Warning: HNSW build failed, using IndexFlatIP: {e}")
                    faiss.normalize_L2(existing_xb)
                    existing_dim = existing_xb.shape[1]
                    existing_index = faiss.IndexFlatIP(existing_dim)
                    existing_index.add(existing_xb)
            
            # Match unassigned faces to existing persons
            batch_xb = np.stack(unassigned_vecs).astype("float32")
            
            try:
                D_match, I_match = index_manager.search(existing_index, batch_xb, k=1, ef_search=50)
            except Exception as e:
                print(f"Warning: Search failed, using direct search: {e}")
                faiss.normalize_L2(batch_xb)
                D_match, I_match = existing_index.search(batch_xb, 1)
            
            # Convert L2 distances to cosine similarity if needed
            if D_match.max() > 1.0:
                # L2 distances - convert to cosine similarity
                max_l2 = D_match.max()
                if max_l2 > 2.0:
                    D_match = 1.0 - (D_match / 2.0)
                else:
                    D_match = 1.0 - ((D_match ** 2) / 2.0)
                D_match = np.clip(D_match, 0.0, 1.0)
            
            for i, face_id in enumerate(unassigned_face_ids):
                # Check if face is still unassigned
                cur.execute("SELECT person_id FROM faces WHERE id=?", (face_id,))
                face_row = cur.fetchone()
                if face_row and face_row[0] is not None:
                    continue
                
                similarity = float(D_match[i, 0]) if D_match[i, 0] > 0 else 0
                if similarity >= threshold:
                    matched_existing_face_idx = I_match[i, 0]
                    matched_existing_face_id = existing_person_face_ids[matched_existing_face_idx]
                    person_id, person_name = existing_person_map[matched_existing_face_id]
                    matched_assignments.append((face_id, person_id))
                else:
                    unmatched_indices.append(i)
        else:
            unmatched_indices = list(range(len(unassigned_face_ids)))
        
        # Batch update matched faces
        if matched_assignments:
            for face_id, person_id in matched_assignments:
                cur.execute("UPDATE faces SET person_id=? WHERE id=?", (person_id, face_id))
            conn.commit()
        matched_count = len(matched_assignments)
        
        # Update FAISS index with matched faces (Phase 1)
        if matched_assignments and existing_index is not None:
            try:
                # Get embeddings for matched faces
                matched_face_ids = [face_id for face_id, _ in matched_assignments]
                placeholders = ','.join(['?'] * len(matched_face_ids))
                cur.execute(f"""
                    SELECT embedding FROM faces WHERE id IN ({placeholders})
                """, matched_face_ids)
                matched_embeddings = cur.fetchall()
                
                if matched_embeddings:
                    # Prepare embeddings for index
                    matched_vecs = []
                    for (emb_bytes,) in matched_embeddings:
                        if emb_bytes:
                            v = np.frombuffer(emb_bytes, dtype=np.float32)
                            if v.size > 0:
                                matched_vecs.append(v)
                    
                    if matched_vecs:
                        matched_xb = np.stack(matched_vecs).astype("float32")
                        faiss.normalize_L2(matched_xb)
                        existing_index.add(matched_xb)
                        index_manager.save_index(existing_index, existing_index_name)
                        print(f"✓ Updated FAISS index: Added {len(matched_vecs)} matched faces to existing persons index")
            except Exception as e:
                print(f"Warning: Failed to update FAISS index with matched faces: {e}")
                # Continue - index will be rebuilt on next clustering
        
        # Phase 2: Cluster unmatched faces into new persons using HDBSCAN
        persons_created = 0
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
                'prediction_data': True  # Enable approximate_predict for future matching
            }
            
            # Only add epsilon if it's positive and reasonable
            if euclidean_threshold > 0:
                clusterer_kwargs['cluster_selection_epsilon'] = float(euclidean_threshold)
            
            clusterer = hdbscan.HDBSCAN(**clusterer_kwargs)
            
            # Perform clustering
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
                    
                    # Update FAISS index with newly assigned faces (Phase 2)
                    try:
                        # Get embeddings for cluster-assigned faces
                        cluster_face_ids = [face_id for face_id, _ in cluster_assignments]
                        placeholders = ','.join(['?'] * len(cluster_face_ids))
                        cur.execute(f"""
                            SELECT embedding FROM faces WHERE id IN ({placeholders})
                        """, cluster_face_ids)
                        cluster_embeddings = cur.fetchall()
                        
                        if cluster_embeddings:
                            # Prepare embeddings for index
                            cluster_vecs = []
                            for (emb_bytes,) in cluster_embeddings:
                                if emb_bytes:
                                    v = np.frombuffer(emb_bytes, dtype=np.float32)
                                    if v.size > 0:
                                        cluster_vecs.append(v)
                            
                            if cluster_vecs:
                                cluster_xb = np.stack(cluster_vecs).astype("float32")
                                faiss.normalize_L2(cluster_xb)
                                
                                # Add to existing index if it exists, otherwise create new one
                                if existing_index is not None:
                                    # Add to existing index
                                    existing_index.add(cluster_xb)
                                    index_manager.save_index(existing_index, existing_index_name)
                                    print(f"✓ Updated FAISS index: Added {len(cluster_vecs)} newly clustered faces to index")
                                else:
                                    # No existing index - create new one with these faces
                                    # This will be rebuilt properly on next clustering, but create it for now
                                    try:
                                        new_index = index_manager.build_hnsw_index(
                                            cluster_xb, existing_index_name, m=32, ef_construction=200
                                        )
                                        print(f"✓ Created new FAISS index with {len(cluster_vecs)} faces")
                                    except Exception as e:
                                        print(f"Warning: Could not create new index: {e}")
                                        # Index will be built on next clustering
                    except Exception as e:
                        print(f"Warning: Failed to update FAISS index with clustered faces: {e}")
                        # Continue - index will be rebuilt on next clustering
        
        total_clustered = matched_count + len(unmatched_indices) if unmatched_indices else matched_count
        print(f"✓ Clustering complete: {total_clustered} faces clustered, {persons_created} persons created")
        
        return {"clustered": total_clustered, "persons_created": persons_created}
        
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
        
        # Get all faces that were merged (now assigned to target_person_id)
        # Query faces that were recently merged - we can identify them by checking
        # if they exist in the target person (they were just merged)
        cur.execute(
            """
            SELECT id, embedding 
            FROM faces 
            WHERE person_id = ? AND embedding IS NOT NULL
            """,
            (target_person_id,)
        )
        merged_faces = cur.fetchall()
        
        if not merged_faces:
            print(f"ℹ No faces with embeddings found for person {target_person_id} after merge")
            return {"updated": 0, "message": "No faces to update"}
        
        # Load existing FAISS index
        existing_index_name = "existing_person_faces"
        existing_index = index_manager.load_index(existing_index_name)
        
        if existing_index is None:
            print(f"ℹ FAISS index not found - will be built on next clustering. Merged {len(merged_faces)} faces to person {target_person_id} ({target_person_name})")
            return {"updated": 0, "message": "Index not found, will be built on next clustering"}
        
        # Prepare embeddings for batch update
        merged_embeddings = []
        face_ids = []
        
        for face_row in merged_faces:
            face_id = face_row[0]  # id
            face_embedding_bytes = face_row[1]  # embedding
            
            if face_embedding_bytes:
                face_embedding = np.frombuffer(face_embedding_bytes, dtype=np.float32)
                face_embedding = face_embedding.reshape(1, -1).astype(np.float32)
                merged_embeddings.append(face_embedding)
                face_ids.append(face_id)
        
        if not merged_embeddings:
            print(f"ℹ No valid embeddings found for merged faces")
            return {"updated": 0, "message": "No valid embeddings"}
        
        # Stack all embeddings into a single array
        merged_xb = np.vstack(merged_embeddings)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(merged_xb)
        
        # Add to existing index
        existing_index.add(merged_xb)
        
        # Save updated index
        index_manager.save_index(existing_index, existing_index_name)
        
        print(f"✓ Updated FAISS index: Added {len(merged_embeddings)} merged faces from person {source_person_id} to person {target_person_id} ({target_person_name})")
        
        return {
            "updated": len(merged_embeddings),
            "message": f"Successfully updated FAISS index with {len(merged_embeddings)} face embeddings"
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
        
        # Get all faces assigned to persons with embeddings
        cur.execute(
            """
            SELECT f.id, f.embedding, f.person_id, p.name as person_name
            FROM faces f
            JOIN persons p ON f.person_id = p.id
            WHERE f.embedding IS NOT NULL
            ORDER BY f.person_id, f.id
            """
        )
        person_faces = cur.fetchall()
        
        if not person_faces:
            print("ℹ No faces with embeddings found for any person")
            # Delete existing index if it exists
            existing_index_name = "existing_person_faces"
            index_path = index_manager.index_dir / f"{existing_index_name}.index"
            if index_path.exists():
                index_path.unlink()
                print("✓ Deleted empty FAISS index")
            return {"updated": 0, "message": "No faces to index"}
        
        # Prepare embeddings
        embeddings = []
        face_ids = []
        person_ids = []
        
        for row in person_faces:
            face_id = row[0]  # id
            embedding_bytes = row[1]  # embedding
            person_id = row[2]  # person_id
            person_name = row[3]  # person_name
            
            if embedding_bytes:
                face_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                face_embedding = face_embedding.reshape(1, -1).astype(np.float32)
                embeddings.append(face_embedding)
                face_ids.append(face_id)
                person_ids.append(person_id)
        
        if not embeddings:
            print("ℹ No valid embeddings found")
            return {"updated": 0, "message": "No valid embeddings"}
        
        # Stack all embeddings into a single array
        xb = np.vstack(embeddings)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(xb)
        
        # Build new index
        existing_index_name = "existing_person_faces"
        dim = xb.shape[1]
        
        try:
            # Try to build HNSW index for faster search
            new_index = index_manager.build_hnsw_index(
                xb,
                existing_index_name,
                m=32,  # Connections per node
                ef_construction=200
            )
            print(f"✓ Rebuilt HNSW FAISS index with {len(embeddings)} faces from {len(set(person_ids))} persons")
        except Exception as e:
            # Fall back to IndexFlatIP if HNSW fails
            print(f"Warning: HNSW index build failed, using IndexFlatIP: {e}")
            faiss.normalize_L2(xb)
            new_index = faiss.IndexFlatIP(dim)
            new_index.add(xb)
            index_manager.save_index(new_index, existing_index_name)
            print(f"✓ Rebuilt FAISS index (IndexFlatIP) with {len(embeddings)} faces from {len(set(person_ids))} persons")
        
        return {
            "updated": len(embeddings),
            "persons": len(set(person_ids)),
            "message": f"Successfully rebuilt FAISS index with {len(embeddings)} face embeddings from {len(set(person_ids))} persons"
        }
        
    except Exception as e:
        print(f"Error rebuilding FAISS index: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        conn.close()
