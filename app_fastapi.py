"""
FastAPI application for Chitra Photo Management.
Migrated from Flask with full async support and MinIO storage.
"""
from __future__ import annotations

import os
import io
import tempfile
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, Any, List, Optional
from pathlib import Path

from fastapi import FastAPI, Depends, HTTPException, Query, UploadFile, File, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

import aiosqlite
import asyncio
import numpy as np
from concurrent.futures import ThreadPoolExecutor


from core import db_async
from core.storage_client import MinIOStorageClient
from core.schemas import (
    ScanPathRequest, AddPhotoTagsRequest, IndexEmbeddingsRequest,
    IndexFacesRequest, ClusterFacesRequest, CreatePersonRequest,
    UpdatePersonRequest, AssignFacePersonRequest, MergePersonsRequest,
    PhotoResponse, PhotoListResponse, TagListResponse, TagResponse,
    FaceListResponse, PersonListResponse, PersonResponse,
    JobStatusResponse, SearchResultsResponse, ScanPathResponse,
    UploadPhotosResponse, HealthResponse, RootResponse, StatusResponse,
    ClusterFacesResponse, IndexJobResponse, FaceIndexJobResponse,
    PersonFacesResponse
)
from core.embedder import ClipEmbedder
from core.extractor import collect_metadata, load_image, iter_images, RAW_EXTS
from core.face import face_encodings
from core.tagger import auto_tags
from core.gallery import ensure_thumb
from core.cache import get_cached_thumbnail, cache_thumbnail
from core.worker import get_queue
from core.jobs import (
    process_photo_embedding_job,
    process_photo_faces_job,
    index_embeddings_batch_job,
    index_faces_batch_job,
)
from rq.job import Job
from PIL import Image

# Register HEIC/HEIF support
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass  # pillow-heif not installed, HEIC support will be limited


# Global instances
_STORAGE_CLIENT: MinIOStorageClient | None = None
_EMBEDDER: ClipEmbedder | None = None

# Database path
DB_PATH = os.environ.get("CHITRA_DB_PATH", db_async.DB_DEFAULT_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup
    global _STORAGE_CLIENT, _EMBEDDER
    
    # Initialize storage client (lazy - will connect when needed)
    try:
        _STORAGE_CLIENT = MinIOStorageClient()
        print("✓ MinIO storage client initialized")
    except Exception as e:
        print(f"Warning: MinIO storage client initialization failed: {e}")
        print("  MinIO will be initialized on first use")
        _STORAGE_CLIENT = None
    
    # Initialize database if needed
    try:
        async with db_async.connect_async(DB_PATH) as conn:
            await db_async.init_db_async(DB_PATH)
        print("✓ Database initialized")
    except Exception as e:
        print(f"Warning: Database initialization failed: {e}")
    
    yield
    
    # Shutdown
    _STORAGE_CLIENT = None
    _EMBEDDER = None


# Create FastAPI app
app = FastAPI(
    title="Chitra Photo Management API",
    description="Photo management with embeddings and face detection",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# DEPENDENCY INJECTION
# -----------------------------------------------------------------------------

async def get_db_async() -> AsyncIterator[aiosqlite.Connection]:
    """Dependency: Get async database connection."""
    async with db_async.connect_async(DB_PATH) as conn:
        yield conn


def get_storage_client() -> MinIOStorageClient:
    """Dependency: Get MinIO storage client."""
    global _STORAGE_CLIENT
    if _STORAGE_CLIENT is None:
        try:
            _STORAGE_CLIENT = MinIOStorageClient()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Storage service unavailable: {str(e)}"
            )
    return _STORAGE_CLIENT


def get_embedder() -> ClipEmbedder:
    """Dependency: Get CLIP embedder (sync, use in thread pool)."""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = ClipEmbedder()
    return _EMBEDDER


# -----------------------------------------------------------------------------
# HEALTH CHECK
# -----------------------------------------------------------------------------

@app.get("/api/health")
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    Tests database and MinIO connectivity.
    """
    health_info = {
        "status": "ok",
        "version": "2.0.0",
        "db_path": str(DB_PATH),
    }
    
    # Test database
    try:
        async with db_async.connect_async(DB_PATH) as conn:
            await conn.execute("SELECT 1")
        health_info["db_status"] = "ok"
    except Exception as e:
        health_info["db_status"] = f"error: {str(e)}"
        health_info["status"] = "degraded"
    
    # Test MinIO storage (optional - don't fail if unavailable)
    try:
        storage = get_storage_client()
        # Try to list buckets (lightweight operation)
        storage.client.list_buckets()
        health_info["storage_status"] = "ok"
        health_info["storage_endpoint"] = storage.endpoint
        health_info["storage_bucket"] = storage.bucket_name
    except Exception as e:
        health_info["storage_status"] = f"unavailable: {str(e)[:100]}"
        health_info["status"] = "degraded"
        health_info["storage_note"] = "MinIO server not running or unreachable"
    
    # Return 200 even if degraded (service is still functional)
    return JSONResponse(content=health_info, status_code=200)


# -----------------------------------------------------------------------------
# ROOT ENDPOINT
# -----------------------------------------------------------------------------

@app.get("/")
async def root() -> RootResponse:
    """Root endpoint."""
    return RootResponse(message="Chitra Photo Management API", version="2.0.0", docs="/docs", health="/api/health")


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

def row_to_photo_dto(row) -> Dict[str, Any]:
    """Convert database row to photo DTO."""
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
        "thumb_path": row.get("thumb_path"),
    }


async def ensure_photo_thumb_async(
    file_path: str,
    photo_id: int,
    storage: MinIOStorageClient,
    conn: aiosqlite.Connection
) -> str:
    """
    Ensure thumbnail exists on MinIO. Returns MinIO object key.
    """
    thumb_path = storage.generate_thumbnail_path(photo_id, "photo")
    
    # Check if thumbnail exists on MinIO
    if not await storage.file_exists_async(thumb_path):
        # Download original, generate thumb, upload
        try:
            file_data = await storage.download_file_async(file_path)
        except FileNotFoundError:
            # Original file doesn't exist in MinIO - cannot generate thumbnail
            raise HTTPException(
                status_code=404,
                detail=f"Original photo file not found in storage: {file_path}"
            )
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file_path).suffix) as tmp:
            tmp.write(file_data)
            tmp_path = tmp.name
        
        try:
            # Generate thumbnail
            thumb_file_path = str(Path(tmp_path).with_suffix('.jpg'))
            # Run thumbnail generation in thread pool (CPU-bound)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, ensure_thumb, tmp_path, thumb_file_path)
            
            # Check if thumbnail was actually created
            if not os.path.exists(thumb_file_path):
                raise Exception(f"Thumbnail generation failed: {thumb_file_path} was not created")
            
            # Read thumbnail and upload
            with open(thumb_file_path, 'rb') as f:
                thumb_data = f.read()
            await storage.upload_file_async(thumb_data, thumb_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            if os.path.exists(thumb_file_path):
                os.unlink(thumb_file_path)
    
    return thumb_path


# -----------------------------------------------------------------------------
# PHOTO ENDPOINTS
# -----------------------------------------------------------------------------

@app.get("/api/photos")
async def list_photos(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    conn: aiosqlite.Connection = Depends(get_db_async)
):
    """List photos with pagination."""
    cur = await conn.cursor()
    await cur.execute(
        """SELECT * FROM photos ORDER BY id DESC LIMIT ? OFFSET ?""",
        (limit, offset),
    )
    rows = await cur.fetchall()
    
    items = [PhotoResponse(**row_to_photo_dto(dict(row))) for row in rows]
    return PhotoListResponse(items=items, limit=limit, offset=offset)


@app.get("/api/photos/{photo_id}")
async def get_photo(
    photo_id: int,
    conn: aiosqlite.Connection = Depends(get_db_async)
):
    """Get photo details by ID."""
    cur = await conn.cursor()
    await cur.execute("SELECT * FROM photos WHERE id=?", (photo_id,))
    row = await cur.fetchone()
    
    if not row:
        raise HTTPException(status_code=404, detail="photo_not_found")
    
    return PhotoResponse(**row_to_photo_dto(dict(row)))


@app.get("/api/photos/{photo_id}/image")
async def get_photo_image(
    photo_id: int,
    conn: aiosqlite.Connection = Depends(get_db_async),
    storage: MinIOStorageClient = Depends(get_storage_client)
):
    """Download photo image."""
    cur = await conn.cursor()
    await cur.execute("SELECT file_path FROM photos WHERE id=?", (photo_id,))
    row = await cur.fetchone()
    
    if not row:
        raise HTTPException(status_code=404, detail="photo_not_found")
    
    file_path = row["file_path"]
    
    # Download from MinIO
    try:
        file_data = await storage.download_file_async(file_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="file_not_found_on_storage")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"storage_error: {str(e)}")
    
    # Check if file needs conversion (RAW or HEIC/HEIF)
    file_ext = Path(file_path).suffix.lower()
    heic_exts = {".heic", ".heif"}
    
    if file_ext in RAW_EXTS or file_ext in heic_exts:
        # Process in temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(file_data)
            tmp_path = tmp.name
        
        try:
            # Run image loading in thread pool (I/O and CPU-bound)
            loop = asyncio.get_event_loop()
            img = await loop.run_in_executor(None, load_image, Path(tmp_path))
            # Convert PIL Image to JPEG bytes
            img_io = io.BytesIO()
            img.save(img_io, format='JPEG', quality=95)
            img_io.seek(0)
            return Response(
                content=img_io.read(),
                media_type='image/jpeg',
                headers={"Cache-Control": "public, max-age=86400"}
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"failed_to_convert: {str(e)}")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    # Serve directly
    ext = Path(file_path).suffix.lower()
    mimetype = {
        '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
        '.png': 'image/png', '.gif': 'image/gif',
        '.webp': 'image/webp',
    }.get(ext, 'application/octet-stream')
    
    return Response(
        content=file_data,
        media_type=mimetype,
        headers={"Cache-Control": "public, max-age=86400"}
    )


@app.get("/api/photos/{photo_id}/thumbnail")
async def get_photo_thumbnail(
    photo_id: int,
    conn: aiosqlite.Connection = Depends(get_db_async),
    storage: MinIOStorageClient = Depends(get_storage_client)
):
    """Get photo thumbnail."""
    cur = await conn.cursor()
    await cur.execute("SELECT file_path, thumb_path FROM photos WHERE id=?", (photo_id,))
    row = await cur.fetchone()
    
    if not row:
        raise HTTPException(status_code=404, detail="photo_not_found")
    
    file_path = row["file_path"]
    thumb_path = row["thumb_path"]
    
    # If thumb_path not in DB, generate it and store
    if not thumb_path:
        thumb_path = await ensure_photo_thumb_async(file_path, photo_id, storage, conn)
        await conn.execute("UPDATE photos SET thumb_path = ? WHERE id = ?", (thumb_path, photo_id))
        await conn.commit()
    else:
        # Check if thumbnail exists on MinIO (only if we have path in DB)
        if not await storage.file_exists_async(thumb_path):
            # Thumbnail was deleted or doesn't exist, regenerate
            thumb_path = await ensure_photo_thumb_async(file_path, photo_id, storage, conn)
            await conn.execute("UPDATE photos SET thumb_path = ? WHERE id = ?", (thumb_path, photo_id))
            await conn.commit()
    
    # Check cache first
    thumb_data = get_cached_thumbnail(thumb_path)
    
    if thumb_data is None:
        # Download from MinIO
        try:
            thumb_data = await storage.download_file_async(thumb_path)
            # Cache for future requests
            cache_thumbnail(thumb_path, thumb_data)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="thumbnail_not_found_on_storage")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"storage_error: {str(e)}")
    
    return Response(
        content=thumb_data,
        media_type='image/jpeg',
        headers={"Cache-Control": "public, max-age=86400"}
    )


@app.post("/api/photos/upload")
async def upload_photos(
    files: Optional[List[UploadFile]] = File(None),
    file: Optional[UploadFile] = File(None),
    auto_process: bool = Form(True),
    storage: MinIOStorageClient = Depends(get_storage_client)
):
    """Upload photos (supports multiple files).
    
    Accepts either:
    - 'files' field (multiple files): files=@file1.jpg files=@file2.jpg
    - 'file' field (single file): file=@photo.jpg
    """
    # Handle both "file" (singular) and "files" (plural) for compatibility
    if file:
        files = [file]
    elif not files:
        files = []
    
    if not files:
        raise HTTPException(status_code=400, detail="no_files")
    
    # Read all file data upfront
    file_data_list = []
    for file in files:
        if not file.filename:
            continue
        file_data = await file.read()
        file_data_list.append((file.filename, file_data))
    
    if not file_data_list:
        raise HTTPException(status_code=400, detail="no_valid_files")
    
    async def upload_single_file_async(filename: str, file_data: bytes) -> Dict[str, Any]:
        """Upload a single file asynchronously."""
        tmp_path = None
        thumb_file_path = None
        try:
            async with db_async.connect_async(DB_PATH) as conn:
                # Upload to MinIO storage
                remote_path = storage.generate_photo_path(filename)
                
                # Ensure uniqueness - check database first
                counter = 1
                base_path = remote_path
                cur = await conn.cursor()
                while True:
                    await cur.execute("SELECT id FROM photos WHERE file_path=?", (remote_path,))
                    row = await cur.fetchone()
                    if row is None:
                        # Not in database, check MinIO only if needed (for safety)
                        if not await storage.file_exists_async(remote_path):
                            break
                    # Path exists, generate new one
                    path_obj = Path(base_path)
                    ext = path_obj.suffix
                    name = path_obj.stem
                    remote_path = f"{path_obj.parent}/{name}_{counter}{ext}"
                    counter += 1
                
                # Upload to MinIO
                stored_path = await storage.upload_file_async(file_data, remote_path)
                
                # Process metadata using temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
                    tmp.write(file_data)
                    tmp_path = tmp.name
                
                # Collect metadata
                # Run metadata collection in thread pool (I/O and CPU-bound)
                loop = asyncio.get_event_loop()
                meta = await loop.run_in_executor(None, collect_metadata, Path(tmp_path))
                # Store MinIO path
                meta['file_path'] = stored_path
                await db_async.upsert_photo_async(conn, **meta)
                
                # Get photo ID
                cur = await conn.cursor()
                await cur.execute("SELECT id FROM photos WHERE file_path=?", (stored_path,))
                row = await cur.fetchone()
                if not row:
                    return {"error": f"Failed to get photo ID for {filename}"}
                photo_id = row["id"]
                
                # Generate thumbnail
                thumb_path = await ensure_photo_thumb_async(stored_path, photo_id, storage, conn)
                
                # Store thumb_path in database
                await conn.execute("UPDATE photos SET thumb_path = ? WHERE id = ?", (thumb_path, photo_id))
                await conn.commit()
                
                # Auto-process: embeddings and faces (if enabled)
                if auto_process:
                    try:
                        queue = get_queue()
                        # Queue embedding processing job
                        queue.enqueue(
                            process_photo_embedding_job,
                            photo_id,
                            stored_path,
                            DB_PATH,
                            job_timeout='10m'
                        )
                        # Queue face processing job
                        queue.enqueue(
                            process_photo_faces_job,
                            photo_id,
                            stored_path,
                            DB_PATH,
                            job_timeout='10m'
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
            print(f"Error in upload_single_file_async: {error_msg}")
            print(traceback.format_exc())
            return {"error": error_msg}
        finally:
            # Clean up temporary files
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
    
    # Upload files in parallel (max 5 concurrent uploads)
    saved = []
    errors = []
    
    # For single file, process directly
    if len(file_data_list) == 1:
        filename, file_data = file_data_list[0]
        try:
            result = await upload_single_file_async(filename, file_data)
            if "error" in result:
                raise HTTPException(status_code=500, detail=result["error"])
            saved.append(result)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Exception uploading {filename}: {str(e)}")
    else:
        # Multiple files: use asyncio.gather for parallel processing
        tasks = [
            upload_single_file_async(filename, file_data)
            for filename, file_data in file_data_list
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                errors.append(f"Exception: {str(result)}")
            elif "error" in result:
                errors.append(result["error"])
            else:
                saved.append(result)
    
    # Return results
    response = {"saved": saved}
    if errors:
        response["errors"] = errors
    
    return response




# -----------------------------------------------------------------------------
# PHOTO ENDPOINTS: SCAN PATH
# -----------------------------------------------------------------------------

@app.post("/api/photos/scan-path")
async def scan_path(
    data: ScanPathRequest,
    conn: aiosqlite.Connection = Depends(get_db_async)
):
    """Scan a directory path for photos and add them to the database."""
    root = data.root
    if not root:
        raise HTTPException(status_code=400, detail="missing_root")
    
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise HTTPException(status_code=400, detail="root_not_found")
    
    processed = 0
    for img_path in iter_images(root_path):
        # Run metadata collection in thread pool
        loop = asyncio.get_event_loop()
        meta = await loop.run_in_executor(None, collect_metadata, img_path)
        await db_async.upsert_photo_async(conn, **meta)
        processed += 1
    
    await conn.commit()
    return ScanPathResponse(root=str(root_path), processed=processed)


# -----------------------------------------------------------------------------
# PHOTO ENDPOINTS: TAGS
# -----------------------------------------------------------------------------

@app.get("/api/photos/{photo_id}/tags")
async def get_photo_tags(
    photo_id: int,
    conn: aiosqlite.Connection = Depends(get_db_async)
):
    """Get tags for a photo."""
    cur = await conn.cursor()
    await cur.execute(
        "SELECT tag, score FROM tags WHERE photo_id=? ORDER BY score DESC",
        (photo_id,)
    )
    rows = await cur.fetchall()
    
    tags = [TagResponse(tag=r["tag"], score=r["score"]) for r in rows]
    return TagListResponse(photo_id=photo_id, tags=tags)


@app.post("/api/photos/{photo_id}/tags")
async def add_photo_tags(
    photo_id: int,
    data: AddPhotoTagsRequest,
    conn: aiosqlite.Connection = Depends(get_db_async)
):
    """Add tags to a photo."""
    tags = data.tags
    if not isinstance(tags, list) or not tags:
        raise HTTPException(status_code=400, detail="invalid_tags")
    
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
        await db_async.add_tag_async(conn, photo_id, tag, score)
    
    await conn.commit()
    return StatusResponse(status="ok")


# -----------------------------------------------------------------------------
# INDEXING ENDPOINTS
# -----------------------------------------------------------------------------

@app.post("/api/index/embeddings")
async def index_embeddings(
    data: IndexEmbeddingsRequest,
    conn: aiosqlite.Connection = Depends(get_db_async)
):
    """Queue embedding indexing job."""
    incremental = data.incremental
    
    cur = await conn.cursor()
    
    if incremental:
        await cur.execute(
            """
            SELECT p.id, p.file_path
            FROM photos p
            LEFT JOIN embeddings e ON e.photo_id = p.id
            WHERE e.photo_id IS NULL
            ORDER BY p.id ASC
            """
        )
    else:
        await cur.execute("SELECT id, file_path FROM photos ORDER BY id ASC")
    
    rows = await cur.fetchall()
    
    if not rows:
        return {"indexed": 0, "job_id": None, "message": "No photos to process"}
    
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
            job_timeout='1h'
        )
        return IndexJobResponse(indexed=0, job_id=job.id, message=f"Queued {len(photo_ids_and_paths)} photos for processing", status="queued")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to queue job: {str(e)}")


@app.post("/api/index/faces")
async def index_faces(
    data: IndexFacesRequest,
    conn: aiosqlite.Connection = Depends(get_db_async)
):
    """Queue face detection job."""
    limit = data.limit
    min_score = data.min_score
    thumb_size = data.thumb_size
    
    cur = await conn.cursor()
    await cur.execute("SELECT id, file_path FROM photos ORDER BY id ASC")
    rows = await cur.fetchall()
    
    if not rows:
        return {"processed_photos": 0, "job_id": None, "message": "No photos to process"}
    
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
            job_timeout='1h'
        )
        return FaceIndexJobResponse(processed_photos=0, job_id=job.id, message=f"Queued {len(photo_ids_and_paths)} photos for face processing", status="queued")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to queue job: {str(e)}")


@app.post("/api/index/faces-cluster")
async def cluster_faces(
    data: ClusterFacesRequest,
    conn: aiosqlite.Connection = Depends(get_db_async)
):
    """Cluster faces into persons using FAISS."""
    import faiss
    
    threshold = data.threshold
    
    rows = await db_async.get_faces_embeddings_async(conn)
    
    if not rows:
        raise HTTPException(
            status_code=400,
            detail="no_faces_found",
            headers={"X-Message": "No faces found. Run face indexing first."}
        )
    
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
        raise HTTPException(status_code=400, detail="no_valid_embeddings")
    
    xb = np.stack(vecs).astype("float32")
    faiss.normalize_L2(xb)
    dim = xb.shape[1]
    
    index = faiss.IndexFlatIP(dim)
    index.add(xb)
    
    # For each face, find neighbors and cluster via union-find
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
        person_id = await db_async.get_or_create_person_async(conn, person_name)
        person_idx += 1
        
        for m in members:
            fid = face_ids[m]
            await db_async.set_face_person_async(conn, fid, person_id)
    
    await conn.commit()
    
    return ClusterFacesResponse(clustered_faces=len(face_ids), persons_created=len(clusters), threshold=threshold)


# -----------------------------------------------------------------------------
# JOB STATUS
# -----------------------------------------------------------------------------

@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str) -> JobStatusResponse:
    """Get status of a background job."""
    try:
        queue = get_queue()
        job = Job.fetch(job_id, connection=queue.connection)
        
        status_info = JobStatusResponse(
            job_id=job.id,
            status=job.get_status(),
            created_at=job.created_at.isoformat() if job.created_at else None,
            started_at=job.started_at.isoformat() if job.started_at else None,
            ended_at=job.ended_at.isoformat() if job.ended_at else None,
            result=job.result if job.is_finished else None,
            error=str(job.exc_info) if job.is_failed and job.exc_info else None
        )
        
        return status_info
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Job not found: {str(e)}")


# -----------------------------------------------------------------------------
# SEARCH ENDPOINTS
# -----------------------------------------------------------------------------

@app.get("/api/search/photos")
async def search_photos(
    q: Optional[str] = Query(None, alias="query"),
    limit: int = Query(20, ge=1, le=100),
    conn: aiosqlite.Connection = Depends(get_db_async),
    embedder: ClipEmbedder = Depends(get_embedder)
):
    """Search photos by text query using embeddings."""
    query = q
    if not query:
        raise HTTPException(status_code=400, detail="missing_query")
    
    # Get text embedding
    q_vec = embedder.text_embedding(query)  # 1D float32
    q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-9)
    
    rows = await db_async.get_embeddings_async(conn)
    if not rows:
        return {"results": []}
    
    photo_ids: List[int] = []
    vecs: List[np.ndarray] = []
    
    for photo_id, dim, vec_bytes in rows:
        v = np.frombuffer(vec_bytes, dtype="float32")
        if v.shape[0] != dim:
            continue
        photo_ids.append(photo_id)
        vecs.append(v)
    
    if not vecs:
        return {"results": []}
    
    mat = np.stack(vecs, axis=0)
    mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    
    sims = mat @ q_vec
    idx = np.argsort(-sims)[:limit]
    
    cur = await conn.cursor()
    results = []
    for i in idx:
        pid = photo_ids[int(i)]
        score = float(sims[int(i)])
        await cur.execute("SELECT * FROM photos WHERE id=?", (pid,))
        row = await cur.fetchone()
        if not row:
            continue
        dto = row_to_photo_dto(dict(row))
        dto["score"] = score
        results.append(PhotoResponse(**dto))
    
    return SearchResultsResponse(query=query, results=results)


@app.get("/api/search/by-person")
async def search_by_person(
    name: Optional[str] = Query(None, alias="q"),
    conn: aiosqlite.Connection = Depends(get_db_async)
):
    """Search photos containing a specific person."""
    person_name = name
    if not person_name:
        raise HTTPException(status_code=400, detail="missing_person_name")
    
    cur = await conn.cursor()
    
    # Find photos that have faces tagged with this person (case-insensitive partial match)
    await cur.execute(
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
    rows = await cur.fetchall()
    
    results = [PhotoResponse(**row_to_photo_dto(dict(r))) for r in rows]
    return SearchResultsResponse(query=person_name, results=results)


# -----------------------------------------------------------------------------
# FACE ENDPOINTS
# -----------------------------------------------------------------------------

@app.get("/api/faces/{face_id}/thumbnail")
async def get_face_thumbnail(
    face_id: int,
    conn: aiosqlite.Connection = Depends(get_db_async),
    storage: MinIOStorageClient = Depends(get_storage_client)
):
    """Serve face thumbnail image."""
    cur = await conn.cursor()
    await cur.execute(
        """
        SELECT ft.thumb_path
        FROM face_thumbs ft
        WHERE ft.face_id = ?
        """,
        (face_id,),
    )
    row = await cur.fetchone()
    
    if not row or not row["thumb_path"]:
        raise HTTPException(status_code=404, detail="face_thumb_not_found")
    
    thumb_path = row["thumb_path"]
    
    # Check cache first
    thumb_data = get_cached_thumbnail(thumb_path)
    
    if thumb_data is None:
        # Download from MinIO
        try:
            thumb_data = await storage.download_file_async(thumb_path)
            # Cache for future requests
            cache_thumbnail(thumb_path, thumb_data)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="thumb_not_found_on_storage")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"storage_error: {str(e)}")
    
    return Response(
        content=thumb_data,
        media_type='image/jpeg',
        headers={"Cache-Control": "public, max-age=86400"}
    )


@app.get("/api/faces")
async def list_faces(
    conn: aiosqlite.Connection = Depends(get_db_async)
) -> FaceListResponse:
    """List all faces with thumbnails."""
    rows = await db_async.get_face_thumbs_async(conn)
    
    items = []
    for r in rows:
        items.append(FaceResponse(
            face_id=r["face_id"],
            photo_id=r["photo_id"],
            photo_path=r["file_path"],
            thumb_path=r["thumb_path"],
            person_id=r["person_id"],
            person_name=r["person_name"]
        ))
    
    return FaceListResponse(items=items)

@app.get("/api/persons")
async def list_persons(
    conn: aiosqlite.Connection = Depends(get_db_async)
):
    """List all persons."""
    rows = await db_async.list_persons_async(conn)
    
    persons = [PersonResponse(id=r["id"], name=r["name"]) for r in rows]
    return PersonListResponse(persons=persons)


@app.post("/api/persons")
async def create_person(
    data: CreatePersonRequest,
    conn: aiosqlite.Connection = Depends(get_db_async)
):
    """Create a new person."""
    name = data.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="missing_name")
    
    person_id = await db_async.create_person_async(conn, name)
    await conn.commit()
    return PersonResponse(id=person_id, name=name)


@app.put("/api/persons/{person_id}")
async def update_person(
    person_id: int,
    data: UpdatePersonRequest,
    conn: aiosqlite.Connection = Depends(get_db_async)
):
    """Update a person's name."""
    name = data.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="missing_name")
    
    await db_async.rename_person_async(conn, person_id, name)
    await conn.commit()
    return PersonResponse(id=person_id, name=name)


@app.get("/api/persons/{person_id}/faces")
async def get_person_faces(
    person_id: int,
    conn: aiosqlite.Connection = Depends(get_db_async)
):
    """Get face thumbnails for a specific person."""
    rows = await db_async.get_face_thumbs_for_person_async(conn, person_id)
    
    items = []
    for r in rows:
        items.append({
            "face_id": r["face_id"],
            "photo_id": r["photo_id"],
            "thumb_path": r["thumb_path"],
        })
    return PersonFacesResponse(items=items)


@app.post("/api/persons/{target_person_id}/merge")
async def merge_persons_endpoint(
    target_person_id: int,
    data: MergePersonsRequest,
    conn: aiosqlite.Connection = Depends(get_db_async)
):
    """Merge source person into target person."""
    source_person_id = data.source_person_id
    
    if not isinstance(source_person_id, int):
        raise HTTPException(status_code=400, detail="missing_source_person_id")
    
    if source_person_id == target_person_id:
        raise HTTPException(status_code=400, detail="cannot_merge_same_person")
    
    try:
        await db_async.merge_persons_async(conn, source_person_id, target_person_id)
        await conn.commit()
        return StatusResponse(status="merged")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------------------------
# STORAGE ENDPOINT
# -----------------------------------------------------------------------------

@app.get("/api/storage/{file_path:path}")
async def get_storage_file(
    file_path: str,
    storage: MinIOStorageClient = Depends(get_storage_client)
):
    """Generic endpoint to serve any file from MinIO storage."""
    try:
        file_data = await storage.download_file_async(file_path)
        ext = Path(file_path).suffix.lower()
        mimetype = {
            '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
            '.png': 'image/png', '.gif': 'image/gif',
            '.webp': 'image/webp', '.bmp': 'image/bmp',
            '.tif': 'image/tiff', '.tiff': 'image/tiff',
        }.get(ext, 'application/octet-stream')
        
        return Response(content=file_data, media_type=mimetype)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="file_not_found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"storage_error: {str(e)}")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app_fastapi:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        reload=True
    )

