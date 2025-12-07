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
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

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
    FaceResponse, FaceListResponse, PersonListResponse, PersonResponse,
    JobStatusResponse, SearchResultsResponse, ScanPathResponse,
    UploadPhotosResponse, HealthResponse, RootResponse, StatusResponse,
    ClusterFacesResponse, IndexJobResponse, FaceIndexJobResponse,
    PersonFacesResponse
)
from core.embedder import ClipEmbedder
from core.extractor import collect_metadata, load_image, iter_images, RAW_EXTS, _normalize_exif_date
from core.face import face_encodings
from core.tagger import auto_tags
from core.gallery import ensure_thumb
from core.cache import get_cached_thumbnail, cache_thumbnail
from core.worker import get_queue
from core.faiss_index import FAISSIndexManager
from core.jobs import (
    process_photo_embedding_job,
    process_photo_faces_job,
    index_embeddings_batch_job,
    index_faces_batch_job,
    cluster_faces_job,
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

# Size limit middleware for large file uploads
class MaxUploadSizeMiddleware(BaseHTTPMiddleware):
    """Middleware to limit upload size and prevent memory issues."""
    MAX_SIZE = 5 * 1024 * 1024 * 1024  # 5GB default
    
    async def dispatch(self, request: Request, call_next):
        if request.method == "POST" and "/api/photos/upload" in str(request.url):
            content_length = request.headers.get("content-length")
            if content_length:
                try:
                    size = int(content_length)
                    if size > self.MAX_SIZE:
                        return JSONResponse(
                            {
                                "error": f"File too large. Maximum size: {self.MAX_SIZE / (1024**3):.1f}GB",
                                "max_size": self.MAX_SIZE,
                                "received_size": size
                            },
                            status_code=413
                        )
                except (ValueError, TypeError):
                    pass  # Invalid content-length, let it proceed
        return await call_next(request)

app.add_middleware(MaxUploadSizeMiddleware)

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
        try:
            storage.client.list_buckets()
            health_info["storage_status"] = "ok"
            health_info["storage_endpoint"] = storage.endpoint
            health_info["storage_bucket"] = storage.bucket_name
        except Exception as e:
            # Storage client exists but connection failed
            health_info["storage_status"] = f"unavailable: {str(e)[:100]}"
            health_info["status"] = "degraded"
            health_info["storage_endpoint"] = getattr(storage, 'endpoint', 'unknown')
            health_info["storage_bucket"] = getattr(storage, 'bucket_name', 'unknown')
    except Exception as e:
        # Storage client creation failed
        health_info["storage_status"] = f"error: {str(e)[:100]}"
        health_info["status"] = "degraded"
    
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
    """Convert database row to photo DTO, normalizing date formats."""
    from datetime import datetime
    
    # Normalize exif_datetime (should already be normalized, but handle legacy data)
    exif_dt = row.get("exif_datetime") or ""
    if exif_dt:
        # Re-normalize if needed (handles legacy data with old formats)
        normalized = _normalize_exif_date(exif_dt)
        exif_dt = normalized if normalized else ""
    
    # Normalize created_at to ISO format if not already
    created_at = row.get("created_at") or ""
    if created_at:
        try:
            # If it already has T, assume it's ISO-like
            if "T" not in created_at and " " in created_at:
                # Convert space to T: "2025-12-07 10:30:45" -> "2025-12-07T10:30:45"
                created_at = created_at.replace(" ", "T", 1)
            # Validate by parsing
            datetime.fromisoformat(created_at)
        except (ValueError, AttributeError):
            # If invalid, try to normalize using the same function
            normalized = _normalize_exif_date(created_at)
            if normalized:
                created_at = normalized
            elif not created_at.strip():
                created_at = ""
    
    return {
        "id": row["id"],
        "file_path": row["file_path"],
        "size": row["size"],
        "created_at": created_at,
        "checksum": row["checksum"],
        "phash": row["phash"],
        "exif_datetime": exif_dt,
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
    """List photos with pagination, sorted by date (newest first)."""
    cur = await conn.cursor()
    
    # Get total count
    await cur.execute("SELECT COUNT(*) FROM photos")
    total = (await cur.fetchone())[0]
    
    # Sort by exif_datetime if available, otherwise created_at, newest first
    # Normalize EXIF dates (YYYY:MM:DD HH:MM:SS -> YYYY-MM-DDTHH:MM:SS) for proper sorting
    await cur.execute(
        """
        SELECT * FROM photos 
        ORDER BY 
            CASE 
                WHEN exif_datetime != '' AND exif_datetime IS NOT NULL THEN
                    -- Normalize EXIF format: YYYY:MM:DD HH:MM:SS -> YYYY-MM-DDTHH:MM:SS
                    -- Extract date parts and time part, then combine with proper separators
                    SUBSTR(exif_datetime, 1, 4) || '-' || 
                    SUBSTR(exif_datetime, 6, 2) || '-' || 
                    SUBSTR(exif_datetime, 9, 2) || 'T' || 
                    SUBSTR(exif_datetime, 12)
                ELSE created_at
            END DESC,
            id DESC
        LIMIT ? OFFSET ?
        """,
        (limit, offset),
    )
    rows = await cur.fetchall()
    
    items = [PhotoResponse(**row_to_photo_dto(dict(row))) for row in rows]
    return PhotoListResponse(items=items, limit=limit, offset=offset, total=total)


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


@app.delete("/api/photos/{photo_id}")
async def delete_photo(
    photo_id: int,
    conn: aiosqlite.Connection = Depends(get_db_async),
    storage: MinIOStorageClient = Depends(get_storage_client)
):
    """Delete a photo and all related data."""
    cur = await conn.cursor()
    
    # Get photo details before deletion
    await cur.execute("SELECT file_path, thumb_path FROM photos WHERE id=?", (photo_id,))
    row = await cur.fetchone()
    
    if not row:
        raise HTTPException(status_code=404, detail="photo_not_found")
    
    file_path = row["file_path"]
    thumb_path = row["thumb_path"] if row["thumb_path"] else None
    
    # Get face thumbnails for this photo before deletion
    face_thumb_paths = []
    await cur.execute(
        """
        SELECT ft.thumb_path 
        FROM face_thumbs ft
        JOIN faces f ON ft.face_id = f.id
        WHERE f.photo_id = ?
        """,
        (photo_id,)
    )
    face_thumb_rows = await cur.fetchall()
    for face_row in face_thumb_rows:
        if face_row["thumb_path"]:
            face_thumb_paths.append(face_row["thumb_path"])
    
    # Delete files from MinIO storage
    try:
        # Delete original photo file
        if file_path:
            await storage.delete_file_async(file_path)
        
        # Delete photo thumbnail if it exists
        if thumb_path:
            await storage.delete_file_async(thumb_path)
        
        # Delete face thumbnails
        for face_thumb_path in face_thumb_paths:
            try:
                await storage.delete_file_async(face_thumb_path)
            except Exception as e:
                print(f"Warning: Failed to delete face thumbnail '{face_thumb_path}': {e}")
    except Exception as e:
        # Log error but continue with database deletion
        print(f"Warning: Failed to delete some files from storage for photo {photo_id}: {e}")
    
    # Delete from database (CASCADE will handle related records: tags, embeddings, clusters, faces, face_thumbs)
    await cur.execute("DELETE FROM photos WHERE id=?", (photo_id,))
    await conn.commit()
    
    return {"message": "Photo deleted successfully", "photo_id": photo_id}


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
                headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"}
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
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"}
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
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"}
    )


@app.delete("/api/photos/{photo_id}")
async def delete_photo(
    photo_id: int,
    conn: aiosqlite.Connection = Depends(get_db_async),
    storage: MinIOStorageClient = Depends(get_storage_client)
):
    """Delete a photo and all related data."""
    cur = await conn.cursor()
    
    # Get photo details before deletion
    await cur.execute("SELECT file_path, thumb_path FROM photos WHERE id=?", (photo_id,))
    row = await cur.fetchone()
    
    if not row:
        raise HTTPException(status_code=404, detail="photo_not_found")
    
    file_path = row["file_path"]
    thumb_path = row["thumb_path"] if row["thumb_path"] else None
    
    # Get face thumbnails for this photo before deletion
    face_thumb_paths = []
    await cur.execute(
        """
        SELECT ft.thumb_path 
        FROM face_thumbs ft
        JOIN faces f ON ft.face_id = f.id
        WHERE f.photo_id = ?
        """,
        (photo_id,)
    )
    face_thumb_rows = await cur.fetchall()
    for face_row in face_thumb_rows:
        if face_row["thumb_path"]:
            face_thumb_paths.append(face_row["thumb_path"])
    
    # Delete files from MinIO storage
    try:
        # Delete original photo file
        if file_path:
            await storage.delete_file_async(file_path)
        
        # Delete photo thumbnail if it exists
        if thumb_path:
            await storage.delete_file_async(thumb_path)
        
        # Delete face thumbnails
        for face_thumb_path in face_thumb_paths:
            try:
                await storage.delete_file_async(face_thumb_path)
            except Exception as e:
                print(f"Warning: Failed to delete face thumbnail '{face_thumb_path}': {e}")
    except Exception as e:
        # Log error but continue with database deletion
        print(f"Warning: Failed to delete some files from storage for photo {photo_id}: {e}")
    
    # Delete from database (CASCADE will handle related records: tags, embeddings, clusters, faces, face_thumbs)
    await cur.execute("DELETE FROM photos WHERE id=?", (photo_id,))
    await conn.commit()
    
    return {"message": "Photo deleted successfully", "photo_id": photo_id}


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
    
    # Filter out files without filenames
    valid_files = [f for f in files if f.filename]
    if not valid_files:
        raise HTTPException(status_code=400, detail="no_valid_files")
    
    async def upload_single_file_streaming(file: UploadFile) -> Dict[str, Any]:
        """Upload a single file asynchronously using streaming to avoid loading entire file into memory."""
        filename = file.filename
        if not filename:
            return {"error": "No filename provided"}
        
        tmp_path = None
        thumb_file_path = None
        try:
            # Stream file directly to temporary file (avoids loading entire file into memory)
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
                # Stream in chunks to avoid memory issues
                chunk_size = 1024 * 1024  # 1MB chunks
                while chunk := await file.read(chunk_size):
                    tmp.write(chunk)
                tmp_path = tmp.name
            
            # Reset file pointer for potential reuse (though we won't reuse it)
            await file.seek(0)
            
            # Collect metadata first to check for duplicates
            # Run metadata collection in thread pool (I/O and CPU-bound)
            loop = asyncio.get_event_loop()
            meta = await loop.run_in_executor(None, collect_metadata, Path(tmp_path))
            
            async with db_async.connect_async(DB_PATH) as conn:
                # Check for duplicate by checksum BEFORE uploading to MinIO
                if meta.get('checksum'):
                    cur = await conn.cursor()
                    await cur.execute(
                        "SELECT id, file_path FROM photos WHERE checksum = ?",
                        (meta['checksum'],)
                    )
                    existing_photo = await cur.fetchone()
                    if existing_photo:
                        # Duplicate found - skip upload and return existing photo info
                        return {
                            "id": existing_photo["id"],
                            "file_path": existing_photo["file_path"],
                            "duplicate": True,
                            "message": f"Duplicate photo detected (same checksum as photo #{existing_photo['id']})"
                        }
                
                # No duplicate found - proceed with upload
                # Upload to MinIO storage
                remote_path = storage.generate_photo_path(filename)
                
                # Ensure path uniqueness - check database first
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
                
                # Upload to MinIO directly from temp file (avoids loading entire file into memory)
                # Check if file is empty
                if os.path.getsize(tmp_path) == 0:
                    return {"error": f"Empty file: {filename}"}
                
                # Upload directly from file path for large files
                stored_path = await storage.upload_file_from_path_async(tmp_path, remote_path)
                
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
            print(f"Error in upload_single_file_streaming: {error_msg}")
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
    duplicates = []
    errors = []
    
    # Process files (streaming, handles large files efficiently)
    if len(valid_files) == 1:
        # Single file
        try:
            result = await upload_single_file_streaming(valid_files[0])
            if "error" in result:
                raise HTTPException(status_code=500, detail=result["error"])
            if result.get("duplicate"):
                duplicates.append(result)
            else:
                saved.append(result)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Exception uploading {valid_files[0].filename}: {str(e)}")
    else:
        # Multiple files: use asyncio.gather for parallel processing
        # Limit concurrent uploads to avoid overwhelming system
        semaphore = asyncio.Semaphore(3)  # Max 3 concurrent uploads for large files
        
        async def upload_with_semaphore(file: UploadFile):
            async with semaphore:
                return await upload_single_file_streaming(file)
        
        tasks = [upload_with_semaphore(file) for file in valid_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for file, result in zip(valid_files, results):
            filename = file.filename or "unknown"
            if isinstance(result, Exception):
                errors.append(f"{filename}: Exception: {str(result)}")
            elif "error" in result:
                errors.append(f"{filename}: {result['error']}")
            elif result.get("duplicate"):
                duplicates.append(result)
            else:
                saved.append(result)
    
    # Auto-trigger clustering after face detection completes
    # Queue clustering job with a delay to allow face detection jobs to start
    if auto_process and saved:
        try:
            # Extract photo IDs from successfully uploaded photos
            uploaded_photo_ids = [result["id"] for result in saved if "id" in result and not result.get("duplicate", False)]
            
            if uploaded_photo_ids:
                queue = get_queue()
                # Queue clustering job with 30 second delay to allow face detection to complete
                # Pass photo_ids to only cluster faces from newly uploaded photos (more efficient)
                try:
                    # Try using enqueue_in if available (RQ scheduler)
                    from datetime import timedelta
                    queue.enqueue_in(
                        timedelta(seconds=30),
                        cluster_faces_job,
                        DB_PATH,
                        0.75,  # threshold
                        uploaded_photo_ids,  # Only cluster faces from these photos
                        job_timeout='5m'
                    )
                    print(f"✓ Queued auto-clustering job for {len(uploaded_photo_ids)} photos (will run in 30 seconds)")
                except AttributeError:
                    # enqueue_in not available, use regular enqueue (will run immediately)
                    # Face detection jobs will complete first due to their processing time
                    queue.enqueue(
                        cluster_faces_job,
                        DB_PATH,
                        0.75,  # threshold
                        uploaded_photo_ids,  # Only cluster faces from these photos
                        job_timeout='5m'
                    )
                    print(f"✓ Queued auto-clustering job for {len(uploaded_photo_ids)} photos (will run after face detection completes)")
        except Exception as e:
            # Don't fail upload if clustering job queuing fails
            print(f"Warning: Failed to queue auto-clustering job: {e}")
    
    # Return results
    response = {"saved": saved}
    if duplicates:
        response["duplicates"] = duplicates
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
    """Cluster faces into persons using FAISS HNSW with persistence, matching to existing persons first."""
    import faiss
    
    threshold = data.threshold
    
    # Initialize index manager
    index_manager = FAISSIndexManager()
    
    # Get ONLY unassigned faces (person_id IS NULL) - this prevents reassigning already-assigned faces
    unassigned_faces = await db_async.get_unassigned_faces_embeddings_async(conn)
    
    if not unassigned_faces:
        # All faces already assigned or no faces with embeddings
        return ClusterFacesResponse(
            clustered_faces=0,
            persons_created=0,
            threshold=threshold
        )
    
    # Get existing person faces for matching
    existing_person_faces = await db_async.get_person_faces_embeddings_async(conn)
    
    # Process unassigned faces
    unassigned_face_ids = []
    unassigned_vecs = []
    
    for row in unassigned_faces:
        fid = row["id"]
        emb_bytes = row["embedding"]
        if not emb_bytes:
            continue
        v = np.frombuffer(emb_bytes, dtype=np.float32)
        if v.size == 0:
            continue
        
        unassigned_face_ids.append(fid)
        unassigned_vecs.append(v)
    
    if not unassigned_vecs:
        # No valid embeddings in unassigned faces
        return ClusterFacesResponse(
            clustered_faces=0,
            persons_created=0,
            threshold=threshold
        )
    
    # Build index of existing person faces for matching
    existing_person_vecs = []
    existing_person_face_ids = []
    existing_person_map = {}  # face_id -> (person_id, person_name)
    
    for row in existing_person_faces:
        emb_bytes = row["embedding"]
        if not emb_bytes:
            continue
        v = np.frombuffer(emb_bytes, dtype=np.float32)
        if v.size == 0:
            continue
        
        existing_person_vecs.append(v)
        existing_person_face_ids.append(row["id"])
        existing_person_map[row["id"]] = (row["person_id"], row["person_name"])
    
    # Phase 1: Match unassigned faces to existing persons (with batch processing)
    matched_assignments = []  # List of (face_id, person_id) tuples for batch update
    unmatched_indices = []
    
    # Process in batches to avoid loading all embeddings into memory at once
    BATCH_SIZE = 1000
    
    if existing_person_vecs:
        # Load or build HNSW index for existing person faces (with persistence)
        existing_xb = np.stack(existing_person_vecs).astype("float32")
        existing_index_name = "existing_person_faces"
        
        # Try to load existing index, or build new one
        existing_index = index_manager.load_index(existing_index_name)
        
        # Check if index exists and is valid
        index_needs_rebuild = False
        if existing_index is None:
            index_needs_rebuild = True
        else:
            # Check if index size matches current data
            try:
                if hasattr(existing_index, 'ntotal') and existing_index.ntotal != len(existing_person_vecs):
                    index_needs_rebuild = True
                elif not hasattr(existing_index, 'ntotal'):
                    # Index doesn't have ntotal attribute, might be corrupted
                    index_needs_rebuild = True
            except Exception:
                # Error checking index, rebuild
                index_needs_rebuild = True
        
        if index_needs_rebuild:
            # Index doesn't exist or is outdated, build new HNSW index
            try:
                existing_index = index_manager.build_hnsw_index(
                    existing_xb,
                    existing_index_name,
                    m=32,  # Connections per node (tune for accuracy/speed)
                    ef_construction=200
                )
            except Exception as e:
                # If HNSW fails, fall back to IndexFlatIP
                print(f"Warning: HNSW index build failed, using IndexFlatIP: {e}")
                faiss.normalize_L2(existing_xb)
                existing_dim = existing_xb.shape[1]
                existing_index = faiss.IndexFlatIP(existing_dim)
                existing_index.add(existing_xb)
        
        # Process unassigned faces in batches
        for batch_start in range(0, len(unassigned_face_ids), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(unassigned_face_ids))
            batch_face_ids = unassigned_face_ids[batch_start:batch_end]
            batch_vecs = unassigned_vecs[batch_start:batch_end]
            
            # Check batch against existing persons
            batch_xb = np.stack(batch_vecs).astype("float32")
            
            # Search for best match in existing persons using HNSW
            try:
                D_match, I_match = index_manager.search(
                    existing_index,
                    batch_xb,
                    k=1,
                    ef_search=50  # Tune for accuracy/speed
                )
            except Exception as e:
                # Fallback to direct FAISS search if manager search fails
                print(f"Warning: Index manager search failed, using direct search: {e}")
                faiss.normalize_L2(batch_xb)
                D_match, I_match = existing_index.search(batch_xb, 1)
            
            for i, face_id in enumerate(batch_face_ids):
                # Double-check face is still unassigned before processing
                cur = await conn.cursor()
                await cur.execute("SELECT person_id FROM faces WHERE id=?", (face_id,))
                face_row = await cur.fetchone()
                if face_row and face_row["person_id"] is not None:
                    # Face was already assigned, skip it
                    print(f"Warning: Face {face_id} was already assigned, skipping")
                    continue
                
                similarity = D_match[i, 0] if D_match[i, 0] > 0 else 0
                if similarity >= threshold:
                    # Match found! Assign to existing person
                    matched_existing_face_idx = I_match[i, 0]
                    matched_existing_face_id = existing_person_face_ids[matched_existing_face_idx]
                    person_id, person_name = existing_person_map[matched_existing_face_id]
                    
                    matched_assignments.append((face_id, person_id))
                else:
                    # No match found, will cluster later
                    global_idx = batch_start + i
                    unmatched_indices.append(global_idx)
    else:
        # No existing persons, all faces are unmatched
        unmatched_indices = list(range(len(unassigned_face_ids)))
    
    # Batch update all matched faces
    if matched_assignments:
        await db_async.set_faces_person_batch_async(conn, matched_assignments)
    matched_count = len(matched_assignments)
    
    # Phase 2: Cluster unmatched faces into new persons
    if unmatched_indices:
        unmatched_face_ids = [unassigned_face_ids[i] for i in unmatched_indices]
        unmatched_vecs = [unassigned_vecs[i] for i in unmatched_indices]
        
        print(f"Debug: Phase 2 - Starting clustering for {len(unmatched_face_ids)} unmatched faces with threshold {threshold}")
        
        # Create a copy for index building (build_hnsw_index normalizes in place)
        xb = np.stack(unmatched_vecs).astype("float32")
        xb_for_index = xb.copy()  # Copy to avoid modifying original
        dim = xb.shape[1]
        n = xb.shape[0]
        
        # Validate threshold
        if threshold < 0.5 or threshold > 1.0:
            print(f"Warning: Threshold {threshold} is outside recommended range [0.5, 1.0]")
        
        # Build HNSW index for clustering (with persistence) - faster than IndexFlatIP
        cluster_index_name = "unmatched_faces_cluster"
        use_hnsw = True
        try:
            cluster_index = index_manager.build_hnsw_index(
                xb_for_index,
                cluster_index_name,
                m=32,  # Connections per node
                ef_construction=200
            )
            print(f"Debug: Built HNSW index for {n} faces (fast approximate search)")
        except Exception as e:
            # If HNSW fails, fall back to IndexFlatIP
            print(f"Warning: HNSW index build failed, using IndexFlatIP: {e}")
            faiss.normalize_L2(xb_for_index)
            cluster_index = faiss.IndexFlatIP(dim)
            cluster_index.add(xb_for_index)
            use_hnsw = False
        
        # For each face, find neighbors and cluster via union-find
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
        
        # Search for neighbors
        if use_hnsw:
            # HNSW returns L2 distances - convert to cosine similarity
            try:
                D_l2, I = index_manager.search(
                    cluster_index,
                    xb.copy(),  # Use copy for search to avoid side effects
                    k=k,
                    ef_search=50
                )
            except Exception as e:
                # Fallback to direct FAISS search if manager search fails
                print(f"Warning: Index manager search failed, using direct search: {e}")
                xb_search = xb.copy()
                faiss.normalize_L2(xb_search)
                D_l2, I = cluster_index.search(xb_search, k)
            
            # Convert L2 distances to cosine similarity
            # For normalized vectors: L2² = 2(1 - cosine_sim) => cosine_sim = 1 - L2²/2
            # Check if distances are squared (max > 2) or regular (max <= 2)
            max_l2 = D_l2.max()
            if max_l2 > 2.0:
                # Distances are already squared
                D = 1.0 - (D_l2 / 2.0)
            else:
                # Distances are regular L2, square them first
                D = 1.0 - ((D_l2 ** 2) / 2.0)
            # Clamp to [0, 1] range (should already be in range, but ensure it)
            D = np.clip(D, 0.0, 1.0)
            print(f"Debug: Converted L2 distances (max={max_l2:.3f}) to cosine similarity (range: {D.min():.3f} to {D.max():.3f})")
        else:
            # IndexFlatIP returns inner products = cosine similarity for normalized vectors
            xb_search = xb.copy()
            faiss.normalize_L2(xb_search)
            D, I = cluster_index.search(xb_search, k)
            # Clamp to [0, 1] range
            D = np.clip(D, 0.0, 1.0)
        
        # Track similarity scores for debugging
        similarity_scores = []
        union_count = 0
        
        for i in range(n):
            for j in range(1, k):  # skip self at j=0
                if I[i, j] < 0:
                    continue
                similarity = float(D[i, j])
                similarity_scores.append(similarity)
                if similarity >= threshold:
                    union(i, I[i, j])
                    union_count += 1
        
        # Log similarity statistics
        if similarity_scores:
            print(f"Debug: Similarity scores - min={min(similarity_scores):.3f}, max={max(similarity_scores):.3f}, "
                  f"avg={sum(similarity_scores)/len(similarity_scores):.3f}, "
                  f"median={sorted(similarity_scores)[len(similarity_scores)//2]:.3f}")
            above_threshold = sum(1 for s in similarity_scores if s >= threshold)
            print(f"Debug: {above_threshold}/{len(similarity_scores)} similarities >= threshold {threshold}")
            print(f"Debug: Performed {union_count} unions")
        
        # Collect clusters
        clusters: Dict[int, List[int]] = {}
        for i in range(n):
            root = find(i)
            clusters.setdefault(root, []).append(i)
        
        # Log cluster statistics
        cluster_sizes = [len(members) for members in clusters.values()]
        if cluster_sizes:
            print(f"Debug: Created {len(clusters)} clusters - sizes: min={min(cluster_sizes)}, "
                  f"max={max(cluster_sizes)}, avg={sum(cluster_sizes)/len(cluster_sizes):.1f}")
            
            # Warn if one cluster is too large (might indicate threshold too low)
            max_cluster_size = max(cluster_sizes)
            if max_cluster_size > n * 0.8:  # More than 80% of faces in one cluster
                print(f"⚠ Warning: Largest cluster contains {max_cluster_size}/{n} faces ({max_cluster_size/n*100:.1f}%)")
                print(f"⚠ Consider increasing threshold above {threshold} to get more granular clusters")
            
            # Warn if too many singleton clusters (might indicate threshold too high)
            singleton_count = sum(1 for size in cluster_sizes if size == 1)
            if singleton_count > len(clusters) * 0.5:  # More than 50% are singletons
                print(f"⚠ Warning: {singleton_count}/{len(clusters)} clusters are singletons ({singleton_count/len(clusters)*100:.1f}%)")
                print(f"⚠ Consider decreasing threshold below {threshold} to merge similar faces")
        
        # Validate we have clusters and faces to assign
        if len(clusters) == 0:
            print("Warning: No clusters created, cannot assign faces")
            persons_created = 0
        elif len(unmatched_face_ids) == 0:
            print("Warning: No unmatched face IDs available")
            persons_created = 0
        elif len(clusters) == 1 and len(list(clusters.values())[0]) == n:
            # All faces in one cluster - this might be a threshold issue
            print(f"⚠ Warning: All {n} faces clustered into a single group!")
            print(f"⚠ This suggests threshold {threshold} may be too low, or faces are very similar")
            print(f"⚠ Consider increasing threshold or checking face embeddings")
            # Still proceed with assignment, but warn user
            # Continue to assignment logic below
        else:
            # Get existing person names to avoid duplicates
            existing_person_names = set()
            async with conn.execute("SELECT name FROM persons") as cur:
                for row in await cur.fetchall():
                    existing_person_names.add(row["name"])
            
            # Assign new persons to clusters (collect all assignments first)
            person_idx = 1
            persons_created = 0
            cluster_assignments = []  # List of (face_id, person_id) tuples for batch update
            
            for root, members in clusters.items():
                # Skip empty clusters
                if not members or len(members) == 0:
                    print(f"Warning: Cluster {root} is empty, skipping")
                    continue
                
                print(f"Debug: Processing cluster {root} with {len(members)} members")
                
                # Find next available person name (avoid duplicates)
                while True:
                    person_name = f"Person {person_idx}"
                    if person_name not in existing_person_names:
                        break
                    person_idx += 1
                
                # Create or get person (get_or_create handles existing names)
                person_id = await db_async.get_or_create_person_async(conn, person_name)
                existing_person_names.add(person_name)  # Track created names
                person_idx += 1
                persons_created += 1
                
                print(f"Debug: Created person '{person_name}' (ID: {person_id}) for cluster {root}")
                
                # Assign all faces in this cluster to the person
                faces_assigned_this_cluster = 0
                for m in members:
                    # Validate index bounds
                    if m >= len(unmatched_face_ids):
                        print(f"Error: Member index {m} out of range (max: {len(unmatched_face_ids)-1})")
                        continue
                    
                    fid = unmatched_face_ids[m]
                    
                    # Double-check face is still unassigned before assigning
                    cur = await conn.cursor()
                    await cur.execute("SELECT person_id FROM faces WHERE id=?", (fid,))
                    face_row = await cur.fetchone()
                    
                    if not face_row:
                        print(f"Warning: Face {fid} not found in database, skipping")
                        continue
                    
                    if face_row["person_id"] is None:
                        cluster_assignments.append((fid, person_id))
                        faces_assigned_this_cluster += 1
                    else:
                        # Face was assigned by another process, skip it
                        print(f"Warning: Face {fid} was already assigned to person {face_row['person_id']}, skipping")
                
                print(f"Debug: Assigned {faces_assigned_this_cluster} faces to person '{person_name}' (ID: {person_id})")
            
            # Batch update all cluster assignments
            if cluster_assignments:
                print(f"Debug: Batch assigning {len(cluster_assignments)} faces to {persons_created} persons")
                await db_async.set_faces_person_batch_async(conn, cluster_assignments)
                print(f"✓ Successfully assigned {len(cluster_assignments)} faces to {persons_created} new persons")
            else:
                print(f"⚠ ERROR: Created {persons_created} persons but collected 0 face assignments!")
                print(f"⚠ This indicates a bug - faces should have been assigned to the created persons")
    else:
        persons_created = 0
    
    # Ensure all changes are committed before returning
    await conn.commit()
    
    total_clustered = matched_count + len(unmatched_indices)
    
    return ClusterFacesResponse(
        clustered_faces=total_clustered,
        persons_created=persons_created,
        threshold=threshold
    )


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
    min_score: float = Query(0.2, ge=0.0, le=1.0, description="Minimum similarity score (0.0-1.0)"),
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
    
    # Filter by minimum score and get top results
    valid_indices = np.where(sims >= min_score)[0]
    if len(valid_indices) == 0:
        return {"results": []}
    
    # Sort valid results by similarity (descending)
    sorted_valid = valid_indices[np.argsort(-sims[valid_indices])]
    # Take top limit results
    idx = sorted_valid[:limit]
    
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
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"}
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


@app.post("/api/faces/{face_id}/assign-person")
async def assign_face_person(
    face_id: int,
    data: AssignFacePersonRequest,
    conn: aiosqlite.Connection = Depends(get_db_async)
):
    """Assign a person to a face and update matching index for future auto-matching."""
    import faiss
    import numpy as np
    
    person_id = data.person_id
    
    # Verify face exists and get its embedding
    cur = await conn.cursor()
    await cur.execute("SELECT id, embedding FROM faces WHERE id=?", (face_id,))
    face_row = await cur.fetchone()
    if not face_row:
        raise HTTPException(status_code=404, detail="face_not_found")
    
    # Verify person exists
    await cur.execute("SELECT id, name FROM persons WHERE id=?", (person_id,))
    person_row = await cur.fetchone()
    if not person_row:
        raise HTTPException(status_code=404, detail="person_not_found")
    
    person_name = person_row["name"]
    face_embedding_bytes = face_row["embedding"]
    
    # Assign person to face in database first
    await db_async.set_face_person_async(conn, face_id, person_id)
    
    # Update FAISS index to include this newly assigned face for future matching
    if face_embedding_bytes:
        try:
            index_manager = FAISSIndexManager()
            existing_index_name = "existing_person_faces"
            
            # Load existing index
            existing_index = index_manager.load_index(existing_index_name)
            
            if existing_index is not None:
                # Index exists - update it incrementally with new face
                face_embedding = np.frombuffer(face_embedding_bytes, dtype=np.float32)
                face_embedding = face_embedding.reshape(1, -1).astype(np.float32)
                
                # Normalize embedding for cosine similarity
                faiss.normalize_L2(face_embedding)
                
                # Add to existing index
                existing_index.add(face_embedding)
                
                # Save updated index
                index_manager.save_index(existing_index, existing_index_name)
                print(f"✓ Updated FAISS index: Added face {face_id} to person {person_id} ({person_name}) for future matching")
            else:
                # Index doesn't exist yet - will be built on next clustering
                # This is fine, the face will be included when index is built
                print(f"ℹ FAISS index not found - will be built on next clustering. Face {face_id} assigned to person {person_id} ({person_name})")
        except Exception as e:
            # Don't fail the assignment if index update fails
            print(f"Warning: Failed to update FAISS index after manual assignment: {e}")
            # Assignment still succeeded in database
    else:
        # Face has no embedding, can't update index but assignment succeeded
        print(f"Warning: Face {face_id} has no embedding, skipping index update")
    
    return StatusResponse(status="ok")


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
    """Merge source person into target person and update FAISS index if needed."""
    source_person_id = data.source_person_id
    
    if not isinstance(source_person_id, int):
        raise HTTPException(status_code=400, detail="missing_source_person_id")
    
    if source_person_id == target_person_id:
        raise HTTPException(status_code=400, detail="cannot_merge_same_person")
    
    try:
        # Get target person name for logging
        cur = await conn.cursor()
        await cur.execute("SELECT name FROM persons WHERE id=?", (target_person_id,))
        target_person_row = await cur.fetchone()
        target_person_name = target_person_row["name"] if target_person_row else f"Person #{target_person_id}"
        
        # Get count of faces being merged
        await cur.execute("SELECT COUNT(*) as count FROM faces WHERE person_id=?", (source_person_id,))
        count_row = await cur.fetchone()
        faces_count = count_row["count"] if count_row else 0
        
        # Perform the merge in database
        await db_async.merge_persons_async(conn, source_person_id, target_person_id)
        await conn.commit()
        
        # Note: FAISS index doesn't need updating because:
        # 1. Face embeddings haven't changed (only person_id in database changed)
        # 2. FAISS index stores embeddings, person_id is in database
        # 3. Next clustering will read updated person_id from database
        # The index will work correctly on next use since it reads person_id from DB
        if faces_count > 0:
            print(f"✓ Merged {faces_count} faces from person {source_person_id} to {target_person_id} ({target_person_name}). FAISS index will reflect changes on next clustering.")
        
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

