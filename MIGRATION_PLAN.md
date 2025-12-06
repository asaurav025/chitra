# Migration Plan: Flask → FastAPI + MinIO

## Overview
Complete migration from Flask to FastAPI with full async support, replacing FTP storage with MinIO (S3-compatible object storage).

## Migration Strategy
- Full migration (replace Flask completely)
- Full async conversion (maximum performance)
- Maintain API compatibility
- Incremental implementation with checkpoints

---

## Phase 1: Infrastructure Setup

### 1.1 Install MinIO Server
- Set up MinIO via Docker or native installation
- Create bucket: `chitra-photos`
- Configure access keys and endpoint
- Test connectivity

### 1.2 Update Dependencies
**File:** `requirements.txt`

**Remove:**
- `Flask>=3.0.0`
- `flask-cors`
- `pyftpdlib>=1.5.9`

**Add:**
- `fastapi>=0.104.0`
- `uvicorn[standard]>=0.24.0`
- `python-multipart>=0.0.6` (for file uploads)
- `minio>=7.2.0`
- `aiosqlite>=0.19.0` (async SQLite)
- `aiofiles>=23.2.0` (async file operations)

---

## Phase 2: Create MinIO Storage Client

### 2.1 Create Storage Client
**File:** `core/storage_client.py` (new file, replaces `ftp_client.py`)

**Key Features:**
- MinIO client initialization
- Async upload/download methods
- File existence checks
- Pre-signed URL generation
- Path generation (same logic as FTP client)
- Error handling with proper exceptions

**Methods:**
- `__init__()` - Initialize MinIO client, ensure bucket exists
- `upload_file(file_data: bytes, remote_path: str) -> str` - Sync upload
- `upload_file_async(file_data: bytes, remote_path: str) -> str` - Async upload
- `download_file(remote_path: str) -> bytes` - Sync download
- `download_file_async(remote_path: str) -> bytes` - Async download
- `file_exists(remote_path: str) -> bool` - Sync check
- `file_exists_async(remote_path: str) -> bool` - Async check
- `delete_file(remote_path: str) -> bool` - Delete object
- `generate_presigned_url(remote_path: str, expires: int) -> str` - Generate temporary URL
- `generate_photo_path(filename: str, photo_id: int) -> str` - Same as FTP version
- `generate_thumbnail_path(item_id: int, item_type: str) -> str` - Same as FTP version

### 2.2 Environment Variables
**Add to `.env` or environment:**
```bash
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_SECURE=false
MINIO_BUCKET=chitra-photos
```

**Checkpoint:** Storage client created and tested

---

## Phase 3: Async Database Layer

### 3.1 Create Async Database Module
**File:** `core/db_async.py` (new file, parallel to `db.py`)

**Key Changes:**
- Use `aiosqlite` instead of `sqlite3`
- All functions become async
- Connection context manager for proper cleanup
- Connection pooling via async context managers

**Functions to Convert:**
- `connect_async(db_path: str) -> AsyncContextManager` - Async connection
- `init_db_async(db_path: str)` - Async table creation
- `upsert_photo_async(conn, **meta)` - Async photo upsert
- `put_embedding_async(conn, photo_id, vec_bytes, dim)` - Async embedding
- All other database functions converted to async

**Checkpoint:** Async database layer functional

---

## Phase 4: FastAPI Application Setup

### 4.1 Create FastAPI App
**File:** `app_fastapi.py` (new file, will replace `app.py`)

**Structure:**
- FastAPI app initialization
- CORS middleware
- Dependency injection setup
- Health check endpoint (test)

### 4.2 Dependency Injection
- Database connection: `get_db_async()`
- Storage client: `get_storage_client()`
- Embedder: `get_embedder()` (can stay sync, use thread pool)

**Checkpoint:** FastAPI app runs and health check works

---

## Phase 5: Endpoint Migration (24 endpoints)

### 5.1 Health Check
**Route:** `GET /api/health`
- Convert to FastAPI
- Test MinIO connectivity instead of FTP
- Return JSON response

### 5.2 Photo Endpoints (8 endpoints)
- `GET /api/photos` - List photos
- `GET /api/photos/{photo_id}` - Get photo details
- `GET /api/photos/{photo_id}/image` - Download image (async)
- `GET /api/photos/{photo_id}/thumbnail` - Get thumbnail (async, use cache)
- `POST /api/photos/upload` - Upload photos (async, handle multiple files)
- `POST /api/photos/scan-path` - Scan directory
- `GET /api/photos/{photo_id}/tags` - Get tags
- `POST /api/photos/{photo_id}/tags` - Add tags

### 5.3 Indexing Endpoints (3 endpoints)
- `POST /api/index/embeddings` - Queue embedding jobs
- `POST /api/index/faces` - Queue face detection jobs
- `POST /api/index/faces-cluster` - Cluster faces

### 5.4 Job Status
**Route:** `GET /api/jobs/{job_id}`

### 5.5 Search Endpoints (2 endpoints)
- `GET /api/search/photos` - Text search
- `GET /api/search/by-person` - Person search

### 5.6 Face Endpoints (3 endpoints)
- `GET /api/faces` - List faces
- `GET /api/faces/{face_id}/thumbnail` - Get face thumbnail (async)
- `POST /api/faces/{face_id}/assign-person` - Assign person to face

### 5.7 Person Endpoints (5 endpoints)
- `GET /api/persons` - List persons
- `POST /api/persons` - Create person
- `PUT /api/persons/{person_id}` - Update person
- `GET /api/persons/{person_id}/faces` - Get person's faces
- `POST /api/persons/{target_person_id}/merge` - Merge persons

### 5.8 Storage Endpoint
**Route:** `GET /api/storage/{file_path:path}`

### 5.9 Frontend Serving
**Route:** `GET /` and `GET /{path:path}`

**Checkpoint:** All endpoints migrated and tested

---

## Phase 6: Background Jobs Update

### 6.1 Update Job Functions
**File:** `core/jobs.py`

**Changes:**
- Replace `FTPStorageClient` with `MinIOStorageClient`
- Update `_get_ftp_client()` → `_get_storage_client()`
- Convert storage operations to async (or use sync with thread pool)

**Checkpoint:** Background jobs working with MinIO

---

## Phase 7: Pydantic Models

### 7.1 Request Models
**File:** `core/schemas.py` (new file)

### 7.2 Response Models
**File:** `core/schemas.py`

**Checkpoint:** All models defined and used

---

## Phase 8: Update Core Modules

### 8.1 Update Cache Module
**File:** `core/cache.py` - Keep as-is (works with FastAPI)

### 8.2 Update Extractor Module
**File:** `core/extractor.py` - Keep sync, use `run_in_executor`

### 8.3 Update Embedder Module
**File:** `core/embedder.py` - Keep sync, use `run_in_executor`

### 8.4 Update Face Module
**File:** `core/face.py` - Keep sync, use `run_in_executor`

**Checkpoint:** All core modules compatible

---

## Phase 9: Testing and Validation

### 9.1 Unit Tests
- Test MinIO client operations
- Test async database operations
- Test endpoint conversions

### 9.2 Integration Tests
- Test file upload/download flow
- Test background job processing
- Test search functionality

**Checkpoint:** All tests passing

---

## Phase 10: Migration and Deployment

### 10.1 Data Migration (Optional)
**Script:** `scripts/migrate_ftp_to_minio.py`

### 10.2 Deployment
- Update startup script to use `uvicorn`
- Update Dockerfile if using containers
- Update environment variables

**Checkpoint:** Application deployed and running

---

## Risk Mitigation

1. Keep Flask app as backup during migration
2. Test each endpoint before removing Flask version
3. Database schema unchanged (backward compatible)
4. MinIO can run alongside FTP during transition
5. Use feature flags if needed

---

## Success Criteria

- All 24 endpoints migrated and working
- Async operations functional
- MinIO storage working
- Background jobs processing correctly
- Performance improved (2-5x for concurrent operations)
- API documentation auto-generated at `/docs`
- All tests passing

