# Chitra Test Suite

Test suite for Chitra Photo Management API migration from Flask to FastAPI with MinIO storage.

## Test Structure

### Unit Tests

- **test_storage_client.py**: Tests for MinIO storage client operations
  - File upload/download
  - File existence checks
  - Path generation
  - Connection and bucket operations

- **test_db_async.py**: Tests for async database operations
  - Async connection handling
  - Photo upsert operations
  - Embedding storage and retrieval
  - Tag management

### Integration Tests

- **test_endpoints.py**: Tests for FastAPI endpoints
  - Health check endpoint
  - Photo listing and retrieval
  - Person management
  - Search functionality
  - Error handling

- **test_background_jobs.py**: Tests for background job processing
  - Job function signatures
  - Batch processing structure
  - Job execution flow

- **test_search.py**: Tests for search functionality
  - Text embedding generation
  - Search with embeddings
  - Empty result handling

## Running Tests

### Prerequisites

1. **MinIO Server**: Must be running for storage tests
   ```bash
   # Default connection: localhost:9000
   # Credentials: minioadmin/minioadmin
   ```

2. **Dependencies**: Install test dependencies
   ```bash
   pip install httpx  # For FastAPI TestClient
   ```

3. **Database**: Tests use temporary databases (auto-created/cleaned)

### Run All Tests

```bash
python3 tests/run_tests.py
```

### Run Specific Test Module

```bash
python3 tests/run_tests.py test_storage_client
python3 tests/run_tests.py test_db_async
python3 tests/run_tests.py test_endpoints
```

### Run with unittest directly

```bash
python3 -m unittest discover tests
python3 -m unittest tests.test_storage_client
```

## Test Environment Variables

Tests use default values if environment variables are not set:

- `MINIO_ENDPOINT`: Defaults to `localhost:9000`
- `MINIO_ACCESS_KEY`: Defaults to `minioadmin`
- `MINIO_SECRET_KEY`: Defaults to `minioadmin`
- `MINIO_BUCKET`: Defaults to `chitra-photos`

## Test Coverage

### ✅ Covered

- MinIO storage operations (upload, download, delete, exists)
- Async database operations
- FastAPI endpoint structure and responses
- Background job function signatures
- Search functionality basics

### ⚠️ Requires External Dependencies

Some tests are skipped if dependencies are unavailable:

- **Storage tests**: Skip if MinIO server not running
- **Embedding tests**: Skip if CLIP model not available
- **Face detection tests**: Skip if face detection model not available
- **Endpoint tests**: Skip if FastAPI TestClient not installed

## Notes

- Tests create temporary databases that are cleaned up automatically
- Tests use test buckets/prefixes to avoid conflicts with production data
- Some tests require actual ML models (CLIP, face detection) which may be large
- Background job tests verify structure but may skip actual execution if models unavailable

## Continuous Integration

For CI/CD pipelines:

```bash
# Install dependencies
pip install -r requirements.txt
pip install httpx pytest  # Optional: for extended testing

# Run tests
python3 tests/run_tests.py

# Or with pytest (if installed)
pytest tests/
```

