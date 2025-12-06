#!/bin/bash
# Start RQ worker with MinIO environment variables
# Make sure to set these before running:
#   export MINIO_ENDPOINT=localhost:9000
#   export MINIO_ACCESS_KEY=your_access_key
#   export MINIO_SECRET_KEY=your_secret_key
#   export MINIO_SECURE=false  # or true for HTTPS
#   export MINIO_BUCKET=chitra-photos

cd "$(dirname "$0")"

# Check if MinIO env vars are set
if [ -z "$MINIO_ACCESS_KEY" ] || [ -z "$MINIO_SECRET_KEY" ]; then
    echo "⚠ Warning: MINIO_ACCESS_KEY or MINIO_SECRET_KEY not set"
    echo "Using defaults (minioadmin/minioadmin)"
    echo ""
fi

# Enable debug output
export DEBUG_MINIO=true
export RQ_WORKER=true

echo "Starting RQ worker..."
echo "MinIO Endpoint: ${MINIO_ENDPOINT:-localhost:9000}"
echo "MinIO Access Key: ${MINIO_ACCESS_KEY:-minioadmin} (default)"
echo "MinIO Secure: ${MINIO_SECURE:-false}"
echo ""

python3 worker.py
