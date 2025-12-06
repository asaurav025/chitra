#!/bin/bash
# MinIO Setup Script for Chitra
# This script helps set up MinIO server for the migration

set -e

echo "=== MinIO Setup for Chitra ==="
echo ""

# Check if Docker is available
if command -v docker &> /dev/null; then
    echo "Docker detected. Setting up MinIO via Docker..."
    echo ""
    
    # Check if MinIO container already exists
    if docker ps -a --format '{{.Names}}' | grep -q "^minio$"; then
        echo "MinIO container already exists."
        read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Stopping and removing existing MinIO container..."
            docker stop minio 2>/dev/null || true
            docker rm minio 2>/dev/null || true
        else
            echo "Starting existing MinIO container..."
            docker start minio
            echo "MinIO is running at:"
            echo "  API: http://localhost:9000"
            echo "  Console: http://localhost:9001"
            echo "  Access Key: minioadmin"
            echo "  Secret Key: minioadmin"
            exit 0
        fi
    fi
    
    # Create data directory
    mkdir -p ./minio-data
    
    # Run MinIO container
    docker run -d \
        --name minio \
        -p 9000:9000 \
        -p 9001:9001 \
        -e "MINIO_ROOT_USER=minioadmin" \
        -e "MINIO_ROOT_PASSWORD=minioadmin" \
        -v "$(pwd)/minio-data:/data" \
        minio/minio server /data --console-address ":9001"
    
    echo ""
    echo "MinIO server started successfully!"
    echo ""
    echo "Access information:"
    echo "  API Endpoint: http://localhost:9000"
    echo "  Console URL: http://localhost:9001"
    echo "  Access Key: minioadmin"
    echo "  Secret Key: minioadmin"
    echo ""
    echo "Next steps:"
    echo "1. Open http://localhost:9001 in your browser"
    echo "2. Login with minioadmin/minioadmin"
    echo "3. Create a bucket named 'chitra-photos'"
    echo "4. Or run: docker exec minio mc alias set local http://localhost:9000 minioadmin minioadmin"
    echo "5. Then: docker exec minio mc mb local/chitra-photos"
    echo ""
    
elif command -v minio &> /dev/null; then
    echo "MinIO binary detected. Setting up MinIO server..."
    echo ""
    echo "To start MinIO server manually, run:"
    echo "  export MINIO_ROOT_USER=minioadmin"
    echo "  export MINIO_ROOT_PASSWORD=minioadmin"
    echo "  minio server ./minio-data --console-address :9001"
    echo ""
    echo "Or create a systemd service for automatic startup."
    echo ""
else
    echo "Neither Docker nor MinIO binary found."
    echo ""
    echo "Options:"
    echo "1. Install Docker and run this script again"
    echo "2. Download MinIO binary from https://min.io/download"
    echo "3. Use a cloud MinIO service"
    echo ""
    echo "For Docker installation:"
    echo "  Ubuntu/Debian: sudo apt-get install docker.io"
    echo "  Or visit: https://docs.docker.com/get-docker/"
    echo ""
    exit 1
fi

