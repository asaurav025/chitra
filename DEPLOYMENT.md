# Chitra Deployment Guide

## Quick Start

### Single File Startup

The simplest way to start the service:

```bash
python3 run.py
```

Or with custom options:

```bash
python3 run.py --port 8000 --reload  # Development mode
python3 run.py --host 0.0.0.0 --port 5000 --workers 4  # Production mode
```

### Using Startup Script

```bash
./start_server.sh
```

## Prerequisites

1. **Python 3.12+** with dependencies installed:
   ```bash
   pip install -r requirements.txt
   ```

2. **MinIO Server** (standalone installation)
   - Default connection: `localhost:9000`
   - Default credentials: `minioadmin` / `minioadmin`
   - Bucket: `chitra-photos` (created automatically)

3. **Redis** (for background jobs - optional but recommended)
   - Default connection: `localhost:6379`

## Environment Variables

Set these environment variables before starting:

```bash
# MinIO Configuration (required)
export MINIO_ENDPOINT=localhost:9000
export MINIO_ACCESS_KEY=minioadmin
export MINIO_SECRET_KEY=minioadmin
export MINIO_SECURE=false
export MINIO_BUCKET=chitra-photos

# Database Configuration (optional)
export CHITRA_DB_PATH=/path/to/photo.db  # Default: ./photo.db

# Server Configuration (optional)
export HOST=0.0.0.0
export PORT=5000
export WORKERS=4
export LOG_LEVEL=info
export RELOAD=false  # Set to 'true' for development

# Redis Configuration (optional, for background jobs)
export REDIS_URL=redis://localhost:6379/0

# Authentication Configuration (required for production)
export JWT_SECRET_KEY=your-secret-key-change-in-production  # Use a strong random secret
export JWT_ALGORITHM=HS256
export JWT_ACCESS_TOKEN_EXPIRE_MINUTES=1440  # 24 hours
```

## Running the Service

### Development Mode

```bash
# Auto-reload enabled
python3 run.py --reload

# Or set environment variable
export RELOAD=true
python3 run.py
```

### Production Mode

```bash
# Single worker
python3 run.py

# Multiple workers
python3 run.py --workers 4

# Or set environment variable
export WORKERS=4
python3 run.py
```

### Using Startup Script

```bash
# Development
RELOAD=true ./start_server.sh

# Production
WORKERS=4 ./start_server.sh
```

## Verifying the Service

Once started, verify it's running:

```bash
# Health check
curl http://localhost:5000/api/health

# API documentation
open http://localhost:5000/docs
```

## Background Jobs

If you're using background jobs (embeddings, face detection), make sure:

1. **Redis is running**:
   ```bash
   redis-server
   ```

2. **Worker process is running** (in separate terminal):
   ```bash
   python3 worker.py
   ```

## Docker Deployment (Optional)

If you prefer Docker:

```bash
# Build image
docker build -t chitra-api .

# Run container
docker run -d \
  -p 5000:5000 \
  -e MINIO_ENDPOINT=your-minio-host:9000 \
  -e MINIO_ACCESS_KEY=your-key \
  -e MINIO_SECRET_KEY=your-secret \
  -v $(pwd)/data:/app/data \
  chitra-api
```

## Troubleshooting

### MinIO Connection Issues

```bash
# Test MinIO connection
python3 -c "from core.storage_client import MinIOStorageClient; c = MinIOStorageClient(); print('Connected!')"
```

### Port Already in Use

```bash
# Use different port
python3 run.py --port 8000
```

### Database Issues

```bash
# Check database path
export CHITRA_DB_PATH=/path/to/photo.db
python3 run.py
```

## Production Checklist

- [ ] Set secure MinIO credentials
- [ ] Set `MINIO_SECURE=true` for HTTPS
- [ ] Configure proper `CHITRA_DB_PATH`
- [ ] Set appropriate `WORKERS` count
- [ ] Set `LOG_LEVEL=info` or `warning`
- [ ] Ensure Redis is running for background jobs
- [ ] Set up reverse proxy (nginx, etc.) if needed
- [ ] Configure firewall rules
- [ ] Set up monitoring and logging

