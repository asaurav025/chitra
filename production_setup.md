# Chitra Production Setup Guide

This guide covers setting up the Chitra backend service for production with systemd auto-start on server reboot.

## Overview

The production setup consists of:
- **Main API Service**: FastAPI application running with uvicorn (4 workers)
- **Worker Service**: 4 RQ workers for background job processing
- **Systemd Services**: Auto-start on server reboot

## Files Structure

```
/home/saurav/services/chitra/
├── start_production.sh      # Main API startup script
├── start_workers.sh         # Workers startup script
├── stop_workers.sh          # Workers stop script
├── .env.production          # Production environment variables
├── logs/                    # Application logs directory
└── ecosystem.config.js      # (Optional) PM2 config
```

## Step 1: Create Production Start Scripts

### 1.1 Main API Service Script

Create `/home/saurav/services/chitra/start_production.sh`:

```bash
#!/bin/bash
# Production startup script for Chitra FastAPI main service

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load environment variables
if [ -f .env.production ]; then
    export $(cat .env.production | grep -v '^#' | xargs)
fi

# Default values
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5000}"
WORKERS="${WORKERS:-4}"
LOG_LEVEL="${LOG_LEVEL:-info}"

# Create logs directory if it doesn't exist
mkdir -p logs

# Determine Python interpreter (use venv if available)
if [ -f ".venv/bin/uvicorn" ]; then
    UVICORN_CMD=".venv/bin/uvicorn"
    PYTHON_CMD=".venv/bin/python3"
elif [ -f "venv/bin/uvicorn" ]; then
    UVICORN_CMD="venv/bin/uvicorn"
    PYTHON_CMD="venv/bin/python3"
else
    UVICORN_CMD="uvicorn"
    PYTHON_CMD="python3"
fi

echo "Starting Chitra API in production mode..."
echo "Host: $HOST"
echo "Port: $PORT"
echo "Workers: $WORKERS"
echo "Log Level: $LOG_LEVEL"

# Start uvicorn with multiple workers
exec $UVICORN_CMD app_fastapi:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers "$WORKERS" \
    --log-level "$LOG_LEVEL" \
    --access-logfile logs/access.log \
    --error-logfile logs/error.log \
    --timeout-keep-alive 120 \
    --timeout-graceful-shutdown 30
```

### 1.2 Workers Startup Script

Create `/home/saurav/services/chitra/start_workers.sh`:

```bash
#!/bin/bash
# Production startup script for Chitra RQ workers (starts 4 workers)

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load environment variables
if [ -f .env.production ]; then
    export $(cat .env.production | grep -v '^#' | xargs)
fi

# Number of workers
WORKER_COUNT="${WORKER_COUNT:-4}"

# Create logs directory if it doesn't exist
mkdir -p logs

# Determine Python interpreter (use venv if available)
if [ -f ".venv/bin/python3" ]; then
    PYTHON_CMD=".venv/bin/python3"
elif [ -f "venv/bin/python3" ]; then
    PYTHON_CMD="venv/bin/python3"
else
    PYTHON_CMD="python3"
fi

echo "Starting $WORKER_COUNT Chitra RQ workers..."

# Start workers in background and save PIDs
for i in $(seq 1 $WORKER_COUNT); do
    echo "Starting worker $i..."
    $PYTHON_CMD worker.py > "logs/worker_$i.log" 2>&1 &
    echo $! > "logs/worker_$i.pid"
    echo "Worker $i started with PID $(cat logs/worker_$i.pid)"
done

echo "All $WORKER_COUNT workers started"
echo "Worker PIDs:"
cat logs/worker_*.pid | xargs echo
```

### 1.3 Workers Stop Script

Create `/home/saurav/services/chitra/stop_workers.sh`:

```bash
#!/bin/bash
# Stop all Chitra RQ workers

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Stopping Chitra RQ workers..."

# Kill workers by PID files
if [ -d logs ]; then
    for pidfile in logs/worker_*.pid; do
        if [ -f "$pidfile" ]; then
            pid=$(cat "$pidfile")
            if ps -p "$pid" > /dev/null 2>&1; then
                echo "Stopping worker with PID $pid..."
                kill "$pid" || true
            fi
            rm -f "$pidfile"
        fi
    done
fi

# Also kill any remaining worker processes
pkill -f "worker.py" || true

echo "All workers stopped"
```

Make scripts executable:

```bash
cd /home/saurav/services/chitra
chmod +x start_production.sh start_workers.sh stop_workers.sh
```

## Step 2: Create Production Environment File

Create `/home/saurav/services/chitra/.env.production`:

```bash
# Production environment variables for Chitra

# Server Configuration
HOST=0.0.0.0
PORT=5000
WORKERS=4
LOG_LEVEL=info

# MinIO Configuration
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_SECURE=false
MINIO_BUCKET=chitra-photos

# Database Configuration
CHITRA_DB_PATH=/home/saurav/services/chitra/photo.db

# Redis Configuration (for RQ)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Worker Configuration
WORKER_COUNT=4
```

**Important**: Update the values in `.env.production` with your actual production credentials and paths.

## Step 3: Create Logs Directory

```bash
cd /home/saurav/services/chitra
mkdir -p logs
```

## Step 4: Create Systemd Service Files

### 4.1 Main API Service

Create `/etc/systemd/system/chitra-api.service`:

```ini
[Unit]
Description=Chitra FastAPI Backend Service
After=network.target redis.service postgresql.service
Wants=redis.service

[Service]
Type=simple
User=saurav
Group=saurav
WorkingDirectory=/home/saurav/services/chitra
Environment="PATH=/home/saurav/services/chitra/.venv/bin:/usr/local/bin:/usr/bin:/bin"
EnvironmentFile=/home/saurav/services/chitra/.env.production

# Main service command
ExecStart=/home/saurav/services/chitra/start_production.sh

# Restart configuration
Restart=always
RestartSec=10
StartLimitInterval=200
StartLimitBurst=5

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=chitra-api

# Security
NoNewPrivileges=true
PrivateTmp=true

# Resource limits
LimitNOFILE=65536
MemoryMax=4G
CPUQuota=200%

[Install]
WantedBy=multi-user.target
```

### 4.2 Workers Service

Create `/etc/systemd/system/chitra-workers.service`:

```ini
[Unit]
Description=Chitra RQ Workers Service (4 workers)
After=network.target redis.service chitra-api.service
Wants=redis.service
Requires=chitra-api.service

[Service]
Type=forking
User=saurav
Group=saurav
WorkingDirectory=/home/saurav/services/chitra
Environment="PATH=/home/saurav/services/chitra/.venv/bin:/usr/local/bin:/usr/bin:/bin"
EnvironmentFile=/home/saurav/services/chitra/.env.production
Environment="WORKER_COUNT=4"

# Start workers
ExecStart=/home/saurav/services/chitra/start_workers.sh

# Stop workers
ExecStop=/home/saurav/services/chitra/stop_workers.sh

# Restart configuration
Restart=always
RestartSec=10
StartLimitInterval=200
StartLimitBurst=5

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=chitra-workers

# Security
NoNewPrivileges=true
PrivateTmp=true

# Resource limits
LimitNOFILE=65536
MemoryMax=8G
CPUQuota=400%

[Install]
WantedBy=multi-user.target
```

## Step 5: Install and Enable Systemd Services

```bash
# Copy service files (if you created them in the chitra directory)
sudo cp /home/saurav/services/chitra/chitra-api.service /etc/systemd/system/
sudo cp /home/saurav/services/chitra/chitra-workers.service /etc/systemd/system/

# Or create them directly in /etc/systemd/system/

# Reload systemd daemon
sudo systemctl daemon-reload

# Enable services to start on boot
sudo systemctl enable chitra-api.service
sudo systemctl enable chitra-workers.service
```

## Step 6: Start Services

```bash
# Start the main API service
sudo systemctl start chitra-api

# Start the workers service
sudo systemctl start chitra-workers

# Check status
sudo systemctl status chitra-api
sudo systemctl status chitra-workers
```

## Step 7: Verify Services

### Check Service Status

```bash
# Check if services are running
sudo systemctl is-active chitra-api
sudo systemctl is-active chitra-workers

# Check if services are enabled (will start on boot)
sudo systemctl is-enabled chitra-api
sudo systemctl is-enabled chitra-workers
```

### Check Logs

```bash
# Systemd journal logs
sudo journalctl -u chitra-api -f
sudo journalctl -u chitra-workers -f

# Application logs
tail -f /home/saurav/services/chitra/logs/access.log
tail -f /home/saurav/services/chitra/logs/error.log
tail -f /home/saurav/services/chitra/logs/worker_*.log

# View last 100 lines
sudo journalctl -u chitra-api -n 100
sudo journalctl -u chitra-workers -n 100
```

### Test API Endpoint

```bash
# Test health endpoint (adjust port if different)
curl http://localhost:5000/health

# Or test root endpoint
curl http://localhost:5000/
```

## Step 8: Common Operations

### Restart Services

```bash
sudo systemctl restart chitra-api
sudo systemctl restart chitra-workers
```

### Stop Services

```bash
sudo systemctl stop chitra-api
sudo systemctl stop chitra-workers
```

### Disable Auto-Start (if needed)

```bash
sudo systemctl disable chitra-api
sudo systemctl disable chitra-workers
```

### View Service Status

```bash
sudo systemctl status chitra-api
sudo systemctl status chitra-workers
```

## Step 9: Cloudflare Tunnel Configuration

Since you're using Cloudflare Tunnel, configure it to route traffic to your services:

1. **Install Cloudflare Tunnel** (if not already installed):
   ```bash
   # For Linux
   wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
   sudo dpkg -i cloudflared-linux-amd64.deb
   ```

2. **Authenticate**:
   ```bash
   cloudflared tunnel login
   ```

3. **Create Tunnel Configuration** (`~/.cloudflared/config.yml`):
   ```yaml
   tunnel: <your-tunnel-id>
   credentials-file: /home/saurav/.cloudflared/<tunnel-id>.json

   ingress:
     - hostname: api.yourdomain.com
       service: http://localhost:5000
     - hostname: yourdomain.com
       service: http://localhost:5173  # Frontend (will configure later)
     - service: http_status:404
   ```

4. **Run Tunnel as Systemd Service** (optional, for auto-start):
   ```bash
   sudo cloudflared service install
   sudo systemctl enable cloudflared
   sudo systemctl start cloudflared
   ```

## Troubleshooting

### Service Fails to Start

1. Check service status:
   ```bash
   sudo systemctl status chitra-api
   ```

2. Check logs:
   ```bash
   sudo journalctl -u chitra-api -n 50
   ```

3. Verify environment file:
   ```bash
   cat /home/saurav/services/chitra/.env.production
   ```

4. Test script manually:
   ```bash
   cd /home/saurav/services/chitra
   ./start_production.sh
   ```

### Workers Not Starting

1. Check Redis is running:
   ```bash
   sudo systemctl status redis
   redis-cli ping  # Should return PONG
   ```

2. Check worker logs:
   ```bash
   tail -f /home/saurav/services/chitra/logs/worker_*.log
   ```

3. Verify worker script:
   ```bash
   cd /home/saurav/services/chitra
   ./start_workers.sh
   ```

### Port Already in Use

If port 5000 is already in use:

```bash
# Find process using port
sudo lsof -i :5000

# Kill the process or change PORT in .env.production
```

### Permission Issues

Ensure scripts are executable and user has proper permissions:

```bash
chmod +x /home/saurav/services/chitra/*.sh
chown -R saurav:saurav /home/saurav/services/chitra
```

## Resource Limits

Adjust resource limits in systemd service files if needed:

- **MemoryMax**: Maximum memory (currently 4G for API, 8G for workers)
- **CPUQuota**: CPU limit (200% = 2 cores for API, 400% = 4 cores for workers)
- **LimitNOFILE**: Maximum open file descriptors

## Security Notes

1. **Environment Variables**: Never commit `.env.production` to git
2. **File Permissions**: Ensure `.env.production` has restricted permissions:
   ```bash
   chmod 600 /home/saurav/services/chitra/.env.production
   ```
3. **Firewall**: Configure firewall to only allow necessary ports
4. **Updates**: Keep dependencies updated regularly

## Next Steps

After setting up the backend:
1. Configure frontend (`chitra_ui`) production setup
2. Set up Cloudflare Tunnel for both services
3. Configure monitoring and alerting
4. Set up backup strategy for database

## Support

For issues or questions:
- Check logs: `sudo journalctl -u chitra-api -f`
- Review application logs in `/home/saurav/services/chitra/logs/`
- Verify all dependencies are installed in virtual environment

