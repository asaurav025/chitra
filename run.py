#!/usr/bin/env python3
"""
Simple startup script for Chitra FastAPI application.
Run this file directly to start the server.

Usage:
    python3 run.py
    python3 run.py --port 8000
    python3 run.py --host 127.0.0.1 --port 5000 --reload
"""
import os
import sys
import argparse
import uvicorn


def main():
    """Start the FastAPI application."""
    parser = argparse.ArgumentParser(description='Start Chitra Photo Management API')
    parser.add_argument('--host', default=os.environ.get('HOST', '0.0.0.0'),
                        help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', '5000')),
                        help='Port to bind to (default: 5000)')
    parser.add_argument('--workers', type=int, default=int(os.environ.get('WORKERS', '1')),
                        help='Number of worker processes (default: 1)')
    parser.add_argument('--reload', action='store_true',
                        default=os.environ.get('RELOAD', 'false').lower() == 'true',
                        help='Enable auto-reload for development')
    parser.add_argument('--log-level', default=os.environ.get('LOG_LEVEL', 'info'),
                        choices=['critical', 'error', 'warning', 'info', 'debug', 'trace'],
                        help='Log level (default: info)')
    
    args = parser.parse_args()
    
    # Determine if we're in development mode
    is_dev = args.reload or os.environ.get('ENV', '').lower() == 'development'
    
    if is_dev:
        print(f"Starting Chitra API in development mode on {args.host}:{args.port}")
        uvicorn.run(
            "app_fastapi:app",
            host=args.host,
            port=args.port,
            reload=True,
            log_level=args.log_level,
            timeout_keep_alive=300,  # 5 minutes for large uploads
            limit_concurrency=100
        )
    else:
        print(f"Starting Chitra API in production mode on {args.host}:{args.port} with {args.workers} workers")
        uvicorn.run(
            "app_fastapi:app",
            host=args.host,
            port=args.port,
            workers=args.workers,
            log_level=args.log_level,
            access_log=True,
            timeout_keep_alive=300,  # 5 minutes for large uploads
            limit_concurrency=100
        )


if __name__ == '__main__':
    main()

