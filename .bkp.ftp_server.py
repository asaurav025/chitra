#!/usr/bin/env python3
"""
Chitra FTP Server

A lightweight FTP server for serving and receiving media files.
Runs alongside the Flask API, sharing the same photos directory.

Usage:
    python ftp_server.py

Environment Variables:
    FTP_PORT          - Server port (default: 2121)
    FTP_USER          - Username (default: chitra)
    FTP_PASS          - Password (required)
    FTP_PASSIVE_PORTS - Passive port range (default: 60000-60100)
    CHITRA_PHOTO_ROOT - Photos directory (default: ./photos)
"""

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path

from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.handlers import FTPHandler
from pyftpdlib.servers import FTPServer

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

# Server settings
FTP_PORT = int(os.environ.get("FTP_PORT", 2121))
FTP_HOST = os.environ.get("FTP_HOST", "0.0.0.0")

# Passive mode port range (for NAT/firewall traversal)
PASSIVE_PORTS_STR = os.environ.get("FTP_PASSIVE_PORTS", "60000-60100")
try:
    start, end = PASSIVE_PORTS_STR.split("-")
    PASSIVE_PORTS = range(int(start), int(end) + 1)
except ValueError:
    PASSIVE_PORTS = range(60000, 60101)

# User credentials
FTP_USER = os.environ.get("FTP_USER", "chitra")
FTP_PASS = os.environ.get("FTP_PASS", "")

# Directory paths
PHOTO_ROOT = Path(os.environ.get("CHITRA_PHOTO_ROOT", "photos")).resolve()
UPLOAD_DIR = PHOTO_ROOT / "uploads"
THUMB_DIR = PHOTO_ROOT / ".thumbs"
FACES_DIR = PHOTO_ROOT / ".faces"

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("chitra-ftp")


# -----------------------------------------------------------------------------
# CUSTOM HANDLER
# -----------------------------------------------------------------------------

class ChitraFTPHandler(FTPHandler):
    """Custom FTP handler with logging for uploads."""

    def on_file_received(self, file):
        """Called when a file is fully received."""
        logger.info(f"File uploaded: {file}")

    def on_file_sent(self, file):
        """Called when a file is fully sent."""
        logger.info(f"File downloaded: {file}")

    def on_incomplete_file_received(self, file):
        """Called when a file upload is incomplete."""
        logger.warning(f"Incomplete upload (removing): {file}")
        # Remove incomplete uploads
        try:
            os.remove(file)
        except OSError:
            pass


# -----------------------------------------------------------------------------
# SERVER SETUP
# -----------------------------------------------------------------------------

def create_directories():
    """Ensure all required directories exist."""
    PHOTO_ROOT.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    FACES_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Photo root: {PHOTO_ROOT}")
    logger.info(f"Upload directory: {UPLOAD_DIR}")


def create_server() -> FTPServer:
    """Create and configure the FTP server."""
    
    if not FTP_PASS:
        logger.error("FTP_PASS environment variable is required!")
        logger.error("Set it with: export FTP_PASS=your_password")
        sys.exit(1)

    create_directories()

    # Create authorizer with user permissions
    authorizer = DummyAuthorizer()

    # Main user with full access to photo root
    # Permissions:
    #   e = change directory (CWD, CDUP)
    #   l = list files (LIST, NLST, STAT, MLSD, MLST, SIZE)
    #   r = retrieve file (RETR)
    #   a = append data to file (APPE)
    #   d = delete file or directory (DELE, RMD)
    #   f = rename file or directory (RNFR, RNTO)
    #   m = create directory (MKD)
    #   w = store file (STOR, STOU)
    #   M = change file mode / permission (SITE CHMOD) - not available on Windows
    authorizer.add_user(
        username=FTP_USER,
        password=FTP_PASS,
        homedir=str(PHOTO_ROOT),
        perm="elradfmw",  # Full read/write access
    )

    # Optional: Add anonymous read-only user for browsing
    # authorizer.add_anonymous(str(PHOTO_ROOT), perm="elr")

    # Configure handler
    handler = ChitraFTPHandler
    handler.authorizer = authorizer
    handler.passive_ports = PASSIVE_PORTS

    # Banner message
    handler.banner = "Welcome to Chitra FTP Server. Ready for media transfers."

    # Timeout settings
    handler.timeout = 300  # 5 minutes idle timeout

    # Create server
    server = FTPServer((FTP_HOST, FTP_PORT), handler)

    # Server-level settings
    server.max_cons = 50  # Max simultaneous connections
    server.max_cons_per_ip = 10  # Max connections per IP

    return server


def main():
    """Run the FTP server."""
    logger.info("=" * 60)
    logger.info("Chitra FTP Server")
    logger.info("=" * 60)

    server = create_server()

    logger.info(f"Server starting on {FTP_HOST}:{FTP_PORT}")
    logger.info(f"Passive ports: {PASSIVE_PORTS.start}-{PASSIVE_PORTS.stop - 1}")
    logger.info(f"User: {FTP_USER}")
    logger.info("-" * 60)
    logger.info("Directory structure:")
    logger.info(f"  Root:      {PHOTO_ROOT}")
    logger.info(f"  Uploads:   {UPLOAD_DIR}")
    logger.info(f"  Thumbs:    {THUMB_DIR}")
    logger.info(f"  Faces:     {FACES_DIR}")
    logger.info("-" * 60)
    logger.info("Press Ctrl+C to stop the server")
    logger.info("=" * 60)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.close_all()


if __name__ == "__main__":
    main()

