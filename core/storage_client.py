"""
MinIO/S3-compatible storage client.
Replaces FTP client with object storage.
"""
import os
import io
import asyncio
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta

from minio import Minio
from minio.error import S3Error


class MinIOStorageClient:
    """
    Client for MinIO object storage (S3-compatible).
    All file operations go through this client.
    No local file storage - everything is on MinIO server.
    
    Features:
    - Sync and async operations
    - Automatic bucket creation
    - Pre-signed URL generation
    - Path generation (same as FTP client)
    """
    
    def __init__(self):
        self.endpoint = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
        self.access_key = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
        self.secret_key = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
        # Explicitly handle MINIO_SECURE - default to False (HTTP)
        minio_secure = os.environ.get("MINIO_SECURE", "false")
        if isinstance(minio_secure, str):
            self.secure = minio_secure.lower() in ("true", "1", "yes")
        else:
            self.secure = bool(minio_secure)
        self.bucket_name = os.environ.get("MINIO_BUCKET", "chitra-photos")
        
        # Debug output (can be removed in production)
        # Always show config in worker context to help debug credential issues
        debug_minio = os.environ.get("DEBUG_MINIO", "false").lower() == "true"
        if debug_minio or os.environ.get("RQ_WORKER", "false").lower() == "true":
            print(f"MinIO Config: endpoint={self.endpoint}, secure={self.secure}, bucket={self.bucket_name}")
        
        # Initialize MinIO client
        self.client = Minio(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.secure
        )
        
        # Ensure bucket exists
        self._ensure_bucket()
    
    def _ensure_bucket(self):
        """Create bucket if it doesn't exist."""
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                print(f"Created MinIO bucket: {self.bucket_name}")
        except S3Error as e:
            if e.code == "SignatureDoesNotMatch":
                raise Exception(
                    f"MinIO authentication failed. Check your credentials:\n"
                    f"  Endpoint: {self.endpoint}\n"
                    f"  Access Key: {self.access_key[:4]}... (from MINIO_ACCESS_KEY env var)\n"
                    f"  Secret Key: {'*' * len(self.secret_key)} (from MINIO_SECRET_KEY env var)\n"
                    f"  Secure: {self.secure}\n"
                    f"Error: {e}"
                )
            raise Exception(f"Failed to create bucket '{self.bucket_name}': {e}")
        except Exception as e:
            raise Exception(f"Failed to create bucket '{self.bucket_name}': {e}")
    
    def _get_content_type(self, path: str) -> str:
        """Get content type from file extension."""
        ext = Path(path).suffix.lower()
        content_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.arw': 'image/x-sony-arw',
            '.heic': 'image/heic',
            '.heif': 'image/heif',
        }
        return content_types.get(ext, 'application/octet-stream')
    
    def upload_file(self, file_data: bytes, remote_path: str) -> str:
        """
        Upload file to MinIO (sync).
        remote_path: Object key (e.g., 'photos/2024/12/image.jpg')
        Returns: Object key for storage in database
        """
        try:
            file_obj = io.BytesIO(file_data)
            self.client.put_object(
                self.bucket_name,
                remote_path,
                file_obj,
                length=len(file_data),
                content_type=self._get_content_type(remote_path)
            )
            return remote_path
        except Exception as e:
            raise Exception(f"MinIO upload failed for '{remote_path}': {e}")
    
    async def upload_file_async(self, file_data: bytes, remote_path: str) -> str:
        """
        Upload file to MinIO (async).
        Runs sync upload in thread pool.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.upload_file, file_data, remote_path)
    
    def download_file(self, remote_path: str) -> bytes:
        """
        Download file from MinIO (sync).
        remote_path: Object key stored in database
        """
        try:
            response = self.client.get_object(self.bucket_name, remote_path)
            data = response.read()
            response.close()
            response.release_conn()
            return data
        except Exception as e:
            # Handle S3Error from minio library
            from minio.error import S3Error
            if isinstance(e, S3Error) and e.code == "NoSuchKey":
                raise FileNotFoundError(f"File not found in MinIO: {remote_path}")
            # Re-raise FileNotFoundError as-is
            if isinstance(e, FileNotFoundError):
                raise
            raise Exception(f"MinIO download failed for '{remote_path}': {e}")
    
    async def download_file_async(self, remote_path: str) -> bytes:
        """
        Download file from MinIO (async).
        Runs sync download in thread pool.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.download_file, remote_path)
    
    def file_exists(self, remote_path: str) -> bool:
        """Check if file exists in MinIO (sync)."""
        try:
            self.client.stat_object(self.bucket_name, remote_path)
            return True
        except S3Error:
            return False
    
    async def file_exists_async(self, remote_path: str) -> bool:
        """Check if file exists in MinIO (async)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.file_exists, remote_path)
    
    def delete_file(self, remote_path: str) -> bool:
        """Delete file from MinIO."""
        try:
            self.client.remove_object(self.bucket_name, remote_path)
            return True
        except Exception as e:
            print(f"Failed to delete file '{remote_path}': {e}")
            return False
    
    async def delete_file_async(self, remote_path: str) -> bool:
        """Delete file from MinIO (async)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.delete_file, remote_path)
    
    def generate_presigned_url(self, remote_path: str, expires_seconds: int = 3600) -> str:
        """
        Generate pre-signed URL for temporary access.
        Useful for direct browser access without proxying through API.
        """
        try:
            url = self.client.presigned_get_object(
                self.bucket_name,
                remote_path,
                expires=timedelta(seconds=expires_seconds)
            )
            return url
        except Exception as e:
            raise Exception(f"Failed to generate presigned URL for '{remote_path}': {e}")
    
    def generate_photo_path(self, filename: str, photo_id: int = None) -> str:
        """
        Generate unique path for photo.
        Format: photos/{year}/{month}/{filename}
        Same logic as FTP client for compatibility.
        """
        now = datetime.now()
        year = now.year
        month = f"{now.month:02d}"
        
        # Use original filename, ensure uniqueness if needed
        if photo_id:
            # If photo_id available, use it for uniqueness
            ext = Path(filename).suffix
            name = Path(filename).stem
            filename = f"{name}_{photo_id}{ext}"
        
        return f"photos/{year}/{month}/{filename}"
    
    def generate_thumbnail_path(self, item_id: int, item_type: str = "photo") -> str:
        """
        Generate object key for thumbnail.
        Same logic as FTP client for compatibility.
        """
        if item_type == "photo":
            return f"thumbnails/photos/{item_id}.jpg"
        elif item_type == "face":
            return f"thumbnails/faces/face_{item_id}.jpg"
        else:
            return f"thumbnails/{item_type}/{item_id}.jpg"
    
    def test_connection(self) -> bool:
        """Test MinIO connection by checking bucket existence."""
        try:
            return self.client.bucket_exists(self.bucket_name)
        except Exception as e:
            print(f"MinIO connection test failed: {e}")
            return False

