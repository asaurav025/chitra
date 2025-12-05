"""
FTP Client for accessing external FTP storage server.
This is the ONLY way chitra interacts with file storage.
All files are stored on independent FTP server, not locally.
"""
import ftplib
import io
from pathlib import Path
from typing import Optional
import os
from datetime import datetime


class FTPStorageClient:
    """
    Client for external FTP storage server.
    All file operations go through this client.
    No local file storage - everything is on FTP server.
    """
    
    def __init__(self):
        self.host = os.environ.get("FTP_STORAGE_HOST", "localhost")
        self.port = int(os.environ.get("FTP_STORAGE_PORT", "21"))
        self.user = os.environ.get("FTP_STORAGE_USER", "ftpstorage")
        self.password = os.environ.get("FTP_STORAGE_PASS", "")
        # Default to empty string (use FTP user's home directory)
        # Can be set to relative path like "storage" or absolute if not chrooted
        base_path = os.environ.get("FTP_STORAGE_BASE", "")
        self.base_path = base_path.strip() if base_path else ""
        self._ftp = None
    
    def _connect(self) -> ftplib.FTP:
        """Get or create FTP connection"""
        if self._ftp is None:
            self._ftp = ftplib.FTP()
            self._ftp.connect(self.host, self.port)
            self._ftp.login(self.user, self.password)
            # Set to passive mode for better compatibility
            self._ftp.set_pasv(True)
        return self._ftp
    
    def _disconnect(self):
        """Close FTP connection"""
        if self._ftp:
            try:
                self._ftp.quit()
            except:
                try:
                    self._ftp.close()
                except:
                    pass
            self._ftp = None
    
    def _ensure_directory(self, remote_path: str):
        """Ensure directory structure exists on FTP server"""
        ftp = self._connect()
        
        # Navigate to base path if specified
        if self.base_path:
            try:
                ftp.cwd(self.base_path)
            except:
                # Base path might not exist, try to create it
                try:
                    ftp.mkd(self.base_path)
                    ftp.cwd(self.base_path)
                except:
                    pass
        
        # Create directory structure for the file
        path_parts = Path(remote_path).parent.parts
        for part in path_parts:
            if part:
                try:
                    ftp.cwd(part)
                except:
                    try:
                        ftp.mkd(part)
                        ftp.cwd(part)
                    except:
                        pass
    
    def upload_file(self, file_data: bytes, remote_path: str) -> str:
        """
        Upload file to FTP server.
        remote_path: Path relative to FTP base (e.g., 'photos/2024/12/image.jpg')
        Returns: Relative path for storage in database
        """
        ftp = self._connect()
        
        # Ensure directory exists
        self._ensure_directory(remote_path)
        
        # Navigate to base if specified
        if self.base_path:
            try:
                ftp.cwd(self.base_path)
            except:
                pass
        
        # Construct full path for STOR command
        if self.base_path:
            full_remote_path = f"{self.base_path}/{remote_path}"
        else:
            full_remote_path = remote_path
        
        file_obj = io.BytesIO(file_data)
        
        try:
            ftp.storbinary(f'STOR {full_remote_path}', file_obj)
        except Exception as e:
            self._disconnect()
            raise Exception(f"FTP upload failed: {str(e)}")
        
        return remote_path  # Return relative path for database
    
    def download_file(self, remote_path: str) -> bytes:
        """
        Download file from FTP server.
        remote_path: Path stored in database (relative to base)
        """
        ftp = self._connect()
        
        # Construct full path for RETR command
        if self.base_path:
            full_path = f"{self.base_path}/{remote_path}"
        else:
            full_path = remote_path
        
        file_obj = io.BytesIO()
        try:
            ftp.retrbinary(f'RETR {full_path}', file_obj.write)
        except ftplib.error_perm as e:
            self._disconnect()
            raise FileNotFoundError(f"File not found on FTP: {remote_path}") from e
        except Exception as e:
            self._disconnect()
            raise Exception(f"FTP download failed: {str(e)}") from e
        
        file_obj.seek(0)
        return file_obj.read()
    
    def file_exists(self, remote_path: str) -> bool:
        """Check if file exists on FTP server"""
        try:
            ftp = self._connect()
            # Construct full path
            if self.base_path:
                full_path = f"{self.base_path}/{remote_path}"
            else:
                full_path = remote_path
            ftp.size(full_path)
            return True
        except:
            return False
    
    def delete_file(self, remote_path: str) -> bool:
        """Delete file from FTP server"""
        try:
            ftp = self._connect()
            # Construct full path
            if self.base_path:
                full_path = f"{self.base_path}/{remote_path}"
            else:
                full_path = remote_path
            ftp.delete(full_path)
            return True
        except:
            return False
    
    def generate_photo_path(self, filename: str, photo_id: int = None) -> str:
        """
        Generate unique path for photo.
        Format: photos/{year}/{month}/{filename}
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
    
    def generate_thumbnail_path(self, photo_id: int, thumb_type: str = "photo") -> str:
        """Generate path for thumbnail"""
        if thumb_type == "photo":
            return f"thumbnails/photos/{photo_id}.jpg"
        elif thumb_type == "face":
            return f"thumbnails/faces/face_{photo_id}.jpg"
        else:
            return f"thumbnails/{thumb_type}/{photo_id}.jpg"
    
    def __del__(self):
        """Cleanup on destruction"""
        self._disconnect()

