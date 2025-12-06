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
import threading
import time


class FTPStorageClient:
    """
    Client for external FTP storage server.
    All file operations go through this client.
    No local file storage - everything is on FTP server.
    
    Features:
    - Connection pooling: Reuses connections to avoid overhead
    - Thread-safe: Safe for concurrent uploads
    - Connection keep-alive: Maintains connections between operations
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
        
        # Connection management
        self._ftp = None
        self._lock = threading.Lock()  # Thread-safe connection management
        self._last_used = None  # Track last connection use for timeout
        self._connection_timeout = 300  # 5 minutes - disconnect idle connections
    
    def _connect(self) -> ftplib.FTP:
        """
        Get or create FTP connection with thread safety.
        Reuses existing connection if still valid.
        """
        with self._lock:
            # Check if we have a valid connection
            if self._ftp is not None:
                # Check if connection has timed out
                if self._last_used and (time.time() - self._last_used) > self._connection_timeout:
                    # Connection is too old, reconnect
                    try:
                        self._ftp.quit()
                    except:
                        try:
                            self._ftp.close()
                        except:
                            pass
                    self._ftp = None
                else:
                    # Test connection with a simple command to ensure it's alive
                    try:
                        self._ftp.voidcmd("NOOP")
                        # Connection is good
                        self._last_used = time.time()
                        return self._ftp
                    except (ftplib.error_temp, ftplib.error_perm, OSError, AttributeError, EOFError):
                        # Connection is dead, need to reconnect
                        self._ftp = None
            
            # Create new connection
            try:
                self._ftp = ftplib.FTP()
                self._ftp.connect(self.host, self.port, timeout=30)
                self._ftp.login(self.user, self.password)
                # Set to passive mode for better compatibility
                self._ftp.set_pasv(True)
                self._last_used = time.time()
                return self._ftp
            except Exception as e:
                # If connection fails, clear it
                self._ftp = None
                self._last_used = None
                raise
    
    def _disconnect(self):
        """Close FTP connection (thread-safe)"""
        with self._lock:
            if self._ftp:
                try:
                    self._ftp.quit()
                except:
                    try:
                        self._ftp.close()
                    except:
                        pass
                self._ftp = None
                self._last_used = None
    
    def _ensure_directory(self, remote_path: str):
        """Ensure directory structure exists on FTP server"""
        ftp = self._connect()
        
        # Start from root/home directory
        try:
            ftp.cwd("/")
        except:
            pass  # Some FTP servers don't support CWD /
        
        # Navigate to base path if specified, creating it if needed
        if self.base_path:
            try:
                ftp.cwd(self.base_path)
            except ftplib.error_perm:
                # Base path doesn't exist, try to create it
                try:
                    ftp.mkd(self.base_path)
                    ftp.cwd(self.base_path)
                except ftplib.error_perm as e:
                    raise Exception(f"Cannot create base directory '{self.base_path}': {str(e)}. Check FTP permissions.")
            except Exception as e:
                raise Exception(f"Cannot access base directory '{self.base_path}': {str(e)}")
        
        # Create directory structure for the file (relative to base or root)
        path_parts = Path(remote_path).parent.parts
        for part in path_parts:
            if part:
                try:
                    ftp.cwd(part)
                except ftplib.error_perm:
                    # Directory doesn't exist, try to create it
                    try:
                        ftp.mkd(part)
                        ftp.cwd(part)
                    except ftplib.error_perm as e:
                        raise Exception(f"Cannot create directory '{part}': {str(e)}. Check FTP write permissions.")
                except Exception as e:
                    raise Exception(f"Cannot access directory '{part}': {str(e)}")
    
    def upload_file(self, file_data: bytes, remote_path: str) -> str:
        """
        Upload file to FTP server.
        remote_path: Path relative to FTP base (e.g., 'photos/2024/12/image.jpg')
        Returns: Relative path for storage in database
        
        Note: Connection is kept alive after upload for reuse.
        """
        ftp = self._connect()
        
        # Try to ensure directory exists
        try:
            self._ensure_directory(remote_path)
        except Exception as e:
            # If directory creation fails, provide helpful error
            error_msg = str(e)
            if "550" in error_msg or "Permission denied" in error_msg:
                raise Exception(
                    f"FTP permission denied when creating directories for '{remote_path}'. "
                    f"Please ensure:\n"
                    f"1. The FTP user has write permissions\n"
                    f"2. Required directories exist (e.g., 'photos' directory)\n"
                    f"3. The FTP user can create subdirectories, or create them manually\n"
                    f"Original error: {error_msg}"
                )
            raise
        
        # Navigate to the directory containing the file
        path_obj = Path(remote_path)
        filename = path_obj.name
        dir_path = str(path_obj.parent) if path_obj.parent != Path('.') else ''
        
        # Start from base_path if specified
        if self.base_path:
            try:
                ftp.cwd(self.base_path)
            except ftplib.error_perm as e:
                raise Exception(f"Cannot access base directory '{self.base_path}': {str(e)}")
        
        # Navigate to the subdirectory
        if dir_path:
            try:
                for part in Path(dir_path).parts:
                    if part:
                        ftp.cwd(part)
            except ftplib.error_perm as e:
                raise Exception(
                    f"Cannot navigate to directory '{dir_path}': {str(e)}. "
                    f"Directory may not exist. Create it manually or check FTP permissions."
                )
        
        # Upload the file (we're now in the correct directory)
        file_obj = io.BytesIO(file_data)
        
        try:
            # Use just the filename since we're already in the correct directory
            ftp.storbinary(f'STOR {filename}', file_obj, blocksize=8192*4)  # 32KB chunks for better performance
            # Update last used time
            with self._lock:
                if self._last_used:
                    self._last_used = time.time()
        except (ftplib.error_temp, OSError, AttributeError) as e:
            # Temporary errors - connection might be dead, disconnect and let next operation reconnect
            self._disconnect()
            raise Exception(f"FTP upload failed (connection error): {str(e)}")
        except ftplib.error_perm as e:
            # Only disconnect on permission errors (likely permanent)
            error_str = str(e)
            if "550" in error_str:
                raise Exception(
                    f"FTP upload failed: Permission denied. "
                    f"Cannot write file '{filename}' to directory. "
                    f"Check that:\n"
                    f"1. The FTP user has write permissions in the current directory\n"
                    f"2. The directory exists and is accessible\n"
                    f"3. There is sufficient disk space\n"
                    f"Original error: {error_str}"
                )
            raise Exception(f"FTP upload failed: {error_str}")
        except (ftplib.error_temp, OSError) as e:
            # Temporary errors - disconnect and let next operation reconnect
            self._disconnect()
            raise Exception(f"FTP upload failed (temporary error): {str(e)}")
        except Exception as e:
            # Other errors - disconnect to be safe
            self._disconnect()
            raise Exception(f"FTP upload failed: {str(e)}")
        
        # Connection is kept alive for reuse - don't disconnect!
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
            ftp.retrbinary(f'RETR {full_path}', file_obj.write, blocksize=8192*4)  # 32KB chunks
            # Update last used time
            with self._lock:
                self._last_used = time.time()
        except (ftplib.error_temp, OSError, EOFError, AttributeError) as e:
            # Connection errors - disconnect and let next operation reconnect
            self._disconnect()
            raise Exception(f"FTP download failed (connection error): {str(e)}") from e
        except ftplib.error_perm as e:
            # Permission error - disconnect
            self._disconnect()
            raise FileNotFoundError(f"File not found on FTP: {remote_path}") from e
        except Exception as e:
            # Other error - disconnect
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
            # Update last used time
            with self._lock:
                self._last_used = time.time()
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
            # Update last used time
            with self._lock:
                self._last_used = time.time()
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
