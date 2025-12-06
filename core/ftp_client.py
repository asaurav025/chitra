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
from queue import Queue, Empty


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
        
        # Connection pool management
        self._pool_size = int(os.environ.get("FTP_POOL_SIZE", "5"))
        self._connection_pool = Queue(maxsize=self._pool_size)
        self._lock = threading.Lock()  # Thread-safe connection pool management
        self._connection_timeout = 300  # 5 minutes - disconnect idle connections
        self._connection_created = {}  # Track when connections were created
        self._pool_initialized = False
    
    def _create_connection(self) -> ftplib.FTP:
        """Create a new FTP connection."""
        ftp = ftplib.FTP()
        ftp.connect(self.host, self.port, timeout=30)
        ftp.login(self.user, self.password)
        ftp.set_pasv(True)
        return ftp
    
    def _initialize_pool(self):
        """Initialize connection pool with connections."""
        if self._pool_initialized:
            return
        
        with self._lock:
            if self._pool_initialized:
                return
            
            # Create initial connections
            for _ in range(min(2, self._pool_size)):  # Start with 2 connections
                try:
                    ftp = self._create_connection()
                    conn_id = id(ftp)
                    self._connection_pool.put(ftp)
                    self._connection_created[conn_id] = time.time()
                except Exception as e:
                    print(f"Warning: Failed to create initial FTP connection: {e}")
            
            self._pool_initialized = True
    
    def _get_connection_from_pool(self) -> ftplib.FTP:
        """
        Get a connection from the pool, creating new ones if needed.
        Returns a connection that should be returned to pool after use.
        """
        self._initialize_pool()
        
        # Try to get connection from pool
        try:
            ftp = self._connection_pool.get(timeout=5)
            conn_id = id(ftp)
            
            # Check if connection has timed out
            if conn_id in self._connection_created:
                created_time = self._connection_created[conn_id]
                if (time.time() - created_time) > self._connection_timeout:
                    # Connection is too old, close it and create new one
                    try:
                        ftp.quit()
                    except:
                        try:
                            ftp.close()
                        except:
                            pass
                    del self._connection_created[conn_id]
                    ftp = self._create_connection()
                    conn_id = id(ftp)
                    self._connection_created[conn_id] = time.time()
            
            return ftp
        except Empty:
            # Pool is empty, create new connection
            ftp = self._create_connection()
            conn_id = id(ftp)
            self._connection_created[conn_id] = time.time()
            return ftp
    
    def _return_connection_to_pool(self, ftp: ftplib.FTP):
        """Return a connection to the pool."""
        if ftp is None:
            return
        
        conn_id = id(ftp)
        try:
            # Try to put back in pool (non-blocking)
            self._connection_pool.put_nowait(ftp)
        except:
            # Pool is full or connection is bad, close it
            try:
                ftp.quit()
            except:
                try:
                    ftp.close()
                except:
                    pass
            if conn_id in self._connection_created:
                del self._connection_created[conn_id]
    
    def _connect(self) -> ftplib.FTP:
        """
        Get FTP connection from pool (for backward compatibility).
        Note: This returns a connection that should NOT be returned to pool
        (for methods that manage their own connection lifecycle).
        """
        return self._get_connection_from_pool()
    
    def _disconnect(self, ftp: ftplib.FTP = None):
        """
        Close FTP connection or return to pool.
        If ftp is provided, returns it to pool. Otherwise closes current connection.
        """
        if ftp:
            self._return_connection_to_pool(ftp)
        else:
            # Legacy behavior - close all connections in pool
            with self._lock:
                while not self._connection_pool.empty():
                    try:
                        ftp = self._connection_pool.get_nowait()
                        try:
                            ftp.quit()
                        except:
                            try:
                                ftp.close()
                            except:
                                pass
                        conn_id = id(ftp)
                        if conn_id in self._connection_created:
                            del self._connection_created[conn_id]
                    except:
                        break
                self._pool_initialized = False
    
    def _ensure_directory(self, remote_path: str, ftp: ftplib.FTP = None):
        """Ensure directory structure exists on FTP server"""
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
        
        Note: Connection is returned to pool after upload.
        """
        ftp = self._get_connection_from_pool()
        
        # Try to ensure directory exists (using the same connection)
        try:
            self._ensure_directory(remote_path, ftp)
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
        except (ftplib.error_temp, OSError, AttributeError, EOFError) as e:
            # Temporary errors - connection might be dead, close it
            try:
                ftp.quit()
            except:
                try:
                    ftp.close()
                except:
                    pass
            conn_id = id(ftp)
            if conn_id in self._connection_created:
                del self._connection_created[conn_id]
            raise Exception(f"FTP upload failed (connection error): {str(e)}")
        except ftplib.error_perm as e:
            # Permission error - connection is likely still valid, return to pool
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
        except Exception as e:
            # Other errors - close connection
            try:
                ftp.quit()
            except:
                try:
                    ftp.close()
                except:
                    pass
            conn_id = id(ftp)
            if conn_id in self._connection_created:
                del self._connection_created[conn_id]
            raise Exception(f"FTP upload failed: {str(e)}")
        finally:
            # Return connection to pool if not already closed
            conn_id = id(ftp)
            if conn_id in self._connection_created:
                self._return_connection_to_pool(ftp)
        
        return remote_path  # Return relative path for database
    
    def download_file_streaming(self, remote_path: str, chunk_size: int = 8192*4):
        """
        Download file from FTP server as a generator (streaming).
        Useful for large files to reduce memory usage.
        
        Note: This keeps the connection open until all chunks are consumed.
        The connection is returned to pool after the generator is exhausted.
        
        Args:
            remote_path: Path stored in database (relative to base)
            chunk_size: Size of chunks to yield
            
        Yields:
            bytes: Chunks of file data
        """
        ftp = self._get_connection_from_pool()
        
        # Construct full path
        if self.base_path:
            full_path = f"{self.base_path}/{remote_path}"
        else:
            full_path = remote_path
        
        chunks = []
        error_occurred = None
        
        def collect_chunk(chunk):
            chunks.append(chunk)
        
        try:
            ftp.retrbinary(f'RETR {full_path}', collect_chunk, blocksize=chunk_size)
        except (ftplib.error_temp, OSError, EOFError, AttributeError) as e:
            # Connection errors - close connection
            error_occurred = Exception(f"FTP download failed (connection error): {str(e)}")
            try:
                ftp.quit()
            except:
                try:
                    ftp.close()
                except:
                    pass
            conn_id = id(ftp)
            if conn_id in self._connection_created:
                del self._connection_created[conn_id]
        except ftplib.error_perm as e:
            # Permission error - return connection to pool (might be valid)
            error_occurred = FileNotFoundError(f"File not found on FTP: {remote_path}")
            self._return_connection_to_pool(ftp)
        except Exception as e:
            # Other error - close connection
            error_occurred = Exception(f"FTP download failed: {str(e)}")
            try:
                ftp.quit()
            except:
                try:
                    ftp.close()
                except:
                    pass
            conn_id = id(ftp)
            if conn_id in self._connection_created:
                del self._connection_created[conn_id]
        finally:
            # Return connection to pool if not already closed
            if error_occurred is None:
                conn_id = id(ftp)
                if conn_id in self._connection_created:
                    self._return_connection_to_pool(ftp)
        
        if error_occurred:
            raise error_occurred
        
        # Yield chunks as they were collected
        for chunk in chunks:
            yield chunk
    
    def download_file(self, remote_path: str) -> bytes:
        """
        Download file from FTP server.
        remote_path: Path stored in database (relative to base)
        """
        ftp = self._get_connection_from_pool()
        
        # Construct full path for RETR command
        if self.base_path:
            full_path = f"{self.base_path}/{remote_path}"
        else:
            full_path = remote_path
        
        file_obj = io.BytesIO()
        try:
            ftp.retrbinary(f'RETR {full_path}', file_obj.write, blocksize=8192*4)  # 32KB chunks
        except (ftplib.error_temp, OSError, EOFError, AttributeError) as e:
            # Connection errors - close connection
            try:
                ftp.quit()
            except:
                try:
                    ftp.close()
                except:
                    pass
            conn_id = id(ftp)
            if conn_id in self._connection_created:
                del self._connection_created[conn_id]
            raise Exception(f"FTP download failed (connection error): {str(e)}") from e
        except ftplib.error_perm as e:
            # Permission error - return connection to pool (might be valid)
            self._return_connection_to_pool(ftp)
            raise FileNotFoundError(f"File not found on FTP: {remote_path}") from e
        except Exception as e:
            # Other error - close connection
            try:
                ftp.quit()
            except:
                try:
                    ftp.close()
                except:
                    pass
            conn_id = id(ftp)
            if conn_id in self._connection_created:
                del self._connection_created[conn_id]
            raise Exception(f"FTP download failed: {str(e)}") from e
        finally:
            # Return connection to pool if not already closed
            conn_id = id(ftp)
            if conn_id in self._connection_created:
                self._return_connection_to_pool(ftp)
        
        file_obj.seek(0)
        return file_obj.read()
    
    def file_exists(self, remote_path: str) -> bool:
        """Check if file exists on FTP server"""
        ftp = None
        try:
            ftp = self._get_connection_from_pool()
            # Construct full path
            if self.base_path:
                full_path = f"{self.base_path}/{remote_path}"
            else:
                full_path = remote_path
            ftp.size(full_path)
            return True
        except:
            return False
        finally:
            if ftp:
                self._return_connection_to_pool(ftp)
    
    def delete_file(self, remote_path: str) -> bool:
        """Delete file from FTP server"""
        ftp = None
        try:
            ftp = self._get_connection_from_pool()
            # Construct full path
            if self.base_path:
                full_path = f"{self.base_path}/{remote_path}"
            else:
                full_path = remote_path
            ftp.delete(full_path)
            return True
        except:
            return False
        finally:
            if ftp:
                self._return_connection_to_pool(ftp)
    
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
