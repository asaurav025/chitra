#!/usr/bin/env python3
"""
Test FTP server permissions and directory access.
"""
import ftplib
import os
import sys
from io import BytesIO

def test_ftp_permissions():
    """Test FTP connection and permissions"""
    host = os.environ.get("FTP_STORAGE_HOST", "localhost")
    port = int(os.environ.get("FTP_STORAGE_PORT", "21"))
    user = os.environ.get("FTP_STORAGE_USER", "ftpstorage")
    password = os.environ.get("FTP_STORAGE_PASS", "")
    base_path = os.environ.get("FTP_STORAGE_BASE", "")
    
    # If password not set, try to read from stdin (for non-interactive environments)
    if not password:
        import sys
        if sys.stdin.isatty():
            # Interactive terminal - can prompt
            try:
                import getpass
                password = getpass.getpass(f"Enter FTP password for {user}@{host}: ")
            except:
                print("Cannot prompt for password in this environment.")
                print("Please set FTP_STORAGE_PASS environment variable.")
                sys.exit(1)
        else:
            # Non-interactive - read from stdin if available
            print("FTP password not found in environment variables.")
            print("Please set FTP_STORAGE_PASS environment variable or provide via stdin.")
            sys.exit(1)
    
    print(f"Connecting to FTP server: {host}:{port}")
    print(f"User: {user}")
    print(f"Base path: '{base_path}' (empty = home directory)")
    print("-" * 60)
    
    try:
        # Connect
        ftp = ftplib.FTP()
        ftp.connect(host, port)
        print("✓ Connected to FTP server")
        
        # Login
        ftp.login(user, password)
        print("✓ Login successful")
        
        # Set passive mode
        ftp.set_pasv(True)
        print("✓ Passive mode enabled")
        
        # Get current directory
        try:
            pwd = ftp.pwd()
            print(f"✓ Current directory: {pwd}")
        except Exception as e:
            print(f"⚠ Cannot get current directory: {e}")
        
        # Navigate to base path if specified
        if base_path:
            try:
                ftp.cwd(base_path)
                print(f"✓ Navigated to base path: {base_path}")
                pwd = ftp.pwd()
                print(f"  Current directory: {pwd}")
            except ftplib.error_perm as e:
                print(f"✗ Cannot access base path '{base_path}': {e}")
                print("  Try creating it manually or check permissions")
                return
            except Exception as e:
                print(f"✗ Error accessing base path: {e}")
                return
        
        # Test listing current directory
        try:
            files = ftp.nlst()
            print(f"✓ Can list directory ({len(files)} items)")
            if files:
                print(f"  Sample items: {', '.join(files[:5])}")
        except Exception as e:
            print(f"⚠ Cannot list directory: {e}")
        
        # Test if 'photos' directory exists
        print("\n" + "-" * 60)
        print("Testing 'photos' directory:")
        try:
            ftp.cwd("photos")
            print("✓ 'photos' directory exists and is accessible")
            photos_pwd = ftp.pwd()
            print(f"  Current directory: {photos_pwd}")
            
            # Try to list photos directory
            try:
                photos_files = ftp.nlst()
                print(f"✓ Can list photos directory ({len(photos_files)} items)")
            except Exception as e:
                print(f"⚠ Cannot list photos directory: {e}")
            
            # Go back
            ftp.cwd("..")
        except ftplib.error_perm as e:
            print(f"✗ 'photos' directory does not exist or is not accessible: {e}")
            print("  You need to create it manually:")
            if base_path:
                print(f"    mkdir -p {base_path}/photos")
            else:
                print(f"    mkdir -p photos  (on FTP server)")
        
        # Test creating a subdirectory
        print("\n" + "-" * 60)
        print("Testing directory creation permissions:")
        test_dir = "test_permissions_12345"
        try:
            # Try to create a test directory
            ftp.mkd(test_dir)
            print(f"✓ Can create directories (created '{test_dir}')")
            
            # Try to navigate into it
            try:
                ftp.cwd(test_dir)
                print(f"✓ Can navigate into created directory")
                ftp.cwd("..")
            except Exception as e:
                print(f"⚠ Created directory but cannot navigate: {e}")
            
            # Try to delete it
            try:
                ftp.rmd(test_dir)
                print(f"✓ Can delete directories (removed '{test_dir}')")
            except Exception as e:
                print(f"⚠ Cannot delete test directory: {e}")
                print(f"  You may need to manually remove: {test_dir}")
                
        except ftplib.error_perm as e:
            print(f"✗ Cannot create directories: {e}")
            print("  The FTP user does not have permission to create directories")
            print("  You will need to create directories manually on the FTP server")
        
        # Test file upload (small test file)
        print("\n" + "-" * 60)
        print("Testing file upload permissions:")
        test_content = b"test file content"
        test_filename = "test_upload_12345.txt"
        
        try:
            # Make sure we're in the right directory
            if base_path:
                try:
                    ftp.cwd(base_path)
                except:
                    pass
            try:
                ftp.cwd("photos")
            except:
                print("⚠ Cannot navigate to photos directory for upload test")
                print("  Skipping upload test")
                ftp.quit()
                return
            
            # Try to upload
            file_obj = BytesIO(test_content)
            ftp.storbinary(f'STOR {test_filename}', file_obj)
            print(f"✓ Can upload files (uploaded '{test_filename}')")
            
            # Try to delete the test file
            try:
                ftp.delete(test_filename)
                print(f"✓ Can delete files (removed '{test_filename}')")
            except Exception as e:
                print(f"⚠ Cannot delete test file: {e}")
                print(f"  You may need to manually remove: {test_filename}")
                
        except ftplib.error_perm as e:
            print(f"✗ Cannot upload files: {e}")
            print("  The FTP user does not have write permission")
        except Exception as e:
            print(f"✗ Upload test failed: {e}")
        
        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY:")
        print("=" * 60)
        print("If you can create directories: Use year/month structure (photos/2025/12/)")
        print("If you cannot create directories: Use flat structure (photos/filename)")
        print("=" * 60)
        
        ftp.quit()
        print("\n✓ FTP connection closed")
        
    except ftplib.error_perm as e:
        print(f"\n✗ FTP Permission Error: {e}")
        print("Check your FTP credentials and permissions")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ FTP Connection Error: {e}")
        print("Check your FTP server configuration")
        sys.exit(1)

if __name__ == "__main__":
    test_ftp_permissions()

