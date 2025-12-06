#!/usr/bin/env python3
"""
Clear all Chitra data - database and MinIO bucket.
Use with caution - this will delete all photos, metadata, and storage!

Usage:
    python3 clear_data.py                    # Interactive mode (asks for confirmation)
    python3 clear_data.py --yes              # Non-interactive (auto-confirm)
    python3 clear_data.py --db-only          # Only clear database
    python3 clear_data.py --minio-only       # Only clear MinIO bucket
"""
import os
import sys
import argparse
from pathlib import Path


def clear_database(db_path: str) -> bool:
    """Clear the database file."""
    try:
        if os.path.exists(db_path):
            os.unlink(db_path)
            print(f"✓ Deleted database: {db_path}")
            return True
        else:
            print(f"⚠ Database not found: {db_path}")
            return False
    except Exception as e:
        print(f"✗ Error deleting database: {e}")
        return False


def clear_minio_bucket() -> bool:
    """Clear all objects from MinIO bucket."""
    try:
        from core.storage_client import MinIOStorageClient
        storage = MinIOStorageClient()
        
        print(f"Connecting to MinIO bucket: {storage.bucket_name}")
        
        # List all objects
        objects = storage.client.list_objects(
            storage.bucket_name,
            recursive=True
        )
        
        object_list = list(objects)
        count = len(object_list)
        
        if count == 0:
            print("✓ MinIO bucket is already empty")
            return True
        
        print(f"Found {count} objects in bucket")
        
        # Delete all objects
        deleted = 0
        for obj in object_list:
            try:
                storage.client.remove_object(storage.bucket_name, obj.object_name)
                deleted += 1
                if deleted % 100 == 0:
                    print(f"  Deleted {deleted}/{count} objects...")
            except Exception as e:
                print(f"  ⚠ Error deleting {obj.object_name}: {e}")
        
        print(f"✓ Deleted {deleted} objects from MinIO bucket")
        return True
        
    except ImportError:
        print("✗ Could not import MinIOStorageClient - is minio installed?")
        return False
    except Exception as e:
        print(f"✗ Error clearing MinIO bucket: {e}")
        return False


def clear_cache():
    """Clear application cache."""
    try:
        from core.cache import clear_cache
        clear_cache()
        print("✓ Cleared application cache")
        return True
    except Exception as e:
        print(f"⚠ Could not clear cache: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Clear all Chitra data')
    parser.add_argument('--yes', action='store_true',
                        help='Skip confirmation prompt')
    parser.add_argument('--db-only', action='store_true',
                        help='Only clear database')
    parser.add_argument('--minio-only', action='store_true',
                        help='Only clear MinIO bucket')
    parser.add_argument('--db-path', default=None,
                        help='Custom database path (default: from env or photo.db)')
    
    args = parser.parse_args()
    
    # Get database path
    db_path = args.db_path or os.environ.get("CHITRA_DB_PATH", "photo.db")
    
    print("=" * 60)
    print("CHITRA DATA CLEARANCE")
    print("=" * 60)
    print()
    print("This will delete:")
    if not args.minio_only:
        print(f"  - Database: {db_path}")
    if not args.db_only:
        print("  - All files in MinIO bucket")
    print("  - Application cache")
    print()
    
    # Confirmation
    if not args.yes:
        response = input("Are you sure you want to continue? (yes/no): ")
        if response.lower() not in ('yes', 'y'):
            print("Cancelled.")
            sys.exit(0)
    
    print()
    print("Clearing data...")
    print()
    
    success = True
    
    # Clear database
    if not args.minio_only:
        if not clear_database(db_path):
            success = False
    
    # Clear MinIO
    if not args.db_only:
        if not clear_minio_bucket():
            success = False
    
    # Clear cache
    clear_cache()
    
    print()
    print("=" * 60)
    if success:
        print("✓ All data cleared successfully!")
        print()
        print("You can now start fresh with:")
        print("  python3 run.py")
    else:
        print("⚠ Some operations had errors. Check output above.")
        sys.exit(1)
    print("=" * 60)


if __name__ == '__main__':
    main()

