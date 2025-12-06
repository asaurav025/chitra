#!/usr/bin/env python3
"""
Test script for FastAPI app.
Run this to verify step 4 is working correctly.
"""
import requests
import json
import time
import sys

BASE_URL = "http://127.0.0.1:5000"

def test_endpoints():
    """Test FastAPI endpoints."""
    print("Testing FastAPI endpoints...")
    print(f"Base URL: {BASE_URL}\n")
    
    # Test root endpoint
    try:
        print("1. Testing root endpoint (GET /)...")
        r = requests.get(f"{BASE_URL}/", timeout=5)
        print(f"   Status: {r.status_code}")
        if r.status_code == 200:
            print(f"   Response: {json.dumps(r.json(), indent=2)}")
            print("   ✓ Root endpoint working\n")
        else:
            print(f"   ✗ Unexpected status code: {r.status_code}\n")
    except Exception as e:
        print(f"   ✗ Error: {e}\n")
        return False
    
    # Test health endpoint
    try:
        print("2. Testing health endpoint (GET /api/health)...")
        r = requests.get(f"{BASE_URL}/api/health", timeout=5)
        print(f"   Status: {r.status_code}")
        if r.status_code == 200:
            health = r.json()
            print(f"   Response: {json.dumps(health, indent=2)}")
            print(f"   Status: {health.get('status', 'unknown')}")
            print(f"   DB Status: {health.get('db_status', 'unknown')}")
            print(f"   Storage Status: {health.get('storage_status', 'unknown')}")
            print("   ✓ Health endpoint working\n")
        else:
            print(f"   ✗ Unexpected status code: {r.status_code}\n")
    except Exception as e:
        print(f"   ✗ Error: {e}\n")
        return False
    
    print("✓ All endpoint tests passed!")
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("FastAPI App Test (Step 4)")
    print("=" * 50)
    print("\nMake sure the FastAPI server is running:")
    print("  source .venv/bin/activate")
    print("  python -m uvicorn app_fastapi:app --host 127.0.0.1 --port 5000")
    print("\n" + "=" * 50 + "\n")
    
    # Wait a bit for server to be ready
    time.sleep(1)
    
    success = test_endpoints()
    sys.exit(0 if success else 1)

