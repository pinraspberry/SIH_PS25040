#!/usr/bin/env python3
"""
Test script to verify the Argo Ocean Analysis System setup
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set PostgreSQL library path
if os.getenv('DYLD_LIBRARY_PATH'):
    os.environ['DYLD_LIBRARY_PATH'] = os.getenv('DYLD_LIBRARY_PATH')

def test_imports():
    """Test all required imports"""
    print("🔍 Testing imports...")
    
    try:
        import psycopg2
        print("✅ psycopg2 imported successfully")
    except ImportError as e:
        print(f"❌ psycopg2 import failed: {e}")
        return False
    
    try:
        import flask
        print("✅ Flask imported successfully")
    except ImportError as e:
        print(f"❌ Flask import failed: {e}")
        return False
    
    try:
        import netCDF4
        print("✅ netCDF4 imported successfully")
    except ImportError as e:
        print(f"❌ netCDF4 import failed: {e}")
        return False
    
    try:
        import numpy
        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ numpy import failed: {e}")
        return False
    
    return True

def test_database_connection():
    """Test database connection"""
    print("\n🔍 Testing database connection...")
    
    try:
        sys.path.append('.')
        from config.database import get_psycopg2_url, test_connection
        
        if test_connection():
            print("✅ Database connection successful!")
            return True
        else:
            print("❌ Database connection failed!")
            return False
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_flask_app():
    """Test Flask app creation"""
    print("\n🔍 Testing Flask app...")
    
    try:
        sys.path.append('.')
        from backend.app import create_app
        
        app = create_app()
        print("✅ Flask app created successfully!")
        print(f"✅ Upload folder: {app.config.get('UPLOAD_FOLDER')}")
        return True
    except Exception as e:
        print(f"❌ Flask app test failed: {e}")
        return False

def test_directories():
    """Test required directories"""
    print("\n🔍 Testing directories...")
    
    required_dirs = [
        'backend',
        'frontend',
        'config',
        'data/uploads'
    ]
    
    all_good = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path} exists")
        else:
            print(f"❌ {dir_path} missing")
            all_good = False
    
    return all_good

def main():
    """Run all tests"""
    print("🌊 Argo Ocean Analysis System - Setup Test\n")
    
    tests = [
        ("Imports", test_imports),
        ("Directories", test_directories),
        ("Database Connection", test_database_connection),
        ("Flask App", test_flask_app),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "="*50)
    print("📊 TEST RESULTS:")
    print("="*50)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if not passed:
            all_passed = False
    
    print("="*50)
    if all_passed:
        print("🎉 All tests passed! Your system is ready to go!")
        print("\nTo start the application, run:")
        print("  ./run_app.sh")
        print("\nOr manually:")
        print("  export DYLD_LIBRARY_PATH='/Library/PostgreSQL/17/lib:$DYLD_LIBRARY_PATH'")
        print("  source .venv/bin/activate")
        print("  python backend/app.py")
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())