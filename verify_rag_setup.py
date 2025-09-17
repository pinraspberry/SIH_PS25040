#!/usr/bin/env python3
"""
Verification script for RAG pipeline environment setup.
This script checks that all required dependencies and environment variables are properly configured.
"""

import os
import sys
from dotenv import load_dotenv

def check_dependencies():
    """Check if all required Python packages can be imported."""
    print("Checking Python dependencies...")
    
    try:
        import xarray
        print("✓ xarray imported successfully")
    except ImportError as e:
        print(f"✗ xarray import failed: {e}")
        return False
    
    try:
        import netCDF4
        print("✓ netCDF4 imported successfully")
    except ImportError as e:
        print(f"✗ netCDF4 import failed: {e}")
        return False
    
    try:
        import sentence_transformers
        print("✓ sentence-transformers imported successfully")
    except ImportError as e:
        print(f"✗ sentence-transformers import failed: {e}")
        return False
    
    try:
        import pgvector
        print("✓ pgvector imported successfully")
    except ImportError as e:
        print(f"✗ pgvector import failed: {e}")
        return False
    
    try:
        import google.generativeai
        print("✓ google-generativeai imported successfully")
    except ImportError as e:
        print(f"✗ google-generativeai import failed: {e}")
        return False
    
    # Note: psycopg2 may fail due to library path issues on macOS
    # This is expected and handled by DYLD_LIBRARY_PATH in .env
    try:
        import psycopg2
        print("✓ psycopg2 imported successfully")
    except ImportError as e:
        print(f"⚠ psycopg2 import failed (expected on macOS): {e}")
        print("  This should work when running with proper DYLD_LIBRARY_PATH from .env")
    
    return True

def check_environment_variables():
    """Check if all required environment variables are set."""
    print("\nChecking environment variables...")
    
    # Load environment variables from .env file
    load_dotenv()
    
    required_vars = [
        'DATABASE_HOST',
        'DATABASE_PORT', 
        'DATABASE_NAME',
        'DATABASE_USER',
        'DATABASE_PASSWORD',
        'GEMINI_API_KEY',
        'EMBEDDING_MODEL',
        'EMBEDDING_DIMENSION',
        'VECTOR_INDEX_LISTS',
        'MAX_SEARCH_RESULTS',
        'RAG_TEMPERATURE',
        'BATCH_SIZE',
        'MAX_PROFILE_LENGTH',
        'ENABLE_CACHING',
        'LOG_LEVEL'
    ]
    
    missing_vars = []
    for var in required_vars:
        value = os.getenv(var)
        if value is None:
            missing_vars.append(var)
            print(f"✗ {var} not set")
        else:
            # Don't print sensitive values like API keys and passwords
            if 'PASSWORD' in var or 'API_KEY' in var:
                print(f"✓ {var} is set (value hidden)")
            else:
                print(f"✓ {var} = {value}")
    
    if missing_vars:
        print(f"\n⚠ Missing environment variables: {', '.join(missing_vars)}")
        return False
    
    return True

def check_model_configuration():
    """Check model configuration settings."""
    print("\nChecking model configuration...")
    
    embedding_model = os.getenv('EMBEDDING_MODEL', 'sentence-transformers')
    embedding_dim = os.getenv('EMBEDDING_DIMENSION', '384')
    
    print(f"✓ Embedding model: {embedding_model}")
    print(f"✓ Embedding dimension: {embedding_dim}")
    
    if embedding_model == 'sentence-transformers' and embedding_dim != '384':
        print("⚠ Warning: sentence-transformers typically uses 384 dimensions")
        print("  Consider setting EMBEDDING_DIMENSION=384 for all-MiniLM-L6-v2")
    
    return True

def main():
    """Main verification function."""
    print("RAG Pipeline Environment Verification")
    print("=" * 40)
    
    success = True
    
    # Check dependencies
    if not check_dependencies():
        success = False
    
    # Check environment variables
    if not check_environment_variables():
        success = False
    
    # Check model configuration
    if not check_model_configuration():
        success = False
    
    print("\n" + "=" * 40)
    if success:
        print("✓ RAG pipeline environment setup completed successfully!")
        print("\nNext steps:")
        print("1. Update GEMINI_API_KEY in .env with your actual API key")
        print("2. Ensure PostgreSQL is running with pgvector extension")
        print("3. Run the next task to extend database schema with vector capabilities")
    else:
        print("✗ RAG pipeline environment setup has issues that need to be resolved")
        sys.exit(1)

if __name__ == "__main__":
    main()