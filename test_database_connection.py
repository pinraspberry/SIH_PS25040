#!/usr/bin/env python3
"""
Database connection test script for RAG pipeline.
Tests database connectivity and pgvector extension availability.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_basic_connection():
    """Test basic database connection without psycopg2."""
    print("Testing basic database connectivity...")
    
    try:
        # Try using asyncpg for basic connection test
        import asyncio
        import asyncpg
        
        async def check_connection():
            try:
                # Get connection parameters from environment
                host = os.getenv('DATABASE_HOST', 'localhost')
                port = int(os.getenv('DATABASE_PORT', 5432))
                database = os.getenv('DATABASE_NAME', 'argo_sih')
                user = os.getenv('DATABASE_USER', 'postgres')
                password = os.getenv('DATABASE_PASSWORD', '')
                
                conn = await asyncpg.connect(
                    host=host,
                    port=port,
                    database=database,
                    user=user,
                    password=password
                )
                
                # Test basic query
                version = await conn.fetchval('SELECT version()')
                print(f"✓ Connected to: {version}")
                
                # Check tables
                tables = await conn.fetch("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                    ORDER BY table_name;
                """)
                table_names = [row['table_name'] for row in tables]
                print(f"✓ Tables found: {table_names}")
                
                # Check record count
                count = await conn.fetchval("SELECT COUNT(*) FROM argo_profiles;")
                print(f"✓ Records in argo_profiles: {count}")
                
                await conn.close()
                return True
                
            except Exception as e:
                print(f"✗ Connection failed: {e}")
                return False
        
        return asyncio.run(check_connection())
        
    except ImportError:
        print("✗ asyncpg not available, trying alternative method...")
        return False
    except Exception as e:
        print(f"✗ Connection test failed: {e}")
        return False

def test_psycopg2_connection():
    """Test psycopg2 connection with proper library path."""
    print("\nTesting psycopg2 connection...")
    
    try:
        # Set library path for macOS
        os.environ['DYLD_LIBRARY_PATH'] = '/Library/PostgreSQL/17/lib'
        
        from config.database import test_connection
        return test_connection()
        
    except Exception as e:
        print(f"✗ psycopg2 connection failed: {e}")
        return False

def check_pgvector_extension():
    """Check if pgvector extension is available and installed."""
    print("\nChecking pgvector extension...")
    
    try:
        import asyncio
        import asyncpg
        
        async def check_extension():
            try:
                # Get connection parameters from environment
                host = os.getenv('DATABASE_HOST', 'localhost')
                port = int(os.getenv('DATABASE_PORT', 5432))
                database = os.getenv('DATABASE_NAME', 'argo_sih')
                user = os.getenv('DATABASE_USER', 'postgres')
                password = os.getenv('DATABASE_PASSWORD', '')
                
                conn = await asyncpg.connect(
                    host=host,
                    port=port,
                    database=database,
                    user=user,
                    password=password
                )
                
                # Check if pgvector extension is available
                extensions = await conn.fetch("""
                    SELECT name, default_version, installed_version 
                    FROM pg_available_extensions 
                    WHERE name = 'vector';
                """)
                
                if extensions:
                    ext = extensions[0]
                    print(f"✓ pgvector extension found: {ext['name']}")
                    print(f"  Default version: {ext['default_version']}")
                    print(f"  Installed version: {ext['installed_version'] or 'Not installed'}")
                    
                    if ext['installed_version']:
                        print("✅ pgvector extension is installed and ready!")
                        
                        # Test vector operations
                        try:
                            await conn.execute("SELECT '[1,2,3]'::vector;")
                            print("✓ Vector operations working correctly")
                        except Exception as e:
                            print(f"⚠ Vector operations test failed: {e}")
                        
                        await conn.close()
                        return True
                    else:
                        print("⚠ pgvector extension available but not installed")
                        print("\nTo install pgvector extension, run:")
                        print("  psql -d argo_sih -c 'CREATE EXTENSION vector;'")
                        print("  Or use the PostgreSQL admin interface")
                        
                        await conn.close()
                        return False
                else:
                    print("❌ pgvector extension not found")
                    print("\nTo install pgvector:")
                    print("1. Install pgvector for PostgreSQL 17:")
                    print("   brew install pgvector")
                    print("   OR download from: https://github.com/pgvector/pgvector")
                    print("2. Then enable it in your database:")
                    print("   psql -d argo_sih -c 'CREATE EXTENSION vector;'")
                    
                    await conn.close()
                    return False
                    
            except Exception as e:
                print(f"✗ Extension check failed: {e}")
                return False
        
        return asyncio.run(check_extension())
        
    except Exception as e:
        print(f"✗ pgvector check failed: {e}")
        return False

def check_rag_dependencies():
    """Check RAG pipeline Python dependencies."""
    print("\nChecking RAG pipeline dependencies...")
    
    dependencies = [
        ('xarray', 'NetCDF processing'),
        ('netCDF4', 'NetCDF file handling'),
        ('sentence_transformers', 'Local embeddings'),
        ('pgvector', 'Vector database operations'),
        ('google.generativeai', 'Gemini API integration')
    ]
    
    all_good = True
    for module, description in dependencies:
        try:
            __import__(module)
            print(f"✓ {module} - {description}")
        except ImportError as e:
            print(f"✗ {module} - {description}: {e}")
            all_good = False
    
    return all_good

def main():
    """Main test function."""
    print("Database Connection and RAG Pipeline Test")
    print("=" * 50)
    
    # Test basic connection
    basic_ok = test_basic_connection()
    
    # Test psycopg2 connection
    psycopg2_ok = test_psycopg2_connection()
    
    # Check pgvector extension
    pgvector_ok = check_pgvector_extension()
    
    # Check RAG dependencies
    deps_ok = check_rag_dependencies()
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"✓ Basic database connection: {'✅' if basic_ok else '❌'}")
    print(f"✓ psycopg2 connection: {'✅' if psycopg2_ok else '❌'}")
    print(f"✓ pgvector extension: {'✅' if pgvector_ok else '❌'}")
    print(f"✓ RAG dependencies: {'✅' if deps_ok else '❌'}")
    
    if all([basic_ok, psycopg2_ok, pgvector_ok, deps_ok]):
        print("\n🎉 All systems ready for RAG pipeline!")
    elif basic_ok and deps_ok:
        print("\n⚠ Database connection works, but pgvector needs setup")
        print("   Install pgvector extension to proceed with RAG pipeline")
    else:
        print("\n❌ Some issues need to be resolved before proceeding")
    
    return all([basic_ok, psycopg2_ok, pgvector_ok, deps_ok])

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)