#!/usr/bin/env python3
"""
Vector Migration Runner Script
Executes the vector capabilities migration and validates the setup
Requirements: 1.4, 4.2, 6.4
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def run_migration():
    """Run the vector capabilities migration"""
    print("🚀 Running Vector Capabilities Migration...")
    print("=" * 50)
    
    # Get database connection parameters
    db_host = os.getenv('DATABASE_HOST', 'localhost')
    db_port = os.getenv('DATABASE_PORT', '5432')
    db_name = os.getenv('DATABASE_NAME', 'argo_sih')
    db_user = os.getenv('DATABASE_USER', 'postgres')
    psql_path = os.getenv('PSQL_PATH', '/Library/PostgreSQL/17/bin/psql')
    
    # Check if migration file exists
    migration_file = Path('migrations/001_add_vector_capabilities.sql')
    if not migration_file.exists():
        print(f"❌ Migration file not found: {migration_file}")
        return False
    
    # Construct psql command
    cmd = [
        psql_path,
        '-h', db_host,
        '-p', str(db_port),
        '-U', db_user,
        '-d', db_name,
        '-f', str(migration_file)
    ]
    
    # Set password environment variable
    env = os.environ.copy()
    env['PGPASSWORD'] = os.getenv('POSTGRES_PASSWORD')
    
    try:
        print(f"📝 Executing migration: {migration_file}")
        print(f"🔗 Connecting to: {db_user}@{db_host}:{db_port}/{db_name}")
        
        # Run the migration
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Migration executed successfully!")
            print("\n📄 Migration Output:")
            print(result.stdout)
            return True
        else:
            print("❌ Migration failed!")
            print(f"Error: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print(f"❌ psql command not found at: {psql_path}")
        print("Please check PSQL_PATH in your .env file")
        return False
    except Exception as e:
        print(f"❌ Error running migration: {e}")
        return False

def run_validation():
    """Run the validation script"""
    print("\n🔍 Running Vector Schema Validation...")
    print("=" * 50)
    
    try:
        # Import and run the validator
        sys.path.append('scripts')
        from validate_vector_schema import VectorSchemaValidator
        
        validator = VectorSchemaValidator()
        return validator.run_validation()
        
    except ImportError as e:
        print(f"❌ Error importing validator: {e}")
        return False
    except Exception as e:
        print(f"❌ Error running validation: {e}")
        return False

def main():
    """Main function"""
    print("🎯 Vector Capabilities Setup")
    print("=" * 50)
    
    # Step 1: Run migration
    if not run_migration():
        print("\n❌ Migration failed. Please check the error messages above.")
        sys.exit(1)
    
    # Step 2: Run validation
    if not run_validation():
        print("\n⚠️ Validation completed with warnings. Check the details above.")
        sys.exit(1)
    
    print("\n🎉 Vector capabilities setup completed successfully!")
    print("\n📋 Next Steps:")
    print("   1. Run data ingestion to populate embeddings")
    print("   2. Vector similarity index will be created automatically after ingestion")
    print("   3. Test semantic search functionality")
    
    sys.exit(0)

if __name__ == "__main__":
    main()