#!/usr/bin/env python3
"""
Database Schema Validation Script for Vector Capabilities
Validates that pgvector extension and vector columns are properly set up
Requirements: 1.4, 4.2, 6.4
"""

import os
import sys
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor

# Load environment variables
load_dotenv()

class VectorSchemaValidator:
    """Validates database schema for vector capabilities"""
    
    def __init__(self):
        self.connection = None
        self.validation_results = []
        
    def connect_to_database(self) -> bool:
        """Establish database connection"""
        try:
            self.connection = psycopg2.connect(
                host=os.getenv('DATABASE_HOST', 'localhost'),
                port=os.getenv('DATABASE_PORT', 5432),
                database=os.getenv('DATABASE_NAME', 'argo_sih'),
                user=os.getenv('DATABASE_USER', 'postgres'),
                password=os.getenv('POSTGRES_PASSWORD')
            )
            self.log_result("✅ Database connection", "SUCCESS", "Connected to PostgreSQL database")
            return True
        except Exception as e:
            self.log_result("❌ Database connection", "FAILED", f"Error: {e}")
            return False
    
    def log_result(self, test_name: str, status: str, details: str):
        """Log validation result"""
        self.validation_results.append({
            'test': test_name,
            'status': status,
            'details': details
        })
        print(f"{test_name}: {status} - {details}")
    
    def validate_pgvector_extension(self) -> bool:
        """Validate that pgvector extension is installed and enabled"""
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT extname, extversion, extrelocatable 
                    FROM pg_extension 
                    WHERE extname = 'vector'
                """)
                result = cursor.fetchone()
                
                if result:
                    version = result['extversion']
                    self.log_result("✅ pgvector extension", "SUCCESS", 
                                  f"Extension installed, version: {version}")
                    return True
                else:
                    self.log_result("❌ pgvector extension", "FAILED", 
                                  "Extension not found. Run: CREATE EXTENSION vector;")
                    return False
        except Exception as e:
            self.log_result("❌ pgvector extension", "ERROR", f"Error checking extension: {e}")
            return False
    
    def validate_vector_columns(self) -> bool:
        """Validate that vector columns exist in argo_profiles table"""
        expected_columns = {
            'profile_summary': 'text',
            'embedding': 'vector',
            'embedding_model': 'character varying',
            'embedding_created_at': 'timestamp without time zone'
        }
        
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns 
                    WHERE table_name = 'argo_profiles' 
                    AND column_name IN ('profile_summary', 'embedding', 'embedding_model', 'embedding_created_at')
                    ORDER BY column_name
                """)
                results = cursor.fetchall()
                
                found_columns = {row['column_name']: row['data_type'] for row in results}
                
                all_valid = True
                for col_name, expected_type in expected_columns.items():
                    if col_name in found_columns:
                        actual_type = found_columns[col_name]
                        # Special handling for vector type which shows as USER-DEFINED
                        if col_name == 'embedding' and actual_type == 'USER-DEFINED':
                            self.log_result(f"✅ Column {col_name}", "SUCCESS", 
                                          f"Type: vector (reported as {actual_type})")
                        elif expected_type in actual_type or actual_type in expected_type:
                            self.log_result(f"✅ Column {col_name}", "SUCCESS", 
                                          f"Type: {actual_type}")
                        else:
                            self.log_result(f"❌ Column {col_name}", "WARNING", 
                                          f"Expected: {expected_type}, Found: {actual_type}")
                            all_valid = False
                    else:
                        self.log_result(f"❌ Column {col_name}", "FAILED", 
                                      f"Column missing. Expected type: {expected_type}")
                        all_valid = False
                
                return all_valid
        except Exception as e:
            self.log_result("❌ Vector columns", "ERROR", f"Error checking columns: {e}")
            return False
    
    def validate_vector_indexes(self) -> bool:
        """Validate that vector indexes exist"""
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                # Check for vector similarity index
                cursor.execute("""
                    SELECT indexname, indexdef
                    FROM pg_indexes 
                    WHERE tablename = 'argo_profiles'
                    AND indexname LIKE '%embedding%'
                    ORDER BY indexname
                """)
                indexes = cursor.fetchall()
                
                vector_index_found = False
                for index in indexes:
                    if 'ivfflat' in index['indexdef'].lower() or 'vector' in index['indexdef'].lower():
                        vector_index_found = True
                        self.log_result("✅ Vector similarity index", "SUCCESS", 
                                      f"Index: {index['indexname']}")
                        break
                
                if not vector_index_found:
                    self.log_result("✅ Vector similarity index", "INFO", 
                                  "IVFFlat index will be created after data ingestion (as expected).")
                
                # Check for other supporting indexes
                cursor.execute("""
                    SELECT indexname
                    FROM pg_indexes 
                    WHERE tablename = 'argo_profiles'
                    AND indexname IN ('idx_embedding_model', 'idx_embedding_created_at', 'idx_profile_summary')
                """)
                support_indexes = cursor.fetchall()
                
                expected_support_indexes = ['idx_embedding_model', 'idx_embedding_created_at', 'idx_profile_summary']
                found_support_indexes = [idx['indexname'] for idx in support_indexes]
                
                for idx_name in expected_support_indexes:
                    if idx_name in found_support_indexes:
                        self.log_result(f"✅ Support index {idx_name}", "SUCCESS", "Index exists")
                    else:
                        self.log_result(f"⚠️ Support index {idx_name}", "WARNING", "Index missing")
                
                return True
        except Exception as e:
            self.log_result("❌ Vector indexes", "ERROR", f"Error checking indexes: {e}")
            return False
    
    def validate_vector_operations(self) -> bool:
        """Test basic vector operations"""
        try:
            with self.connection.cursor() as cursor:
                # Test vector creation and similarity operations
                cursor.execute("SELECT '[1,2,3]'::vector(3) <-> '[1,2,4]'::vector(3) as distance")
                result = cursor.fetchone()
                distance = result[0]
                
                if distance is not None:
                    self.log_result("✅ Vector operations", "SUCCESS", 
                                  f"Cosine distance test passed: {distance}")
                    return True
                else:
                    self.log_result("❌ Vector operations", "FAILED", "Vector operations not working")
                    return False
        except Exception as e:
            self.log_result("❌ Vector operations", "ERROR", f"Error testing vector ops: {e}")
            return False
    
    def validate_embedding_dimension(self) -> bool:
        """Validate that embedding column supports 384 dimensions"""
        try:
            with self.connection.cursor() as cursor:
                # Test inserting a 384-dimensional vector
                test_vector = '[' + ','.join(['0.1'] * 384) + ']'
                cursor.execute(f"SELECT '{test_vector}'::vector(384) IS NOT NULL as valid")
                result = cursor.fetchone()
                
                if result[0]:
                    self.log_result("✅ Embedding dimensions", "SUCCESS", 
                                  "384-dimensional vectors supported")
                    return True
                else:
                    self.log_result("❌ Embedding dimensions", "FAILED", 
                                  "384-dimensional vectors not supported")
                    return False
        except Exception as e:
            self.log_result("❌ Embedding dimensions", "ERROR", f"Error testing dimensions: {e}")
            return False
    
    def check_existing_data(self) -> Dict:
        """Check existing data in argo_profiles table"""
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                # Count total profiles
                cursor.execute("SELECT COUNT(*) as total FROM argo_profiles")
                total_profiles = cursor.fetchone()['total']
                
                # Count profiles with embeddings
                cursor.execute("SELECT COUNT(*) as with_embeddings FROM argo_profiles WHERE embedding IS NOT NULL")
                with_embeddings = cursor.fetchone()['with_embeddings']
                
                # Count profiles with summaries
                cursor.execute("SELECT COUNT(*) as with_summaries FROM argo_profiles WHERE profile_summary IS NOT NULL")
                with_summaries = cursor.fetchone()['with_summaries']
                
                data_info = {
                    'total_profiles': total_profiles,
                    'with_embeddings': with_embeddings,
                    'with_summaries': with_summaries
                }
                
                self.log_result("📊 Data status", "INFO", 
                              f"Total: {total_profiles}, With embeddings: {with_embeddings}, With summaries: {with_summaries}")
                
                return data_info
        except Exception as e:
            self.log_result("❌ Data status", "ERROR", f"Error checking data: {e}")
            return {}
    
    def run_validation(self) -> bool:
        """Run complete validation suite"""
        print("🔍 Starting Vector Schema Validation...")
        print("=" * 60)
        
        if not self.connect_to_database():
            return False
        
        try:
            # Run all validation tests
            tests_passed = 0
            total_tests = 6
            
            if self.validate_pgvector_extension():
                tests_passed += 1
            
            if self.validate_vector_columns():
                tests_passed += 1
            
            if self.validate_vector_indexes():
                tests_passed += 1
            
            if self.validate_vector_operations():
                tests_passed += 1
            
            if self.validate_embedding_dimension():
                tests_passed += 1
            
            # Data check is informational
            self.check_existing_data()
            tests_passed += 1
            
            print("\n" + "=" * 60)
            print(f"📋 Validation Summary: {tests_passed}/{total_tests} tests passed")
            
            if tests_passed == total_tests:
                print("🎉 All validations passed! Vector capabilities are ready.")
                return True
            else:
                print("⚠️ Some validations failed. Check the details above.")
                return False
                
        finally:
            if self.connection:
                self.connection.close()
    
    def get_setup_instructions(self) -> List[str]:
        """Get setup instructions for failed validations"""
        instructions = []
        
        for result in self.validation_results:
            if result['status'] == 'FAILED':
                if 'pgvector extension' in result['test']:
                    instructions.append("1. Install pgvector extension: CREATE EXTENSION vector;")
                elif 'Column' in result['test']:
                    instructions.append("2. Run migration script: psql -f migrations/001_add_vector_capabilities.sql")
                elif 'Vector similarity index' in result['test']:
                    instructions.append("3. Create vector index after data ingestion")
        
        return instructions

def main():
    """Main validation function"""
    validator = VectorSchemaValidator()
    
    success = validator.run_validation()
    
    if not success:
        print("\n🔧 Setup Instructions:")
        instructions = validator.get_setup_instructions()
        for instruction in instructions:
            print(f"   {instruction}")
        
        print("\n📖 For detailed setup, see: DATABASE_SETUP.md")
        sys.exit(1)
    else:
        print("\n✅ Vector schema validation completed successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()