"""
Integration tests for vector database manager
Tests vector storage, retrieval, and similarity search operations
"""

import pytest
import os
import sys
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from models.argo_model import DatabaseManager, ArgoProfile
from sqlalchemy import text

class TestVectorDatabaseManager:
    """Test suite for vector database operations"""
    
    @classmethod
    def setup_class(cls):
        """Set up test database manager"""
        cls.db_manager = DatabaseManager()
        cls.test_embeddings = [
            [0.1, 0.2, 0.3] + [0.0] * 381,  # 384-dim vector
            [0.4, 0.5, 0.6] + [0.0] * 381,  # Similar to first
            [0.9, 0.8, 0.7] + [0.0] * 381,  # Different from first two
        ]
        cls.test_profiles = []
    
    def test_vector_extension_available(self):
        """Test that pgvector extension is available"""
        assert self.db_manager.test_vector_extension(), "pgvector extension not available"
    
    def test_batch_insert_profiles_with_embeddings(self):
        """Test batch insertion of profiles with embeddings"""
        # Create test profile data with embeddings
        profile_records = []
        for i in range(3):
            profile_data = {
                'platform_number': f'TEST{i:04d}',
                'cycle_number': 1,
                'profile_date': datetime.now() - timedelta(days=i),
                'latitude': 20.0 + i,
                'longitude': 60.0 + i,
                'level_index': 1,
                'pressure': 10.0 + i,
                'temperature': 25.0 + i * 0.5,
                'salinity': 35.0 + i * 0.1,
                'profile_summary': f'Test profile {i} with temperature {25.0 + i * 0.5}°C',
                'embedding': self.test_embeddings[i],
                'embedding_model': 'test-model-v1'
            }
            profile_records.append(profile_data)
        
        # Insert profiles
        inserted_count = self.db_manager.batch_insert_profiles_with_embeddings(
            profile_records, batch_size=2
        )
        
        assert inserted_count == 3, f"Expected 3 profiles inserted, got {inserted_count}"
        
        # Store test profile IDs for cleanup
        session = self.db_manager.get_session()
        try:
            test_profiles = session.query(ArgoProfile).filter(
                ArgoProfile.platform_number.like('TEST%')
            ).all()
            self.test_profiles = [p.id for p in test_profiles]
            assert len(self.test_profiles) == 3, "Test profiles not found in database"
        finally:
            self.db_manager.close_session(session)
    
    def test_vector_similarity_search(self):
        """Test vector similarity search functionality"""
        # Use first test embedding as query
        query_embedding = self.test_embeddings[0]
        
        # Perform similarity search
        results = self.db_manager.vector_similarity_search(
            query_embedding=query_embedding,
            limit=5,
            similarity_threshold=0.5
        )
        
        assert len(results) > 0, "No similarity search results returned"
        
        # Check that results are sorted by similarity (highest first)
        similarities = [similarity for _, similarity in results]
        assert similarities == sorted(similarities, reverse=True), "Results not sorted by similarity"
        
        # Check that the most similar result is the exact match (should be very high similarity)
        best_match, best_similarity = results[0]
        assert best_similarity > 0.99, f"Best match similarity too low: {best_similarity}"
        assert best_match.platform_number == 'TEST0000', "Best match is not the expected profile"
    
    def test_update_profile_embedding(self):
        """Test updating embedding for a specific profile"""
        if not self.test_profiles:
            pytest.skip("No test profiles available")
        
        profile_id = self.test_profiles[0]
        new_embedding = [0.2, 0.3, 0.4] + [0.1] * 381
        new_summary = "Updated test profile summary"
        
        # Update embedding
        success = self.db_manager.update_profile_embedding(
            profile_id=profile_id,
            embedding=new_embedding,
            embedding_model='test-model-v2',
            profile_summary=new_summary
        )
        
        assert success, "Failed to update profile embedding"
        
        # Verify update
        session = self.db_manager.get_session()
        try:
            profile = session.query(ArgoProfile).filter(ArgoProfile.id == profile_id).first()
            assert profile is not None, "Profile not found after update"
            assert profile.embedding_model == 'test-model-v2', "Embedding model not updated"
            assert profile.profile_summary == new_summary, "Profile summary not updated"
            # Note: Direct embedding comparison is complex due to pgvector format
        finally:
            self.db_manager.close_session(session)
    
    def test_batch_update_embeddings(self):
        """Test batch updating of embeddings"""
        if len(self.test_profiles) < 2:
            pytest.skip("Not enough test profiles available")
        
        # Prepare batch updates
        updates = []
        for i, profile_id in enumerate(self.test_profiles[:2]):
            updates.append({
                'profile_id': profile_id,
                'embedding': [0.5 + i * 0.1, 0.6 + i * 0.1, 0.7 + i * 0.1] + [0.2] * 381,
                'embedding_model': 'batch-test-model',
                'profile_summary': f'Batch updated profile {i}'
            })
        
        # Perform batch update
        updated_count = self.db_manager.batch_update_embeddings(updates, batch_size=1)
        
        assert updated_count == 2, f"Expected 2 profiles updated, got {updated_count}"
        
        # Verify updates
        session = self.db_manager.get_session()
        try:
            for i, profile_id in enumerate(self.test_profiles[:2]):
                profile = session.query(ArgoProfile).filter(ArgoProfile.id == profile_id).first()
                assert profile.embedding_model == 'batch-test-model', f"Profile {i} model not updated"
                assert profile.profile_summary == f'Batch updated profile {i}', f"Profile {i} summary not updated"
        finally:
            self.db_manager.close_session(session)
    
    def test_get_profiles_without_embeddings(self):
        """Test retrieving profiles without embeddings"""
        # Insert a profile without embedding
        profile_data = {
            'platform_number': 'NOEMBEDDING',
            'cycle_number': 1,
            'profile_date': datetime.now(),
            'latitude': 25.0,
            'longitude': 65.0,
            'level_index': 1,
            'pressure': 15.0,
            'temperature': 26.0,
            'salinity': 35.5
        }
        
        no_embedding_count = self.db_manager.batch_insert_profiles([profile_data])
        assert no_embedding_count == 1, "Failed to insert profile without embedding"
        
        # Get profiles without embeddings
        profiles_without_embeddings = self.db_manager.get_profiles_without_embeddings(limit=10)
        
        assert len(profiles_without_embeddings) > 0, "No profiles without embeddings found"
        
        # Check that returned profiles indeed have no embeddings
        for profile in profiles_without_embeddings:
            assert profile.embedding is None, f"Profile {profile.id} has embedding but shouldn't"
        
        # Clean up
        session = self.db_manager.get_session()
        try:
            session.query(ArgoProfile).filter(
                ArgoProfile.platform_number == 'NOEMBEDDING'
            ).delete()
            session.commit()
        finally:
            self.db_manager.close_session(session)
    
    def test_get_embedding_statistics(self):
        """Test embedding statistics retrieval"""
        stats = self.db_manager.get_embedding_statistics()
        
        assert isinstance(stats, dict), "Statistics should be a dictionary"
        assert 'total_profiles' in stats, "Missing total_profiles in statistics"
        assert 'profiles_with_embeddings' in stats, "Missing profiles_with_embeddings in statistics"
        assert 'profiles_without_embeddings' in stats, "Missing profiles_without_embeddings in statistics"
        assert 'embedding_coverage' in stats, "Missing embedding_coverage in statistics"
        assert 'embedding_models' in stats, "Missing embedding_models in statistics"
        
        assert stats['total_profiles'] >= 3, "Should have at least 3 test profiles"
        assert stats['profiles_with_embeddings'] >= 3, "Should have at least 3 profiles with embeddings"
        assert 0 <= stats['embedding_coverage'] <= 1, "Coverage should be between 0 and 1"
        
        # Check that our test models are present
        assert 'batch-test-model' in stats['embedding_models'], "batch-test-model not found in statistics"
    
    def test_vector_index_creation(self):
        """Test vector index creation"""
        # This should succeed since we have embeddings
        index_created = self.db_manager.create_vector_index_if_needed()
        assert index_created, "Vector index creation failed"
        
        # Verify index exists
        session = self.db_manager.get_session()
        try:
            result = session.execute(text("""
                SELECT indexname FROM pg_indexes 
                WHERE tablename = 'argo_profiles' 
                AND indexname = 'argo_profiles_embedding_idx'
            """))
            index_exists = result.fetchone() is not None
            assert index_exists, "Vector index not found after creation"
        finally:
            self.db_manager.close_session(session)
    
    def test_similarity_search_with_threshold(self):
        """Test similarity search with different thresholds"""
        query_embedding = self.test_embeddings[0]
        
        # High threshold - should return fewer results
        high_threshold_results = self.db_manager.vector_similarity_search(
            query_embedding=query_embedding,
            limit=10,
            similarity_threshold=0.95
        )
        
        # Low threshold - should return more results
        low_threshold_results = self.db_manager.vector_similarity_search(
            query_embedding=query_embedding,
            limit=10,
            similarity_threshold=0.1
        )
        
        assert len(high_threshold_results) <= len(low_threshold_results), \
            "High threshold should return fewer or equal results"
        
        # All results should meet the threshold
        for _, similarity in high_threshold_results:
            assert similarity >= 0.95, f"Result similarity {similarity} below threshold 0.95"
    
    def test_connection_pooling(self):
        """Test that connection pooling works correctly"""
        # Create multiple database managers to test pooling
        managers = [DatabaseManager(pool_size=2, max_overflow=1) for _ in range(3)]
        
        # Perform operations with each manager
        for i, manager in enumerate(managers):
            stats = manager.get_embedding_statistics()
            assert stats['total_profiles'] >= 3, f"Manager {i} couldn't retrieve statistics"
        
        # Clean up
        for manager in managers:
            manager.engine.dispose()
    
    @classmethod
    def teardown_class(cls):
        """Clean up test data"""
        if hasattr(cls, 'test_profiles') and cls.test_profiles:
            session = cls.db_manager.get_session()
            try:
                # Delete test profiles
                session.query(ArgoProfile).filter(
                    ArgoProfile.id.in_(cls.test_profiles)
                ).delete(synchronize_session=False)
                
                # Also clean up any profiles with TEST platform numbers
                session.query(ArgoProfile).filter(
                    ArgoProfile.platform_number.like('TEST%')
                ).delete(synchronize_session=False)
                
                session.commit()
                print(f"Cleaned up {len(cls.test_profiles)} test profiles")
            except Exception as e:
                session.rollback()
                print(f"Error during cleanup: {e}")
            finally:
                cls.db_manager.close_session(session)
        
        # Dispose of engine
        cls.db_manager.engine.dispose()


def test_database_manager_initialization():
    """Test DatabaseManager initialization with different parameters"""
    # Test default initialization
    db_manager = DatabaseManager()
    assert db_manager.engine is not None, "Engine not initialized"
    assert db_manager.SessionLocal is not None, "SessionLocal not initialized"
    db_manager.engine.dispose()
    
    # Test custom pool parameters
    db_manager_custom = DatabaseManager(pool_size=5, max_overflow=10)
    assert db_manager_custom.engine.pool.size() == 5, "Custom pool size not set"
    db_manager_custom.engine.dispose()


def test_argo_profile_to_dict():
    """Test ArgoProfile to_dict method"""
    # Create a mock profile (not saved to database)
    profile = ArgoProfile(
        platform_number='TEST9999',
        cycle_number=1,
        latitude=20.0,
        longitude=60.0,
        level_index=1,
        pressure=10.0,
        temperature=25.0,
        salinity=35.0,
        profile_summary='Test profile',
        embedding_model='test-model'
    )
    
    profile_dict = profile.to_dict()
    
    assert isinstance(profile_dict, dict), "to_dict should return a dictionary"
    assert profile_dict['platform_number'] == 'TEST9999', "Platform number not in dict"
    assert profile_dict['latitude'] == 20.0, "Latitude not in dict"
    assert profile_dict['profile_summary'] == 'Test profile', "Profile summary not in dict"


if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v', '--tb=short'])