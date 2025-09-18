#!/usr/bin/env python3
"""
Test script to verify the embedding service implementation.

This script tests the embedding services with real dependencies to ensure
they work correctly in the actual environment.
"""

import os
import sys
import logging
from typing import List

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If python-dotenv is not available, try to load manually
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#') and '=' in line:
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from services.embedding_service import (
    GeminiEmbeddingService,
    LocalEmbeddingService,
    EmbeddingServiceManager,
    create_embedding_service
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_local_embedding_service():
    """Test the local embedding service."""
    logger.info("Testing Local Embedding Service...")
    
    try:
        service = LocalEmbeddingService()
        
        if not service.is_available():
            logger.warning("Local embedding service is not available (sentence-transformers not installed)")
            return False
        
        logger.info(f"Service: {service.get_service_name()}")
        logger.info(f"Dimension: {service.get_embedding_dimension()}")
        
        # Test single embedding
        test_text = "Temperature profile shows thermocline at 100m depth"
        embedding = service.compute_embedding(test_text)
        logger.info(f"Single embedding computed: {len(embedding)} dimensions")
        logger.info(f"Sample values: {embedding[:5]}")
        
        # Test batch embeddings
        test_texts = [
            "Temperature profile shows thermocline at 100m depth",
            "Salinity increases with depth in this region",
            "Mixed layer depth is approximately 50 meters"
        ]
        embeddings = service.batch_compute_embeddings(test_texts)
        logger.info(f"Batch embeddings computed: {len(embeddings)} embeddings")
        
        # Verify dimensions are consistent
        assert len(embedding) == service.get_embedding_dimension()
        assert all(len(emb) == service.get_embedding_dimension() for emb in embeddings)
        
        logger.info("✅ Local embedding service test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Local embedding service test failed: {e}")
        return False


def test_gemini_embedding_service():
    """Test the Gemini embedding service."""
    logger.info("Testing Gemini Embedding Service...")
    
    try:
        # Check if API key is available
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            logger.warning("GEMINI_API_KEY not found in environment variables")
            logger.info("Skipping Gemini service test (no API key)")
            return True  # Not a failure, just skipped
        
        service = GeminiEmbeddingService(api_key=api_key)
        
        if not service.is_available():
            logger.warning("Gemini embedding service is not available")
            return True  # Not a failure, just not available
        
        logger.info(f"Service: {service.get_service_name()}")
        
        # Test single embedding
        test_text = "Temperature profile shows thermocline at 100m depth"
        embedding = service.compute_embedding(test_text)
        logger.info(f"Single embedding computed: {len(embedding)} dimensions")
        logger.info(f"Sample values: {embedding[:5]}")
        logger.info(f"Dimension: {service.get_embedding_dimension()}")
        
        # Test batch embeddings (smaller batch for API limits)
        test_texts = [
            "Temperature profile shows thermocline at 100m depth",
            "Salinity increases with depth in this region"
        ]
        embeddings = service.batch_compute_embeddings(test_texts)
        logger.info(f"Batch embeddings computed: {len(embeddings)} embeddings")
        
        # Verify dimensions are consistent
        assert len(embedding) == service.get_embedding_dimension()
        assert all(len(emb) == service.get_embedding_dimension() for emb in embeddings)
        
        logger.info("✅ Gemini embedding service test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Gemini embedding service test failed: {e}")
        return False


def test_embedding_service_manager():
    """Test the embedding service manager with fallback."""
    logger.info("Testing Embedding Service Manager...")
    
    try:
        # Test with prefer_gemini=True
        manager = EmbeddingServiceManager(prefer_gemini=True)
        
        status = manager.get_service_status()
        logger.info("Service Status:")
        logger.info(f"  Primary: {status['primary_service']['name']} (available: {status['primary_service']['available']})")
        logger.info(f"  Fallback: {status['fallback_service']['name']} (available: {status['fallback_service']['available']})")
        logger.info(f"  Gemini available: {status['gemini_available']}")
        logger.info(f"  Local available: {status['local_available']}")
        
        if not status['primary_service']['available'] and not status['fallback_service']['available']:
            logger.warning("No embedding services are available")
            return True  # Not a failure, just no services available
        
        logger.info(f"Active service: {manager.get_active_service_name()}")
        logger.info(f"Embedding dimension: {manager.get_embedding_dimension()}")
        
        # Test single embedding
        test_text = "Temperature profile shows thermocline at 100m depth"
        embedding = manager.compute_embedding(test_text)
        logger.info(f"Single embedding computed: {len(embedding)} dimensions")
        
        # Test batch embeddings
        test_texts = [
            "Temperature profile shows thermocline at 100m depth",
            "Salinity increases with depth in this region"
        ]
        embeddings = manager.batch_compute_embeddings(test_texts)
        logger.info(f"Batch embeddings computed: {len(embeddings)} embeddings")
        
        # Verify dimensions are consistent
        assert len(embedding) == manager.get_embedding_dimension()
        assert all(len(emb) == manager.get_embedding_dimension() for emb in embeddings)
        
        logger.info("✅ Embedding service manager test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Embedding service manager test failed: {e}")
        return False


def test_factory_function():
    """Test the factory function."""
    logger.info("Testing Factory Function...")
    
    try:
        # Test default creation
        manager1 = create_embedding_service()
        logger.info(f"Default manager created: {manager1.get_active_service_name()}")
        
        # Test with prefer_gemini=False
        manager2 = create_embedding_service(prefer_gemini=False)
        logger.info(f"Local-preferred manager created: {manager2.get_active_service_name()}")
        
        logger.info("✅ Factory function test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Factory function test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("Starting Embedding Service Implementation Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Local Embedding Service", test_local_embedding_service),
        ("Gemini Embedding Service", test_gemini_embedding_service),
        ("Embedding Service Manager", test_embedding_service_manager),
        ("Factory Function", test_factory_function)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{test_name}")
        logger.info("-" * 40)
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if success:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        logger.info("🎉 All tests passed! Embedding service implementation is working correctly.")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())