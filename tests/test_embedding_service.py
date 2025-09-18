"""
Unit tests for embedding services.

Tests both Gemini and local embedding services, including fallback behavior.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import os
import sys
from typing import List

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from services.embedding_service import (
    EmbeddingService,
    GeminiEmbeddingService,
    LocalEmbeddingService,
    EmbeddingServiceManager,
    create_embedding_service
)


class TestEmbeddingServiceBase(unittest.TestCase):
    """Base test class with common utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_text = "Temperature profile shows thermocline at 100m depth"
        self.sample_texts = [
            "Temperature profile shows thermocline at 100m depth",
            "Salinity increases with depth in this region",
            "Mixed layer depth is approximately 50 meters"
        ]
        self.expected_dimension = 384  # all-MiniLM-L6-v2 dimension


class TestGeminiEmbeddingService(TestEmbeddingServiceBase):
    """Test cases for Gemini embedding service."""
    
    def setUp(self):
        super().setUp()
        self.api_key = "test_api_key"
        self.mock_embedding = [0.1] * 768  # Typical Gemini dimension
    
    @patch('services.embedding_service.GEMINI_AVAILABLE', True)
    @patch('services.embedding_service.genai')
    def test_initialization_success(self, mock_genai):
        """Test successful initialization of Gemini service."""
        service = GeminiEmbeddingService(api_key=self.api_key)
        
        self.assertEqual(service.api_key, self.api_key)
        self.assertEqual(service.model_name, "models/text-embedding-004")
        mock_genai.configure.assert_called_once_with(api_key=self.api_key)
        self.assertTrue(service.is_available())
    
    @patch('services.embedding_service.GEMINI_AVAILABLE', False)
    def test_initialization_no_package(self):
        """Test initialization when Gemini package is not available."""
        service = GeminiEmbeddingService(api_key=self.api_key)
        
        self.assertFalse(service.is_available())
    
    def test_initialization_no_api_key(self):
        """Test initialization without API key."""
        with patch.dict(os.environ, {}, clear=True):
            service = GeminiEmbeddingService()
            self.assertFalse(service.is_available())
    
    @patch('services.embedding_service.GEMINI_AVAILABLE', True)
    @patch('services.embedding_service.genai')
    def test_compute_embedding_success(self, mock_genai):
        """Test successful embedding computation."""
        mock_genai.embed_content.return_value = {'embedding': self.mock_embedding}
        
        service = GeminiEmbeddingService(api_key=self.api_key)
        result = service.compute_embedding(self.sample_text)
        
        self.assertEqual(result, self.mock_embedding)
        mock_genai.embed_content.assert_called_once_with(
            model="models/text-embedding-004",
            content=self.sample_text,
            task_type="retrieval_document"
        )
    
    @patch('services.embedding_service.GEMINI_AVAILABLE', True)
    @patch('services.embedding_service.genai')
    def test_compute_embedding_api_error(self, mock_genai):
        """Test embedding computation with API error."""
        mock_genai.embed_content.side_effect = Exception("API Error")
        
        service = GeminiEmbeddingService(api_key=self.api_key)
        
        with self.assertRaises(RuntimeError) as context:
            service.compute_embedding(self.sample_text)
        
        self.assertIn("Failed to compute Gemini embedding", str(context.exception))
    
    @patch('services.embedding_service.GEMINI_AVAILABLE', True)
    @patch('services.embedding_service.genai')
    def test_batch_compute_embeddings(self, mock_genai):
        """Test batch embedding computation."""
        mock_genai.embed_content.return_value = {'embedding': self.mock_embedding}
        
        service = GeminiEmbeddingService(api_key=self.api_key)
        results = service.batch_compute_embeddings(self.sample_texts)
        
        self.assertEqual(len(results), len(self.sample_texts))
        self.assertEqual(results[0], self.mock_embedding)
        self.assertEqual(mock_genai.embed_content.call_count, len(self.sample_texts))
    
    @patch('services.embedding_service.GEMINI_AVAILABLE', True)
    @patch('services.embedding_service.genai')
    def test_get_embedding_dimension(self, mock_genai):
        """Test getting embedding dimension."""
        mock_genai.embed_content.return_value = {'embedding': self.mock_embedding}
        
        service = GeminiEmbeddingService(api_key=self.api_key)
        dimension = service.get_embedding_dimension()
        
        self.assertEqual(dimension, len(self.mock_embedding))
    
    def test_service_unavailable_operations(self):
        """Test operations when service is unavailable."""
        service = GeminiEmbeddingService()  # No API key
        
        with self.assertRaises(RuntimeError):
            service.compute_embedding(self.sample_text)
        
        with self.assertRaises(RuntimeError):
            service.batch_compute_embeddings(self.sample_texts)


class TestLocalEmbeddingService(TestEmbeddingServiceBase):
    """Test cases for local embedding service."""
    
    def setUp(self):
        super().setUp()
        self.mock_embedding = [0.1] * self.expected_dimension
    
    @patch('services.embedding_service.SENTENCE_TRANSFORMERS_AVAILABLE', True)
    @patch('services.embedding_service.SentenceTransformer')
    def test_initialization_success(self, mock_sentence_transformer):
        """Test successful initialization of local service."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = self.expected_dimension
        mock_sentence_transformer.return_value = mock_model
        
        service = LocalEmbeddingService()
        
        self.assertTrue(service.is_available())
        self.assertEqual(service.get_embedding_dimension(), self.expected_dimension)
        mock_sentence_transformer.assert_called_once_with("all-MiniLM-L6-v2")
    
    @patch('services.embedding_service.SENTENCE_TRANSFORMERS_AVAILABLE', False)
    def test_initialization_no_package(self):
        """Test initialization when sentence-transformers package is not available."""
        service = LocalEmbeddingService()
        
        self.assertFalse(service.is_available())
    
    @patch('services.embedding_service.SENTENCE_TRANSFORMERS_AVAILABLE', True)
    @patch('services.embedding_service.SentenceTransformer')
    def test_initialization_model_load_error(self, mock_sentence_transformer):
        """Test initialization with model loading error."""
        mock_sentence_transformer.side_effect = Exception("Model load error")
        
        service = LocalEmbeddingService()
        
        self.assertFalse(service.is_available())
    
    @patch('services.embedding_service.SENTENCE_TRANSFORMERS_AVAILABLE', True)
    @patch('services.embedding_service.SentenceTransformer')
    def test_compute_embedding_success(self, mock_sentence_transformer):
        """Test successful embedding computation."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = self.expected_dimension
        mock_model.encode.return_value = Mock()
        mock_model.encode.return_value.tolist.return_value = self.mock_embedding
        mock_sentence_transformer.return_value = mock_model
        
        service = LocalEmbeddingService()
        result = service.compute_embedding(self.sample_text)
        
        self.assertEqual(result, self.mock_embedding)
        mock_model.encode.assert_called_once_with(self.sample_text, convert_to_numpy=True)
    
    @patch('services.embedding_service.SENTENCE_TRANSFORMERS_AVAILABLE', True)
    @patch('services.embedding_service.SentenceTransformer')
    def test_batch_compute_embeddings(self, mock_sentence_transformer):
        """Test batch embedding computation."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = self.expected_dimension
        mock_embeddings = Mock()
        mock_embeddings.tolist.return_value = [self.mock_embedding] * len(self.sample_texts)
        mock_model.encode.return_value = mock_embeddings
        mock_sentence_transformer.return_value = mock_model
        
        service = LocalEmbeddingService()
        results = service.batch_compute_embeddings(self.sample_texts)
        
        self.assertEqual(len(results), len(self.sample_texts))
        self.assertEqual(results[0], self.mock_embedding)
        mock_model.encode.assert_called_once_with(self.sample_texts, convert_to_numpy=True)
    
    @patch('services.embedding_service.SENTENCE_TRANSFORMERS_AVAILABLE', False)
    def test_service_unavailable_operations(self):
        """Test operations when service is unavailable."""
        service = LocalEmbeddingService()  # No sentence-transformers available
        
        with self.assertRaises(RuntimeError):
            service.compute_embedding(self.sample_text)
        
        with self.assertRaises(RuntimeError):
            service.batch_compute_embeddings(self.sample_texts)


class TestEmbeddingServiceManager(TestEmbeddingServiceBase):
    """Test cases for embedding service manager with fallback."""
    
    def setUp(self):
        super().setUp()
        self.mock_embedding = [0.1] * self.expected_dimension
    
    def create_mock_service(self, available=True, embedding=None):
        """Create a mock embedding service."""
        mock_service = Mock(spec=EmbeddingService)
        mock_service.is_available.return_value = available
        mock_service.get_service_name.return_value = "Mock Service"
        mock_service.get_embedding_dimension.return_value = self.expected_dimension
        
        if embedding:
            mock_service.compute_embedding.return_value = embedding
            mock_service.batch_compute_embeddings.return_value = [embedding] * 3
        
        return mock_service
    
    @patch('services.embedding_service.GeminiEmbeddingService')
    @patch('services.embedding_service.LocalEmbeddingService')
    def test_initialization_prefer_gemini_both_available(self, mock_local_cls, mock_gemini_cls):
        """Test initialization when both services are available and Gemini is preferred."""
        mock_gemini = self.create_mock_service(available=True, embedding=self.mock_embedding)
        mock_local = self.create_mock_service(available=True, embedding=self.mock_embedding)
        mock_gemini_cls.return_value = mock_gemini
        mock_local_cls.return_value = mock_local
        
        manager = EmbeddingServiceManager(prefer_gemini=True)
        
        self.assertEqual(manager.primary_service, mock_gemini)
        self.assertEqual(manager.fallback_service, mock_local)
    
    @patch('services.embedding_service.GeminiEmbeddingService')
    @patch('services.embedding_service.LocalEmbeddingService')
    def test_initialization_prefer_local_both_available(self, mock_local_cls, mock_gemini_cls):
        """Test initialization when both services are available and local is preferred."""
        mock_gemini = self.create_mock_service(available=True, embedding=self.mock_embedding)
        mock_local = self.create_mock_service(available=True, embedding=self.mock_embedding)
        mock_gemini_cls.return_value = mock_gemini
        mock_local_cls.return_value = mock_local
        
        manager = EmbeddingServiceManager(prefer_gemini=False)
        
        self.assertEqual(manager.primary_service, mock_local)
        self.assertEqual(manager.fallback_service, mock_gemini)
    
    @patch('services.embedding_service.GeminiEmbeddingService')
    @patch('services.embedding_service.LocalEmbeddingService')
    def test_initialization_only_gemini_available(self, mock_local_cls, mock_gemini_cls):
        """Test initialization when only Gemini is available."""
        mock_gemini = self.create_mock_service(available=True, embedding=self.mock_embedding)
        mock_local = self.create_mock_service(available=False)
        mock_gemini_cls.return_value = mock_gemini
        mock_local_cls.return_value = mock_local
        
        manager = EmbeddingServiceManager()
        
        self.assertEqual(manager.primary_service, mock_gemini)
        self.assertIsNone(manager.fallback_service)
    
    @patch('services.embedding_service.GeminiEmbeddingService')
    @patch('services.embedding_service.LocalEmbeddingService')
    def test_initialization_only_local_available(self, mock_local_cls, mock_gemini_cls):
        """Test initialization when only local service is available."""
        mock_gemini = self.create_mock_service(available=False)
        mock_local = self.create_mock_service(available=True, embedding=self.mock_embedding)
        mock_gemini_cls.return_value = mock_gemini
        mock_local_cls.return_value = mock_local
        
        manager = EmbeddingServiceManager()
        
        self.assertEqual(manager.primary_service, mock_local)
        self.assertIsNone(manager.fallback_service)
    
    @patch('services.embedding_service.GeminiEmbeddingService')
    @patch('services.embedding_service.LocalEmbeddingService')
    def test_initialization_no_services_available(self, mock_local_cls, mock_gemini_cls):
        """Test initialization when no services are available."""
        mock_gemini = self.create_mock_service(available=False)
        mock_local = self.create_mock_service(available=False)
        mock_gemini_cls.return_value = mock_gemini
        mock_local_cls.return_value = mock_local
        
        manager = EmbeddingServiceManager()
        
        self.assertIsNone(manager.primary_service)
        self.assertIsNone(manager.fallback_service)
    
    @patch('services.embedding_service.GeminiEmbeddingService')
    @patch('services.embedding_service.LocalEmbeddingService')
    def test_compute_embedding_primary_success(self, mock_local_cls, mock_gemini_cls):
        """Test successful embedding computation with primary service."""
        mock_gemini = self.create_mock_service(available=True, embedding=self.mock_embedding)
        mock_local = self.create_mock_service(available=True, embedding=self.mock_embedding)
        mock_gemini_cls.return_value = mock_gemini
        mock_local_cls.return_value = mock_local
        
        manager = EmbeddingServiceManager(prefer_gemini=True)
        result = manager.compute_embedding(self.sample_text)
        
        self.assertEqual(result, self.mock_embedding)
        mock_gemini.compute_embedding.assert_called_once_with(self.sample_text)
        mock_local.compute_embedding.assert_not_called()
    
    @patch('services.embedding_service.GeminiEmbeddingService')
    @patch('services.embedding_service.LocalEmbeddingService')
    def test_compute_embedding_fallback_success(self, mock_local_cls, mock_gemini_cls):
        """Test successful embedding computation with fallback service."""
        mock_gemini = self.create_mock_service(available=True, embedding=self.mock_embedding)
        mock_local = self.create_mock_service(available=True, embedding=self.mock_embedding)
        
        # Make primary service fail
        mock_gemini.compute_embedding.side_effect = Exception("Primary failed")
        
        mock_gemini_cls.return_value = mock_gemini
        mock_local_cls.return_value = mock_local
        
        manager = EmbeddingServiceManager(prefer_gemini=True)
        result = manager.compute_embedding(self.sample_text)
        
        self.assertEqual(result, self.mock_embedding)
        mock_gemini.compute_embedding.assert_called_once_with(self.sample_text)
        mock_local.compute_embedding.assert_called_once_with(self.sample_text)
    
    @patch('services.embedding_service.GeminiEmbeddingService')
    @patch('services.embedding_service.LocalEmbeddingService')
    def test_compute_embedding_both_fail(self, mock_local_cls, mock_gemini_cls):
        """Test embedding computation when both services fail."""
        mock_gemini = self.create_mock_service(available=True)
        mock_local = self.create_mock_service(available=True)
        
        # Make both services fail
        mock_gemini.compute_embedding.side_effect = Exception("Primary failed")
        mock_local.compute_embedding.side_effect = Exception("Fallback failed")
        
        mock_gemini_cls.return_value = mock_gemini
        mock_local_cls.return_value = mock_local
        
        manager = EmbeddingServiceManager(prefer_gemini=True)
        
        with self.assertRaises(RuntimeError) as context:
            manager.compute_embedding(self.sample_text)
        
        self.assertIn("All embedding services failed", str(context.exception))
    
    @patch('services.embedding_service.GeminiEmbeddingService')
    @patch('services.embedding_service.LocalEmbeddingService')
    def test_compute_embedding_no_services(self, mock_local_cls, mock_gemini_cls):
        """Test embedding computation when no services are available."""
        mock_gemini = self.create_mock_service(available=False)
        mock_local = self.create_mock_service(available=False)
        mock_gemini_cls.return_value = mock_gemini
        mock_local_cls.return_value = mock_local
        
        manager = EmbeddingServiceManager()
        
        with self.assertRaises(RuntimeError) as context:
            manager.compute_embedding(self.sample_text)
        
        self.assertIn("No embedding services are available", str(context.exception))
    
    @patch('services.embedding_service.GeminiEmbeddingService')
    @patch('services.embedding_service.LocalEmbeddingService')
    def test_batch_compute_embeddings_fallback(self, mock_local_cls, mock_gemini_cls):
        """Test batch embedding computation with fallback."""
        mock_gemini = self.create_mock_service(available=True)
        mock_local = self.create_mock_service(available=True, embedding=self.mock_embedding)
        
        # Make primary service fail for batch
        mock_gemini.batch_compute_embeddings.side_effect = Exception("Batch failed")
        
        mock_gemini_cls.return_value = mock_gemini
        mock_local_cls.return_value = mock_local
        
        manager = EmbeddingServiceManager(prefer_gemini=True)
        results = manager.batch_compute_embeddings(self.sample_texts)
        
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0], self.mock_embedding)
        mock_gemini.batch_compute_embeddings.assert_called_once_with(self.sample_texts)
        mock_local.batch_compute_embeddings.assert_called_once_with(self.sample_texts)
    
    @patch('services.embedding_service.GeminiEmbeddingService')
    @patch('services.embedding_service.LocalEmbeddingService')
    def test_get_service_status(self, mock_local_cls, mock_gemini_cls):
        """Test getting service status information."""
        mock_gemini = self.create_mock_service(available=True, embedding=self.mock_embedding)
        mock_local = self.create_mock_service(available=True, embedding=self.mock_embedding)
        mock_gemini.get_service_name.return_value = "Gemini Service"
        mock_local.get_service_name.return_value = "Local Service"
        
        mock_gemini_cls.return_value = mock_gemini
        mock_local_cls.return_value = mock_local
        
        manager = EmbeddingServiceManager(prefer_gemini=True)
        status = manager.get_service_status()
        
        self.assertEqual(status["primary_service"]["name"], "Gemini Service")
        self.assertEqual(status["fallback_service"]["name"], "Local Service")
        self.assertTrue(status["primary_service"]["available"])
        self.assertTrue(status["fallback_service"]["available"])
        self.assertTrue(status["gemini_available"])
        self.assertTrue(status["local_available"])


class TestFactoryFunction(TestEmbeddingServiceBase):
    """Test cases for the factory function."""
    
    @patch('services.embedding_service.EmbeddingServiceManager')
    def test_create_embedding_service_default(self, mock_manager_cls):
        """Test factory function with default parameters."""
        mock_manager = Mock()
        mock_manager_cls.return_value = mock_manager
        
        result = create_embedding_service()
        
        mock_manager_cls.assert_called_once_with(
            gemini_api_key=None,
            prefer_gemini=True
        )
        self.assertEqual(result, mock_manager)
    
    @patch('services.embedding_service.EmbeddingServiceManager')
    def test_create_embedding_service_custom(self, mock_manager_cls):
        """Test factory function with custom parameters."""
        mock_manager = Mock()
        mock_manager_cls.return_value = mock_manager
        api_key = "test_key"
        
        result = create_embedding_service(prefer_gemini=False, gemini_api_key=api_key)
        
        mock_manager_cls.assert_called_once_with(
            gemini_api_key=api_key,
            prefer_gemini=False
        )
        self.assertEqual(result, mock_manager)


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.INFO)
    
    unittest.main()