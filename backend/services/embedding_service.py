"""
Embedding Service Implementation for FloatChat RAG Pipeline

This module provides embedding computation services with Gemini API and local fallback.
Supports both cloud-based (Gemini) and local (sentence-transformers) embedding generation.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import numpy as np

# Import for local embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Import for Gemini embeddings
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

logger = logging.getLogger(__name__)


class EmbeddingService(ABC):
    """Abstract base class for embedding services."""
    
    @abstractmethod
    def compute_embedding(self, text: str) -> List[float]:
        """Compute embedding for a single text."""
        pass
    
    @abstractmethod
    def batch_compute_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Compute embeddings for multiple texts."""
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this service."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the service is available and properly configured."""
        pass
    
    @abstractmethod
    def get_service_name(self) -> str:
        """Get the name of the embedding service."""
        pass


class GeminiEmbeddingService(EmbeddingService):
    """Gemini API-based embedding service with error handling."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "models/text-embedding-004"):
        """
        Initialize Gemini embedding service.
        
        Args:
            api_key: Gemini API key. If None, will try to get from environment.
            model_name: Name of the Gemini embedding model to use.
        """
        self.api_key = api_key or os.getenv('AIzaSyD8PWnbVRU_B45ilYUF22TGZtKRK3Ixswk')
        self.model_name = model_name
        self._client = None
        self._dimension = None
        
        if not GEMINI_AVAILABLE:
            logger.warning("google-generativeai package not available")
            return
            
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self._client = genai
                logger.info(f"Gemini embedding service initialized with model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}")
                self._client = None
        else:
            logger.warning("No Gemini API key provided")
    
    def compute_embedding(self, text: str) -> List[float]:
        """Compute embedding for a single text using Gemini API."""
        if not self.is_available():
            raise RuntimeError("Gemini embedding service is not available")
        
        try:
            result = self._client.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document"
            )
            embedding = result['embedding']
            
            # Cache dimension on first successful call
            if self._dimension is None:
                self._dimension = len(embedding)
                logger.info(f"Gemini embedding dimension: {self._dimension}")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Gemini embedding computation failed: {e}")
            raise RuntimeError(f"Failed to compute Gemini embedding: {e}")
    
    def batch_compute_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Compute embeddings for multiple texts using Gemini API."""
        if not self.is_available():
            raise RuntimeError("Gemini embedding service is not available")
        
        embeddings = []
        for text in texts:
            try:
                embedding = self.compute_embedding(text)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Failed to compute embedding for text: {text[:50]}... Error: {e}")
                raise
        
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of Gemini embeddings."""
        if self._dimension is None:
            # Try to determine dimension with a test embedding
            try:
                test_embedding = self.compute_embedding("test")
                self._dimension = len(test_embedding)
            except Exception:
                # Default dimension for text-embedding-004
                self._dimension = 768
                logger.warning(f"Could not determine Gemini embedding dimension, using default: {self._dimension}")
        
        return self._dimension
    
    def is_available(self) -> bool:
        """Check if Gemini service is available."""
        return (GEMINI_AVAILABLE and 
                self._client is not None and 
                self.api_key is not None)
    
    def get_service_name(self) -> str:
        """Get the service name."""
        return f"Gemini ({self.model_name})"


class LocalEmbeddingService(EmbeddingService):
    """Local sentence-transformers based embedding service."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize local embedding service.
        
        Args:
            model_name: Name of the sentence-transformers model to use.
        """
        self.model_name = model_name
        self._model = None
        self._dimension = None
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("sentence-transformers package not available")
            return
        
        try:
            logger.info(f"Loading local embedding model: {model_name}")
            self._model = SentenceTransformer(model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(f"Local embedding service initialized. Dimension: {self._dimension}")
        except Exception as e:
            logger.error(f"Failed to load local embedding model: {e}")
            self._model = None
    
    def compute_embedding(self, text: str) -> List[float]:
        """Compute embedding for a single text using local model."""
        if not self.is_available():
            raise RuntimeError("Local embedding service is not available")
        
        try:
            embedding = self._model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Local embedding computation failed: {e}")
            raise RuntimeError(f"Failed to compute local embedding: {e}")
    
    def batch_compute_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Compute embeddings for multiple texts using local model."""
        if not self.is_available():
            raise RuntimeError("Local embedding service is not available")
        
        try:
            embeddings = self._model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Batch embedding computation failed: {e}")
            raise RuntimeError(f"Failed to compute batch embeddings: {e}")
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of local embeddings."""
        if self._dimension is None:
            if self.is_available():
                self._dimension = self._model.get_sentence_embedding_dimension()
            else:
                # Default dimension for all-MiniLM-L6-v2
                self._dimension = 384
                logger.warning(f"Could not determine local embedding dimension, using default: {self._dimension}")
        
        return self._dimension
    
    def is_available(self) -> bool:
        """Check if local service is available."""
        return SENTENCE_TRANSFORMERS_AVAILABLE and self._model is not None
    
    def get_service_name(self) -> str:
        """Get the service name."""
        return f"Local ({self.model_name})"


class EmbeddingServiceManager:
    """
    Manager class that handles automatic fallback between Gemini and local embedding services.
    """
    
    def __init__(self, 
                 gemini_api_key: Optional[str] = None,
                 gemini_model: str = "models/text-embedding-004",
                 local_model: str = "all-MiniLM-L6-v2",
                 prefer_gemini: bool = True):
        """
        Initialize embedding service manager with fallback capability.
        
        Args:
            gemini_api_key: Gemini API key. If None, will try to get from environment.
            gemini_model: Gemini model name to use.
            local_model: Local sentence-transformers model name to use.
            prefer_gemini: Whether to prefer Gemini over local service when both are available.
        """
        self.prefer_gemini = prefer_gemini
        
        # Initialize services
        self.gemini_service = GeminiEmbeddingService(gemini_api_key, gemini_model)
        self.local_service = LocalEmbeddingService(local_model)
        
        # Determine primary and fallback services
        self._setup_service_priority()
        
        logger.info(f"Embedding service manager initialized:")
        logger.info(f"  Primary: {self.primary_service.get_service_name() if self.primary_service else 'None'}")
        logger.info(f"  Fallback: {self.fallback_service.get_service_name() if self.fallback_service else 'None'}")
    
    def _setup_service_priority(self):
        """Setup primary and fallback services based on availability and preference."""
        gemini_available = self.gemini_service.is_available()
        local_available = self.local_service.is_available()
        
        if self.prefer_gemini and gemini_available:
            self.primary_service = self.gemini_service
            self.fallback_service = self.local_service if local_available else None
        elif local_available:
            self.primary_service = self.local_service
            self.fallback_service = self.gemini_service if gemini_available else None
        elif gemini_available:
            self.primary_service = self.gemini_service
            self.fallback_service = None
        else:
            self.primary_service = None
            self.fallback_service = None
            logger.error("No embedding services are available!")
    
    def compute_embedding(self, text: str) -> List[float]:
        """
        Compute embedding with automatic fallback.
        
        Args:
            text: Text to embed.
            
        Returns:
            List of embedding values.
            
        Raises:
            RuntimeError: If no embedding service is available or all services fail.
        """
        if not self.primary_service:
            raise RuntimeError("No embedding services are available")
        
        # Try primary service
        try:
            return self.primary_service.compute_embedding(text)
        except Exception as e:
            logger.warning(f"Primary embedding service failed: {e}")
            
            # Try fallback service
            if self.fallback_service:
                try:
                    logger.info("Attempting fallback embedding service")
                    return self.fallback_service.compute_embedding(text)
                except Exception as fallback_error:
                    logger.error(f"Fallback embedding service also failed: {fallback_error}")
                    raise RuntimeError(f"All embedding services failed. Primary: {e}, Fallback: {fallback_error}")
            else:
                raise RuntimeError(f"Primary embedding service failed and no fallback available: {e}")
    
    def batch_compute_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Compute embeddings for multiple texts with automatic fallback.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            List of embedding lists.
            
        Raises:
            RuntimeError: If no embedding service is available or all services fail.
        """
        if not self.primary_service:
            raise RuntimeError("No embedding services are available")
        
        # Try primary service
        try:
            return self.primary_service.batch_compute_embeddings(texts)
        except Exception as e:
            logger.warning(f"Primary embedding service failed for batch: {e}")
            
            # Try fallback service
            if self.fallback_service:
                try:
                    logger.info("Attempting fallback embedding service for batch")
                    return self.fallback_service.batch_compute_embeddings(texts)
                except Exception as fallback_error:
                    logger.error(f"Fallback embedding service also failed for batch: {fallback_error}")
                    raise RuntimeError(f"All embedding services failed for batch. Primary: {e}, Fallback: {fallback_error}")
            else:
                raise RuntimeError(f"Primary embedding service failed for batch and no fallback available: {e}")
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings from the primary service."""
        if not self.primary_service:
            raise RuntimeError("No embedding services are available")
        
        return self.primary_service.get_embedding_dimension()
    
    def get_active_service_name(self) -> str:
        """Get the name of the currently active primary service."""
        if not self.primary_service:
            return "None"
        return self.primary_service.get_service_name()
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get status information about all services."""
        return {
            "primary_service": {
                "name": self.primary_service.get_service_name() if self.primary_service else "None",
                "available": self.primary_service.is_available() if self.primary_service else False,
                "dimension": self.primary_service.get_embedding_dimension() if self.primary_service else None
            },
            "fallback_service": {
                "name": self.fallback_service.get_service_name() if self.fallback_service else "None",
                "available": self.fallback_service.is_available() if self.fallback_service else False,
                "dimension": self.fallback_service.get_embedding_dimension() if self.fallback_service else None
            },
            "gemini_available": self.gemini_service.is_available(),
            "local_available": self.local_service.is_available()
        }


# Factory function for easy service creation
def create_embedding_service(prefer_gemini: bool = True, 
                           gemini_api_key: Optional[str] = None) -> EmbeddingServiceManager:
    """
    Factory function to create an embedding service manager.
    
    Args:
        prefer_gemini: Whether to prefer Gemini over local service.
        gemini_api_key: Gemini API key. If None, will try to get from environment.
        
    Returns:
        Configured EmbeddingServiceManager instance.
    """
    return EmbeddingServiceManager(
        gemini_api_key=gemini_api_key,
        prefer_gemini=prefer_gemini
    )