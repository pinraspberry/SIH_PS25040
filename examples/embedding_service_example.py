#!/usr/bin/env python3
"""
Example usage of the embedding service for FloatChat RAG pipeline.

This script demonstrates how to use the embedding services in a real application.
"""

import os
import sys
import logging

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If python-dotenv is not available, try to load manually
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#') and '=' in line:
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from services.embedding_service import create_embedding_service

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Demonstrate embedding service usage."""
    logger.info("FloatChat Embedding Service Example")
    logger.info("=" * 50)
    
    # Create embedding service with automatic fallback
    embedding_service = create_embedding_service(prefer_gemini=True)
    
    # Check service status
    status = embedding_service.get_service_status()
    logger.info(f"Active service: {embedding_service.get_active_service_name()}")
    logger.info(f"Embedding dimension: {embedding_service.get_embedding_dimension()}")
    
    # Sample oceanographic texts for embedding
    sample_texts = [
        "Temperature profile shows thermocline at 100m depth with rapid temperature decrease",
        "Salinity increases linearly with depth from 34.5 to 35.2 PSU in the upper 500m",
        "Mixed layer depth varies seasonally from 30m in summer to 80m in winter",
        "Oxygen minimum zone detected between 200-800m depth with concentrations below 2 ml/L",
        "Chlorophyll maximum observed at 75m depth indicating deep chlorophyll maximum layer"
    ]
    
    logger.info("\nComputing embeddings for oceanographic data...")
    
    try:
        # Compute embeddings for all texts
        embeddings = embedding_service.batch_compute_embeddings(sample_texts)
        
        logger.info(f"Successfully computed {len(embeddings)} embeddings")
        logger.info(f"Each embedding has {len(embeddings[0])} dimensions")
        
        # Show sample embedding values
        logger.info("\nSample embedding values:")
        for i, (text, embedding) in enumerate(zip(sample_texts, embeddings)):
            logger.info(f"Text {i+1}: {text[:50]}...")
            logger.info(f"  Embedding preview: {embedding[:5]} ... {embedding[-5:]}")
            logger.info(f"  Magnitude: {sum(x*x for x in embedding)**0.5:.4f}")
        
        # Demonstrate similarity computation (cosine similarity)
        logger.info("\nComputing similarity between first two texts:")
        emb1, emb2 = embeddings[0], embeddings[1]
        
        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(emb1, emb2))
        magnitude1 = sum(x * x for x in emb1) ** 0.5
        magnitude2 = sum(x * x for x in emb2) ** 0.5
        cosine_similarity = dot_product / (magnitude1 * magnitude2)
        
        logger.info(f"Cosine similarity: {cosine_similarity:.4f}")
        
        # Single embedding example
        logger.info("\nComputing single embedding...")
        query_text = "What is the temperature at 200 meters depth?"
        query_embedding = embedding_service.compute_embedding(query_text)
        logger.info(f"Query: {query_text}")
        logger.info(f"Query embedding dimension: {len(query_embedding)}")
        
        # Find most similar text to query
        similarities = []
        for i, text_embedding in enumerate(embeddings):
            dot_product = sum(a * b for a, b in zip(query_embedding, text_embedding))
            magnitude1 = sum(x * x for x in query_embedding) ** 0.5
            magnitude2 = sum(x * x for x in text_embedding) ** 0.5
            similarity = dot_product / (magnitude1 * magnitude2)
            similarities.append((i, similarity, sample_texts[i]))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        logger.info("\nMost similar texts to query:")
        for i, (idx, sim, text) in enumerate(similarities[:3]):
            logger.info(f"{i+1}. Similarity: {sim:.4f} - {text}")
        
        logger.info("\n✅ Embedding service example completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error during embedding computation: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())