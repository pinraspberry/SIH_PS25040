-- Migration: Add Vector Capabilities to Argo Database
-- Description: Extends existing argo_profiles table with pgvector support for RAG pipeline
-- Requirements: 1.4, 4.2, 6.4

-- Connect to the database
\c argo_sih;

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify pgvector extension is installed
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';

-- Add vector columns to existing argo_profiles table
ALTER TABLE argo_profiles 
ADD COLUMN IF NOT EXISTS profile_summary TEXT,
ADD COLUMN IF NOT EXISTS embedding vector(384),
ADD COLUMN IF NOT EXISTS embedding_model VARCHAR(100),
ADD COLUMN IF NOT EXISTS embedding_created_at TIMESTAMP DEFAULT NOW();

-- Create vector similarity index using IVFFlat for efficient nearest neighbor search
-- Note: IVFFlat index requires some data to be present, so we'll create it conditionally
DO $$
BEGIN
    -- Check if we have any embeddings before creating the index
    IF (SELECT COUNT(*) FROM argo_profiles WHERE embedding IS NOT NULL) > 0 THEN
        -- Create IVFFlat index for cosine similarity search
        CREATE INDEX IF NOT EXISTS argo_profiles_embedding_idx 
        ON argo_profiles USING ivfflat (embedding vector_cosine_ops) 
        WITH (lists = 100);
        
        RAISE NOTICE 'IVFFlat index created successfully';
    ELSE
        RAISE NOTICE 'No embeddings found. Index will be created after data ingestion.';
        RAISE NOTICE 'Run the following command after ingesting data with embeddings:';
        RAISE NOTICE 'CREATE INDEX argo_profiles_embedding_idx ON argo_profiles USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);';
    END IF;
END $$;

-- Create additional indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_embedding_model ON argo_profiles(embedding_model);
CREATE INDEX IF NOT EXISTS idx_embedding_created_at ON argo_profiles(embedding_created_at);
CREATE INDEX IF NOT EXISTS idx_profile_summary ON argo_profiles USING gin(to_tsvector('english', profile_summary));

-- Add comments to document the new columns
COMMENT ON COLUMN argo_profiles.profile_summary IS 'Text summary of the oceanographic profile for embedding computation';
COMMENT ON COLUMN argo_profiles.embedding IS 'Vector embedding of the profile summary (384 dimensions for all-MiniLM-L6-v2)';
COMMENT ON COLUMN argo_profiles.embedding_model IS 'Name of the model used to generate the embedding (e.g., sentence-transformers, gemini)';
COMMENT ON COLUMN argo_profiles.embedding_created_at IS 'Timestamp when the embedding was computed';

-- Display updated table structure
\d argo_profiles;

-- Show vector extension info
SELECT 
    extname as "Extension Name",
    extversion as "Version",
    extrelocatable as "Relocatable"
FROM pg_extension 
WHERE extname = 'vector';

-- Show new columns
SELECT 
    column_name,
    data_type,
    is_nullable,
    column_default
FROM information_schema.columns 
WHERE table_name = 'argo_profiles' 
AND column_name IN ('profile_summary', 'embedding', 'embedding_model', 'embedding_created_at')
ORDER BY ordinal_position;

-- Show indexes on the table
SELECT 
    indexname,
    indexdef
FROM pg_indexes 
WHERE tablename = 'argo_profiles'
ORDER BY indexname;

SELECT 'Vector capabilities migration completed successfully!' as status;