-- Setup script for weather_embeddings table
-- Run this manually in your Lakebase Postgres database before running the ingest script
-- Replace {{EMBEDDING_DIM}} with your model's dimension (default below uses 384)

CREATE EXTENSION IF NOT EXISTS vector;

-- IMPORTANT: replace {{EMBEDDING_DIM}} with your embedding model's output dim
CREATE TABLE IF NOT EXISTS weather_embeddings (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    text TEXT NOT NULL,
    embedding VECTOR({{EMBEDDING_DIM}}) NOT NULL,
    model_name TEXT NOT NULL,
    embedded_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_weather_embeddings_embedding
ON weather_embeddings
USING hnsw (embedding vector_cosine_ops);

-- Verify
SELECT table_name, column_name, data_type, udt_name
FROM information_schema.columns
WHERE table_name = 'weather_embeddings'
ORDER BY ordinal_position;
