-- Setup script for weather_documents table
-- Run this manually in your Lakebase Postgres database before running the ingest script

CREATE TABLE IF NOT EXISTS weather_documents (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    title TEXT,
    description TEXT,
    instruction TEXT,
    published TIMESTAMPTZ,
    payload JSONB NOT NULL,
    synced_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_weather_documents_source ON weather_documents (source);

-- Verify
SELECT table_name, column_name, data_type, udt_name
FROM information_schema.columns
WHERE table_name = 'weather_documents'
ORDER BY ordinal_position;
