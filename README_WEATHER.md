# Weather RAG System

A complete weather retrieval-augmented generation (RAG) system for ingesting, embedding, and searching National Weather Service (NWS) alerts and forecasts using Lakebase (Postgres + pgvector).

## Architecture

This system implements a three-part pipeline:

1. **Harvest (Ingestion)**: Fetch weather data from NWS API and store in Lakebase
2. **Vectorize (Embedding)**: Generate semantic embeddings for weather documents
3. **Retrieve (Search)**: Semantic search API for finding relevant weather information

## Components

### Core Modules

* **`lakebase.py`**: Postgres connection helper and schema management
  * `get_connection()` - Context manager for psycopg connections
  * `ensure_weather_tables()` - Create database tables (idempotent)
  * `run_query()` - Execute SELECT queries
  * `run_write()` - Execute INSERT/UPDATE/DELETE

* **`weather_client.py`**: NWS API client
  * `geocode_location()` - Resolve city/state to lat/lon
  * `resolve_gridpoint()` - Get NWS grid information
  * `fetch_alerts_for_point()` - Fetch active alerts
  * `fetch_forecast_for_grid()` - Fetch forecast periods
  * `sync_locations()` - Main sync function

* **`app.py`**: Flask REST API
  * `POST /weather/sync` - Trigger data ingestion
  * `POST /weather/search` - Semantic search over weather documents

### Notebooks

* **`notebooks/ingest_weather_embeddings`**: Embedding generation pipeline
  * Reads documents from `weather_documents` table
  * Chunks narrative text (800 chars, 100 char overlap)
  * Generates 384-dim embeddings using sentence-transformers/all-MiniLM-L6-v2
  * Writes to `weather_embeddings` table
  * **Note**: Requires Serverless GPU or ML Runtime for PyTorch support

## Database Schema

### `weather_documents`

```sql
CREATE TABLE weather_documents (
    id TEXT PRIMARY KEY,               -- Stable dedup key
    location TEXT NOT NULL,             -- City/state or lat/lon
    source_type TEXT NOT NULL,          -- 'alert' or 'forecast'
    headline TEXT,                      -- Event name or period name
    narrative_text TEXT,                -- Full text for embedding
    issued_at TIMESTAMPTZ,              -- When issued/effective
    payload JSONB NOT NULL,             -- Raw NWS JSON
    synced_at TIMESTAMPTZ NOT NULL      -- Last sync timestamp
);
```

### `weather_embeddings`

```sql
CREATE TABLE weather_embeddings (
    id TEXT PRIMARY KEY,                -- document_id + chunk_index
    document_id TEXT NOT NULL,          -- FK to weather_documents.id
    chunk_index INT NOT NULL,           -- Chunk position
    chunk_text TEXT NOT NULL,           -- Chunk content
    embedding VECTOR(384) NOT NULL,     -- 384-dim embedding vector
    model_name TEXT NOT NULL,           -- Model used
    created_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX idx_weather_embeddings_embedding 
    ON weather_embeddings USING hnsw (embedding vector_cosine_ops);
```

## Setup

### Prerequisites

1. **Lakebase Database**: You need a Lakebase Postgres instance
2. **Secret Configuration**: Store the Lakebase connection URL in Databricks secrets
3. **Dependencies**: Install required packages

### Initialize Database

```python
import lakebase
lakebase.ensure_weather_tables(embedding_dim=384)
```

## Usage

### Part 1: Harvest Weather Data

#### Via Python

```python
import weather_client

# Sync weather for specific locations
locations = ["Chicago, IL", "San Francisco, CA", "Austin, TX"]
count = weather_client.sync_locations(locations, limit=50)
print(f"Synced {count} documents")
```

#### Via REST API

```bash
curl -X POST http://localhost:8000/weather/sync \
  -H "Content-Type: application/json" \
  -d '{
    "locations": ["Chicago, IL", "Austin, TX"],
    "limit": 50
  }'

# Response:
# {"synced": 120, "locations": ["Chicago, IL", "Austin, TX"]}
```

### Part 2: Generate Embeddings

Run the embedding notebook:

1. Open `notebooks/ingest_weather_embeddings`
2. **Important**: Use Serverless GPU compute or ML Runtime (sentence-transformers requires PyTorch)
3. Run all cells

The notebook will:
* Find documents without embeddings
* Chunk the narrative text
* Generate embeddings
* Write to `weather_embeddings` table

### Part 3: Semantic Search

#### Via REST API

```bash
curl -X POST http://localhost:8000/weather/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "risk of flooding near rivers",
    "top_k": 5
  }'

# Response: Array of matching documents with similarity scores
```

#### Via Python

```python
import lakebase
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
query = "severe thunderstorm warning"

# Embed query
query_vec = model.encode([query])[0].tolist()
vec_literal = "[" + ",".join(map(str, query_vec)) + "]"

# Search
sql = """
    SELECT d.location, d.headline, e.chunk_text,
           (e.embedding <=> %s::vector) AS distance
    FROM weather_embeddings e
    JOIN weather_documents d ON d.id = e.document_id
    ORDER BY distance ASC
    LIMIT 5
"""

results = lakebase.run_query(sql, (vec_literal,))
for r in results:
    similarity = 1.0 - float(r['distance'])
    print(f"{r['location']}: {r['headline']} (similarity: {similarity:.3f})")
```

## Testing

Run the comprehensive test suite:

```bash
python test_weather_pipeline.py
```

The test script validates:
1. ✓ Database table creation
2. ✓ Weather data syncing
3. ✓ Document queries
4. ✓ Embedding presence
5. ✓ Semantic search functionality

## API Reference

### POST /weather/sync

Fetch weather documents and store in database.

**Request:**
```json
{
  "locations": ["City, State", "lat,lon"],
  "limit": 50  // Optional, default 50
}
```

**Response:**
```json
{
  "synced": 120,
  "locations": ["Chicago, IL", "Austin, TX"]
}
```

### POST /weather/search

Semantic search over weather embeddings.

**Request:**
```json
{
  "query": "flash flood risk this weekend",
  "top_k": 5  // Optional, default 5, max 20
}
```

**Response:**
```json
[
  {
    "document_id": "abc123",
    "location": "Chicago, IL",
    "source_type": "alert",
    "headline": "Flash Flood Warning",
    "chunk_text": "Heavy rainfall expected...",
    "similarity": 0.87
  }
]
```

## Data Sources

### National Weather Service (NWS) API

* **Alerts**: `https://api.weather.gov/alerts/active?point={lat},{lon}`
* **Forecasts**: `https://api.weather.gov/points/{lat},{lon}`
* **Rate Limits**: Be respectful - use delays between requests
* **User-Agent**: Required (set in weather_client.py)

### Geocoding

* Uses OpenStreetMap Nominatim for city/state → lat/lon conversion
* Fallback: Direct lat,lon parsing

## Chunking Strategy

* **Chunk Size**: 800 characters
* **Overlap**: 100 characters
* **Rationale**: Most NWS alerts and forecasts are short (200-600 chars)

## Embedding Model

* **Model**: `sentence-transformers/all-MiniLM-L6-v2`
* **Dimensions**: 384
* **Why**: Fast, lightweight, good performance on short documents

## Troubleshooting

### "Fatal error: The Python kernel is unresponsive" in notebook

**Cause**: sentence-transformers requires PyTorch, which needs proper ML environment

**Solution**: Use Serverless GPU or classic cluster with ML Runtime

### "No documents found"

**Cause**: Weather sync hasn't been run or returned no results

**Solution**: Run `POST /weather/sync` or `weather_client.sync_locations()`

### "No embeddings found"

**Cause**: Embedding notebook hasn't been run

**Solution**: Run the `ingest_weather_embeddings` notebook

## Future Enhancements

* [ ] Scheduled sync (daily refresh of weather data)
* [ ] Incremental embedding (only embed new documents)
* [ ] Multiple embedding models (switch by use case)
* [ ] Hybrid search (combine vector + keyword)
* [ ] Location-based filtering in search
* [ ] Time-based filtering (only recent alerts)
* [ ] Web UI for search
