# Weather RAG Pipeline Architecture

## Overview

This weather-rag project implements the same architecture pattern as the ticker news embeddings pipeline from `databricks-lakebase-app-day-2`. Both follow a **4-stage RAG pipeline** using Lakebase Postgres + pgvector.

---

## Architecture Comparison

### Pipeline Stages (Side-by-Side)

| Stage | Ticker News Pipeline | Weather RAG Pipeline |
|-------|---------------------|---------------------|
| **1. Harvest** | Massive API → ticker_news_documents | NWS API → weather_documents |
| **2. Vectorize** | sentence-transformers → ticker_news_embeddings | sentence-transformers → weather_embeddings |
| **3. Chunk** | Article body → ticker_news_chunk_embeddings | Alert/forecast text → weather_embeddings (chunks) |
| **4. Retrieve** | POST /news/search | POST /weather/search |

---

## Detailed Component Mapping

### 1️⃣ Data Ingestion (Harvest)

#### Ticker News
```python
# massive_client.py
def fetch_news(ticker, limit=50):
    # GET https://api.polygon.io/v2/reference/news
    # Returns: id, ticker, title, description, article_url, published_utc
    
def upsert_news(articles):
    # INSERT INTO ticker_news_documents ... ON CONFLICT DO NOTHING
```

**Endpoint**: `POST /news/sync` with `{"tickers": ["AAPL", "MSFT"], "limit": 50}`

#### Weather RAG ✓
```python
# weather_client.py
def sync_locations(locations, limit=50):
    # 1. geocode_location() → lat/lon
    # 2. resolve_gridpoint() → NWS grid
    # 3. fetch_alerts_for_point() + fetch_forecast_for_grid()
    # Returns: id, location, source_type, headline, narrative_text, issued_at
    
def upsert_documents(docs):
    # INSERT INTO weather_documents ... ON CONFLICT DO UPDATE
```

**Endpoint**: `POST /weather/sync` with `{"locations": ["Chicago, IL"], "limit": 50}`

---

### 2️⃣ Text Chunking

#### Ticker News
```python
# notebooks/ingest_ticker_news_embeddings (Cell 21-22)
# Fetch full article body from article_url using trafilatura
for start in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
    chunk = text[start:start+CHUNK_SIZE]
    chunks.append(chunk)
```

**Parameters**: 800 char chunks, 100 char overlap

#### Weather RAG ✓
```python
# notebooks/ingest_weather_embeddings (Cell 6)
# Split narrative_text from weather_documents
for start in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
    chunk = text[start:start+CHUNK_SIZE]
    chunks.append(chunk)
```

**Parameters**: 800 char chunks, 100 char overlap *(same)*

---

### 3️⃣ Embedding Computation

#### Ticker News
```python
# notebooks/ingest_ticker_news_embeddings (Cell 24)
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
vectors = model.encode(texts, show_progress_bar=False)
# Result: 384-dimensional vectors
```

#### Weather RAG ✓
```python
# notebooks/ingest_weather_embeddings (Cell 8)
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
vectors = model.encode(chunks, show_progress_bar=False)
# Result: 384-dimensional vectors
```

**Model**: Same (all-MiniLM-L6-v2, 384 dims)  
**Batch Size**: Same (32 per batch)

---

### 4️⃣ Vector Storage

#### Ticker News Schema
```sql
CREATE TABLE ticker_news_chunk_embeddings (
    id TEXT PRIMARY KEY,
    article_id TEXT NOT NULL,
    chunk_index INT NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL,
    model_name TEXT NOT NULL,
    embedded_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_ticker_news_chunk_embeddings_embedding
ON ticker_news_chunk_embeddings
USING hnsw (embedding vector_cosine_ops);
```

#### Weather RAG Schema ✓
```sql
CREATE TABLE weather_embeddings (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    chunk_index INT NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL,
    model_name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_weather_embeddings_embedding
ON weather_embeddings
USING hnsw (embedding vector_cosine_ops);
```

**Difference**: Column names (`embedded_at` vs `created_at`), otherwise identical

---

### 5️⃣ Batch Insertion Pattern

#### Ticker News
```python
# notebooks/ingest_ticker_news_embeddings (Cell 28)
import psycopg2

conn = psycopg2.connect(host=..., port=..., dbname=..., user=..., password=...)
cursor = conn.cursor()

insert_sql = """
    INSERT INTO ticker_news_chunk_embeddings 
    (id, article_id, chunk_index, chunk_text, embedding, model_name, embedded_at)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (id) DO NOTHING
"""

rows = [
    (row['id'], row['article_id'], row['chunk_index'], row['chunk_text'],
     '{' + ','.join(map(str, row['embedding'])) + '}',  # PostgreSQL array
     row['model_name'], row['embedded_at'])
    for row in embedding_rows
]

cursor.executemany(insert_sql, rows)
conn.commit()
```

**Critical Step**: `UPDATE table SET embedding = embedding::vector` (array → vector cast)

#### Weather RAG ✓
```python
# notebooks/ingest_weather_embeddings (Cell 10)
import psycopg2

conn = psycopg2.connect(host=..., port=..., dbname=..., user=..., password=...)
cursor = conn.cursor()

insert_sql = """
    INSERT INTO weather_embeddings 
    (id, document_id, chunk_index, chunk_text, embedding, model_name, created_at)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (id) DO NOTHING
"""

rows = [
    (row['id'], row['document_id'], row['chunk_index'], row['chunk_text'],
     '{' + ','.join(map(str, row['embedding'])) + '}',  # PostgreSQL array
     row['model_name'], row['created_at'])
    for row in embedding_rows
]

cursor.executemany(insert_sql, rows)
conn.commit()
```

**Same pattern**: PostgreSQL array format + manual vector cast

---

### 6️⃣ Semantic Search

#### Ticker News
```python
# app.py - /news/search endpoint
vec = model.encode([query], show_progress_bar=False)[0].tolist()
qvec_literal = '[' + ','.join(map(lambda x: repr(float(x)), vec)) + ']'

sql = """
    SELECT d.id, d.ticker, d.title, e.chunk_text,
           (e.embedding <=> %s::vector) AS distance
    FROM ticker_news_chunk_embeddings e
    JOIN ticker_news_documents d ON d.id = e.article_id
    ORDER BY distance ASC
    LIMIT %s
"""
results = run_query(sql, (qvec_literal, top_k))
```

**Returns**: `{"article_id", "ticker", "title", "chunk_text", "similarity": 1-distance}`

#### Weather RAG ✓
```python
# app.py - /weather/search endpoint
vec = WEATHER_EMBEDDING_MODEL_INSTANCE.encode([query], show_progress_bar=False)[0].tolist()
qvec_literal = '[' + ','.join(map(lambda x: repr(float(x)), vec)) + ']'

sql = """
    SELECT d.id AS document_id, d.location, d.source_type, d.headline,
           e.chunk_text, (e.embedding <=> %s::vector) AS distance
    FROM weather_embeddings e
    JOIN weather_documents d ON d.id = e.document_id
    ORDER BY distance ASC
    LIMIT %s
"""
results = run_query(sql, (qvec_literal, top_k))
```

**Returns**: `{"document_id", "location", "source_type", "headline", "chunk_text", "similarity": 1-distance}`

---

## Key Technical Patterns (Shared)

### ✅ Connection Management
```python
# Both use psycopg (not psycopg2 in lakebase.py, but psycopg2 in notebooks for compatibility)
import base64
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
secret = w.secrets.get_secret(scope="database", key="lakebase-url")
lakebase_url = base64.b64decode(secret.value).decode("utf-8")
# Parse postgresql://user:password@host:port/dbname?sslmode=require
```

### ✅ Rate Limiting (API Courtesy)
```python
# Ticker news: Massive API rate limit
time.sleep(60 / MAX_REQUESTS_PER_MINUTE)

# Weather: NWS API courtesy delay
time.sleep(1)  # 1 second between locations
```

### ✅ Error Handling (Graceful Degradation)
```python
# Both pipelines: try/except around external fetches
try:
    content = fetch_from_api(url)
except Exception as e:
    # Skip this item, continue processing
    continue
```

### ✅ Batch Processing (Memory Efficiency)
```python
# Both: Process embeddings in batches of 32
for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i+batch_size]
    vectors = model.encode(batch['text'].tolist())
```

### ✅ Deduplication Strategy
```python
# Both: ON CONFLICT DO NOTHING for idempotent re-runs
INSERT INTO embeddings_table (...) VALUES (...)
ON CONFLICT (id) DO NOTHING
```

---

## Compute Requirements

| Notebook | Required Compute | Reason |
|----------|------------------|--------|
| Ticker news embeddings | Serverless GPU or ML Runtime | sentence-transformers needs PyTorch |
| Weather embeddings | Serverless GPU or ML Runtime | sentence-transformers needs PyTorch |

**Why not CPU?** sentence-transformers → transformers → PyTorch → CUDA libraries (unavailable on Serverless CPU Standard)

---

## File Structure Comparison

### Ticker News Project
```
databricks-lakebase-app-day-2/
├── app.py                              # Flask API
├── massive_client.py                   # Massive API client
├── lakebase.py                         # DB helper
├── notebooks/
│   └── ingest_ticker_news_embeddings   # Embedding pipeline
└── sql/
    └── 03_setup_chunk_embeddings_table.sql
```

### Weather RAG Project ✓
```
weather-rag/
├── app.py                              # Flask API (mirrors structure)
├── weather_client.py                   # NWS API client (mirrors massive_client.py)
├── lakebase.py                         # DB helper (same pattern)
├── notebooks/
│   └── ingest_weather_embeddings       # Embedding pipeline (mirrors ticker news)
└── sql/
    └── 06_setup_weather_embeddings_table.sql
```

---

## Learning Objectives ✅

| Objective | Implementation |
|-----------|----------------|
| Harvest unstructured data from API | weather_client.py fetches NWS alerts/forecasts |
| Design Postgres/pgvector schema | weather_documents + weather_embeddings tables |
| Chunk and embed text | notebooks/ingest_weather_embeddings |
| Write vectors via psycopg2 | Batch insert with executemany() |
| Implement cosine similarity endpoint | POST /weather/search with pgvector <=> |

---

## Summary

The weather-rag pipeline **exactly mirrors** the ticker news architecture:

1. **Same embedding model** (all-MiniLM-L6-v2, 384 dims)
2. **Same chunking strategy** (800 chars, 100 overlap)
3. **Same database pattern** (psycopg2, array→vector cast, HNSW index)
4. **Same Flask endpoints** (/sync, /search)
5. **Same notebook structure** (read → chunk → embed → upsert)
6. **Same compute requirements** (needs GPU for PyTorch)

**Key Differences**:
* Data source: Massive API (finance) → NWS API (weather)
* Domain: Ticker news → Weather alerts/forecasts
* Column names: Minor (article_id → document_id, embedded_at → created_at)

Everything else is **architecturally identical**! 🎯