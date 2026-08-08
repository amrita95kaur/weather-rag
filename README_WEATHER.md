# Weather RAG: Harvest → Embed → Retrieve

This document describes the weather RAG pipeline added to the project.

Data source
- National Weather Service (NWS) API: `api.weather.gov` — no API key required, provides active alerts and gridpoint forecasts with narrative text.

Schema decisions
- `weather_documents` (raw documents): `id, source, title, description, instruction, published, payload, synced_at` — stores alerts and forecast periods.
- `weather_embeddings` (vectors): `id, document_id, chunk_index, chunk_text, embedding VECTOR(384), model_name, created_at` — uses `pgvector` (384 dims for `all-MiniLM-L6-v2`).

Chunking / model
- Chunking: `CHUNK_SIZE=800`, `CHUNK_OVERLAP=100` (sliding window). Most NWS text is short; chunking applies primarily when combining headline+description+instruction.
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (384-dim) — matches the existing news pipeline.

How to run end-to-end
1. Install deps:
```bash
pip install -r requirements.txt
```
2. Create DB tables (run in Lakebase/Postgres):
```sql
-- Run these SQL files and replace {{EMBEDDING_DIM}} with 384 if needed
\i sql/05_setup_weather_table.sql
\i sql/06_setup_weather_embeddings_table.sql
```
Alternatively, from Python you can call `lakebase.ensure_weather_tables(embedding_dim=384)`.

3. Sync documents (via Flask endpoint):
```bash
curl -X POST http://localhost:5000/weather/sync -H "Content-Type: application/json" \
  -d '{"locations":["Chicago, IL","Austin, TX"],"limit":50}'
```

4. Run the embedding ingestion (psycopg2-backed):
```bash
python scripts/ingest_weather_embeddings.py --states CA,TX --model sentence-transformers/all-MiniLM-L6-v2
```

5. Query retrieval:
```bash
curl -X POST http://localhost:5000/weather/search -H "Content-Type: application/json" \
  -d '{"query":"flash flood risk this weekend","top_k":5}'
```

Limitations & improvements
- Geocoding uses Nominatim (OpenStreetMap). For production use, consider a rate-limited geocoding service.
- Forecast discussion extraction is from the `/forecast` gridpoint periods — there are other NWS discussion feeds that can be added.
- Current ingestion is single-threaded with per-location sleeps to be polite to NWS. For higher throughput, add batching and retries with backoff.
- Consider adding periodic scheduling (cron/Databricks Job) and deduplication checks beyond simple `id` upserts.
