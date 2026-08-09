# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "databricks_ai_v5"
# environment_version = "5"
# ///
# DBTITLE 1,Install Dependencies
# MAGIC %md
# MAGIC # Weather Document Embeddings Pipeline
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Reads weather documents from `weather_documents` table
# MAGIC 2. Computes embeddings for narrative text using sentence-transformers
# MAGIC 3. Chunks long narratives (alerts/forecasts can be lengthy)
# MAGIC 4. Writes embeddings to `weather_embeddings` table
# MAGIC
# MAGIC **Compute requirement**: This notebook requires **Serverless GPU** (not CPU) because sentence-transformers needs PyTorch.
# MAGIC
# MAGIC **Architecture mirrors**: `ingest_ticker_news_embeddings` notebook pattern from databricks-lakebase-app-day-2

# COMMAND ----------

# DBTITLE 1,Install required packages
# MAGIC %pip install psycopg sentence-transformers --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Configuration
# Embedding model (matches the app.py default)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension

# Chunking parameters for long weather narratives
CHUNK_SIZE = 800  # characters per chunk
CHUNK_OVERLAP = 100  # overlap to preserve context

# Table names
DOCUMENTS_TABLE = "weather_documents"
EMBEDDINGS_TABLE = "weather_embeddings"

print(f"Model: {EMBEDDING_MODEL_NAME}")
print(f"Embedding dimension: {EMBEDDING_DIM}")
print(f"Chunk size: {CHUNK_SIZE} chars, overlap: {CHUNK_OVERLAP} chars")

# COMMAND ----------

# DBTITLE 1,Parse Lakebase credentials from secrets
import base64
import os
from urllib.parse import urlparse
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# Fetch Lakebase URL from secrets (same pattern as lakebase.py)
SCOPE = os.environ.get("LAKEBASE_SECRET_SCOPE", "database")
KEY = os.environ.get("LAKEBASE_SECRET_KEY", "lakebase-url")

secret = w.secrets.get_secret(scope=SCOPE, key=KEY)
lakebase_url = base64.b64decode(secret.value).decode("utf-8")

# Parse URL: postgresql://user:password@host:port/dbname?sslmode=require
parsed = urlparse(lakebase_url)
db_host = parsed.hostname
db_port = parsed.port or 5432
db_name = parsed.path.lstrip('/')
db_user = parsed.username
db_password = parsed.password

print(f"✅ Connected to Lakebase: {db_host}:{db_port}/{db_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load weather documents from Lakebase

# COMMAND ----------

# DBTITLE 1,Read weather_documents table
import pandas as pd
import psycopg

# Connect with explicit password authentication (no OAuth)
conn = psycopg.connect(
    host=db_host,
    port=db_port,
    dbname=db_name,
    user=db_user,
    password=db_password,
    sslmode='require'
)

query = f"""
    SELECT id, location, source_type, headline, narrative_text, issued_at
    FROM {DOCUMENTS_TABLE}
    WHERE narrative_text IS NOT NULL AND narrative_text != ''
    ORDER BY issued_at DESC
"""

weather_df = pd.read_sql(query, conn)
conn.close()

print(f"Loaded {len(weather_df)} weather documents with narrative text")
print(f"\nSource type breakdown:")
print(weather_df['source_type'].value_counts())
weather_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chunk long narratives
# MAGIC
# MAGIC Weather alerts and forecasts can have lengthy text. We'll split them into overlapping chunks for fine-grained retrieval (matches the ticker news chunk pattern).

# COMMAND ----------

# DBTITLE 1,Create overlapping chunks
print(f"Chunking {len(weather_df)} documents...")

out_doc_ids, out_chunk_indexes, out_chunk_texts = [], [], []

for idx, row in weather_df.iterrows():
    doc_id = row['id']
    text = row['narrative_text'] or ""
    
    if not text.strip():
        continue
    
    # Split into overlapping chunks
    for chunk_index, start in enumerate(range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP)):
        chunk_text = text[start : start + CHUNK_SIZE].strip()
        if not chunk_text:
            continue
        out_doc_ids.append(doc_id)
        out_chunk_indexes.append(chunk_index)
        out_chunk_texts.append(chunk_text)
        if start + CHUNK_SIZE >= len(text):
            break  # Last chunk

chunks_df = pd.DataFrame({
    "document_id": out_doc_ids,
    "chunk_index": out_chunk_indexes,
    "chunk_text": out_chunk_texts
})

print(f"✅ Created {len(chunks_df)} chunks from {len(weather_df)} documents")
print(f"   Average {len(chunks_df)/len(weather_df):.1f} chunks per document")
chunks_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute embeddings
# MAGIC
# MAGIC Load sentence-transformers model and compute vectors for each chunk.

# COMMAND ----------

# DBTITLE 1,Load model and compute embeddings
import os
from sentence_transformers import SentenceTransformer

# Set cache directory
os.environ["HF_HOME"] = "/tmp/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/.cache/huggingface"
os.environ["HF_HUB_CACHE"] = "/tmp/.cache/huggingface"

print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
model = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder="/tmp/.cache/huggingface")
print("✅ Model loaded")

# Compute embeddings in batches
batch_size = 32
all_embeddings = []

print(f"\nComputing embeddings for {len(chunks_df)} chunks...")
for i in range(0, len(chunks_df), batch_size):
    batch = chunks_df.iloc[i:i+batch_size]
    vectors = model.encode(batch["chunk_text"].tolist(), show_progress_bar=False)
    all_embeddings.extend(vectors.tolist())
    
    if (i + batch_size) % 128 == 0 or i + batch_size >= len(chunks_df):
        print(f"  Processed {min(i + batch_size, len(chunks_df))}/{len(chunks_df)} chunks")

chunks_df['embedding'] = all_embeddings
print(f"\n✅ Computed {len(all_embeddings)} embeddings (dimension: {len(all_embeddings[0])})")

# Show sample
chunks_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Upsert embeddings into Lakebase
# MAGIC
# MAGIC Write embeddings to `weather_embeddings` table using psycopg2 batch insert.

# COMMAND ----------

# DBTITLE 1,Insert embeddings
from datetime import datetime

# Prepare rows for insertion
chunks_df['id'] = chunks_df['document_id'] + '_' + chunks_df['chunk_index'].astype(str)
chunks_df['model_name'] = EMBEDDING_MODEL_NAME
chunks_df['created_at'] = datetime.now()

embedding_rows = chunks_df.to_dict('records')

if len(embedding_rows) > 0:
    print(f"Inserting {len(embedding_rows)} embeddings into {EMBEDDINGS_TABLE}...")
    
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password
    )
    cursor = conn.cursor()
    
    insert_sql = f"""
        INSERT INTO {EMBEDDINGS_TABLE} 
        (id, document_id, chunk_index, chunk_text, embedding, model_name, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO NOTHING
    """
    
    rows_data = [
        (
            row['id'],
            row['document_id'],
            int(row['chunk_index']),
            row['chunk_text'],
            '{' + ','.join(map(str, row['embedding'])) + '}',  # PostgreSQL array format
            row['model_name'],
            row['created_at']
        )
        for row in embedding_rows
    ]
    
    cursor.executemany(insert_sql, rows_data)
    inserted_count = cursor.rowcount
    conn.commit()
    cursor.close()
    conn.close()
    
    print(f"✅ Successfully inserted {inserted_count} embeddings")
    print("   (Duplicates were skipped via ON CONFLICT DO NOTHING)")
    print(f"\n⚠️  IMPORTANT: Run this SQL in Lakebase to cast arrays to vectors:")
    print(f"   UPDATE {EMBEDDINGS_TABLE} SET embedding = embedding::vector WHERE embedding IS NOT NULL;")
else:
    print("No embeddings to insert.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify embeddings were written
# MAGIC
# MAGIC Query the embeddings table to confirm data is present.

# COMMAND ----------

# DBTITLE 1,Check embeddings table
conn = psycopg2.connect(
    host=db_host,
    port=db_port,
    dbname=db_name,
    user=db_user,
    password=db_password,
    cursor_factory=RealDictCursor
)

verify_query = f"""
    SELECT 
        COUNT(*) as total_embeddings,
        COUNT(DISTINCT document_id) as unique_documents,
        model_name
    FROM {EMBEDDINGS_TABLE}
    GROUP BY model_name
"""

verify_df = pd.read_sql(verify_query, conn)
conn.close()

print(f"📊 Embeddings table summary:")
print(verify_df)

print(f"\n✅ Pipeline complete! You can now use POST /weather/search in the Flask app.")