#!/usr/bin/env python3
"""Simple batch job: fetch weather text from NWS, chunk, embed, and write to Lakebase.

Usage: python scripts/ingest_weather_embeddings.py --states CA,WA --model sentence-transformers/all-MiniLM-L6-v2

Before running: execute `sql/05_setup_weather_table.sql` and
`sql/06_setup_weather_embeddings_table.sql` (replace {{EMBEDDING_DIM}} with your model dim).
"""

import argparse
import json
import time
from datetime import datetime
from typing import List

import requests
from sentence_transformers import SentenceTransformer

import lakebase


EMBEDDING_MODEL_DEFAULT = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM_DEFAULT = 384
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def fetch_alerts_for_state(state: str) -> List[dict]:
    url = f"https://api.weather.gov/alerts/active?area={state}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json().get("features", [])


def normalize_alert(feature: dict) -> dict:
    props = feature.get("properties", {})
    doc_id = feature.get("id") or props.get("id") or props.get("@id")
    title = props.get("event") or props.get("headline")
    description = props.get("description")
    instruction = props.get("instruction")
    published = props.get("sent" ) or props.get("effective")
    payload = feature
    return {
        "id": doc_id,
        "source": "nws.alerts",
        "title": title,
        "description": description,
        "instruction": instruction,
        "published": published,
        "payload": payload,
    }


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if not text:
        return []
    chunks = []
    step = size - overlap
    for start in range(0, len(text), step):
        chunk = text[start : start + size].strip()
        if chunk:
            chunks.append(chunk)
        if start + size >= len(text):
            break
    return chunks


def upsert_documents(docs: List[dict]):
    sql = (
        "INSERT INTO weather_documents (id, source, title, description, instruction, published, payload, synced_at) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,now()) "
        "ON CONFLICT (id) DO UPDATE SET description = EXCLUDED.description, instruction = EXCLUDED.instruction, payload = EXCLUDED.payload, synced_at = now()"
    )
    params = []
    for d in docs:
        params.append(
            (
                d.get("id"),
                d.get("source"),
                d.get("title"),
                d.get("description"),
                d.get("instruction"),
                d.get("published"),
                json.dumps(d.get("payload") or {}),
            )
        )

    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, params)
            conn.commit()


def upsert_embeddings(rows: List[tuple], model_name: str):
    """Rows: list of (id, document_id, text, embedding_list)

    Uses psycopg2.extras.execute_values for batched insertion.
    """
    from psycopg2.extras import execute_values

    sql = (
        "INSERT INTO weather_embeddings (id, document_id, chunk_index, chunk_text, embedding, model_name, created_at) "
        "VALUES %s ON CONFLICT (id) DO NOTHING"
    )

    # Prepare value tuples where embedding is passed as a literal string like [0.1,0.2,...]
    values = []
    for _id, doc_id, chunk_index, text, vec in rows:
        vec_literal = "[" + ",".join(map(lambda x: repr(float(x)), vec)) + "]"
        values.append((_id, doc_id, int(chunk_index), text, vec_literal, model_name))

    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, values, template="(%s,%s,%s,%s,%s::vector,%s,now())")
            conn.commit()


def main(states: List[str], model_name: str):
    model = SentenceTransformer(model_name)

    all_docs = []
    for state in states:
        try:
            features = fetch_alerts_for_state(state)
        except Exception as exc:
            print(f"Failed to fetch alerts for {state}: {exc}")
            continue
        for feat in features:
            doc = normalize_alert(feat)
            if not doc.get("id"):
                # skip malformed
                continue
            all_docs.append(doc)

    if not all_docs:
        print("No documents found for the requested states.")
        return

    print(f"Upserting {len(all_docs)} documents...")
    upsert_documents(all_docs)

    # Build chunks and embed
    embed_rows = []
    for doc in all_docs:
        text_to_embed = "\n\n".join([str(doc.get("title") or ""), str(doc.get("description") or ""), str(doc.get("instruction") or "")])
        chunks = chunk_text(text_to_embed)
        if not chunks:
            # fallback to title if nothing else
            chunks = [doc.get("title") or ""]

        vectors = model.encode(chunks, show_progress_bar=False)
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            row_id = f"{doc['id']}_{i}"
            embed_rows.append((row_id, doc["id"], i, chunk, vec.tolist()))

    print(f"Writing {len(embed_rows)} embeddings to Lakebase...")
    upsert_embeddings(embed_rows, model_name)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", default="CA", help="Comma-separated state codes to fetch alerts for (e.g. CA,WA,OR)")
    parser.add_argument("--model", default=EMBEDDING_MODEL_DEFAULT, help="SentenceTransformer model name")
    args = parser.parse_args()
    states = [s.strip().upper() for s in args.states.split(",") if s.strip()]
    main(states, args.model)
