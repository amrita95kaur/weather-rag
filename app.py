"""
Databricks App boilerplate:
- Serves a small Flask API
- Reads/writes to Lakebase (Databricks-managed Postgres) via lakebase.py
- Pulls data from the Massive API via massive_client.py and syncs it into Lakebase

Run locally:
    python app.py
Deploy as a Databricks App using app.yaml.
"""

import logging
import os
import re

import requests
from databricks.sdk import WorkspaceClient
from flask import Flask, jsonify, render_template, request

import lakebase
from sentence_transformers import SentenceTransformer
import weather_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

app = Flask(__name__)
_w = WorkspaceClient()






def _current_user_email() -> str:
    """
    Resolve the current user's email so the watchlist can be personalized.

    Databricks Apps inject the logged-in user's identity via the
    X-Forwarded-Email header on every request. Fall back to the Databricks
    SDK's current_user API for local development where that header isn't set.
    """
    header_email = request.headers.get("X-Forwarded-Email")
    if header_email:
        return header_email
    return _w.current_user.me().user_name


# Load the embedding model once at module import for weather search
WEATHER_EMBEDDING_MODEL = os.environ.get("WEATHER_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
WEATHER_EMBEDDING_MODEL_INSTANCE = SentenceTransformer(WEATHER_EMBEDDING_MODEL)


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.errorhandler(Exception)
def handle_exception(err):
    """Ensure all unhandled errors return JSON (not an HTML error page),
    so the frontend's resp.json() call never chokes on HTML."""
    logger.exception("Unhandled exception while processing request")
    status_code = getattr(err, "code", 500)
    if not isinstance(status_code, int):
        status_code = 500
    return jsonify({"error": str(err)}), status_code


@app.route("/")
def index():
    """Simple UI to submit a list of stock symbols to sync from Massive."""
    return render_template("index.html")


@app.route("/records")
def list_records():
    return jsonify({"error": "Records endpoint removed (Massive/news features cleaned)."}), 404


@app.route("/sync", methods=["POST"])
def sync_from_massive():
    return jsonify({"error": "Massive sync removed."}), 404


@app.route("/news/sync", methods=["POST"])
def sync_news_from_massive():
    return jsonify({"error": "News sync removed."}), 404


@app.route("/weather/search", methods=["POST"])
def weather_search():
    """Semantic search over ingested weather documents.

    Body: {"query": "flash flood risk this weekend", "limit": 5}
    """
    body = request.json if request.is_json else {}
    query_text = body.get("query")
    if not query_text:
        return jsonify({"error": "Missing 'query' in request body"}), 400

    top_k = int(body.get("top_k", body.get("limit", 5)))
    # clamp
    top_k = max(1, min(20, top_k))

    vec = WEATHER_EMBEDDING_MODEL_INSTANCE.encode([query_text], show_progress_bar=False)[0].tolist()
    qvec_literal = "[" + ",".join(map(lambda x: repr(float(x)), vec)) + "]"

    sql = (
        "SELECT d.id AS document_id, d.source AS source, d.title AS headline, d.description AS description, "
        "e.chunk_text, (e.embedding <=> %s::vector) AS distance "
        "FROM weather_embeddings e JOIN weather_documents d ON d.id = e.document_id "
        "ORDER BY distance ASC LIMIT %s"
    )

    rows = lakebase.run_query(sql, (qvec_literal, top_k))
    # Convert distance -> similarity (1 - distance) for cosine distance
    results = []
    for r in rows:
        dist = r.get("distance")
        similarity = None if dist is None else 1.0 - float(dist)
        results.append(
            {
                "document_id": r.get("document_id"),
                "location": r.get("source"),
                "headline": r.get("headline"),
                "chunk_text": r.get("chunk_text"),
                "similarity": similarity,
            }
        )
    return jsonify(results)



@app.route("/weather/sync", methods=["POST"])
def weather_sync():
    """Trigger syncing of weather docs for a list of locations.

    Body: {"locations": ["Chicago, IL", "40.7128,-74.0060"], "limit": 50}
    """
    body = request.json if request.is_json else {}
    locations = body.get("locations") or []
    if not isinstance(locations, list) or not locations:
        return jsonify({"error": "Missing or invalid 'locations' list in request body"}), 400

    limit = int(body.get("limit", 50))
    # ensure tables exist
    lakebase.ensure_weather_tables(embedding_dim=384)
    synced = weather_client.sync_locations(locations, limit=limit)
    return jsonify({"synced": synced, "locations": locations})


@app.route("/watchlist", methods=["GET", "POST", "DELETE"])
def watchlist_removed():
    return jsonify({"error": "Watchlist/Massive features removed."}), 404





def _upsert_batch(items: list[dict]) -> int:
    """Upsert a batch of Massive API items into Lakebase, one statement per row.

    For very large batches, consider psycopg2.extras.execute_values for
    higher throughput instead of per-row execute calls.
    """
    import json as _json

    count = 0
    # Massive batch upsert removed. This helper is deprecated.
    return 0


def _upsert_news_batch(ticker: str, articles: list[dict]) -> int:
    # Deprecated: news upsert removed from this project
    return 0


if __name__ == '__main__':
    host = os.getenv('FLASK_RUN_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_RUN_PORT', 8000))
    app.run(debug=True, host=host, port=port)
    print(f"Flask app running on http://{host}:{port}")