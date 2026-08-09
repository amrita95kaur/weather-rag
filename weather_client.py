"""NWS (National Weather Service) helper client for resolving locations,
fetching alerts and forecasts, and normalizing documents for Lakebase.

Usage: call `sync_locations(locations, limit)` to fetch and upsert documents
into the `weather_documents` table using `lakebase.get_connection()`.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Dict, List, Optional, Tuple

import requests

import lakebase


NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
NWS_BASE = "https://api.weather.gov"


def geocode_location(location: str) -> Optional[Tuple[float, float, str]]:
    """Resolve a free-text `city, state` string to (lat, lon, display_name).
    Falls back to treating `location` as `lat,lon` if parseable.
    """
    # If user passed lat,lon
    if "," in location:
        parts = [p.strip() for p in location.split(",")]
        if len(parts) == 2:
            try:
                lat = float(parts[0])
                lon = float(parts[1])
                return lat, lon, f"{lat},{lon}"
            except Exception:
                pass

    params = {"q": location, "format": "json", "limit": 1}
    headers = {"User-Agent": "weather-rag/1.0 (email@example.com)"}
    resp = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        return None
    item = data[0]
    return float(item["lat"]), float(item["lon"]), item.get("display_name", location)


def resolve_gridpoint(lat: float, lon: float) -> Optional[Dict]:
    """Call NWS GET /points/{lat},{lon} to get gridpoint metadata.
    Returns the JSON object or None on error.
    """
    url = f"{NWS_BASE}/points/{lat},{lon}"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_alerts_for_point(lat: float, lon: float, limit: int = 50) -> List[Dict]:
    url = f"{NWS_BASE}/alerts/active"
    params = {"point": f"{lat},{lon}", "limit": limit}
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json().get("features", [])


def fetch_forecast_for_grid(grid_json: Dict) -> List[Dict]:
    """Fetch forecast periods and normalize them into list of features.
    `grid_json` is the result of GET /points/{lat},{lon}.
    """
    try:
        props = grid_json.get("properties", {})
        forecast_url = props.get("forecast")
        if not forecast_url:
            return []
        resp = requests.get(forecast_url, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        periods = data.get("properties", {}).get("periods", [])
        return periods
    except Exception:
        return []


def _make_id(prefix: str, raw: Dict) -> str:
    """Stable id for deduplication: hash of canonical fields."""
    if prefix == "alert":
        # NWS alert features typically have an 'id' field
        return raw.get("id") or hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()
    else:
        # For forecast periods, use start time + gridpoint + short hash
        key = json.dumps({"name": raw.get("name"), "startTime": raw.get("startTime"), "shortForecast": raw.get("shortForecast")}, sort_keys=True)
        return hashlib.sha256(key.encode()).hexdigest()


def normalize_alert(feature: Dict, location_label: str) -> Dict:
    props = feature.get("properties", {})
    doc_id = _make_id("alert", feature)
    headline = props.get("headline") or props.get("event")
    narrative = (props.get("description") or "") + "\n\n" + (props.get("instruction") or "")
    issued = props.get("sent") or props.get("effective") or props.get("onset")
    return {
        "id": doc_id,
        "location": location_label,
        "source_type": "alert",
        "headline": headline,
        "narrative_text": narrative.strip(),
        "issued_at": issued,
        "payload": feature,
        "synced_at": None,
    }


def normalize_forecast(period: Dict, location_label: str) -> Dict:
    doc_id = _make_id("forecast", period)
    headline = period.get("name")
    narrative = period.get("detailedForecast") or period.get("detailedforecast") or period.get("shortForecast")
    issued = period.get("startTime")
    return {
        "id": doc_id,
        "location": location_label,
        "source_type": "forecast",
        "headline": headline,
        "narrative_text": narrative,
        "issued_at": issued,
        "payload": period,
        "synced_at": None,
    }


def upsert_documents(docs: List[Dict]) -> int:
    """Upsert into weather_documents. Returns number of upserted rows."""
    if not docs:
        return 0
    sql = (
        "INSERT INTO weather_documents (id, location, source_type, headline, narrative_text, issued_at, payload, synced_at) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,now()) "
        "ON CONFLICT (id) DO UPDATE SET narrative_text = EXCLUDED.narrative_text, payload = EXCLUDED.payload, synced_at = now()"
    )
    params = []
    for d in docs:
        params.append((
            d.get("id"),
            d.get("location"),
            d.get("source_type"),
            d.get("headline"),
            d.get("narrative_text"),
            d.get("issued_at"),
            json.dumps(d.get("payload") or {}),
        ))

    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, params)
            conn.commit()
            return cur.rowcount


def sync_locations(locations: List[str], limit: int = 50) -> int:
    """Resolve each location, fetch alerts and forecasts, normalize and upsert.
    Returns total documents synced.
    """
    total = 0
    for loc in locations:
        resolved = None
        try:
            resolved = geocode_location(loc)
        except Exception:
            resolved = None
        if not resolved:
            continue
        lat, lon, label = resolved
        try:
            grid = resolve_gridpoint(lat, lon)
        except Exception:
            grid = None

        # Fetch alerts
        try:
            alerts = fetch_alerts_for_point(lat, lon, limit=limit)
        except Exception:
            alerts = []
        norm_alerts = [normalize_alert(a, label) for a in alerts]

        # Fetch forecast periods
        periods = []
        try:
            periods = fetch_forecast_for_grid(grid) if grid else []
        except Exception:
            periods = []
        norm_periods = [normalize_forecast(p, label) for p in periods]

        count = upsert_documents(norm_alerts + norm_periods)
        total += count
        # be nice to the NWS API
        time.sleep(1)

    return total
