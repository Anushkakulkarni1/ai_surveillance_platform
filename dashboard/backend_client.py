

from __future__ import annotations

from typing import Optional

import pandas as pd
import requests


def fetch_recent_telemetry(
    backend_url: str, count: int = 200, timeout: float = 2.0
) -> "tuple[pd.DataFrame, bool]":
    
    try:
        response = requests.get(
            f"{backend_url.rstrip('/')}/telemetry/recent",
            params={"count": count},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.exceptions.RequestException:
        return pd.DataFrame(), False
    except ValueError:
       
        return pd.DataFrame(), False

    rows = []
    for item in payload:
        anomaly = item.get("anomaly")
        if anomaly is None:
            
            continue
        rows.append(
            {
                "Timestamp": item.get("published_at"),
                "Anomaly_Score": anomaly.get("anomaly_score"),
                "Zone": anomaly.get("zone", "-"),
                "Description": anomaly.get("description", "-"),
                "Frame_ID": item.get("frame_id"),
                "Is_Critical": anomaly.get("is_critical"),
                "Processing_Latency_Ms": item.get("processing_latency_ms"),
            }
        )

    return pd.DataFrame(rows), True


def fetch_backend_health(backend_url: str, timeout: float = 1.5) -> Optional[dict]:
    
    try:
        response = requests.get(f"{backend_url.rstrip('/')}/health", timeout=timeout)
        response.raise_for_status()
        return response.json()
    except (requests.exceptions.RequestException, ValueError):
        return None
