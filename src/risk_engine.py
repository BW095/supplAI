"""
risk_engine.py
---------------
Computes per-node risk scores combining:
  - Static: centrality, tier, product criticality
  - Dynamic: cascade depth, weather signals, news intensity
  - Anomaly: Isolation Forest on shipment patterns
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import networkx as nx

ROOT = Path(__file__).resolve().parent.parent


# ──────────────────────────────────────────────
# Centrality
# ──────────────────────────────────────────────

def compute_centrality(G: nx.DiGraph) -> Dict[str, float]:
    """Betweenness centrality normalised 0→1."""
    try:
        bc = nx.betweenness_centrality(G, normalized=True, weight="weight")
    except Exception:
        bc = {n: 0.0 for n in G.nodes}
    max_bc = max(bc.values(), default=1.0)
    return {n: round(v / max(max_bc, 1e-9), 4) for n, v in bc.items()}


# ──────────────────────────────────────────────
# Per-node scoring
# ──────────────────────────────────────────────

TIER_WEIGHT   = {1: 0.9, 2: 0.7, 3: 0.5, 4: 0.3, 5: 0.2}
CRITICAL_PRODUCTS = {"Semiconductors", "Pharmaceuticals", "Electronics", "Automotive"}


def score_nodes(
    G: nx.DiGraph,
    cascade: Dict[str, int],       # node → depth
    severity: str = "medium",
    centrality: Optional[Dict[str, float]] = None,
    weather_signals: Optional[Dict[str, float]] = None,  # node → severity 0–1
) -> pd.DataFrame:
    """
    Compute risk score per node.
    Returns DataFrame: node, city_name, country, product_category, tier,
                       risk_score, risk_level, cascade_depth, centrality, delay_prob
    """
    central = centrality or {}
    weather = weather_signals or {}
    cascade_set = set(cascade.keys())

    SEV_MULT = {"low": 0.5, "medium": 1.0, "high": 1.4, "critical": 1.8}
    sev_m = SEV_MULT.get(severity, 1.0)

    rows = []
    for node in G.nodes:
        if node not in cascade_set:
            continue
        attrs = G.nodes[node]
        depth = cascade.get(node, 99)
        tier = int(attrs.get("tier", 3))
        product = attrs.get("product_category", "General")
        cent = central.get(node, 0.0)

        # Cascade depth penalty (shallower = more affected)
        depth_score = max(0, 1.0 - depth * 0.2)

        # Tier importance
        tier_score = TIER_WEIGHT.get(tier, 0.3)

        # Critical product bonus
        prod_score = 0.85 if product in CRITICAL_PRODUCTS else 0.45

        # Centrality (how many routes pass through)
        cent_score = min(cent * 3.0, 1.0)

        # Weather
        w_score = weather.get(node, 0.0) * 0.5

        # Composite
        raw = (
            depth_score  * 0.40 +
            tier_score   * 0.20 +
            prod_score   * 0.15 +
            cent_score   * 0.15 +
            w_score      * 0.10
        ) * sev_m

        risk_score = round(min(raw, 1.0), 4)

        if risk_score >= 0.70:
            risk_level = "Critical"
        elif risk_score >= 0.50:
            risk_level = "High"
        elif risk_score >= 0.30:
            risk_level = "Medium"
        elif risk_score > 0.0:
            risk_level = "Low"
        else:
            risk_level = "None"

        delay_prob = round(min(0.9, risk_score * 1.1), 4)

        rows.append({
            "node":             node,
            "city_name":        attrs.get("city_name", node),
            "country":          attrs.get("country", ""),
            "product_category": product,
            "tier":             tier,
            "risk_score":       risk_score,
            "risk_level":       risk_level,
            "cascade_depth":    depth,
            "centrality":       round(cent, 4),
            "delay_prob":       delay_prob,
            "lat":              attrs.get("lat", 0.0),
            "lon":              attrs.get("lon", 0.0),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("risk_score", ascending=False).reset_index(drop=True)
    return df


# ──────────────────────────────────────────────
# Anomaly detection (Isolation Forest)
# ──────────────────────────────────────────────

def load_anomaly_model():
    """Load trained Isolation Forest model."""
    model_path = ROOT / "models" / "anomaly_model.pkl"
    if not model_path.exists():
        return None
    import joblib
    return joblib.load(model_path)


def score_anomalies(
    risk_df: pd.DataFrame,
    model=None,
) -> pd.DataFrame:
    """
    Flag anomalous nodes using Isolation Forest.
    Adds columns: anomaly_score, is_anomalous, anomaly_level
    """
    if risk_df.empty:
        return risk_df.copy()

    df = risk_df.copy()
    features = ["risk_score", "centrality", "delay_prob", "tier", "cascade_depth"]
    available = [c for c in features if c in df.columns]

    if not available:
        df["anomaly_score"] = 0.0
        df["is_anomalous"]  = False
        df["anomaly_level"] = "Normal"
        return df

    X = df[available].fillna(0).values

    if model is not None:
        try:
            scores = model.decision_function(X)
            # Normalise: lower IF score = more anomalous → invert and normalise
            norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
            df["anomaly_score"] = np.round(1 - norm, 4)
            df["is_anomalous"]  = model.predict(X) == -1
        except Exception:
            df["anomaly_score"] = df["risk_score"].values
            df["is_anomalous"]  = df["risk_score"] > 0.6
    else:
        # Heuristic fallback
        df["anomaly_score"] = df["risk_score"].values
        df["is_anomalous"]  = df["risk_score"] > 0.6

    df["anomaly_level"] = df["anomaly_score"].apply(
        lambda s: "Critical" if s > 0.8 else ("High" if s > 0.6 else "Normal")
    )
    return df
