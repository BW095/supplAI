"""
anomaly_detector.py
--------------------
Detects anomalous shipment patterns in supply chain routes using
Isolation Forest — an unsupervised ML algorithm that identifies
outliers without needing labelled data.

How it works:
  1. Train Isolation Forest on the full historical shipment dataset
     (same features as the delay model: distance, weight, SLA, etc.)
  2. Score each supply chain graph node based on the anomaly scores
     of its incoming shipments
  3. Flag nodes where anomaly score deviates significantly from baseline
     (these are routes behaving unusually — possible early disruption signal)

Returns an anomaly_df with columns: node, city_name, anomaly_score,
anomaly_level, is_anomalous — integrated into the risk scoring pipeline.
"""

import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
TRAIN_PATH    = PROJECT_ROOT / "datasets" / "Is_delayed_prediction_Train_2_Avatar_2_Version_1_05_09_2019.csv"
MODEL_DIR     = PROJECT_ROOT / "models"
ANOMALY_PATH  = MODEL_DIR / "anomaly_model.pkl"

# Same feature set as delay_model.py
FEATURE_COLS = [
    "distance", "shipment_weight", "SLA",
    "pickup_metro", "pickup_non_metro",
    "drop_metro",   "drop_non_metro",
    "cp_delay_per_quarter", "cp_ontime_per_quarter",
    "cp_delay_per_month",   "cp_ontime_per_month",
    "holiday_in_between",   "is_sunday_in_between",
]

# Anomaly score threshold (Isolation Forest scores: -1=anomaly, closer to -1 = more anomalous)
# sklearn returns scores in range (-0.5, 0.5) with lower = more anomalous
ANOMALY_THRESHOLD = -0.05   # below this = anomalous


# ---------------------------------------------------------------------------
# Train / load Isolation Forest
# ---------------------------------------------------------------------------
def _train_isolation_forest(
    train_path:  Path = TRAIN_PATH,
    model_path:  Path = ANOMALY_PATH,
) -> Dict[str, Any]:
    from sklearn.ensemble import IsolationForest

    print(f"  [anomaly] Loading training data …")
    cols = FEATURE_COLS + ["is_delayed"]
    df   = pd.read_csv(train_path, usecols=cols).dropna(subset=cols)

    X = df[FEATURE_COLS].values.astype(np.float32)

    print(f"  [anomaly] Training Isolation Forest on {len(X):,} samples …")
    model = IsolationForest(
        n_estimators  = 200,
        contamination = 0.08,   # ~8% anomaly rate (mirrors delay class imbalance)
        max_samples   = "auto",
        random_state  = 42,
        n_jobs        = -1,
    )
    model.fit(X)

    # Compute baseline score distribution
    scores = model.decision_function(X)
    artifact = {
        "model":        model,
        "features":     FEATURE_COLS,
        "score_mean":   float(np.mean(scores)),
        "score_std":    float(np.std(scores)),
        "score_p5":     float(np.percentile(scores, 5)),
        "train_rows":   len(X),
    }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(artifact, model_path)
    print(f"  [anomaly] Model saved to {model_path}")
    print(f"  [anomaly] Baseline score: mean={artifact['score_mean']:.4f} std={artifact['score_std']:.4f} p5={artifact['score_p5']:.4f}")
    return artifact


def load_or_train_anomaly(
    train_path:    Path = TRAIN_PATH,
    model_path:    Path = ANOMALY_PATH,
    force_retrain: bool = False,
) -> Dict[str, Any]:
    if model_path.exists() and not force_retrain:
        print(f"  [anomaly] Loading cached model from {model_path}")
        return joblib.load(model_path)
    return _train_isolation_forest(train_path, model_path)


# ---------------------------------------------------------------------------
# Score a single route feature vector
# ---------------------------------------------------------------------------
def _route_anomaly_score(
    artifact: Dict[str, Any],
    distance_m:     float = 500_000,
    weight_kg:      float = 200,
    sla:            int   = 1,
    pickup_metro:   int   = 1,
    pickup_non_metro: int = 0,
    drop_metro:     int   = 1,
    drop_non_metro: int   = 0,
    cp_delay_q:     float = 0.15,
    cp_ontime_q:    float = 0.85,
    cp_delay_m:     float = 0.15,
    cp_ontime_m:    float = 0.85,
    holiday:        int   = 0,
    is_sunday:      int   = 0,
) -> float:
    """
    Returns the anomaly score for a route (lower = more anomalous).
    sklearn IsolationForest.decision_function: negative = anomaly.
    """
    model = artifact["model"]
    X = np.array([[
        distance_m, weight_kg * 1000,  # kg → g to match training data
        sla, pickup_metro, pickup_non_metro,
        drop_metro, drop_non_metro,
        cp_delay_q, cp_ontime_q,
        cp_delay_m, cp_ontime_m,
        holiday, is_sunday,
    ]], dtype=np.float32)
    return float(model.decision_function(X)[0])


# ---------------------------------------------------------------------------
# Score all graph nodes
# ---------------------------------------------------------------------------
def score_anomalies(
    artifact:   Dict[str, Any],
    G,
    supply_df:  Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute anomaly score for every node in the graph using its
    incoming edge statistics as the feature proxy.

    Returns DataFrame with columns:
        node, city_name, country, anomaly_score, anomaly_z,
        anomaly_level, is_anomalous
    """
    rows = []
    score_mean = artifact["score_mean"]
    score_std  = artifact["score_std"]

    for node in G.nodes():
        node_data = G.nodes[node]
        in_edges  = list(G.in_edges(node, data=True))

        if in_edges:
            dists   = [d.get("distance_m",     500_000) for _, _, d in in_edges]
            weights = [d.get("avg_weight_kg",   200)    for _, _, d in in_edges]
            avg_dist   = float(np.mean(dists))
            avg_weight = float(np.mean(weights))
        else:
            tier       = node_data.get("tier", 3)
            avg_dist   = 500_000 - tier * 50_000
            avg_weight = 200.0

        score = _route_anomaly_score(
            artifact,
            distance_m  = avg_dist,
            weight_kg   = avg_weight,
        )

        # Z-score relative to training baseline
        z = (score - score_mean) / (score_std + 1e-9)

        # Anomaly level
        if score < artifact["score_p5"]:
            level = "🔴 High Anomaly"
        elif score < ANOMALY_THRESHOLD:
            level = "🟠 Moderate Anomaly"
        else:
            level = "🟢 Normal"

        rows.append({
            "node":          node,
            "city_name":     node_data.get("city_name", node),
            "country":       node_data.get("country",  "Unknown"),
            "product":       node_data.get("product_category", "General"),
            "anomaly_score": round(score, 5),
            "anomaly_z":     round(z,     3),
            "anomaly_level": level,
            "is_anomalous":  score < ANOMALY_THRESHOLD,
        })

    df = pd.DataFrame(rows)
    df.sort_values("anomaly_score", ascending=True, inplace=True)   # most anomalous first
    df.reset_index(drop=True, inplace=True)
    return df


# ---------------------------------------------------------------------------
# Plotly: anomaly score bar chart (top N most anomalous)
# ---------------------------------------------------------------------------
def anomaly_bar_figure(anomaly_df: pd.DataFrame, top_n: int = 20):
    import plotly.graph_objects as go

    if anomaly_df.empty:
        return go.Figure()

    df = anomaly_df.head(top_n).copy()
    df = df.sort_values("anomaly_score", ascending=True)

    colours = [
        "#ef4444" if r["anomaly_score"] < anomaly_df["anomaly_score"].quantile(0.05)
        else ("#f97316" if r["is_anomalous"] else "#22c55e")
        for _, r in df.iterrows()
    ]

    fig = go.Figure(go.Bar(
        x = df["anomaly_score"],
        y = df["city_name"] + " (" + df["country"] + ")",
        orientation = "h",
        marker      = dict(color=colours, line=dict(width=0)),
        text        = [f"{s:.4f}" for s in df["anomaly_score"]],
        textposition = "outside",
        textfont    = dict(color="#94a3b8", size=10),
        hovertemplate = "<b>%{y}</b><br>Anomaly Score: %{x:.5f}<extra></extra>",
    ))

    fig.add_vline(
        x=ANOMALY_THRESHOLD, line_dash="dash",
        line_color="#f97316", line_width=1.5,
        annotation_text="Anomaly threshold",
        annotation_font_color="#f97316",
    )

    fig.update_layout(
        title       = dict(text=f"Shipment Anomaly Detection — Top {top_n} Nodes",
                           font=dict(color="#e2e8f0", size=14, family="Inter"), x=0),
        paper_bgcolor = "#0a0e1a",
        plot_bgcolor  = "#0a0e1a",
        xaxis = dict(title="Isolation Forest Score (lower = more anomalous)",
                     color="#94a3b8", gridcolor="#1e293b"),
        yaxis = dict(color="#e2e8f0", tickfont=dict(size=10)),
        font  = dict(color="#e2e8f0", family="Inter"),
        height = 500,
        margin = dict(t=40, b=40, l=10, r=80),
    )
    return fig


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from graph_builder import build_graph

    artifact = load_or_train_anomaly()
    G        = build_graph()
    df       = score_anomalies(artifact, G)

    print(f"\nAnomaly scores computed for {len(df)} nodes")
    print(f"Anomalous nodes: {df['is_anomalous'].sum()}")
    print("\nTop 10 most anomalous:")
    print(df[["node", "city_name", "country", "anomaly_score", "anomaly_level"]].head(10).to_string(index=False))
