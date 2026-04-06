"""
shap_explain.py
---------------
ML explainability layer for the delay prediction model using SHAP
(SHapley Additive exPlanations).

Provides:
  compute_shap()          → dict of {node_id: {feature: shap_val}} for top-N nodes
  shap_bar_figure()       → Plotly Figure — global mean |SHAP| feature importance
  shap_waterfall_figure() → Plotly Figure — per-node waterfall (push/pull per feature)
  shap_to_text()          → plain-English summary string (for Gemini prompt injection)

Works with both XGBoost (TreeExplainer, fast) and RandomForest (TreeExplainer) backends.
Falls back gracefully if SHAP is not installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Feature display names — human-readable labels for the 13 model features
# ---------------------------------------------------------------------------
FEATURE_LABELS: Dict[str, str] = {
    "distance":              "Route Distance",
    "shipment_weight":       "Shipment Weight",
    "SLA":                   "SLA (days)",
    "pickup_metro":          "Pickup Metro",
    "pickup_non_metro":      "Pickup Non-Metro",
    "drop_metro":            "Drop Metro",
    "drop_non_metro":        "Drop Non-Metro",
    "cp_delay_per_quarter":  "Carrier Delay Rate (Q)",
    "cp_ontime_per_quarter": "Carrier On-Time Rate (Q)",
    "cp_delay_per_month":    "Carrier Delay Rate (M)",
    "cp_ontime_per_month":   "Carrier On-Time Rate (M)",
    "holiday_in_between":    "Holiday En Route",
    "is_sunday_in_between":  "Sunday En Route",
}

FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "distance":              "Total route distance in metres. Longer routes carry higher delay probability.",
    "shipment_weight":       "Cargo weight in grams. Heavier loads face more handling complexity.",
    "SLA":                   "Service Level Agreement in days. Tighter SLAs leave less buffer.",
    "pickup_metro":          "1 if the pickup location is a metro (urban) hub — generally lower delay.",
    "pickup_non_metro":      "1 if the pickup location is non-metropolitan — higher delay risk.",
    "drop_metro":            "1 if the delivery location is a metro hub — generally lower delay.",
    "drop_non_metro":        "1 if the delivery location is non-metropolitan — higher delay risk.",
    "cp_delay_per_quarter":  "Fraction of carrier shipments delayed this quarter (0–1). High = riskier carrier.",
    "cp_ontime_per_quarter": "Fraction of carrier shipments on-time this quarter (0–1). High = reliable carrier.",
    "cp_delay_per_month":    "Fraction of carrier shipments delayed this month (0–1).",
    "cp_ontime_per_month":   "Fraction of carrier shipments on-time this month (0–1).",
    "holiday_in_between":    "1 if a public holiday falls within the shipment window — adds buffer days.",
    "is_sunday_in_between":  "1 if a Sunday falls within the shipment window — logistics slowdown.",
}

# ---------------------------------------------------------------------------
# Build feature matrix for a set of nodes
# ---------------------------------------------------------------------------

def _node_features(
    artifact: Dict[str, Any],
    G,
    nodes: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    Build an (N × 13) float32 feature matrix for the supplied node list
    using the same defaults as delay_model.predict_delay_proba.

    Returns (X, valid_nodes) — rows in X map 1-to-1 to valid_nodes.
    """
    rows: List[np.ndarray] = []
    valid: List[str] = []

    for node in nodes:
        if node not in G.nodes:
            continue

        in_edges = list(G.in_edges(node, data=True))
        if in_edges:
            dists   = [d.get("distance_m",     500_000) for _, _, d in in_edges]
            weights = [d.get("avg_weight_kg",   200)    for _, _, d in in_edges]
            avg_dist   = float(np.mean(dists))
            avg_weight = float(np.mean(weights)) * 1000   # kg → g
        else:
            tier = G.nodes[node].get("tier", 3)
            avg_dist   = 500_000 - tier * 50_000
            avg_weight = 200_000

        row = np.array([
            avg_dist,
            avg_weight,
            1,      # SLA default
            1,      # pickup_metro
            0,
            1,      # drop_metro
            0,
            0.15,   # cp_delay_per_quarter
            0.85,
            0.15,   # cp_delay_per_month
            0.85,
            0,      # holiday_in_between
            0,      # is_sunday_in_between
        ], dtype=np.float32)

        rows.append(row)
        valid.append(node)

    if not rows:
        return np.empty((0, 13), dtype=np.float32), []

    return np.vstack(rows), valid


# ---------------------------------------------------------------------------
# Core SHAP computation
# ---------------------------------------------------------------------------

def compute_shap(
    artifact:  Dict[str, Any],
    risk_df:   pd.DataFrame,
    G,
    top_n:     int = 20,
) -> Dict[str, Dict[str, float]]:
    """
    Compute SHAP values for the top-N highest-risk nodes.

    Parameters
    ----------
    artifact  : dict from delay_model.load_or_train()
    risk_df   : DataFrame from risk_scoring.score_nodes()
    G         : NetworkX DiGraph
    top_n     : max nodes to explain (keep small for speed)

    Returns
    -------
    dict  {node_id: {feature_name: shap_value_float}}
    Empty dict if SHAP is unavailable or no nodes to explain.
    """
    try:
        import shap
    except ImportError:
        print("  [shap_explain] shap not installed — skipping explainability")
        return {}

    if risk_df.empty:
        return {}

    model    = artifact["model"]
    features = artifact["features"]   # ordered list of 13 feature names

    # Select top-N nodes by risk score
    top_nodes = risk_df.nlargest(top_n, "risk_score")["node"].tolist()

    X, valid_nodes = _node_features(artifact, G, top_nodes)
    if X.shape[0] == 0:
        return {}

    # Use TreeExplainer (works for XGBoost + RandomForest)
    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # RandomForest returns list[array] (one per class) — take class-1
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

    except Exception as e:
        print(f"  [shap_explain] TreeExplainer failed: {e} — using KernelExplainer")
        try:
            bg = shap.kmeans(X, min(10, X.shape[0]))
            explainer   = shap.KernelExplainer(model.predict_proba, bg)
            shap_values = explainer.shap_values(X, nsamples=50)[:, :, 1]
        except Exception as e2:
            print(f"  [shap_explain] KernelExplainer also failed: {e2}")
            return {}

    # Map back to dict
    result: Dict[str, Dict[str, float]] = {}
    for i, node in enumerate(valid_nodes):
        result[node] = {
            feat: float(shap_values[i, j])
            for j, feat in enumerate(features)
        }

    print(f"  [shap_explain] Computed SHAP for {len(result)} nodes")
    return result


# ---------------------------------------------------------------------------
# Plotly: Global feature importance bar chart
# ---------------------------------------------------------------------------

def shap_bar_figure(
    shap_results: Dict[str, Dict[str, float]],
) -> go.Figure:
    """
    Horizontal bar chart of mean |SHAP| across all explained nodes.
    Styled to match the SupplAI dark theme.
    """
    if not shap_results:
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor="#0a0e1a",
            plot_bgcolor="#0a0e1a",
            annotations=[dict(
                text="No SHAP data available", x=0.5, y=0.5,
                xref="paper", yref="paper", showarrow=False,
                font=dict(color="#64748b", size=14),
            )],
        )
        return fig

    # Aggregate mean |SHAP| per feature
    all_features = list(next(iter(shap_results.values())).keys())
    mean_abs: Dict[str, float] = {}
    for feat in all_features:
        vals = [abs(shap_results[node].get(feat, 0.0)) for node in shap_results]
        mean_abs[feat] = float(np.mean(vals))

    # Sort by importance
    sorted_items = sorted(mean_abs.items(), key=lambda x: x[1])
    feats  = [FEATURE_LABELS.get(f, f) for f, _ in sorted_items]
    values = [v for _, v in sorted_items]

    # Colour gradient: low importance = slate, high = purple-red
    max_v  = max(values) if values else 1.0
    colours = [
        f"rgba({int(99 + 150*(v/max_v))}, {int(102 - 60*(v/max_v))}, {int(241 - 150*(v/max_v))}, 0.85)"
        for v in values
    ]

    fig = go.Figure(go.Bar(
        x=values,
        y=feats,
        orientation="h",
        marker=dict(color=colours, line=dict(width=0)),
        text=[f"{v:.4f}" for v in values],
        textposition="outside",
        textfont=dict(color="#94a3b8", size=11),
        hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.5f}<extra></extra>",
    ))

    fig.update_layout(
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0a0e1a",
        xaxis=dict(
            title="Mean |SHAP value| — impact on delay probability",
            color="#94a3b8", gridcolor="#1e293b",
            title_font=dict(size=12),
        ),
        yaxis=dict(color="#e2e8f0", tickfont=dict(size=11)),
        font=dict(color="#e2e8f0", family="Inter"),
        height=420,
        margin=dict(t=30, b=40, l=10, r=80),
        title=dict(
            text="Global Feature Importance (mean |SHAP| across all risk nodes)",
            font=dict(color="#e2e8f0", size=14, family="Inter"),
            x=0,
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Plotly: Per-node waterfall chart
# ---------------------------------------------------------------------------

def shap_waterfall_figure(
    node_shap:  Dict[str, float],
    node_name:  str,
    base_value: float = 0.5,
) -> go.Figure:
    """
    Waterfall chart showing how each feature pushes the delay probability
    up (red) or down (green) from the baseline for a specific node.
    """
    if not node_shap:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a")
        return fig

    # Sort by absolute SHAP value (most impactful first)
    sorted_items = sorted(node_shap.items(), key=lambda x: abs(x[1]), reverse=True)

    labels = ["Base"] + [FEATURE_LABELS.get(f, f) for f, _ in sorted_items] + ["Final Score"]
    values = [base_value] + [v for _, v in sorted_items]

    # Build cumulative for waterfall
    running = base_value
    measures = ["absolute"]
    texts    = [f"{base_value:.3f}"]
    for _, v in sorted_items:
        running += v
        measures.append("relative")
        texts.append(f"{v:+.4f}")

    measures.append("total")
    texts.append(f"{running:.3f}")

    # Colour: positive SHAP = risk-increasing (red), negative = risk-reducing (green)
    marker_colours = ["#6366f1"]   # base purple
    for _, v in sorted_items:
        marker_colours.append("#ef4444" if v > 0 else "#22c55e")
    marker_colours.append("#f97316")  # final total = orange

    shap_vals_for_fig = [base_value] + [v for _, v in sorted_items]

    fig = go.Figure(go.Waterfall(
        name="SHAP",
        orientation="v",
        measure=measures,
        x=labels,
        y=shap_vals_for_fig + [None],  # last point is "total"
        text=texts,
        textposition="outside",
        connector=dict(line=dict(color="#334155", width=1, dash="dot")),
        increasing=dict(marker=dict(color="#ef4444")),
        decreasing=dict(marker=dict(color="#22c55e")),
        totals=dict(marker=dict(color="#f97316")),
        textfont=dict(color="#e2e8f0", size=11),
    ))

    fig.update_layout(
        title=dict(
            text=f"SHAP Waterfall — Why <b>{node_name}</b> is at risk",
            font=dict(color="#e2e8f0", size=14, family="Inter"),
            x=0,
        ),
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0a0e1a",
        xaxis=dict(
            color="#94a3b8", tickangle=-35,
            tickfont=dict(size=10),
            gridcolor="#1e293b",
        ),
        yaxis=dict(
            title="Delay Probability Contribution",
            color="#94a3b8", gridcolor="#1e293b",
        ),
        font=dict(color="#e2e8f0", family="Inter"),
        height=420,
        margin=dict(t=50, b=80, l=20, r=20),
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Plain-English SHAP summary (for Gemini prompt injection)
# ---------------------------------------------------------------------------

def shap_to_text(
    node_shap: Dict[str, float],
    node_name: str,
    top_k: int = 5,
) -> str:
    """
    Produce a compact plain-English bullet list of the top-K SHAP drivers
    for a given node.  Injected directly into the Gemini prompt.

    Example output:
      • Route Distance: +0.182 (long haul increases delay risk significantly)
      • Carrier Delay Rate (Q): +0.141 (carrier has poor quarterly track record)
      • Holiday En Route: +0.089 (public holiday adds buffer time)
      • SLA (days): -0.061 (tight SLA reduces margin for error)
      • Carrier On-Time Rate (Q): -0.048 (reliable carrier partially mitigates risk)
    """
    if not node_shap:
        return "No SHAP data available."

    sorted_items = sorted(node_shap.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
    lines = [f"SHAP feature drivers for {node_name}:"]
    for feat, val in sorted_items:
        label = FEATURE_LABELS.get(feat, feat)
        desc  = FEATURE_DESCRIPTIONS.get(feat, "")
        direction = "increases" if val > 0 else "decreases"
        lines.append(f"  • {label}: {val:+.4f}  ({desc.split('.')[0].lower()} — {direction} delay risk)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quick standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))

    from delay_model import load_or_train
    from graph_builder import build_graph

    print("Loading model …")
    artifact = load_or_train()

    print("Building graph …")
    G = build_graph()

    # Mock a tiny risk_df
    nodes = list(G.nodes())[:10]
    risk_df = pd.DataFrame({"node": nodes, "risk_score": [0.9 - i*0.05 for i in range(len(nodes))]})

    print("Computing SHAP …")
    results = compute_shap(artifact, risk_df, G, top_n=5)

    if results:
        top_node = next(iter(results))
        print(f"\nTop node: {top_node}")
        print(shap_to_text(results[top_node], top_node))
    else:
        print("No SHAP results (shap may not be installed).")
