"""
graph_engine.py
----------------
Builds and manages the real-world supply chain graph using NetworkX.
City IDs are real IATA/port codes (SHA, RTM, LAX, etc.)
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


# ──────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────

def load_metadata() -> pd.DataFrame:
    """Return node metadata DataFrame indexed by city_id."""
    path = DATA_DIR / "supply_chain.csv"
    if not path.exists():
        raise FileNotFoundError(
            "Run data/generate_world_network.py first to create supply_chain.csv"
        )
    df = pd.read_csv(path)
    df = df.set_index("city_id")
    return df


def load_routes() -> pd.DataFrame:
    """Return edges DataFrame."""
    path = DATA_DIR / "routes.csv"
    if not path.exists():
        raise FileNotFoundError(
            "Run data/generate_world_network.py first to create routes.csv"
        )
    return pd.read_csv(path)


def build_graph(
    metadata: Optional[pd.DataFrame] = None,
    routes: Optional[pd.DataFrame] = None,
) -> nx.DiGraph:
    """
    Build a directed supply chain graph.
    Node attrs: city_name, country, lat, lon, tier, type, product_category
    Edge attrs: distance_km, tariff_rate, transport_mode, transit_days, weight
    """
    meta = metadata if metadata is not None else load_metadata()
    rts = routes if routes is not None else load_routes()

    G = nx.DiGraph()

    # Add nodes
    for city_id, row in meta.iterrows():
        G.add_node(city_id, **{
            "city_name":       row["city_name"],
            "country":         row["country"],
            "lat":             float(row["lat"]),
            "lon":             float(row["lon"]),
            "tier":            int(row["tier"]),
            "type":            row["type"],
            "product_category": row["product_category"],
            "risk_score":      0.0,
            "is_disrupted":    False,
            "is_sanctioned":   row["country"] in ("Russia",),  # basic OFAC placeholder
        })

    # Add edges
    for _, r in rts.iterrows():
        if r["source"] in G and r["destination"] in G:
            G.add_edge(
                r["source"], r["destination"],
                distance_km=float(r["distance_km"]),
                tariff_rate=float(r["tariff_rate"]),
                transport_mode=str(r["transport_mode"]),
                transit_days=float(r["transit_days"]),
                weight=float(r["weight"]),
            )

    return G


def get_graph_summary(G: nx.DiGraph) -> Dict[str, Any]:
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "countries": len({G.nodes[n]["country"] for n in G.nodes}),
        "hubs": [G.nodes[n]["city_name"] for n in G.nodes if G.nodes[n]["tier"] == 1],
        "avg_degree": round(sum(dict(G.degree()).values()) / max(G.number_of_nodes(), 1), 2),
    }


def get_safe_subgraph(G: nx.DiGraph, banned_nodes: set) -> nx.DiGraph:
    """Return a subgraph excluding banned (disrupted / sanctioned) nodes."""
    safe = [n for n in G.nodes if n not in banned_nodes]
    return G.subgraph(safe).copy()


def node_label(G: nx.DiGraph, node_id: str) -> str:
    """Human-readable label for a node."""
    attrs = G.nodes.get(node_id, {})
    city = attrs.get("city_name", node_id)
    country = attrs.get("country", "")
    return f"{city}, {country}" if country else city


def get_hub_cities(G: nx.DiGraph) -> List[str]:
    """Return Tier-1 hub node IDs."""
    return [n for n in G.nodes if G.nodes[n].get("tier", 3) == 1]
