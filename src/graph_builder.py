"""
graph_builder.py
-----------------
Builds a NetworkX DiGraph representing the supply chain network.

Data sources used:
  - order_large.csv        : edges (Source → Destination), material, weight, deadline
  - distance.csv           : edge weights in meters (Source, Destination, Distance)
  - data/supply_chain.csv  : node metadata (country, product, tier, lat/lon)

The resulting graph has:
  - Nodes: City_XX with attributes (country, city_name, product_category, tier, risk_factor, lat, lon)
  - Edges: directional routes with attributes (distance_m, weight_kg, material, danger_type)
"""

import networkx as nx
import pandas as pd
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Default dataset paths (relative to project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
ORDERS_PATH   = PROJECT_ROOT / "datasets" / "order_large.csv"
DISTANCE_PATH = PROJECT_ROOT / "datasets" / "distance.csv"
SUPPLY_PATH   = PROJECT_ROOT / "data" / "supply_chain.csv"


def load_supply_metadata(supply_path: Path = SUPPLY_PATH) -> pd.DataFrame:
    """
    Load the city → country/product metadata table.
    Returns a DataFrame indexed by city_id.
    """
    df = pd.read_csv(supply_path)
    df.set_index("city_id", inplace=True)
    return df


def load_distance_data(distance_path: Path = DISTANCE_PATH) -> pd.DataFrame:
    """
    Load the pairwise distance data.
    Columns: Source, Destination, Distance(M)
    """
    df = pd.read_csv(distance_path)
    # Rename columns for consistency
    df.columns = ["source", "destination", "distance_m"]
    return df


def load_orders_data(orders_path: Path = ORDERS_PATH) -> pd.DataFrame:
    """
    Load supply chain order routes.
    Columns: Order_ID, Material_ID, Item_ID, Source, Destination,
             Available_Time, Deadline, Danger_Type, Area, Weight
    """
    df = pd.read_csv(orders_path, parse_dates=["Available_Time", "Deadline"])
    return df


def build_graph(
    orders_path:   Path = ORDERS_PATH,
    distance_path: Path = DISTANCE_PATH,
    supply_path:   Path = SUPPLY_PATH,
) -> nx.DiGraph:
    """
    Build and return the supply chain DiGraph.

    Steps:
    1. Load distance data as the backbone
    2. Merge with order data to overlay shipment attributes
    3. Add node metadata from supply_chain.csv
    4. Return graph

    Parameters
    ----------
    orders_path, distance_path, supply_path : Path
        Paths to the three CSV files.

    Returns
    -------
    nx.DiGraph
        Nodes: City_XX
        Edges: directional supply routes with distance/weight attributes
    """
    # ------------------------------------------------------------------
    # 1. Load raw data
    # ------------------------------------------------------------------
    orders_df   = load_orders_data(orders_path)
    distance_df = load_distance_data(distance_path)
    supply_meta = load_supply_metadata(supply_path)

    # ------------------------------------------------------------------
    # 2. Build edge list from distance backbone
    # ------------------------------------------------------------------
    edge_df = distance_df.copy()

    # ------------------------------------------------------------------
    # 3. Aggregate orders and merge with distance backbone
    # ------------------------------------------------------------------
    order_agg = (
        orders_df
        .groupby(["Source", "Destination"], as_index=False)
        .agg(
            total_orders  = ("Order_ID", "count"),
            avg_weight_kg = ("Weight", "mean"),
            danger_type   = ("Danger_Type", "first"),
            material      = ("Material_ID", "first"),
        )
    )
    order_agg.rename(columns={"Source": "source", "Destination": "destination"}, inplace=True)

    edge_df = edge_df.merge(order_agg, on=["source", "destination"], how="left")
    
    # Fill missing values for non-order routes
    edge_df["total_orders"] = edge_df["total_orders"].fillna(0)
    edge_df["avg_weight_kg"] = edge_df["avg_weight_kg"].fillna(0.0)
    edge_df["danger_type"] = edge_df["danger_type"].fillna("None")
    edge_df["material"] = edge_df["material"].fillna("None")

    # ------------------------------------------------------------------
    # 4. Create the DiGraph
    # ------------------------------------------------------------------
    G = nx.DiGraph()

    # Add edges
    for _, row in edge_df.iterrows():
        G.add_edge(
            row["source"],
            row["destination"],
            distance_m    = float(row["distance_m"]),
            avg_weight_kg = float(row["avg_weight_kg"]),
            total_orders  = int(row["total_orders"]),
            danger_type   = str(row["danger_type"]),
            material      = str(row["material"]),
            # Normalised weight for Dijkstra (distance in km)
            weight        = float(row["distance_m"]) / 1000.0,
        )

    # ------------------------------------------------------------------
    # 5. Attach node metadata
    # ------------------------------------------------------------------
    for node in G.nodes():
        if node in supply_meta.index:
            meta = supply_meta.loc[node].to_dict()
        else:
            # Unknown city — assign sensible defaults
            meta = {
                "city_name":        node,
                "country":          "Unknown",
                "region":           "Unknown",
                "product_category": "General",
                "tier":             3,
                "risk_factor":      "medium",
                "lat":              0.0,
                "lon":              0.0,
            }
        # Merge each attribute directly onto the node
        for key, val in meta.items():
            G.nodes[node][key] = val

    return G


def get_graph_summary(G: nx.DiGraph) -> dict:
    """Return a simple summary dict for display in the dashboard."""
    return {
        "nodes":          G.number_of_nodes(),
        "edges":          G.number_of_edges(),
        "countries":      len(set(nx.get_node_attributes(G, "country").values())),
        "is_connected":   nx.is_weakly_connected(G),
        "avg_degree":     round(sum(d for _, d in G.degree()) / G.number_of_nodes(), 2),
    }


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    G = build_graph()
    print("Graph built successfully!")
    summary = get_graph_summary(G)
    for k, v in summary.items():
        print(f"  {k:20s}: {v}")
    # Show a sample of node attributes
    sample_node = list(G.nodes())[0]
    print(f"\nSample node '{sample_node}':")
    print(f"  {G.nodes[sample_node]}")
