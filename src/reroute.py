"""
reroute.py
-----------
Suggests alternate supply routes to avoid disrupted nodes.

Strategy
--------
1. Identify the most critical Source→Destination pairs currently routed
   through disrupted nodes.
2. For each such pair, find the shortest path that avoids ALL disrupted nodes,
   using Dijkstra's algorithm on the distance-weighted graph.
3. Compare the alternate path against the original in terms of:
     - route length (hops)
     - total distance (km)
     - estimated added delay

Returns a list of recommendation dicts ready for the dashboard.
"""

import networkx as nx
import pandas as pd
from typing import Dict, List, Optional, Any


def find_alternates(
    G:               nx.DiGraph,
    affected_nodes:  List[str],
    cascade_result:  Dict[str, int],
    supply_df:       Optional[pd.DataFrame] = None,
    top_pairs:       int = 8,
) -> List[Dict[str, Any]]:
    """
    Find alternate routes avoiding all disrupted nodes.

    Parameters
    ----------
    G               : nx.DiGraph   — full supply chain graph
    affected_nodes  : list[str]    — seed nodes from disruption_input
    cascade_result  : dict         — full cascade (including downstream)
    supply_df       : pd.DataFrame — city metadata (optional, for display)
    top_pairs       : int          — max number of rerouting suggestions

    Returns
    -------
    list of dicts, each containing:
        source          : str   — origin city ID
        destination     : str   — destination city ID
        source_name     : str   — human-readable origin
        destination_name: str   — human-readable destination
        disrupted_path  : list  — original path (passes through disrupted nodes)
        alternate_path  : list  — new safe path
        original_dist_km: float
        alternate_dist_km: float
        distance_delta_km: float  — extra km added by rerouting
        detour_pct      : float  — % increase in distance
        status          : str   — "✅ Alternate Found" / "⚠️ No Alternate"
        hops_original   : int
        hops_alternate  : int
    """
    # Only block the SEED nodes (depth=0) as fully unavailable.
    # Cascade nodes (depth>=1) are "at risk" but can still be used for routing.
    seed_nodes    = {n for n, d in cascade_result.items() if d == 0}
    all_disrupted = seed_nodes   # for rerouting purposes

    # Build a "safe" subgraph that excludes only the directly disrupted seed nodes
    safe_nodes = [n for n in G.nodes if n not in seed_nodes]
    G_safe     = G.subgraph(safe_nodes).copy()

    results: List[Dict[str, Any]] = []

    # Find (source, destination) pairs whose path passes through a seed node
    candidate_pairs = set()

    for disrupted_node in seed_nodes:
        # Predecessors (suppliers feeding into the blocked node)
        for pred in G.predecessors(disrupted_node):
            if pred not in seed_nodes:
                # Route goes: pred → disrupted_node → succ
                for succ in G.successors(disrupted_node):
                    if succ not in seed_nodes:
                        candidate_pairs.add((pred, succ))

    # Also try pairs from seed nodes to major distribution hubs
    hubs = ["City_24", "City_35", "City_38", "City_53", "City_63", "City_47", "City_17"]
    for seed in affected_nodes:
        if seed in G:
            for hub in hubs:
                if hub in G and hub not in all_disrupted:
                    candidate_pairs.add((seed, hub))

    # Evaluate each candidate pair
    for source, destination in list(candidate_pairs)[:top_pairs * 2]:

        if source not in G or destination not in G:
            continue

        # ---- Original path (may pass through disrupted nodes) ----
        try:
            orig_path = nx.shortest_path(G, source, destination, weight="weight")
            orig_dist = sum(
                G[orig_path[i]][orig_path[i+1]].get("weight", 1.0)
                for i in range(len(orig_path) - 1)
            )
        except nx.NetworkXNoPath:
            orig_path  = [source, destination]
            orig_dist  = float("inf")

        # ---- Alternate path (avoid all disrupted nodes) ----
        alt_status   = "✅ Alternate Found"
        alt_path     = []
        alt_dist     = float("inf")
        dist_delta   = 0.0
        detour_pct   = 0.0

        if source in G_safe and destination in G_safe:
            try:
                alt_path = nx.shortest_path(G_safe, source, destination, weight="weight")
                alt_dist = sum(
                    G_safe[alt_path[i]][alt_path[i+1]].get("weight", 1.0)
                    for i in range(len(alt_path) - 1)
                )
                if orig_dist < float("inf"):
                    dist_delta = alt_dist - orig_dist
                    detour_pct = (dist_delta / orig_dist) * 100 if orig_dist > 0 else 0.0
            except nx.NetworkXNoPath:
                alt_status = "⚠️ No Alternate Route"
        else:
            alt_status = "⚠️ No Alternate Route"

        # ---- Human-readable names ----
        def _name(node_id):
            if supply_df is not None and node_id in supply_df.index:
                row = supply_df.loc[node_id]
                return f"{row['city_name']}, {row['country']}"
            return node_id

        results.append({
            "source":              source,
            "destination":         destination,
            "source_name":         _name(source),
            "destination_name":    _name(destination),
            "disrupted_path":      orig_path,
            "alternate_path":      alt_path if alt_path else [],
            "original_dist_km":    round(orig_dist, 1) if orig_dist < float("inf") else None,
            "alternate_dist_km":   round(alt_dist,  1) if alt_dist  < float("inf") else None,
            "distance_delta_km":   round(dist_delta, 1),
            "detour_pct":          round(detour_pct, 1),
            "status":              alt_status,
            "hops_original":       len(orig_path) - 1,
            "hops_alternate":      len(alt_path)  - 1 if alt_path else 0,
        })

    # De-duplicate and sort: found alternates first, then by detour %
    results = _deduplicate(results)
    results.sort(key=lambda r: (r["status"] != "✅ Alternate Found", r["detour_pct"]))

    return results[:top_pairs]


def _deduplicate(results: List[Dict]) -> List[Dict]:
    """Remove duplicate (source, destination) pairs, keeping the best result."""
    seen   = {}
    for r in results:
        key = (r["source"], r["destination"])
        if key not in seen or r["status"] == "✅ Alternate Found":
            seen[key] = r
    return list(seen.values())


def format_path(path: List[str], supply_df: Optional[pd.DataFrame] = None) -> str:
    """Convert a list of city IDs into a human-readable route string."""
    if not path:
        return "No path"

    def _label(node_id: str) -> str:
        if supply_df is not None and node_id in supply_df.index:
            row = supply_df.loc[node_id]
            return f"{row['city_name']} ({row['country']})"
        return node_id

    return " → ".join(_label(n) for n in path)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from graph_builder import build_graph, load_supply_metadata
    from cascade_model import run_cascade

    G        = build_graph()
    supply   = load_supply_metadata()
    cascade  = run_cascade(G, ["City_1", "City_2"], max_depth=3)
    routes   = find_alternates(G, ["City_1", "City_2"], cascade, supply_df=supply)

    print(f"Found {len(routes)} rerouting suggestions:\n")
    for r in routes:
        print(f"  {r['source_name']} → {r['destination_name']}")
        print(f"    Status: {r['status']}")
        if r["alternate_path"]:
            print(f"    Alt path: {format_path(r['alternate_path'], supply)}")
        print(f"    Extra km: +{r['distance_delta_km']} ({r['detour_pct']}%)\n")
