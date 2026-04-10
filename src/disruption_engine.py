"""
disruption_engine.py
---------------------
Unified disruption pipeline:
  1. Receive external signal (news event, weather, manual simulation)
  2. Map to affected graph nodes
  3. Run cascade propagation
  4. Compute impact metrics (shipments affected, cost, delay)
  5. Return structured DisruptionReport
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set

import networkx as nx
import pandas as pd


# ──────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────

@dataclass
class DisruptionReport:
    event_id: str
    event_text: str
    source: str                    # "news" | "weather" | "earthquake" | "simulation"
    severity: str                  # "low" | "medium" | "high" | "critical"
    category: str                  # "geopolitical" | "natural_disaster" | "labor" | etc.
    seed_nodes: List[str]          # directly hit nodes
    cascade_nodes: Dict[str, int]  # node_id → cascade_depth
    affected_countries: List[str]
    affected_sectors: List[str]
    estimated_delay_days: float
    estimated_cost_usd: float
    shipments_at_risk: int
    timestamp: str
    raw_signal: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


# ──────────────────────────────────────────────
# Country → node mapping (based on real hub list)
# ──────────────────────────────────────────────

COUNTRY_NODE_MAP: Dict[str, List[str]] = {
    "china":        ["SHA", "SZX", "GZH", "TJN", "CHG", "HKG"],
    "singapore":    ["SGP"],
    "japan":        ["TYO", "OSK"],
    "south korea":  ["BUS", "ICN"],
    "india":        ["MUM", "DEL", "CHE"],
    "malaysia":     ["KUL"],
    "thailand":     ["BKK"],
    "indonesia":    ["JKT"],
    "vietnam":      ["HCM"],
    "taiwan":       ["TPE"],
    "australia":    ["SYD"],
    "netherlands":  ["RTM"],
    "germany":      ["HAM", "FRA"],
    "belgium":      ["ANT"],
    "uk":           ["FLX", "LON"],
    "france":       ["PAR"],
    "italy":        ["MIL"],
    "spain":        ["BCN"],
    "greece":       ["PIR"],
    "turkey":       ["IST"],
    "poland":       ["WAR"],
    "uae":          ["DXB", "ABU"],
    "saudi arabia": ["JED"],
    "egypt":        ["SUZ"],
    "south africa": ["CPT"],
    "djibouti":     ["DJI"],
    "nigeria":      ["LGS"],
    "usa":          ["LAX", "NYC", "CHI", "HOU", "MIA", "SEA"],
    "canada":       ["YVR"],
    "mexico":       ["MEX"],
    "panama":       ["PAN"],
    "brazil":       ["SAO", "SSZ"],
    "colombia":     ["BOG"],
    "argentina":    ["BUE"],
    "russia":       ["MSC", "NVS"],
    "kazakhstan":   ["ALM"],
    "pakistan":     ["KHI"],
    "sri lanka":    ["COL"],
    "qatar":        ["DOH"],
    "red sea":      ["SUZ", "DJI", "JED"],
    "suez":         ["SUZ"],
    "suez canal":   ["SUZ"],
    "strait of malacca": ["SGP", "KUL"],
    "panama canal": ["PAN"],
    "middle east":  ["DXB", "ABU", "DJI", "JED", "DOH"],
    "europe":       ["RTM", "HAM", "ANT", "FLX", "LON", "PAR", "FRA"],
    "southeast asia": ["SGP", "KUL", "BKK", "JKT", "HCM"],
    "east asia":    ["SHA", "HKG", "TYO", "BUS", "TPE"],
}

SECTOR_NODE_MAP: Dict[str, List[str]] = {
    "electronics":    ["SHA", "SZX", "ICN", "SGP", "TPE", "KUL"],
    "semiconductors": ["TPE", "ICN", "SGP", "SHA", "SZX"],
    "automotive":     ["TYO", "HAM", "BUS", "CHE", "MEX"],
    "pharmaceuticals":["MUM", "FRA", "NYC", "LON"],
    "textiles":       ["HCM", "DEL", "IST", "KHI", "BCN"],
    "oil & gas":      ["DXB", "JED", "HOU", "LGS", "ABU", "DOH"],
    "food":           ["BKK", "SAO", "MIA", "BOG", "BUE", "SSZ"],
    "chemicals":      ["RTM", "ANT", "HAM"],
    "shipping":       ["SHA", "SGP", "RTM", "LAX", "DXB"],
    "retail":         ["LAX", "NYC", "FLX", "SHA"],
    "manufacturing":  ["SHA", "GZH", "HAM", "OSK", "CHI"],
    "raw materials":  ["SYD", "CPT", "LGS", "MSC", "YVR"],
    "luxury":         ["MIL", "PAR", "LON"],
    "transshipment":  ["SGP", "SUZ", "DXB", "PAN", "PIR", "COL"],
}


def map_signal_to_nodes(
    countries: List[str],
    sectors: List[str],
    keywords: List[str] = None,
) -> List[str]:
    """Map a disruption signal's countries/sectors to hub node IDs."""
    nodes: Set[str] = set()

    for c in countries:
        key = c.lower().strip()
        # exact match
        if key in COUNTRY_NODE_MAP:
            nodes.update(COUNTRY_NODE_MAP[key])
        else:
            # partial match
            for map_key, ids in COUNTRY_NODE_MAP.items():
                if map_key in key or key in map_key:
                    nodes.update(ids)
                    break

    for s in sectors:
        key = s.lower().strip()
        if key in SECTOR_NODE_MAP:
            nodes.update(SECTOR_NODE_MAP[key])
        else:
            for map_key, ids in SECTOR_NODE_MAP.items():
                if map_key in key or key in map_key:
                    nodes.update(ids)
                    break

    # keyword fallback
    if keywords and not nodes:
        for kw in keywords:
            k = kw.lower()
            for map_key, ids in COUNTRY_NODE_MAP.items():
                if k in map_key or map_key in k:
                    nodes.update(ids)

    return list(nodes)


# ──────────────────────────────────────────────
# Cascade propagation
# ──────────────────────────────────────────────

def run_cascade(
    G: nx.DiGraph,
    seed_nodes: List[str],
    max_depth: int = 3,
    decay: float = 0.6,
) -> Dict[str, int]:
    """
    BFS cascade from seed_nodes outward — limited to 3 hops.
    Returns dict: node_id → cascade_depth (0 = seed)
    Only propagates through HIGH-CONNECTIVITY nodes to avoid flooding whole graph.
    """
    visited: Dict[str, int] = {}
    frontier: Set[str] = set()

    # Degree threshold: only cascade through nodes with degree >= 2
    for n in seed_nodes:
        if n in G:
            visited[n] = 0
            frontier.add(n)

    for depth in range(1, max_depth + 1):
        next_frontier: Set[str] = set()
        for node in frontier:
            # Only propagate to successors that are hubs (tier <= 2) or high-degree
            for succ in G.successors(node):
                if succ not in visited:
                    succ_tier = G.nodes[succ].get("tier", 3)
                    # Depth 1: always cascade; depth 2+: only to tier ≤ 2 nodes
                    if depth == 1 or succ_tier <= 2:
                        visited[succ] = depth
                        next_frontier.add(succ)
        frontier = next_frontier
        if not frontier:
            break

    return visited


# ──────────────────────────────────────────────
# Impact estimator
# ──────────────────────────────────────────────

SEVERITY_WEIGHTS = {"low": 0.15, "medium": 0.35, "high": 0.65, "critical": 0.90}
SEVERITY_DELAY   = {"low": 1.5,  "medium": 4.0,  "high": 9.0,  "critical": 18.0}
SEVERITY_COST    = {"low": 50_000, "medium": 250_000, "high": 800_000, "critical": 3_000_000}


def estimate_impact(
    G: nx.DiGraph,
    cascade: Dict[str, int],
    severity: str,
    shipments: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """Estimate cost, delay, and shipment impact from a cascade."""
    n_affected = len(cascade)
    weight = SEVERITY_WEIGHTS.get(severity, 0.35)

    # Base estimates scaled by cascade size
    delay_days = SEVERITY_DELAY.get(severity, 4.0) * (1 + n_affected * 0.03)
    cost_usd = SEVERITY_COST.get(severity, 250_000) * (1 + n_affected * 0.15)

    # Count affected shipments
    cascade_set = set(cascade.keys())
    at_risk = 0
    if shipments:
        at_risk = sum(
            1 for s in shipments
            if s.get("origin") in cascade_set or s.get("destination") in cascade_set
        )

    return {
        "nodes_affected": n_affected,
        "estimated_delay_days": round(delay_days, 1),
        "estimated_cost_usd": round(cost_usd, 0),
        "shipments_at_risk": at_risk,
        "cascade_depth_max": max(cascade.values()) if cascade else 0,
    }


# ──────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────

def process_disruption(
    G: nx.DiGraph,
    signal: Dict[str, Any],
    shipments: Optional[List[Dict]] = None,
    event_id: Optional[str] = None,
) -> DisruptionReport:
    """
    Full pipeline: map signal → cascade → estimate impact → return report.

    signal fields (standard):
      event_text, source, severity, category,
      affected_countries, affected_sectors,
      affected_nodes (if pre-mapped), keywords_hit
    """
    import uuid
    from datetime import datetime, timezone

    eid = event_id or f"EVT-{uuid.uuid4().hex[:8].upper()}"
    now = datetime.now(timezone.utc).isoformat()

    countries = signal.get("affected_countries", signal.get("country_hit", []))
    sectors   = signal.get("affected_sectors",   signal.get("product_hit", []))
    keywords  = signal.get("keywords_hit", [])
    severity  = signal.get("severity", "medium")
    category  = signal.get("category", "other")

    # Resolve seed nodes
    seed_nodes: List[str] = list(signal.get("affected_nodes", []))
    if not seed_nodes:
        seed_nodes = map_signal_to_nodes(countries, sectors, keywords)
    # Validate against graph
    seed_nodes = [n for n in seed_nodes if n in G]

    # Cascade
    cascade = run_cascade(G, seed_nodes, max_depth=4)

    # Impact
    impact = estimate_impact(G, cascade, severity, shipments)

    # Countries affected in cascade
    cascade_countries = list({
        G.nodes[n].get("country", "") for n in cascade if n in G
    })

    return DisruptionReport(
        event_id=eid,
        event_text=signal.get("event_text", "Unknown disruption"),
        source=signal.get("source", "unknown"),
        severity=severity,
        category=category,
        seed_nodes=seed_nodes,
        cascade_nodes=cascade,
        affected_countries=cascade_countries,
        affected_sectors=sectors,
        estimated_delay_days=impact["estimated_delay_days"],
        estimated_cost_usd=impact["estimated_cost_usd"],
        shipments_at_risk=impact["shipments_at_risk"],
        timestamp=now,
        raw_signal=signal,
    )
