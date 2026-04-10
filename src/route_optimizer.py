"""
route_optimizer.py
-------------------
Finds and ranks alternative supply routes that avoid disrupted nodes.
Scores alternatives by: delay, cost delta, tariff exposure, route safety.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import networkx as nx
import pandas as pd


# ──────────────────────────────────────────────
# Data structure
# ──────────────────────────────────────────────

@dataclass
class RouteAlternative:
    source: str
    destination: str
    source_label: str
    dest_label: str
    status: str                    # "found" | "no_path" | "blocked"
    original_path: List[str]
    alternate_path: List[str]
    original_dist_km: float
    alternate_dist_km: float
    distance_delta_km: float
    detour_pct: float
    original_cost_usd: float
    alternate_cost_usd: float
    cost_delta_usd: float
    original_transit_days: float
    alternate_transit_days: float
    delay_delta_days: float
    max_tariff_pct: float
    route_safety: str              # "clean" | "partial_exposure" | "hidden_dependency"
    safety_note: str
    score: float                   # composite score (lower = better)
    alt_path_labels: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


# ──────────────────────────────────────────────
# Cost model (similar to V1 but improved)
# ──────────────────────────────────────────────

BASE_RATE_NORMAL    = 0.45   # USD/km (bulk contract)
BASE_RATE_EMERGENCY = 2.80   # USD/km (spot market)
EMERGENCY_PENALTY   = 28_000 # flat emergency premium


def _calc_cost(path: List[str], G: nx.DiGraph, is_alternate: bool) -> float:
    if len(path) < 2:
        return 0.0
    cost = 0.0
    rate = BASE_RATE_EMERGENCY if is_alternate else BASE_RATE_NORMAL
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if not G.has_edge(u, v):
            continue
        edge = G[u][v]
        dist = edge.get("distance_km", 1000)
        tariff = edge.get("tariff_rate", 0.0)
        cost += dist * rate + 500 + (50_000 * tariff)
    if is_alternate:
        cost += EMERGENCY_PENALTY
    return round(cost, 2)


def _calc_transit(path: List[str], G: nx.DiGraph) -> float:
    if len(path) < 2:
        return 0.0
    days = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if G.has_edge(u, v):
            days += G[u][v].get("transit_days", 5)
    return round(days, 1)


def _calc_dist(path: List[str], G: nx.DiGraph) -> float:
    dist = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if G.has_edge(u, v):
            dist += G[u][v].get("distance_km", 0)
    return round(dist, 1)


def _node_label(node: str, G: nx.DiGraph) -> str:
    attrs = G.nodes.get(node, {})
    city = attrs.get("city_name", node)
    country = attrs.get("country", "")
    return f"{city}, {country}" if country else city


def _max_tariff(path: List[str], G: nx.DiGraph) -> float:
    rates = [G[path[i]][path[i+1]].get("tariff_rate", 0.0)
             for i in range(len(path) - 1) if G.has_edge(path[i], path[i+1])]
    return round(max(rates, default=0.0) * 100, 2)


def _check_route_safety(
    alt_path: List[str],
    cascade_set: set,
    G: nx.DiGraph,
) -> tuple[str, str]:
    """Check if any intermediate node on alternate path has upstream cascade exposure."""
    if len(alt_path) < 3:
        return "clean", "Direct connection — no intermediate hops."

    intermediates = alt_path[1:-1]
    worst_ratio = 0.0
    exposed = []

    for node in intermediates:
        ancestors = set()
        frontier = {node}
        for _ in range(3):  # 3-hop upstream
            next_f = set()
            for n in frontier:
                for pred in G.predecessors(n):
                    if pred not in ancestors:
                        next_f.add(pred)
            ancestors |= next_f
            frontier = next_f

        if ancestors:
            ratio = len(ancestors & cascade_set) / len(ancestors)
            worst_ratio = max(worst_ratio, ratio)
            if ratio > 0.1:
                exposed.append(_node_label(node, G))

    if worst_ratio >= 0.50:
        return "hidden_dependency", f"⛔ {', '.join(exposed)} source 50%+ inputs from disrupted region — this route may fail too."
    elif worst_ratio >= 0.20:
        return "partial_exposure", f"⚠️ {', '.join(exposed)} have partial upstream exposure. Monitor closely."
    else:
        return "clean", "✅ No significant upstream exposure. Route is genuinely independent."


# ──────────────────────────────────────────────
# Main optimizer
# ──────────────────────────────────────────────

def find_alternates(
    G: nx.DiGraph,
    cascade: Dict[str, int],   # node → depth
    top_k: int = 8,
) -> List[RouteAlternative]:
    """
    Find top_k alternate routes that avoid disrupted seed nodes.
    Returns list sorted by composite score (best first).
    """
    cascade_set = set(cascade.keys())
    seed_nodes  = {n for n, d in cascade.items() if d == 0}

    # Safe subgraph: remove seed nodes (and sanctioned nodes)
    banned = seed_nodes | {n for n in G.nodes if G.nodes[n].get("is_sanctioned", False)}
    safe_nodes = [n for n in G.nodes if n not in banned]
    G_safe = G.subgraph(safe_nodes).copy()

    # Build candidate (src, via, dst) triples
    triples = []
    seen_pairs: set = set()
    for disrupted in seed_nodes:
        for pred in G.predecessors(disrupted):
            if pred not in seed_nodes:
                for succ in G.successors(disrupted):
                    if succ not in seed_nodes and succ != pred:  # skip self-loops
                        pair = (pred, succ)
                        if pair not in seen_pairs:
                            seen_pairs.add(pair)
                            triples.append((pred, disrupted, succ))

    random.shuffle(triples)
    triples = triples[: top_k * 5]

    results: List[RouteAlternative] = []

    for src, via, dst in triples:
        if src not in G or dst not in G:
            continue

        # Original path (2 hops through disrupted node)
        orig_path = [src, via, dst] if G.has_edge(src, via) and G.has_edge(via, dst) else [src, dst]
        orig_dist  = _calc_dist(orig_path, G)
        orig_cost  = _calc_cost(orig_path, G, is_alternate=False)
        orig_days  = _calc_transit(orig_path, G)

        # Alternate path
        alt_path: List[str] = []
        status = "found"

        if src in G_safe and dst in G_safe:
            try:
                alt_path = nx.shortest_path(G_safe, src, dst, weight="weight")
            except nx.NetworkXNoPath:
                status = "no_path"
                alt_path = []

            # Avoid trivial same-path (can happen if direct edge exists in safe graph)
            if alt_path == orig_path:
                try:
                    # Try 2nd shortest path
                    paths = list(nx.shortest_simple_paths(G_safe, src, dst, weight="weight"))
                    if len(paths) > 1:
                        alt_path = paths[1]
                except Exception:
                    pass
        else:
            status = "blocked"

        alt_dist  = _calc_dist(alt_path, G_safe) if alt_path else 0.0
        alt_cost  = _calc_cost(alt_path, G_safe, is_alternate=True) if alt_path else 0.0
        alt_days  = _calc_transit(alt_path, G_safe) if alt_path else 0.0

        dist_delta  = round(alt_dist - orig_dist, 1)
        detour_pct  = round((dist_delta / orig_dist) * 100, 1) if orig_dist > 0 else 0.0
        cost_delta  = round(alt_cost - orig_cost, 2)
        delay_delta = round(alt_days - orig_days, 1)
        tariff_pct  = _max_tariff(alt_path, G_safe) if alt_path else 0.0

        safety, safety_note = ("n/a", "No alternate path.") if not alt_path else \
            _check_route_safety(alt_path, cascade_set, G)

        # Composite score: lower = better
        if status == "found" and alt_path:
            score = (
                detour_pct * 0.4 +
                (cost_delta / 10_000) * 0.3 +
                tariff_pct * 0.2 +
                delay_delta * 0.1
            )
            if safety == "hidden_dependency":
                score += 50
            elif safety == "partial_exposure":
                score += 15
        else:
            score = 9999.0

        results.append(RouteAlternative(
            source=src,
            destination=dst,
            source_label=_node_label(src, G),
            dest_label=_node_label(dst, G),
            status=status,
            original_path=orig_path,
            alternate_path=alt_path,
            original_dist_km=orig_dist,
            alternate_dist_km=alt_dist,
            distance_delta_km=dist_delta,
            detour_pct=detour_pct,
            original_cost_usd=orig_cost,
            alternate_cost_usd=alt_cost,
            cost_delta_usd=cost_delta,
            original_transit_days=orig_days,
            alternate_transit_days=alt_days,
            delay_delta_days=delay_delta,
            max_tariff_pct=tariff_pct,
            route_safety=safety,
            safety_note=safety_note,
            score=score,
            alt_path_labels=[_node_label(n, G) for n in alt_path],
        ))

    # Deduplicate (keep best per pair)
    seen: Dict[tuple, RouteAlternative] = {}
    for r in results:
        key = (r.source, r.destination)
        if key not in seen or r.score < seen[key].score:
            seen[key] = r

    ranked = sorted(seen.values(), key=lambda r: r.score)
    return ranked[:top_k]
