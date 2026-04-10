"""
simulate_shipments.py
----------------------
Generates ~500 realistic active shipments across the supply chain network.
Outputs: data/active_shipments.json
"""

import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

CARRIERS = [
    "Maersk", "MSC", "CMA CGM", "COSCO", "Evergreen",
    "Hapag-Lloyd", "ONE", "Yang Ming", "FedEx Freight",
    "DHL Supply Chain", "UPS Supply Chain", "DB Schenker",
    "Kuehne+Nagel", "Nippon Yusen", "OOCL",
]

CARGO_TYPES = [
    "Electronics", "Automotive Parts", "Pharmaceuticals", "Textiles",
    "Semiconductors", "Food & Agriculture", "Industrial Machinery",
    "Chemicals", "Consumer Goods", "Luxury Goods", "Raw Materials",
    "Oil & Gas Equipment", "Aerospace Components",
]

STATUSES = ["on_time", "on_time", "on_time", "at_risk", "delayed", "rerouted"]


def load_routes():
    path = DATA_DIR / "routes.csv"
    if not path.exists():
        raise FileNotFoundError("Run generate_world_network.py first")
    return pd.read_csv(path)


def generate_shipments(n: int = 500) -> list:
    routes_df = load_routes()
    # Only use direct sea/air lanes (not road/rail for shipment simulation)
    valid = routes_df[routes_df["transport_mode"].isin(["sea", "air"])].to_dict("records")

    if not valid:
        valid = routes_df.to_dict("records")

    shipments = []
    now = datetime.utcnow()

    for _ in range(n):
        lane = random.choice(valid)
        status = random.choice(STATUSES)
        cargo = random.choice(CARGO_TYPES)
        carrier = random.choice(CARRIERS)
        transit_days = float(lane.get("transit_days", 10))

        # Departure was 0-transit_days ago
        depart_offset = random.uniform(0, transit_days)
        departed_at = now - timedelta(days=depart_offset)
        eta = departed_at + timedelta(days=transit_days)

        # At risk / delayed: push ETA
        delay_days = 0
        if status == "at_risk":
            delay_days = random.uniform(0.5, 2.0)
            eta += timedelta(days=delay_days)
        elif status == "delayed":
            delay_days = random.uniform(2.0, 7.0)
            eta += timedelta(days=delay_days)

        value_usd = random.randint(50_000, 5_000_000)
        weight_tons = random.randint(5, 500)

        shipment = {
            "shipment_id": f"SHP-{uuid.uuid4().hex[:8].upper()}",
            "origin": lane["source"],
            "origin_city": lane["source_city"],
            "origin_country": lane["source_country"],
            "destination": lane["destination"],
            "destination_city": lane["dest_city"],
            "destination_country": lane["dest_country"],
            "carrier": carrier,
            "cargo_type": cargo,
            "value_usd": value_usd,
            "weight_tons": weight_tons,
            "transport_mode": lane["transport_mode"],
            "status": status,
            "delay_days": round(delay_days, 1),
            "departed_at": departed_at.isoformat() + "Z",
            "eta": eta.isoformat() + "Z",
            "distance_km": lane["distance_km"],
            "tariff_rate": lane["tariff_rate"],
            "progress_pct": round(min(99, (depart_offset / transit_days) * 100), 1),
            "disruption_cause": (
                random.choice([
                    "Port congestion", "Weather delay", "Customs hold",
                    "Equipment failure", "Documentation issue",
                ]) if status in ("delayed", "at_risk") else None
            ),
            "reroute_reason": (
                random.choice([
                    "Disruption at primary route", "Tariff avoidance",
                    "Weather routing", "Capacity optimization",
                ]) if status == "rerouted" else None
            ),
        }
        shipments.append(shipment)

    # Sort: delayed first for better demo visibility
    priority = {"delayed": 0, "at_risk": 1, "rerouted": 2, "on_time": 3}
    shipments.sort(key=lambda s: priority.get(s["status"], 3))

    return shipments


def main():
    print("📦 Simulating active shipments...")
    shipments = generate_shipments(500)

    with open(DATA_DIR / "active_shipments.json", "w") as f:
        json.dump(shipments, f, indent=2)

    counts = {}
    for s in shipments:
        counts[s["status"]] = counts.get(s["status"], 0) + 1

    print(f"  ✅ {len(shipments)} shipments generated → data/active_shipments.json")
    for status, count in counts.items():
        print(f"     {status}: {count}")


if __name__ == "__main__":
    main()
