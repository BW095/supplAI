"""
generate_world_network.py
--------------------------
Builds a realistic global supply chain network with 60 real logistics hubs.
Outputs:
  - data/supply_chain.csv  (nodes: city metadata + coordinates)
  - data/routes.csv        (edges: shipping lanes with distance, tariff, mode)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import math

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 60 Real global logistics hubs
# ─────────────────────────────────────────────────────────────
HUBS = [
    # Asia-Pacific
    {"city_id": "SHA", "city_name": "Shanghai",      "country": "China",        "lat": 31.23, "lon": 121.47, "tier": 1, "type": "port",    "product_category": "Electronics"},
    {"city_id": "SZX", "city_name": "Shenzhen",      "country": "China",        "lat": 22.54, "lon": 114.06, "tier": 1, "type": "port",    "product_category": "Electronics"},
    {"city_id": "GZH", "city_name": "Guangzhou",     "country": "China",        "lat": 23.13, "lon": 113.26, "tier": 2, "type": "city",    "product_category": "Manufacturing"},
    {"city_id": "TJN", "city_name": "Tianjin",       "country": "China",        "lat": 39.34, "lon": 117.36, "tier": 2, "type": "port",    "product_category": "Automotive"},
    {"city_id": "CHG", "city_name": "Chengdu",        "country": "China",        "lat": 30.57, "lon": 104.07, "tier": 3, "type": "city",    "product_category": "Technology"},
    {"city_id": "SGP", "city_name": "Singapore",     "country": "Singapore",    "lat":  1.35, "lon": 103.82, "tier": 1, "type": "port",    "product_category": "Semiconductors"},
    {"city_id": "HKG", "city_name": "Hong Kong",     "country": "China",        "lat": 22.32, "lon": 114.17, "tier": 1, "type": "port",    "product_category": "Finance/Logistics"},
    {"city_id": "TYO", "city_name": "Tokyo",          "country": "Japan",        "lat": 35.68, "lon": 139.69, "tier": 1, "type": "city",    "product_category": "Automotive"},
    {"city_id": "OSK", "city_name": "Osaka",          "country": "Japan",        "lat": 34.69, "lon": 135.50, "tier": 2, "type": "port",    "product_category": "Manufacturing"},
    {"city_id": "BUS", "city_name": "Busan",          "country": "South Korea",  "lat": 35.18, "lon": 129.07, "tier": 1, "type": "port",    "product_category": "Automotive"},
    {"city_id": "ICN", "city_name": "Incheon",        "country": "South Korea",  "lat": 37.46, "lon": 126.44, "tier": 2, "type": "airport", "product_category": "Semiconductors"},
    {"city_id": "MUM", "city_name": "Mumbai",         "country": "India",        "lat": 19.08, "lon": 72.88,  "tier": 2, "type": "port",    "product_category": "Pharmaceuticals"},
    {"city_id": "DEL", "city_name": "Delhi",          "country": "India",        "lat": 28.61, "lon": 77.21,  "tier": 2, "type": "city",    "product_category": "Textiles"},
    {"city_id": "CHE", "city_name": "Chennai",        "country": "India",        "lat": 13.08, "lon": 80.27,  "tier": 3, "type": "port",    "product_category": "Automotive"},
    {"city_id": "KUL", "city_name": "Kuala Lumpur",  "country": "Malaysia",     "lat":  3.14, "lon": 101.69, "tier": 2, "type": "city",    "product_category": "Electronics"},
    {"city_id": "BKK", "city_name": "Bangkok",        "country": "Thailand",     "lat": 13.75, "lon": 100.52, "tier": 2, "type": "port",    "product_category": "Food & Agriculture"},
    {"city_id": "JKT", "city_name": "Jakarta",        "country": "Indonesia",    "lat": -6.21, "lon": 106.85, "tier": 3, "type": "port",    "product_category": "Raw Materials"},
    {"city_id": "HCM", "city_name": "Ho Chi Minh",   "country": "Vietnam",      "lat": 10.82, "lon": 106.63, "tier": 2, "type": "port",    "product_category": "Textiles"},
    {"city_id": "TPE", "city_name": "Taipei",         "country": "Taiwan",       "lat": 25.04, "lon": 121.56, "tier": 1, "type": "city",    "product_category": "Semiconductors"},
    {"city_id": "SYD", "city_name": "Sydney",         "country": "Australia",    "lat":-33.87, "lon": 151.21, "tier": 3, "type": "port",    "product_category": "Mining/Resources"},

    # Europe
    {"city_id": "RTM", "city_name": "Rotterdam",      "country": "Netherlands",  "lat": 51.92, "lon":   4.48, "tier": 1, "type": "port",    "product_category": "Chemicals"},
    {"city_id": "HAM", "city_name": "Hamburg",         "country": "Germany",      "lat": 53.55, "lon":   9.99, "tier": 1, "type": "port",    "product_category": "Automotive"},
    {"city_id": "ANT", "city_name": "Antwerp",         "country": "Belgium",      "lat": 51.22, "lon":   4.40, "tier": 2, "type": "port",    "product_category": "Chemicals"},
    {"city_id": "FLX", "city_name": "Felixstowe",     "country": "UK",           "lat": 51.96, "lon":   1.35, "tier": 2, "type": "port",    "product_category": "Consumer Goods"},
    {"city_id": "LON", "city_name": "London",          "country": "UK",           "lat": 51.51, "lon":  -0.13, "tier": 1, "type": "city",    "product_category": "Finance/Logistics"},
    {"city_id": "PAR", "city_name": "Paris",           "country": "France",       "lat": 48.86, "lon":   2.35, "tier": 2, "type": "city",    "product_category": "Luxury Goods"},
    {"city_id": "FRA", "city_name": "Frankfurt",       "country": "Germany",      "lat": 50.11, "lon":   8.68, "tier": 2, "type": "airport", "product_category": "Pharmaceuticals"},
    {"city_id": "MIL", "city_name": "Milan",           "country": "Italy",        "lat": 45.47, "lon":   9.19, "tier": 2, "type": "city",    "product_category": "Luxury Goods"},
    {"city_id": "BCN", "city_name": "Barcelona",       "country": "Spain",        "lat": 41.39, "lon":   2.17, "tier": 3, "type": "port",    "product_category": "Food & Agriculture"},
    {"city_id": "PIR", "city_name": "Piraeus",         "country": "Greece",       "lat": 37.94, "lon":  23.65, "tier": 2, "type": "port",    "product_category": "Transshipment"},
    {"city_id": "IST", "city_name": "Istanbul",        "country": "Turkey",       "lat": 41.01, "lon":  28.95, "tier": 2, "type": "port",    "product_category": "Textiles"},
    {"city_id": "WAR", "city_name": "Warsaw",          "country": "Poland",       "lat": 52.23, "lon":  21.01, "tier": 3, "type": "city",    "product_category": "Manufacturing"},

    # Middle East
    {"city_id": "DXB", "city_name": "Dubai",           "country": "UAE",          "lat": 25.20, "lon":  55.27, "tier": 1, "type": "port",    "product_category": "Transshipment"},
    {"city_id": "ABU", "city_name": "Abu Dhabi",       "country": "UAE",          "lat": 24.47, "lon":  54.37, "tier": 2, "type": "city",    "product_category": "Oil & Gas"},
    {"city_id": "JED", "city_name": "Jeddah",          "country": "Saudi Arabia", "lat": 21.49, "lon":  39.19, "tier": 2, "type": "port",    "product_category": "Oil & Gas"},
    {"city_id": "SUZ", "city_name": "Suez",            "country": "Egypt",        "lat": 29.97, "lon":  32.55, "tier": 1, "type": "canal",   "product_category": "Transshipment"},

    # Africa
    {"city_id": "CPT", "city_name": "Cape Town",       "country": "South Africa", "lat":-33.93, "lon":  18.42, "tier": 3, "type": "port",    "product_category": "Mining/Resources"},
    {"city_id": "DJI", "city_name": "Djibouti",        "country": "Djibouti",     "lat": 11.59, "lon":  43.15, "tier": 2, "type": "port",    "product_category": "Transshipment"},
    {"city_id": "LGS", "city_name": "Lagos",           "country": "Nigeria",      "lat":  6.53, "lon":   3.38, "tier": 3, "type": "port",    "product_category": "Oil & Gas"},

    # Americas – North
    {"city_id": "LAX", "city_name": "Los Angeles",     "country": "USA",          "lat": 33.93, "lon":-118.41, "tier": 1, "type": "port",    "product_category": "Consumer Goods"},
    {"city_id": "NYC", "city_name": "New York",        "country": "USA",          "lat": 40.71, "lon": -74.01, "tier": 1, "type": "port",    "product_category": "Finance/Logistics"},
    {"city_id": "CHI", "city_name": "Chicago",         "country": "USA",          "lat": 41.88, "lon": -87.63, "tier": 2, "type": "city",    "product_category": "Manufacturing"},
    {"city_id": "HOU", "city_name": "Houston",         "country": "USA",          "lat": 29.76, "lon": -95.37, "tier": 2, "type": "port",    "product_category": "Oil & Gas"},
    {"city_id": "MIA", "city_name": "Miami",           "country": "USA",          "lat": 25.76, "lon": -80.19, "tier": 2, "type": "port",    "product_category": "Food & Agriculture"},
    {"city_id": "SEA", "city_name": "Seattle",         "country": "USA",          "lat": 47.61, "lon":-122.33, "tier": 2, "type": "port",    "product_category": "Aerospace"},
    {"city_id": "YVR", "city_name": "Vancouver",       "country": "Canada",       "lat": 49.25, "lon":-123.12, "tier": 2, "type": "port",    "product_category": "Raw Materials"},
    {"city_id": "MEX", "city_name": "Mexico City",     "country": "Mexico",       "lat": 19.43, "lon": -99.13, "tier": 3, "type": "city",    "product_category": "Automotive"},
    {"city_id": "PAN", "city_name": "Panama City",     "country": "Panama",       "lat":  8.99, "lon": -79.52, "tier": 1, "type": "canal",   "product_category": "Transshipment"},

    # Americas – South
    {"city_id": "SAO", "city_name": "São Paulo",       "country": "Brazil",       "lat":-23.55, "lon": -46.63, "tier": 2, "type": "city",    "product_category": "Food & Agriculture"},
    {"city_id": "SSZ", "city_name": "Santos",          "country": "Brazil",       "lat":-23.95, "lon": -46.33, "tier": 2, "type": "port",    "product_category": "Food & Agriculture"},
    {"city_id": "BOG", "city_name": "Bogotá",          "country": "Colombia",     "lat":  4.71, "lon": -74.07, "tier": 3, "type": "city",    "product_category": "Food & Agriculture"},
    {"city_id": "BUE", "city_name": "Buenos Aires",    "country": "Argentina",    "lat":-34.60, "lon": -58.44, "tier": 3, "type": "port",    "product_category": "Food & Agriculture"},

    # Central Asia / Russia
    {"city_id": "MSC", "city_name": "Moscow",          "country": "Russia",       "lat": 55.75, "lon":  37.62, "tier": 3, "type": "city",    "product_category": "Raw Materials"},
    {"city_id": "NVS", "city_name": "Novosibirsk",     "country": "Russia",       "lat": 54.99, "lon":  82.90, "tier": 4, "type": "city",    "product_category": "Raw Materials"},
    {"city_id": "ALM", "city_name": "Almaty",          "country": "Kazakhstan",   "lat": 43.25, "lon":  76.94, "tier": 4, "type": "city",    "product_category": "Raw Materials"},
    {"city_id": "KHI", "city_name": "Karachi",         "country": "Pakistan",     "lat": 24.86, "lon":  67.01, "tier": 3, "type": "port",    "product_category": "Textiles"},
    {"city_id": "COL", "city_name": "Colombo",         "country": "Sri Lanka",    "lat":  6.93, "lon":  79.85, "tier": 3, "type": "port",    "product_category": "Transshipment"},
    {"city_id": "DOH", "city_name": "Doha",            "country": "Qatar",        "lat": 25.29, "lon":  51.53, "tier": 2, "type": "airport", "product_category": "Oil & Gas"},
]

# ─────────────────────────────────────────────────────────────
# Tariff rates 2025 (approximate) — origin_country → tariff%
# ─────────────────────────────────────────────────────────────
TARIFF_RATES = {
    ("China", "USA"):          0.145,  # 145% US tariffs on China
    ("USA", "China"):          0.125,
    ("China", "EU"):           0.035,
    ("EU", "China"):           0.035,
    ("China", "India"):        0.018,
    ("India", "USA"):          0.026,
    ("Russia", "EU"):          0.30,   # sanctions-related
    ("Russia", "USA"):         0.35,
    ("Taiwan", "China"):       0.075,  # cross-strait tension
    ("USA", "EU"):             0.015,
    ("EU", "USA"):             0.010,
    ("USA", "Mexico"):         0.025,  # USMCA
    ("Mexico", "USA"):         0.025,
}

def get_tariff(c1: str, c2: str) -> float:
    return TARIFF_RATES.get((c1, c2), TARIFF_RATES.get((c2, c1), 0.005))


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    d = math.radians
    dlat = d(lat2 - lat1)
    dlon = d(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(d(lat1)) * math.cos(d(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ─────────────────────────────────────────────────────────────
# Shipping lane connectivity rules
# ─────────────────────────────────────────────────────────────
# Major global trade corridors: manually define key lane pairs
CORRIDORS = [
    # Trans-Pacific
    ("SHA", "LAX"), ("SHA", "SEA"),  ("SHA", "YVR"),
    ("SZX", "LAX"), ("HKG", "LAX"), ("HKG", "NYC"),
    ("SGP", "LAX"), ("SGP", "RTM"), ("TPE", "LAX"),
    ("BUS", "LAX"), ("TYO", "LAX"), ("TYO", "SEA"),

    # Asia - Europe (via Suez)
    ("SHA", "RTM"), ("SHA", "HAM"), ("SGP", "RTM"),
    ("SGP", "PIR"), ("MUM", "RTM"), ("MUM", "HAM"),
    ("HKG", "RTM"), ("HKG", "ANT"), ("KHI", "RTM"),
    ("SUZ", "RTM"), ("SUZ", "PIR"), ("DJI", "SUZ"),
    ("DXB", "RTM"), ("DXB", "SHA"), ("DXB", "LAX"),

    # Asia internal
    ("SHA", "SGP"), ("SHA", "HKG"), ("SHA", "BUS"),
    ("SHA", "TYO"), ("SHA", "TPE"), ("SGP", "HKG"),
    ("SGP", "JKT"), ("SGP", "BKK"), ("SGP", "KUL"),
    ("SGP", "COL"), ("SGP", "MUM"), ("SGP", "CHE"),
    ("HKG", "SGP"), ("HKG", "BUS"), ("TPE", "HKG"),
    ("TPE", "SGP"), ("HCM", "SGP"), ("HCM", "SHA"),
    ("KUL", "SHA"), ("BKK", "SHA"), ("JKT", "SGP"),
    ("ICN", "SHA"), ("ICN", "SGP"), ("BUS", "SHA"),
    ("BUS", "SGP"), ("OSK", "SHA"), ("SZX", "SGP"),
    ("GZH", "SGP"), ("GZH", "SHA"), ("TJN", "SHA"),
    ("CHG", "SHA"), ("CHG", "GZH"),
    ("MUM", "DXB"), ("DEL", "MUM"), ("CHE", "MUM"),
    ("DOH", "DXB"), ("ABU", "DXB"), ("JED", "DXB"),
    ("KHI", "MUM"), ("KHI", "DXB"), ("COL", "MUM"),
    ("SYD", "SGP"), ("SYD", "SHA"),

    # Europe internal
    ("RTM", "HAM"), ("RTM", "ANT"), ("RTM", "FLX"),
    ("HAM", "FRA"), ("ANT", "PAR"), ("FLX", "LON"),
    ("LON", "PAR"), ("LON", "FRA"), ("PAR", "MIL"),
    ("FRA", "MIL"), ("FRA", "WAR"), ("MIL", "BCN"),
    ("BCN", "PIR"), ("PIR", "IST"), ("IST", "DXB"),
    ("IST", "SHA"),

    # Americas internal
    ("LAX", "NYC"), ("LAX", "CHI"), ("LAX", "HOU"),
    ("LAX", "SEA"), ("LAX", "MIA"), ("LAX", "MEX"),
    ("NYC", "MIA"), ("NYC", "CHI"), ("NYC", "LON"),
    ("NYC", "RTM"), ("SEA", "YVR"), ("HOU", "MIA"),
    ("MEX", "CHI"), ("MEX", "MIA"), ("PAN", "MIA"),
    ("PAN", "LAX"), ("PAN", "SAO"), ("SAO", "SSZ"),
    ("SSZ", "RTM"), ("SSZ", "LAX"), ("BOG", "PAN"),
    ("BUE", "SSZ"), ("YVR", "LAX"),

    # Africa
    ("CPT", "RTM"), ("CPT", "LAX"), ("CPT", "DXB"),
    ("DJI", "MUM"), ("DJI", "SGP"), ("DJI", "DXB"),
    ("LGS", "RTM"), ("LGS", "DXB"),

    # Trans-Siberian / Central Asia rail
    ("MSC", "WAR"), ("MSC", "SHA"), ("NVS", "MSC"),
    ("NVS", "SHA"), ("ALM", "SHA"), ("ALM", "MSC"),
    ("ALM", "DXB"),
]


def build_route_df(hubs: list, corridors: list) -> pd.DataFrame:
    hub_map = {h["city_id"]: h for h in hubs}
    rows = []
    seen = set()

    for src_id, dst_id in corridors:
        if src_id not in hub_map or dst_id not in hub_map:
            continue
        key = tuple(sorted([src_id, dst_id]))
        if key in seen:
            continue
        seen.add(key)

        src = hub_map[src_id]
        dst = hub_map[dst_id]
        dist = haversine_km(src["lat"], src["lon"], dst["lat"], dst["lon"])
        tariff = get_tariff(src["country"], dst["country"])

        # Determine transport mode
        if dist > 5000:
            mode = "sea"
        elif src["type"] == "airport" or dst["type"] == "airport":
            mode = "air"
        elif src["type"] == "canal" or dst["type"] == "canal":
            mode = "sea"
        else:
            mode = "sea" if dist > 800 else "road/rail"

        # Transit time days (rough)
        speed_km_per_day = {"sea": 800, "air": 8000, "road/rail": 600}
        transit_days = max(1, round(dist / speed_km_per_day.get(mode, 800), 1))

        rows.append({
            "source": src_id,
            "destination": dst_id,
            "source_city": src["city_name"],
            "dest_city": dst["city_name"],
            "source_country": src["country"],
            "dest_country": dst["country"],
            "distance_km": round(dist, 1),
            "tariff_rate": tariff,
            "transport_mode": mode,
            "transit_days": transit_days,
            "weight": round(dist / 1000 * (1 + tariff * 2), 3),  # lower = preferred
        })

        # Add reverse direction
        if src_id != dst_id:
            rows.append({
                "source": dst_id,
                "destination": src_id,
                "source_city": dst["city_name"],
                "dest_city": src["city_name"],
                "source_country": dst["country"],
                "dest_country": src["country"],
                "distance_km": round(dist, 1),
                "tariff_rate": get_tariff(dst["country"], src["country"]),
                "transport_mode": mode,
                "transit_days": transit_days,
                "weight": round(dist / 1000 * (1 + get_tariff(dst["country"], src["country"]) * 2), 3),
            })

    return pd.DataFrame(rows)


def main():
    print("🌏 Generating global supply chain network...")

    # Save nodes
    nodes_df = pd.DataFrame(HUBS)
    nodes_df.to_csv(DATA_DIR / "supply_chain.csv", index=False)
    print(f"  ✅ {len(nodes_df)} logistics hubs saved → data/supply_chain.csv")

    # Save edges/routes
    routes_df = build_route_df(HUBS, CORRIDORS)
    routes_df.to_csv(DATA_DIR / "routes.csv", index=False)
    print(f"  ✅ {len(routes_df)} shipping lanes saved → data/routes.csv")

    # Save hub_ids list for quick reference
    hub_ids = [h["city_id"] for h in HUBS]
    with open(DATA_DIR / "hub_ids.json", "w") as f:
        json.dump(hub_ids, f)

    print(f"\n📊 Network summary:")
    print(f"  Nodes  : {len(nodes_df)}")
    print(f"  Edges  : {len(routes_df)}")
    print(f"  Countries: {nodes_df['country'].nunique()}")
    print(f"  Avg distance: {routes_df['distance_km'].mean():.0f} km")
    print(f"\nDone! ✅")


if __name__ == "__main__":
    main()
