"""
delay_predictor.py
-------------------
Live delay prediction wrapper around the trained XGBoost model.
Provides per-shipment delay probability and confidence scores.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

FEATURE_COLS = [
    "distance_km", "tariff_rate", "transit_days", "carrier_tier",
    "cargo_risk", "mode_risk", "country_risk", "weather_severity",
    "port_congestion", "value_usd", "weight_tons",
]

CARRIER_TIER = {
    "Maersk": 1, "MSC": 1, "CMA CGM": 1, "COSCO": 1, "Hapag-Lloyd": 1,
    "Evergreen": 2, "ONE": 2, "Yang Ming": 2, "OOCL": 2,
    "FedEx Freight": 2, "DHL Supply Chain": 2, "UPS Supply Chain": 2,
    "DB Schenker": 3, "Kuehne+Nagel": 3, "Nippon Yusen": 3,
}
CARGO_RISK = {
    "Electronics": 0.7, "Semiconductors": 0.8, "Pharmaceuticals": 0.75,
    "Automotive Parts": 0.6, "Food & Agriculture": 0.65, "Textiles": 0.4,
    "Industrial Machinery": 0.45, "Chemicals": 0.55, "Consumer Goods": 0.35,
    "Luxury Goods": 0.5, "Raw Materials": 0.3, "Oil & Gas Equipment": 0.5,
    "Aerospace Components": 0.85,
}
MODE_RISK = {"sea": 0.5, "air": 0.2, "road/rail": 0.35}
HIGH_RISK_COUNTRIES = {
    "Russia", "China", "Ukraine", "Iran", "Myanmar", "Belarus",
    "Venezuela", "North Korea", "Libya", "Yemen", "Syria",
}


# ─────────────────────────────────────────
# Singleton model loader
# ─────────────────────────────────────────

_model_cache: Optional[Dict] = None


def _load_model() -> Optional[Dict]:
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    path = MODELS_DIR / "delay_model.pkl"
    if not path.exists():
        return None
    try:
        import joblib
        _model_cache = joblib.load(path)
        return _model_cache
    except Exception as e:
        print(f"[delay_predictor] Model load failed: {e}")
        return None


# ─────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────

def _extract_features(shipment: Dict[str, Any]) -> np.ndarray:
    """Convert shipment dict → feature vector."""
    carrier = shipment.get("carrier", "Unknown")
    cargo   = shipment.get("cargo_type", "Consumer Goods")
    mode    = shipment.get("transport_mode", "sea")
    origin_country = shipment.get("origin_country", "")
    dest_country   = shipment.get("destination_country", "")

    carrier_tier = CARRIER_TIER.get(carrier, 2)
    cargo_risk   = CARGO_RISK.get(cargo, 0.45)
    mode_risk    = MODE_RISK.get(mode.split("/")[0], 0.5)
    country_risk = max(
        1.2 if origin_country in HIGH_RISK_COUNTRIES else 0.5,
        1.2 if dest_country   in HIGH_RISK_COUNTRIES else 0.5,
    )

    # Port congestion heuristic from current delay status
    status = shipment.get("status", "on_time")
    port_congestion = {"on_time": 0.2, "at_risk": 0.55, "delayed": 0.85, "rerouted": 0.6}.get(status, 0.3)

    # Weather severity — passive heuristic (0 without live API)
    weather_severity = 0.3 if status in ("delayed", "at_risk") else 0.1

    value_usd   = float(shipment.get("value_usd", 500_000)) / 1_000_000
    weight_tons = float(shipment.get("weight_tons", 100)) / 500
    distance_km = float(shipment.get("distance_km", 5000))
    tariff_rate = float(shipment.get("tariff_rate", 0.01))
    transit_days = float(shipment.get("transit_days", 10)) if "transit_days" in shipment else distance_km / 800

    return np.array([[
        distance_km, tariff_rate, transit_days, carrier_tier,
        cargo_risk, mode_risk, country_risk, weather_severity,
        port_congestion, value_usd, weight_tons,
    ]])


# ─────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────

def predict_delay_prob(shipment: Dict[str, Any]) -> Tuple[float, str]:
    """
    Predict delay probability for a single shipment.
    Returns: (probability 0-1, confidence label)
    """
    mdl = _load_model()
    if mdl is None:
        # Heuristic fallback
        status = shipment.get("status", "on_time")
        return {
            "on_time": 0.12, "at_risk": 0.52, "delayed": 0.88, "rerouted": 0.45,
        }.get(status, 0.25), "heuristic"

    try:
        X = _extract_features(shipment)
        model = mdl["model"]
        prob = float(model.predict_proba(X)[0][1])
        confidence = "high" if prob > 0.75 or prob < 0.25 else "medium"
        return round(prob, 4), confidence
    except Exception as e:
        print(f"[delay_predictor] Predict error: {e}")
        return 0.3, "error"


def predict_batch(shipments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Batch predict delay probability for a list of shipments.
    Returns list of dicts with: shipment_id, delay_prob, confidence, risk_tier
    """
    mdl = _load_model()
    results = []

    if mdl is None:
        for s in shipments:
            prob, conf = predict_delay_prob(s)
            results.append({
                "shipment_id": s.get("shipment_id", ""),
                "delay_prob": prob,
                "confidence": conf,
                "risk_tier": _risk_tier(prob),
            })
        return results

    try:
        X = np.vstack([_extract_features(s) for s in shipments])
        model = mdl["model"]
        probs = model.predict_proba(X)[:, 1]
        for s, prob in zip(shipments, probs):
            p = round(float(prob), 4)
            results.append({
                "shipment_id": s.get("shipment_id", ""),
                "delay_prob": p,
                "confidence": "high" if p > 0.75 or p < 0.25 else "medium",
                "risk_tier": _risk_tier(p),
            })
    except Exception as e:
        print(f"[delay_predictor] Batch predict error: {e}")
        for s in shipments:
            prob, conf = predict_delay_prob(s)
            results.append({
                "shipment_id": s.get("shipment_id", ""),
                "delay_prob": prob,
                "confidence": conf,
                "risk_tier": _risk_tier(prob),
            })
    return results


def _risk_tier(prob: float) -> str:
    if prob >= 0.70: return "High Risk"
    if prob >= 0.45: return "Moderate"
    if prob >= 0.25: return "Low Risk"
    return "On Track"


def get_model_info() -> Dict[str, Any]:
    """Return metadata about the loaded model."""
    meta_path = MODELS_DIR / "model_metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                return json.load(f)
        except Exception:
            pass
    mdl = _load_model()
    if mdl:
        return {"auc": mdl.get("auc", "unknown"), "features": mdl.get("features", FEATURE_COLS)}
    return {"status": "not_loaded"}
