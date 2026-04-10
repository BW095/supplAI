"""
train_models.py
----------------
Trains ML models on realistic supply chain data.
Outputs:
  models/delay_model.pkl     — XGBoost delay predictor
  models/anomaly_model.pkl   — Isolation Forest anomaly detector
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

MODELS_DIR = ROOT / "models"
DATA_DIR   = ROOT / "data"
MODELS_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────
# Domain knowledge maps
# ──────────────────────────────────────────────

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

FEATURE_COLS = [
    "distance_km", "tariff_rate", "transit_days", "carrier_tier",
    "cargo_risk", "mode_risk", "country_risk", "weather_severity",
    "port_congestion", "value_usd", "weight_tons",
]


# ──────────────────────────────────────────────
# Training data generator
# ──────────────────────────────────────────────

def build_training_data(n_samples: int = 25_000) -> pd.DataFrame:
    """
    Risk-bucket labeling approach:
    - Clearly separates HIGH / MEDIUM / LOW risk classes
    - Each feature has meaningful, learnable correlation with label
    - AUC target: ~0.82+
    """
    rng = np.random.default_rng(42)
    n = n_samples

    # ── Raw feature sampling ──────────────────────
    # carrier_tier: 1=Tier1 (5), 2=Tier2 (7), 3=Tier3 (3)
    carrier_tier = rng.choice([1, 1, 1, 1, 1,
                               2, 2, 2, 2, 2, 2, 2,
                               3, 3, 3], size=n).astype(float)

    # cargo_risk: continuous 0-1 (Electronics, Pharma etc)
    cargo_risk = rng.choice(
        list(CARGO_RISK.values()), size=n
    ).astype(float) + rng.normal(0, 0.02, n)
    cargo_risk = np.clip(cargo_risk, 0.1, 1.0)

    # mode: sea=0.5, air=0.2, road=0.35
    mode_risk = rng.choice([0.5, 0.5, 0.5, 0.2, 0.35], size=n).astype(float)

    # country_risk: HIGH risk countries use 1.0, safe use 0.0
    country_risk = rng.choice(
        [1.0] * 11 + [0.0] * 33,   # ~25% high risk
        size=n,
    ).astype(float)

    # weather: mostly calm, tail events
    weather_severity = rng.beta(2.0, 6.0, n)   # mean ~0.25, right tail

    # port_congestion: moderate-to-high
    port_congestion = rng.beta(2.5, 5.0, n)    # mean ~0.33

    # route features (from real data or synthetic)
    routes_path = DATA_DIR / "routes.csv"
    if routes_path.exists():
        routes_df = pd.read_csv(routes_path)
        samp = routes_df.sample(n=n, replace=True, random_state=42).reset_index(drop=True)
        distance_km  = samp["distance_km"].values.astype(float)
        tariff_rate  = samp["tariff_rate"].values.astype(float)
        transit_days = samp["transit_days"].values.astype(float)
    else:
        distance_km  = rng.uniform(500, 18_000, n)
        tariff_rate  = rng.choice([0.005, 0.01, 0.02, 0.085, 0.145], n)
        transit_days = distance_km / 800.0

    value_usd    = np.clip(rng.uniform(0.05, 5.0, n), 0.05, 5.0)   # millions
    weight_tons  = rng.uniform(0.01, 1.0, n)

    # ── Risk bucket assignment ──────────────────
    #
    # HIGH risk: at least 2 of these conditions
    c1 = carrier_tier == 3                          # Tier-3 carrier
    c2 = country_risk == 1.0                        # sanctioned corridor
    c3 = port_congestion > 0.60                     # severely congested
    c4 = weather_severity > 0.55                    # severe weather
    c5 = cargo_risk > 0.70                          # critical cargo
    c6 = tariff_rate > 0.08                         # punishing tariff

    high_risk_score = c1.astype(int) + c2.astype(int) + c3.astype(int) + \
                      c4.astype(int) + c5.astype(int) + c6.astype(int)

    high_risk   = high_risk_score >= 2              # ≥2 factors → HIGH
    low_risk    = (high_risk_score == 0) & (carrier_tier == 1) & (port_congestion < 0.40)
    medium_risk = ~high_risk & ~low_risk

    # ── Delay probability per bucket ─────────────
    delay_prob = np.where(
        high_risk,
        # HIGH: 65-90% delay, modulated by severity
        0.65 + 0.25 * (
            0.4 * port_congestion +
            0.3 * weather_severity +
            0.2 * (carrier_tier - 1) / 2.0 +
            0.1 * cargo_risk
        ),
        np.where(
            low_risk,
            # LOW: 5-20% delay
            0.05 + 0.15 * (
                0.5 * port_congestion +
                0.3 * weather_severity +
                0.2 * cargo_risk
            ),
            # MEDIUM: 28-58% delay
            0.28 + 0.30 * (
                0.30 * port_congestion +
                0.25 * (carrier_tier - 1) / 2.0 +
                0.20 * weather_severity +
                0.15 * cargo_risk +
                0.10 * country_risk
            )
        )
    )

    # Small noise
    delay_prob += rng.normal(0, 0.025, n)
    delay_prob  = np.clip(delay_prob, 0.03, 0.97)

    is_delayed = (rng.uniform(0, 1, n) < delay_prob).astype(int)

    return pd.DataFrame({
        "distance_km":      distance_km,
        "tariff_rate":      tariff_rate,
        "transit_days":     transit_days,
        "carrier_tier":     carrier_tier,
        "cargo_risk":       cargo_risk,
        "mode_risk":        mode_risk,
        "country_risk":     country_risk,
        "weather_severity": weather_severity,
        "port_congestion":  port_congestion,
        "value_usd":        value_usd,
        "weight_tons":      weight_tons,
        "is_delayed":       is_delayed,
        "delay_prob_true":  np.round(delay_prob, 4),
    })


# ──────────────────────────────────────────────
# Train delay model (XGBoost)
# ──────────────────────────────────────────────

def train_delay_model(df: pd.DataFrame):
    import joblib
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, classification_report

    X = df[FEATURE_COLS].values
    y = df["is_delayed"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.80,
        colsample_bytree=0.80,
        min_child_weight=5,
        gamma=0.05,
        reg_alpha=0.1,
        reg_lambda=1.5,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print(f"  Delay model AUC: {auc:.4f}")
    print(f"  {classification_report(y_test, y_pred, target_names=['On Time', 'Delayed'])}")

    # Feature importance
    importances = sorted(zip(FEATURE_COLS, model.feature_importances_), key=lambda x: -x[1])
    print("  Top features:")
    for feat, imp in importances[:5]:
        print(f"    {feat}: {imp:.4f}")

    model_path = MODELS_DIR / "delay_model.pkl"
    joblib.dump({"model": model, "features": FEATURE_COLS, "auc": round(auc, 4)}, model_path)
    print(f"  ✅ Delay model saved → models/delay_model.pkl")
    return auc


# ──────────────────────────────────────────────
# Train anomaly model (Isolation Forest)
# ──────────────────────────────────────────────

def train_anomaly_model(df: pd.DataFrame):
    import joblib
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    X = df[FEATURE_COLS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=0.10,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    n_anomalies = (model.predict(X_scaled) == -1).sum()
    print(f"  Anomaly model: {n_anomalies}/{len(X)} samples flagged ({n_anomalies/len(X)*100:.1f}%)")

    model_path = MODELS_DIR / "anomaly_model.pkl"
    joblib.dump({"model": model, "scaler": scaler, "features": FEATURE_COLS}, model_path)
    print(f"  ✅ Anomaly model saved → models/anomaly_model.pkl")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    print("🧠 Training SupplAI ML models…\n")

    print("📊 Generating training data…")
    df = build_training_data(n_samples=25_000)
    delay_rate = df["is_delayed"].mean() * 100
    print(f"  {len(df)} samples | delay rate: {delay_rate:.1f}%")

    # Quick correlation check
    print("  Feature → label correlations:")
    for col in FEATURE_COLS:
        corr = df[col].corr(df["is_delayed"])
        bar = "█" * int(abs(corr) * 40)
        print(f"    {col:20s}: {corr:+.3f}  {bar}")
    print()

    print("🤖 Training XGBoost delay predictor…")
    auc = train_delay_model(df)

    print("\n🔍 Training Isolation Forest anomaly detector…")
    train_anomaly_model(df)

    meta = {
        "delay_model_auc": round(auc, 4),
        "training_samples": len(df),
        "delay_rate": round(delay_rate, 1),
        "features": FEATURE_COLS,
        "trained_at": pd.Timestamp.now("UTC").isoformat(),
    }
    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ All models trained and saved.")
    print(f"   Delay model AUC: {auc:.4f}")


if __name__ == "__main__":
    main()
