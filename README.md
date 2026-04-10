# 🛰️ SupplAI — Autonomous Supply Chain Watchtower

An AI-powered, autonomous supply chain monitoring system with a real-world 60-city global logistics network, Gemini 2.5 Flash reasoning, and a live Control Tower dashboard.

---

## 🚀 Quick Start (Local)

```bash
# 1. Activate conda environment
conda activate condaVE

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API key in .env
cp .env.example .env
# then edit .env with your GEMINI_API_KEY (and optionally GROQ_API_KEY, OPEN_WEATHER_API_KEY)

# 4. Generate dataset + train models (once)
python data/generate_world_network.py
python data/simulate_shipments.py
python models/train_models.py

# 5. Launch the Control Tower
streamlit run dashboard/app.py --server.port 8501
```

Open → **http://localhost:8501**

---

## 🐳 Docker

```bash
# Copy env file
cp .env.example .env   # fill in your API keys

# Build + run
docker-compose up --build
```

Open → **http://localhost:8501**

---

## 🏗️ Architecture

```
supplAI/
├── data/
│   ├── generate_world_network.py   # 60-city real logistics graph
│   ├── simulate_shipments.py       # 500 realistic active shipments
│   ├── supply_chain.csv            # Node metadata
│   ├── routes.csv                  # Shipping lane edges
│   └── active_shipments.json       # Live shipment state
│
├── src/
│   ├── graph_engine.py             # NetworkX supply chain graph
│   ├── disruption_engine.py        # Cascade propagation engine
│   ├── route_optimizer.py          # Alternate route finder
│   ├── risk_engine.py              # Risk scoring + anomaly detection
│   ├── intelligence_feeds.py       # News / weather / earthquake feeds
│   ├── gemini_agent.py             # AI agent (Gemini → Groq → deterministic)
│   └── notification_engine.py      # Supplier notification generator
│
├── daemon/
│   ├── watchtower.py               # Background monitoring daemon
│   └── state_manager.py            # Shared state (JSON files)
│
├── dashboard/
│   └── app.py                      # Streamlit Control Tower (5 tabs)
│
├── models/
│   ├── train_models.py             # XGBoost + Isolation Forest training
│   ├── delay_model.pkl             # Delay prediction model
│   └── anomaly_model.pkl           # Anomaly detection model
│
├── state/                          # Runtime state (auto-generated)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── run.sh
```

---

## 🌐 The Global Network

- **58 real logistics hubs** — Shanghai, Rotterdam, Dubai, Los Angeles, Singapore, and more
- **240 shipping lanes** across Trans-Pacific, Asia-Europe (Suez), Trans-Atlantic routes
- **39 countries** with real 2025 tariff rates (US-China 145%, EU-Russia sanctions, etc.)
- **500 simulated active shipments** with realistic delay patterns

---

## 🤖 AI Agent Architecture

```
External Signal
    │
    ▼
Intelligence Feeds (news RSS + OpenWeather + USGS earthquakes)
    │
    ▼
Disruption Engine (BFS cascade propagation across graph)
    │
    ▼
Risk Engine (betweenness centrality + Isolation Forest anomaly score)
    │
    ▼
Route Optimizer (Dijkstra on safe subgraph + upstream dependency check)
    │
    ▼
Gemini Agent (function-calling loop: assess → score → approve → flag → finalize)
    │ (fallback: Groq → deterministic)
    ▼
Notification Engine (route change + delay alert + emergency procurement)
    │
    ▼
State Files → Dashboard (15s auto-refresh)
```

---

## 🎯 Simulation Console

8 pre-built disruption scenarios for live demos:

| Scenario | Severity | Key Nodes |
|----------|----------|-----------|
| 🌪️ Port of Shanghai Closure | Critical | SHA, SZX, GZH |
| ⚓ Suez Canal Blockage | Critical | SUZ |
| 📈 US-China Tariff Escalation | High | SHA, LAX, NYC |
| 🌊 Rotterdam Flood Damage | High | RTM |
| ✈️ European Air Cargo Strike | High | FRA, HAM, LON |
| 🔥 Singapore Port Fire | High | SGP |
| ⚡ Taiwan Strait Tension | Critical | TPE, HKG |
| 🚢 Panama Canal Drought | Medium | PAN |

---

## 🔑 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes (primary) | Google Gemini 2.5 Flash |
| `GROQ_API_KEY` | No (fallback) | Groq LLaMA 70B fallback |
| `OPEN_WEATHER_API_KEY` | No (optional) | Live weather data |
| `SCAN_INTERVAL_MINUTES` | No | Default: 5 |

---

## 📊 ML Models

| Model | Algorithm | Purpose |
|-------|-----------|---------|
| `delay_model.pkl` | XGBoost Classifier | Predict shipment delay probability |
| `anomaly_model.pkl` | Isolation Forest | Detect anomalous supply chain nodes |

Retrain anytime: `python models/train_models.py`

---

## ☁️ Google Cloud Deployment

```bash
# Build & push to Artifact Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT/supplai .

# Deploy to Cloud Run
gcloud run deploy supplai \
  --image gcr.io/YOUR_PROJECT/supplai \
  --platform managed \
  --port 8501 \
  --memory 2Gi \
  --set-env-vars GEMINI_API_KEY=your_key
```
