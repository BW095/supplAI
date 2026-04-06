# 🔗 SupplAI — AI Supply Chain Disruption Monitor

> **Hackathon Project** | AI-powered supply chain risk simulation, ML explainability (SHAP), and operations brief generation

---

## 🚀 Quick Start

```bash
# 1. Activate environment
conda activate condaVE

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Gemini API key
echo "AIzaSyD_MFUolJqM1NNEDYwaizMX7IQjzAzH23U" > .env

# 4. Launch the dashboard
streamlit run app.py
```

> 💡 The delay prediction model trains automatically on first launch (CUDA GPU accelerated). Cached for all future runs.

---

## 🎯 Demo Scenario

Use the **"🏭 China Electronics Shutdown"** button in the sidebar to instantly run:

> *"Factory shutdown in China affecting electronics supply chain"*

This demonstrates all 4 dashboard tabs in one click.

---

## 📁 Project Structure

```
supplAI/
├── app.py                      ← Streamlit dashboard (entry point)
├── requirements.txt
├── .env                        ← GEMINI_API_KEY here
├── data/
│   └── supply_chain.csv        ← 70-city node metadata
├── datasets/                   ← Your existing datasets
│   ├── order_large.csv         ← Supply chain edges (routes)
│   ├── distance.csv            ← Edge weights (distances)
│   └── Is_delayed_*.csv        ← Delay prediction training data
├── models/                     ← Auto-created; stores trained model
│   └── delay_model.pkl
└── src/
    ├── disruption_input.py     ← Keyword-based event parser
    ├── graph_builder.py        ← NetworkX graph from CSVs
    ├── delay_model.py          ← XGBoost CUDA delay predictor
    ├── shap_explain.py         ← SHAP ML explainability layer  ← NEW
    ├── cascade_model.py        ← BFS cascade simulation
    ├── risk_scoring.py         ← Composite risk scoring
    ├── reroute.py              ← Dijkstra alternate paths
    └── llm_brief.py            ← Gemini AI operations brief (SHAP-enhanced)
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Dashboard | Streamlit |
| Graph engine | NetworkX |
| ML model | XGBoost (CUDA GPU) |
| ML Explainability | SHAP (TreeExplainer) |
| Visualisation | Plotly |
| AI brief | Google Gemini 1.5 Flash |
| Data | Pandas + real logistics datasets |

---

## 📊 How It Works

```
1. User types disruption event
        ↓
2. Keyword parser → identifies affected city nodes
        ↓
3. BFS cascade → propagates through supply chain graph
        ↓
4. Risk scoring → depth + centrality + ML delay probability
        ↓
5. SHAP explainability → per-feature breakdown of WHY each node is at risk
        ↓
6. Dijkstra rerouting → finds safe alternate paths
        ↓
7. Gemini API → generates AI brief (citing SHAP feature drivers)
        ↓
8. Streamlit dashboard → 5 tabs: Network · Risk · Rerouting · AI Brief · ML Explainability
```

---

## 🔑 API Key

Get a free Gemini API key at: https://aistudio.google.com/app/apikey

The dashboard works without an API key (uses template brief as fallback).
