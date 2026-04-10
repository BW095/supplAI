#!/bin/bash
# SupplAI — Start watchtower daemon + Streamlit dashboard

set -e

echo "🚀 Starting SupplAI Watchtower..."

# Ensure state dir exists
mkdir -p state data models

# Generate data if not already present
if [ ! -f data/supply_chain.csv ]; then
    echo "📊 Generating world supply chain network..."
    python data/generate_world_network.py
fi

if [ ! -f data/active_shipments.json ]; then
    echo "📦 Simulating active shipments..."
    python data/simulate_shipments.py
fi

if [ ! -f models/delay_model.pkl ]; then
    echo "🧠 Training ML models..."
    python models/train_models.py
fi

# Start Streamlit (daemon is launched from within the dashboard)
echo "🖥️  Launching Control Tower dashboard..."
exec streamlit run dashboard/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
