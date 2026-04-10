"""
app.py — SupplAI Control Tower Dashboard
==========================================
5-Tab Streamlit dashboard:
  🗺️ Live World Map | 📡 Watchtower Feed | 🤖 AI Decisions | 📦 Shipments | 🎯 Simulation
"""

import sys
import json
import time
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "daemon"))
sys.path.insert(0, str(ROOT / "dashboard"))

# ─────────────────────────────────────────────────────────────
# Page config — MUST be first Streamlit call
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SupplAI — Autonomous Supply Chain Watchtower",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CSS: premium dark theme
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── Animations ── */
@keyframes pulse-glow  { 0%,100%{box-shadow:0 0 8px rgba(99,102,241,.3)} 50%{box-shadow:0 0 28px rgba(99,102,241,.7),0 0 48px rgba(239,68,68,.15)} }
@keyframes shimmer     { 0%{background-position:-200% center} 100%{background-position:200% center} }
@keyframes float       { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-6px)} }
@keyframes ping        { 75%,100%{transform:scale(2.2);opacity:0} }
@keyframes fadeInUp    { from{opacity:0;transform:translateY(18px)} to{opacity:1;transform:translateY(0)} }
@keyframes scanLine    { 0%{top:0} 100%{top:100%} }
@keyframes borderPulse { 0%,100%{border-color:rgba(99,102,241,.35)} 50%{border-color:rgba(239,68,68,.55)} }

/* ── Global ── */
html,body,[data-testid="stAppViewContainer"],[data-testid="stApp"],.stApp,.main,
[data-testid="stAppViewBlockContainer"],[data-testid="block-container"],.block-container,
section[data-testid="stMain"],div[data-testid="stMainBlockContainer"] {
    background: #03040d !important; color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}
.stMarkdown p,.stMarkdown div,.element-container p,.element-container div { color:#e2e8f0; }

/* ── Sidebar ── */
[data-testid="stSidebar"],[data-testid="stSidebar"]>div {
    background: linear-gradient(180deg,#06091a 0%,#0c1230 50%,#0f162e 100%) !important;
    border-right: 1px solid rgba(99,102,241,.2) !important;
}
[data-testid="stSidebar"] label,[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,[data-testid="stSidebar"] .stMarkdown p { color:#cbd5e1 !important; }
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 { color:#e2e8f0 !important; }
[data-testid="stSidebar"] .stTextArea label,[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label {
    color:#94a3b8 !important; font-size:.8rem !important; font-weight:600 !important;
    text-transform:uppercase; letter-spacing:.07em;
}

/* ── Tabs ── */
[data-baseweb="tab-list"] {
    background:rgba(15,18,40,.9) !important; border-radius:12px; padding:5px;
    border:1px solid rgba(99,102,241,.18); gap:4px; backdrop-filter:blur(10px);
}
[data-baseweb="tab"] {
    border-radius:8px; font-weight:500; color:#475569 !important;
    padding:10px 20px; font-size:.88rem; transition:all .2s ease; font-family:'Inter',sans-serif;
}
[data-baseweb="tab"]:hover { color:#cbd5e1 !important; background:rgba(99,102,241,.1) !important; }
[aria-selected="true"] {
    background:linear-gradient(135deg,#4f46e5,#6366f1) !important; color:white !important;
    box-shadow:0 2px 14px rgba(79,70,229,.5) !important; font-weight:600 !important;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
    background:linear-gradient(135deg,rgba(30,35,66,.9),rgba(12,14,35,.95)) !important;
    border:1px solid rgba(99,102,241,.22) !important; border-radius:14px !important;
    padding:1.1rem 1.2rem !important;
    box-shadow:0 4px 24px rgba(0,0,0,.55),inset 0 1px 0 rgba(255,255,255,.04) !important;
    backdrop-filter:blur(12px); animation:fadeInUp .4s ease both;
}
[data-testid="metric-container"]:hover {
    border-color:rgba(99,102,241,.5) !important;
    box-shadow:0 6px 32px rgba(99,102,241,.2) !important; transform:translateY(-2px) !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color:#f1f5f9 !important; font-size:1.6rem !important; font-weight:800 !important;
    font-family:'Space Grotesk',sans-serif !important;
}
[data-testid="metric-container"] label { color:#64748b !important; font-size:.82rem !important; font-weight:500 !important; }
[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size:.82rem !important; }

/* ── Buttons ── */
.stButton>button {
    background:linear-gradient(135deg,#4f46e5,#6366f1) !important; color:white !important;
    border:none !important; border-radius:9px !important; font-weight:600 !important;
    font-size:.87rem !important; padding:.55rem 1.5rem !important;
    transition:all .22s cubic-bezier(.4,0,.2,1); width:100%; letter-spacing:.02em;
    box-shadow:0 2px 10px rgba(79,70,229,.35);
}
.stButton>button:hover {
    background:linear-gradient(135deg,#3730a3,#4f46e5) !important;
    box-shadow:0 4px 22px rgba(79,70,229,.55); transform:translateY(-1px);
}
.stButton>button:active { transform:translateY(0); }

/* ── Danger button variation ── */
.danger .stButton>button {
    background:linear-gradient(135deg,#dc2626,#ef4444) !important;
    box-shadow:0 2px 10px rgba(239,68,68,.35) !important;
}
.danger .stButton>button:hover { background:linear-gradient(135deg,#b91c1c,#dc2626) !important; }

/* ── Inputs ── */
.stTextArea textarea {
    background:rgba(30,35,66,.75) !important; border:1px solid rgba(71,85,105,.8) !important;
    border-radius:10px !important; color:#e2e8f0 !important; font-family:'Inter',sans-serif !important;
}
.stTextArea textarea:focus { border-color:#6366f1 !important; box-shadow:0 0 0 3px rgba(99,102,241,.18) !important; }
[data-baseweb="select"]>div {
    background:rgba(30,35,66,.8) !important; border:1px solid rgba(71,85,105,.8) !important;
    border-radius:9px !important; color:#e2e8f0 !important;
}
[data-baseweb="select"] span { color:#e2e8f0 !important; }
[data-baseweb="popover"],[data-baseweb="menu"] { background:#1e2547 !important; }
[data-baseweb="list-item"] { background:#1e2547 !important; color:#e2e8f0 !important; }
[data-baseweb="list-item"]:hover { background:rgba(99,102,241,.2) !important; }

/* ── Cards (general) ── */
.wt-card {
    background:linear-gradient(135deg,rgba(22,27,65,.92),rgba(10,12,28,.95));
    border:1px solid rgba(51,65,85,.65); border-radius:14px; padding:1.2rem 1.4rem;
    backdrop-filter:blur(10px); animation:fadeInUp .35s ease both;
    transition:all .22s ease; margin:.4rem 0;
}
.wt-card:hover { border-color:rgba(99,102,241,.45); transform:translateY(-2px); box-shadow:0 8px 32px rgba(99,102,241,.12); }

/* ── Severity borders ── */
.sev-critical { border-left:4px solid #ef4444 !important; }
.sev-high     { border-left:4px solid #f97316 !important; }
.sev-medium   { border-left:4px solid #eab308 !important; }
.sev-low      { border-left:4px solid #22c55e !important; }

/* ── Badges ── */
.badge { display:inline-flex; align-items:center; border-radius:6px; padding:3px 10px; font-size:.75rem; font-weight:700; letter-spacing:.05em; }
.badge-critical { background:rgba(239,68,68,.18); color:#f87171; border:1px solid rgba(239,68,68,.45); }
.badge-high     { background:rgba(249,115,22,.15); color:#fb923c; border:1px solid rgba(249,115,22,.4); }
.badge-medium   { background:rgba(234,179,8,.15);  color:#fde047; border:1px solid rgba(234,179,8,.4); }
.badge-low      { background:rgba(34,197,94,.15);  color:#4ade80; border:1px solid rgba(34,197,94,.4); }
.badge-info     { background:rgba(99,102,241,.15); color:#a5b4fc; border:1px solid rgba(99,102,241,.4); }
.badge-gemini   { background:rgba(16,185,129,.15); color:#6ee7b7; border:1px solid rgba(16,185,129,.4); }
.badge-groq     { background:rgba(245,158,11,.15); color:#fcd34d; border:1px solid rgba(245,158,11,.4); }
.badge-det      { background:rgba(99,102,241,.1);  color:#c4b5fd; border:1px solid rgba(99,102,241,.3); }

/* ── Log entries ── */
.log-entry {
    background:rgba(15,18,40,.6); border-radius:8px; padding:.55rem 1rem;
    margin:.25rem 0; border-left:3px solid rgba(99,102,241,.4);
    font-size:.82rem; color:#cbd5e1; font-family:'Inter',monospace;
    animation:fadeInUp .25s ease both;
}
.log-entry.disruption { border-color:#ef4444; }
.log-entry.success    { border-color:#22c55e; }
.log-entry.agent      { border-color:#a855f7; }
.log-entry.route      { border-color:#3b82f6; }
.log-entry.warning    { border-color:#eab308; }
.log-entry.notification { border-color:#f59e0b; }
.log-timestamp { color:#475569; font-size:.73rem; margin-right:.5rem; }

/* ── Hero banner ── */
.hero-banner {
    background:linear-gradient(135deg,#06091a 0%,#0c1230 40%,#070a1e 70%,#03040d 100%);
    border:1px solid rgba(99,102,241,.28); border-radius:20px;
    padding:1.8rem 2.2rem; margin-bottom:1.2rem; position:relative; overflow:hidden;
    animation:pulse-glow 4s ease-in-out infinite;
}
.hero-banner::before {
    content:''; position:absolute; top:-40%; right:-5%; width:380px; height:380px;
    background:radial-gradient(circle,rgba(99,102,241,.09) 0%,transparent 70%);
    border-radius:50%; pointer-events:none;
}

/* ── Live dot ── */
.live-dot {
    display:inline-block; width:9px; height:9px; border-radius:50%;
    background:#22c55e; box-shadow:0 0 8px rgba(34,197,94,.8);
    animation:ping 1.5s cubic-bezier(0,0,.2,1) infinite; vertical-align:middle; margin-right:5px;
}
.live-dot-red { background:#ef4444; box-shadow:0 0 8px rgba(239,68,68,.8); }

/* ── Section header ── */
.section-hdr {
    background:linear-gradient(90deg,#6366f1,#8b5cf6,#ec4899,#f59e0b);
    background-size:200% auto; -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    font-size:1.35rem; font-weight:800; font-family:'Space Grotesk',sans-serif;
    animation:shimmer 4s linear infinite;
}

/* ── Sim buttons ── */
.sim-btn .stButton>button { text-align:left !important; font-size:.85rem !important; padding:.65rem 1rem !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:#03040d; }
::-webkit-scrollbar-thumb { background:rgba(99,102,241,.35); border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background:rgba(99,102,241,.65); }

/* ── Typography ── */
h1,h2,h3,h4,h5,h6 { color:#f1f5f9 !important; font-family:'Space Grotesk',sans-serif !important; }
hr { border-color:rgba(51,65,85,.5) !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background:rgba(15,18,40,.75) !important; border:1px solid rgba(51,65,85,.55) !important;
    border-radius:12px !important; backdrop-filter:blur(8px);
}
[data-testid="stExpander"] summary { color:#94a3b8 !important; font-weight:500; padding:.75rem 1rem; }
[data-testid="stExpander"] summary:hover { color:#e2e8f0 !important; }

/* ── Progress bar ── */
[data-testid="stProgressBar"]>div>div { background:linear-gradient(90deg,#4f46e5,#22c55e) !important; }

/* ── Checkbox/Radio ── */
[data-testid="stCheckbox"] label,[data-testid="stRadio"] label { color:#94a3b8 !important; font-size:.88rem !important; }

/* ── Hide Streamlit chrome ── */
[data-testid="stHeader"],header[data-testid="stHeader"] { background:#03040d !important; border-bottom:1px solid rgba(99,102,241,.12) !important; }
[data-testid="stStatusWidget"],[data-testid="stDecoration"],#stDecoration { display:none !important; }
footer { visibility:hidden !important; }

/* ── Notification card ── */
.notif-card {
    background:linear-gradient(135deg,rgba(22,27,65,.9),rgba(10,12,28,.95));
    border:1px solid rgba(51,65,85,.6); border-radius:12px; padding:1rem 1.2rem;
    margin:.4rem 0; transition:all .2s ease; animation:fadeInUp .3s ease both;
}
.notif-card:hover { border-color:rgba(99,102,241,.4); transform:translateY(-1px); }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius:12px; overflow:hidden; border:1px solid rgba(51,65,85,.5); }
[data-testid="stDataFrame"] thead th {
    background:rgba(79,70,229,.15) !important; color:#a5b4fc !important;
    font-weight:600 !important; font-size:.81rem !important; text-transform:uppercase; letter-spacing:.05em;
}

/* ── Toast ── */
[data-testid="stToast"] {
    background:rgba(10,12,28,.96) !important; border:1px solid rgba(34,197,94,.4) !important;
    border-radius:12px !important; backdrop-filter:blur(12px); color:#e2e8f0 !important;
    box-shadow:0 8px 36px rgba(0,0,0,.5) !important;
}
div[data-testid="column"] { padding:0 .35rem !important; }
[data-testid="stHorizontalBlock"] { gap:.8rem !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Import backend modules
# ─────────────────────────────────────────────────────────────
from state_manager import (
    load_system_state, update_system_state, load_log,
    load_decisions, load_notifications, init_state_files,
)


@st.cache_resource(show_spinner="🌐 Building supply chain graph…")
def _load_graph():
    from graph_engine import build_graph, load_metadata, load_routes
    meta = load_metadata()
    G    = build_graph(meta, load_routes())
    return G, meta


@st.cache_data(show_spinner=False, ttl=600)
def _load_shipments():
    p = ROOT / "data" / "active_shipments.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return []


# ─────────────────────────────────────────────────────────────
# Init
# ─────────────────────────────────────────────────────────────
init_state_files()

# Auto-refresh every 15 s
st_autorefresh(interval=15_000, key="main_refresh")

# Load graph (cached)
try:
    G, meta_df = _load_graph()
    graph_ok = True
except Exception as e:
    st.error(f"❌ Graph not loaded — run `python data/generate_world_network.py` first\n{e}")
    graph_ok = False
    G, meta_df = None, None

sys_state = load_system_state()


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _sev_class(s: str) -> str:
    return {"critical": "sev-critical", "high": "sev-high", "medium": "sev-medium", "low": "sev-low"}.get(str(s).lower(), "sev-low")

def _badge(text: str, cls: str = "info") -> str:
    return f'<span class="badge badge-{cls}">{text}</span>'

def _status_dot(running: bool) -> str:
    cls = "" if running else " live-dot-red"
    s = "LIVE" if running else "IDLE"
    return f'<span class="live-dot{cls}"></span> <b>{s}</b>'

def _fmt_usd(v) -> str:
    if v is None: return "—"
    v = float(v)
    if v >= 1e6: return f"${v/1e6:.2f}M"
    if v >= 1e3: return f"${v/1e3:.0f}K"
    return f"${v:.0f}"

def _agent_badge(src: str) -> str:
    if "gemini" in src.lower():
        return _badge("⚡ Gemini", "gemini")
    if "groq" in src.lower():
        return _badge("🔥 Groq", "groq")
    return _badge("🔧 Deterministic", "det")


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center;margin-bottom:1rem'>
        <div style='font-size:2.4rem;animation:float 3s ease-in-out infinite'>🛰️</div>
        <div style='font-size:1.25rem;font-weight:800;color:#e2e8f0;font-family:Space Grotesk,sans-serif'>SupplAI</div>
        <div style='font-size:.72rem;color:#6366f1;font-weight:600;letter-spacing:.12em;text-transform:uppercase'>Autonomous Watchtower</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Status
    running = sys_state.get("watchtower_running", False)
    st.markdown(f"**Status:** {_status_dot(running)}", unsafe_allow_html=True)
    activity = sys_state.get("current_activity", "Idle")
    st.caption(f"📋 {activity}")

    last_scan = sys_state.get("last_scan_utc")
    if last_scan:
        st.caption(f"🕐 Last scan: {last_scan[11:19]} UTC")

    st.divider()

    # Watchtower controls
    scan_interval = st.slider("Scan interval (min)", 1, 30, 5, key="scan_interval_slider")

    col_start, col_stop = st.columns(2)
    with col_start:
        if st.button("▶ Start", key="btn_start", disabled=running):
            sys.path.insert(0, str(ROOT / "daemon"))
            from watchtower import start_watchtower
            ok = start_watchtower(scan_interval)
            if ok:
                st.toast("✅ Watchtower started!", icon="🛰️")
            else:
                st.toast("Already running", icon="ℹ️")
            st.rerun()

    with col_stop:
        if st.button("⏹ Stop", key="btn_stop", disabled=not running):
            from watchtower import stop_watchtower
            stop_watchtower()
            st.toast("🛑 Watchtower stopped", icon="ℹ️")
            st.rerun()

    st.divider()

    # KPIs
    st.markdown("#### 📊 Network Stats")
    m1, m2 = st.columns(2)
    m1.metric("Nodes", sys_state.get("network_nodes", len(G.nodes) if G else 0))
    m2.metric("Edges", sys_state.get("network_edges", len(G.edges) if G else 0))
    m3, m4 = st.columns(2)
    m3.metric("Shipments", sys_state.get("active_shipments", "—"))
    m4.metric("Disruptions", sys_state.get("active_disruptions", 0))

    st.divider()
    m5, m6 = st.columns(2)
    m5.metric("Decisions Today", sys_state.get("decisions_today", 0))
    m6.metric("Notifications", sys_state.get("total_notifications", 0))

    st.divider()
    st.caption("🛰️ SupplAI • Autonomous Watchtower\n\nPowered by Gemini 2.5 Flash")


# ─────────────────────────────────────────────────────────────
# Main tabs
# ─────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Live World Map",
    "📡 Watchtower Feed",
    "🤖 AI Decisions",
    "📦 Shipments",
    "🎯 Simulation",
])


# ═══════════════════════════════════════════════════════════════
# TAB 1 — Live World Map
# ═══════════════════════════════════════════════════════════════

with tab1:
    st.markdown('<p class="section-hdr">🌐 Live Supply Chain World Map</p>', unsafe_allow_html=True)
    st.caption("Real-time view of global logistics network — node color indicates risk level")

    if not graph_ok:
        st.warning("Graph not loaded. Run data generation script first.")
    else:
        decisions = load_decisions(5)
        # Collect all disrupted nodes from recent decisions
        disrupted_nodes = set()
        cascade_depth: dict = {}
        for dec in decisions[:1]:
            if dec.get("suggested_alternatives") or dec.get("approved_reroutes") or dec.get("flagged_suppliers"):
                for fnode in dec.get("flagged_suppliers", []):
                    n = fnode.get("node", "")
                    if n:
                        disrupted_nodes.add(n)
                        cascade_depth[n] = 0

        # Build node traces
        node_lats, node_lons, node_texts, node_colors, node_sizes = [], [], [], [], []
        for node_id, attrs in G.nodes(data=True):
            lat = attrs.get("lat", 0)
            lon = attrs.get("lon", 0)
            city = attrs.get("city_name", node_id)
            country = attrs.get("country", "")
            product = attrs.get("product_category", "")
            tier = attrs.get("tier", 3)

            if node_id in disrupted_nodes:
                col = "#ef4444"
                size = 16
            elif attrs.get("is_sanctioned"):
                col = "#dc2626"
                size = 10
            else:
                tier_colors = {1: "#6366f1", 2: "#8b5cf6", 3: "#64748b", 4: "#475569", 5: "#334155"}
                col  = tier_colors.get(tier, "#475569")
                size = {1: 14, 2: 11, 3: 9, 4: 7, 5: 6}.get(tier, 8)

            node_lats.append(lat)
            node_lons.append(lon)
            node_sizes.append(size)
            node_colors.append(col)
            node_texts.append(
                f"<b>{city}</b><br>{country}<br>"
                f"Tier {tier} | {product}<br>"
                f"{'⚠️ DISRUPTED' if node_id in disrupted_nodes else '✅ Active'}"
            )

        node_trace = go.Scattergeo(
            lat=node_lats, lon=node_lons,
            mode="markers",
            marker=dict(size=node_sizes, color=node_colors, opacity=0.9,
                        line=dict(width=0.5, color="rgba(255,255,255,0.15)")),
            hovertext=node_texts, hoverinfo="text",
            name="Supply Nodes",
        )

        # Sample shipment arcs (show 80 for visibility)
        shipments = _load_shipments()
        arc_lats, arc_lons = [], []
        arc_colors = {"on_time": "rgba(99,102,241,0.25)", "at_risk": "rgba(234,179,8,0.4)",
                       "delayed": "rgba(239,68,68,0.5)", "rerouted": "rgba(34,197,94,0.4)"}

        for ship in shipments[:120]:
            src_id = ship.get("origin", "")
            dst_id = ship.get("destination", "")
            if G.has_node(src_id) and G.has_node(dst_id):
                slat = G.nodes[src_id]["lat"]; slon = G.nodes[src_id]["lon"]
                dlat = G.nodes[dst_id]["lat"]; dlon = G.nodes[dst_id]["lon"]
                arc_lats += [slat, dlat, None]
                arc_lons += [slon, dlon, None]

        arc_trace = go.Scattergeo(
            lat=arc_lats, lon=arc_lons, mode="lines",
            line=dict(width=0.7, color="rgba(99,102,241,0.2)"),
            hoverinfo="none", name="Active Shipments", showlegend=False,
        )

        fig = go.Figure(data=[arc_trace, node_trace])
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="#03040d", plot_bgcolor="#03040d",
            geo=dict(
                projection_type="natural earth",
                bgcolor="#03040d",
                showland=True, landcolor="#0d1117",
                showocean=True, oceancolor="#030812",
                showcoastlines=True, coastlinecolor="rgba(71,85,105,0.4)",
                showframe=False,
                showcountries=True, countrycolor="rgba(51,65,85,0.3)",
            ),
            showlegend=False,
            height=540,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Legend
        lc1, lc2, lc3, lc4, lc5 = st.columns(5)
        lc1.markdown('<div style="color:#6366f1;font-size:.8rem">🔵 Tier 1 Hub</div>', unsafe_allow_html=True)
        lc2.markdown('<div style="color:#8b5cf6;font-size:.8rem">🟣 Tier 2 Node</div>', unsafe_allow_html=True)
        lc3.markdown('<div style="color:#64748b;font-size:.8rem">⚫ Tier 3 Node</div>', unsafe_allow_html=True)
        lc4.markdown('<div style="color:#ef4444;font-size:.8rem">🔴 Disrupted</div>', unsafe_allow_html=True)
        lc5.markdown('<div style="color:#94a3b8;font-size:.8rem">— Active Route</div>', unsafe_allow_html=True)

        # Network stats row
        st.divider()
        kc = st.columns(5)
        kc[0].metric("🌐 Hub Cities",    sum(1 for _, a in G.nodes(data=True) if a.get("tier") == 1))
        kc[1].metric("🔗 Trade Lanes",   G.number_of_edges())
        kc[2].metric("🌍 Countries",     len({a.get("country","") for _,a in G.nodes(data=True)}))
        kc[3].metric("🚨 Disrupted Now", len(disrupted_nodes))
        kc[4].metric("📦 Shipments",     len(shipments))


# ═══════════════════════════════════════════════════════════════
# TAB 2 — Watchtower Feed
# ═══════════════════════════════════════════════════════════════

with tab2:
    st.markdown('<p class="section-hdr">📡 Live Watchtower Feed</p>', unsafe_allow_html=True)
    st.caption("Real-time log of all system activity — auto-refreshes every 15 seconds")

    logs = load_log(150)

    if not logs:
        st.markdown("""
        <div class="wt-card" style="text-align:center;padding:3rem">
            <div style="font-size:3rem;margin-bottom:1rem">🛰️</div>
            <div style="color:#6366f1;font-size:1.1rem;font-weight:600">Watchtower is idle</div>
            <div style="color:#475569;margin-top:.5rem">Start the watchtower from the sidebar to begin monitoring</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Filter controls
        fc1, fc2 = st.columns([3, 1])
        with fc1:
            filter_cat = st.multiselect(
                "Filter by category",
                ["disruption", "agent", "route", "notification", "weather", "news",
                 "earthquake", "simulation", "scan", "success", "warning", "error"],
                default=[],
                key="log_filter",
                label_visibility="collapsed",
            )
        with fc2:
            show_n = st.selectbox("Show", [50, 100, 150], index=0, label_visibility="collapsed")

        filtered = [l for l in logs if not filter_cat or l.get("category") in filter_cat][:show_n]

        log_html = ""
        for entry in filtered:
            cat = entry.get("category", "info")
            ts  = entry.get("timestamp", "")[-8:-1] if entry.get("timestamp") else ""
            icon = entry.get("icon", "ℹ️")
            msg  = entry.get("message", "")
            lvl  = entry.get("level", "info")

            cls_map = {
                "disruption": "disruption", "success": "success", "agent": "agent",
                "route": "route", "warning": "warning", "notification": "notification",
                "error": "disruption",
            }
            cls = cls_map.get(cat, "")

            log_html += f"""
            <div class="log-entry {cls}">
                <span class="log-timestamp">{ts}</span>
                <span>{icon}</span>
                <span style="margin-left:.4rem">{msg}</span>
            </div>"""

        st.markdown(log_html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 3 — AI Decisions
# ═══════════════════════════════════════════════════════════════

with tab3:
    st.markdown('<p class="section-hdr">🤖 AI Decision Center</p>', unsafe_allow_html=True)
    st.caption("Every autonomous decision the AI agent made — with full explainable reasoning")

    decisions = load_decisions(20)

    if not decisions:
        st.markdown("""
        <div class="wt-card" style="text-align:center;padding:3rem">
            <div style="font-size:3rem;margin-bottom:1rem">🤖</div>
            <div style="color:#6366f1;font-size:1.1rem;font-weight:600">No decisions yet</div>
            <div style="color:#475569;margin-top:.5rem">Start the watchtower or fire a simulation to see AI decisions here</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for dec in decisions:
            sev       = dec.get("severity", "medium").lower()
            sev_cls   = _sev_class(sev)
            src       = dec.get("agent_source", "deterministic")
            ts        = dec.get("timestamp", "")[:16].replace("T", " ")
            event     = dec.get("event_text", "Unknown event")[:80]
            rl        = dec.get("risk_level", "Medium")
            recovery  = dec.get("estimated_recovery_days", 7)
            n_routes  = len(dec.get("suggested_alternatives") or dec.get("approved_reroutes", []))
            n_flagged = len(dec.get("flagged_suppliers", []))
            n_steps   = len(dec.get("steps", []))
            did       = dec.get("decision_id", "")
            trigger   = dec.get("trigger_source", "")

            trigger_icon = {"news": "📰", "weather": "🌩️", "earthquake": "🌍",
                            "simulation": "🎯", "unknown": "📡"}.get(trigger, "📡")

            with st.expander(f"{trigger_icon} {event} — {ts}", expanded=False):
                # Header row
                hc = st.columns([2, 1, 1, 1, 1])
                hc[0].markdown(f"**Decision ID:** `{did}`")
                hc[1].markdown(f"**Risk:** {_badge(rl, sev)}", unsafe_allow_html=True)
                hc[2].markdown(f"**Agent:** {_agent_badge(src)}", unsafe_allow_html=True)
                hc[3].markdown(f"**Recovery:** {recovery} days")
                hc[4].markdown(f"**Time:** {dec.get('elapsed_seconds', 0):.1f}s")

                st.divider()

                # Summary
                st.markdown(f"**📋 Executive Summary**")
                st.info(dec.get("final_summary", "No summary available."))

                # Priority actions
                st.markdown("**🎯 Priority Actions**")
                for i, action in enumerate(dec.get("priority_actions", []), 1):
                    st.markdown(f"{i}. {action}")

                st.divider()

                # Agent reasoning steps
                steps = dec.get("steps", [])
                if steps:
                    st.markdown(f"**🔍 Agent Reasoning Chain ({n_steps} steps)**")
                    for step_data in steps:
                        step_num  = step_data.get("step", "?")
                        tool      = step_data.get("tool", "unknown")
                        thought   = step_data.get("thought", "")
                        result    = step_data.get("result", {})
                        duration  = step_data.get("duration_ms", 0)

                        tool_icons = {
                            "assess_disruptions": "🔍", "score_route_alternatives": "🗺️",
                            "approve_reroute": "📋", "suggest_reroute": "📋", "flag_critical_supplier": "🚨",
                            "estimate_recovery": "📅", "notify_suppliers": "📬",
                            "finalize_plan": "📋",
                        }
                        t_icon = tool_icons.get(tool, "⚙️")

                        with st.expander(f"Step {step_num}: {t_icon} `{tool}()` — {duration}ms", expanded=False):
                            if thought:
                                st.markdown(f"💭 **Thought:** *{thought}*")
                            st.markdown("**Result:**")
                            st.json(result, expanded=False)

                st.divider()

                # Approved reroutes
                reroutes = dec.get("suggested_alternatives") or dec.get("approved_reroutes", [])
                if reroutes:
                    st.markdown(f"**📋 Suggested Route Alternatives ({n_routes})** — *Advisory only, supplier decides*")
                    st.caption("ℹ️ These are recommendations. Suppliers should evaluate with their freight forwarder before making changes.")
                    for rt in reroutes:
                        st.markdown(
                            f'<div class="wt-card sev-low" style="margin:.3rem 0">'
                            f'<b>🗺️ Suggested: {rt.get("source","?")} → {rt.get("destination","?")}</b><br>'
                            f'<span style="color:#94a3b8;font-size:.85rem">{rt.get("reason","")[:160]}</span>'
                            f'</div>', unsafe_allow_html=True
                        )

                # Flagged suppliers
                flagged = dec.get("flagged_suppliers", [])
                if flagged:
                    st.markdown(f"**🚨 Flagged Critical Suppliers ({n_flagged})**")
                    for fl in flagged:
                        priority = fl.get("priority", "high")
                        p_badge  = _badge(priority.upper(), "critical" if priority == "high" else "medium")
                        st.markdown(
                            f'<div class="wt-card sev-critical" style="margin:.3rem 0">'
                            f'<b>🏭 {fl.get("city", fl.get("node","?"))}</b> {p_badge}<br>'
                            f'<span style="color:#94a3b8;font-size:.85rem">{fl.get("reason","")[:120]}</span>'
                            f'</div>', unsafe_allow_html=True
                        )


# ═══════════════════════════════════════════════════════════════
# TAB 4 — Shipment Tracker
# ═══════════════════════════════════════════════════════════════

with tab4:
    st.markdown('<p class="section-hdr">📦 Active Shipment Tracker</p>', unsafe_allow_html=True)
    st.caption("Live view of 500 global shipments — status updates as disruptions are detected")

    shipments = _load_shipments()
    if not shipments:
        st.warning("No shipments loaded. Run `python data/simulate_shipments.py` first.")
    else:
        # Summary metrics
        sc = st.columns(4)
        stati = {s.get("status", "?"): 0 for s in shipments}
        for s in shipments:
            stati[s.get("status", "?")] = stati.get(s.get("status", "?"), 0) + 1

        sc[0].metric("✅ On Time",    stati.get("on_time", 0))
        sc[1].metric("⚠️ At Risk",   stati.get("at_risk", 0),  delta=f"-{stati.get('at_risk',0)} shipments")
        sc[2].metric("🔴 Delayed",   stati.get("delayed", 0),  delta=f"-{stati.get('delayed',0)}", delta_color="inverse")
        sc[3].metric("🔄 Rerouted",  stati.get("rerouted", 0))

        st.divider()

        # Filters
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            status_filter = st.multiselect(
                "Status", ["on_time", "at_risk", "delayed", "rerouted"],
                default=["delayed", "at_risk", "rerouted"],
                key="ship_status_filter",
            )
        with fc2:
            mode_filter = st.multiselect(
                "Transport mode", ["sea", "air", "road/rail"],
                default=[], key="ship_mode_filter",
            )
        with fc3:
            cargo_opts = sorted(list({s.get("cargo_type", "") for s in shipments if s.get("cargo_type")}))
            cargo_filter = st.multiselect("Cargo type", cargo_opts, default=[], key="ship_cargo_filter")

        filtered_ships = [
            s for s in shipments
            if (not status_filter or s.get("status") in status_filter)
            and (not mode_filter or s.get("transport_mode") in mode_filter)
            and (not cargo_filter or s.get("cargo_type") in cargo_filter)
        ]

        # Display as cards (first 30) then table
        status_icons = {"on_time": "✅", "at_risk": "⚠️", "delayed": "🔴", "rerouted": "🔄"}
        status_colors = {"on_time": "#22c55e", "at_risk": "#eab308", "delayed": "#ef4444", "rerouted": "#3b82f6"}
        sev_classes   = {"on_time": "sev-low", "at_risk": "sev-medium", "delayed": "sev-critical", "rerouted": "sev-low"}

        st.markdown(f"**Showing {len(filtered_ships)} shipments**")

        # Card view for first 15
        if filtered_ships:
            cols = st.columns(2)
            for i, ship in enumerate(filtered_ships[:16]):
                with cols[i % 2]:
                    st_code = ship.get("status", "on_time")
                    sc_cls  = sev_classes.get(st_code, "sev-low")
                    s_icon  = status_icons.get(st_code, "📦")
                    s_color = status_colors.get(st_code, "#64748b")
                    progress = ship.get("progress_pct", 0)
                    cause   = ship.get("disruption_cause") or ship.get("reroute_reason") or ""

                    st.markdown(f"""
                    <div class="wt-card {sc_cls}">
                        <div style="display:flex;justify-content:space-between;align-items:flex-start">
                            <div>
                                <div style="font-weight:700;font-size:.9rem">{s_icon} {ship.get("shipment_id","")}</div>
                                <div style="color:#94a3b8;font-size:.8rem;margin-top:.2rem">
                                    {ship.get("origin_city","?")} → {ship.get("destination_city","?")}
                                </div>
                            </div>
                            <span class="badge badge-{'critical' if st_code=='delayed' else 'medium' if st_code=='at_risk' else 'info'}">
                                {st_code.replace('_',' ').upper()}
                            </span>
                        </div>
                        <div style="margin-top:.6rem;font-size:.8rem;color:#64748b">
                            🚢 {ship.get("carrier","?")} &nbsp;|&nbsp; 📦 {ship.get("cargo_type","?")} &nbsp;|&nbsp; 💰 {_fmt_usd(ship.get("value_usd",0))}
                        </div>
                        <div style="margin-top:.5rem;background:rgba(255,255,255,.06);border-radius:4px;height:4px;overflow:hidden">
                            <div style="width:{progress}%;height:100%;background:{s_color};border-radius:4px;transition:width .5s"></div>
                        </div>
                        <div style="font-size:.72rem;color:#64748b;margin-top:.3rem">{progress:.0f}% complete{' — ' + cause if cause else ''}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # Table view for all
        if len(filtered_ships) > 0:
            st.divider()
            st.markdown("**Full table**")
            tbl = pd.DataFrame([{
                "ID": s.get("shipment_id", ""),
                "Status": s.get("status", "").replace("_", " ").title(),
                "Origin": s.get("origin_city", ""),
                "Destination": s.get("destination_city", ""),
                "Carrier": s.get("carrier", ""),
                "Cargo": s.get("cargo_type", ""),
                "Mode": s.get("transport_mode", ""),
                "Value": _fmt_usd(s.get("value_usd", 0)),
                "Progress %": s.get("progress_pct", 0),
                "Delay Days": s.get("delay_days", 0),
            } for s in filtered_ships[:200]])
            st.dataframe(tbl, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════
# TAB 5 — Simulation Console
# ═══════════════════════════════════════════════════════════════

with tab5:
    st.markdown('<p class="section-hdr">🎯 Simulation Console</p>', unsafe_allow_html=True)
    st.caption("Inject any disruption scenario and watch the AI respond in real-time")

    st.markdown("""
    <div class="wt-card" style="padding:1rem 1.4rem;margin-bottom:1rem;border-left:4px solid #6366f1">
        <b>How it works:</b> Select a preset scenario or type a custom event, then click <b>Fire Disruption →</b>
        The AI agent will immediately analyze the event, compute optimized reroutes, and send supplier notifications.
        Watch the results appear in the <b>📡 Watchtower Feed</b> and <b>🤖 AI Decisions</b> tabs.
    </div>
    """, unsafe_allow_html=True)

    # Preset scenarios
    SCENARIOS = [
        {
            "label": "🌪️ Port of Shanghai Closure",
            "event_text": "Port of Shanghai closed due to typhoon — all outbound container traffic suspended",
            "affected_countries": ["China"],
            "affected_sectors": ["Electronics", "Manufacturing", "Consumer Goods"],
            "affected_nodes": ["SHA", "SZX", "GZH"],
            "severity": "critical",
            "category": "natural_disaster",
            "source": "simulation",
        },
        {
            "label": "⚓ Suez Canal Blockage",
            "event_text": "Suez Canal blocked by grounded vessel — all trans-Suez shipping halted indefinitely",
            "affected_countries": ["Egypt"],
            "affected_sectors": ["Transshipment"],
            "affected_nodes": ["SUZ"],
            "severity": "critical",
            "category": "logistics",
            "source": "simulation",
        },
        {
            "label": "📈 US-China Tariff Escalation",
            "event_text": "US imposes emergency 200% tariff surcharge on all Chinese electronics imports",
            "affected_countries": ["China", "USA"],
            "affected_sectors": ["Electronics", "Semiconductors"],
            "affected_nodes": ["SHA", "SZX", "LAX", "NYC"],
            "severity": "high",
            "category": "geopolitical",
            "source": "simulation",
        },
        {
            "label": "🌊 Rotterdam Flood Damage",
            "event_text": "Severe flooding in Rotterdam port area — 60% of dock operations suspended",
            "affected_countries": ["Netherlands"],
            "affected_sectors": ["Chemicals", "Consumer Goods"],
            "affected_nodes": ["RTM"],
            "severity": "high",
            "category": "natural_disaster",
            "source": "simulation",
        },
        {
            "label": "✈️ European Air Cargo Strike",
            "event_text": "European air cargo handlers on indefinite strike — air freight across EU halted",
            "affected_countries": ["Germany", "France", "Netherlands", "UK"],
            "affected_sectors": ["Pharmaceuticals", "Luxury Goods"],
            "affected_nodes": ["FRA", "CDG", "HAM", "LON"],
            "severity": "high",
            "category": "labor",
            "source": "simulation",
        },
        {
            "label": "🔥 Singapore Port Fire",
            "event_text": "Major fire at Singapore Jurong Port — terminal 2 evacuated, 40% capacity offline",
            "affected_countries": ["Singapore"],
            "affected_sectors": ["Transshipment", "Semiconductors"],
            "affected_nodes": ["SGP"],
            "severity": "high",
            "category": "industrial_accident",
            "source": "simulation",
        },
        {
            "label": "⚡ Taiwan Strait Tension",
            "event_text": "Escalating Taiwan Strait military exercises disrupt all commercial shipping lanes",
            "affected_countries": ["Taiwan", "China"],
            "affected_sectors": ["Semiconductors", "Electronics"],
            "affected_nodes": ["TPE", "HKG", "SHA"],
            "severity": "critical",
            "category": "geopolitical",
            "source": "simulation",
        },
        {
            "label": "🚢 Panama Canal Drought",
            "event_text": "Extreme drought reduces Panama Canal water levels — vessel transit limited by 40%",
            "affected_countries": ["Panama"],
            "affected_sectors": ["Transshipment"],
            "affected_nodes": ["PAN"],
            "severity": "medium",
            "category": "natural_disaster",
            "source": "simulation",
        },
    ]

    st.markdown("#### ⚡ Preset Scenarios")
    # 4 columns of scenario buttons
    rows = [SCENARIOS[i:i+2] for i in range(0, len(SCENARIOS), 2)]
    for row in rows:
        cols = st.columns(2)
        for j, scenario in enumerate(row):
            with cols[j]:
                if st.button(scenario["label"], key=f"sim_{scenario['label'][:15]}", use_container_width=True):
                    st.session_state["pending_sim"] = scenario

    st.divider()

    # Custom event
    st.markdown("#### ✍️ Custom Event")
    custom_text = st.text_area(
        "Describe a supply chain disruption",
        placeholder="e.g. Major earthquake in Japan disrupts automotive parts supply from Toyota factories in Nagoya",
        height=80, key="custom_event_text",
    )
    cc1, cc2, cc3 = st.columns(3)
    custom_severity = cc1.selectbox("Severity", ["high", "critical", "medium", "low"], key="custom_sev")
    custom_category = cc2.selectbox("Category", ["geopolitical","natural_disaster","labor","industrial_accident","logistics","other"], key="custom_cat")
    custom_countries = cc3.text_input("Affected countries (comma-sep)", key="custom_countries")

    if st.button("🎯 Fire Custom Disruption →", key="btn_custom_sim", use_container_width=True):
        if custom_text.strip():
            countries_list = [c.strip() for c in custom_countries.split(",") if c.strip()]
            st.session_state["pending_sim"] = {
                "event_text": custom_text.strip(),
                "affected_countries": countries_list,
                "affected_sectors": [],
                "severity": custom_severity,
                "category": custom_category,
                "source": "simulation",
                "title": custom_text.strip()[:60],
            }
        else:
            st.warning("Please describe the disruption event.")

    # Execute pending simulation
    if "pending_sim" in st.session_state:
        scenario = st.session_state.pop("pending_sim")
        with st.spinner(f"🎯 Firing simulation: {scenario.get('event_text','')[:60]}…"):
            try:
                sys.path.insert(0, str(ROOT / "daemon"))
                from watchtower import trigger_simulation
                trigger_simulation(scenario)
                st.success(f"✅ Simulation fired! Check the **📡 Watchtower Feed** and **🤖 AI Decisions** tabs.")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Simulation error: {e}")
                import traceback; traceback.print_exc()

    # Notifications panel
    st.divider()
    st.markdown("#### 📬 Supplier Notifications")
    notifs = load_notifications(20)

    if not notifs:
        st.markdown('<div style="color:#475569;text-align:center;padding:1.5rem">No notifications sent yet</div>',
                    unsafe_allow_html=True)
    else:
        for notif in notifs[:15]:
            ntype = notif.get("type", "info")
            color = notif.get("color", "#6366f1")
            icon  = notif.get("icon", "📬")
            send_ts = notif.get("sent_at", "")[:16].replace("T", " ")
            sev_n   = notif.get("severity", "medium")

            with st.expander(
                f"{icon} {notif.get('type_label','')} → {notif.get('recipient_name','')} ({send_ts})",
                expanded=False,
            ):
                nc1, nc2 = st.columns([3, 1])
                nc1.markdown(f"**To:** {notif.get('recipient_name','')} `{notif.get('recipient_email','')}`")
                nc2.markdown(f"**Severity:** {_badge(sev_n.upper(), 'critical' if sev_n=='high' else 'medium')}", unsafe_allow_html=True)
                st.markdown(f"**Subject:** {notif.get('subject','')}")
                st.markdown("**Message:**")
                st.code(notif.get("body", ""), language=None)
