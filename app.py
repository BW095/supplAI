"""
app.py — SupplAI: AI Supply Chain Disruption Monitor
=====================================================
Streamlit dashboard with 4 tabs:
  🌐 Network Graph  |  🔥 Risk Analysis  |  🔁 Rerouting  |  🤖 AI Brief

Run with:
    conda activate condaVE
    streamlit run app.py
"""

import sys
import time
import json
from pathlib import Path

import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import numpy as np

# ---------------------------------------------------------------------------
# Add src/ to Python path so we can import our modules
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from disruption_input import parse_disruption
from graph_builder    import build_graph, load_supply_metadata, get_graph_summary
from cascade_model    import run_cascade, get_cascade_stats
from risk_scoring     import score_nodes, compute_centrality
from reroute          import find_alternates, format_path
from llm_brief        import generate_brief
from shap_explain     import compute_shap, shap_bar_figure, shap_waterfall_figure, shap_to_text, FEATURE_DESCRIPTIONS, FEATURE_LABELS

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title = "SupplAI — Supply Chain Monitor",
    page_icon  = "🔗",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark premium theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ===== Google Fonts ===== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ===== Global Dark Background — Override Streamlit Defaults ===== */
html, body {
    background-color: #0a0e1a !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}

/* Force dark bg on all Streamlit wrappers */
.stApp,
.stApp > div,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="block-container"],
.block-container,
.main,
.main > div,
section[data-testid="stMain"],
div[data-testid="stMainBlockContainer"] {
    background-color: #0a0e1a !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}

/* All generic divs and paragraphs in main — ensure visible text */
.stMarkdown, .stMarkdown p, .stMarkdown div,
.element-container p, .element-container div {
    color: #e2e8f0;
}

/* ===== Sidebar ===== */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
    border-right: 1px solid #334155;
}
[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1rem;
}

/* Sidebar text visibility */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown div,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: #cbd5e1 !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #e2e8f0 !important;
}

/* Sidebar input labels */
[data-testid="stSidebar"] .stTextArea label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label {
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

/* ===== Main content area ===== */
.main .block-container {
    padding: 1.5rem 2rem;
    max-width: 100%;
}

/* ===== Metric cards ===== */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1e293b, #0f172a) !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4) !important;
}
[data-testid="metric-container"]:hover {
    border-color: #6366f1 !important;
    box-shadow: 0 4px 25px rgba(99, 102, 241, 0.2) !important;
    transition: all 0.2s ease;
}
[data-testid="metric-container"] label,
[data-testid="metric-container"] div {
    color: #94a3b8 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e2e8f0 !important;
    font-size: 1.4rem !important;
    font-weight: 700 !important;
}

/* ===== Tab styling ===== */
[data-baseweb="tab-list"] {
    background: #0f172a !important;
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
}
[data-baseweb="tab"] {
    border-radius: 6px;
    font-weight: 500;
    color: #94a3b8 !important;
    padding: 8px 20px;
}
[aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
}
[data-baseweb="tab-panel"] {
    background: transparent !important;
}

/* ===== Horizontal rule ===== */
hr {
    border-color: #1e293b !important;
}

/* ===== Dataframe / tables ===== */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
}

/* ===== Buttons ===== */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s ease;
    width: 100%;
}
.stButton > button:hover {
    opacity: 0.9;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4);
    transform: translateY(-1px);
}

/* ===== Text inputs ===== */
.stTextArea textarea {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextArea textarea::placeholder {
    color: #475569 !important;
}
.stTextArea textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
}

/* ===== Selectbox ===== */
[data-baseweb="select"] > div {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}
[data-baseweb="select"] span {
    color: #e2e8f0 !important;
}

/* Selectbox dropdown menu */
[data-baseweb="popover"] {
    background: #1e293b !important;
}
[data-baseweb="menu"] {
    background: #1e293b !important;
}
[data-baseweb="list-item"] {
    background: #1e293b !important;
    color: #e2e8f0 !important;
}
[data-baseweb="list-item"]:hover {
    background: #334155 !important;
}

/* ===== Slider ===== */
[data-testid="stSlider"] .stSlider > div {
    color: #e2e8f0 !important;
}
[data-testid="stSlider"] [data-testid="stTickBar"] {
    color: #475569 !important;
}

/* ===== Alert / info boxes ===== */
[data-testid="stAlert"] {
    background: #1e293b !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}

/* ===== Custom cards ===== */
.risk-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 0.5rem 0;
    box-shadow: 0 2px 10px rgba(0,0,0,0.3);
    color: #e2e8f0;
}
.risk-card:hover {
    border-color: #6366f1;
    transform: translateY(-2px);
    box-shadow: 0 6px 25px rgba(99, 102, 241, 0.2);
    transition: all 0.2s ease;
}
.risk-card-critical { border-left: 4px solid #ef4444; }
.risk-card-high     { border-left: 4px solid #f97316; }
.risk-card-medium   { border-left: 4px solid #eab308; }
.risk-card-low      { border-left: 4px solid #22c55e; }

.route-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 0.5rem 0;
    color: #e2e8f0;
}
.route-found  { border-left: 4px solid #22c55e; }
.route-none   { border-left: 4px solid #ef4444; }

/* ===== Section headers ===== */
.section-header {
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

/* ===== Brief sections ===== */
.brief-section {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 0.8rem 0;
}
.brief-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: #6366f1;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
}
.brief-content {
    color: #cbd5e1;
    line-height: 1.7;
    font-size: 0.95rem;
}

/* ===== Severity badge ===== */  
.badge-high   { background: #7f1d1d; color: #fca5a5; border-radius: 6px; padding: 2px 10px; font-size: 0.8rem; font-weight: 600; }
.badge-medium { background: #78350f; color: #fde68a; border-radius: 6px; padding: 2px 10px; font-size: 0.8rem; font-weight: 600; }
.badge-low    { background: #14532d; color: #86efac; border-radius: 6px; padding: 2px 10px; font-size: 0.8rem; font-weight: 600; }

/* ===== Spinner / progress ===== */
.stSpinner > div { border-top-color: #6366f1 !important; }

/* ===== Scrollbar ===== */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0f172a; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #6366f1; }

/* ===== Headings in main area ===== */
h1, h2, h3, h4, h5, h6 {
    color: #e2e8f0 !important;
}

/* ===== stInfo / stWarning ===== */
[data-testid="stNotification"] {
    background: #1e293b !important;
    color: #e2e8f0 !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached loaders — run once per session, not on every re-render
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="🔗 Building supply chain graph …")
def _load_graph():
    G = build_graph()
    return G

@st.cache_resource(show_spinner="📊 Loading city metadata …")
def _load_supply():
    return load_supply_metadata()

@st.cache_resource(show_spinner="🧠 Loading / training delay model …")
def _load_delay_model():
    from delay_model import load_or_train
    return load_or_train()

@st.cache_data(show_spinner=False)
def _compute_centrality(_G):
    return compute_centrality(_G)


# ---------------------------------------------------------------------------
# Graph visualisation with Plotly
# ---------------------------------------------------------------------------
def build_plotly_graph(
    G:              nx.DiGraph,
    cascade_result: dict,
    risk_df:        pd.DataFrame,
    supply_df:      pd.DataFrame,
    seed_nodes:     list,
) -> go.Figure:
    """Build an interactive Plotly geo-scatter map coloured by risk level."""

    # Build node risk lookup
    risk_lookup = {}
    if not risk_df.empty:
        for _, row in risk_df.iterrows():
            risk_lookup[row["node"]] = row["risk_score"]

    cascade_nodes = set(cascade_result.keys())
    seed_set      = set(seed_nodes)

    # Colour + size helpers
    def node_colour(node):
        if node in seed_set:   return "#ef4444"
        rs = risk_lookup.get(node, 0)
        if rs >= 0.65:         return "#f97316"
        if rs >= 0.40:         return "#eab308"
        if rs >= 0.10:         return "#3b82f6"
        return "#334155"

    def node_size(node):
        if node in seed_set:   return 16
        rs = risk_lookup.get(node, 0)
        if rs >= 0.65:         return 13
        if rs >= 0.40:         return 11
        if rs >= 0.10:         return 9
        return 7

    # ------------------------------------------------------------------ #
    # Arc lines for edges that are within the cascade                     #
    # ------------------------------------------------------------------ #
    arc_lats, arc_lons = [], []
    for src, dst in G.edges():
        src_lat = G.nodes[src].get("lat", 0)
        src_lon = G.nodes[src].get("lon", 0)
        dst_lat = G.nodes[dst].get("lat", 0)
        dst_lon = G.nodes[dst].get("lon", 0)
        if src in cascade_nodes and dst in cascade_nodes:
            arc_lats += [src_lat, dst_lat, None]
            arc_lons += [src_lon, dst_lon, None]

    arc_trace = go.Scattergeo(
        lat=arc_lats, lon=arc_lons,
        mode="lines",
        line=dict(width=0.8, color="rgba(239,68,68,0.35)"),
        hoverinfo="none",
        name="Disrupted Routes",
        showlegend=False,
    )

    # ------------------------------------------------------------------ #
    # Node scatter                                                        #
    # ------------------------------------------------------------------ #
    lats, lons, texts, colours, sizes = [], [], [], [], []

    for node in G.nodes():
        lat = G.nodes[node].get("lat", 0)
        lon = G.nodes[node].get("lon", 0)
        if lat == 0 and lon == 0:
            continue

        nd    = G.nodes[node]
        rs    = risk_lookup.get(node, 0)
        depth = cascade_result.get(node, -1)
        status_str = f"⚡ Cascade depth: {depth}" if depth >= 0 else "✅ Unaffected"
        hover = (
            f"<b>{nd.get('city_name', node)}</b><br>"
            f"🌍 {nd.get('country', '?')} · {nd.get('region', '?')}<br>"
            f"📦 {nd.get('product_category', '?')} | Tier {nd.get('tier', '?')}<br>"
            f"{status_str}<br>"
            f"🎯 Risk Score: <b>{rs:.3f}</b>"
        )
        lats.append(lat)
        lons.append(lon)
        texts.append(hover)
        colours.append(node_colour(node))
        sizes.append(node_size(node))

    node_trace = go.Scattergeo(
        lat=lats, lon=lons,
        mode="markers",
        hoverinfo="text",
        text=texts,
        marker=dict(
            color=colours,
            size=sizes,
            line=dict(width=0.8, color="rgba(255,255,255,0.15)"),
            symbol="circle",
        ),
        name="Supply Chain Nodes",
        showlegend=False,
    )

    fig = go.Figure(data=[arc_trace, node_trace])
    fig.update_geos(
        projection_type="natural earth",
        showland=True,       landcolor="#1a2235",
        showocean=True,      oceancolor="#0a0e1a",
        showcoastlines=True, coastlinecolor="#2d3f5f",
        showframe=False,
        showcountries=True,  countrycolor="#1e3050",
        bgcolor="#0a0e1a",
    )
    fig.update_layout(
        title=dict(
            text="Supply Chain Network — Global Disruption Impact Map",
            font=dict(color="#e2e8f0", size=16, family="Inter"),
        ),
        paper_bgcolor="#0a0e1a",
        margin=dict(l=0, r=0, t=45, b=0),
        height=580,
        annotations=[
            dict(x=0.01, y=0.02, xref="paper", yref="paper", showarrow=False,
                 text="🔴 Disruption Source  🟠 Critical Cascade  🟡 High Risk  🔵 Monitoring  ⚫ Unaffected",
                 font=dict(color="#94a3b8", size=11, family="Inter"),
                 bgcolor="rgba(10,14,26,0.7)", borderpad=6, align="left"),
        ],
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        # Logo/title
        st.markdown("""
        <div style="text-align:center; padding: 1rem 0;">
            <div style="font-size: 2.5rem; margin-bottom: 0.3rem;">🔗</div>
            <div style="font-size: 1.3rem; font-weight: 700; 
                        background: linear-gradient(135deg, #6366f1, #8b5cf6);
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                SupplAI
            </div>
            <div style="font-size: 0.75rem; color: #64748b; margin-top: 0.2rem;">
                AI Supply Chain Monitor
            </div>
        </div>
        <hr style="border-color: #1e293b; margin: 0.5rem 0 1rem;">
        """, unsafe_allow_html=True)

        st.markdown("### 🔍 Disruption Input")

        # Text area for disruption description
        event_text = st.text_area(
            label       = "Describe the disruption event:",
            value       = st.session_state.get("event_text", ""),
            height      = 120,
            placeholder = "e.g. Factory shutdown in China affecting electronics supply chain...",
            help        = "Describe the supply chain disruption in plain English.",
            key         = "event_text_input",
        )

        # Severity override
        severity_override = st.selectbox(
            "Severity Override",
            options=["Auto-detect", "High", "Medium", "Low"],
            index=0,
            help="Override the auto-detected severity level.",
        )

        # Cascade depth
        max_depth = st.slider(
            "Cascade Depth",
            min_value=1, max_value=6,
            value=4,
            help="How many hops downstream to propagate the disruption.",
        )

        st.markdown("---")

        # Demo scenario button
        st.markdown("**🧪 Quick Demo:**")
        if st.button("🏭 China Electronics Shutdown"):
            st.session_state["event_text"] = (
                "Factory shutdown in China affecting electronics supply chain"
            )
            st.rerun()

        if st.button("🌊 Southeast Asia Flood"):
            st.session_state["event_text"] = (
                "Severe flooding in Vietnam affecting textile and electronics manufacturing"
            )
            st.rerun()

        if st.button("⚡ Korea Semiconductor Strike"):
            st.session_state["event_text"] = (
                "Labor strike in South Korea disrupting semiconductor production"
            )
            st.rerun()

        st.markdown("---")

        # Analyse button
        run_analysis = st.button("🚀 Analyse Disruption", type="primary", use_container_width=True)

        st.markdown("---")
        st.markdown("""
        <div style="font-size: 0.7rem; color: #475569; text-align: center;">
            Powered by NetworkX · XGBoost · Gemini AI<br>
            Built for AI Hackathon 2024
        </div>
        """, unsafe_allow_html=True)

    # Sync text area value to session state
    st.session_state["event_text"] = event_text

    return event_text, severity_override, max_depth, run_analysis


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    # ---- Init session state ----
    if "results" not in st.session_state:
        st.session_state["results"] = None
    if "event_text" not in st.session_state:
        st.session_state["event_text"] = ""

    # ---- Load resources ----
    G         = _load_graph()
    supply_df = _load_supply()
    centrality = _compute_centrality(G)

    # ---- Sidebar ----
    event_text, severity_override, max_depth, run_analysis = render_sidebar()

    # ---- Header ----
    col1, col2, col3, col4, col5 = st.columns(5)
    summary = get_graph_summary(G)
    col1.metric("🌐 Network Nodes",   f"{summary['nodes']:,}")
    col2.metric("🔗 Supply Routes",   f"{summary['edges']:,}")
    col3.metric("🌍 Countries",       f"{summary['countries']}")
    col4.metric("📊 Avg Connections", f"{summary['avg_degree']:,.1f}")
    col5.metric("⚡ Graph Connected", "Yes" if summary["is_connected"] else "Partial")

    st.markdown("---")

    # ---- Run analysis on button click ----
    if run_analysis and event_text.strip():
        with st.spinner("🔍 Parsing disruption event …"):
            disruption_info = parse_disruption(event_text.strip())
            if severity_override != "Auto-detect":
                disruption_info["severity"] = severity_override.lower()

        st.info(
            f"**Event parsed** | 🏙️ {len(disruption_info['affected_nodes'])} seed nodes identified "
            f"| ⚠️ Severity: **{disruption_info['severity'].upper()}** "
            f"| 📂 Category: **{disruption_info['category'].replace('_', ' ').title()}**"
        )

        with st.spinner("⚡ Simulating cascade propagation …"):
            cascade_result = run_cascade(G, disruption_info["affected_nodes"], max_depth)
            cascade_stats  = get_cascade_stats(cascade_result)

        with st.spinner("📊 Scoring risk nodes …"):
            try:
                delay_artifact = _load_delay_model()
            except Exception:
                delay_artifact = None
            risk_df = score_nodes(G, cascade_result, centrality, delay_artifact)

        with st.spinner("🔁 Finding alternate routes …"):
            reroute_suggestions = find_alternates(
                G, disruption_info["affected_nodes"], cascade_result, supply_df=supply_df
            )

        with st.spinner("🔍 Computing SHAP explainability …"):
            shap_results = {}
            shap_context = None
            if delay_artifact is not None:
                try:
                    shap_results = compute_shap(delay_artifact, risk_df, G, top_n=20)
                    if shap_results and not risk_df.empty:
                        top_node = risk_df.iloc[0]["node"]
                        top_name = risk_df.iloc[0]["city_name"]
                        if top_node in shap_results:
                            shap_context = shap_to_text(shap_results[top_node], top_name)
                except Exception as _shap_err:
                    st.warning(f"SHAP computation skipped: {_shap_err}")

        with st.spinner("🤖 Generating AI operations brief …"):
            brief = generate_brief(
                disruption_info, risk_df, reroute_suggestions,
                shap_context=shap_context,
            )

        st.session_state["results"] = {
            "disruption_info":     disruption_info,
            "cascade_result":      cascade_result,
            "cascade_stats":       cascade_stats,
            "risk_df":             risk_df,
            "reroute_suggestions": reroute_suggestions,
            "brief":               brief,
            "shap_results":        shap_results,
        }

    elif run_analysis and not event_text.strip():
        st.warning("Please enter a disruption description in the sidebar.")

    # ---- Display results ----
    results = st.session_state.get("results")

    if results is None:
        # Welcome screen
        st.markdown("""
        <div style="text-align:center; padding: 4rem 2rem; background: transparent;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🔗</div>
            <div style="font-size: 2rem; font-weight: 700; 
                        background: linear-gradient(135deg, #6366f1, #8b5cf6);
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                        margin-bottom: 0.8rem;">
                SupplAI — Supply Chain Disruption Monitor
            </div>
            <div style="color: #94a3b8; font-size: 1.1rem; max-width: 600px; margin: 0 auto 2rem;">
                Enter a disruption event in the sidebar and click <b style="color: #a5b4fc;">Analyse Disruption</b>
                to see real-time cascade simulation, risk scoring, rerouting suggestions,
                and an AI-generated operations brief.
            </div>
            <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap; margin-top: 2rem;">
                <div style="background: linear-gradient(135deg, #1e293b, #0f172a); border-radius: 16px; padding: 1.8rem 1.5rem; width: 190px; border: 1px solid #334155; text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.4); transition: all 0.2s ease;">
                    <div style="font-size: 2.2rem; margin-bottom: 0.6rem;">🌐</div>
                    <div style="font-weight: 700; font-size: 1rem; color: #e2e8f0; margin-bottom: 0.4rem;">Network Graph</div>
                    <div style="color: #94a3b8; font-size: 0.82rem; line-height: 1.4;">Interactive supply chain map</div>
                </div>
                <div style="background: linear-gradient(135deg, #1e293b, #0f172a); border-radius: 16px; padding: 1.8rem 1.5rem; width: 190px; border: 1px solid #334155; text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.4); transition: all 0.2s ease;">
                    <div style="font-size: 2.2rem; margin-bottom: 0.6rem;">🔥</div>
                    <div style="font-weight: 700; font-size: 1rem; color: #e2e8f0; margin-bottom: 0.4rem;">Risk Analysis</div>
                    <div style="color: #94a3b8; font-size: 0.82rem; line-height: 1.4;">ML-powered scoring</div>
                </div>
                <div style="background: linear-gradient(135deg, #1e293b, #0f172a); border-radius: 16px; padding: 1.8rem 1.5rem; width: 190px; border: 1px solid #334155; text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.4); transition: all 0.2s ease;">
                    <div style="font-size: 2.2rem; margin-bottom: 0.6rem;">🔁</div>
                    <div style="font-weight: 700; font-size: 1rem; color: #e2e8f0; margin-bottom: 0.4rem;">Rerouting</div>
                    <div style="color: #94a3b8; font-size: 0.82rem; line-height: 1.4;">Dijkstra alternate paths</div>
                </div>
                <div style="background: linear-gradient(135deg, #1e293b, #0f172a); border-radius: 16px; padding: 1.8rem 1.5rem; width: 190px; border: 1px solid #334155; text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.4); transition: all 0.2s ease;">
                    <div style="font-size: 2.2rem; margin-bottom: 0.6rem;">🤖</div>
                    <div style="font-weight: 700; font-size: 1rem; color: #e2e8f0; margin-bottom: 0.4rem;">AI Brief</div>
                    <div style="color: #94a3b8; font-size: 0.82rem; line-height: 1.4;">Gemini-powered insights</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ---- Unpack results ----
    disruption_info     = results["disruption_info"]
    cascade_result      = results["cascade_result"]
    cascade_stats       = results["cascade_stats"]
    risk_df             = results["risk_df"]
    reroute_suggestions = results["reroute_suggestions"]
    brief               = results["brief"]
    shap_results        = results.get("shap_results", {})

    # ---- Cascade stats banner ----
    sev = disruption_info["severity"].upper()
    badge_class = {"HIGH": "badge-high", "MEDIUM": "badge-medium", "LOW": "badge-low"}.get(sev, "badge-medium")

    st.markdown(f"""
    <div class="risk-card" style="display:flex; align-items:center; gap: 1.5rem; flex-wrap: wrap;">
        <span class="{badge_class}">{sev} SEVERITY</span>
        <span>📂 {disruption_info['category'].replace('_',' ').title()}</span>
        <span>🏙️ <b>{cascade_stats['seed_count']}</b> origin nodes</span>
        <span>⚡ <b>{cascade_stats['total_affected']}</b> total nodes in cascade</span>
        <span>📏 <b>{cascade_stats['max_depth']}</b> max cascade depth</span>
        <span>🌍 <b>{risk_df['country'].nunique() if not risk_df.empty else 0}</b> countries affected</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- 5 tabs ----
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌐 Network Graph",
        "🔥 Risk Analysis",
        "🔁 Rerouting",
        "🤖 AI Brief",
        "🔍 ML Explainability",
    ])

    # ==================================================================
    # TAB 1 — Network Graph
    # ==================================================================
    with tab1:
        st.markdown('<div class="section-header">Supply Chain Network</div>', unsafe_allow_html=True)
        st.markdown(
            f"Showing **{G.number_of_nodes()}** supply chain nodes. "
            f"**{cascade_stats['total_affected']}** nodes are impacted in this disruption cascade.",
            unsafe_allow_html=True,
        )

        fig = build_plotly_graph(
            G, cascade_result, risk_df, supply_df,
            seed_nodes=disruption_info["affected_nodes"],
        )
        st.plotly_chart(fig, use_container_width=True)

        # Depth breakdown
        st.markdown("#### Cascade Depth Breakdown")
        depth_data = cascade_stats.get("by_depth", {})
        if depth_data:
            depth_df = pd.DataFrame(
                [{"Cascade Depth": f"Depth {d}", "Nodes Affected": n}
                 for d, n in sorted(depth_data.items())]
            )
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.dataframe(depth_df, hide_index=True, use_container_width=True)
            with col_b:
                # Simple bar chart using plotly
                bar_fig = go.Figure(go.Bar(
                    x=[str(d) for d in sorted(depth_data.keys())],
                    y=[depth_data[d] for d in sorted(depth_data.keys())],
                    marker=dict(
                        color=["#ef4444", "#f97316", "#eab308", "#3b82f6", "#22c55e", "#6366f1"][:len(depth_data)],
                    ),
                    text=[depth_data[d] for d in sorted(depth_data.keys())],
                    textposition="outside",
                ))
                bar_fig.update_layout(
                    paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
                    xaxis=dict(title="Cascade Depth", color="#94a3b8", gridcolor="#1e293b"),
                    yaxis=dict(title="Nodes Affected", color="#94a3b8", gridcolor="#1e293b"),
                    font=dict(color="#e2e8f0", family="Inter"),
                    height=280, margin=dict(t=20, b=30, l=20, r=20),
                )
                st.plotly_chart(bar_fig, use_container_width=True)

    # ==================================================================
    # TAB 2 — Risk Analysis
    # ==================================================================
    with tab2:
        st.markdown('<div class="section-header">Risk Scoring Analysis</div>', unsafe_allow_html=True)

        if risk_df.empty:
            st.warning("No risk data — no disruption nodes matched the graph.")
        else:
            # Top 3 critical nodes as metric cards
            top3 = risk_df.head(3)
            cols = st.columns(3)
            for i, (_, row) in enumerate(top3.iterrows()):
                with cols[i]:
                    st.markdown(f"""
                    <div class="risk-card {'risk-card-critical' if row['risk_score']>=0.65 else 'risk-card-high' if row['risk_score']>=0.40 else 'risk-card-medium'}">
                        <div style="font-size:0.75rem;color:#64748b;text-transform:uppercase;letter-spacing:0.08em;">#{i+1} Highest Risk</div>
                        <div style="font-size:1.15rem;font-weight:700;margin:0.3rem 0">{row['city_name']}</div>
                        <div style="color:#94a3b8;font-size:0.85rem;">{row['country']} · {row['product_category']}</div>
                        <div style="margin-top:0.6rem;display:flex;align-items:center;gap:0.5rem;">
                            <span style="font-size:1.4rem;font-weight:700;color:{'#ef4444' if row['risk_score']>=0.65 else '#f97316' if row['risk_score']>=0.40 else '#eab308'}">{row['risk_score']:.3f}</span>
                            <span style="font-size:0.8rem;">{row['risk_level']}</span>
                        </div>
                        <div style="color:#64748b;font-size:0.75rem;margin-top:0.3rem;">{row['status']}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Full risk table
            tbl_col, export_col = st.columns([5, 1])
            with tbl_col:
                st.markdown("#### All Affected Nodes")
            with export_col:
                st.markdown("<div style='padding-top:1.6rem'></div>", unsafe_allow_html=True)
                display_cols = [
                    "node", "city_name", "country", "product_category",
                    "tier", "cascade_depth", "delay_prob",
                    "centrality_score", "risk_score", "risk_level", "status"
                ]
                available_cols = [c for c in display_cols if c in risk_df.columns]
                export_df = risk_df[available_cols].rename(columns={
                    "node": "Node ID", "city_name": "City", "country": "Country",
                    "product_category": "Product", "tier": "Tier",
                    "cascade_depth": "Depth", "delay_prob": "Delay Prob",
                    "centrality_score": "Centrality", "risk_score": "Risk Score",
                    "risk_level": "Risk Level", "status": "Status",
                })
                st.download_button(
                    label="⬇️ Export CSV",
                    data=export_df.to_csv(index=False),
                    file_name="supplai_risk_report.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            st.dataframe(
                export_df,
                hide_index=True,
                use_container_width=True,
                height=400,
            )

            # Risk score distribution
            st.markdown("#### Risk Score Distribution")
            # Build per-score color list for the histogram bars
            _score_colours = risk_df["risk_score"].apply(
                lambda s: "#ef4444" if s >= 0.65 else "#f97316" if s >= 0.40 else "#eab308" if s >= 0.20 else "#22c55e"
            ).tolist()
            risk_fig = go.Figure()
            risk_fig.add_trace(go.Histogram(
                x=risk_df["risk_score"],
                nbinsx=20,
                marker=dict(
                    color=risk_df["risk_score"].tolist(),
                    colorscale=[[0, "#22c55e"], [0.33, "#eab308"], [0.66, "#f97316"], [1, "#ef4444"]],
                    cmin=0, cmax=1,
                    showscale=False,
                ),
                name="Risk Scores",
            ))
            risk_fig.update_layout(
                paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
                xaxis=dict(title="Risk Score", color="#94a3b8", gridcolor="#1e293b"),
                yaxis=dict(title="Node Count", color="#94a3b8", gridcolor="#1e293b"),
                font=dict(color="#e2e8f0", family="Inter"),
                height=260, margin=dict(t=10, b=30, l=20, r=20),
                bargap=0.1,
            )
            st.plotly_chart(risk_fig, use_container_width=True)

    # ==================================================================
    # TAB 3 — Rerouting
    # ==================================================================
    with tab3:
        st.markdown('<div class="section-header">Alternate Route Recommendations</div>', unsafe_allow_html=True)

        if not reroute_suggestions:
            st.info("No rerouting suggestions — the disruption may be too isolated or no alternate paths exist.")
        else:
            n_found   = len([r for r in reroute_suggestions if r["status"] == "✅ Alternate Found"])
            n_blocked = len(reroute_suggestions) - n_found

            col_r1, col_r2, col_r3 = st.columns(3)
            col_r1.metric("🔄 Total Suggestions", len(reroute_suggestions))
            col_r2.metric("✅ Alternates Found",  n_found)
            col_r3.metric("⚠️ Blocked Routes",    n_blocked)

            st.markdown("<br>", unsafe_allow_html=True)

            for i, route in enumerate(reroute_suggestions):
                found   = route["status"] == "✅ Alternate Found"
                card_cls = "route-found" if found else "route-none"

                alt_path_str  = format_path(route.get("alternate_path", []), supply_df)
                orig_path_str = format_path(route.get("disrupted_path", []), supply_df)

                delta_str = ""
                if found and route.get("distance_delta_km") is not None:
                    delta = route["distance_delta_km"]
                    detour = route["detour_pct"]
                    sign  = "+" if delta >= 0 else ""
                    delta_str = f"<span style='color:#eab308'>{sign}{delta:,.0f} km ({sign}{detour:.1f}%)</span>"

                st.markdown(f"""
                <div class="route-card {card_cls}">
                    <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:0.5rem;">
                        <div>
                            <span style="font-size:0.75rem;color:#64748b;text-transform:uppercase;">Route {i+1}</span>
                            <div style="font-weight:700;font-size:1rem;margin:0.2rem 0">
                                📍 {route['source_name']} → 📍 {route['destination_name']}
                            </div>
                        </div>
                        <span style="font-size:0.9rem;">{route['status']}</span>
                    </div>
                    {'<div style="margin-top:0.8rem;padding:0.6rem;background:#0f172a;border-radius:8px;font-size:0.82rem;color:#94a3b8;"><b style=color:#ef4444>⚠️ Original:</b> ' + orig_path_str + '</div>' if orig_path_str and orig_path_str != 'No path' else ''}
                    {'<div style="margin-top:0.4rem;padding:0.6rem;background:#0f172a;border-radius:8px;font-size:0.82rem;color:#94a3b8;"><b style=color:#22c55e>✅ Alternate:</b> ' + alt_path_str + '</div>' if found and alt_path_str and alt_path_str != 'No path' else ''}
                    <div style="margin-top:0.6rem;display:flex;gap:1.5rem;font-size:0.82rem;color:#64748b;flex-wrap:wrap;">
                        <span>🛤️ Orig hops: <b>{route['hops_original']}</b></span>
                        <span>🛤️ Alt hops: <b>{route['hops_alternate']}</b></span>
                        {'<span>📏 Extra distance: <b>' + delta_str + '</b></span>' if delta_str else ''}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ==================================================================
    # TAB 4 — AI Brief
    # ==================================================================
    with tab4:
        st.markdown('<div class="section-header">AI Operations Brief</div>', unsafe_allow_html=True)

        source_label = (
            "🤖 Generated by Gemini 1.5 Flash"
            if brief.get("source", "template") != "template"
            else "📋 Template Brief (Add GEMINI_API_KEY to .env for AI generation)"
        )
        confidence = brief.get("confidence", "Medium")
        conf_colour = {"High": "#22c55e", "Medium": "#eab308", "Low": "#ef4444"}.get(confidence, "#eab308")

        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem; flex-wrap:wrap; gap:0.5rem;">
            <span style="color:#64748b;font-size:0.85rem;">{source_label}</span>
            <span style="color:{conf_colour};font-weight:600;background:#1e293b;padding:4px 12px;border-radius:8px;font-size:0.85rem;">
                Confidence: {confidence}
            </span>
        </div>
        """, unsafe_allow_html=True)

        # Executive Summary
        st.markdown(f"""
        <div class="brief-section">
            <div class="brief-title">📋 Executive Summary</div>
            <div class="brief-content">{brief.get('executive_summary', 'N/A')}</div>
        </div>
        """, unsafe_allow_html=True)

        # Two columns: top risks + actions
        bc1, bc2 = st.columns(2)

        with bc1:
            risks = brief.get("top_risks", [])
            risks_html = "".join(
                f"<div style='padding:0.4rem 0;border-bottom:1px solid #1e293b;color:#cbd5e1;font-size:0.9rem;'>"
                f"<span style='color:#ef4444;font-weight:600;'>{i+1}.</span> {r}</div>"
                for i, r in enumerate(risks)
            )
            st.markdown(f"""
            <div class="brief-section">
                <div class="brief-title">⚠️ Top Risks</div>
                {risks_html}
            </div>
            """, unsafe_allow_html=True)

        with bc2:
            actions = brief.get("immediate_actions", [])
            actions_html = "".join(
                f"<div style='padding:0.4rem 0;border-bottom:1px solid #1e293b;color:#cbd5e1;font-size:0.9rem;'>"
                f"<span style='color:#22c55e;font-weight:600;'>{i+1}.</span> {a}</div>"
                for i, a in enumerate(actions)
            )
            st.markdown(f"""
            <div class="brief-section">
                <div class="brief-title">🚀 Immediate Actions</div>
                {actions_html}
            </div>
            """, unsafe_allow_html=True)

        # Impact + Timeline
        ic1, ic2 = st.columns(2)
        with ic1:
            st.markdown(f"""
            <div class="brief-section">
                <div class="brief-title">💥 Estimated Impact</div>
                <div class="brief-content">{brief.get('estimated_impact', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)
        with ic2:
            st.markdown(f"""
            <div class="brief-section">
                <div class="brief-title">⏱️ Response Timeline</div>
                <div class="brief-content">{brief.get('timeline', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)

        # Export brief as JSON
        st.markdown("---")
        brief_json = json.dumps(brief, indent=2)
        st.download_button(
            label     = "⬇️ Download Brief as JSON",
            data      = brief_json,
            file_name = "operations_brief.json",
            mime      = "application/json",
        )

    # ==================================================================
    # TAB 5 — ML Explainability
    # ==================================================================
    with tab5:
        st.markdown('<div class="section-header">ML Explainability — SHAP Analysis</div>', unsafe_allow_html=True)

        if not shap_results:
            # SHAP not installed — show feature-importance placeholder using raw model info
            st.markdown("""
            <div class="brief-section" style="text-align:center; padding: 2.5rem;">
                <div style="font-size:2.5rem; margin-bottom:0.8rem;">🔍</div>
                <div style="font-size:1.1rem; font-weight:600; color:#e2e8f0; margin-bottom:0.5rem;">
                    SHAP Library Not Available
                </div>
                <div style="color:#64748b; font-size:0.9rem; max-width:500px; margin:0 auto; line-height:1.6;">
                    Install <code style="background:#0f172a;padding:2px 6px;border-radius:4px;color:#a5b4fc;">shap</code>
                    to enable full ML explainability with feature-level breakdowns.<br><br>
                    The model currently uses <b style="color:#a5b4fc;">13 features</b> to predict shipment delay probability.
                    Below is the feature glossary.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            shap_node_ids = list(shap_results.keys())

            # ── Intro banner ──────────────────────────────────────────────
            n_explained = len(shap_results)
            top_node_id = risk_df.iloc[0]["node"] if not risk_df.empty else shap_node_ids[0]
            top_city    = risk_df.iloc[0]["city_name"] if not risk_df.empty else top_node_id

            st.markdown(f"""
            <div class="risk-card" style="display:flex; align-items:center; gap:1.5rem; flex-wrap:wrap;">
                <span style="font-size:1.5rem;">🔍</span>
                <div>
                    <div style="font-weight:700; font-size:1rem; color:#e2e8f0;">
                        SHAP computed for <span style="color:#a5b4fc;">{n_explained} nodes</span>
                    </div>
                    <div style="color:#64748b; font-size:0.82rem; margin-top:0.2rem;">
                        Using XGBoost TreeExplainer — each bar shows how much a feature pushes the delay probability up or down.
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Global Feature Importance ─────────────────────────────────
            st.markdown("#### 📊 Global Feature Importance")
            st.markdown(
                "<span style='color:#64748b;font-size:0.85rem;'>"
                "Mean absolute SHAP value across all explained nodes — higher = more influential on delay prediction."
                "</span>",
                unsafe_allow_html=True,
            )
            bar_fig = shap_bar_figure(shap_results)
            st.plotly_chart(bar_fig, use_container_width=True)

            st.markdown("---")

            # ── Per-Node Waterfall ────────────────────────────────────────
            st.markdown("#### 🌊 Per-Node SHAP Waterfall")
            st.markdown(
                "<span style='color:#64748b;font-size:0.85rem;'>"
                "Select any at-risk node to see exactly which features are pushing its delay probability up "
                "<span style='color:#ef4444;'>▲ (red)</span> or down "
                "<span style='color:#22c55e;'>▼ (green)</span>."
                "</span>",
                unsafe_allow_html=True,
            )

            # Build display names for the dropdown
            node_display = {}
            for nid in shap_node_ids:
                row = risk_df[risk_df["node"] == nid]
                if not row.empty:
                    city    = row.iloc[0]["city_name"]
                    country = row.iloc[0]["country"]
                    score   = row.iloc[0]["risk_score"]
                    level   = row.iloc[0]["risk_level"]
                    node_display[nid] = f"{city}, {country} — Risk: {score:.3f} {level}"
                else:
                    node_display[nid] = nid

            selected_node = st.selectbox(
                "Select node to inspect:",
                options=shap_node_ids,
                format_func=lambda x: node_display.get(x, x),
                index=0,
                key="shap_node_selector",
            )

            sel_row = risk_df[risk_df["node"] == selected_node]
            sel_name = sel_row.iloc[0]["city_name"] if not sel_row.empty else selected_node

            waterfall_fig = shap_waterfall_figure(
                shap_results[selected_node],
                node_name=sel_name,
            )
            st.plotly_chart(waterfall_fig, use_container_width=True)

            # ── Key driver callout ────────────────────────────────────────
            node_shap = shap_results[selected_node]
            sorted_drivers = sorted(node_shap.items(), key=lambda x: abs(x[1]), reverse=True)
            top3 = sorted_drivers[:3]

            driver_cols = st.columns(3)
            for i, (feat, val) in enumerate(top3):
                label     = FEATURE_LABELS.get(feat, feat)
                direction = "↑ Increases Risk" if val > 0 else "↓ Reduces Risk"
                col_hex   = "#ef4444" if val > 0 else "#22c55e"
                with driver_cols[i]:
                    st.markdown(f"""
                    <div class="risk-card" style="text-align:center; padding:1rem;">
                        <div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.3rem;">Driver #{i+1}</div>
                        <div style="font-weight:700;font-size:0.95rem;color:#e2e8f0;margin-bottom:0.4rem;">{label}</div>
                        <div style="font-size:1.3rem;font-weight:700;color:{col_hex};">{val:+.4f}</div>
                        <div style="font-size:0.78rem;color:{col_hex};margin-top:0.2rem;">{direction}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")

            # ── Plain-English SHAP brief ──────────────────────────────────
            st.markdown("#### 🤖 SHAP Natural Language Summary")
            plain_text = shap_to_text(node_shap, sel_name, top_k=5)
            st.markdown(f"""
            <div class="brief-section">
                <div class="brief-title">Why <b style='color:#a5b4fc;'>{sel_name}</b> is at risk — feature breakdown</div>
                <div class="brief-content" style="font-family: monospace; font-size:0.88rem; line-height:1.9; white-space:pre-line;">{plain_text}</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Feature Glossary (always shown) ───────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📖 Feature Glossary — what each ML feature means", expanded=False):
            for feat, label in FEATURE_LABELS.items():
                desc = FEATURE_DESCRIPTIONS.get(feat, "")
                st.markdown(f"""
                <div style="padding:0.5rem 0; border-bottom:1px solid #1e293b;">
                    <span style="font-weight:600;color:#a5b4fc;">{label}</span>
                    <span style="color:#64748b; font-size:0.82rem;"> ({feat})</span><br>
                    <span style="color:#94a3b8;font-size:0.85rem;">{desc}</span>
                </div>
                """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
