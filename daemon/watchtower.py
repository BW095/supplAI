"""
watchtower.py
--------------
The autonomous background daemon that continuously monitors the global supply chain.

Runs in a background thread launched from the Streamlit dashboard.
Every scan cycle:
  1. Fetch fresh signals (news, weather, earthquakes)
  2. Run disruption engine on any new events
  3. Run risk engine to update node scores
  4. For each new disruption detected:
     a. Run Gemini agent to analyze and decide
      b. Generate advisory notifications for affected suppliers
      c. Persist decision to state files
  5. Update system state
"""

from __future__ import annotations
import sys
import json
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "daemon"))

from state_manager import (
    log, save_decision, save_notifications,
    update_system_state, load_system_state, init_state_files,
)
from graph_engine import build_graph, load_metadata, load_routes, get_graph_summary
from disruption_engine import process_disruption
from route_optimizer import find_alternates
from risk_engine import score_nodes, compute_centrality, load_anomaly_model, score_anomalies
from intelligence_feeds import fetch_all_signals
from gemini_agent import run_agent
from notification_engine import (
    generate_route_advisory,
    generate_risk_advisory,
    generate_situation_report,
    generate_delay_status_update,
    generate_emergency_procurement,   # repurposed as critical-node advisory
)

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass


# ──────────────────────────────────────────────
# Singleton daemon state
# ──────────────────────────────────────────────

_daemon_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()
_seen_events: set = set()  # de-duplicate signals across scans


# ──────────────────────────────────────────────
# Graph cache (loaded once)
# ──────────────────────────────────────────────

_G = None
_meta_df = None
_centrality = None
_anomaly_model = None
_shipments: List[Dict] = []


def _ensure_graph():
    global _G, _meta_df, _centrality, _anomaly_model, _shipments
    if _G is None:
        log("📊 Loading supply chain graph…", category="info")
        _meta_df = load_metadata()
        _G = build_graph(_meta_df, load_routes())
        _centrality = compute_centrality(_G)
        _anomaly_model = load_anomaly_model()

        summary = get_graph_summary(_G)
        update_system_state(
            network_nodes=summary["nodes"],
            network_edges=summary["edges"],
        )
        log(f"✅ Graph loaded: {summary['nodes']} nodes, {summary['edges']} edges, "
            f"{summary['countries']} countries", category="success")

    # Load shipments
    shipments_path = ROOT / "data" / "active_shipments.json"
    if shipments_path.exists():
        try:
            with open(shipments_path) as f:
                _shipments = json.load(f)
            update_system_state(active_shipments=len(_shipments))
        except Exception:
            pass


# ──────────────────────────────────────────────
# Single scan cycle
# ──────────────────────────────────────────────

def _run_scan(include_news: bool = True):
    """Execute one full monitoring scan cycle."""
    global _seen_events

    now_str = datetime.now(timezone.utc).strftime("%H:%M UTC")
    log(f"🔍 Scan started at {now_str}", category="scan")
    update_system_state(
        status="monitoring",
        last_scan_utc=datetime.now(timezone.utc).isoformat(),
        current_activity="Fetching intelligence signals…",
    )

    _ensure_graph()

    # 1. Fetch external signals
    log("📡 Fetching live intelligence signals…", category="scan")
    try:
        signals = fetch_all_signals(_meta_df, include_news=include_news)
    except Exception as e:
        log(f"⚠️ Signal fetch error: {e}", level="warning", category="warning")
        signals = []

    if not signals:
        log("✅ No disruptions detected — supply chain stable", category="success")
        update_system_state(status="monitoring", current_activity="All clear — monitoring", active_disruptions=0)
        return

    # 2. Process each signal
    new_disruptions = []
    for signal in signals:
        # Avoid re-processing same event
        fingerprint = f"{signal.get('event_text', '')}:{signal.get('severity', '')}"
        if fingerprint in _seen_events:
            continue
        _seen_events.add(fingerprint)

        # Cap seen events memory
        if len(_seen_events) > 200:
            _seen_events = set(list(_seen_events)[-100:])

        log(f"🚨 Disruption detected: {signal.get('title', signal.get('event_text', ''))[:80]}",
            level="warning", category="disruption")

        report = process_disruption(_G, signal, _shipments)

        if not report.seed_nodes:
            log(f"ℹ️ Signal could not be mapped to network nodes — skipping",
                category="info")
            continue

        new_disruptions.append(report)
        log(f"   🌐 Cascade: {len(report.cascade_nodes)} nodes, {len(report.affected_countries)} countries",
            category="disruption")
        log(f"   💰 Estimated impact: ${report.estimated_cost_usd:,.0f} | "
            f"⏱️ {report.estimated_delay_days:.1f} day delay | "
            f"📦 {report.shipments_at_risk} shipments at risk",
            category="disruption")

    update_system_state(active_disruptions=len(new_disruptions))

    # 3. For each credible disruption, run the AI agent
    for report in new_disruptions[:2]:  # limit to 2 per cycle to avoid rate limits
        log(f"🤖 Agent analyzing: {report.event_text[:60]}…", category="agent")
        update_system_state(current_activity=f"AI analyzing: {report.event_text[:50]}…")

        # Risk scoring
        risk_df = score_nodes(_G, report.cascade_nodes, severity=report.severity, centrality=_centrality)
        risk_df = score_anomalies(risk_df, _anomaly_model)

        # Route optimization
        log("🔄 Computing alternate routes…", category="route")
        routes = find_alternates(_G, report.cascade_nodes, top_k=8)
        found_routes = [r for r in routes if r.status == "found"]
        log(f"   ✅ {len(found_routes)}/{len(routes)} alternate routes found", category="route")

        # Run agent
        try:
            decision = run_agent(report, routes, risk_df, _shipments)
            log(f"🤖 Agent ({decision.agent_source}) completed in {decision.elapsed_seconds:.1f}s — "
                f"{len(decision.steps)} reasoning steps", category="agent")
            log(f"   📋 Risk level: {decision.risk_level} | "
                f"Recovery: {decision.estimated_recovery_days} days | "
                f"Route suggestions: {len(decision.suggested_alternatives or [])}",
                category="agent")

            save_decision(decision.to_dict())
        except Exception as e:
            log(f"❌ Agent error: {e}", level="error", category="error")
            continue

        # 4. Generate ADVISORY notifications (no autonomous rerouting)
        notifications = []
        product_cat = "Consumer Goods"
        if report.seed_nodes and _G.has_node(report.seed_nodes[0]):
            product_cat = _G.nodes[report.seed_nodes[0]].get("product_category", "Consumer Goods")

        delay_days = report.estimated_delay_days

        # Per-route advisory: what happened + suggested alternative
        for route in found_routes[:3]:
            try:
                disruption_reason = (
                    f"Your shipment corridor passes through {', '.join(report.seed_nodes[:2])}, "
                    f"which is directly affected by this event. {len(report.cascade_nodes)} "
                    f"supply chain nodes are in the cascade zone."
                )
                affected_ids = [
                    s.get("shipment_id", "?") for s in _shipments
                    if s.get("origin") in report.cascade_nodes
                    or s.get("destination") in report.cascade_nodes
                ][:5]
                notif = generate_route_advisory(
                    route_alt=route,
                    disruption_event_text=report.event_text,
                    disruption_reason=disruption_reason,
                    delay_days=delay_days,
                    affected_shipment_ids=affected_ids or ["N/A"],
                    product_category=product_cat,
                )
                notifications.append(notif)
                log(f"📬 Route advisory → {notif['recipient_name']} "
                    f"(suggested: {route.source_label} → {route.dest_label})",
                    category="notification")
            except Exception as e:
                log(f"⚠️ Advisory error: {e}", level="warning", category="warning")

        # Situation report (covers all alternatives at once)
        if found_routes:
            try:
                sitreport = generate_situation_report(
                    event_text=report.event_text,
                    cascade_node_count=len(report.cascade_nodes),
                    shipments_at_risk=report.shipments_at_risk,
                    estimated_delay_days=delay_days,
                    top_alternatives=[
                        {
                            "source": r.source_label, "destination": r.dest_label,
                            "path": " → ".join(r.alt_path_labels[:6]),
                            "cost_delta_usd": r.cost_delta_usd,
                            "delay_delta_days": r.delay_delta_days,
                            "safety": r.route_safety,
                        } for r in found_routes[:4]
                    ],
                    recovery_days=decision.estimated_recovery_days,
                    affected_sectors=report.affected_sectors,
                )
                notifications.append(sitreport)
                log("📡 Situation report dispatched to procurement managers", category="notification")
            except Exception as e:
                log(f"⚠️ Sit-rep error: {e}", level="warning", category="warning")

        # Broad risk advisory to all affected sectors
        if report.affected_sectors:
            try:
                alts_summary = "; ".join(
                    f"{r.source_label}→{r.dest_label} (+{r.delay_delta_days:.0f}d, ${r.cost_delta_usd:+,.0f})"
                    for r in found_routes[:3]
                ) or "Computing alternatives…"
                advs = generate_risk_advisory(
                    report.event_text,
                    report.affected_countries,
                    report.affected_sectors,
                    report.severity,
                    alternatives_summary=alts_summary,
                )
                notifications.extend(advs)
                for adv in advs:
                    log(f"📋 Risk advisory → {adv['recipient_name']}", category="notification")
            except Exception as e:
                log(f"⚠️ Advisory gen error: {e}", level="warning", category="warning")

        if notifications:
            save_notifications(notifications)
            log(f"✅ {len(notifications)} supplier advisory notifications dispatched", category="success")

    log(f"✅ Scan complete — {len(new_disruptions)} active disruption(s)", category="success")
    update_system_state(
        status="alert" if new_disruptions else "monitoring",
        current_activity=f"Monitoring — {len(new_disruptions)} active alert(s)" if new_disruptions else "All clear",
    )


# ──────────────────────────────────────────────
# Simulation trigger
# ──────────────────────────────────────────────

def trigger_simulation(scenario: Dict[str, Any]) -> None:
    """Inject a simulated disruption event immediately (called from UI)."""
    _ensure_graph()

    event_text = scenario.get("event_text", "Simulated disruption")
    log(f"🎯 SIMULATION TRIGGERED: {event_text}", level="warning", category="simulation")

    try:
        report = process_disruption(_G, scenario, _shipments, event_id=f"SIM-{int(time.time())}")

        if not report.seed_nodes:
            log("⚠️ Simulation: could not map to network nodes — using random hub", category="warning")
            import random
            from graph_engine import get_hub_cities
            hubs = get_hub_cities(_G)
            if hubs:
                scenario["affected_nodes"] = [random.choice(hubs)]
                report = process_disruption(_G, scenario, _shipments)

        log(f"🌐 Simulation cascade: {len(report.cascade_nodes)} nodes, "
            f"{report.shipments_at_risk} shipments at risk", category="simulation")

        risk_df = score_nodes(_G, report.cascade_nodes, severity=report.severity, centrality=_centrality)
        risk_df = score_anomalies(risk_df, _anomaly_model)

        routes = find_alternates(_G, report.cascade_nodes, top_k=8)
        found_routes = [r for r in routes if r.status == "found"]
        log(f"🔄 Simulation: {len(found_routes)} alternate routes found", category="route")

        log("🤖 Agent analyzing simulation…", category="agent")
        decision = run_agent(report, routes, risk_df, _shipments)
        decision.trigger_source = "simulation"

        log(f"🤖 Simulation decision ({decision.agent_source}): "
            f"{len(decision.suggested_alternatives or [])} route suggestions, "
            f"{len(decision.flagged_suppliers)} flagged, "
            f"{decision.risk_level} risk", category="agent")

        save_decision(decision.to_dict())

        # Advisory notifications for simulation
        notifications = []
        product_cat = "Consumer Goods"
        if report.seed_nodes and _G.has_node(report.seed_nodes[0]):
            product_cat = _G.nodes[report.seed_nodes[0]].get("product_category", "Consumer Goods")

        delay_days = report.estimated_delay_days

        for route in found_routes[:3]:
            try:
                affected_ids = [
                    s.get("shipment_id", "SIM-?") for s in _shipments
                    if s.get("origin") in report.cascade_nodes
                    or s.get("destination") in report.cascade_nodes
                ][:5]
                n = generate_route_advisory(
                    route_alt=route,
                    disruption_event_text=event_text,
                    disruption_reason=(
                        f"The simulated event affects {len(report.cascade_nodes)} nodes. "
                        f"Your shipment corridor through {', '.join(report.seed_nodes[:2])} is impacted."
                    ),
                    delay_days=delay_days,
                    affected_shipment_ids=affected_ids or ["SIM-001"],
                    product_category=product_cat,
                )
                notifications.append(n)
            except Exception:
                pass

        if found_routes:
            try:
                notifications.append(generate_situation_report(
                    event_text=event_text,
                    cascade_node_count=len(report.cascade_nodes),
                    shipments_at_risk=report.shipments_at_risk,
                    estimated_delay_days=delay_days,
                    top_alternatives=[
                        {
                            "source": r.source_label, "destination": r.dest_label,
                            "path": " → ".join(r.alt_path_labels[:6]),
                            "cost_delta_usd": r.cost_delta_usd,
                            "delay_delta_days": r.delay_delta_days,
                            "safety": r.route_safety,
                        } for r in found_routes[:4]
                    ],
                    recovery_days=decision.estimated_recovery_days,
                    affected_sectors=report.affected_sectors,
                ))
            except Exception:
                pass

        if report.affected_sectors:
            try:
                notifications.extend(generate_risk_advisory(
                    event_text, report.affected_countries,
                    report.affected_sectors, report.severity,
                    alternatives_summary="See situation report for full list of alternatives.",
                ))
            except Exception:
                pass

        if notifications:
            save_notifications(notifications)
            log(f"📬 {len(notifications)} advisory notifications dispatched to suppliers",
                category="notification")

        update_system_state(active_disruptions=1, status="alert",
                             current_activity=f"Simulation: {event_text[:50]}")

    except Exception as e:
        log(f"❌ Simulation error: {e}", level="error", category="error")
        import traceback; traceback.print_exc()


# ──────────────────────────────────────────────
# Daemon loop
# ──────────────────────────────────────────────

def _daemon_loop(scan_interval_seconds: int = 300):
    log("🚀 SupplAI Watchtower daemon started", category="success")
    update_system_state(watchtower_running=True, started_at=datetime.now(timezone.utc).isoformat())

    first_scan = True
    while not _stop_event.is_set():
        try:
            _run_scan(include_news=True)
        except Exception as e:
            log(f"❌ Scan cycle error: {e}", level="error", category="error")

        first_scan = False

        # Wait for next interval or until stopped
        for _ in range(scan_interval_seconds):
            if _stop_event.is_set():
                break
            time.sleep(1)

    log("🛑 Watchtower daemon stopped", category="info")
    update_system_state(watchtower_running=False, status="idle", current_activity="Stopped")


# ──────────────────────────────────────────────
# Public control functions
# ──────────────────────────────────────────────

def start_watchtower(scan_interval_minutes: int = 5) -> bool:
    """Start the background daemon thread. Returns True if started, False if already running."""
    global _daemon_thread, _stop_event

    if _daemon_thread and _daemon_thread.is_alive():
        return False

    _stop_event = threading.Event()
    _seen_events.clear()

    init_state_files()

    _daemon_thread = threading.Thread(
        target=_daemon_loop,
        args=(scan_interval_minutes * 60,),
        daemon=True,
        name="SupplAI-Watchtower",
    )
    _daemon_thread.start()
    return True


def stop_watchtower() -> None:
    """Signal the daemon to stop."""
    global _daemon_thread
    if _daemon_thread and _daemon_thread.is_alive():
        _stop_event.set()
        _daemon_thread.join(timeout=5)
    update_system_state(watchtower_running=False, status="idle")


def is_running() -> bool:
    return _daemon_thread is not None and _daemon_thread.is_alive()
