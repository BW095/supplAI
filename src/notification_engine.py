"""
notification_engine.py
-----------------------
Sends ADVISORY notifications to suppliers whose shipments are on affected
supply chains. The system does NOT reroute — it informs suppliers of:

  1. What happened and why their shipment is affected
  2. Current delay estimate for their specific shipment
  3. Alternative route OPTIONS they may consider
  4. Actions they can take

The supplier or their logistics partner makes the final decision.
"""

from __future__ import annotations
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / "state"
STATE_DIR.mkdir(exist_ok=True)
NOTIFICATION_FILE = STATE_DIR / "notifications.json"


# ──────────────────────────────────────────────
# Templates
# ──────────────────────────────────────────────

TEMPLATES = {
    "route_advisory": {
        "subject": "⚠️ Supply Chain Alert: Disruption Detected — Route Advisory Enclosed",
        "type_label": "Route Advisory",
        "icon": "📋",
        "color": "#6366f1",
    },
    "delay_status": {
        "subject": "🕐 Shipment Status Update: Delay Expected — Current Situation & Options",
        "type_label": "Delay Status Update",
        "icon": "🕐",
        "color": "#f59e0b",
    },
    "risk_advisory": {
        "subject": "🚨 Supply Chain Risk Advisory — Immediate Awareness Required",
        "type_label": "Risk Advisory",
        "icon": "🚨",
        "color": "#ef4444",
    },
    "situation_report": {
        "subject": "📡 Situation Report: Supply Chain Disruption Intelligence Briefing",
        "type_label": "Situation Report",
        "icon": "📡",
        "color": "#8b5cf6",
    },
}

# Representative supplier contacts per product category
SUPPLIER_CONTACTS = {
    "Electronics":        [("Samsung Electronics", "procurement@samsung-supply.com"),   ("Foxconn", "logistics@foxconn-scm.com")],
    "Semiconductors":     [("TSMC Logistics", "supply@tsmc-ops.com"),                   ("ASML", "logistics@asml.com")],
    "Automotive Parts":   [("Toyota Supply Chain", "sc@toyota-global.com"),             ("BMW AG", "logistics@bmw-supply.de")],
    "Pharmaceuticals":    [("Pfizer Global SC", "supply@pfizer-logistics.com"),         ("Roche", "procurement@roche.com")],
    "Textiles":           [("Inditex Sourcing", "sourcing@inditex.com"),                ("H&M Supply", "supply@hm-co.com")],
    "Oil & Gas Equipment":[("Shell Logistics", "ops@shell-supply.com"),                 ("Aramco", "logistics@aramco.com")],
    "Food & Agriculture": [("Cargill", "logistics@cargill.com"),                        ("Olam Group", "supply@olamgroup.com")],
    "Consumer Goods":     [("Unilever Supply Chain", "supply@unilever.com"),            ("P&G Logistics", "scm@pg.com")],
    "Transshipment":      [("Maersk Line", "ops@maersk.com"),                           ("MSC", "logistics@msc.com")],
    "Industrial Machinery":[("Siemens SC", "supply@siemens.com"),                      ("Caterpillar", "logistics@cat.com")],
    "Raw Materials":      [("BHP", "supply@bhp.com"),                                   ("Rio Tinto", "logistics@riotinto.com")],
    "Luxury Goods":       [("LVMH", "supply@lvmh.com"),                                 ("Kering", "logistics@kering.com")],
    "Aerospace Components":[("Boeing SC", "supply@boeing-sc.com"),                     ("Airbus", "logistics@airbus.com")],
    "Chemicals":          [("BASF Logistics", "supply@basf.com"),                       ("Dow Chemical", "logistics@dow.com")],
    "Default":            [("Global Logistics Partner", "ops@global-supply.com")],
}

CARRIER_CONTACTS = {
    "Maersk":          "customer-ops@maersk.com",
    "MSC":             "freight@msc.com",
    "CMA CGM":         "logistics@cma-cgm.com",
    "COSCO":           "ops@cosco-ship.com",
    "Hapag-Lloyd":     "track@hapag-lloyd.com",
    "Evergreen":       "ops@evergreen-line.com",
    "DHL Supply Chain":"support@dhl-supply.com",
    "DB Schenker":     "ops@dbschenker.com",
    "FedEx Freight":   "freight@fedex-sc.com",
    "Yang Ming":       "ops@yangming.com",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _get_suppliers(product_category: str) -> List[tuple]:
    return SUPPLIER_CONTACTS.get(product_category,
           SUPPLIER_CONTACTS.get(product_category.split(" ")[0],
           SUPPLIER_CONTACTS["Default"]))


# ──────────────────────────────────────────────
# Core notification generators
# ──────────────────────────────────────────────

def generate_route_advisory(
    route_alt,               # RouteAlternative dataclass
    disruption_event_text: str,
    disruption_reason: str,  # human-readable cause
    delay_days: float,
    affected_shipment_ids: List[str],
    product_category: str = "Consumer Goods",
) -> Dict[str, Any]:
    """
    Advisory to supplier informing them of disruption + SUGGESTED alternate routes.
    The supplier decides whether to use the suggested route.
    """
    tmpl = TEMPLATES["route_advisory"]
    suppliers = _get_suppliers(product_category)
    supplier_name, supplier_email = suppliers[0]
    n_ships = len(affected_shipment_ids)
    ship_list = ", ".join(affected_shipment_ids[:5])
    if len(affected_shipment_ids) > 5:
        ship_list += f" (+{len(affected_shipment_ids)-5} more)"

    alt_path = " → ".join(route_alt.alt_path_labels[:7]) if route_alt.alt_path_labels else "Alternative routing available"
    cost_str  = f"+${route_alt.cost_delta_usd:,.0f}" if route_alt.cost_delta_usd >= 0 else f"-${abs(route_alt.cost_delta_usd):,.0f}"
    time_str  = f"+{route_alt.delay_delta_days:.1f} days" if route_alt.delay_delta_days >= 0 else f"{route_alt.delay_delta_days:.1f} days"

    body = f"""Dear {supplier_name} Procurement Team,

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SUPPLY CHAIN DISRUPTION ALERT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

We are writing to inform you of a disruption affecting your current supply route that requires your immediate attention.

WHAT HAPPENED
─────────────
{disruption_event_text}

WHY THIS AFFECTS YOUR SHIPMENT(S)
───────────────────────────────────
{disruption_reason}
Your {n_ships} active shipment(s) on this corridor are currently impacted:
  Affected Shipments: {ship_list}

CURRENT STATUS
──────────────
• Current Route: {route_alt.source_label} → {route_alt.dest_label}
• Status: ⚠️  DISRUPTED — Transit currently suspended/delayed
• Expected Delay on Current Route: +{delay_days:.1f} days

ALTERNATIVE ROUTE SUGGESTION (YOUR DECISION)
──────────────────────────────────────────────
Our routing intelligence has identified the following alternative:

  Suggested Path: {alt_path}
  Additional Distance: {route_alt.detour_pct:+.1f}%
  Estimated Cost Difference: {cost_str}
  Transit Time Difference: {time_str}
  Route Safety Assessment: {route_alt.route_safety.replace('_', ' ').title()}

⚠️  IMPORTANT: This is a RECOMMENDATION only. Your contract terms, agreed
rates, and operational constraints should guide your decision. Please
coordinate with your freight forwarder before making any routing changes.

RECOMMENDED ACTIONS FOR YOUR TEAM
───────────────────────────────────
1. Review your contract terms with your carrier for disruption clauses
2. Contact your freight forwarder to discuss rerouting feasibility
3. Assess your inventory buffer — additional {delay_days:.0f} days may be needed
4. Notify your production/procurement team of possible delay
5. Document this disruption for insurance and compliance purposes

We will continue monitoring this situation and provide updates every 2 hours.

SupplAI Autonomous Watchtower
Powered by Gemini AI | {_now_iso()[:16].replace("T", " ")} UTC
"""

    return _build_notification(
        ntype="route_advisory",
        subject=tmpl["subject"],
        type_label=tmpl["type_label"],
        icon=tmpl["icon"],
        color=tmpl["color"],
        recipient_name=supplier_name,
        recipient_email=supplier_email,
        body=body,
        severity="high",
        metadata={
            "source": route_alt.source_label,
            "destination": route_alt.dest_label,
            "suggested_path": alt_path,
            "cost_delta_usd": route_alt.cost_delta_usd,
            "detour_pct": route_alt.detour_pct,
            "delay_days": delay_days,
            "affected_shipments": n_ships,
        },
    )


def generate_delay_status_update(
    shipment_id: str,
    origin: str,
    destination: str,
    carrier: str,
    cargo_type: str,
    delay_days: float,
    disruption_cause: str,
    disruption_detail: str,
    current_position: str = "",
    progress_pct: float = 0,
    product_category: str = "Consumer Goods",
) -> Dict[str, Any]:
    """
    Status update to supplier about their specific delayed shipment.
    Explains what's happening, current delay, and what to expect.
    """
    tmpl = TEMPLATES["delay_status"]
    suppliers = _get_suppliers(product_category)
    supplier_name, supplier_email = suppliers[0]
    carrier_email = CARRIER_CONTACTS.get(carrier, "ops@carrier.com")

    severity = "high" if delay_days > 7 else "medium" if delay_days > 3 else "low"
    urgency  = "URGENT" if delay_days > 7 else "ATTENTION REQUIRED"
    position_str = f"Last known position: {current_position}\n  " if current_position else ""

    body = f"""Dear {supplier_name},

{urgency}: YOUR SHIPMENT IS EXPERIENCING A DELAY
{"━" * 50}

SHIPMENT DETAILS
────────────────
  Shipment ID:    {shipment_id}
  Route:          {origin} → {destination}
  Carrier:        {carrier} ({carrier_email})
  Cargo:          {cargo_type}
  Progress:       {progress_pct:.0f}% complete
  {position_str}

WHAT IS HAPPENING
──────────────────
Cause: {disruption_cause}

{disruption_detail}

DELAY IMPACT ON YOUR SHIPMENT
──────────────────────────────
  Original ETA:    As originally agreed
  Expected Delay:  +{delay_days:.1f} days
  Revised ETA:     Approximately {delay_days:.0f} additional days from today

This delay is beyond our control and driven by external disruption events.
Your goods remain safe and in transit — this is a timing impact only.

WHAT YOU SHOULD DO NOW
───────────────────────
1. Notify your downstream customers/production lines of the revised ETA
2. Contact {carrier} ({carrier_email}) for real-time vessel/truck tracking
3. Check if your shipment qualifies for force majeure under your contract
4. Consider emergency air freight for critical items if delay is unacceptable
{f"5. Buffer stock recommendation: {int(delay_days)} additional days of {cargo_type} inventory" if delay_days > 3 else ""}

NEXT UPDATE
───────────
We will send another status update in 4 hours or sooner if the situation changes.

SupplAI Autonomous Watchtower
Real-time Supply Chain Intelligence | {_now_iso()[:16].replace("T", " ")} UTC
"""

    return _build_notification(
        ntype="delay_status",
        subject=tmpl["subject"],
        type_label=tmpl["type_label"],
        icon=tmpl["icon"],
        color=tmpl["color"],
        recipient_name=supplier_name,
        recipient_email=supplier_email,
        body=body,
        severity=severity,
        metadata={
            "shipment_id": shipment_id,
            "origin": origin,
            "destination": destination,
            "carrier": carrier,
            "delay_days": delay_days,
            "progress_pct": progress_pct,
        },
    )


def generate_risk_advisory(
    event_text: str,
    affected_countries: List[str],
    affected_sectors: List[str],
    severity: str,
    alternatives_summary: str = "",
) -> List[Dict[str, Any]]:
    """
    Broad risk advisory to all suppliers in affected sectors.
    Informs them of the situation and available route options.
    """
    tmpl = TEMPLATES["risk_advisory"]
    notifications = []
    country_str = ", ".join(affected_countries[:5])
    if len(affected_countries) > 5:
        country_str += f" and {len(affected_countries)-5} more"

    for sector in affected_sectors[:3]:
        suppliers = _get_suppliers(sector)
        for supplier_name, supplier_email in suppliers[:1]:
            body = f"""Dear {supplier_name},

SUPPLY CHAIN RISK ADVISORY — {severity.upper()} SEVERITY
{"━" * 50}

The SupplAI Autonomous Watchtower has detected a significant disruption
that may impact your supply chain operations.

SITUATION SUMMARY
──────────────────
{event_text}

REGIONS AFFECTED: {country_str}
YOUR SECTOR:      {sector}
RISK LEVEL:       {severity.upper()}

HOW THIS MAY AFFECT YOU
─────────────────────────
Suppliers and manufacturers sourcing from or shipping through the affected
regions may experience:
  • Transit delays of 3-14 days on affected corridors
  • Increased freight rates due to reduced capacity
  • Port congestion on alternative routes
  • Potential documentation/customs delays

{f"ALTERNATIVE ROUTES AVAILABLE{chr(10)}──────────────────────────────{chr(10)}{alternatives_summary}{chr(10)}" if alternatives_summary else ""}
RECOMMENDED PRECAUTIONARY ACTIONS
────────────────────────────────────
1. Review your inventory levels for {sector} items — build 2-3 week buffer
2. Identify backup suppliers outside the affected region as contingency
3. Contact your freight forwarder to assess your specific shipment risk
4. Review your supply contracts for force majeure / disruption clauses
5. Document this advisory for insurance and compliance records
6. Monitor https://supplai.watchtower for real-time updates

This is an advisory only. No action on your part is mandatory.
Your logistics team should assess applicability to your specific contracts.

SupplAI Autonomous Watchtower
{_now_iso()[:16].replace("T", " ")} UTC | Gemini AI-Powered Intelligence
"""

            notifications.append(_build_notification(
                ntype="risk_advisory",
                subject=tmpl["subject"],
                type_label=tmpl["type_label"],
                icon=tmpl["icon"],
                color=tmpl["color"],
                recipient_name=supplier_name,
                recipient_email=supplier_email,
                body=body,
                severity=severity,
                metadata={
                    "event_text": event_text,
                    "affected_countries": affected_countries,
                    "sector": sector,
                },
            ))
    return notifications


def generate_situation_report(
    event_text: str,
    cascade_node_count: int,
    shipments_at_risk: int,
    estimated_delay_days: float,
    top_alternatives: List[Dict],
    recovery_days: int,
    affected_sectors: List[str],
) -> Dict[str, Any]:
    """
    High-level situation report for procurement/logistics managers.
    Summary of disruption + all available alternative options.
    """
    tmpl = TEMPLATES["situation_report"]
    alt_lines = ""
    for i, alt in enumerate(top_alternatives[:4], 1):
        alt_lines += (
            f"  Option {i}: {alt.get('source','')} → {alt.get('destination','')}\n"
            f"           Path: {alt.get('path','N/A')}\n"
            f"           Cost impact: ${alt.get('cost_delta_usd',0):+,.0f} | "
            f"Time impact: {alt.get('delay_delta_days',0):+.1f} days\n"
            f"           Safety: {alt.get('safety','unknown').replace('_',' ').title()}\n\n"
        )
    if not alt_lines:
        alt_lines = "  No alternative routes computed yet. Monitoring ongoing.\n"

    sectors_str = ", ".join(affected_sectors[:4]) if affected_sectors else "Multiple sectors"

    body = f"""SITUATION REPORT — SUPPLY CHAIN DISRUPTION INTELLIGENCE BRIEFING
{"━" * 60}
Generated: {_now_iso()[:16].replace("T", " ")} UTC
Watchtower: SupplAI Autonomous System (Gemini AI)

INCIDENT OVERVIEW
──────────────────
{event_text}

IMPACT ASSESSMENT
──────────────────
  Nodes in cascade:     {cascade_node_count}
  Shipments at risk:    {shipments_at_risk}
  Sectors affected:     {sectors_str}
  Expected delay:       +{estimated_delay_days:.1f} days on disrupted corridor
  Estimated recovery:   {recovery_days} days

AVAILABLE ROUTE ALTERNATIVES (FOR YOUR CONSIDERATION)
──────────────────────────────────────────────────────
These alternatives have been computed by our routing intelligence.
They are SUGGESTIONS — your logistics team and carriers must validate
feasibility, check contract terms, and approve any actual route changes.

{alt_lines}
KEY TAKEAWAY
────────────
• Your current shipments on the affected corridor will experience delays
• Alternatives exist but carry additional cost and/or time
• No automatic rerouting has been performed — your decision is required
• Contact your freight forwarder with these options for rate quotations

SupplAI Autonomous Watchtower | Gemini 2.5 Flash Intelligence
"""

    # Send to transshipment + all affected sectors' main contact
    all_sectors   = (affected_sectors or ["Default"])[:1]
    suppliers     = _get_suppliers(all_sectors[0])
    supplier_name, supplier_email = suppliers[0]

    return _build_notification(
        ntype="situation_report",
        subject=tmpl["subject"],
        type_label=tmpl["type_label"],
        icon=tmpl["icon"],
        color=tmpl["color"],
        recipient_name=supplier_name,
        recipient_email=supplier_email,
        body=body,
        severity="high",
        metadata={
            "cascade_nodes": cascade_node_count,
            "shipments_at_risk": shipments_at_risk,
            "recovery_days": recovery_days,
            "n_alternatives": len(top_alternatives),
        },
    )


# ──────────────────────────────────────────────
# Builder helper
# ──────────────────────────────────────────────

def _build_notification(
    ntype: str, subject: str, type_label: str, icon: str, color: str,
    recipient_name: str, recipient_email: str, body: str,
    severity: str, metadata: Dict,
) -> Dict[str, Any]:
    return {
        "notification_id": f"NOTIF-{uuid.uuid4().hex[:8].upper()}",
        "type": ntype,
        "type_label": type_label,
        "icon": icon,
        "color": color,
        "subject": subject,
        "recipient_name": recipient_name,
        "recipient_email": recipient_email,
        "body": body,
        "severity": severity,
        "status": "sent",
        "sent_at": _now_iso(),
        "metadata": metadata,
    }


# Backward-compat alias (old code may call this)
def generate_route_change_notification(route_alt, disruption_event_text, product_category="Consumer Goods"):
    return generate_route_advisory(
        route_alt=route_alt,
        disruption_event_text=disruption_event_text,
        disruption_reason="This corridor is part of the affected supply chain network.",
        delay_days=max(0, route_alt.delay_delta_days) if hasattr(route_alt, "delay_delta_days") else 5,
        affected_shipment_ids=[],
        product_category=product_category,
    )

def generate_emergency_procurement(node_label, product_category, reason):
    # Repurposed as a risk advisory for that specific node
    suppliers = _get_suppliers(product_category)
    supplier_name, supplier_email = suppliers[-1] if len(suppliers) > 1 else suppliers[0]
    return _build_notification(
        ntype="risk_advisory",
        subject=f"🚨 Supply Risk Alert: {node_label} — {product_category} Sourcing Review Required",
        type_label="Critical Node Alert",
        icon="🚨",
        color="#ef4444",
        recipient_name=supplier_name,
        recipient_email=supplier_email,
        body=(
            f"Dear {supplier_name},\n\n"
            f"CRITICAL SUPPLY NODE ALERT\n{'─'*40}\n\n"
            f"Node:    {node_label}\n"
            f"Product: {product_category}\n"
            f"Reason:  {reason}\n\n"
            f"RECOMMENDED ACTION\n{'─'*40}\n"
            f"Please review your sourcing dependency on {node_label} for {product_category}.\n"
            f"Consider identifying backup suppliers or increasing safety stock\n"
            f"as a precautionary measure while this situation is being monitored.\n\n"
            f"No action is mandatory — this is an advisory alert only.\n\n"
            f"SupplAI Autonomous Watchtower | {_now_iso()[:16].replace('T',' ')} UTC"
        ),
        severity="high",
        metadata={"node": node_label, "product_category": product_category, "reason": reason},
    )


# ──────────────────────────────────────────────
# State persistence
# ──────────────────────────────────────────────

def load_notifications() -> List[Dict]:
    if NOTIFICATION_FILE.exists():
        try:
            with open(NOTIFICATION_FILE) as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_notification(notification: Dict) -> None:
    existing = load_notifications()
    existing.insert(0, notification)
    existing = existing[:100]
    with open(NOTIFICATION_FILE, "w") as f:
        json.dump(existing, f, indent=2)

def save_notifications(notifications: List[Dict]) -> None:
    for n in notifications:
        save_notification(n)
