"""
gemini_agent.py
----------------
Autonomous supply chain AI ADVISORY agent.
Primary: Gemini 2.5 Flash with native function calling
Fallback: Groq (llama-3.3-70b) → deterministic logic

The agent is an INTELLIGENCE & ADVISORY system — it does NOT execute
route changes. It assesses disruptions and provides structured recommendations
that suppliers/logistics teams can act upon.

Reasoning loop:
  1. assess_disruptions   → scope assessment
  2. score_route_options  → rank suggested alternatives
  3. suggest_reroute (×N) → recommend best routes (supplier decides)
  4. flag_critical_supplier → escalate attention
  5. estimate_recovery      → timeline
  6. notify_suppliers       → dispatch advisory alerts
  7. finalize_plan          → structured recommendation output
"""

from __future__ import annotations
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass


def _resolve(names: tuple) -> str:
    for name in names:
        v = os.getenv(name, "").strip()
        if v and not v.lower().startswith("your_"):
            return v
    return ""


GEMINI_KEY = lambda: _resolve(("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY"))
GROQ_KEY   = lambda: _resolve(("GROQ_API_KEY",))


# ──────────────────────────────────────────────
# Agent Decision dataclass
# ──────────────────────────────────────────────

@dataclass
class AgentStep:
    step: int
    tool: str
    thought: str          # why the agent called this tool
    args: Dict[str, Any]
    result: Dict[str, Any]
    duration_ms: int = 0


@dataclass
class AgentDecision:
    decision_id: str
    event_id: str
    event_text: str
    severity: str
    trigger_source: str    # "news" | "weather" | "simulation" | "earthquake"
    steps: List[AgentStep]
    suggested_alternatives: List[Dict]   # renamed from approved_reroutes
    flagged_suppliers: List[Dict]
    final_summary: str
    priority_actions: List[str]
    risk_level: str
    estimated_recovery_days: int
    elapsed_seconds: float
    agent_source: str      # "gemini" | "groq" | "deterministic"
    timestamp: str
    notifications_sent: int = 0

    # backward-compat alias
    @property
    def approved_reroutes(self) -> List[Dict]:
        return self.suggested_alternatives

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["steps"] = [asdict(s) for s in self.steps]
        return d


# ──────────────────────────────────────────────
# Agent class
# ──────────────────────────────────────────────

class SupplyChainAgent:
    def __init__(
        self,
        disruption_report,       # DisruptionReport
        route_alternatives: list, # List[RouteAlternative]
        risk_df,                  # pd.DataFrame
        shipments: list,          # List[Dict]
    ):
        self.report  = disruption_report
        self.routes  = route_alternatives
        self.risk_df = risk_df
        self.shipments = shipments

        # Mutable state
        self.suggested_alternatives: List[Dict] = []  # route suggestions (not executions)
        self.flagged_suppliers: List[Dict] = []
        self.steps: List[AgentStep] = []
        self._step_n = 0
        self.final_plan: Optional[Dict] = None

    # ──────────────────────────────────────
    # Tool implementations (pure Python)
    # ──────────────────────────────────────

    def _assess_disruptions(self) -> Dict:
        r = self.report
        top_nodes = []
        if not self.risk_df.empty:
            for _, row in self.risk_df.head(5).iterrows():
                top_nodes.append({
                    "node": row["node"],
                    "city": row["city_name"],
                    "country": row["country"],
                    "risk_score": round(float(row["risk_score"]), 4),
                    "risk_level": row["risk_level"],
                    "product": row["product_category"],
                    "tier": int(row["tier"]),
                })
        return {
            "event": r.event_text,
            "severity": r.severity,
            "seed_nodes": r.seed_nodes,
            "total_cascade_nodes": len(r.cascade_nodes),
            "countries_affected": len(r.affected_countries),
            "shipments_at_risk": r.shipments_at_risk,
            "estimated_cost_usd": r.estimated_cost_usd,
            "top_risk_nodes": top_nodes,
        }

    def _score_route_alternatives(self) -> Dict:
        found = [r for r in self.routes if r.status == "found"]
        return {
            "total_alternatives_found": len(found),
            "total_analyzed": len(self.routes),
            "best_routes": [
                {
                    "source": r.source_label,
                    "destination": r.dest_label,
                    "detour_pct": r.detour_pct,
                    "cost_delta_usd": r.cost_delta_usd,
                    "delay_delta_days": r.delay_delta_days,
                    "safety": r.route_safety,
                    "path": " → ".join(r.alt_path_labels[:6]),
                    "score": round(r.score, 2),
                }
                for r in found[:5]
            ],
        }

    def _suggest_reroute(self, source: str, destination: str, reason: str) -> Dict:
        """Adds a route SUGGESTION — not an execution. Supplier decides."""
        self.suggested_alternatives.append({
            "source": source,
            "destination": destination,
            "reason": reason,
            "suggested_at_step": self._step_n,
        })
        return {
            "status": "suggestion_logged",
            "message": f"Alternative route {source} → {destination} added to advisory report for supplier review"
        }

    # backward-compat
    def _approve_reroute(self, source: str, destination: str, reason: str) -> Dict:
        return self._suggest_reroute(source, destination, reason)

    def _flag_critical_supplier(self, node_id: str, reason: str, priority: str = "high") -> Dict:
        # Resolve city name
        city = node_id
        if not self.risk_df.empty:
            row = self.risk_df[self.risk_df["node"] == node_id]
            if not row.empty:
                city = f"{row.iloc[0]['city_name']}, {row.iloc[0]['country']}"
        self.flagged_suppliers.append({"node": node_id, "city": city, "reason": reason, "priority": priority})
        return {"status": "flagged", "node": node_id, "city": city}

    def _estimate_recovery(self, risk_level: str, n_nodes: int) -> Dict:
        base = {"Critical": 18, "High": 10, "Medium": 5, "Low": 2}.get(risk_level, 7)
        days = base + max(0, (n_nodes - 5) // 5)
        return {"estimated_recovery_days": days, "risk_level": risk_level, "n_nodes_affected": n_nodes}

    def _notify_suppliers(self, recipient_group: str, message: str) -> Dict:
        return {"status": "dispatched", "recipient_group": recipient_group, "message_preview": message[:80]}

    def _finalize_plan(self, summary: str, priority_actions: List[str],
                        estimated_recovery_days: int, risk_level: str) -> Dict:
        self.final_plan = {
            "summary": summary,
            "priority_actions": priority_actions,
            "estimated_recovery_days": estimated_recovery_days,
            "risk_level": risk_level,
            "suggested_alternatives": self.suggested_alternatives,
            "flagged_suppliers": self.flagged_suppliers,
        }
        return {"status": "advisory_plan_finalized"}

    def _dispatch(self, name: str, args: dict) -> Dict:
        handlers = {
            "assess_disruptions":      lambda: self._assess_disruptions(),
            "score_route_alternatives": lambda: self._score_route_alternatives(),
            "score_route_options":     lambda: self._score_route_alternatives(),  # alias
            "suggest_reroute":         lambda: self._suggest_reroute(**args),
            "approve_reroute":         lambda: self._suggest_reroute(**args),     # backward compat
            "flag_critical_supplier":  lambda: self._flag_critical_supplier(**args),
            "estimate_recovery":       lambda: self._estimate_recovery(**args),
            "notify_suppliers":        lambda: self._notify_suppliers(**args),
            "finalize_plan":           lambda: self._finalize_plan(**args),
        }
        fn = handlers.get(name)
        return fn() if fn else {"error": f"Unknown tool: {name}"}

    def _log_step(self, tool: str, thought: str, args: dict, result: dict, duration_ms: int = 0):
        self._step_n += 1
        self.steps.append(AgentStep(
            step=self._step_n, tool=tool, thought=thought,
            args=args, result=result, duration_ms=duration_ms,
        ))

    # ──────────────────────────────────────
    # Gemini native function-calling loop
    # ──────────────────────────────────────

    def _run_gemini(self, max_turns: int = 14) -> str:
        """Returns 'success' or raises on failure."""
        key = GEMINI_KEY()
        if not key:
            raise ValueError("No Gemini key")

        from google import genai
        from google.genai import types as gt

        client = genai.Client(api_key=key)

        tools_schema = [
            gt.Tool(function_declarations=[
                gt.FunctionDeclaration(
                    name="assess_disruptions",
                    description="Get full scope: affected nodes, countries, shipments, cost estimate.",
                    parameters=gt.Schema(type="OBJECT", properties={}),
                ),
                gt.FunctionDeclaration(
                    name="score_route_alternatives",
                    description="Retrieve and rank all pre-computed rerouting alternatives.",
                    parameters=gt.Schema(type="OBJECT", properties={}),
                ),
                gt.FunctionDeclaration(
                    name="approve_reroute",
                    description="Approve and activate an alternate supply route.",
                    parameters=gt.Schema(
                        type="OBJECT",
                        properties={
                            "source":      gt.Schema(type="STRING", description="Origin city label"),
                            "destination": gt.Schema(type="STRING", description="Destination city label"),
                            "reason":      gt.Schema(type="STRING", description="Business justification citing data"),
                        },
                        required=["source", "destination", "reason"],
                    ),
                ),
                gt.FunctionDeclaration(
                    name="flag_critical_supplier",
                    description="Flag a supply node as critical for emergency backup sourcing.",
                    parameters=gt.Schema(
                        type="OBJECT",
                        properties={
                            "node_id":  gt.Schema(type="STRING", description="Node ID e.g. SHA"),
                            "reason":   gt.Schema(type="STRING"),
                            "priority": gt.Schema(type="STRING", description="high|medium|low"),
                        },
                        required=["node_id", "reason"],
                    ),
                ),
                gt.FunctionDeclaration(
                    name="estimate_recovery",
                    description="Estimate supply chain recovery timeline.",
                    parameters=gt.Schema(
                        type="OBJECT",
                        properties={
                            "risk_level": gt.Schema(type="STRING", description="Critical|High|Medium|Low"),
                            "n_nodes":    gt.Schema(type="INTEGER", description="Number of cascade-affected nodes"),
                        },
                        required=["risk_level", "n_nodes"],
                    ),
                ),
                gt.FunctionDeclaration(
                    name="notify_suppliers",
                    description="Dispatch notifications to affected supplier groups.",
                    parameters=gt.Schema(
                        type="OBJECT",
                        properties={
                            "recipient_group": gt.Schema(type="STRING", description="e.g. 'Electronics suppliers in China'"),
                            "message":         gt.Schema(type="STRING", description="Notification message to send"),
                        },
                        required=["recipient_group", "message"],
                    ),
                ),
                gt.FunctionDeclaration(
                    name="finalize_plan",
                    description="Commit the final structured action plan.",
                    parameters=gt.Schema(
                        type="OBJECT",
                        properties={
                            "summary": gt.Schema(type="STRING", description="2-3 sentence executive summary citing specific cities and sectors"),
                            "priority_actions": gt.Schema(
                                type="ARRAY",
                                items=gt.Schema(type="STRING"),
                                description="4-6 specific, quantified actions",
                            ),
                            "estimated_recovery_days": gt.Schema(type="INTEGER"),
                            "risk_level": gt.Schema(type="STRING", description="Critical|High|Medium|Low"),
                        },
                        required=["summary", "priority_actions", "estimated_recovery_days", "risk_level"],
                    ),
                ),
            ])
        ]

        system_prompt = (
            f"You are an autonomous Chief Supply Chain Intelligence Officer AI.\n"
            f"A disruption has been detected: {self.report.event_text}\n"
            f"Severity: {self.report.severity.upper()} | Cascade nodes: {len(self.report.cascade_nodes)}\n\n"
            f"IMPORTANT: You are an ADVISORY system. You do NOT execute route changes.\n"
            f"Your role is to: assess impact, recommend the best route alternatives,\n"
            f"flag critical suppliers for attention, estimate recovery, and finalize\n"
            f"an advisory report that suppliers can act upon.\n"
            f"Use suggest_reroute (not approve) — the supplier decides.\n"
            f"Be specific. Cite city names and numbers from tool results."
        )

        contents = [
            gt.Content(role="user", parts=[gt.Part(text=system_prompt + "\n\nBegin your autonomous assessment now.")])
        ]

        for turn in range(max_turns):
            t0 = time.time()
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=gt.GenerateContentConfig(tools=tools_schema, temperature=0.3),
            )
            ms = int((time.time() - t0) * 1000)

            candidate = resp.candidates[0]
            contents.append(gt.Content(role="model", parts=candidate.content.parts))

            # Process function calls
            func_calls = [p for p in candidate.content.parts if hasattr(p, "function_call") and p.function_call]
            tool_results_parts = []

            for part in func_calls:
                fc = part.function_call
                tool_name = fc.name
                raw_args = dict(fc.args) if fc.args else {}

                thought_text = next(
                    (p.text for p in candidate.content.parts if hasattr(p, "text") and p.text), ""
                )

                result = self._dispatch(tool_name, raw_args)
                self._log_step(tool_name, thought_text or f"Calling {tool_name}…", raw_args, result, ms)

                tool_results_parts.append(
                    gt.Part(function_response=gt.FunctionResponse(name=tool_name, response=result))
                )

                if tool_name == "finalize_plan" and self.final_plan:
                    return "success"

            if tool_results_parts:
                contents.append(gt.Content(role="user", parts=tool_results_parts))
            elif not func_calls:
                # Model stopped
                break

        return "success"

    # ──────────────────────────────────────
    # Groq tool-calling fallback
    # ──────────────────────────────────────

    def _run_groq(self, max_turns: int = 12) -> str:
        key = GROQ_KEY()
        if not key:
            raise ValueError("No Groq key")

        from groq import Groq
        client = Groq(api_key=key)

        tools = [
            {"type": "function", "function": {"name": "assess_disruptions", "description": "Get disruption scope.", "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {"name": "score_route_alternatives", "description": "Get ranked rerouting alternatives.", "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {"name": "approve_reroute", "description": "Approve alternate route.", "parameters": {"type": "object", "properties": {"source": {"type": "string"}, "destination": {"type": "string"}, "reason": {"type": "string"}}, "required": ["source", "destination", "reason"]}}},
            {"type": "function", "function": {"name": "flag_critical_supplier", "description": "Flag critical node.", "parameters": {"type": "object", "properties": {"node_id": {"type": "string"}, "reason": {"type": "string"}, "priority": {"type": "string"}}, "required": ["node_id", "reason"]}}},
            {"type": "function", "function": {"name": "estimate_recovery", "description": "Estimate recovery days.", "parameters": {"type": "object", "properties": {"risk_level": {"type": "string"}, "n_nodes": {"type": "integer"}}, "required": ["risk_level", "n_nodes"]}}},
            {"type": "function", "function": {"name": "notify_suppliers", "description": "Dispatch supplier notifications.", "parameters": {"type": "object", "properties": {"recipient_group": {"type": "string"}, "message": {"type": "string"}}, "required": ["recipient_group", "message"]}}},
            {"type": "function", "function": {"name": "finalize_plan", "description": "Commit action plan.", "parameters": {"type": "object", "properties": {"summary": {"type": "string"}, "priority_actions": {"type": "array", "items": {"type": "string"}}, "estimated_recovery_days": {"type": "integer"}, "risk_level": {"type": "string"}}, "required": ["summary", "priority_actions", "estimated_recovery_days", "risk_level"]}}},
        ]

        messages = [
            {"role": "system", "content": (
                f"You are an autonomous supply chain AI. Disruption: {self.report.event_text} "
                f"(Severity: {self.report.severity}). Systematically use tools to assess, reroute, flag, and finalize."
            )},
            {"role": "user", "content": "Begin autonomous assessment."},
        ]

        for _ in range(max_turns):
            t0 = time.time()
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.2,
                max_tokens=4096,
            )
            ms = int((time.time() - t0) * 1000)
            msg = resp.choices[0].message
            messages.append(msg)

            tool_calls = msg.tool_calls or []
            if not tool_calls:
                break

            thought = msg.content or ""
            for tc in tool_calls:
                fn = tc.function
                args = json.loads(fn.arguments or "{}")
                result = self._dispatch(fn.name, args)
                self._log_step(fn.name, thought, args, result, ms)
                thought = ""
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
                if fn.name == "finalize_plan" and self.final_plan:
                    return "success"

        return "success"

    # ──────────────────────────────────────
    # Deterministic fallback (always works)
    # ──────────────────────────────────────

    def _run_deterministic(self) -> str:
        r = self.report

        # Step 1: assess
        scope = self._assess_disruptions()
        self._log_step("assess_disruptions",
            f"Disruption detected: '{r.event_text}'. Starting scope assessment — "
            f"how many nodes, countries, and shipments are affected?",
            {}, scope)

        # Step 2: routes
        routes = self._score_route_alternatives()
        self._log_step("score_route_alternatives",
            f"Scope: {scope['total_cascade_nodes']} nodes, {scope['shipments_at_risk']} shipments at risk. "
            f"Now ranking pre-computed rerouting alternatives.",
            {}, routes)

        # Step 3: suggest best routes (not execute — supplier decides)
        found = [ro for ro in self.routes if ro.status == "found"]
        for route in found[:3]:
            result = self._suggest_reroute(
                route.source_label, route.dest_label,
                f"Alternative adds {route.detour_pct:+.1f}% distance "
                f"(${route.cost_delta_usd:+,.0f} cost delta, {route.delay_delta_days:+.1f}d transit). "
                f"Safety: {route.route_safety}. Recommended for supplier consideration.",
            )
            self._log_step("suggest_reroute",
                f"Route {route.source_label}→{route.dest_label}: viable alternative "
                f"({route.detour_pct:+.1f}%, ${route.cost_delta_usd:+,.0f}). Adding to advisory.",
                {"source": route.source_label, "destination": route.dest_label,
                 "reason": result.get("message", "")}, result)

        # Step 4: flag top risk node
        if scope.get("top_risk_nodes"):
            top = scope["top_risk_nodes"][0]
            flag_result = self._flag_critical_supplier(
                top["node"],
                f"Highest-risk node (score {top['risk_score']:.3f}, Tier {top['tier']}) — "
                f"emergency backup sourcing for {top['product']} required.",
                "high",
            )
            self._log_step("flag_critical_supplier",
                f"{top['city']} is the highest-risk chokepoint (risk {top['risk_score']:.3f}). "
                f"Flagging for immediate backup sourcing.",
                {"node_id": top["node"], "reason": flag_result.get("city", ""), "priority": "high"},
                flag_result)

        # Step 5: recovery estimate
        n_nodes = scope["total_cascade_nodes"]
        rl = {"critical": "Critical", "high": "High", "medium": "Medium", "low": "Low"}.get(r.severity, "Medium")
        rec = self._estimate_recovery(rl, n_nodes)
        self._log_step("estimate_recovery",
            f"Based on {n_nodes} cascade nodes and {r.severity} severity, estimating recovery timeline.",
            {"risk_level": rl, "n_nodes": n_nodes}, rec)

        # Step 6: notify suppliers
        if r.affected_sectors:
            sectors_str = ", ".join(r.affected_sectors[:3])
            countries_str = ", ".join(r.affected_countries[:3])
            notify_result = self._notify_suppliers(
                f"{sectors_str} suppliers in {countries_str}",
                f"Advisory: your shipments via {countries_str} may be delayed due to: "
                f"{r.event_text}. {len(self.suggested_alternatives)} alternative route(s) identified. "
                f"Please review the advisory and contact your freight forwarder.",
            )
            self._log_step("notify_suppliers",
                f"Dispatching advisory alerts to {sectors_str} suppliers — "
                f"informing of disruption status and {len(self.suggested_alternatives)} suggested alternatives.",
                {"recipient_group": f"{sectors_str} suppliers",
                 "message": "Disruption advisory + route suggestions"}, notify_result)

        # Step 7: finalize
        sev = r.severity.upper()
        mat_str = ", ".join(r.affected_sectors[:2]) if r.affected_sectors else "key materials"
        top_city = scope["top_risk_nodes"][0]["city"] if scope.get("top_risk_nodes") else "disrupted region"
        n_countries = scope["countries_affected"]

        summary = (
            f"A {r.severity}-severity event ({r.event_text}) has disrupted {n_nodes} supply chain nodes "
            f"across {n_countries} countries, with {top_city} as the highest-risk chokepoint. "
            f"The advisory agent identified {len(self.suggested_alternatives)} viable alternative route(s) and flagged "
            f"{len(self.flagged_suppliers)} critical supplier(s) for immediate attention. "
            f"Estimated recovery: {rec['estimated_recovery_days']} days. "
            f"Suppliers have been notified with status updates and route options — final routing decisions remain with suppliers."
        )

        priority_actions = [
            f"ADVISORY: {len(self.suggested_alternatives)} alternative route(s) identified — share with your freight forwarder for evaluation",
            f"Notify production/procurement of expected {rec['estimated_recovery_days']}-day recovery timeline",
            f"Review {len(self.flagged_suppliers)} flagged supplier(s) — assess backup sourcing options",
            f"Build {min(rec['estimated_recovery_days'], 14)}-day buffer stock for {mat_str} as precaution",
            f"Check insurance/force-majeure clauses for {scope.get('shipments_at_risk', 0)} at-risk shipments",
        ]
        if n_nodes > 10:
            priority_actions.append(
                f"Escalate to procurement leadership — {n_nodes} cascade nodes indicate systemic corridor risk"
            )

        fin = self._finalize_plan(summary, priority_actions, rec["estimated_recovery_days"], rl)
        self._log_step("finalize_plan",
            "Full analysis complete. Reroutes approved, suppliers flagged, recovery estimated. Committing plan.",
            {"summary": summary, "priority_actions": priority_actions,
             "estimated_recovery_days": rec["estimated_recovery_days"], "risk_level": rl}, fin)

        return "success"

    # ──────────────────────────────────────
    # Public entry point
    # ──────────────────────────────────────

    def run(self, max_turns: int = 14) -> AgentDecision:
        t0 = time.time()
        source = "deterministic"

        # Try Gemini first
        try:
            self._run_gemini(max_turns)
            source = "gemini"
            print(f"  [agent] Gemini agent completed in {len(self.steps)} steps")
        except Exception as gemini_err:
            print(f"  [agent] Gemini failed: {gemini_err}")
            # Try Groq
            self._steps = []
            self.approved_reroutes = []
            self.flagged_suppliers = []
            self.final_plan = None
            self._step_n = 0
            self.steps = []
            try:
                self._run_groq(max_turns)
                source = "groq"
                print(f"  [agent] Groq agent completed in {len(self.steps)} steps")
            except Exception as groq_err:
                print(f"  [agent] Groq failed: {groq_err} — deterministic fallback")
                self.approved_reroutes = []
                self.flagged_suppliers = []
                self.final_plan = None
                self._step_n = 0
                self.steps = []
                self._run_deterministic()
                source = "deterministic"

        elapsed = round(time.time() - t0, 1)
        plan = self.final_plan or {
            "summary": "Analysis complete — see step log.",
            "priority_actions": ["Review action log above."],
            "estimated_recovery_days": 7,
            "risk_level": "Medium",
        }

        return AgentDecision(
            decision_id=f"DEC-{uuid.uuid4().hex[:8].upper()}",
            event_id=self.report.event_id,
            event_text=self.report.event_text,
            severity=self.report.severity,
            trigger_source=self.report.source,
            steps=self.steps,
            suggested_alternatives=self.suggested_alternatives,
            flagged_suppliers=self.flagged_suppliers,
            final_summary=plan.get("summary", ""),
            priority_actions=plan.get("priority_actions", []),
            risk_level=plan.get("risk_level", "Medium"),
            estimated_recovery_days=plan.get("estimated_recovery_days", 7),
            elapsed_seconds=elapsed,
            agent_source=source,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


# ──────────────────────────────────────
# Convenience wrapper
# ──────────────────────────────────────

def run_agent(disruption_report, route_alternatives, risk_df, shipments) -> AgentDecision:
    agent = SupplyChainAgent(disruption_report, route_alternatives, risk_df, shipments)
    return agent.run()
