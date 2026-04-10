"""
state_manager.py
-----------------
Thread-safe shared state between background daemon and Streamlit dashboard.
All state persisted to JSON files under state/ for decoupled operation.
"""

from __future__ import annotations
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

STATE_DIR = Path(__file__).resolve().parent.parent / "state"
STATE_DIR.mkdir(exist_ok=True)

_SYSTEM_STATE_FILE  = STATE_DIR / "system_state.json"
_DECISIONS_FILE     = STATE_DIR / "agent_decisions.json"
_LIVE_LOG_FILE      = STATE_DIR / "live_log.json"
_NOTIFS_FILE        = STATE_DIR / "notifications.json"

MAX_LOG_ENTRIES      = 300
MAX_DECISIONS        = 50
MAX_NOTIFICATIONS    = 100

_lock = threading.RLock()


# ──────────────────────────────────────────────
# System State
# ──────────────────────────────────────────────

DEFAULT_SYSTEM_STATE = {
    "status": "idle",          # "idle" | "monitoring" | "alert" | "critical"
    "watchtower_running": False,
    "last_scan_utc": None,
    "scan_interval_minutes": 5,
    "active_disruptions": 0,
    "decisions_today": 0,
    "total_decisions": 0,
    "total_notifications": 0,
    "network_nodes": 0,
    "network_edges": 0,
    "active_shipments": 0,
    "current_activity": "Idle",
    "started_at": None,
}


def load_system_state() -> Dict[str, Any]:
    with _lock:
        if _SYSTEM_STATE_FILE.exists():
            try:
                with open(_SYSTEM_STATE_FILE) as f:
                    s = json.load(f)
                # Always restore to not-running on fresh process start
                s["watchtower_running"] = False
                # Auto-populate from data files if stats are missing
                if not s.get("network_nodes"):
                    try:
                        import pandas as pd
                        _p = Path(__file__).resolve().parent.parent
                        sc = pd.read_csv(_p / "data" / "supply_chain.csv")
                        rt = pd.read_csv(_p / "data" / "routes.csv")
                        s["network_nodes"] = len(sc)
                        s["network_edges"] = len(rt)
                    except Exception:
                        pass
                if not s.get("active_shipments"):
                    try:
                        import json as _json, pathlib
                        _sp = Path(__file__).resolve().parent.parent / "data" / "active_shipments.json"
                        if _sp.exists():
                            with open(_sp) as _f:
                                s["active_shipments"] = len(_json.load(_f))
                    except Exception:
                        pass
                return s
            except Exception:
                pass
        # Build default with auto-populated stats
        state = DEFAULT_SYSTEM_STATE.copy()
        try:
            import pandas as pd
            _p = Path(__file__).resolve().parent.parent
            state["network_nodes"] = len(pd.read_csv(_p / "data" / "supply_chain.csv"))
            state["network_edges"] = len(pd.read_csv(_p / "data" / "routes.csv"))
        except Exception:
            pass
        try:
            import json as _json
            _sp = Path(__file__).resolve().parent.parent / "data" / "active_shipments.json"
            if _sp.exists():
                with open(_sp) as _f:
                    state["active_shipments"] = len(_json.load(_f))
        except Exception:
            pass
        return state


def save_system_state(state: Dict[str, Any]) -> None:
    with _lock:
        with open(_SYSTEM_STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)


def update_system_state(**kwargs) -> None:
    with _lock:
        state = load_system_state()
        state.update(kwargs)
        save_system_state(state)


# ──────────────────────────────────────────────
# Live Log
# ──────────────────────────────────────────────

LOG_ICONS = {
    "info":        "ℹ️",
    "success":     "✅",
    "warning":     "⚠️",
    "error":       "❌",
    "disruption":  "🚨",
    "agent":       "🤖",
    "route":       "🔄",
    "notification":"📬",
    "weather":     "🌩️",
    "news":        "📰",
    "earthquake":  "🌍",
    "simulation":  "🎯",
    "scan":        "🔍",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(message: str, level: str = "info", category: str = "info") -> None:
    """Append a log entry to the live log."""
    with _lock:
        entries = _load_log()
        entries.insert(0, {
            "timestamp": _now_iso(),
            "level": level,
            "category": category,
            "icon": LOG_ICONS.get(category, "ℹ️"),
            "message": message,
        })
        entries = entries[:MAX_LOG_ENTRIES]
        with open(_LIVE_LOG_FILE, "w") as f:
            json.dump(entries, f, indent=2)


def _load_log() -> List[Dict]:
    if _LIVE_LOG_FILE.exists():
        try:
            with open(_LIVE_LOG_FILE) as f:
                return json.load(f)
        except Exception:
            return []
    return []


def load_log(n: int = 100) -> List[Dict]:
    with _lock:
        return _load_log()[:n]


# ──────────────────────────────────────────────
# Agent Decisions
# ──────────────────────────────────────────────

def save_decision(decision_dict: Dict) -> None:
    with _lock:
        existing = _load_decisions()
        existing.insert(0, decision_dict)
        existing = existing[:MAX_DECISIONS]
        with open(_DECISIONS_FILE, "w") as f:
            json.dump(existing, f, indent=2)

        # Increment counter
        state = load_system_state()
        state["total_decisions"] = state.get("total_decisions", 0) + 1
        today = _now_iso()[:10]
        if state.get("_today") != today:
            state["decisions_today"] = 0
            state["_today"] = today
        state["decisions_today"] = state.get("decisions_today", 0) + 1
        save_system_state(state)


def _load_decisions() -> List[Dict]:
    if _DECISIONS_FILE.exists():
        try:
            with open(_DECISIONS_FILE) as f:
                return json.load(f)
        except Exception:
            return []
    return []


def load_decisions(n: int = 20) -> List[Dict]:
    with _lock:
        return _load_decisions()[:n]


# ──────────────────────────────────────────────
# Notifications
# ──────────────────────────────────────────────

def save_notifications(notifications: List[Dict]) -> None:
    with _lock:
        existing = _load_notifications()
        for n in notifications:
            existing.insert(0, n)
        existing = existing[:MAX_NOTIFICATIONS]
        with open(_NOTIFS_FILE, "w") as f:
            json.dump(existing, f, indent=2)

        state = load_system_state()
        state["total_notifications"] = state.get("total_notifications", 0) + len(notifications)
        save_system_state(state)


def _load_notifications() -> List[Dict]:
    if _NOTIFS_FILE.exists():
        try:
            with open(_NOTIFS_FILE) as f:
                return json.load(f)
        except Exception:
            return []
    return []


def load_notifications(n: int = 50) -> List[Dict]:
    with _lock:
        return _load_notifications()[:n]


# ──────────────────────────────────────────────
# Init state files
# ──────────────────────────────────────────────

def init_state_files() -> None:
    """Ensure all state files exist with valid defaults."""
    if not _SYSTEM_STATE_FILE.exists():
        save_system_state(DEFAULT_SYSTEM_STATE.copy())
    if not _LIVE_LOG_FILE.exists():
        with open(_LIVE_LOG_FILE, "w") as f:
            json.dump([], f)
    if not _DECISIONS_FILE.exists():
        with open(_DECISIONS_FILE, "w") as f:
            json.dump([], f)
    if not _NOTIFS_FILE.exists():
        with open(_NOTIFS_FILE, "w") as f:
            json.dump([], f)
