"""
intelligence_feeds.py
----------------------
Unified data feed layer:
  - News RSS → Gemini/Groq extracts structured disruption events
  - Weather: OpenWeatherMap + Open-Meteo (free fallback)
  - Earthquakes: USGS GeoJSON feed
  - Tariffs: static 2025 lookup + enrichment
  - All outputs normalised to ExternalSignal dicts
"""

from __future__ import annotations
import json
import math
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import feedparser
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass


# ──────────────────────────────────────────────────────
# Key resolution (Gemini primary, Groq fallback)
# ──────────────────────────────────────────────────────

def _resolve(env_names: tuple) -> str:
    for name in env_names:
        v = os.getenv(name, "").strip()
        if v and not v.lower().startswith("your_"):
            return v
    return ""

GEMINI_KEY = lambda: _resolve(("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY"))
GROQ_KEY   = lambda: _resolve(("GROQ_API_KEY",))
OWM_KEY    = lambda: _resolve(("OPEN_WEATHER_API_KEY", "OPENWEATHER_API_KEY", "OWM_API_KEY"))


# ──────────────────────────────────────────────────────
# LLM helper (Gemini → Groq fallback)
# ──────────────────────────────────────────────────────

def llm_generate(prompt: str, temperature: float = 0.1, max_tokens: int = 4096) -> str:
    """Call Gemini first; fall back to Groq if unavailable."""
    gemini_key = GEMINI_KEY()
    if gemini_key:
        try:
            from google import genai
            from google.genai import types as gt
            client = genai.Client(api_key=gemini_key)
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=gt.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            return resp.text.strip()
        except Exception as e:
            print(f"  [llm] Gemini failed: {e} — trying Groq")

    groq_key = GROQ_KEY()
    if groq_key:
        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  [llm] Groq failed: {e}")

    return ""


# ──────────────────────────────────────────────────────
# News RSS
# ──────────────────────────────────────────────────────

RSS_FEEDS = [
    {"name": "Reuters World",  "url": "https://feeds.reuters.com/reuters/worldNews"},
    {"name": "BBC World",      "url": "https://feeds.bbci.co.uk/news/world/rss.xml"},
    {"name": "Al Jazeera",     "url": "https://www.aljazeera.com/xml/rss/all.xml"},
    {"name": "Guardian World", "url": "https://www.theguardian.com/world/rss"},
]

FILTER_KEYWORDS = {
    "war","conflict","sanction","tariff","trade","embargo","blockade",
    "port","shipping","supply chain","factory","shutdown","closure",
    "earthquake","tsunami","typhoon","hurricane","flood","wildfire","storm",
    "strike","protest","riot",
    "semiconductor","chip","electronics","oil","gas","steel","pharma",
    "automotive","textile","logistics","cargo","container",
    "china","russia","ukraine","taiwan","iran",
    "middle east","red sea","suez","strait","malacca",
    "trump","military","missile","pipeline","energy","mineral","rare earth",
    "export","import","border","attack","india","pakistan","israel","gaza",
}

_NEWS_PROMPT = """Extract supply chain disruption events from these headlines.
Return ONLY a JSON array. Each item must have:
{{"title":"...", "affected_countries":["..."], "affected_sectors":["..."],
 "severity":"high/medium/low", "category":"geopolitical/natural_disaster/labor/industrial_accident/logistics/other",
 "event_text":"one sentence describing the physical supply chain impact",
 "source_headline":"..."}}
Rules: only real physical supply chain impact, max 5 events, return [] if none.

Headlines:
{headlines}"""


def fetch_news_disruptions() -> List[Dict[str, Any]]:
    """Pull RSS headlines → LLM extraction → structured disruption events."""
    headlines = _fetch_headlines()
    if not headlines:
        return []

    headline_block = "\n".join(
        f"{i+1}. [{h['source']}] {h['title']}"
        + (f" — {h['summary'][:150]}" if h.get("summary") else "")
        for i, h in enumerate(headlines[:30])
    )

    raw = llm_generate(_NEWS_PROMPT.format(headlines=headline_block))
    if not raw:
        return _keyword_fallback(headlines)

    try:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return _keyword_fallback(headlines)
        events = json.loads(match.group(0))
        if not isinstance(events, list):
            return _keyword_fallback(headlines)
        print(f"  [news] LLM extracted {len(events)} disruption event(s)")
        return events
    except Exception as e:
        print(f"  [news] parse error: {e} — keyword fallback")
        return _keyword_fallback(headlines)


def _fetch_headlines(max_per_feed: int = 15) -> List[Dict[str, str]]:
    seen, unique = set(), []
    for feed_info in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_info["url"])
            for entry in feed.entries[:max_per_feed]:
                title   = (getattr(entry, "title",   "") or "").strip()
                summary = (getattr(entry, "summary", "") or "").strip()[:200]
                combined = (title + " " + summary).lower()
                if any(kw in combined for kw in FILTER_KEYWORDS):
                    t = title.lower()
                    if t not in seen:
                        seen.add(t)
                        unique.append({"title": title, "summary": summary, "source": feed_info["name"]})
        except Exception:
            pass
    print(f"  [news] {len(unique)} relevant headlines fetched")
    return unique


def _keyword_fallback(headlines: List[Dict]) -> List[Dict]:
    """Simple keyword-based fallback when LLM is unavailable."""
    results = []
    severity_keywords = {
        "high": ["war", "blocked", "earthquake", "major", "critical", "explosion"],
        "medium": ["strike", "delay", "disruption", "sanction", "shortage", "flood"],
        "low": ["warning", "risk", "concern", "slowdown"],
    }
    for h in headlines[:5]:
        title_lower = h["title"].lower()
        sev = "low"
        for level, kws in severity_keywords.items():
            if any(kw in title_lower for kw in kws):
                sev = level
                break
        results.append({
            "title": h["title"],
            "affected_countries": [],
            "affected_sectors": [],
            "severity": sev,
            "category": "other",
            "event_text": h["title"],
            "source_headline": f"[{h['source']}] {h['title']}",
        })
    return results[:3]


# ──────────────────────────────────────────────────────
# Weather
# ──────────────────────────────────────────────────────

SEVERE_OWM_IDS   = {200,201,202,210,211,212,221,230,231,232,502,503,504,511,522,531,602,611,612,613,615,616,621,622,781,771}
OPEN_METEO_SEVERE = {65,75,82,86,95,96,99}
WIND_SEVERE_MS   = 15.0
PRECIP_HEAVY_MM  = 10.0
EARTHQUAKE_MIN_MAG = 5.5


def _haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    d = math.radians
    a = math.sin(d(lat2-lat1)/2)**2 + math.cos(d(lat1))*math.cos(d(lat2))*math.sin(d(lon2-lon1)/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def fetch_weather_disruptions(meta_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Check weather conditions at supply chain nodes. Returns disruption dicts."""
    owm_key = OWM_KEY()
    events: List[Dict] = []

    cities = meta_df.reset_index()
    # Sample every 4th city to stay within free-tier rate limits
    sample = cities.iloc[::4]

    for _, row in sample.iterrows():
        lat, lon = float(row["lat"]), float(row["lon"])
        city_id  = row.get("city_id", row.name)
        city_name = row["city_name"]
        country   = row["country"]
        product   = row["product_category"]

        data = None
        source_name = "Open-Meteo"

        if owm_key:
            try:
                url = (f"https://api.openweathermap.org/data/2.5/weather"
                       f"?lat={lat}&lon={lon}&appid={owm_key}&units=metric")
                with urllib.request.urlopen(url, timeout=6) as r:
                    data = json.loads(r.read())
                source_name = "OpenWeatherMap"
            except Exception:
                data = None

        if data is None:
            try:
                url = (f"https://api.open-meteo.com/v1/forecast"
                       f"?latitude={lat}&longitude={lon}"
                       f"&current=weather_code,wind_speed_10m,precipitation"
                       f"&forecast_days=1&timezone=UTC")
                with urllib.request.urlopen(url, timeout=6) as r:
                    om = json.loads(r.read())
                cur = om.get("current", {})
                wc = int(cur.get("weather_code", 0) or 0)
                wind_ms = float(cur.get("wind_speed_10m", 0) or 0) / 3.6  # km/h → m/s
                precip  = float(cur.get("precipitation", 0) or 0)

                severe = wc in OPEN_METEO_SEVERE
                if not severe and wind_ms < WIND_SEVERE_MS and precip < PRECIP_HEAVY_MM:
                    continue

                sev = "high" if wc in {95,96,99} else ("medium" if severe else "low")
                events.append({
                    "event_text": f"Severe weather (code {wc}) at {city_name}, {country}",
                    "affected_nodes": [city_id],
                    "affected_countries": [country],
                    "affected_sectors": [product],
                    "severity": sev,
                    "category": "natural_disaster",
                    "source": "Open-Meteo",
                    "title": f"Weather alert: {city_name}",
                    "source_headline": f"Open-Meteo: weather code {wc} at {city_name}",
                    "keywords_hit": [country.lower()],
                })
                continue
            except Exception:
                continue

        # OpenWeatherMap parse
        if data:
            wx_list = data.get("weather", [{}])
            owm_id  = wx_list[0].get("id", 800)
            desc    = wx_list[0].get("description", "clear")
            wind_ms = data.get("wind", {}).get("speed", 0.0)
            precip  = data.get("rain", {}).get("3h", 0.0) + data.get("snow", {}).get("3h", 0.0)

            if not (owm_id in SEVERE_OWM_IDS or wind_ms >= WIND_SEVERE_MS or precip >= PRECIP_HEAVY_MM):
                continue

            sev = "high" if (wind_ms >= 25 or precip >= 30 or owm_id in {202,212,504,781}) else "medium"
            events.append({
                "event_text": f"{desc.title()} disrupting {product} supply chain at {city_name}, {country}",
                "affected_nodes": [city_id],
                "affected_countries": [country],
                "affected_sectors": [product],
                "severity": sev,
                "category": "natural_disaster",
                "source": "OpenWeatherMap",
                "title": f"Weather alert: {desc} at {city_name}",
                "source_headline": f"OWM: {desc} ({wind_ms:.0f} m/s wind, {precip:.0f} mm rain) at {city_name}",
                "keywords_hit": [country.lower()],
            })

    print(f"  [weather] {len(events)} weather event(s) detected")
    return events


def fetch_earthquake_disruptions(meta_df: pd.DataFrame, radius_km: float = 800) -> List[Dict]:
    """Check USGS earthquake feed and map quakes to nearby supply nodes."""
    try:
        url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_week.geojson"
        with urllib.request.urlopen(url, timeout=8) as r:
            quakes = json.loads(r.read()).get("features", [])
    except Exception:
        print("  [earthquake] USGS feed unavailable")
        return []

    events = []
    cities = meta_df.reset_index()

    for q in quakes:
        props  = q.get("properties", {})
        geo    = q.get("geometry", {})
        mag    = props.get("mag", 0) or 0
        place  = props.get("place", "unknown")
        if mag < EARTHQUAKE_MIN_MAG:
            continue

        coords = geo.get("coordinates", [])
        if len(coords) < 2:
            continue
        q_lon, q_lat = float(coords[0]), float(coords[1])

        nearby, countries = [], []
        for _, row in cities.iterrows():
            dist = _haversine(q_lat, q_lon, float(row["lat"]), float(row["lon"]))
            if dist <= radius_km:
                nearby.append(row.get("city_id", row.name))
                countries.append(row["country"])

        if not nearby:
            continue

        sev = "high" if mag >= 7.0 else ("medium" if mag >= 6.0 else "low")
        unique_countries = list(dict.fromkeys(c for c in countries if c))

        events.append({
            "event_text": f"M{mag:.1f} earthquake near {place} affecting supply chains",
            "affected_nodes": nearby,
            "affected_countries": unique_countries,
            "affected_sectors": [],
            "severity": sev,
            "category": "natural_disaster",
            "source": "USGS",
            "title": f"M{mag:.1f} Earthquake — {place}",
            "source_headline": f"USGS: M{mag:.1f} earthquake near {place}",
            "keywords_hit": ["earthquake"] + [c.lower() for c in unique_countries],
        })

    print(f"  [earthquake] {len(events)} earthquake event(s) found")
    return events


# ──────────────────────────────────────────────────────
# Tariff enrichment
# ──────────────────────────────────────────────────────

TARIFF_ALERTS = [
    {
        "event_text": "US-China tariff at 145% significantly disrupting trans-Pacific Electronics and Semiconductor trade",
        "affected_countries": ["China", "USA"],
        "affected_sectors": ["Electronics", "Semiconductors"],
        "severity": "high",
        "category": "geopolitical",
        "source": "tariff_data",
        "title": "US-China Tariff 145%",
        "source_headline": "2025 US tariff schedule: 145% on Chinese imports",
        "keywords_hit": ["tariff", "china", "usa", "electronics"],
    },
]


def get_active_tariff_alerts() -> List[Dict]:
    return TARIFF_ALERTS


# ──────────────────────────────────────────────────────
# Master fetch
# ──────────────────────────────────────────────────────

def fetch_all_signals(meta_df: pd.DataFrame, include_news: bool = True) -> List[Dict[str, Any]]:
    """
    Fetch all external intelligence signals.
    Returns unified list sorted by severity.
    """
    signals = []

    if include_news:
        try:
            news = fetch_news_disruptions()
            signals.extend(news)
        except Exception as e:
            print(f"  [feeds] News fetch error: {e}")

    try:
        quakes = fetch_earthquake_disruptions(meta_df)
        signals.extend(quakes)
    except Exception as e:
        print(f"  [feeds] Earthquake fetch error: {e}")

    try:
        weather = fetch_weather_disruptions(meta_df)
        signals.extend(weather)
    except Exception as e:
        print(f"  [feeds] Weather fetch error: {e}")

    priority = {"high": 0, "medium": 1, "low": 2}
    signals.sort(key=lambda s: priority.get(s.get("severity", "low"), 1))

    print(f"  [feeds] Total signals: {len(signals)}")
    return signals
