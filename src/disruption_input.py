"""
disruption_input.py
--------------------
Parses a free-text disruption event into a structured object containing:
  - event_text     : original text
  - affected_nodes : list of City_XX IDs likely impacted
  - severity       : "high" / "medium" / "low"
  - category       : type of disruption (natural_disaster, strike, etc.)
  - keywords_hit   : which keywords / countries triggered the match
  - reasoning      : LLM explanation of WHY those nodes were chosen (LLM path only)
  - llm_source     : "gemini" | "keyword-matching"

Strategy
--------
1. Try Gemini LLM first — it understands geopolitical context, implied effects,
   unusual phrasing (e.g. "Hormuz tensions" → oil routes through Middle East).
2. Fall back to fast keyword matching when the API key is missing or the call fails.
   The fallback is deterministic and works fully offline.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Country / city mapping
# ---------------------------------------------------------------------------
COUNTRY_CITY_MAP: Dict[str, List[str]] = {
    "china":          ["City_61", "City_1",  "City_2",  "City_3",  "City_4",  "City_5",
                       "City_6",  "City_7",  "City_8",  "City_9",  "City_10", "City_11", "City_12"],
    "chinese":        ["City_61", "City_1",  "City_2",  "City_3",  "City_4",  "City_5",
                       "City_6",  "City_7",  "City_8",  "City_9",  "City_10", "City_11", "City_12"],
    "shenzhen":       ["City_1",  "City_61"],
    "shanghai":       ["City_2",  "City_61"],
    "beijing":        ["City_3",  "City_61"],
    "wuhan":          ["City_6",  "City_61"],
    "guangzhou":      ["City_4",  "City_61"],
    "chengdu":        ["City_5",  "City_61"],
    "hanoi":          ["City_61"],
    "vietnam":        ["City_55", "City_61"],

    "india":          [f"City_{i}" for i in range(13, 23)],
    "indian":         [f"City_{i}" for i in range(13, 23)],
    "mumbai":         ["City_13"],
    "bangalore":      ["City_14"],
    "chennai":        ["City_15"],
    "hyderabad":      ["City_18"],
    "kolkata":        ["City_19"],
    "pune":           ["City_16"],

    "usa":            [f"City_{i}" for i in range(23, 33)],
    "america":        [f"City_{i}" for i in range(23, 33)],
    "american":       [f"City_{i}" for i in range(23, 33)],
    "houston":        ["City_25"],
    "los angeles":    ["City_24"],
    "new york":       ["City_23"],

    "germany":        [f"City_{i}" for i in range(33, 38)],
    "german":         [f"City_{i}" for i in range(33, 38)],
    "europe":         [f"City_{i}" for i in range(33, 43)],
    "european":       [f"City_{i}" for i in range(33, 43)],
    "netherlands":    ["City_38"],
    "rotterdam":      ["City_38"],
    "hamburg":        ["City_35"],

    "japan":          ["City_43", "City_44", "City_45", "City_50", "City_52"],
    "japanese":       ["City_43", "City_44", "City_45", "City_50", "City_52"],
    "tokyo":          ["City_43"],

    "korea":          ["City_46", "City_47", "City_49"],
    "korean":         ["City_46", "City_47", "City_49"],
    "taiwan":         ["City_48", "City_51"],
    "taipei":         ["City_48"],
    "seoul":          ["City_46"],

    "southeast asia": [f"City_{i}" for i in range(53, 62)],
    "thailand":       ["City_54"],
    "singapore":      ["City_53"],
    "malaysia":       ["City_57"],
    "indonesia":      ["City_56"],
    "manila":         ["City_58"],

    "middle east":    ["City_63", "City_64", "City_68", "City_69"],
    "uae":            ["City_63"],
    "dubai":          ["City_63"],
    "saudi":          ["City_64"],

    "africa":         ["City_65", "City_66", "City_67", "City_70"],
    "nigeria":        ["City_65"],
    "egypt":          ["City_66"],
}

PRODUCT_CITY_MAP: Dict[str, List[str]] = {
    "semiconductor":  ["City_2",  "City_46", "City_48", "City_27"],
    "semiconductors": ["City_2",  "City_46", "City_48", "City_27"],
    "chip":           ["City_2",  "City_46", "City_48", "City_27"],
    "chips":          ["City_2",  "City_46", "City_48", "City_27"],
    "microchip":      ["City_2",  "City_46", "City_48"],

    "electronics":    ["City_61", "City_1",  "City_5",  "City_12", "City_43", "City_49", "City_57", "City_58"],
    "electronic":     ["City_61", "City_1",  "City_5",  "City_12", "City_43", "City_49", "City_57", "City_58"],
    "factory":        ["City_61", "City_1",  "City_4",  "City_12"],

    "automotive":     ["City_3",  "City_9",  "City_15", "City_33", "City_37", "City_42", "City_45", "City_54"],
    "automobile":     ["City_3",  "City_9",  "City_15", "City_33", "City_37", "City_42", "City_45"],
    "car":            ["City_30", "City_33", "City_37", "City_45"],
    "vehicle":        ["City_30", "City_33", "City_37", "City_45"],

    "pharmaceutical": ["City_10", "City_13", "City_18", "City_40"],
    "pharmaceuticals":["City_10", "City_13", "City_18", "City_40"],
    "pharma":         ["City_10", "City_13", "City_18", "City_40"],
    "medicine":       ["City_13", "City_18", "City_40"],
    "drug":           ["City_13", "City_18"],

    "textile":        ["City_4",  "City_19", "City_20", "City_41", "City_55", "City_68"],
    "textiles":       ["City_4",  "City_19", "City_20", "City_41", "City_55", "City_68"],
    "clothing":       ["City_4",  "City_19", "City_41", "City_68"],
    "apparel":        ["City_4",  "City_19", "City_41"],

    "oil":            ["City_25", "City_64", "City_65", "City_69"],
    "gas":            ["City_25", "City_64", "City_69"],
    "petroleum":      ["City_25", "City_64", "City_69"],
    "petrochemical":  ["City_69"],

    "steel":          ["City_6",  "City_26"],
    "chemical":       ["City_8",  "City_21", "City_36", "City_40"],
    "chemicals":      ["City_8",  "City_21", "City_36", "City_40"],
    "aerospace":      ["City_7",  "City_28"],
    "mineral":        ["City_59", "City_67"],
    "minerals":       ["City_59", "City_67"],
    "agriculture":    ["City_60", "City_70"],
    "food":           ["City_60", "City_70"],

    "port":           ["City_35", "City_38", "City_47", "City_53", "City_63"],
    "logistics":      ["City_17", "City_29", "City_35", "City_38", "City_47", "City_53"],
    "distribution":   ["City_17", "City_23", "City_24", "City_29", "City_31", "City_63"],

    "supply chain":   ["City_61", "City_53", "City_47", "City_35"],
    "manufacturing":  ["City_61", "City_16", "City_26", "City_33", "City_44"],
    "production":     ["City_61", "City_1",  "City_12", "City_46"],
}

# ---------------------------------------------------------------------------
# LLM output → internal keyword mappings
# These translate the full country / industry names Gemini returns into the
# lowercase keys used by COUNTRY_CITY_MAP and PRODUCT_CITY_MAP above.
# ---------------------------------------------------------------------------
_COUNTRY_NAME_TO_KEYS: Dict[str, List[str]] = {
    "china":          ["china"],
    "taiwan":         ["taiwan", "taipei"],
    "south korea":    ["korea", "seoul"],
    "korea":          ["korea"],
    "japan":          ["japan", "tokyo"],
    "india":          ["india"],
    "vietnam":        ["vietnam", "hanoi"],
    "thailand":       ["thailand"],
    "singapore":      ["singapore"],
    "malaysia":       ["malaysia"],
    "indonesia":      ["indonesia"],
    "philippines":    ["manila"],
    "united states":  ["usa", "houston", "new york", "los angeles"],
    "usa":            ["usa"],
    "germany":        ["germany"],
    "netherlands":    ["netherlands", "rotterdam"],
    "europe":         ["europe"],
    "middle east":    ["middle east"],
    "saudi arabia":   ["saudi"],
    "uae":            ["uae", "dubai"],
    "nigeria":        ["nigeria"],
    "egypt":          ["egypt"],
    "africa":         ["africa"],
    "southeast asia": ["southeast asia"],
}

_INDUSTRY_NAME_TO_KEYS: Dict[str, List[str]] = {
    "electronics":       ["electronics", "electronic", "factory"],
    "semiconductors":    ["semiconductor", "semiconductors", "chip", "chips", "microchip"],
    "semiconductor":     ["semiconductor", "semiconductors", "chip", "chips"],
    "automotive":        ["automotive", "automobile", "car", "vehicle"],
    "pharmaceuticals":   ["pharmaceutical", "pharmaceuticals", "pharma", "medicine"],
    "pharmaceutical":    ["pharmaceutical", "pharmaceuticals", "pharma"],
    "textiles":          ["textile", "textiles", "clothing", "apparel"],
    "textile":           ["textile", "textiles"],
    "oil":               ["oil", "petroleum"],
    "gas":               ["gas"],
    "oil and gas":       ["oil", "gas", "petroleum"],
    "petrochemicals":    ["petrochemical"],
    "steel":             ["steel"],
    "chemicals":         ["chemical", "chemicals"],
    "chemical":          ["chemical", "chemicals"],
    "aerospace":         ["aerospace"],
    "minerals":          ["mineral", "minerals"],
    "agriculture":       ["agriculture", "food"],
    "logistics":         ["logistics", "port", "distribution"],
    "port":              ["port"],
    "manufacturing":     ["manufacturing", "production", "factory"],
    "it hardware":       ["electronics", "electronic"],
    "luxury goods":      ["distribution"],
}

# ---------------------------------------------------------------------------
# Severity / category keywords (used by keyword fallback)
# ---------------------------------------------------------------------------
HIGH_SEVERITY_KEYWORDS = {
    "earthquake", "tsunami", "hurricane", "typhoon", "cyclone",
    "wildfire", "war", "conflict", "sanctions", "explosion",
    "nuclear", "collapse", "catastrophic", "critical", "severe",
    "shutdown", "factory shutdown", "closure",
}
MEDIUM_SEVERITY_KEYWORDS = {
    "flood", "flooding", "strike", "protest", "shortage", "disruption",
    "damage", "fire", "storm", "drought",
}
LOW_SEVERITY_KEYWORDS = {
    "delay", "delayed", "slowdown", "congestion", "traffic",
    "minor", "partial", "limited",
}
CATEGORY_MAP = {
    "natural_disaster":   {"earthquake", "tsunami", "hurricane", "typhoon", "cyclone",
                           "flood", "flooding", "wildfire", "storm", "drought", "fire"},
    "labor":              {"strike", "protest", "walkout", "labor", "labour", "workers",
                           "union", "employee"},
    "geopolitical":       {"war", "conflict", "sanctions", "tariff", "ban", "embargo",
                           "political", "government", "trade war", "tensions", "naval"},
    "industrial_accident":{"explosion", "fire", "collapse", "shutdown", "closure",
                           "accident", "incident"},
    "logistics":          {"port", "congestion", "delay", "traffic", "route",
                           "shipping", "container"},
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dedup(cities: List[str]) -> List[str]:
    seen, out = set(), []
    for c in cities:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _tokenize(text: str) -> str:
    return text.lower().strip()


def _get_severity_kw(text: str) -> str:
    t = _tokenize(text)
    for kw in HIGH_SEVERITY_KEYWORDS:
        if kw in t:
            return "high"
    for kw in MEDIUM_SEVERITY_KEYWORDS:
        if kw in t:
            return "medium"
    for kw in LOW_SEVERITY_KEYWORDS:
        if kw in t:
            return "low"
    return "medium"


def _get_category_kw(text: str) -> str:
    t = _tokenize(text)
    for cat, keywords in CATEGORY_MAP.items():
        for kw in keywords:
            if kw in t:
                return cat
    return "other"


# ---------------------------------------------------------------------------
# LLM-based parser  (Gemini)
# ---------------------------------------------------------------------------

_LLM_PROMPT = """You are a senior supply chain risk analyst with deep knowledge of global trade networks, geopolitics, and industrial dependencies.

Analyse this disruption event and extract structured intelligence:

EVENT: "{event}"

Instructions:
- Consider DIRECT impact: which countries / cities are explicitly mentioned?
- Consider IMPLIED impact: e.g. "Strait of Hormuz tensions" implies oil shipping disruption affecting all of Asia & Europe; "TSMC fab issue" implies global semiconductor shortage
- Consider DOWNSTREAM effects: a factory shutdown in a key hub disrupts industries that depend on it
- Assess severity based on scale, criticality, and recovery difficulty
- Classify the disruption type accurately

Return ONLY valid JSON with exactly these fields:
{{
  "severity": "<high|medium|low>",
  "category": "<natural_disaster|labor|geopolitical|industrial_accident|logistics|other>",
  "affected_countries": ["<country1>", "<country2>"],
  "affected_industries": ["<industry1>", "<industry2>"],
  "reasoning": "<2-3 sentences explaining the supply chain impact, including any implied or downstream effects not obvious from the text>"
}}

For affected_countries use: China, Taiwan, South Korea, Japan, India, Vietnam, Thailand, Singapore, Malaysia, Indonesia, Philippines, United States, Germany, Netherlands, Europe, Middle East, Saudi Arabia, UAE, Nigeria, Egypt, Africa, Southeast Asia

For affected_industries use: electronics, semiconductors, automotive, pharmaceuticals, textiles, oil, gas, oil and gas, petrochemicals, steel, chemicals, aerospace, minerals, agriculture, logistics, port, manufacturing, it hardware, luxury goods

Be comprehensive — list all plausible affected countries and industries, not just the most obvious ones."""


def _llm_parse(text: str) -> dict | None:
    """
    Call Gemini to extract structured disruption data.
    Returns None on any failure (missing key, rate limit, parse error).
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key in ("your_gemini_api_key_here", ""):
        return None

    try:
        from google import genai
        from google.genai import types as genai_types

        client   = genai.Client(api_key=api_key)
        prompt   = _LLM_PROMPT.format(event=text.replace('"', "'"))

        print("  [disruption_input] Calling Gemini for LLM classification …")
        response = client.models.generate_content(
            model    = "gemini-2.5-flash",
            contents = prompt,
            config   = genai_types.GenerateContentConfig(
                temperature       = 0.1,
                max_output_tokens = 1024,
            ),
        )

        raw_text = response.text.strip()
        match    = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not match:
            print("  [disruption_input] LLM returned no JSON — falling back")
            return None

        data = json.loads(match.group(0))

        # Validate required fields
        if not all(k in data for k in ("severity", "category", "affected_countries", "affected_industries")):
            return None

        print(
            f"  [disruption_input] LLM classification: severity={data['severity']}, "
            f"category={data['category']}, "
            f"countries={data['affected_countries']}, "
            f"industries={data['affected_industries']}"
        )
        return data

    except Exception as exc:
        print(f"  [disruption_input] LLM parse failed: {exc} — falling back to keywords")
        return None


def _nodes_from_llm(data: dict) -> tuple[list[str], list[str], list[str]]:
    """
    Map LLM output (full names) → City_XX node IDs using the internal lookup tables.
    Returns (city_ids, country_keywords_hit, product_keywords_hit)
    """
    cities: list[str] = []
    country_hit: list[str] = []
    product_hit: list[str] = []

    for country in data.get("affected_countries", []):
        c_lower = country.lower().strip()
        keys    = _COUNTRY_NAME_TO_KEYS.get(c_lower, [c_lower])
        for key in keys:
            if key in COUNTRY_CITY_MAP:
                cities.extend(COUNTRY_CITY_MAP[key])
                country_hit.append(key)

    for industry in data.get("affected_industries", []):
        i_lower = industry.lower().strip()
        keys    = _INDUSTRY_NAME_TO_KEYS.get(i_lower, [i_lower])
        for key in keys:
            if key in PRODUCT_CITY_MAP:
                cities.extend(PRODUCT_CITY_MAP[key])
                product_hit.append(key)

    return _dedup(cities), list(set(country_hit)), list(set(product_hit))


# ---------------------------------------------------------------------------
# Keyword-matching fallback
# ---------------------------------------------------------------------------

def _keyword_parse(text: str) -> Dict[str, Any]:
    t = _tokenize(text)
    cities: list[str] = []
    keywords_hit: list[str] = []
    country_hit: list[str] = []
    product_hit: list[str] = []

    for keyword, node_list in COUNTRY_CITY_MAP.items():
        if keyword in t:
            cities.extend(node_list)
            keywords_hit.append(keyword)
            country_hit.append(keyword)

    for keyword, node_list in PRODUCT_CITY_MAP.items():
        if keyword in t:
            cities.extend(node_list)
            keywords_hit.append(keyword)
            product_hit.append(keyword)

    unique = _dedup(cities)
    if not unique:
        unique       = ["City_1", "City_2", "City_13", "City_24"]
        keywords_hit = ["general"]

    return {
        "event_text":     text,
        "affected_nodes": unique,
        "severity":       _get_severity_kw(text),
        "category":       _get_category_kw(text),
        "keywords_hit":   list(set(keywords_hit)),
        "country_hit":    list(set(country_hit)),
        "product_hit":    list(set(product_hit)),
        "reasoning":      "",
        "llm_source":     "keyword-matching",
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def parse_disruption(text: str) -> Dict[str, Any]:
    """
    Parse a free-text disruption event.

    Tries Gemini LLM first for deep contextual understanding.
    Falls back to keyword matching when the API key is absent or the call fails.

    Returns
    -------
    dict with keys:
        event_text      : str
        affected_nodes  : list[str]  e.g. ["City_1", "City_5", …]
        severity        : "high" | "medium" | "low"
        category        : disruption type string
        keywords_hit    : list of matched keywords / node-mapping keys
        country_hit     : countries identified
        product_hit     : industries identified
        reasoning       : LLM explanation (empty string for keyword path)
        llm_source      : "gemini" | "keyword-matching"
    """
    # ── Try LLM ──────────────────────────────────────────────────────────────
    llm_data = _llm_parse(text)

    if llm_data:
        cities, country_hit, product_hit = _nodes_from_llm(llm_data)

        if not cities:
            # LLM returned countries/industries we can't map — fall through to keywords
            print("  [disruption_input] LLM output unmappable — using keyword fallback")
        else:
            return {
                "event_text":     text,
                "affected_nodes": cities,
                "severity":       llm_data.get("severity", "medium"),
                "category":       llm_data.get("category", "other"),
                "keywords_hit":   country_hit + product_hit,
                "country_hit":    country_hit,
                "product_hit":    product_hit,
                "reasoning":      llm_data.get("reasoning", ""),
                "llm_source":     "gemini",
            }

    # ── Keyword fallback ──────────────────────────────────────────────────────
    return _keyword_parse(text)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        "Factory shutdown in China affecting electronics supply chain",
        "Tensions at the Strait of Hormuz escalating due to naval exercises",
        "Labor strike at South Korean semiconductor fabs",
        "Severe flooding in Vietnam affecting textile manufacturing",
    ]
    for t in tests:
        print(f"\nINPUT: {t}")
        result = parse_disruption(t)
        print(f"  source   : {result['llm_source']}")
        print(f"  severity : {result['severity']}")
        print(f"  category : {result['category']}")
        print(f"  countries: {result['country_hit']}")
        print(f"  products : {result['product_hit']}")
        print(f"  nodes    : {len(result['affected_nodes'])} nodes")
        if result.get("reasoning"):
            print(f"  reason   : {result['reasoning']}")
