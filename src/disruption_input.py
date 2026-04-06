"""
disruption_input.py
--------------------
Parses a free-text disruption event into a structured object containing:
  - event_text     : original text
  - affected_nodes : list of City_XX IDs likely impacted
  - severity       : "high" / "medium" / "low"
  - category       : type of disruption (natural_disaster, strike, etc.)
  - keywords_hit   : which keywords triggered the match (for transparency)

No LLM needed here — fast keyword matching keeps this deterministic and
hackathon-safe (works even without internet / API keys).
"""

import re
from typing import Dict, List, Any


# ---------------------------------------------------------------------------
# Keyword → Country mapping
# Each entry maps a keyword to the city IDs in that country/region
# ---------------------------------------------------------------------------
COUNTRY_CITY_MAP: Dict[str, List[str]] = {
    # China — City_61 is the real data hub; others are manufacturing nodes
    "china":          ["City_61", "City_1", "City_2", "City_3", "City_4", "City_5",
                       "City_6", "City_7", "City_8", "City_9", "City_10", "City_11", "City_12"],
    "chinese":        ["City_61", "City_1", "City_2", "City_3", "City_4", "City_5",
                       "City_6", "City_7", "City_8", "City_9", "City_10", "City_11", "City_12"],
    "shenzhen":       ["City_1",  "City_61"],
    "shanghai":       ["City_2",  "City_61"],
    "beijing":        ["City_3",  "City_61"],
    "wuhan":          ["City_6",  "City_61"],
    "guangzhou":      ["City_4",  "City_61"],
    "chengdu":        ["City_5",  "City_61"],
    "hanoi":          ["City_61"],
    "vietnam":        ["City_55", "City_61"],

    # India
    "india":          [f"City_{i}" for i in range(13, 23)],
    "indian":         [f"City_{i}" for i in range(13, 23)],
    "mumbai":         ["City_13"],
    "bangalore":      ["City_14"],
    "chennai":        ["City_15"],
    "hyderabad":      ["City_18"],
    "kolkata":        ["City_19"],
    "pune":           ["City_16"],

    # USA
    "usa":            [f"City_{i}" for i in range(23, 33)],
    "america":        [f"City_{i}" for i in range(23, 33)],
    "american":       [f"City_{i}" for i in range(23, 33)],
    "houston":        ["City_25"],
    "los angeles":    ["City_24"],
    "new york":       ["City_23"],

    # Europe
    "germany":        [f"City_{i}" for i in range(33, 38)],
    "german":         [f"City_{i}" for i in range(33, 38)],
    "europe":         [f"City_{i}" for i in range(33, 43)],
    "european":       [f"City_{i}" for i in range(33, 43)],
    "netherlands":    ["City_38"],
    "rotterdam":      ["City_38"],
    "hamburg":        ["City_35"],

    # Japan
    "japan":          ["City_43", "City_44", "City_45", "City_50", "City_52"],
    "japanese":       ["City_43", "City_44", "City_45", "City_50", "City_52"],
    "tokyo":          ["City_43"],

    # Korea / Taiwan
    "korea":          ["City_46", "City_47", "City_49"],
    "korean":         ["City_46", "City_47", "City_49"],
    "taiwan":         ["City_48", "City_51"],
    "taipei":         ["City_48"],
    "seoul":          ["City_46"],

    # Southeast Asia
    "southeast asia": [f"City_{i}" for i in range(53, 62)],
    "thailand":       ["City_54"],
    "singapore":      ["City_53"],
    "malaysia":       ["City_57"],
    "indonesia":      ["City_56"],
    "manila":         ["City_58"],

    # Middle East
    "middle east":    ["City_63", "City_64", "City_68", "City_69"],
    "uae":            ["City_63"],
    "dubai":          ["City_63"],
    "saudi":          ["City_64"],

    # Africa
    "africa":         ["City_65", "City_66", "City_67", "City_70"],
    "nigeria":        ["City_65"],
    "egypt":          ["City_66"],
}

# ---------------------------------------------------------------------------
# Keyword → Product / Sector mapping
# Each entry maps a product keyword to the city IDs that produce it
# ---------------------------------------------------------------------------
PRODUCT_CITY_MAP: Dict[str, List[str]] = {
    "semiconductor":    ["City_2",  "City_46", "City_48", "City_27"],
    "semiconductors":   ["City_2",  "City_46", "City_48", "City_27"],
    "chip":             ["City_2",  "City_46", "City_48", "City_27"],
    "chips":            ["City_2",  "City_46", "City_48", "City_27"],
    "microchip":        ["City_2",  "City_46", "City_48"],

    # Electronics — City_61 is the key real-data hub for electronics
    "electronics":      ["City_61", "City_1", "City_5", "City_12", "City_43", "City_49", "City_57", "City_58"],
    "electronic":       ["City_61", "City_1", "City_5", "City_12", "City_43", "City_49", "City_57", "City_58"],
    "factory":          ["City_61", "City_1", "City_4", "City_12"],

    "automotive":       ["City_3",  "City_9",  "City_15", "City_33", "City_37", "City_42", "City_45", "City_54"],
    "automobile":       ["City_3",  "City_9",  "City_15", "City_33", "City_37", "City_42", "City_45"],
    "car":              ["City_30", "City_33", "City_37", "City_45"],
    "vehicle":          ["City_30", "City_33", "City_37", "City_45"],

    "pharmaceutical":   ["City_10", "City_13", "City_18", "City_40"],
    "pharmaceuticals":  ["City_10", "City_13", "City_18", "City_40"],
    "pharma":           ["City_10", "City_13", "City_18", "City_40"],
    "medicine":         ["City_13", "City_18", "City_40"],
    "drug":             ["City_13", "City_18"],

    "textile":          ["City_4",  "City_19", "City_20", "City_41", "City_55", "City_68"],
    "textiles":         ["City_4",  "City_19", "City_20", "City_41", "City_55", "City_68"],
    "clothing":         ["City_4",  "City_19", "City_41", "City_68"],
    "apparel":          ["City_4",  "City_19", "City_41"],

    "oil":              ["City_25", "City_64", "City_65", "City_69"],
    "gas":              ["City_25", "City_64", "City_69"],
    "petroleum":        ["City_25", "City_64", "City_69"],
    "petrochemical":    ["City_69"],

    "steel":            ["City_6",  "City_26"],
    "chemical":         ["City_8",  "City_21", "City_36", "City_40"],
    "chemicals":        ["City_8",  "City_21", "City_36", "City_40"],
    "aerospace":        ["City_7",  "City_28"],
    "mineral":          ["City_59", "City_67"],
    "minerals":         ["City_59", "City_67"],
    "agriculture":      ["City_60", "City_70"],
    "food":             ["City_60", "City_70"],

    "port":             ["City_35", "City_38", "City_47", "City_53", "City_63"],
    "logistics":        ["City_17", "City_29", "City_35", "City_38", "City_47", "City_53"],
    "distribution":     ["City_17", "City_23", "City_24", "City_29", "City_31", "City_63"],

    "supply chain":     ["City_61", "City_53", "City_47", "City_35"],
    "manufacturing":    ["City_61", "City_16", "City_26", "City_33", "City_44"],
    "production":       ["City_61", "City_1",  "City_12", "City_46"],
}

# ---------------------------------------------------------------------------
# Severity keywords — used to classify how critical the disruption is
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

# ---------------------------------------------------------------------------
# Category keywords
# ---------------------------------------------------------------------------
CATEGORY_MAP = {
    "natural_disaster": {
        "earthquake", "tsunami", "hurricane", "typhoon", "cyclone",
        "flood", "flooding", "wildfire", "storm", "drought", "fire",
    },
    "labor": {
        "strike", "protest", "walkout", "labor", "labour", "workers",
        "union", "employee",
    },
    "geopolitical": {
        "war", "conflict", "sanctions", "tariff", "ban", "embargo",
        "political", "government", "trade war",
    },
    "industrial_accident": {
        "explosion", "fire", "collapse", "shutdown", "closure",
        "accident", "incident",
    },
    "logistics": {
        "port", "congestion", "delay", "traffic", "route",
        "shipping", "container",
    },
}


def _tokenize(text: str) -> str:
    """Lowercase and normalise text for matching."""
    return text.lower().strip()


def get_severity(text: str) -> str:
    """
    Returns 'high', 'medium', or 'low' based on keywords in the text.
    Defaults to 'medium' if no keywords match.
    """
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
    return "medium"  # sensible default


def get_event_category(text: str) -> str:
    """
    Classifies the disruption type.
    Returns one of: natural_disaster, labor, geopolitical, industrial_accident, logistics, other
    """
    t = _tokenize(text)
    for category, keywords in CATEGORY_MAP.items():
        for kw in keywords:
            if kw in t:
                return category
    return "other"


def parse_disruption(text: str) -> Dict[str, Any]:
    """
    Main entry point. Parse a free-text disruption description.

    Parameters
    ----------
    text : str
        e.g. "Factory shutdown in China affecting electronics supply chain"

    Returns
    -------
    dict with keys:
        event_text      : str
        affected_nodes  : list[str]  — e.g. ["City_1", "City_5", ...]
        severity        : str        — "high" / "medium" / "low"
        category        : str        — disruption type
        keywords_hit    : list[str]  — which keywords triggered the match
        country_hit     : list[str]  — matched countries/regions
        product_hit     : list[str]  — matched products/sectors
    """
    t = _tokenize(text)
    affected: List[str] = []
    keywords_hit: List[str] = []
    country_hit: List[str] = []
    product_hit: List[str] = []

    # Match country / region keywords first
    for keyword, cities in COUNTRY_CITY_MAP.items():
        if keyword in t:
            affected.extend(cities)
            keywords_hit.append(keyword)
            country_hit.append(keyword)

    # Match product / sector keywords
    for keyword, cities in PRODUCT_CITY_MAP.items():
        if keyword in t:
            affected.extend(cities)
            keywords_hit.append(keyword)
            product_hit.append(keyword)

    # De-duplicate while preserving order
    seen = set()
    unique_affected = []
    for city in affected:
        if city not in seen:
            seen.add(city)
            unique_affected.append(city)

    # If nothing matched, default to a broad global set for demo
    if not unique_affected:
        unique_affected = ["City_1", "City_2", "City_13", "City_24"]
        keywords_hit = ["general"]

    severity = get_severity(text)
    category = get_event_category(text)

    return {
        "event_text":     text,
        "affected_nodes": unique_affected,
        "severity":       severity,
        "category":       category,
        "keywords_hit":   list(set(keywords_hit)),
        "country_hit":    list(set(country_hit)),
        "product_hit":    list(set(product_hit)),
    }


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    result = parse_disruption("Factory shutdown in China affecting electronics supply chain")
    print("DISRUPTION PARSE RESULT")
    for k, v in result.items():
        print(f"  {k:20s}: {v}")
