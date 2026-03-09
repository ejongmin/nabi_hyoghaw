"""Step 4: Assign risk_type labels to each article based on keyword matching."""

import pandas as pd

from .config import load_risk_keywords


def classify_risk(title: str, keyword_map: dict) -> list[str]:
    """Return list of matching risk_type labels for a given title."""
    t = str(title).lower()
    labels = []
    for risk_type, cfg in keyword_map.items():
        for kw in cfg["keywords"]:
            if kw.lower() in t:
                labels.append(risk_type)
                break
    return labels or ["other"]


def classify_articles(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'risk_types' column (list of str) to each article row."""
    keyword_map = load_risk_keywords()
    df["risk_types"] = df["title"].apply(lambda t: classify_risk(t, keyword_map))
    distribution = {}
    for labels in df["risk_types"]:
        for l in labels:
            distribution[l] = distribution.get(l, 0) + 1
    print(f"[classify] risk_type distribution: {distribution}")
    return df
