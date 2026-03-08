from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import pandas as pd

def normalize_for_yfinance(ticker: str, rules: Dict[str,str]) -> str:
    t = (ticker or "").strip()
    for src_suf, dst_suf in rules.items():
        if t.endswith(src_suf):
            return t[: -len(src_suf)] + dst_suf
    return t

def build_ticker_map(universe: pd.DataFrame, rules: Dict[str,str]) -> pd.DataFrame:
    """Map internal company_id -> provider ticker (yfinance)."""
    rows = []
    for r in universe.itertuples(index=False):
        cid = getattr(r, "company_id")
        if pd.isna(cid) or cid == "":
            continue
        prov = normalize_for_yfinance(str(cid), rules)
        rows.append({"company_id": str(cid), "provider_ticker": prov})
    m = pd.DataFrame(rows).drop_duplicates()
    return m
