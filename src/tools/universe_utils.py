from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
import pandas as pd
import yaml

def parse_semicolon_list(x: str) -> List[str]:
    if x is None:
        return []
    s = str(x).strip()
    if s.lower() == "nan" or s == "":
        return []
    items = [t.strip() for t in s.split(";")]
    out = []
    seen = set()
    for it in items:
        if it and it not in seen:
            seen.add(it)
            out.append(it)
    return out

def normalize_suffix_priority(tickers: List[str], priority: List[str]) -> List[str]:
    # stable sort by suffix priority; unknown suffix goes last
    def key(t: str) -> int:
        for i, suf in enumerate(priority):
            if t.endswith(suf):
                return i
        return len(priority) + 999
    return sorted(tickers, key=key)

def choose_primary_ticker(company_id: str, tickers: List[str], priority: List[str]) -> str:
    # If company_id is valid, keep it
    cid = (company_id or "").strip()
    if cid and cid.lower() != "nan":
        return cid
    if not tickers:
        return ""
    ordered = normalize_suffix_priority(tickers, priority)
    return ordered[0]

def load_priority_yaml(path: Path) -> List[str]:
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data.get("ticker_suffix_priority", []) if isinstance(data, dict) else []

def build_ticker_alias_map(universe: pd.DataFrame, normalize_rules: Dict[str,str] | None = None) -> Dict[str,str]:
    """Map any known ticker alias -> company_id (primary). Includes normalized yfinance variants."""
    rules = normalize_rules or {}
    alias: Dict[str,str] = {}
    for _, row in universe.iterrows():
        cid = str(row.get("company_id","")).strip()
        if not cid or cid.lower() == "nan":
            continue
        tickers = parse_semicolon_list(row.get("tickers",""))
        tickers = [cid] + tickers
        for t in tickers:
            t = str(t).strip()
            if not t or t.lower() == "nan":
                continue
            alias.setdefault(t, cid)
            # yfinance normalized alias
            for src_suf, dst_suf in rules.items():
                if t.endswith(src_suf):
                    alias.setdefault(t[: -len(src_suf)] + dst_suf, cid)
    return alias

def normalize_universe_df(universe: pd.DataFrame,
                          priority_suffixes: List[str],
                          normalize_rules: Dict[str,str] | None = None) -> tuple[pd.DataFrame, Dict[str,str], pd.DataFrame]:
    """Return (normalized_universe, ticker_alias_map, duplicates_df)."""
    df = universe.copy()

    required = ["company_id","canonical_name","country","region_group","value_chain_stage","listed","exchanges","tickers","notes"]
    for c in required:
        if c not in df.columns:
            df[c] = pd.NA

    # Ensure tickers includes company_id
    tickers_fixed = []
    primary_fixed = []
    for _, row in df.iterrows():
        cid = str(row.get("company_id","")).strip()
        tickers = parse_semicolon_list(row.get("tickers",""))
        # choose primary if missing
        primary = choose_primary_ticker(cid, tickers, priority_suffixes)
        # ensure primary present
        if primary and primary not in tickers:
            tickers = [primary] + tickers
        # ensure cid set
        primary_fixed.append(primary if primary else cid)
        tickers_fixed.append(";".join(tickers))

    df["company_id"] = primary_fixed
    df["tickers"] = tickers_fixed

    # duplicates report by company_id
    dup = df[df.duplicated(subset=["company_id"], keep=False)].copy()
    # build alias map
    alias = build_ticker_alias_map(df, normalize_rules=normalize_rules)
    # provider ticker (yfinance) for convenience
    if normalize_rules:
        def norm_yf(t: str) -> str:
            t = (t or "").strip()
            for src_suf, dst_suf in normalize_rules.items():
                if t.endswith(src_suf):
                    return t[: -len(src_suf)] + dst_suf
            return t
        df["provider_ticker"] = df["company_id"].apply(norm_yf)
    else:
        df["provider_ticker"] = df["company_id"]

    return df, alias, dup
