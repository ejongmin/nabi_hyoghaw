from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd

def load_universe(universe_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(universe_csv)
    required = ["company_id","canonical_name","country","region_group","value_chain_stage","listed","exchanges","tickers","notes"]
    for c in required:
        if c not in df.columns:
            df[c] = pd.NA
    return df

def load_seed_edges(seed_csv: Path) -> pd.DataFrame:
    if not seed_csv.exists():
        # create empty valid frame
        return pd.DataFrame(columns=["src_company_id","rel_type","dst_company_id","confidence_plink","strength","evidence","source","valid_from","valid_to"])
    df = pd.read_csv(seed_csv)
    # normalize columns
    rename = {}
    if "src_id" in df.columns and "src_company_id" not in df.columns:
        rename["src_id"] = "src_company_id"
    if "dst_id" in df.columns and "dst_company_id" not in df.columns:
        rename["dst_id"] = "dst_company_id"
    if rename:
        df = df.rename(columns=rename)
    # ensure required
    req = ["src_company_id","rel_type","dst_company_id","confidence_plink","strength","evidence","source","valid_from","valid_to"]
    for c in req:
        if c not in df.columns:
            df[c] = pd.NA
    return df[req]

def build_static_kg(universe: pd.DataFrame, seed_edges: pd.DataFrame, allowed_relations: list[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    nodes = universe.copy()
    edges = seed_edges.copy()
    # enforce dtypes
    edges["rel_type"] = edges["rel_type"].astype(str)
    edges = edges[edges["rel_type"].isin(allowed_relations)].copy()
    edges["confidence_plink"] = pd.to_numeric(edges["confidence_plink"], errors="coerce").fillna(0.5)
    edges["strength"] = pd.to_numeric(edges["strength"], errors="coerce").fillna(1.0)
    return nodes, edges
