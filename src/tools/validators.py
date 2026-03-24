from __future__ import annotations
from dataclasses import dataclass
from typing import List
import pandas as pd

@dataclass
class UniverseValidation:
    rows: int
    missing_company_id: int
    duplicated_company_id: int

def validate_universe(universe: pd.DataFrame) -> UniverseValidation:
    df = universe.copy()
    missing = int(df["company_id"].isna().sum() + (df["company_id"].astype(str).str.strip()=="").sum())
    dup = int(df.duplicated(subset=["company_id"]).sum())
    return UniverseValidation(rows=len(df), missing_company_id=missing, duplicated_company_id=dup)

@dataclass
class SeedValidation:
    rows: int
    bad_relation_rows: int
    missing_evidence_rows: int

def validate_seed_edges(seed: pd.DataFrame, allowed_relations: List[str]) -> SeedValidation:
    df = seed.copy()
    bad_rel = int((~df["rel_type"].isin(allowed_relations)).sum()) if "rel_type" in df.columns else len(df)
    miss_ev = int((df.get("evidence","").fillna("").astype(str).str.len()==0).sum())
    return SeedValidation(rows=len(df), bad_relation_rows=bad_rel, missing_evidence_rows=miss_ev)
