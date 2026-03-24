from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

@dataclass
class KGQC:
    edges: int
    edges_with_evidence: int
    evidence_rate: float
    invalid_relation_rows: int
    missing_node_rows: int

def run_kg_qc(nodes: pd.DataFrame, edges: pd.DataFrame, allowed_relations: list[str]) -> KGQC:
    e = edges.copy()
    e["has_evidence"] = e["evidence"].fillna("").astype(str).str.len() > 0
    edges_n = len(e)
    with_ev = int(e["has_evidence"].sum())
    rate = 0.0 if edges_n == 0 else with_ev / edges_n

    invalid_rel = int((~e["rel_type"].isin(allowed_relations)).sum()) if "rel_type" in e.columns else edges_n
    node_set = set(nodes["company_id"].astype(str).tolist())
    missing_node = int((~e["src_company_id"].astype(str).isin(node_set)).sum() + (~e["dst_company_id"].astype(str).isin(node_set)).sum())
    return KGQC(edges=edges_n, edges_with_evidence=with_ev, evidence_rate=float(rate),
                invalid_relation_rows=invalid_rel, missing_node_rows=missing_node)
