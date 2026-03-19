from __future__ import annotations
from typing import Dict, List, Tuple, Any
import math
import pandas as pd
import networkx as nx
import numpy as np

def build_graph(nodes: pd.DataFrame, edges: pd.DataFrame, weight_mode: str = "confidence_times_strength") -> nx.DiGraph:
    G = nx.DiGraph()
    for r in nodes.itertuples(index=False):
        cid = str(getattr(r, "company_id"))
        if cid and cid != "nan":
            G.add_node(cid,
                       name=str(getattr(r, "canonical_name", "")),
                       stage=str(getattr(r, "value_chain_stage", "")),
                       country=str(getattr(r, "country", "")))
    for e in edges.itertuples(index=False):
        u = str(getattr(e, "src_company_id"))
        v = str(getattr(e, "dst_company_id"))
        conf = float(getattr(e, "confidence_plink", 0.5))
        strength = float(getattr(e, "strength", 1.0))
        if weight_mode == "confidence_only":
            w = conf
        elif weight_mode == "strength_only":
            w = strength
        else:
            w = conf * strength
        G.add_edge(u, v, rel=str(getattr(e, "rel_type")), weight=w, evidence=str(getattr(e, "evidence", "")))
    return G

def _exp_decay(dist: int, lam: float) -> float:
    return float(math.exp(-lam * dist))

def exposure_shortest_path(G: nx.Graph, event_nodes: List[str], severity: float, lam: float) -> Dict[str, float]:
    exposure = {n: 0.0 for n in G.nodes()}
    for src in event_nodes:
        if src not in G:
            continue
        lengths = nx.single_source_shortest_path_length(G, src)
        for n, d in lengths.items():
            exposure[n] += severity * _exp_decay(int(d), lam)
    return exposure

def exposure_rwr(G: nx.Graph, event_nodes: List[str], severity: float, restart_prob: float = 0.15, iters: int = 50) -> Dict[str, float]:
    nodes = list(G.nodes())
    idx = {n:i for i,n in enumerate(nodes)}
    N = len(nodes)
    if N == 0:
        return {}
    A = np.zeros((N,N), dtype=float)
    for u,v,data in G.edges(data=True):
        w = float(data.get("weight", 1.0))
        A[idx[u], idx[v]] += w
    # row normalize
    row_sum = A.sum(axis=1)
    row_sum[row_sum == 0] = 1.0
    P = (A.T / row_sum).T

    p0 = np.zeros(N, dtype=float)
    for n in event_nodes:
        if n in idx:
            p0[idx[n]] = 1.0
    if p0.sum() == 0:
        p0[:] = 1.0 / N
    else:
        p0 = p0 / p0.sum()

    p = p0.copy()
    for _ in range(iters):
        p = (1 - restart_prob) * (p @ P) + restart_prob * p0

    return {nodes[i]: float(severity * p[i]) for i in range(N)}

from src.gnn.gat import run_gat_exposure

def compute_exposure(nodes: pd.DataFrame,
                     edges: pd.DataFrame,
                     risk_events: pd.DataFrame,
                     use_undirected: bool,
                     lam: float,
                     restart_prob: float,
                     iters: int,
                     weight_mode: str,
                     cfg_gnn: dict = None) -> pd.DataFrame:
    Gd = build_graph(nodes, edges, weight_mode=weight_mode)
    G = Gd.to_undirected() if use_undirected else Gd

    # 1. Baseline: Shortest Path and RWR
    rows = []
    for ev in risk_events.itertuples(index=False):
        event_id = getattr(ev, "event_id")
        sev = float(getattr(ev, "severity", 1.0))
        ents = getattr(ev, "entity_ids", [])
        if not isinstance(ents, (list, tuple)):
            try:
                ents = list(ents)  # ndarray → list
            except (TypeError, ValueError):
                ents = []
        sp = exposure_shortest_path(G, ents, sev, lam)
        rw = exposure_rwr(G, ents, sev, restart_prob=restart_prob, iters=iters)

        for cid in G.nodes():
            rows.append({
                "event_id": event_id,
                "company_id": cid,
                "exposure_sp": sp.get(cid, 0.0),
                "exposure_rwr": rw.get(cid, 0.0),
            })
    
    baseline_df = pd.DataFrame(rows)
    
    # 2. GNN: GAT (Graph Attention Network)
    gat_df = run_gat_exposure(nodes, edges, risk_events, cfg_gnn or {})
    
    # Merge results
    final_df = baseline_df.merge(gat_df, on=["event_id", "company_id"], how="left")
    return final_df
