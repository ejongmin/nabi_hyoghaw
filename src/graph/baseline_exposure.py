from __future__ import annotations
from typing import Dict, List, Tuple, Any
import math
import pandas as pd
import networkx as nx
import numpy as np
import logging

log = logging.getLogger("exposure")


def build_graph(nodes: pd.DataFrame, edges: pd.DataFrame,
                weight_mode: str = "confidence_times_strength") -> nx.DiGraph:
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
        G.add_edge(u, v, rel=str(getattr(e, "rel_type")), weight=w,
                   evidence=str(getattr(e, "evidence", "")))
    return G


def _exp_decay(dist: int, lam: float) -> float:
    return float(math.exp(-lam * dist))


def exposure_shortest_path(G: nx.Graph, event_nodes: List[str],
                           severity: float, lam: float) -> Dict[str, float]:
    exposure = {n: 0.0 for n in G.nodes()}
    for src in event_nodes:
        if src not in G:
            continue
        lengths = nx.single_source_shortest_path_length(G, src)
        for n, d in lengths.items():
            exposure[n] += severity * _exp_decay(int(d), lam)
    return exposure


def exposure_rwr(G: nx.Graph, event_nodes: List[str], severity: float,
                 restart_prob: float = 0.15, max_iters: int = 100,
                 tol: float = 1e-8) -> Dict[str, float]:
    """
    Random Walk with Restart.

    개선: 고정 반복 대신 수렴 체크 (||p_t - p_{t-1}|| < tol).
    max_iters는 수렴하지 않을 때의 상한.
    """
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    if N == 0:
        return {}

    A = np.zeros((N, N), dtype=float)
    for u, v, data in G.edges(data=True):
        w = float(data.get("weight", 1.0))
        if u in idx and v in idx:
            A[idx[u], idx[v]] += w

    # Row-normalize → transition matrix
    row_sum = A.sum(axis=1)
    # Dangling nodes: self-loop 추가 (uniform random jump 대신)
    dangling = row_sum == 0
    row_sum[dangling] = 1.0
    A[dangling, :] = 1.0 / N  # dangling nodes → uniform distribution
    row_sum[dangling] = 1.0
    P = (A.T / row_sum).T

    p0 = np.zeros(N, dtype=float)
    for n in event_nodes:
        if n in idx:
            p0[idx[n]] = 1.0
    if p0.sum() == 0:
        log.warning(f"RWR: event_nodes {event_nodes} not in graph, skipping")
        return {n: 0.0 for n in nodes}
    else:
        p0 = p0 / p0.sum()

    p = p0.copy()
    for i in range(max_iters):
        p_new = (1 - restart_prob) * (p @ P) + restart_prob * p0
        diff = np.abs(p_new - p).sum()
        p = p_new
        if diff < tol:
            log.debug(f"RWR converged in {i + 1} iterations (diff={diff:.2e})")
            break

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
    """
    Exposure 계산.

    use_undirected=True: 기존 undirected 모드 (하위 호환)
    use_undirected=False: directed graph 유지 → 상류→하류 방향성 반영

    Directed 모드에서는 추가로 upstream/downstream exposure를 분리 계산:
      - downstream: 원본 방향 (supplier→buyer) — 공급 중단이 하류에 미치는 영향
      - upstream: 역방향 — 수요 충격이 상류에 미치는 영향
    """
    Gd = build_graph(nodes, edges, weight_mode=weight_mode)

    if use_undirected:
        G = Gd.to_undirected()
        log.info(f"Exposure: undirected graph ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
    else:
        G = Gd
        log.info(f"Exposure: DIRECTED graph ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")

    # Reversed graph for upstream exposure (directed mode only)
    G_rev = Gd.reverse() if not use_undirected else None

    rows = []
    n_events = len(risk_events)
    for i, ev in enumerate(risk_events.itertuples(index=False)):
        event_id = getattr(ev, "event_id")
        sev = float(getattr(ev, "severity", 1.0))
        ents = getattr(ev, "entity_ids", [])
        if not isinstance(ents, (list, tuple)):
            try:
                ents = list(ents)
            except (TypeError, ValueError):
                ents = []

        # Downstream exposure (or undirected)
        sp = exposure_shortest_path(G, ents, sev, lam)
        rw = exposure_rwr(G, ents, sev, restart_prob=restart_prob, max_iters=iters)

        # Upstream exposure (directed mode only)
        if G_rev is not None:
            sp_up = exposure_shortest_path(G_rev, ents, sev, lam)
            rw_up = exposure_rwr(G_rev, ents, sev, restart_prob=restart_prob, max_iters=iters)
        else:
            sp_up = None
            rw_up = None

        for cid in G.nodes():
            row = {
                "event_id": event_id,
                "company_id": cid,
                "exposure_sp": sp.get(cid, 0.0),
                "exposure_rwr": rw.get(cid, 0.0),
            }
            if sp_up is not None:
                row["exposure_sp_upstream"] = sp_up.get(cid, 0.0)
                row["exposure_rwr_upstream"] = rw_up.get(cid, 0.0)
                # Combined: max of upstream and downstream
                row["exposure_rwr_combined"] = max(rw.get(cid, 0.0), rw_up.get(cid, 0.0))
            rows.append(row)

        if (i + 1) % 5000 == 0:
            log.info(f"  Exposure progress: {i + 1:,}/{n_events:,}")

    baseline_df = pd.DataFrame(rows)

    # GNN: GAT (Graph Attention Network)
    gat_df = run_gat_exposure(nodes, edges, risk_events, cfg_gnn or {})

    # Merge results
    final_df = baseline_df.merge(gat_df, on=["event_id", "company_id"], how="left")
    return final_df
