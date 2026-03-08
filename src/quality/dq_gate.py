from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import pandas as pd

from src.quality.news_qc import run_news_qc
from src.quality.prices_qc import run_prices_qc
from src.quality.entity_qc import run_entity_qc
from src.quality.kg_qc import run_kg_qc

from src.common.md import df_to_markdown

@dataclass
class DQReport:
    news: Any
    prices: Any
    entity: Any
    kg: Any

def build_dq_report(articles: pd.DataFrame | None,
                    prices: pd.DataFrame | None,
                    risk_events: pd.DataFrame | None,
                    nodes: pd.DataFrame | None,
                    edges: pd.DataFrame | None,
                    cfg: Dict[str, Any]) -> str:
    parts = ["# Data Quality Report", ""]

    # News
    if articles is not None:
        dedup_keys = cfg.get("gdelt", {}).get("dedup_on", ["url"])
        news_qc, _ = run_news_qc(articles, dedup_keys)
        parts += ["## News (GDELT)", f"- rows_before: {news_qc.rows_before}", f"- rows_after_dedup: {news_qc.rows_after}", f"- dedup_ratio: {news_qc.dedup_ratio:.3f}", ""]
    else:
        parts += ["## News (GDELT)", "- (skip) articles not found", ""]

    # Prices
    if prices is not None:
        thr = float(cfg.get("quality_gate", {}).get("prices_missing_rate_max", 0.03))
        pqc = run_prices_qc(prices, missing_rate_max=thr)
        parts += ["## Prices", f"- rows: {pqc.rows}", f"- companies: {pqc.tickers}", ""]
        parts += ["### Missing rate (top 15)", df_to_markdown(pqc.missing_by_company, max_rows=15), ""]
        parts += ["### Outlier rate (top 15)", df_to_markdown(pqc.outlier_by_company, max_rows=15), ""]
    else:
        parts += ["## Prices", "- (skip) prices not found", ""]

    # Entity link
    if risk_events is not None:
        eqc = run_entity_qc(risk_events)
        parts += ["## Entity Linking", f"- events: {eqc.rows}", f"- linked_events: {eqc.linked_rows}", f"- success_rate: {eqc.success_rate:.3f}", ""]
        if len(eqc.top_failed_titles) > 0:
            parts += ["### Failed examples (top 10)", df_to_markdown(eqc.top_failed_titles, max_rows=10), ""]
    else:
        parts += ["## Entity Linking", "- (skip) risk_events not found", ""]

    # KG
    if nodes is not None and edges is not None:
        allowed = cfg.get("schema", {}).get("allowed_relations", [])
        kqc = run_kg_qc(nodes, edges, allowed_relations=allowed)
        parts += ["## KG", f"- edges: {kqc.edges}", f"- edges_with_evidence: {kqc.edges_with_evidence}", f"- evidence_rate: {kqc.evidence_rate:.3f}",
                  f"- invalid_relation_rows: {kqc.invalid_relation_rows}", f"- missing_node_rows(src/dst): {kqc.missing_node_rows}", ""]
    else:
        parts += ["## KG", "- (skip) kg not found", ""]

    return "\n".join(parts) + "\n"
