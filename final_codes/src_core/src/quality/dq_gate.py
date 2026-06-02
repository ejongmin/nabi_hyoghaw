from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from src.quality.news_qc import run_news_qc
from src.quality.prices_qc import run_prices_qc
from src.quality.entity_qc import run_entity_qc
from src.quality.kg_qc import run_kg_qc

from src.common.md import df_to_markdown


# ─────────────────────── Exception ───────────────────────

class DQGateError(Exception):
    """파이프라인 DQ Gate 통과 실패 시 발생하는 예외.
    가이드: 'DQ 리포트에 red flag가 남아있으면 다음 단계로 못 간다'
    """
    def __init__(self, failures: List[str], report: str):
        self.failures = failures
        self.report = report
        msg = (
            f"DQ Gate FAILED — {len(failures)} red flag(s) detected:\n"
            + "\n".join(f"  ✗ {f}" for f in failures)
        )
        super().__init__(msg)


# ─────────────────────── Result ───────────────────────

@dataclass
class DQResult:
    """DQ Gate 실행 결과."""
    passed: bool
    failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    report_md: str = ""

    # 개별 메트릭 (디버깅/리포트용)
    news_dedup_ratio: Optional[float] = None
    entity_link_success_rate: Optional[float] = None
    prices_missing_rate_max: Optional[float] = None
    kg_evidence_rate: Optional[float] = None


# ─────────────────────── Gate 로직 ───────────────────────

def run_dq_gate(
    articles: pd.DataFrame | None,
    prices: pd.DataFrame | None,
    risk_events: pd.DataFrame | None,
    nodes: pd.DataFrame | None,
    edges: pd.DataFrame | None,
    cfg: Dict[str, Any],
    *,
    strict: bool = True,
) -> DQResult:
    """
    DQ Gate 실행: 리포트 생성 + threshold 체크.

    Args:
        strict: True이면 threshold 미달 시 DQGateError 발생 (파이프라인 중단).
                False이면 경고만 출력하고 통과 (리포트 모드).

    Returns:
        DQResult
    """
    qg = cfg.get("quality_gate", {})
    thr_dedup = float(qg.get("news_dedup_ratio_warn", 0.35))
    thr_entity = float(qg.get("entity_link_success_min", 0.85))
    thr_prices = float(qg.get("prices_missing_rate_max", 0.03))
    thr_evidence = float(qg.get("kg_edges_need_evidence_min", 0.8))

    failures: List[str] = []
    warnings: List[str] = []
    parts = ["# Data Quality Report", ""]
    result = DQResult(passed=True)

    # ── 1. News dedup ──
    if articles is not None:
        dedup_keys = cfg.get("gdelt", {}).get("dedup_on", ["url"])
        news_qc, _ = run_news_qc(articles, dedup_keys)
        result.news_dedup_ratio = news_qc.dedup_ratio
        parts += [
            "## News (GDELT)",
            f"- rows_before: {news_qc.rows_before}",
            f"- rows_after_dedup: {news_qc.rows_after}",
            f"- dedup_ratio: {news_qc.dedup_ratio:.3f}",
            f"- threshold(warn): {thr_dedup}",
            "",
        ]
        if news_qc.dedup_ratio > thr_dedup:
            warnings.append(
                f"News dedup ratio {news_qc.dedup_ratio:.3f} > {thr_dedup} "
                f"({news_qc.rows_before - news_qc.rows_after} duplicates)"
            )
            parts.append(f"⚠️ WARNING: dedup ratio exceeds threshold\n")
    else:
        parts += ["## News (GDELT)", "- (skip) articles not found", ""]

    # ── 2. Entity linking success rate ──
    if risk_events is not None:
        eqc = run_entity_qc(risk_events)
        result.entity_link_success_rate = eqc.success_rate
        status = "✅ PASS" if eqc.success_rate >= thr_entity else "❌ FAIL"
        parts += [
            "## Entity Linking",
            f"- events: {eqc.rows}",
            f"- linked_events: {eqc.linked_rows}",
            f"- success_rate: {eqc.success_rate:.3f}",
            f"- threshold(min): {thr_entity}",
            f"- status: {status}",
            "",
        ]
        if len(eqc.top_failed_titles) > 0:
            parts += ["### Failed examples (top 10)", df_to_markdown(eqc.top_failed_titles, max_rows=10), ""]

        if eqc.success_rate < thr_entity:
            failures.append(
                f"Entity link success rate {eqc.success_rate:.3f} < {thr_entity} "
                f"(need {thr_entity*100:.0f}%+, got {eqc.success_rate*100:.1f}%)"
            )
    else:
        parts += ["## Entity Linking", "- (skip) risk_events not found", ""]

    # ── 3. Prices missing rate ──
    if prices is not None:
        pqc = run_prices_qc(prices, missing_rate_max=thr_prices)
        worst_missing = float(pqc.missing_by_company["missing_rate"].max()) if len(pqc.missing_by_company) > 0 else 0.0
        result.prices_missing_rate_max = worst_missing
        status = "✅ PASS" if worst_missing <= thr_prices else "❌ FAIL"
        parts += [
            "## Prices",
            f"- rows: {pqc.rows}",
            f"- companies: {pqc.tickers}",
            f"- worst_missing_rate: {worst_missing:.4f}",
            f"- threshold(max): {thr_prices}",
            f"- status: {status}",
            "",
        ]
        parts += ["### Missing rate (top 15)", df_to_markdown(pqc.missing_by_company, max_rows=15), ""]
        parts += ["### Outlier rate (top 15)", df_to_markdown(pqc.outlier_by_company, max_rows=15), ""]

        # 가이드: 가격 결측률 threshold 초과하는 기업 수 세기
        fail_companies = pqc.missing_by_company[pqc.missing_by_company["missing_rate"] > thr_prices]
        if len(fail_companies) > 0:
            failures.append(
                f"Prices missing rate > {thr_prices} for {len(fail_companies)} companies "
                f"(worst: {worst_missing:.4f})"
            )
    else:
        parts += ["## Prices", "- (skip) prices not found", ""]

    # ── 4. KG evidence rate ──
    if nodes is not None and edges is not None:
        allowed = cfg.get("schema", {}).get("allowed_relations", [])
        kqc = run_kg_qc(nodes, edges, allowed_relations=allowed)
        result.kg_evidence_rate = kqc.evidence_rate
        status = "✅ PASS" if kqc.evidence_rate >= thr_evidence else "❌ FAIL"
        parts += [
            "## KG",
            f"- edges: {kqc.edges}",
            f"- edges_with_evidence: {kqc.edges_with_evidence}",
            f"- evidence_rate: {kqc.evidence_rate:.3f}",
            f"- threshold(min): {thr_evidence}",
            f"- status: {status}",
            f"- invalid_relation_rows: {kqc.invalid_relation_rows}",
            f"- missing_node_rows(src/dst): {kqc.missing_node_rows}",
            "",
        ]
        if kqc.evidence_rate < thr_evidence:
            failures.append(
                f"KG evidence rate {kqc.evidence_rate:.3f} < {thr_evidence}"
            )
        if kqc.invalid_relation_rows > 0:
            warnings.append(f"KG has {kqc.invalid_relation_rows} invalid relation type rows")
        if kqc.missing_node_rows > 0:
            warnings.append(f"KG has {kqc.missing_node_rows} edges referencing missing nodes")
    else:
        parts += ["## KG", "- (skip) kg not found", ""]

    # ── Summary ──
    parts += ["---", "## Gate Summary", ""]
    if failures:
        parts.append(f"**RESULT: ❌ FAILED** ({len(failures)} red flag(s))\n")
        for f in failures:
            parts.append(f"- ❌ {f}")
    else:
        parts.append("**RESULT: ✅ PASSED**\n")

    if warnings:
        parts.append("")
        for w in warnings:
            parts.append(f"- ⚠️ {w}")

    parts.append("")

    report_md = "\n".join(parts)
    result.report_md = report_md
    result.failures = failures
    result.warnings = warnings
    result.passed = len(failures) == 0

    # strict 모드: 실패 시 예외 발생 → 파이프라인 중단
    if strict and not result.passed:
        raise DQGateError(failures, report_md)

    return result


# ─────────────────────── 하위호환: build_dq_report ───────────────────────

def build_dq_report(
    articles: pd.DataFrame | None,
    prices: pd.DataFrame | None,
    risk_events: pd.DataFrame | None,
    nodes: pd.DataFrame | None,
    edges: pd.DataFrame | None,
    cfg: Dict[str, Any],
) -> str:
    """하위호환용 래퍼. strict=False로 호출하여 리포트만 반환."""
    result = run_dq_gate(articles, prices, risk_events, nodes, edges, cfg, strict=False)
    return result.report_md
