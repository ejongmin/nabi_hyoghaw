#!/usr/bin/env python3
"""
나머지 파이프라인 실행 스크립트.
risk_events + prices 완료 상태에서 이어서 실행합니다.

실행: python run_pipeline_remaining.py

단계:
  1. aggregate_events  (400만→ ~10만 집계)
  2. build-kg          (seed_edges → KG)
  3. DQ Gate           (품질 검증, warning 모드)
  4. compute-exposure  (RWR + Shortest Path)
  5. event-study       (CAR 계산)
  6. backtest          (포트폴리오 백테스트)
  7. report-all        (최종 리포트)
"""
import sys
import logging
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("pipeline")

import yaml
import pandas as pd

from src.ingest.aggregate_events import aggregate_risk_events
from src.kg.build_static import load_universe, load_seed_edges, build_static_kg
from src.quality.dq_gate import run_dq_gate, DQGateError
from src.graph.baseline_exposure import compute_exposure
from src.finance.event_study import run_event_study
from src.finance.backtest import backtest_exclude, backtest_tilt
from src.reporting.report_all import build_index, write_md
from src.common.io import read_df, write_df


def main():
    with open("configs/base.yaml") as f:
        cfg = yaml.safe_load(f)

    root = Path(".")

    # ── Paths ──
    risk_raw_path = root / cfg["paths"]["risk_events_parquet"]
    risk_agg_path = root / "data/processed/risk_events_agg.parquet"
    prices_path = root / cfg["paths"]["prices_parquet"]
    seed_path = root / cfg["paths"]["seed_edges_csv"]
    uni_path = root / cfg["paths"]["universe_csv"]
    nodes_path = root / cfg["paths"]["kg_nodes_parquet"]
    edges_path = root / cfg["paths"]["kg_edges_parquet"]
    exp_path = root / cfg["paths"]["exposure_parquet"]
    car_path = root / cfg["paths"]["car_panel_parquet"]
    backtest_csv = root / cfg["paths"]["backtest_equity_csv"]
    reports_dir = root / cfg["paths"]["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Aggregate Events ──
    log.info("=" * 60)
    log.info("Step 1: Aggregate risk events")
    log.info("=" * 60)
    risk_agg = aggregate_risk_events(risk_raw_path, risk_agg_path, force=True)
    log.info(f"Aggregated: {len(risk_agg):,} events")

    # ── Step 2: Build KG ──
    log.info("=" * 60)
    log.info("Step 2: Build Knowledge Graph")
    log.info("=" * 60)
    uni = pd.read_csv(uni_path)
    seed = pd.read_csv(seed_path)
    allowed = cfg.get("schema", {}).get("allowed_relations", [])
    nodes, edges = build_static_kg(uni, seed, allowed_relations=allowed)
    write_df(nodes, nodes_path)
    write_df(edges, edges_path)
    log.info(f"KG: {len(nodes)} nodes, {len(edges)} edges")

    # ── Step 3: DQ Gate (warning mode) ──
    log.info("=" * 60)
    log.info("Step 3: DQ Gate (warning mode)")
    log.info("=" * 60)
    prices = read_df(prices_path)
    try:
        dq = run_dq_gate(
            articles=None,  # BQ 모드
            prices=prices,
            risk_events=risk_agg,
            nodes=nodes,
            edges=edges,
            cfg=cfg,
            strict=False,  # warning only
        )
        (reports_dir / "dq_report.md").write_text(dq.report_md, encoding="utf-8")
        if dq.passed:
            log.info("✅ DQ Gate PASSED")
        else:
            log.warning(f"⚠️ DQ Gate: {len(dq.failures)} warnings (non-blocking)")
            for f in dq.failures:
                log.warning(f"  ⚠️ {f}")
    except Exception as e:
        log.error(f"DQ Gate error (non-fatal): {e}")

    # ── Step 4: Compute Exposure ──
    log.info("=" * 60)
    log.info("Step 4: Compute Exposure (RWR + Shortest Path)")
    log.info("=" * 60)
    use_undirected = bool(cfg.get("kg", {}).get("use_undirected_for_exposure", True))
    lam = float(cfg.get("exposure", {}).get("baseline", {}).get("lambda_dist", 0.7))
    rp = float(cfg.get("exposure", {}).get("baseline", {}).get("rwr_restart_prob", 0.15))
    iters = int(cfg.get("exposure", {}).get("baseline", {}).get("rwr_iters", 50))
    weight_mode = str(cfg.get("kg", {}).get("edge_weight", {}).get("weight_mode", "confidence_times_strength"))
    cfg_gnn = cfg.get("gnn", {})

    exp = compute_exposure(nodes, edges, risk_agg, use_undirected, lam, rp, iters, weight_mode, cfg_gnn=cfg_gnn)
    write_df(exp, exp_path)
    log.info(f"Exposure: {len(exp):,} rows")

    # ── Step 5: Event Study (CAR) ──
    log.info("=" * 60)
    log.info("Step 5: Event Study (CAR)")
    log.info("=" * 60)
    windows = cfg.get("finance", {}).get("car", {}).get("event_windows_trading_days", [21, 63, 126, 252])
    est_win = tuple(cfg.get("finance", {}).get("car", {}).get("estimation_window", [-120, -20]))
    topk = int(cfg.get("finance", {}).get("car", {}).get("topk_companies_per_event", 15))

    car = run_event_study(prices, risk_agg, exp, windows=windows, est_win=est_win, topk=topk)
    write_df(car, car_path)
    log.info(f"CAR panel: {len(car):,} rows")

    # ── Step 6: Backtest ──
    log.info("=" * 60)
    log.info("Step 6: Backtest")
    log.info("=" * 60)
    rebalance = cfg.get("finance", {}).get("backtest", {}).get("rebalance", "W-FRI")
    excl_q = float(cfg.get("finance", {}).get("backtest", {}).get("exclude_quantile", 0.2))
    decay_lam = float(cfg.get("finance", {}).get("backtest", {}).get("risk_decay_lambda", 0.03))

    # --- Strategy 1: Exclude ---
    log.info("  Strategy 1: Exclude (remove high-risk stocks)")
    equity, metrics = backtest_exclude(prices, exp, risk_agg, exclude_quantile=excl_q, decay_lambda=decay_lam)
    backtest_csv.parent.mkdir(parents=True, exist_ok=True)
    equity.to_csv(backtest_csv, index=False, encoding="utf-8-sig")

    md = ["# Backtest summary - Exclude strategy", ""]
    for k, v in metrics.items():
        md.append(f"- {k}: {v:.4f}")
    write_md(reports_dir / "backtest_summary.md", "\n".join(md) + "\n")
    log.info(f"  Exclude: {len(equity):,} rows, metrics saved")

    # --- Strategy 2: Tilt (inverse-risk weighting) ---
    log.info("  Strategy 2: Tilt (inverse-risk weighting)")
    backtest_tilt_csv = backtest_csv.parent / "backtest_equity_tilt.csv"
    equity_tilt, metrics_tilt = backtest_tilt(prices, exp, risk_agg, decay_lambda=decay_lam)
    equity_tilt.to_csv(backtest_tilt_csv, index=False, encoding="utf-8-sig")

    md_tilt = ["# Backtest summary - Tilt strategy", ""]
    for k, v in metrics_tilt.items():
        md_tilt.append(f"- {k}: {v:.4f}")
    write_md(reports_dir / "backtest_summary_tilt.md", "\n".join(md_tilt) + "\n")
    log.info(f"  Tilt: {len(equity_tilt):,} rows, metrics saved")

    # --- Comparison ---
    log.info("  Strategy comparison (Exclude vs Tilt):")
    for key in ["strat_cagr", "strat_vol", "strat_sharpe", "strat_mdd", "avg_weekly_turnover", "total_tx_cost_bps"]:
        v_exc = metrics.get(key, float("nan"))
        v_tlt = metrics_tilt.get(key, float("nan"))
        log.info(f"    {key:25s}  exclude={v_exc:+.4f}  tilt={v_tlt:+.4f}")

    md_cmp = ["# Backtest comparison: Exclude vs Tilt", ""]
    md_cmp.append("| Metric | Exclude | Tilt |")
    md_cmp.append("|--------|---------|------|")
    for key in sorted(set(list(metrics.keys()) + list(metrics_tilt.keys()))):
        v_exc = metrics.get(key, float("nan"))
        v_tlt = metrics_tilt.get(key, float("nan"))
        md_cmp.append(f"| {key} | {v_exc:.4f} | {v_tlt:.4f} |")
    write_md(reports_dir / "backtest_comparison.md", "\n".join(md_cmp) + "\n")
    log.info(f"  Comparison report saved")

    # ── Step 7: Report ──
    log.info("=" * 60)
    log.info("Step 7: Generate Reports")
    log.info("=" * 60)
    idx = build_index(reports_dir)
    write_md(reports_dir / "index.md", idx)
    log.info(f"Report index: {reports_dir / 'index.md'}")

    # ── Summary ──
    log.info("=" * 60)
    log.info("🏁 PIPELINE COMPLETE")
    log.info(f"   Risk events (aggregated): {len(risk_agg):,}")
    log.info(f"   KG: {len(nodes)} nodes, {len(edges)} edges")
    log.info(f"   Exposure: {len(exp):,} rows")
    log.info(f"   CAR panel: {len(car):,} rows")
    log.info(f"   Backtest equity (exclude): {len(equity):,} rows")
    log.info(f"   Backtest equity (tilt):    {len(equity_tilt):,} rows")
    log.info("   --- Exclude metrics ---")
    for k, v in metrics.items():
        log.info(f"   {k}: {v:.4f}")
    log.info("   --- Tilt metrics ---")
    for k, v in metrics_tilt.items():
        log.info(f"   {k}: {v:.4f}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
