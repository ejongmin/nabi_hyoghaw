"""
Phase 5 GNN 예측 기반 백테스트
================================
GNN prob 신호를 월별 투자 결정에 사용한 포트폴리오 성과 평가.

전략:
  bench       : 18개 기업 동일가중 (매월 리밸런싱)
  gnn_momentum: prob > 0.5인 기업 2x 오버웨이트 (GNN 긍정 신호 추종)
  gnn_contra  : prob < 0.5인 기업 2x 오버웨이트 (역투자)
  gnn_top6    : prob 상위 6개 기업만 동일가중 (집중 포트폴리오)

신호 규칙: 월 M의 prob → 월 M+1 투자 결정 (look-ahead 없음)
기간: Phase 5 test 기간 (2024-09 ~ 2025-11)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

PATH_PRED   = ROOT / "data/processed/phase5_final_predictions.parquet"
PATH_PRICES = ROOT / "data/processed/prices_daily.parquet"
PATH_MKT    = ROOT / "data/processed/market_proxies.parquet"
PATH_MKTMAP = ROOT / "data/processed/market_proxy_mapping.yaml"
PATH_C18    = ROOT / "data/universe/companies_18.csv"
OUT_RPT     = ROOT / "reports/phase5_backtest.txt"
OUT_CSV     = ROOT / "reports/phase5_backtest_detail.csv"

TRADING_DAYS = 252


def calc_metrics(monthly_rets: pd.Series) -> dict:
    rets = monthly_rets.dropna()
    if len(rets) < 2:
        return dict(cagr=np.nan, vol=np.nan, sharpe=np.nan, mdd=np.nan, n=len(rets))
    ann_ret = (1 + rets.mean()) ** 12 - 1
    ann_vol = rets.std() * np.sqrt(12)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan
    cum     = (1 + rets).cumprod()
    mdd     = float((cum / cum.cummax() - 1).min())
    return dict(cagr=ann_ret, vol=ann_vol, sharpe=sharpe, mdd=mdd, n=len(rets))


def load_monthly_returns(cids: list) -> pd.DataFrame:
    prices = pd.read_parquet(PATH_PRICES)
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices[prices["company_id"].isin(cids)].sort_values(["company_id", "date"])
    prices["ym"] = prices["date"].dt.to_period("M")
    last = prices.groupby(["company_id", "ym"])["adj_close"].last().reset_index()
    last["ret"] = last.groupby("company_id")["adj_close"].pct_change()
    return last.dropna(subset=["ret"])


def main():
    c18  = pd.read_csv(PATH_C18)
    cids = c18["company_id"].tolist()

    pred = pd.read_parquet(PATH_PRED)
    ret_df = load_monthly_returns(cids)
    ret_df["ym"] = ret_df["ym"].astype(str)

    # test 기간 신호만 사용
    test_pred = pred[pred["split"] == "test"].copy()
    signal_months = sorted(test_pred["year_month"].unique())
    print(f"신호 기간: {signal_months[0]} ~ {signal_months[-1]} ({len(signal_months)}개월)")

    records = []
    for ym_str in signal_months:
        # 신호 월 M → 투자 대상 월 M+1
        ym = pd.Period(ym_str, "M")
        next_ym = str(ym + 1)

        sig = test_pred[test_pred["year_month"] == ym_str][["company_id", "prob"]].copy()
        if sig.empty:
            continue

        month_ret = ret_df[ret_df["ym"] == next_ym].set_index("company_id")["ret"]
        available = [c for c in cids if c in month_ret.index]
        if len(available) < 3:
            continue

        sig_map = sig.set_index("company_id")["prob"].to_dict()

        strategies = {
            "bench":         {c: 1.0 for c in available},
            "gnn_momentum":  {c: (2.0 if sig_map.get(c, 0.5) > 0.5 else 1.0)
                              for c in available},
            "gnn_contra":    {c: (2.0 if sig_map.get(c, 0.5) < 0.5 else 1.0)
                              for c in available},
            "gnn_top6":      {},
        }

        # top6: prob 상위 6개 동일가중
        sorted_by_prob = sorted(available, key=lambda c: sig_map.get(c, 0.5), reverse=True)
        top6 = sorted_by_prob[:6]
        strategies["gnn_top6"] = {c: 1.0 for c in top6}

        for strat_name, raw_w in strategies.items():
            if not raw_w:
                continue
            total = sum(raw_w.values())
            w = {c: v / total for c, v in raw_w.items()}
            port_ret = sum(w[c] * month_ret.get(c, 0) for c in w)
            records.append({
                "signal_month": ym_str,
                "invest_month": next_ym,
                "strategy":     strat_name,
                "ret":          port_ret,
            })

    df = pd.DataFrame(records)
    if df.empty:
        print("백테스트 데이터 없음.")
        return

    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    # ── 성과 요약 ──────────────────────────────────────────────────
    SEP  = "=" * 66
    lines = [
        SEP,
        "Phase 5 GNN 예측 기반 백테스트 결과",
        SEP,
        "",
        f"  신호 기간: {signal_months[0]} ~ {signal_months[-1]} ({len(signal_months)}개월)",
        f"  {'전략':<15} {'CAGR':>8} {'Vol':>8} {'Sharpe':>8} {'MDD':>8} {'N월':>5}",
        "  " + "-" * 54,
    ]

    bench_sharpe = None
    results = {}
    for strat in ["bench", "gnn_momentum", "gnn_contra", "gnn_top6"]:
        g = df[df["strategy"] == strat]
        m = calc_metrics(g["ret"].reset_index(drop=True))
        results[strat] = m
        if strat == "bench":
            bench_sharpe = m["sharpe"]
        excess = "" if strat == "bench" or bench_sharpe is None else \
                 f" (vs bench: {m['sharpe']-bench_sharpe:+.3f})"
        lines.append(
            f"  {strat:<15} {m['cagr']*100:7.2f}%  {m['vol']*100:7.2f}%  "
            f"{m['sharpe']:8.3f}  {m['mdd']*100:7.2f}%  {m['n']:4d}{excess}"
        )

    lines += [
        "",
        "■ 월별 수익률 상세",
        f"  {'월':<10} {'bench':>8} {'gnn_mom':>9} {'gnn_con':>9} {'gnn_top6':>10}",
        "  " + "-" * 44,
    ]
    pivot = df.pivot_table(index="invest_month", columns="strategy", values="ret")
    for ym, row in pivot.iterrows():
        b = row.get("bench", np.nan)
        m = row.get("gnn_momentum", np.nan)
        c = row.get("gnn_contra", np.nan)
        t = row.get("gnn_top6", np.nan)
        lines.append(f"  {ym:<10} {b*100:7.2f}%  {m*100:7.2f}%  {c*100:7.2f}%  {t*100:8.2f}%")

    lines.append("")
    lines.append("■ 해석")
    best = max(results, key=lambda k: results[k].get("sharpe", -999) or -999)
    lines.append(f"  최고 Sharpe 전략: {best} (Sharpe={results[best]['sharpe']:.3f})")
    lines.append(f"  벤치마크 Sharpe:  {results['bench']['sharpe']:.3f}")
    lines.append(SEP)

    rpt = "\n".join(lines)
    OUT_RPT.write_text(rpt, encoding="utf-8")
    print("\n" + rpt)


if __name__ == "__main__":
    main()
