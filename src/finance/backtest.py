from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

from src.common.dates import next_trading_day
from src.finance.metrics import max_drawdown, annualized_return, annualized_vol, sharpe

import logging
log = logging.getLogger("backtest")


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values(["company_id", "date"])
    df["ret"] = df.groupby("company_id")["close"].pct_change()
    return df.dropna(subset=["ret"])


def build_daily_risk(exposure: pd.DataFrame, risk_events: pd.DataFrame,
                     decay_lambda: float = 0.03) -> pd.DataFrame:
    """Risk score per day per company: exponential decay + event additions."""
    ev = risk_events[["event_id", "event_time"]].copy()
    ev["event_time"] = pd.to_datetime(ev["event_time"], errors="coerce")
    ev = ev.dropna(subset=["event_time"])
    ev["event_date"] = ev["event_time"].dt.normalize()

    exp = exposure.merge(ev[["event_id", "event_date"]], on="event_id", how="inner")
    exp = exp[["event_date", "company_id", "exposure_rwr"]].copy()
    exp["exposure_rwr"] = pd.to_numeric(exp["exposure_rwr"], errors="coerce").fillna(0.0)

    additions = exp.groupby(["event_date", "company_id"])["exposure_rwr"].sum().reset_index(name="add")
    dates = pd.date_range(additions["event_date"].min(), additions["event_date"].max(), freq="B")
    companies = sorted(additions["company_id"].unique().tolist())
    add_piv = additions.pivot(index="event_date", columns="company_id", values="add").reindex(dates).fillna(0.0)

    decay = float(np.exp(-decay_lambda))
    risk = pd.DataFrame(index=dates, columns=companies, data=0.0)
    prev = np.zeros(len(companies), dtype=float)
    for i, d in enumerate(dates):
        add_vec = add_piv.loc[d].to_numpy(dtype=float)
        cur = prev * decay + add_vec
        risk.loc[d] = cur
        prev = cur
    risk = risk.reset_index().rename(columns={"index": "date"})
    long = risk.melt(id_vars=["date"], var_name="company_id", value_name="risk")
    return long


def backtest_exclude(prices: pd.DataFrame,
                     exposure: pd.DataFrame,
                     risk_events: pd.DataFrame,
                     exclude_quantile: float = 0.2,
                     decay_lambda: float = 0.03,
                     tx_cost_bps: float = 10.0) -> tuple[pd.DataFrame, Dict[str, float]]:
    """
    Risk-based exclusion backtest.

    개선사항 (v3):
      - Lookahead bias 수정: 금요일 리밸런스 → 다음 거래일(월요일)부터 적용
      - 거래비용 모델링: 편도 tx_cost_bps (기본 10bps = 0.1%)
      - 턴오버 추적: 리밸런스마다 포트폴리오 변경률 기록

    Args:
        tx_cost_bps: 편도 거래비용 (basis points). 10 = 0.1%.
                     리밸런스 시 변경 종목에 대해 양방향(매수+매도) 적용.
    """
    rets = compute_returns(prices)
    # benchmark: equal weight
    bench = rets.groupby("date")["ret"].mean().rename("bench_ret").to_frame()
    bench["bench_equity"] = (1 + bench["bench_ret"]).cumprod()

    # risk per day
    risk_daily = build_daily_risk(exposure, risk_events, decay_lambda=decay_lambda)
    df = rets.merge(risk_daily, on=["date", "company_id"], how="left")
    df["risk"] = df["risk"].fillna(0.0)

    # Weekly rebalance (Friday) → weights applied from NEXT trading day (Monday)
    dates = sorted(df["date"].unique())
    dates = pd.to_datetime(dates)
    dates_arr = np.array(dates, dtype="datetime64[ns]")

    fridays = [d for d in dates if pd.Timestamp(d).weekday() == 4]
    if len(fridays) == 0:
        fridays = pd.Series(dates).resample("W-FRI").max().dropna().tolist()
    fridays = [np.datetime64(f, "ns") for f in fridays]

    companies = sorted(df["company_id"].unique().tolist())
    tx_cost_frac = tx_cost_bps / 10000.0  # bps → fraction

    # Build weight table with 1-day lag
    weights = []
    prev_include = set(companies)  # 초기: 전체 포함

    for i, rb in enumerate(fridays):
        # 리밸런스 결정일: rb (금요일)
        # 적용 시작일: 다음 거래일 (월요일 or 그 이후)
        rb_idx_in_dates = np.searchsorted(dates_arr, rb, side="right")
        if rb_idx_in_dates >= len(dates_arr):
            continue
        apply_from = dates_arr[rb_idx_in_dates]  # 다음 거래일

        # 적용 종료일: 다음 리밸런스의 apply_from 직전까지
        if i + 1 < len(fridays):
            next_rb = fridays[i + 1]
            next_rb_idx = np.searchsorted(dates_arr, next_rb, side="right")
            if next_rb_idx < len(dates_arr):
                end_hold = dates_arr[next_rb_idx]
            else:
                end_hold = dates_arr[-1]
        else:
            end_hold = dates_arr[-1]

        # 리밸런스일 기준 리스크 스냅샷
        snap = df[df["date"] == rb][["company_id", "risk"]].copy()
        if snap.empty:
            continue
        thresh = snap["risk"].quantile(1 - exclude_quantile)
        include = set(snap[snap["risk"] <= thresh]["company_id"].tolist())
        if len(include) == 0:
            include = set(companies)

        # 턴오버 계산 (변경 종목 비율)
        added = include - prev_include
        removed = prev_include - include
        turnover = (len(added) + len(removed)) / max(len(companies), 1)

        w = 1.0 / len(include)
        weights.append({
            "date": apply_from,  # ← lookahead bias 수정: 다음 거래일부터 적용
            "include": list(include),
            "w": w,
            "end_hold": end_hold,
            "turnover": turnover,
        })
        prev_include = include

    weights_df = pd.DataFrame(weights)
    if weights_df.empty:
        out = bench.copy()
        out["strat_ret"] = out["bench_ret"]
        out["strat_equity"] = out["bench_equity"]
        metrics = {}
        return out.reset_index(), metrics

    weights_df = weights_df.sort_values("date")
    reb_dates = weights_df["date"].tolist()

    strat_daily = []
    prev_include_set = set(companies)
    total_tx_cost = 0.0

    for d in dates:
        rb_idx = np.searchsorted(reb_dates, d, side="right") - 1

        if rb_idx < 0:
            # 첫 리밸런스 전: 전체 equal weight
            day = df[df["date"] == d]
            strat_ret = day["ret"].mean() if not day.empty else 0.0
            tx = 0.0
        else:
            row = weights_df.iloc[rb_idx]
            include = set(row["include"])
            w = float(row["w"])
            day = df[(df["date"] == d) & (df["company_id"].isin(include))]
            strat_ret = float((day["ret"] * w).sum()) if not day.empty else 0.0

            # 거래비용: 리밸런스 당일(적용 시작일)에만 부과
            if d == reb_dates[rb_idx]:
                changed = (include - prev_include_set) | (prev_include_set - include)
                # 변경 종목 비율 × 양방향(매수+매도) × 편도 비용
                tx = 2.0 * tx_cost_frac * len(changed) / max(len(companies), 1)
                total_tx_cost += tx
                prev_include_set = include
            else:
                tx = 0.0

            strat_ret -= tx  # 거래비용 차감

        strat_daily.append({"date": d, "strat_ret": float(strat_ret)})

    strat = pd.DataFrame(strat_daily).set_index("date")
    out = bench.join(strat, how="inner")
    out["strat_equity"] = (1 + out["strat_ret"]).cumprod()

    avg_turnover = weights_df["turnover"].mean() if not weights_df.empty else 0.0

    metrics = {
        "bench_cagr": annualized_return(out["bench_equity"]),
        "bench_vol": annualized_vol(out["bench_ret"]),
        "bench_sharpe": sharpe(out["bench_ret"]),
        "bench_mdd": max_drawdown(out["bench_equity"]),
        "strat_cagr": annualized_return(out["strat_equity"]),
        "strat_vol": annualized_vol(out["strat_ret"]),
        "strat_sharpe": sharpe(out["strat_ret"]),
        "strat_mdd": max_drawdown(out["strat_equity"]),
        "avg_weekly_turnover": avg_turnover,
        "total_tx_cost_bps": total_tx_cost * 10000,
        "tx_cost_per_trade_bps": tx_cost_bps,
    }

    log.info(f"Backtest: strat_cagr={metrics['strat_cagr']:.4f}, "
             f"bench_cagr={metrics['bench_cagr']:.4f}, "
             f"turnover={avg_turnover:.3f}, tx_cost={total_tx_cost*10000:.1f}bps total")

    return out.reset_index(), metrics


def backtest_tilt(prices: pd.DataFrame,
                  exposure: pd.DataFrame,
                  risk_events: pd.DataFrame,
                  decay_lambda: float = 0.03,
                  tx_cost_bps: float = 10.0,
                  epsilon: float = 0.01) -> tuple[pd.DataFrame, Dict[str, float]]:
    """
    Risk-tilt backtest: inverse-risk weighting instead of exclusion.

    High-risk stocks are underweighted, low-risk stocks are overweighted.
    Weight formula: w_i = (1 / max(risk_i, epsilon)) / sum(1 / max(risk_j, epsilon))

    Same weekly rebalance schedule as backtest_exclude:
      - Friday decision -> next trading day (Monday) application (1-day lag)
      - Transaction cost tracking (default 10bps one-way)

    Args:
        epsilon: Small value to avoid division by zero for zero-risk stocks.
        tx_cost_bps: One-way transaction cost in basis points.

    Returns:
        (equity_df, metrics_dict) in the same format as backtest_exclude.
    """
    rets = compute_returns(prices)
    # benchmark: equal weight
    bench = rets.groupby("date")["ret"].mean().rename("bench_ret").to_frame()
    bench["bench_equity"] = (1 + bench["bench_ret"]).cumprod()

    # risk per day
    risk_daily = build_daily_risk(exposure, risk_events, decay_lambda=decay_lambda)
    df = rets.merge(risk_daily, on=["date", "company_id"], how="left")
    df["risk"] = df["risk"].fillna(0.0)

    # Weekly rebalance (Friday) -> weights applied from NEXT trading day (Monday)
    dates = sorted(df["date"].unique())
    dates = pd.to_datetime(dates)
    dates_arr = np.array(dates, dtype="datetime64[ns]")

    fridays = [d for d in dates if pd.Timestamp(d).weekday() == 4]
    if len(fridays) == 0:
        fridays = pd.Series(dates).resample("W-FRI").max().dropna().tolist()
    fridays = [np.datetime64(f, "ns") for f in fridays]

    companies = sorted(df["company_id"].unique().tolist())
    n_companies = len(companies)
    tx_cost_frac = tx_cost_bps / 10000.0

    # Build weight table with 1-day lag
    weights_list = []
    prev_weights = {c: 1.0 / max(n_companies, 1) for c in companies}  # initial: equal weight

    for i, rb in enumerate(fridays):
        # Rebalance decision date: rb (Friday)
        # Apply from: next trading day (Monday or later)
        rb_idx_in_dates = np.searchsorted(dates_arr, rb, side="right")
        if rb_idx_in_dates >= len(dates_arr):
            continue
        apply_from = dates_arr[rb_idx_in_dates]

        # Hold until next rebalance apply date
        if i + 1 < len(fridays):
            next_rb = fridays[i + 1]
            next_rb_idx = np.searchsorted(dates_arr, next_rb, side="right")
            if next_rb_idx < len(dates_arr):
                end_hold = dates_arr[next_rb_idx]
            else:
                end_hold = dates_arr[-1]
        else:
            end_hold = dates_arr[-1]

        # Risk snapshot on rebalance date
        snap = df[df["date"] == rb][["company_id", "risk"]].copy()
        if snap.empty:
            continue

        # Inverse-risk weighting: w_i = (1/max(risk_i, epsilon)) / sum(1/max(risk_j, epsilon))
        snap["inv_risk"] = 1.0 / np.maximum(snap["risk"].values, epsilon)
        total_inv = snap["inv_risk"].sum()
        snap["weight"] = snap["inv_risk"] / total_inv

        new_weights = dict(zip(snap["company_id"], snap["weight"]))

        # Turnover: sum of absolute weight changes
        turnover = sum(
            abs(new_weights.get(c, 0.0) - prev_weights.get(c, 0.0))
            for c in set(list(new_weights.keys()) + list(prev_weights.keys()))
        )

        weights_list.append({
            "date": apply_from,
            "weights": new_weights,
            "end_hold": end_hold,
            "turnover": turnover,
        })
        prev_weights = new_weights

    weights_df = pd.DataFrame(weights_list)
    if weights_df.empty:
        out = bench.copy()
        out["strat_ret"] = out["bench_ret"]
        out["strat_equity"] = out["bench_equity"]
        metrics = {}
        return out.reset_index(), metrics

    weights_df = weights_df.sort_values("date")
    reb_dates = weights_df["date"].tolist()

    strat_daily = []
    total_tx_cost = 0.0

    for d in dates:
        rb_idx = np.searchsorted(reb_dates, d, side="right") - 1

        if rb_idx < 0:
            # Before first rebalance: equal weight all
            day = df[df["date"] == d]
            strat_ret = day["ret"].mean() if not day.empty else 0.0
            tx = 0.0
        else:
            row = weights_df.iloc[rb_idx]
            w_map = row["weights"]
            day = df[df["date"] == d].copy()
            day["w"] = day["company_id"].map(w_map).fillna(0.0)
            strat_ret = float((day["ret"] * day["w"]).sum()) if not day.empty else 0.0

            # Transaction cost on rebalance application day
            if d == reb_dates[rb_idx]:
                tx = row["turnover"] * tx_cost_frac
                total_tx_cost += tx
            else:
                tx = 0.0

            strat_ret -= tx

        strat_daily.append({"date": d, "strat_ret": float(strat_ret)})

    strat = pd.DataFrame(strat_daily).set_index("date")
    out = bench.join(strat, how="inner")
    out["strat_equity"] = (1 + out["strat_ret"]).cumprod()

    avg_turnover = weights_df["turnover"].mean() if not weights_df.empty else 0.0

    metrics = {
        "bench_cagr": annualized_return(out["bench_equity"]),
        "bench_vol": annualized_vol(out["bench_ret"]),
        "bench_sharpe": sharpe(out["bench_ret"]),
        "bench_mdd": max_drawdown(out["bench_equity"]),
        "strat_cagr": annualized_return(out["strat_equity"]),
        "strat_vol": annualized_vol(out["strat_ret"]),
        "strat_sharpe": sharpe(out["strat_ret"]),
        "strat_mdd": max_drawdown(out["strat_equity"]),
        "avg_weekly_turnover": avg_turnover,
        "total_tx_cost_bps": total_tx_cost * 10000,
        "tx_cost_per_trade_bps": tx_cost_bps,
    }

    log.info(f"Backtest (tilt): strat_cagr={metrics['strat_cagr']:.4f}, "
             f"bench_cagr={metrics['bench_cagr']:.4f}, "
             f"turnover={avg_turnover:.3f}, tx_cost={total_tx_cost*10000:.1f}bps total")

    return out.reset_index(), metrics
