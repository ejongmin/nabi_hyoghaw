from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

from src.common.dates import next_trading_day
from src.finance.metrics import max_drawdown, annualized_return, annualized_vol, sharpe

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values(["company_id","date"])
    df["ret"] = df.groupby("company_id")["close"].pct_change()
    return df.dropna(subset=["ret"])

def build_daily_risk(exposure: pd.DataFrame, risk_events: pd.DataFrame, decay_lambda: float = 0.03) -> pd.DataFrame:
    """Risk score per day per company: exponential decay + event additions."""
    # prepare event_date
    ev = risk_events[["event_id","event_time"]].copy()
    ev["event_time"] = pd.to_datetime(ev["event_time"], errors="coerce")
    ev = ev.dropna(subset=["event_time"])
    ev["event_date"] = ev["event_time"].dt.normalize()

    exp = exposure.merge(ev[["event_id","event_date"]], on="event_id", how="inner")
    exp = exp[["event_date","company_id","exposure_rwr"]].copy()
    exp["exposure_rwr"] = pd.to_numeric(exp["exposure_rwr"], errors="coerce").fillna(0.0)

    additions = exp.groupby(["event_date","company_id"])["exposure_rwr"].sum().reset_index(name="add")
    # build full date range
    dates = pd.date_range(additions["event_date"].min(), additions["event_date"].max(), freq="B")
    companies = sorted(additions["company_id"].unique().tolist())
    # pivot for fast lookup
    add_piv = additions.pivot(index="event_date", columns="company_id", values="add").reindex(dates).fillna(0.0)

    # decay factor per day
    decay = float(np.exp(-decay_lambda))
    risk = pd.DataFrame(index=dates, columns=companies, data=0.0)
    prev = np.zeros(len(companies), dtype=float)
    for i, d in enumerate(dates):
        add_vec = add_piv.loc[d].to_numpy(dtype=float)
        cur = prev * decay + add_vec
        risk.loc[d] = cur
        prev = cur
    risk = risk.reset_index().rename(columns={"index":"date"})
    long = risk.melt(id_vars=["date"], var_name="company_id", value_name="risk")
    return long

def backtest_exclude(prices: pd.DataFrame,
                     exposure: pd.DataFrame,
                     risk_events: pd.DataFrame,
                     exclude_quantile: float = 0.2,
                     decay_lambda: float = 0.03) -> tuple[pd.DataFrame, Dict[str,float]]:
    rets = compute_returns(prices)
    # benchmark: equal weight each day among available returns
    bench = rets.groupby("date")["ret"].mean().rename("bench_ret").to_frame()
    bench["bench_equity"] = (1 + bench["bench_ret"]).cumprod()

    # risk per day
    risk_daily = build_daily_risk(exposure, risk_events, decay_lambda=decay_lambda)
    # join returns
    df = rets.merge(risk_daily, on=["date","company_id"], how="left")
    df["risk"] = df["risk"].fillna(0.0)

    # weekly rebalance (Friday): use risk on that date to set weights for next week
    dates = sorted(df["date"].unique())
    dates = pd.to_datetime(dates)
    # find Fridays among trading days
    fridays = [d for d in dates if pd.Timestamp(d).weekday() == 4]
    if len(fridays) == 0:
        # fallback: last day of each week (resample)
        fridays = pd.Series(dates).to_series().resample("W-FRI").max().dropna().tolist()

    # Build weight table: date -> company -> weight
    weights = []
    companies = sorted(df["company_id"].unique().tolist())

    for i, rb in enumerate(fridays):
        # determine holding period end (next Friday or end)
        start_hold = rb  # rebalance on rb close, apply next trading day ideally; MVP uses same-day for simplicity
        end_hold = fridays[i+1] if i+1 < len(fridays) else dates[-1]

        snap = df[df["date"] == rb][["company_id","risk"]].copy()
        if snap.empty:
            continue
        thresh = snap["risk"].quantile(1 - exclude_quantile)
        include = snap[snap["risk"] <= thresh]["company_id"].tolist()
        if len(include) == 0:
            include = companies
        w = 1.0 / len(include)
        weights.append({"date": rb, "include": include, "w": w, "end_hold": end_hold})

    # apply weights day by day: use latest rebalance <= date
    weights_df = pd.DataFrame(weights)
    if weights_df.empty:
        out = bench.copy()
        out["strat_ret"] = out["bench_ret"]
        out["strat_equity"] = out["bench_equity"]
        metrics = {}
        return out.reset_index(), metrics

    weights_df = weights_df.sort_values("date")
    # create dict of rebalance date -> include list
    reb_dates = weights_df["date"].tolist()

    strat_daily = []
    for d in dates:
        # latest rebalance <= d
        rb_idx = np.searchsorted(reb_dates, d, side="right") - 1
        if rb_idx < 0:
            # before first rebalance: equal weight
            day = df[df["date"] == d]
            strat_ret = day["ret"].mean() if not day.empty else 0.0
        else:
            include = weights_df.iloc[rb_idx]["include"]
            w = float(weights_df.iloc[rb_idx]["w"])
            day = df[(df["date"] == d) & (df["company_id"].isin(include))]
            strat_ret = float((day["ret"] * w).sum()) if not day.empty else 0.0
        strat_daily.append({"date": d, "strat_ret": float(strat_ret)})

    strat = pd.DataFrame(strat_daily).set_index("date")
    out = bench.join(strat, how="inner")
    out["strat_equity"] = (1 + out["strat_ret"]).cumprod()

    metrics = {
        "bench_cagr": annualized_return(out["bench_equity"]),
        "bench_vol": annualized_vol(out["bench_ret"]),
        "bench_sharpe": sharpe(out["bench_ret"]),
        "bench_mdd": max_drawdown(out["bench_equity"]),
        "strat_cagr": annualized_return(out["strat_equity"]),
        "strat_vol": annualized_vol(out["strat_ret"]),
        "strat_sharpe": sharpe(out["strat_ret"]),
        "strat_mdd": max_drawdown(out["strat_equity"]),
    }
    return out.reset_index(), metrics
