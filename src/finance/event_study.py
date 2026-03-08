from __future__ import annotations
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
import statsmodels.api as sm

from src.common.dates import next_trading_day

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values(["company_id","date"])
    df["ret"] = df.groupby("company_id")["close"].pct_change()
    return df.dropna(subset=["ret"])

def build_market_proxy(rets: pd.DataFrame) -> pd.Series:
    mkt = rets.groupby("date")["ret"].mean()
    mkt.name = "mkt_ret"
    return mkt

def compute_car_for_company(rets_one: pd.DataFrame,
                            mkt: pd.Series,
                            event_time: pd.Timestamp,
                            est_win: Tuple[int,int],
                            evt_len: int) -> Optional[float]:
    # align series
    df = rets_one.set_index("date")[["ret"]].join(mkt.to_frame(), how="inner").dropna()
    if df.empty:
        return None
    # sorted unique trading days
    trading_days = df.index.values.astype("datetime64[D]")
    trade_day = next_trading_day(trading_days, event_time)
    if trade_day is None or trade_day not in df.index:
        return None
    idx = df.index.get_indexer([trade_day])[0]
    if idx < 0:
        return None

    est_start = idx + est_win[0]
    est_end = idx + est_win[1]
    if est_start < 0 or est_end <= est_start:
        return None
    est = df.iloc[est_start:est_end].copy()
    if len(est) < 30:
        return None

    X = sm.add_constant(est["mkt_ret"])
    y = est["ret"]
    model = sm.OLS(y, X).fit()

    evt = df.iloc[idx: idx + evt_len].copy()
    if len(evt) < evt_len:
        return None
    X_evt = sm.add_constant(evt["mkt_ret"])
    exp = model.predict(X_evt)
    ar = evt["ret"] - exp
    return float(ar.sum())

def run_event_study(prices: pd.DataFrame,
                    risk_events: pd.DataFrame,
                    exposure: pd.DataFrame,
                    windows: List[int],
                    est_win: Tuple[int,int],
                    topk: int = 15) -> pd.DataFrame:
    rets = compute_returns(prices)
    mkt = build_market_proxy(rets)

    # choose companies per event: mentioned + topK exposure_rwr
    exp = exposure.copy()
    if "exposure_rwr" not in exp.columns:
        exp["exposure_rwr"] = exp.get("exposure_sp", 0.0)

    event_to_companies = {}
    for ev_id, g in exp.groupby("event_id"):
        top = g.sort_values("exposure_rwr", ascending=False).head(topk)["company_id"].tolist()
        event_to_companies[ev_id] = set(top)

    # add directly mentioned companies
    for r in risk_events.itertuples(index=False):
        ev_id = getattr(r, "event_id")
        ents = getattr(r, "entity_ids", [])
        if not isinstance(ents, list):
            ents = []
        if ev_id not in event_to_companies:
            event_to_companies[ev_id] = set()
        for c in ents:
            event_to_companies[ev_id].add(c)

    rows = []
    # group returns by company_id for speed
    grouped = {cid: g.reset_index(drop=True) for cid, g in rets.groupby("company_id")}

    for ev in risk_events.itertuples(index=False):
        ev_id = getattr(ev, "event_id")
        t = pd.to_datetime(getattr(ev, "event_time"), errors="coerce")
        if pd.isna(t):
            continue
        companies = list(event_to_companies.get(ev_id, []))
        for cid in companies:
            if cid not in grouped:
                continue
            g = grouped[cid]
            for w in windows:
                car = compute_car_for_company(g, mkt, t, est_win, int(w))
                if car is None:
                    continue
                rows.append({"event_id": ev_id, "company_id": cid, "window_td": int(w), "CAR": float(car)})

    return pd.DataFrame(rows)
