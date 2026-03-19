from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import pandas as pd
import numpy as np
import logging

from src.common.dates import next_trading_day

log = logging.getLogger("event_study")

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


def _fast_ols_car(ret: np.ndarray, mkt: np.ndarray,
                  est_start: int, est_end: int,
                  evt_start: int, evt_end: int) -> Optional[float]:
    """numpy-only OLS → CAR. statsmodels 호출 없이 직접 계산."""
    est_r = ret[est_start:est_end]
    est_m = mkt[est_start:est_end]
    if len(est_r) < 30:
        return None

    # OLS: ret = alpha + beta * mkt
    n = len(est_r)
    sum_m = est_m.sum()
    sum_r = est_r.sum()
    sum_mm = (est_m * est_m).sum()
    sum_mr = (est_m * est_r).sum()

    denom = n * sum_mm - sum_m * sum_m
    if abs(denom) < 1e-15:
        return None

    beta = (n * sum_mr - sum_m * sum_r) / denom
    alpha = (sum_r - beta * sum_m) / n

    # Event window AR
    evt_r = ret[evt_start:evt_end]
    evt_m = mkt[evt_start:evt_end]
    if len(evt_r) < (evt_end - evt_start):
        return None

    expected = alpha + beta * evt_m
    ar = evt_r - expected
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

    event_to_companies: Dict[str, set] = {}
    for ev_id, g in exp.groupby("event_id"):
        top = g.sort_values("exposure_rwr", ascending=False).head(topk)["company_id"].tolist()
        event_to_companies[ev_id] = set(top)

    # add directly mentioned companies
    for r in risk_events.itertuples(index=False):
        ev_id = getattr(r, "event_id")
        ents = getattr(r, "entity_ids", [])
        if not isinstance(ents, (list, tuple)):
            try:
                ents = list(ents)
            except (TypeError, ValueError):
                ents = []
        if ev_id not in event_to_companies:
            event_to_companies[ev_id] = set()
        for c in ents:
            event_to_companies[ev_id].add(c)

    # Pre-build per-company aligned arrays (ret, mkt, dates)
    log.info("Pre-building per-company aligned arrays...")
    company_data: Dict[str, dict] = {}
    for cid, g in rets.groupby("company_id"):
        df = g.set_index("date")[["ret"]].join(mkt.to_frame(), how="inner").dropna()
        if df.empty:
            continue
        dates = df.index.values.astype("datetime64[D]")
        company_data[cid] = {
            "ret": df["ret"].values,
            "mkt": df["mkt_ret"].values,
            "dates": dates,
            "date_to_idx": {d: i for i, d in enumerate(dates)},
        }

    log.info(f"Companies with data: {len(company_data)}")
    log.info(f"Events to process: {len(risk_events):,}")

    rows = []
    n_processed = 0
    n_total = len(risk_events)

    for ev in risk_events.itertuples(index=False):
        ev_id = getattr(ev, "event_id")
        t = pd.to_datetime(getattr(ev, "event_time"), errors="coerce")
        if pd.isna(t):
            continue
        companies = list(event_to_companies.get(ev_id, []))

        for cid in companies:
            if cid not in company_data:
                continue
            cd = company_data[cid]

            # Find event day index
            trade_day = next_trading_day(cd["dates"], t)
            if trade_day is None or trade_day not in cd["date_to_idx"]:
                continue
            idx = cd["date_to_idx"][trade_day]

            est_start = idx + est_win[0]
            est_end = idx + est_win[1]
            if est_start < 0 or est_end <= est_start:
                continue

            for w in windows:
                car = _fast_ols_car(cd["ret"], cd["mkt"],
                                    est_start, est_end, idx, idx + int(w))
                if car is not None:
                    rows.append({"event_id": ev_id, "company_id": cid,
                                 "window_td": int(w), "CAR": float(car)})

        n_processed += 1
        if n_processed % 10000 == 0:
            log.info(f"  Event study progress: {n_processed:,}/{n_total:,} ({100*n_processed/n_total:.1f}%)")

    log.info(f"  Event study complete: {len(rows):,} CAR results")
    return pd.DataFrame(rows)
