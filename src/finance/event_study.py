from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import pandas as pd
import numpy as np
import logging

from src.common.dates import next_trading_day

log = logging.getLogger("event_study")

# ──────────────────────── Returns & Market ────────────────────────

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values(["company_id", "date"])
    df["ret"] = df.groupby("company_id")["close"].pct_change()
    return df.dropna(subset=["ret"])


def build_market_proxy(rets: pd.DataFrame) -> pd.Series:
    """Equal-weight 시장 수익률. 향후 cap-weight 또는 외부 인덱스로 대체 권장."""
    mkt = rets.groupby("date")["ret"].mean()
    mkt.name = "mkt_ret"
    return mkt


# ──────────────────────── OLS + Statistical Testing ────────────────────────

MIN_EST_DAYS = 60  # 최소 추정 윈도우 (학계 권장: 120+, 절대 하한: 60)

def _fast_ols_car(ret: np.ndarray, mkt: np.ndarray,
                  est_start: int, est_end: int,
                  evt_start: int, evt_end: int
                  ) -> Optional[Dict[str, float]]:
    """
    numpy-only OLS → CAR + t-stat + p-value.

    Market Model: R_i,t = α + β × R_m,t + ε_t
    AR_t = R_i,t - (α_hat + β_hat × R_m,t)
    CAR = Σ AR_t

    통계 검정: Patell(1976) standardized test
      - σ_AR = std(residuals in estimation window)
      - t_CAR = CAR / (σ_AR × sqrt(event_window_length))
      - p-value from t-distribution (df = est_len - 2)

    Returns dict with keys: CAR, t_stat, p_value, alpha, beta, sigma_ar, est_len
    """
    est_r = ret[est_start:est_end]
    est_m = mkt[est_start:est_end]
    n = len(est_r)
    if n < MIN_EST_DAYS:
        return None

    # OLS: ret = alpha + beta * mkt
    sum_m = est_m.sum()
    sum_r = est_r.sum()
    sum_mm = (est_m * est_m).sum()
    sum_mr = (est_m * est_r).sum()

    denom = n * sum_mm - sum_m * sum_m
    if abs(denom) < 1e-15:
        return None

    beta = (n * sum_mr - sum_m * sum_r) / denom
    alpha = (sum_r - beta * sum_m) / n

    # Estimation window residuals → sigma_AR
    expected_est = alpha + beta * est_m
    residuals = est_r - expected_est
    sigma_ar = float(np.std(residuals, ddof=2))  # df = n - 2 (alpha, beta)
    if sigma_ar < 1e-10:
        return None

    # Event window AR & CAR
    evt_r = ret[evt_start:evt_end]
    evt_m = mkt[evt_start:evt_end]
    evt_len = evt_end - evt_start
    if len(evt_r) < max(1, evt_len * 0.8):  # 80% 이상 데이터 필요
        return None

    expected_evt = alpha + beta * evt_m
    ar = evt_r - expected_evt
    car = float(ar.sum())

    # Patell (1976) standardized test statistic
    # Var(CAR) = L * σ²_AR  (L = event window length)
    # 단순 근사 (prediction error 보정 생략 — 추후 개선 가능)
    t_stat = car / (sigma_ar * np.sqrt(len(evt_r)))

    # p-value from t-distribution (two-tailed)
    from scipy import stats
    df_t = n - 2
    p_value = float(2.0 * stats.t.sf(abs(t_stat), df_t))

    return {
        "CAR": car,
        "t_stat": t_stat,
        "p_value": p_value,
        "alpha": alpha,
        "beta": beta,
        "sigma_ar": sigma_ar,
        "est_len": n,
    }


# ──────────────────────── Main Event Study ────────────────────────

def run_event_study(prices: pd.DataFrame,
                    risk_events: pd.DataFrame,
                    exposure: pd.DataFrame,
                    windows: List[int],
                    est_win: Tuple[int, int],
                    topk: int = 15,
                    filter_confounding: bool = True) -> pd.DataFrame:
    """
    Event Study: CAR + 통계 검정.

    Args:
        filter_confounding: True면 이벤트 윈도우 내 동일 기업 다른 이벤트 존재 시 제외
    """
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

    # Build per-event date index for confounding event check
    event_dates: Dict[str, set] = {}  # company_id → set of event dates
    if filter_confounding:
        for r in risk_events.itertuples(index=False):
            ents = getattr(r, "entity_ids", [])
            if not isinstance(ents, (list, tuple)):
                try:
                    ents = list(ents)
                except (TypeError, ValueError):
                    ents = []
            t = pd.to_datetime(getattr(r, "event_time", None), errors="coerce")
            if pd.isna(t):
                continue
            td = np.datetime64(t.date(), "D")
            for c in ents:
                if c not in event_dates:
                    event_dates[c] = set()
                event_dates[c].add(td)

    # Pre-build per-company aligned arrays
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
    n_confounding_skip = 0

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
                evt_end = idx + int(w)
                # bounds check
                if evt_end > len(cd["ret"]):
                    continue

                # Confounding event filter: 이벤트 윈도우 내 동일 기업 다른 이벤트 존재 시 스킵
                if filter_confounding and cid in event_dates:
                    window_dates = set(cd["dates"][idx:evt_end])
                    other_events = event_dates[cid] - {trade_day}
                    if window_dates & other_events:
                        n_confounding_skip += 1
                        continue

                result = _fast_ols_car(cd["ret"], cd["mkt"],
                                       est_start, est_end, idx, evt_end)
                if result is not None:
                    rows.append({
                        "event_id": ev_id,
                        "company_id": cid,
                        "window_td": int(w),
                        "CAR": result["CAR"],
                        "t_stat": result["t_stat"],
                        "p_value": result["p_value"],
                        "beta": result["beta"],
                        "sigma_ar": result["sigma_ar"],
                        "est_len": result["est_len"],
                    })

        n_processed += 1
        if n_processed % 10000 == 0:
            log.info(f"  Event study progress: {n_processed:,}/{n_total:,} "
                     f"({100 * n_processed / n_total:.1f}%)")

    log.info(f"  Event study complete: {len(rows):,} CAR results")
    if filter_confounding:
        log.info(f"  Confounding events skipped: {n_confounding_skip:,}")

    df_result = pd.DataFrame(rows)

    # CAAR (Cross-sectional Average AR) 요약 통계 로깅
    if not df_result.empty:
        for w in windows:
            sub = df_result[df_result["window_td"] == w]
            if sub.empty:
                continue
            caar = sub["CAR"].mean()
            n_obs = len(sub)
            n_sig = (sub["p_value"] < 0.05).sum()
            pct_sig = 100 * n_sig / n_obs if n_obs > 0 else 0
            # CAAR t-test (cross-sectional)
            caar_t = caar / (sub["CAR"].std() / np.sqrt(n_obs)) if n_obs > 1 else 0
            log.info(f"  Window {w}d: CAAR={caar:.4f}, t={caar_t:.2f}, "
                     f"n={n_obs:,}, sig@5%={n_sig:,} ({pct_sig:.1f}%)")

    return df_result
