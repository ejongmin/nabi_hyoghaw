"""
rebuild_zscore_v3_expanding.py
================================
18개 기업 tone_monthly_zscore를 Expanding Window 방식으로 재계산.

기존 rebuild_zscore_v3.py의 per_company_z()는 전체 기간 통계 사용 → look-ahead bias.
이 스크립트는 각 월 t에서 t-1까지 누적된 데이터만으로 z-score 계산.

출력: data/processed/tone_monthly_zscore_18_v3_expanding.parquet
"""
from __future__ import annotations
import logging, sys
from pathlib import Path
import numpy as np, pandas as pd
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
V10  = ROOT / "data/processed/risk_events_classified_v10_new.parquet"
C18  = ROOT / "data/universe/companies_18.csv"
OUT  = ROOT / "data/processed/tone_monthly_zscore_18_v3_expanding.parquet"

FINAL_18 = ['TSLA','BMW.DE','VOW3.DE','MBG.DE','7267.T','1211.HK',
            '300750.SZ','373220.KS','3931.HK','300014.SZ','002074.SZ',
            '051910.KS','BAS.DE','247540.KQ','300919.SZ','ALB','GLEN.L','603799.SH']

FB2_STRICT  = 10
SUP_STRICT  = 5
MIN_HIST    = 12
MIN_SUP_HIST = 6
SHOCK_THR   = 2.0
Z_CLIP      = 5.0

REGIME_DATES = {
    "regime_covid":        ("2020-01", "2021-12"),
    "regime_ukraine":      ("2022-02", "2023-06"),
    "regime_ira":          ("2022-08", "2025-12"),
    "regime_china_export": ("2023-08", "2025-12"),
    "regime_trump_tariff": ("2025-01", "2027-12"),
}

SUPPLY_TYPES = {"natural", "logistics", "geopolitics"}


def regime_flags(ym: str) -> dict:
    return {col: 1.0 if s <= ym <= e else 0.0
            for col, (s, e) in REGIME_DATES.items()}


def build_monthly_signals() -> pd.DataFrame:
    """v10_new에서 18개 기업 월별 집계 (rebuild_zscore_v3.py와 동일)."""
    log.info("v10_new 로드 중 (배치 방식)...")
    pf = pq.ParquetFile(V10)
    batches = []
    for batch in pf.iter_batches(
        batch_size=400_000,
        columns=["company_id","event_time","risk_types","tone",
                 "source_tier_weight","finbert2_score","finbert_score"]
    ):
        b = batch.to_pandas()
        b = b[b["company_id"].isin(FINAL_18)].copy()
        if len(b) == 0: continue

        b["event_time"] = pd.to_datetime(b["event_time"], errors="coerce")
        b = b.dropna(subset=["event_time"])
        b["ym"] = b["event_time"].dt.to_period("M")
        b["weight"] = b["source_tier_weight"].fillna(1.0)

        b["tone_w"] = b["tone"].fillna(0) * b["weight"]
        b["tone_valid"] = b["tone"].notna().astype(int)

        b["fb2_score"] = b["finbert2_score"].fillna(b["finbert_score"])
        b["fb2_valid"] = b["fb2_score"].notna().astype(int)
        b["fb2_w"] = b["fb2_score"].fillna(0) * b["weight"]

        def is_supply(rt):
            if not isinstance(rt, str): return False
            types = {t.strip() for t in rt.split(",")}
            return bool(types & SUPPLY_TYPES) and types != {"other"}
        b["is_supply"] = b["risk_types"].apply(is_supply)

        batches.append(b[["company_id","ym","tone_w","weight","tone_valid",
                           "fb2_valid","fb2_w","is_supply"]])

    df = pd.concat(batches, ignore_index=True)

    agg = df.groupby(["company_id","ym"]).agg(
        tone_wsum=("tone_w","sum"), wsum=("weight","sum"),
        n_tone=("tone_valid","sum"), n_fb2=("fb2_valid","sum"),
        fb2_wsum=("fb2_w","sum"),
    ).reset_index()
    agg["tone_mean"] = np.where(agg["wsum"] > 0, agg["tone_wsum"] / agg["wsum"], np.nan)
    agg["fb2_mean"] = np.where(agg["n_fb2"] > 0,
                               agg["fb2_wsum"] / agg["n_fb2"].replace(0, np.nan), np.nan)

    sup = df[df["is_supply"]].groupby(["company_id","ym"]).agg(
        sup_n_tone=("tone_valid","sum"), sup_n_fb2=("fb2_valid","sum"),
        sup_fb2_wsum=("fb2_w","sum"),
    ).reset_index()
    sup["sup_fb2_mean"] = np.where(
        sup["sup_n_fb2"] > 0,
        sup["sup_fb2_wsum"] / sup["sup_n_fb2"].replace(0, np.nan), np.nan)

    result = agg.merge(
        sup[["company_id","ym","sup_n_tone","sup_n_fb2","sup_fb2_mean"]],
        on=["company_id","ym"], how="left")
    log.info("월별 집계 완료: %d 기업-월", len(result))
    return result


def expanding_zscore(fb2_series: pd.Series, n_series: pd.Series,
                     min_count: int, min_hist: int, z_clip: float) -> pd.Series:
    """
    Expanding window z-score.
    - 월 t에서 t-1까지 누적된 reliable 데이터로 mu/std 계산
    - t의 값은 z-score 계산 후 running list에 추가
    """
    z_out = pd.Series(np.nan, index=fb2_series.index)
    running = []  # t-1까지 accumulated fb2_mean 값

    for idx in fb2_series.index:
        fb2_val = fb2_series[idx]
        n_val = int(n_series[idx]) if not pd.isna(n_series[idx]) else 0

        # t-1까지 통계로 z 계산
        if len(running) >= min_hist and n_val >= min_count and not pd.isna(fb2_val):
            mu = np.mean(running)
            sig = np.std(running, ddof=1)
            if sig > 1e-10:
                z = float(np.clip((fb2_val - mu) / sig, -z_clip, z_clip))
                z_out[idx] = z

        # 현재 t 값을 다음 달부터 사용하도록 추가
        if n_val >= min_count and not pd.isna(fb2_val):
            running.append(fb2_val)

    return z_out


def build_expanding_tone(agg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cid, g in agg.groupby("company_id"):
        g = g.sort_values("ym").copy()

        # finbert2 z (expanding)
        fb2_z = expanding_zscore(
            g["fb2_mean"], g["n_fb2"], FB2_STRICT, MIN_HIST, Z_CLIP)
        g["fb2_z_v3"] = fb2_z.values

        # supply z (expanding)
        sup_z = expanding_zscore(
            g["sup_fb2_mean"].fillna(np.nan),
            g["sup_n_fb2"].fillna(0),
            SUP_STRICT, MIN_SUP_HIST, Z_CLIP)
        g["supply_z_v3"] = sup_z.values

        # z_score_v3 = fb2_z (tone fallback 없이 finbert2만 사용)
        g["z_score_v3"] = g["fb2_z_v3"]
        g["z_source_v3"] = np.where(g["fb2_z_v3"].notna(), "finbert2", "insufficient")

        # reliable_v3: fb2_z가 계산됐고 n_fb2 >= FB2_STRICT
        g["reliable_v3"] = g["fb2_z_v3"].notna() & (g["n_fb2"] >= FB2_STRICT)

        # 파생 피처 (shift 기반, look-ahead 없음)
        g["z_lag1_v3"]     = g["z_score_v3"].shift(1).clip(-Z_CLIP, Z_CLIP)
        g["z_lag2_v3"]     = g["z_score_v3"].shift(2).clip(-Z_CLIP, Z_CLIP)
        g["delta_z_v3"]    = (g["z_score_v3"] - g["z_lag1_v3"]).clip(-Z_CLIP, Z_CLIP)
        g["rolling_3m_v3"] = g["z_score_v3"].rolling(3, min_periods=2).mean()

        # shock flags
        g["neg_shock_v3"] = (g["z_score_v3"] < -SHOCK_THR) & g["reliable_v3"] & g["z_score_v3"].notna()
        g["pos_shock_v3"] = (g["z_score_v3"] > SHOCK_THR)  & g["reliable_v3"] & g["z_score_v3"].notna()
        g["neg_shock_sup_v3"] = (g["supply_z_v3"] < -SHOCK_THR) & (g["sup_n_fb2"].fillna(0) >= SUP_STRICT) & g["supply_z_v3"].notna()
        g["pos_shock_sup_v3"] = (g["supply_z_v3"] > SHOCK_THR)  & (g["sup_n_fb2"].fillna(0) >= SUP_STRICT) & g["supply_z_v3"].notna()

        # regime 변수
        for _, row in g.iterrows():
            ym_str = str(row["ym"])
            for col, (s, e) in REGIME_DATES.items():
                g.loc[row.name, col] = 1.0 if s <= ym_str <= e else 0.0

        # 컬럼 추가
        g["tone_z_v3"] = np.nan  # finbert2 전용이므로 tone_z는 NaN
        g["year_month"] = g["ym"].astype(str)
        g["z_cross_v3"] = np.nan   # 아래에서 cross_z 계산
        g["z_cross_stage"] = np.nan

        rows.append(g)

    result = pd.concat(rows, ignore_index=True)
    return result


def add_cross_z(panel: pd.DataFrame) -> pd.DataFrame:
    """월별 횡단면 z-score (look-ahead 없음 — 같은 달 내 기업간 정규화)."""
    def cx(g):
        v = g["z_score_v3"].dropna()
        if len(v) < 5:
            return g
        mu, sig = v.mean(), v.std(ddof=1)
        if sig > 1e-10:
            g = g.copy()
            g["z_cross_v3"] = (g["z_score_v3"] - mu) / sig
        return g
    return panel.groupby("year_month", group_keys=False).apply(cx)


def main():
    log.info("=== 18개 기업 Expanding Window z-score 재계산 ===")

    agg = build_monthly_signals()

    log.info("Expanding window z-score 계산 중...")
    tone_df = build_expanding_tone(agg)
    tone_df = add_cross_z(tone_df)

    # 최종 컬럼 정리
    keep_cols = [
        "company_id","year_month",
        "tone_wsum","wsum","n_tone","n_fb2","fb2_wsum",
        "tone_mean","fb2_mean",
        "sup_n_tone","sup_n_fb2","sup_fb2_mean",
        "fb2_z_v3","tone_z_v3","z_score_v3","z_source_v3","reliable_v3",
        "supply_z_v3","z_lag1_v3","z_lag2_v3","delta_z_v3","rolling_3m_v3",
        "neg_shock_v3","pos_shock_v3","neg_shock_sup_v3","pos_shock_sup_v3",
        "z_cross_v3","z_cross_stage",
        "regime_covid","regime_ukraine","regime_ira",
        "regime_china_export","regime_trump_tariff",
    ]
    # sup_tone_mean은 있으면 추가
    for c in ["sup_tone_mean","sup_tone_wsum"]:
        if c in tone_df.columns:
            keep_cols.append(c)

    available = [c for c in keep_cols if c in tone_df.columns]
    out_df = tone_df[available].sort_values(["company_id","year_month"]).reset_index(drop=True)

    out_df.to_parquet(OUT, index=False)
    log.info("저장 완료: %s (%d행)", OUT.name, len(out_df))

    # 비교 요약
    log.info("\n=== Expanding vs Full-Period 비교 (18개 기업) ===")
    orig = pd.read_parquet(ROOT/"data/processed/tone_monthly_zscore_18_v3.parquet")
    for cid in FINAL_18[:6]:
        o = orig[orig["company_id"]==cid]["z_score_v3"].dropna()
        n = out_df[out_df["company_id"]==cid]["z_score_v3"].dropna()
        log.info("  %-15s  orig reliable=%2d  expand reliable=%2d  "
                 "orig_mean=%+.3f  expand_mean=%+.3f",
                 cid, len(o), len(n),
                 o.mean() if len(o) else float("nan"),
                 n.mean() if len(n) else float("nan"))


if __name__ == "__main__":
    main()
