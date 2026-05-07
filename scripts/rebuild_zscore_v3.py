"""
tone_monthly_zscore v3 — v10_new(finbert2) 기반 재생성
========================================================

v2 대비 개선:
  [개선 1] finbert_score(v1, ~0% 커버) → finbert2_score(v2, 36% 커버)
           Tesla 0→55%, CATL 0→67%, LGES 0→67%, BMW 0→28%
           → 중국·한국 기업 FinBERT 신뢰도 대폭 향상

  [개선 2] supply_z 도 finbert2_score 기반으로 계산
           → 공급망 특화 감성 신호 품질 향상

  나머지(strict 기준, z_cross, reliable, trump_tariff 등)는 v2와 동일

출력:
  data/processed/tone_monthly_zscore_18_v3.parquet
  reports/zscore_v3_improvement_report.txt
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
V1   = ROOT/"data/processed/tone_monthly_zscore_18.parquet"     # 원본 v1
V10  = ROOT/"data/processed/risk_events_classified_v10_new.parquet"  # finbert2 포함
C18  = ROOT/"data/universe/companies_18.csv"
OUT  = ROOT/"data/processed/tone_monthly_zscore_18_v3.parquet"
RPT  = ROOT/"reports/zscore_v3_improvement_report.txt"

FINAL_18 = ['TSLA','BMW.DE','VOW3.DE','MBG.DE','7267.T','1211.HK',
            '300750.SZ','373220.KS','3931.HK','300014.SZ','002074.SZ',
            '051910.KS','BAS.DE','247540.KQ','300919.SZ','ALB','GLEN.L','603799.SH']

FB2_STRICT  = 10    # finbert2 월 기사 수 임계값
TONE_STRICT = 20    # tone 임계값
SHOCK_THR   = 2.0


# ── 월별 감성 집계 (v10_new 기반) ─────────────────────────────

def build_monthly_signals(v10_path: Path) -> pd.DataFrame:
    """
    v10_new에서 18개 기업 × 월별 집계
    - tone: GDELT v2_tone 가중 평균
    - finbert2: URL 매핑된 FinBERT v2 스코어 (훨씬 높은 커버리지)
    - supply: natural/logistics/geopolitics 특화 finbert2
    """
    SUPPLY_TYPES = {"natural", "logistics", "geopolitics"}
    pf = pq.ParquetFile(v10_path)
    batches = []

    for batch in pf.iter_batches(
        batch_size=400_000,
        columns=["company_id","event_time","risk_types","tone",
                 "source_tier_weight","finbert2_score","finbert2_neg",
                 "finbert2_pos","finbert_score"]
    ):
        b = batch.to_pandas()
        b = b[b["company_id"].isin(FINAL_18)].copy()
        if len(b) == 0: continue

        b["event_time"] = pd.to_datetime(b["event_time"], errors="coerce")
        b = b.dropna(subset=["event_time"])
        b["ym"] = b["event_time"].dt.to_period("M")
        b["weight"] = b["source_tier_weight"].fillna(1.0)

        # tone (GDELT)
        b["tone_w"]     = b["tone"].fillna(0) * b["weight"]
        b["tone_valid"] = b["tone"].notna().astype(int)

        # finbert2 (새 스코어) + finbert1 fallback
        b["fb2_score"] = b["finbert2_score"].fillna(b["finbert_score"])
        b["fb2_valid"] = b["fb2_score"].notna().astype(int)
        b["fb2_w"]     = b["fb2_score"].fillna(0) * b["weight"]

        # supply 여부
        def is_supply(rt):
            if not isinstance(rt,str): return False
            types = {t.strip() for t in rt.split(",")}
            return bool(types & SUPPLY_TYPES) and types != {"other"}
        b["is_supply"] = b["risk_types"].apply(is_supply)

        batches.append(b[["company_id","ym","tone_w","weight","tone_valid",
                           "fb2_valid","fb2_w","is_supply"]])

    df = pd.concat(batches, ignore_index=True)

    # 전체 집계
    agg = df.groupby(["company_id","ym"]).agg(
        tone_wsum  =("tone_w",    "sum"),
        wsum       =("weight",    "sum"),
        n_tone     =("tone_valid","sum"),
        n_fb2      =("fb2_valid", "sum"),
        fb2_wsum   =("fb2_w",     "sum"),
    ).reset_index()
    agg["tone_mean"] = np.where(agg["wsum"]>0, agg["tone_wsum"]/agg["wsum"], np.nan)
    agg["fb2_mean"]  = np.where(agg["n_fb2"]>0, agg["fb2_wsum"]/agg["n_fb2"].replace(0,np.nan), np.nan)

    # supply 집계
    sup = df[df["is_supply"]].groupby(["company_id","ym"]).agg(
        sup_tone_wsum=("tone_w","sum"), sup_wsum=("weight","sum"),
        sup_n_tone=("tone_valid","sum"), sup_n_fb2=("fb2_valid","sum"),
        sup_fb2_wsum=("fb2_w","sum"),
    ).reset_index()
    sup["sup_tone_mean"] = np.where(sup["sup_wsum"]>0, sup["sup_tone_wsum"]/sup["sup_wsum"], np.nan)
    sup["sup_fb2_mean"]  = np.where(sup["sup_n_fb2"]>0, sup["sup_fb2_wsum"]/sup["sup_n_fb2"].replace(0,np.nan), np.nan)

    result = agg.merge(sup[["company_id","ym","sup_n_tone","sup_n_fb2",
                             "sup_tone_mean","sup_fb2_mean"]],
                       on=["company_id","ym"], how="left")
    log.info("월별 집계 완료: %d기업-월", len(result))
    return result


def per_company_z(series: pd.Series, min_obs: int = 12) -> pd.Series:
    valid = series.dropna()
    if len(valid) < min_obs:
        return pd.Series(np.nan, index=series.index)
    mu = valid.mean(); sig = valid.std(ddof=1)
    if sig < 1e-10:
        return pd.Series(0.0, index=series.index)
    return (series - mu) / sig


def build_tone_df(agg: pd.DataFrame, v1: pd.DataFrame) -> pd.DataFrame:
    """
    v3 tone_monthly_zscore 생성:
    - z_score_v3: finbert2 기반 strict z-score
    - supply_z_v3: finbert2 기반 supply z-score
    """
    rows = []
    for cid, g in agg.groupby("company_id"):
        g = g.sort_values("ym").copy()
        T = len(g)

        # finbert2 strict z (n_fb2 >= 10)
        fb2_mask = g["n_fb2"] >= FB2_STRICT
        g["fb2_raw"] = np.where(fb2_mask, g["fb2_mean"], np.nan)
        g["fb2_z_v3"] = per_company_z(g["fb2_raw"])

        # tone strict z
        tone_mask = g["n_tone"] >= TONE_STRICT
        g["tone_raw"] = np.where(tone_mask, g["tone_mean"], np.nan)
        g["tone_z_v3"] = per_company_z(g["tone_raw"])

        # z_score_v3: finbert2 우선, fallback tone
        g["z_score_v3"] = g["fb2_z_v3"]
        g["z_source_v3"] = "finbert2"
        fallback = g["fb2_z_v3"].isna() & g["tone_z_v3"].notna()
        g.loc[fallback, "z_score_v3"] = g.loc[fallback, "tone_z_v3"]
        g.loc[fallback, "z_source_v3"] = "v2tone"

        # reliable_v3
        g["reliable_v3"] = fb2_mask | tone_mask

        # supply z v3 (finbert2 기반)
        sup_fb2_mask = g["sup_n_fb2"].fillna(0) >= 5
        g["sup_fb2_raw"] = np.where(sup_fb2_mask, g["sup_fb2_mean"], np.nan)
        g["supply_z_v3"] = per_company_z(g["sup_fb2_raw"], min_obs=6)
        # supply fallback: tone
        sup_tone_mask = g["sup_n_tone"].fillna(0) >= 5
        g["sup_tone_raw"] = np.where(sup_tone_mask, g["sup_tone_mean"], np.nan)
        g_sup_tone_z = per_company_z(g["sup_tone_raw"], min_obs=6)
        sup_fallback = g["supply_z_v3"].isna() & g_sup_tone_z.notna()
        g.loc[sup_fallback, "supply_z_v3"] = g_sup_tone_z[sup_fallback]

        # lag / delta / rolling (v3)
        g["z_lag1_v3"]    = g["z_score_v3"].shift(1)
        g["z_lag2_v3"]    = g["z_score_v3"].shift(2)
        g["delta_z_v3"]   = g["z_score_v3"] - g["z_lag1_v3"]
        g["rolling_3m_v3"]= g["z_score_v3"].rolling(3,min_periods=2).mean()

        # shock flags v3
        g["neg_shock_v3"]     = (g["z_score_v3"]  < -SHOCK_THR) & g["reliable_v3"] & g["z_score_v3"].notna()
        g["pos_shock_v3"]     = (g["z_score_v3"]  > SHOCK_THR)  & g["reliable_v3"] & g["z_score_v3"].notna()
        g["neg_shock_sup_v3"] = (g["supply_z_v3"] < -SHOCK_THR) & (g["sup_n_fb2"].fillna(0)>=5) & g["supply_z_v3"].notna()
        g["pos_shock_sup_v3"] = (g["supply_z_v3"] > SHOCK_THR)  & (g["sup_n_fb2"].fillna(0)>=5) & g["supply_z_v3"].notna()

        rows.append(g)

    result = pd.concat(rows, ignore_index=True)
    return result


def add_cross_z(panel: pd.DataFrame) -> pd.DataFrame:
    """횡단면 z-score (v3 기반)"""
    def cx_all(group):
        v = group["z_score_v3"].dropna()
        if len(v) < 5: group["z_cross_v3"] = np.nan; return group
        mu=v.mean(); sig=v.std(ddof=1)
        group["z_cross_v3"] = (group["z_score_v3"]-mu)/sig if sig>1e-10 else 0.0
        return group
    return panel.groupby("ym",group_keys=False).apply(cx_all)


def main():
    log.info("tone_monthly_zscore v3 재생성 시작 (v10_new 기반)")
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # v1 로드 (regime 컬럼 등 활용)
    v1 = pd.read_parquet(V1)
    v1["ym"] = pd.to_datetime(v1["year_month"]).dt.to_period("M")

    # v10_new 기반 집계
    log.info("v10_new 월별 집계 중...")
    agg = build_monthly_signals(V10)

    # tone_df 구성
    log.info("z-score 계산 중...")
    panel = build_tone_df(agg, v1)

    # 횡단면 z
    panel = add_cross_z(panel)

    # v2에서 기존 컬럼들 가져오기 (regime, z_cross_stage 등)
    v2 = pd.read_parquet(ROOT/"data/processed/tone_monthly_zscore_18_v2.parquet")
    v2["ym"] = pd.to_datetime(v2["year_month"]).dt.to_period("M")

    merge_cols = ["company_id","ym","regime_covid","regime_ukraine",
                  "regime_ira","regime_china_export","regime_trump_tariff",
                  "z_cross_stage"]
    avail = [c for c in merge_cols if c in v2.columns]
    panel = panel.merge(v2[avail], on=["company_id","ym"], how="left")

    # year_month 문자열로 저장
    panel["year_month"] = panel["ym"].dt.strftime("%Y-%m")
    panel = panel.drop(columns=["ym"] + [c for c in panel.columns if c in ["fb2_raw","tone_raw","sup_fb2_raw","sup_tone_raw"]])

    panel.to_parquet(OUT, index=False)
    log.info("저장: %s (%d행, %d컬럼)", OUT, len(panel), len(panel.columns))

    # 비교 보고서
    c18 = pd.read_csv(C18)
    nm  = dict(zip(c18["company_id"], c18["canonical_name"]))

    lines = ["="*68,"tone_monthly_zscore v3 — v10_new(finbert2) 기반 재생성","="*68,"",
             "■ 주요 변경: finbert_score(v1, ~0%) → finbert2_score(36% 커버)","",
             f"  {'기업':25s}  {'v2 n_fb(월평균)':>15}  {'v3 n_fb2(월평균)':>16}  {'v3 reliable':>12}"]
    lines.append("  "+"-"*72)

    for cid in FINAL_18:
        nv2 = v2[v2["company_id"]==cid]["n_fb_total"].mean() if "n_fb_total" in v2.columns else 0
        g   = panel[panel["company_id"]==cid]
        nv3 = g["n_fb2"].mean() if "n_fb2" in g.columns else 0
        rel = int(g["reliable_v3"].sum()) if "reliable_v3" in g.columns else 0
        lines.append(f"  {nm.get(cid,cid):25s}  {nv2:15.0f}  {nv3:16.0f}  {rel:>12}")

    neg_v2 = int(v2["neg_shock_v2"].sum()) if "neg_shock_v2" in v2.columns else int(v2["neg_shock"].sum())
    pos_v2 = int(v2["pos_shock_v2"].sum()) if "pos_shock_v2" in v2.columns else int(v2["pos_shock"].sum())
    neg_v3 = int(panel["neg_shock_v3"].sum())
    pos_v3 = int(panel["pos_shock_v3"].sum())
    neg_sup = int(panel["neg_shock_sup_v3"].sum())

    lines += ["",
              f"  neg_shock: v2={neg_v2} → v3={neg_v3}",
              f"  pos_shock: v2={pos_v2} → v3={pos_v3}",
              f"  supply NEG (v3): {neg_sup}건","","="*68]
    rpt = "\n".join(lines)
    RPT.write_text(rpt, encoding="utf-8")
    print("\n"+rpt)


if __name__ == "__main__":
    main()
