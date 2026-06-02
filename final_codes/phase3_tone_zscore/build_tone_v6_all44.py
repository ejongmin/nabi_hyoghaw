"""
build_tone_v6_all44.py
======================
44개 기업 전체에 다국어 FinBERT 적용 → tone_monthly_zscore_all44_v6.parquet

변경점 vs v5:
- 26개 신규 기업: 기존 tone_new26_multilingual.parquet 재활용
- 18개 원본 기업: finbert_v2 (영어) + finbert_v2_multilingual (다국어) 통합 적용
  → 중국어/한국어/일본어 기업 커버리지 개선 기대

출력: experiments/universe_44/data/processed/tone_monthly_zscore_all44_v6.parquet
"""
from __future__ import annotations
import logging, sys
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
EXP  = ROOT / "experiments/universe_44"
OUT  = EXP / "data/processed/tone_monthly_zscore_all44_v6.parquet"

ORIG18 = [
    '002074.SZ','051910.KS','1211.HK','247540.KQ','300014.SZ',
    '300750.SZ','300919.SZ','373220.KS','3931.HK','603799.SH',
    '7267.T','ALB','BAS.DE','BMW.DE','GLEN.L','MBG.DE','TSLA','VOW3.DE'
]
NEW26 = [
    '300037.SZ','300207.SZ','920185.BJ','002340.SZ','688388.SH',
    '600110.SH','603659.SH','600733.SH','000625.SZ','600006.SH',
    '601238.SH','0175.HK','601633.SH','600418.SH','603993.SH',
    '002460.SZ','600362.SH','002466.SZ','3407.T','066970.KS',
    '003670.KS','F','GM','LCID','7203.T','RIVN'
]
ALL44 = ORIG18 + NEW26

SUPPLY_KW = {
    "supply","supplier","shortage","disruption","delay","delivery",
    "shipment","contract","mine","mining","refin","lithium","cobalt",
    "nickel","cathode","anode","precursor","production halt","factory",
    "recall","tariff","sanction","export ban","import","procurement",
    "供应","短缺","断供","原材料","电池","锂","钴","镍",
}
REGIME_DATES = {
    "regime_covid":        ("2020-01", "2021-12"),
    "regime_ukraine":      ("2022-02", "2023-06"),
    "regime_ira":          ("2022-08", "2025-12"),
    "regime_china_export": ("2023-08", "2025-12"),
    "regime_trump_tariff": ("2025-01", "2027-12"),
}
FB2_STRICT = 10; SUP_STRICT = 5; MIN_HIST = 12; Z_CLIP = 5.0


def regime_flags(ym):
    return {c: 1.0 if s <= ym <= e else 0.0 for c, (s, e) in REGIME_DATES.items()}

def is_supply(title):
    tl = str(title).lower()
    return any(kw in tl for kw in SUPPLY_KW)

def compute_tone_for_companies(companies, ev_df, fb_url, fb_lang):
    """주어진 기업 목록에 대해 월별 z-score 계산."""
    ev = ev_df[ev_df["company_id"].isin(companies)].copy()
    log.info("  대상 이벤트: %d건", len(ev))

    merged_url = ev.merge(fb_url, on="url", how="inner")
    log.info("  URL 조인 성공: %d건 (%.1f%%)", len(merged_url), 100*len(merged_url)/max(len(ev),1))

    ev_remaining = ev[~ev["url"].isin(merged_url["url"])].copy()
    merged_eid = ev_remaining.merge(fb_lang, left_on="event_id",
                                    right_on="rep_event_id", how="inner")
    log.info("  event_id 조인 추가: %d건", len(merged_eid))

    merged = pd.concat([
        merged_url[["url","title","company_id","event_time","finbert_score"]],
        merged_eid[["url","title","company_id","event_time","finbert_score"]],
    ], ignore_index=True).drop_duplicates(subset=["url","company_id"])
    log.info("  통합 총: %d건", len(merged))

    merged["event_time"] = pd.to_datetime(merged["event_time"], errors="coerce")
    merged = merged.dropna(subset=["event_time"])
    merged["ym"] = merged["event_time"].dt.to_period("M").astype(str)
    merged["is_supply"] = merged["title"].apply(is_supply)
    merged["fb2_score"] = merged["finbert_score"].astype(float)

    agg = (merged.groupby(["company_id","ym"])
               .agg(fb2_wsum=("fb2_score","sum"), n_fb2=("fb2_score","count"))
               .reset_index())
    agg["fb2_mean"] = agg["fb2_wsum"] / agg["n_fb2"].replace(0, np.nan)

    sup = (merged[merged["is_supply"]].groupby(["company_id","ym"])
               .agg(sup_n_fb2=("fb2_score","count"),
                    sup_fb2_wsum=("fb2_score","sum"))
               .reset_index())
    sup["sup_fb2_mean"] = sup["sup_fb2_wsum"] / sup["sup_n_fb2"].replace(0, np.nan)
    agg = agg.merge(sup[["company_id","ym","sup_n_fb2","sup_fb2_mean"]],
                    on=["company_id","ym"], how="left")
    agg["sup_n_fb2"] = agg["sup_n_fb2"].fillna(0).astype(int)

    rows = []
    for cid in companies:
        cid_agg = agg[agg["company_id"]==cid].sort_values("ym")
        if cid_agg.empty:
            log.warning("  %s: 이벤트 없음", cid)
            continue
        all_rel = cid_agg[cid_agg["n_fb2"]>=FB2_STRICT]["fb2_mean"].dropna()
        has_hist = len(all_rel) >= MIN_HIST
        mu_fb2 = all_rel.mean() if has_hist else np.nan
        sig_fb2 = all_rel.std(ddof=1) if has_hist else np.nan
        all_sup = cid_agg[cid_agg["sup_n_fb2"]>=SUP_STRICT]["sup_fb2_mean"].dropna()
        has_sup = len(all_sup) >= 6
        mu_sup = all_sup.mean() if has_sup else np.nan
        sig_sup = all_sup.std(ddof=1) if has_sup else np.nan
        running_z = []
        for _, row in cid_agg.iterrows():
            fb2_mean = row["fb2_mean"]; n_fb2 = int(row["n_fb2"])
            sup_mean = row.get("sup_fb2_mean", np.nan); sup_n = int(row.get("sup_n_fb2",0))
            ym_str = str(row["ym"])
            reliable = bool(n_fb2>=FB2_STRICT and has_hist
                            and not pd.isna(fb2_mean)
                            and not np.isnan(sig_fb2) and sig_fb2>1e-10)
            z  = float(np.clip((fb2_mean-mu_fb2)/sig_fb2, -Z_CLIP, Z_CLIP)) if reliable else np.nan
            sz = float(np.clip((sup_mean-mu_sup)/sig_sup, -Z_CLIP, Z_CLIP)) \
                 if (sup_n>=SUP_STRICT and has_sup and not pd.isna(sup_mean)
                     and not np.isnan(sig_sup) and sig_sup>1e-10) else np.nan
            z_lag1 = running_z[-1] if len(running_z)>=1 else np.nan
            z_lag2 = running_z[-2] if len(running_z)>=2 else np.nan
            recent = running_z[-2:]+([z] if not np.isnan(z) else [])
            rolling = float(np.mean(recent[-3:])) if len(recent)>=2 else np.nan
            delta = float(z-z_lag1) if not np.isnan(z) and not np.isnan(z_lag1) else np.nan
            rows.append({
                "company_id":cid, "year_month":ym_str,
                "n_fb2":n_fb2, "fb2_mean":fb2_mean,
                "sup_n_fb2":sup_n, "sup_fb2_mean":sup_mean,
                "fb2_z_v3":z, "z_score_v3":z,
                "z_source_v3":"finbert2" if reliable else "insufficient",
                "reliable_v3":reliable, "supply_z_v3":sz,
                "z_lag1_v3":z_lag1, "z_lag2_v3":z_lag2,
                "rolling_3m_v3":rolling, "delta_z_v3":delta,
                "neg_shock_v3":bool(not np.isnan(z) and z<-2.0 and reliable),
                "pos_shock_v3":bool(not np.isnan(z) and z>2.0 and reliable),
                **regime_flags(ym_str),
                "tone_wsum":np.nan, "wsum":np.nan, "n_tone":0,
                "tone_mean":np.nan, "tone_z_v3":np.nan,
                "sup_n_tone":0, "sup_tone_mean":np.nan,
                "neg_shock_sup_v3":bool(not np.isnan(sz) and sz<-2.0 and sup_n>=SUP_STRICT),
                "pos_shock_sup_v3":bool(not np.isnan(sz) and sz>2.0 and sup_n>=SUP_STRICT),
                "z_cross_v3":np.nan, "z_cross_stage":np.nan,
            })
            if not np.isnan(z): running_z.append(z)

    return pd.DataFrame(rows)


def add_cross_z(df_):
    def cx(g):
        v = g["z_score_v3"].dropna()
        if len(v) < 5: return g
        mu, sig = v.mean(), v.std(ddof=1)
        if sig > 1e-10:
            g = g.copy()
            g["z_cross_v3"] = (g["z_score_v3"] - mu) / sig
        return g
    return df_.groupby("year_month", group_keys=False).apply(cx)


def main():
    log.info("=== Tone v6: 44개 기업 전체 다국어 FinBERT 적용 ===")

    # FinBERT 데이터 로드
    log.info("FinBERT 데이터 로드...")
    fb_en = pd.read_parquet(ROOT/"data/processed/finbert_v2.parquet",
                            columns=["url","finbert_score"])
    fb_ml = pd.read_parquet(ROOT/"data/processed/finbert_v2_multilingual.parquet",
                            columns=["url","finbert_score"])
    fb_url = pd.concat([fb_en, fb_ml], ignore_index=True)
    fb_url = fb_url.drop_duplicates(subset=["url"], keep="first")
    log.info("  URL 기반 통합: %d건", len(fb_url))

    fb_zh = pd.read_parquet(ROOT/"data/processed/finbert_results_zh.parquet",
                            columns=["rep_event_id","finbert_score"])
    fb_ko = pd.read_parquet(ROOT/"data/processed/finbert_results_ko.parquet",
                            columns=["rep_event_id","finbert_score"])
    fb_lang = pd.concat([fb_zh, fb_ko], ignore_index=True)
    fb_lang = fb_lang.drop_duplicates(subset=["rep_event_id"], keep="first")
    log.info("  ZH+KO FinBERT: %d건", len(fb_lang))

    # 이벤트 로드 (44개 전체)
    log.info("risk_events_v10_new 로드 (44개 기업)...")
    ev = pd.read_parquet(
        ROOT/"data/processed/risk_events_classified_v10_new.parquet",
        columns=["event_id","url","title","company_id","event_time"]
    )
    ev = ev[ev["company_id"].isin(ALL44)].copy()
    log.info("  44개 기업 이벤트: %d건", len(ev))

    # 18개 원본 기업 처리
    log.info("\n--- 18개 원본 기업 (다국어 신규 적용) ---")
    df_orig = compute_tone_for_companies(ORIG18, ev, fb_url, fb_lang)
    log.info("원본 18개 결과: %d행", len(df_orig))

    # 26개 신규 기업 (기존 multilingual 재활용)
    log.info("\n--- 26개 신규 기업 (기존 multilingual 재활용) ---")
    df_new26 = pd.read_parquet(EXP/"data/processed/tone_new26_multilingual.parquet")
    log.info("신규 26개 재사용: %d행", len(df_new26))

    # v5와 컬럼 맞추기
    for col in df_orig.columns:
        if col not in df_new26.columns:
            df_new26[col] = np.nan
    for col in df_new26.columns:
        if col not in df_orig.columns:
            df_orig[col] = np.nan

    # 합치기
    df_all = pd.concat([df_orig, df_new26], ignore_index=True)
    df_all = df_all.sort_values(["year_month","company_id"]).reset_index(drop=True)

    # cross-z 재계산 (전체 44개 기준)
    df_all = add_cross_z(df_all)

    df_all.to_parquet(OUT, index=False)
    log.info("\n저장 완료: %s (%d행, %d개 기업)", OUT.name, len(df_all), df_all["company_id"].nunique())

    # reliable 현황 출력
    summary = df_all.groupby("company_id").agg(
        total=("year_month","count"),
        reliable=("reliable_v3","sum")
    ).reset_index()
    summary["pct"] = (summary["reliable"]/summary["total"]*100).round(1)
    log.info("\n=== v6 reliable 현황 (v5 대비 개선 확인) ===\n%s",
             summary.sort_values("reliable").to_string(index=False))


if __name__ == "__main__":
    main()
