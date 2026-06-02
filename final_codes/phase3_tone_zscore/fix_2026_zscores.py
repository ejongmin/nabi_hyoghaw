"""
2026년 z-score 이상값 패치 스크립트
=====================================
기존 tone_monthly_zscore_18_v3.parquet의 2026년 데이터를
올바른 historical 통계로 재계산합니다.

문제: update_tone_2026.py가 historical < 12개월이면 mu=0, sig=1 기본값 사용
      → z = fb2_mean 자체 (Honda -39.6, TSLA -21.8 등 이상값 발생)

수정:
  - historical >= 12개월인 기업만 z 계산 (나머지는 NaN)
  - reliable_v3 = (n_fb2 >= 10) AND (historical >= 12)
  - |z| > 5 클리핑
  - lag/delta/rolling 재계산
"""
from __future__ import annotations
import logging, sys
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

ROOT    = Path(__file__).resolve().parents[1]
TONE_V3 = ROOT / "data/processed/tone_monthly_zscore_18_v3.parquet"
BACKUP  = ROOT / "data/processed/tone_monthly_zscore_18_v3_pre_fix.parquet"

FB2_STRICT  = 10
SUP_STRICT  = 5
MIN_HIST    = 12
MIN_SUP_HIST = 6
Z_CLIP      = 5.0


def main():
    log.info("=== 2026년 z-score 패치 시작 ===")

    df = pd.read_parquet(TONE_V3)
    log.info("전체 행: %d  (%s ~ %s)", len(df),
             df["year_month"].min(), df["year_month"].max())

    # 백업
    df.to_parquet(BACKUP, index=False)
    log.info("백업 저장: %s", BACKUP.name)

    hist_df = df[df["year_month"] < "2026-01"].copy()
    new_df  = df[df["year_month"] >= "2026-01"].copy()
    log.info("Historical: %d행 | 2026년 신규: %d행", len(hist_df), len(new_df))

    if new_df.empty:
        log.info("2026년 데이터 없음. 종료.")
        return

    # ── 기업별 historical 통계 계산 ──────────────────────────────
    company_stats: dict = {}
    for cid, grp in hist_df.groupby("company_id"):
        rel_rows = grp[grp["reliable_v3"] == True]
        fb2_vals = rel_rows["fb2_mean"].dropna()
        sup_vals = grp.loc[grp["sup_n_fb2"].fillna(0) >= SUP_STRICT,
                           "sup_fb2_mean"].dropna()
        company_stats[cid] = {
            "n_hist":    len(fb2_vals),
            "mu_fb2":    fb2_vals.mean() if len(fb2_vals) >= MIN_HIST else np.nan,
            "sig_fb2":   fb2_vals.std(ddof=1) if len(fb2_vals) >= MIN_HIST else np.nan,
            "n_sup":     len(sup_vals),
            "mu_sup":    sup_vals.mean() if len(sup_vals) >= MIN_SUP_HIST else np.nan,
            "sig_sup":   sup_vals.std(ddof=1) if len(sup_vals) >= MIN_SUP_HIST else np.nan,
        }
        log.info("  [%s] hist=%d  mu_fb2=%.4f  sig_fb2=%.4f",
                 cid, len(fb2_vals),
                 company_stats[cid]["mu_fb2"] if not np.isnan(company_stats[cid]["mu_fb2"]) else float("nan"),
                 company_stats[cid]["sig_fb2"] if not np.isnan(company_stats[cid]["sig_fb2"]) else float("nan"))

    # ── 2026년 z-score 재계산 ─────────────────────────────────────
    fixed_rows: list[dict] = []
    for cid, grp in new_df.groupby("company_id"):
        stats     = company_stats.get(cid, {})
        has_hist  = stats.get("n_hist", 0) >= MIN_HIST and not np.isnan(stats.get("mu_fb2", np.nan))
        has_sup   = stats.get("n_sup", 0) >= MIN_SUP_HIST and not np.isnan(stats.get("mu_sup", np.nan))
        mu_fb2    = stats.get("mu_fb2", np.nan)
        sig_fb2   = stats.get("sig_fb2", np.nan)
        mu_sup    = stats.get("mu_sup", np.nan)
        sig_sup   = stats.get("sig_sup", np.nan)

        # lag 계산을 위해 historical z 시계열 유지
        cid_hist    = hist_df[hist_df["company_id"] == cid].sort_values("year_month")
        running_z   = list(cid_hist["z_score_v3"].dropna())

        for _, row in grp.sort_values("year_month").iterrows():
            fb2_mean     = row.get("fb2_mean", np.nan)
            _n = row.get("n_fb2", 0)
            n_fb2        = int(_n) if not pd.isna(_n) else 0
            sup_fb2_mean = row.get("sup_fb2_mean", np.nan)
            _sn = row.get("sup_n_fb2", 0)
            sup_n_fb2    = int(_sn) if not pd.isna(_sn) else 0

            reliable = bool(n_fb2 >= FB2_STRICT and has_hist and not pd.isna(fb2_mean))

            if reliable and not np.isnan(sig_fb2) and sig_fb2 > 1e-10:
                z = float(np.clip((fb2_mean - mu_fb2) / sig_fb2, -Z_CLIP, Z_CLIP))
            else:
                z = np.nan

            if (sup_n_fb2 >= SUP_STRICT and has_sup
                    and not pd.isna(sup_fb2_mean)
                    and not np.isnan(sig_sup) and sig_sup > 1e-10):
                sz = float(np.clip((sup_fb2_mean - mu_sup) / sig_sup, -Z_CLIP, Z_CLIP))
            else:
                sz = np.nan

            # lag/delta/rolling
            z_lag1   = running_z[-1] if len(running_z) >= 1 else np.nan
            z_lag2   = running_z[-2] if len(running_z) >= 2 else np.nan
            recent   = running_z[-2:] + ([z] if not np.isnan(z) else [])
            rolling  = float(np.mean(recent[-3:])) if len(recent) >= 2 else np.nan
            delta    = float(z - z_lag1) if not np.isnan(z) and not np.isnan(z_lag1) else np.nan

            new_row = row.to_dict()
            new_row.update({
                "z_score_v3":    z,
                "fb2_z_v3":      z,
                "reliable_v3":   reliable,
                "supply_z_v3":   sz,
                "z_source_v3":   "finbert2" if reliable else "insufficient",
                "z_lag1_v3":     z_lag1,
                "z_lag2_v3":     z_lag2,
                "rolling_3m_v3": rolling,
                "delta_z_v3":    delta,
                "neg_shock_v3":  bool(not np.isnan(z) and z < -2.0 and reliable),
                "pos_shock_v3":  bool(not np.isnan(z) and z >  2.0 and reliable),
                "neg_shock_sup_v3": bool(not np.isnan(sz) and sz < -2.0
                                        and sup_n_fb2 >= SUP_STRICT),
                "pos_shock_sup_v3": bool(not np.isnan(sz) and sz >  2.0
                                        and sup_n_fb2 >= SUP_STRICT),
            })
            fixed_rows.append(new_row)

            if not np.isnan(z):
                running_z.append(z)

    fixed_new = pd.DataFrame(fixed_rows)

    # z_cross_v3 (월별 기업간 교차 z) 재계산
    def add_cross_z(df_: pd.DataFrame) -> pd.DataFrame:
        def cx(g):
            v = g["z_score_v3"].dropna()
            if len(v) < 5:
                return g
            mu, sig = v.mean(), v.std(ddof=1)
            if sig > 1e-10:
                g = g.copy()
                g["z_cross_v3"] = (g["z_score_v3"] - mu) / sig
            return g
        return df_.groupby("year_month", group_keys=False).apply(cx)

    fixed_new = add_cross_z(fixed_new)

    # ── 병합 & 저장 ───────────────────────────────────────────────
    combined = pd.concat(
        [hist_df, fixed_new.drop(columns=["ym"], errors="ignore")],
        ignore_index=True
    ).sort_values(["company_id", "year_month"]).reset_index(drop=True)

    combined.to_parquet(TONE_V3, index=False)
    log.info("저장 완료: %s (%d행)", TONE_V3.name, len(combined))

    # ── 결과 요약 ─────────────────────────────────────────────────
    log.info("\n=== 패치 결과 요약 (2026년) ===")
    for ym in sorted(fixed_new["year_month"].unique()):
        sub = fixed_new[fixed_new["year_month"] == ym]
        zvals = sub["z_score_v3"].dropna()
        log.info("  %s: %d기업, reliable=%d, z=[%.2f, %.2f] (수정 전 이상값 제거됨)",
                 ym, len(sub), int(sub["reliable_v3"].sum()),
                 zvals.min() if len(zvals) else float("nan"),
                 zvals.max() if len(zvals) else float("nan"))


if __name__ == "__main__":
    main()
