"""
Phase 2.5: Multi-frequency placebo falsification

Phase 2 (daily placebo) 에서 39개 기업 중 3개만 p_std<0.05 로 통과한 원인을
'일별 집계의 power 부족' 으로 가설하고, 여러 집계 주기 / 여러 tier subset 에서
재검증한다.

선행 연구 근거:
  Baker, Bloom, Davis (2016) *"Measuring Economic Policy Uncertainty"* QJE —
    EPU 는 **월간** 집계를 사용하며, 일별 노이즈를 명시적으로 제거.
  Manela, Moreira (2017) *"News Implied Volatility and Disaster Concerns"* JFE —
    front-page WSJ 기반, **월간** NVIX 구축.
  Bloom (2009) *"The Impact of Uncertainty Shocks"* Econometrica — 월간.
  Bybee, Kelly, Manela, Xiu (2020/2021) *"The Structure of Economic News"* —
    topic attention 월간 집계.

즉 기존 문헌들이 일관되게 월간을 쓰는 것은 뉴스 sentiment 의 일별 noise-to-signal
문제를 경험적으로 알고 있었기 때문. 우리의 Phase 2 placebo 결과는 그 경험을
정량적으로 재현한 것.

본 스크립트:
  4개 조합 x (daily, weekly, monthly) × (all-tier, T1-only) 교차.
  실제로는 4 실행:
    A. daily    / all-tier  (Phase 2 reproduction)
    B. weekly   / all-tier
    C. monthly  / all-tier
    D. weekly   / T1-only
  각각의 (real_std, placebo_std_mean, p_std, sig_count / total_firms) 요약을
  reports/placebo_multifreq_summary.csv 에 누적.

Note: dedup_weight > 0 인 행만 사용 (non-lead 제외).
"""

import argparse
import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
V9 = ROOT / "data" / "processed" / "risk_events_classified_v9.parquet"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)
OUT_CSV = REPORTS / "placebo_multifreq_summary.csv"

PEAK_LOW = -2.0
PEAK_HIGH = +2.0


def load_rows(con, tier_filter: str | None):
    tier_sql = ""
    if tier_filter == "T1":
        tier_sql = "AND source_tier = 1"
    q = f"""
        SELECT company_id,
               event_time,
               finbert_z * dedup_weight AS w_z,
               dedup_weight AS w
        FROM read_parquet('{V9}')
        WHERE finbert_z IS NOT NULL
          AND dedup_weight > 0
          AND company_id IS NOT NULL
          AND title IS NOT NULL AND title != ''
          {tier_sql}
    """
    df = con.execute(q).fetchdf()
    df["event_time"] = pd.to_datetime(df["event_time"])
    return df


def bucket(dates: pd.Series, freq: str) -> pd.Series:
    if freq == "D":
        return dates.dt.floor("D")
    if freq == "W":
        # ISO week start (Mon)
        return dates.dt.to_period("W-SUN").dt.start_time
    if freq == "M":
        return dates.dt.to_period("M").dt.start_time
    raise ValueError(freq)


def agg_series(df_c: pd.DataFrame, freq: str) -> pd.Series:
    b = bucket(df_c["event_time"], freq)
    g = df_c.assign(_b=b).groupby("_b")
    s = g["w_z"].sum() / g["w"].sum().replace(0, np.nan)
    return s.dropna().sort_index()


def run_one(df: pd.DataFrame, freq: str, tier_label: str, B: int,
            min_buckets: int, seed: int):
    rng = np.random.default_rng(seed)
    rows = []
    for cid, grp in df.groupby("company_id"):
        real = agg_series(grp, freq)
        if len(real) < min_buckets:
            continue
        real_std = float(real.std())
        real_neg = int((real < PEAK_LOW).sum())
        real_pos = int((real > PEAK_HIGH).sum())

        dates = grp["event_time"].to_numpy()
        wz = grp["w_z"].to_numpy()
        w = grp["w"].to_numpy()
        n = len(grp)

        null_std = np.empty(B, dtype=np.float32)
        null_neg = np.empty(B, dtype=np.int32)
        null_pos = np.empty(B, dtype=np.int32)
        for b in range(B):
            perm = rng.permutation(n)
            tmp = pd.DataFrame({"event_time": dates[perm], "w_z": wz, "w": w})
            s = agg_series(tmp, freq)
            if len(s) == 0:
                null_std[b] = np.nan
                null_neg[b] = 0
                null_pos[b] = 0
                continue
            null_std[b] = float(s.std())
            null_neg[b] = int((s < PEAK_LOW).sum())
            null_pos[b] = int((s > PEAK_HIGH).sum())

        rows.append({
            "setting": f"{freq}-{tier_label}",
            "company_id": cid,
            "n_rows": n,
            "n_buckets_real": len(real),
            "real_std": real_std,
            "placebo_std_mean": float(np.nanmean(null_std)),
            "p_std": float((null_std >= real_std).mean()),
            "real_neg": real_neg,
            "p_neg": float((null_neg >= real_neg).mean()),
            "real_pos": real_pos,
            "p_pos": float((null_pos >= real_pos).mean()),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    con = duckdb.connect()

    configs = [
        # (freq, tier, min_buckets, label)
        ("D", None, 30, "D-all"),
        ("W", None, 12, "W-all"),
        ("M", None, 6,  "M-all"),
        ("W", "T1", 12, "W-T1"),
    ]

    all_rows = []
    for freq, tier, min_b, label in configs:
        log.info("== Running %s  (freq=%s tier=%s min_buckets=%d) ==",
                 label, freq, tier or "all", min_b)
        df = load_rows(con, tier)
        log.info("  rows loaded: %d (companies: %d)",
                 len(df), df["company_id"].nunique())
        res = run_one(df, freq, tier or "all", args.B, min_b, args.seed)
        log.info("  firms after min_buckets filter: %d", len(res))
        if len(res):
            sig = (res["p_std"] < 0.05).sum()
            log.info("  sig p_std<0.05 : %d/%d (%.1f%%)",
                     sig, len(res), 100 * sig / len(res))
            log.info("  mean ratio real/placebo std: %.4f",
                     (res["real_std"] / res["placebo_std_mean"]).mean())
        all_rows.append(res)

    all_df = pd.concat(all_rows, ignore_index=True)
    all_df.to_csv(OUT_CSV, index=False)

    # Aggregate summary
    log.info("\n== SUMMARY (p_std < 0.05 by setting) ==")
    summ = (all_df.groupby("setting")
                 .agg(n_firms=("company_id", "count"),
                      sig_count=("p_std", lambda s: (s < 0.05).sum()),
                      mean_real_std=("real_std", "mean"),
                      mean_placebo_std=("placebo_std_mean", "mean"),
                      median_p_std=("p_std", "median")))
    summ["sig_pct"] = 100 * summ["sig_count"] / summ["n_firms"]
    summ["std_ratio"] = summ["mean_real_std"] / summ["mean_placebo_std"]
    log.info("\n%s", summ.to_string())

    with open(REPORTS / "placebo_multifreq_summary.txt", "w") as f:
        f.write("Phase 2.5 Multi-frequency Placebo Falsification\n")
        f.write(f"B={args.B}, seed={args.seed}\n\n")
        f.write(summ.to_string())
        f.write("\n")
    log.info("wrote %s", OUT_CSV)


if __name__ == "__main__":
    main()
