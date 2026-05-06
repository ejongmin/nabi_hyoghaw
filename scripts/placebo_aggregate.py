"""
Phase 2.7: Aggregate-level Placebo Falsification

Phase 2.5/2.6 에서 firm-level 은 monthly 에서조차 39 firms 중 6개만 유의했음.
문헌 (BBD 2016, Bloom 2009, Bybee-Kelly-Manela-Xiu 2020) 은 모두 aggregate
수준에서만 명확한 신호를 보임. 여기서는 4개 aggregate level 에서 placebo 를
재실행해 '어느 수준에서 signal 이 null 과 분리되는가' 를 실증한다.

Levels (raw × monthly 기준 — Phase 2.6 에서 최적으로 확인됨)
------
  L1. pooled         : 전체 title+finbert 코퍼스 하나의 시계열
  L2. country        : ["China"/"CN"]→CN, ["Korea"/"KR"]→KR 그룹별
  L3. risk_type_nz   : risk_types != 'other' 인 non-trivial risk 그룹별
  L4. country_risk   : country × risk_type_nz 교차

Weight: raw (source_tier_weight), Freq: monthly (Baker-Bloom-Davis 2016 QJE 와 동일)
Shuffle: 같은 그룹 내에서 event_time permutation, B=1000 (firm-level 보다 더
         많은 반복으로 aggregate 수준의 안정적 null).

가설 (추론):
  - L1 pooled: 45K rows 로 가장 풍부한 신호. std_ratio > 1.1 기대.
  - L2 country: CN/KR 만 있으므로 2 series, 각각 ~60 monthly buckets.
  - L3 risk_type_nz: 5 ~ 7 series, 일부는 low-n 로 불안정할 수 있음.
  - L4 country_risk: 일부 cell 만 충분한 n, 나머지는 drop.
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

PEAK_LOW = -2.0
PEAK_HIGH = +2.0
MIN_BUCKETS = 12  # 월 단위 12개 이상 있어야 std 계산


def normalize_country(cid: str) -> str:
    if cid is None or cid == "" or cid == "[]":
        return ""
    s = str(cid)
    if "China" in s or '"CN"' in s:
        return "CN"
    if "Korea" in s or '"KR"' in s:
        return "KR"
    if "GLOBAL" in s:
        return "GLOBAL"
    if "EU" in s or "Germany" in s or "DE" in s:
        return "EU"
    return "OTHER"


def load_base(con):
    q = f"""
        SELECT event_time,
               finbert_z * source_tier_weight AS w_z,
               source_tier_weight AS w,
               risk_types,
               country_ids
        FROM read_parquet('{V9}')
        WHERE finbert_z IS NOT NULL
          AND source_tier_weight IS NOT NULL
          AND title IS NOT NULL AND title != ''
    """
    df = con.execute(q).fetchdf()
    df["event_time"] = pd.to_datetime(df["event_time"])
    df["country"] = df["country_ids"].apply(normalize_country)
    return df


def monthly_series(df_g: pd.DataFrame) -> pd.Series:
    b = df_g["event_time"].dt.to_period("M").dt.start_time
    g = df_g.assign(_b=b).groupby("_b")
    s = g["w_z"].sum() / g["w"].sum().replace(0, np.nan)
    return s.dropna().sort_index()


def placebo_one(df_g: pd.DataFrame, B: int, rng) -> dict:
    real = monthly_series(df_g)
    if len(real) < MIN_BUCKETS:
        return None
    rstd = float(real.std())
    rneg = int((real < PEAK_LOW).sum())
    rpos = int((real > PEAK_HIGH).sum())

    dates = df_g["event_time"].to_numpy()
    wz = df_g["w_z"].to_numpy()
    w = df_g["w"].to_numpy()
    n = len(df_g)

    nstd = np.empty(B, dtype=np.float32)
    nneg = np.empty(B, dtype=np.int32)
    npos = np.empty(B, dtype=np.int32)
    for b in range(B):
        perm = rng.permutation(n)
        tmp = pd.DataFrame({"event_time": dates[perm], "w_z": wz, "w": w})
        s = monthly_series(tmp)
        if len(s) == 0:
            nstd[b] = np.nan; nneg[b] = 0; npos[b] = 0
            continue
        nstd[b] = float(s.std())
        nneg[b] = int((s < PEAK_LOW).sum())
        npos[b] = int((s > PEAK_HIGH).sum())

    return {
        "n_rows": n,
        "n_buckets": len(real),
        "real_std": rstd,
        "null_std_mean": float(np.nanmean(nstd)),
        "null_std_q05": float(np.nanquantile(nstd, 0.05)),
        "null_std_q95": float(np.nanquantile(nstd, 0.95)),
        "std_ratio": rstd / float(np.nanmean(nstd)) if np.nanmean(nstd) > 0 else np.nan,
        "p_std": float((nstd >= rstd).mean()),
        "real_neg": rneg,
        "p_neg": float((nneg >= rneg).mean()),
        "real_pos": rpos,
        "p_pos": float((npos >= rpos).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    con = duckdb.connect()
    log.info("Loading v9 …")
    base = load_base(con)
    log.info("  rows: %d", len(base))

    rng = np.random.default_rng(args.seed)
    results = []

    # L1 pooled
    log.info("\n[L1] pooled")
    r = placebo_one(base, args.B, rng)
    if r:
        r["level"] = "L1_pooled"; r["group"] = "ALL"
        results.append(r)
        log.info("  n=%d  buckets=%d  real_std=%.3f  null=%.3f  ratio=%.3f  p=%.3f",
                 r["n_rows"], r["n_buckets"], r["real_std"],
                 r["null_std_mean"], r["std_ratio"], r["p_std"])

    # L2 country
    log.info("\n[L2] country (CN/KR/EU/...)")
    for ctry, grp in base.groupby("country"):
        if ctry == "" or len(grp) < 200:
            continue
        r = placebo_one(grp, args.B, rng)
        if r:
            r["level"] = "L2_country"; r["group"] = ctry
            results.append(r)
            log.info("  %-8s n=%6d buckets=%3d real=%.3f null=%.3f ratio=%.3f p=%.3f",
                     ctry, r["n_rows"], r["n_buckets"], r["real_std"],
                     r["null_std_mean"], r["std_ratio"], r["p_std"])

    # L3 risk_type non-other (single only, not composite)
    log.info("\n[L3] risk_type non-other (single only)")
    single_nz = base[
        base["risk_types"].notna()
        & (base["risk_types"] != "other")
        & (~base["risk_types"].str.contains(",", na=False))
    ]
    for rt, grp in single_nz.groupby("risk_types"):
        if len(grp) < 100:
            continue
        r = placebo_one(grp, args.B, rng)
        if r:
            r["level"] = "L3_risk"; r["group"] = rt
            results.append(r)
            log.info("  %-15s n=%5d buckets=%3d real=%.3f null=%.3f ratio=%.3f p=%.3f",
                     rt, r["n_rows"], r["n_buckets"], r["real_std"],
                     r["null_std_mean"], r["std_ratio"], r["p_std"])

    # L4 country × risk
    log.info("\n[L4] country × risk")
    base["_rt"] = base["risk_types"]
    mask = (base["_rt"].notna() & (base["_rt"] != "other") &
            (~base["_rt"].str.contains(",", na=False)) &
            (base["country"] != ""))
    for (ctry, rt), grp in base[mask].groupby(["country", "_rt"]):
        if len(grp) < 100:
            continue
        r = placebo_one(grp, args.B, rng)
        if r:
            r["level"] = "L4_country_risk"; r["group"] = f"{ctry}|{rt}"
            results.append(r)
            log.info("  %-20s n=%5d buckets=%3d real=%.3f null=%.3f ratio=%.3f p=%.3f",
                     f"{ctry}|{rt}", r["n_rows"], r["n_buckets"], r["real_std"],
                     r["null_std_mean"], r["std_ratio"], r["p_std"])

    out = pd.DataFrame(results)
    out = out[["level", "group", "n_rows", "n_buckets", "real_std",
               "null_std_mean", "null_std_q05", "null_std_q95",
               "std_ratio", "p_std", "real_neg", "p_neg",
               "real_pos", "p_pos"]]
    out.to_csv(REPORTS / "placebo_aggregate.csv", index=False)

    log.info("\n== FINAL AGGREGATE SUMMARY ==")
    log.info("\n%s", out.to_string(index=False))
    sig_n = (out["p_std"] < 0.05).sum()
    log.info("\nsig p_std<0.05 : %d / %d groups", sig_n, len(out))


if __name__ == "__main__":
    main()
