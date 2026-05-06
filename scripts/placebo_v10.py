"""
Step 3b: v10 기반 placebo 재실행 (finbert2_z_month)

placebo_aggregate.py 와 동일한 L1/L2/L3/L4 구조를 유지하되
  - 입력을 v9 → v10 로 변경
  - finbert_z (45K 한정) → finbert2_z_month (확장) 로 전환
  - weight 동일: source_tier_weight
  - B=1000 shuffle

출력:
  reports/placebo_v10_aggregate.csv
  reports/placebo_v10_summary.txt
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
V10 = ROOT / "data" / "processed" / "risk_events_classified_v10.parquet"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

PEAK_LOW = -2.0
PEAK_HIGH = +2.0
MIN_BUCKETS = 12


def normalize_country(cid):
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
               finbert2_z_month * source_tier_weight AS w_z,
               source_tier_weight AS w,
               risk_types,
               country_ids
        FROM read_parquet('{V10}')
        WHERE finbert2_z_month IS NOT NULL
          AND source_tier_weight IS NOT NULL
    """
    df = con.execute(q).fetchdf()
    df["event_time"] = pd.to_datetime(df["event_time"])
    df["country"] = df["country_ids"].apply(normalize_country)
    return df


def monthly_series(df_g):
    b = df_g["event_time"].dt.to_period("M").dt.start_time
    g = df_g.assign(_b=b).groupby("_b")
    s = g["w_z"].sum() / g["w"].sum().replace(0, np.nan)
    return s.dropna().sort_index()


def placebo_one(df_g, B, rng):
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
    con.execute("PRAGMA memory_limit='2GB'")
    con.execute("PRAGMA threads=2")
    log.info("Loading v10 (finbert2_z_month NOT NULL)...")
    base = load_base(con)
    log.info("  rows: %d", len(base))

    rng = np.random.default_rng(args.seed)
    results = []

    log.info("\n[L1] pooled")
    r = placebo_one(base, args.B, rng)
    if r:
        r["level"] = "L1_pooled"; r["group"] = "ALL"; results.append(r)
        log.info("  n=%d buckets=%d real=%.3f null=%.3f ratio=%.3f p=%.3f",
                 r["n_rows"], r["n_buckets"], r["real_std"],
                 r["null_std_mean"], r["std_ratio"], r["p_std"])

    log.info("\n[L2] country")
    for ctry, grp in base.groupby("country"):
        if ctry == "" or len(grp) < 200:
            continue
        r = placebo_one(grp, args.B, rng)
        if r:
            r["level"] = "L2_country"; r["group"] = ctry; results.append(r)
            log.info("  %-8s n=%6d b=%3d real=%.3f null=%.3f ratio=%.3f p=%.3f",
                     ctry, r["n_rows"], r["n_buckets"], r["real_std"],
                     r["null_std_mean"], r["std_ratio"], r["p_std"])

    log.info("\n[L3] risk_type single")
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
            r["level"] = "L3_risk"; r["group"] = rt; results.append(r)
            log.info("  %-15s n=%5d b=%3d real=%.3f null=%.3f ratio=%.3f p=%.3f",
                     rt, r["n_rows"], r["n_buckets"], r["real_std"],
                     r["null_std_mean"], r["std_ratio"], r["p_std"])

    log.info("\n[L4] country × risk")
    base["_rt"] = base["risk_types"]
    mask = (base["_rt"].notna() & (base["_rt"] != "other")
            & (~base["_rt"].str.contains(",", na=False))
            & (base["country"] != ""))
    for (ctry, rt), grp in base[mask].groupby(["country", "_rt"]):
        if len(grp) < 100:
            continue
        r = placebo_one(grp, args.B, rng)
        if r:
            r["level"] = "L4_country_risk"; r["group"] = f"{ctry}|{rt}"; results.append(r)
            log.info("  %-20s n=%5d b=%3d real=%.3f null=%.3f ratio=%.3f p=%.3f",
                     f"{ctry}|{rt}", r["n_rows"], r["n_buckets"], r["real_std"],
                     r["null_std_mean"], r["std_ratio"], r["p_std"])

    out = pd.DataFrame(results)[["level", "group", "n_rows", "n_buckets", "real_std",
                                  "null_std_mean", "null_std_q05", "null_std_q95",
                                  "std_ratio", "p_std", "real_neg", "p_neg",
                                  "real_pos", "p_pos"]]
    out.to_csv(REPORTS / "placebo_v10_aggregate.csv", index=False)

    lines = ["=" * 76,
             "Phase 2.7-v10 · Aggregate placebo on finbert2_z_month (B=" + str(args.B) + ")",
             "=" * 76, "",
             f"Source : {V10.name}", f"Base rows (finbert2 not null): {len(base):,}",
             ""]
    lines.append(out.to_string(index=False))
    lines.append("")
    sig_n = (out["p_std"] < 0.05).sum()
    lines.append(f"sig p_std<0.05 : {sig_n} / {len(out)} groups")
    lines.append("")
    lines.append("비교 : v9 (finbert_z, 45K) 의 placebo_aggregate.csv 결과와 std_ratio / p_std 를 나란히 표기해 보고서에 삽입할 것.")
    (REPORTS / "placebo_v10_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    log.info("\n%s", "\n".join(lines))


if __name__ == "__main__":
    main()
