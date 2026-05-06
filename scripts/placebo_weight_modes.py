"""
Phase 2.6: Placebo falsification with 3 competing weight schemes × 3 frequencies

v9 의 cluster 정보를 세 가지 방식으로 weight 화하고 placebo test 재수행.

Weight mode 정의
----------------
  dedup      : source_tier_weight if is_cluster_lead else 0
               — Baker-Bloom-Davis (2016, QJE), Boschee et al. (2015, ICEWS)
               해석: "중복은 syndication noise"

  attention  : source_tier_weight × log(1 + cluster_size)
               — Tetlock (2007, JF), Fang-Peress (2009, JF), Peress (2014, JF)
               해석: "cluster 크기 = 시장이 그 사건에 쏟은 주목도"
               log 형태: count saturation (Tetlock 이 word freq 에 사용한 구조)

  raw        : source_tier_weight (cluster_size 무시, lead/non-lead 모두 유지)
               baseline — 아무 cluster 정보도 쓰지 않음

Frequencies: D(daily), W(weekly), M(monthly)
총 9개 설정. 각각 39 firm × B=500 placebo.

가설 (추론):
  dedup 이 signal 약한 건 Phase 2 에서 확인. attention 이 monthly 에서 가장 강한
  std_ratio 를 보일 것으로 예상. raw 는 그 중간.
  만약 attention 이 dedup 보다 뚜렷이 낫다면 → Phase 3 의 그래프 엣지는
  Tetlock-style attention weighting 으로 가는 것이 논문 방어에 유리.
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


def load_v9(con):
    q = f"""
        SELECT company_id,
               event_time,
               finbert_z,
               source_tier_weight,
               is_cluster_lead,
               cluster_size
        FROM read_parquet('{V9}')
        WHERE finbert_z IS NOT NULL
          AND company_id IS NOT NULL
          AND title IS NOT NULL AND title != ''
          AND source_tier_weight IS NOT NULL
    """
    df = con.execute(q).fetchdf()
    df["event_time"] = pd.to_datetime(df["event_time"])
    df["cluster_size"] = df["cluster_size"].fillna(1).astype("int32")
    return df


def apply_weight(df: pd.DataFrame, mode: str) -> np.ndarray:
    stw = df["source_tier_weight"].to_numpy(dtype="float32")
    if mode == "dedup":
        lead = df["is_cluster_lead"].to_numpy()
        return np.where(lead, stw, np.float32(0.0)).astype("float32")
    if mode == "attention":
        cs = df["cluster_size"].to_numpy(dtype="float32")
        return (stw * np.log1p(cs)).astype("float32")
    if mode == "raw":
        return stw
    raise ValueError(mode)


def bucket(dates: pd.Series, freq: str) -> pd.Series:
    if freq == "D": return dates.dt.floor("D")
    if freq == "W": return dates.dt.to_period("W-SUN").dt.start_time
    if freq == "M": return dates.dt.to_period("M").dt.start_time
    raise ValueError(freq)


def agg_series(df_c: pd.DataFrame, freq: str) -> pd.Series:
    b = bucket(df_c["event_time"], freq)
    g = df_c.assign(_b=b).groupby("_b")
    wsum = g["w"].sum()
    s = g["w_z"].sum() / wsum.replace(0, np.nan)
    return s.dropna().sort_index()


def placebo_one(df: pd.DataFrame, freq: str, min_buckets: int,
                 B: int, seed: int):
    rng = np.random.default_rng(seed)
    rows = []
    for cid, grp in df.groupby("company_id"):
        real = agg_series(grp, freq)
        if len(real) < min_buckets:
            continue
        rstd = float(real.std())
        rneg = int((real < PEAK_LOW).sum())
        rpos = int((real > PEAK_HIGH).sum())

        dates = grp["event_time"].to_numpy()
        wz = grp["w_z"].to_numpy()
        w = grp["w"].to_numpy()
        n = len(grp)

        nstd = np.empty(B, dtype=np.float32)
        nneg = np.empty(B, dtype=np.int32)
        npos = np.empty(B, dtype=np.int32)
        for b in range(B):
            perm = rng.permutation(n)
            tmp = pd.DataFrame({"event_time": dates[perm], "w_z": wz, "w": w})
            s = agg_series(tmp, freq)
            if len(s) == 0:
                nstd[b] = np.nan; nneg[b] = 0; npos[b] = 0
                continue
            nstd[b] = float(s.std())
            nneg[b] = int((s < PEAK_LOW).sum())
            npos[b] = int((s > PEAK_HIGH).sum())

        rows.append({
            "company_id": cid,
            "n_rows": n,
            "n_buckets_real": len(real),
            "real_std": rstd,
            "placebo_std_mean": float(np.nanmean(nstd)),
            "p_std": float((nstd >= rstd).mean()),
            "real_neg": rneg,
            "p_neg": float((nneg >= rneg).mean()),
            "real_pos": rpos,
            "p_pos": float((npos >= rpos).mean()),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    con = duckdb.connect()
    log.info("Loading v9…")
    base = load_v9(con)
    log.info("  base rows: %d  companies: %d",
             len(base), base["company_id"].nunique())

    modes = ["dedup", "attention", "raw"]
    freq_cfg = [("D", 30), ("W", 12), ("M", 6)]

    summaries = []
    all_rows = []

    for mode in modes:
        w = apply_weight(base, mode)
        df_mode = base.copy()
        df_mode["w"] = w
        df_mode["w_z"] = df_mode["finbert_z"].to_numpy(dtype="float32") * w
        df_mode = df_mode[df_mode["w"] > 0]   # 0-weight rows skipped (dedup 모드용)

        for freq, min_b in freq_cfg:
            log.info("== %s × %s (min_buckets=%d) ==", mode, freq, min_b)
            res = placebo_one(df_mode, freq, min_b, args.B, args.seed)
            if len(res) == 0:
                log.info("  (no firms)")
                continue
            res["mode"] = mode
            res["freq"] = freq
            sig = int((res["p_std"] < 0.05).sum())
            ratio = float((res["real_std"] / res["placebo_std_mean"]).mean())
            median_p = float(res["p_std"].median())
            log.info("  n_firms=%d  sig(p<.05)=%d (%.1f%%)  std_ratio=%.4f  median_p=%.3f",
                     len(res), sig, 100 * sig / len(res), ratio, median_p)
            summaries.append({
                "mode": mode, "freq": freq,
                "n_firms": len(res),
                "sig_p05": sig,
                "sig_pct": 100 * sig / len(res),
                "mean_std_ratio": ratio,
                "median_p_std": median_p,
                "mean_real_std": float(res["real_std"].mean()),
                "mean_null_std": float(res["placebo_std_mean"].mean()),
            })
            all_rows.append(res)

    summ = pd.DataFrame(summaries)
    summ.to_csv(REPORTS / "placebo_weight_modes_summary.csv", index=False)
    pd.concat(all_rows, ignore_index=True).to_csv(
        REPORTS / "placebo_weight_modes_detail.csv", index=False)

    with open(REPORTS / "placebo_weight_modes_summary.txt", "w") as f:
        f.write("Phase 2.6  Placebo Weight-Mode Comparison\n")
        f.write(f"B={args.B}, seed={args.seed}\n\n")
        f.write(summ.to_string(index=False))
        f.write("\n")

    log.info("\n== FINAL SUMMARY ==")
    log.info("\n%s", summ.to_string(index=False))


if __name__ == "__main__":
    main()
