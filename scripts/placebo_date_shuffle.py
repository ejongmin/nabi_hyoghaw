"""
Phase 2-E: Placebo / Date-Shuffle Falsification Test

목적: v9 의 finbert_z × dedup_weight 로 구성한 기업별 일일 감정 시계열이,
실제 이벤트 타이밍(event_time) 과 진짜로 연결되어 있는지 — 아니면 단순
매체별 베이스라인 드리프트인지 — 구분한다.

방법 (Cohen-Frazzini 2008, JF 의 placebo shuffling 아이디어 응용):
  1) 각 기업의 '일별 집계 감정' 실제 시계열 S_real(c, t) 계산.
  2) event_time 을 그 기업 내에서 uniformly 셔플한 뒤 같은 방법으로
     S_placebo(c, t) 재계산 — B = 500회 반복.
  3) 실제 시계열의 '고위험 피크' 개수 (z < -2) 가 placebo 분포의 상위
     5% 를 넘는지 확인 → 진짜 신호라는 증거.
  4) 동일하게 '저위험 피크' (z > +2) 도 확인.

또한 전체 코퍼스 수준에서:
  - pooled_real_std      : 실제 시계열 std
  - pooled_placebo_std   : placebo 평균 std
  - ratio real/placebo   > 1.5 이면 진짜 신호가 있음의 증거 (추론)

참고 논문:
  Cohen, Frazzini (2008) *"Economic Links and Predictable Returns"* JF —
    placebo/fake-link 대조군 구성 원리.
  Politis, Romano (1994) — block bootstrap 의 null distribution 해석.
  Baker-Bloom-Davis (2016, QJE) — EPU 의 풍문/noise 구분 방식.

입력: data/processed/risk_events_classified_v9.parquet
출력:
  - reports/placebo_summary.txt  (기업별 실제 vs placebo p-value 표)
  - reports/placebo_null_hist.csv (null 분포 원시값)
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

# 기본 파라미터
DEFAULT_B = 500       # placebo 반복 수
PEAK_LOW = -2.0       # 고위험 피크 (negative sentiment)
PEAK_HIGH = +2.0      # 저위험 피크


def load_clustered(con, min_rows: int):
    """lead 또는 ambient 행만 + finbert_z 보유."""
    q = f"""
        SELECT company_id,
               CAST(event_time AS DATE) AS d,
               finbert_z * dedup_weight AS w_z,
               dedup_weight AS w
        FROM read_parquet('{V9}')
        WHERE finbert_z IS NOT NULL
          AND dedup_weight > 0
          AND company_id IS NOT NULL
    """
    df = con.execute(q).fetchdf()
    # 기업별 최소 행수 필터
    counts = df.groupby("company_id").size()
    keep = counts[counts >= min_rows].index
    df = df[df["company_id"].isin(keep)].copy()
    return df, keep.tolist()


def daily_series(df_c: pd.DataFrame) -> pd.Series:
    """주어진 기업 df → 일별 가중평균 z."""
    g = df_c.groupby("d")
    s = g["w_z"].sum() / g["w"].sum().replace(0, np.nan)
    return s.dropna().sort_index()


def count_peaks(s: pd.Series, low: float, high: float):
    return int((s < low).sum()), int((s > high).sum())


def run_placebo(df: pd.DataFrame, B: int, seed: int):
    rng = np.random.default_rng(seed)
    results = []

    for cid, grp in df.groupby("company_id"):
        real = daily_series(grp)
        if len(real) < 10:
            continue
        real_neg, real_pos = count_peaks(real, PEAK_LOW, PEAK_HIGH)
        real_std = float(real.std())

        # placebo: shuffle dates within company
        dates_arr = grp["d"].to_numpy()
        n = len(dates_arr)
        null_neg = np.empty(B, dtype=np.int32)
        null_pos = np.empty(B, dtype=np.int32)
        null_std = np.empty(B, dtype=np.float32)
        wz = grp["w_z"].to_numpy()
        w = grp["w"].to_numpy()

        for b in range(B):
            perm = rng.permutation(n)
            tmp = pd.DataFrame({"d": dates_arr[perm], "w_z": wz, "w": w})
            s = daily_series(tmp)
            if len(s) == 0:
                null_neg[b] = null_pos[b] = 0
                null_std[b] = np.nan
                continue
            nn, np_ = count_peaks(s, PEAK_LOW, PEAK_HIGH)
            null_neg[b] = nn
            null_pos[b] = np_
            null_std[b] = s.std()

        # empirical p-value (one-sided, right tail)
        p_neg = float((null_neg >= real_neg).mean())
        p_pos = float((null_pos >= real_pos).mean())
        p_std = float((null_std >= real_std).mean())

        results.append({
            "company_id": cid,
            "n_rows": n,
            "n_days_real": len(real),
            "real_neg_peaks": real_neg,
            "placebo_neg_p95": int(np.quantile(null_neg, 0.95)),
            "p_neg": p_neg,
            "real_pos_peaks": real_pos,
            "placebo_pos_p95": int(np.quantile(null_pos, 0.95)),
            "p_pos": p_pos,
            "real_std": real_std,
            "placebo_std_mean": float(np.nanmean(null_std)),
            "p_std": p_std,
        })

    return pd.DataFrame(results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=DEFAULT_B)
    ap.add_argument("--min-rows", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    assert V9.exists(), f"v9 not found: {V9}"
    con = duckdb.connect()
    log.info("Loading v9 finbert rows…")
    df, kept = load_clustered(con, args.min_rows)
    log.info("  companies kept (≥%d rows): %d", args.min_rows, len(kept))
    log.info("  total rows: %d", len(df))

    log.info("Running placebo B=%d …", args.B)
    res = run_placebo(df, args.B, args.seed)
    res = res.sort_values("p_std").reset_index(drop=True)

    out_txt = REPORTS / "placebo_summary.txt"
    res.to_csv(REPORTS / "placebo_summary.csv", index=False)
    with open(out_txt, "w") as f:
        f.write(f"Placebo Date-Shuffle Falsification Test\n")
        f.write(f"B={args.B}, min_rows={args.min_rows}, seed={args.seed}\n")
        f.write(f"PEAK_LOW={PEAK_LOW}, PEAK_HIGH={PEAK_HIGH}\n\n")
        f.write(res.to_string(index=False))
        f.write("\n\n")
        sig_neg = (res["p_neg"] < 0.05).sum()
        sig_pos = (res["p_pos"] < 0.05).sum()
        sig_std = (res["p_std"] < 0.05).sum()
        f.write(f"Companies with p_neg < 0.05: {sig_neg}/{len(res)}\n")
        f.write(f"Companies with p_pos < 0.05: {sig_pos}/{len(res)}\n")
        f.write(f"Companies with p_std < 0.05: {sig_std}/{len(res)}\n")

    log.info("Wrote %s", out_txt)
    log.info("sig p_std < 0.05: %d / %d companies",
             int((res["p_std"] < 0.05).sum()), len(res))


if __name__ == "__main__":
    main()
