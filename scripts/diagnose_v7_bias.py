#!/usr/bin/env python3
"""
diagnose_v7_bias.py
───────────────────
v6 → v7 병합 후 기업별 편중 재진단.

비교 지표:
  1. scored=0 기업 수 (title-bearing 데이터 0개인 기업)
  2. Per-company row count 분포 + Gini coefficient
  3. Top/bottom N 기업
  4. classify_source별 row count (google_news_zero8 기여도)
  5. zero8 복구 대상 8개 기업의 before/after

Usage:
    python scripts/diagnose_v7_bias.py
    python scripts/diagnose_v7_bias.py --v6-only   # v6만 (v7 없어도 됨)
"""
import argparse
import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

log = logging.getLogger("diagnose_v7")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ROOT = Path(__file__).resolve().parent.parent
V6 = ROOT / "data" / "processed" / "risk_events_classified_v6.parquet"
V7 = ROOT / "data" / "processed" / "risk_events_classified_v7_final.parquet"
V7_RAW = ROOT / "data" / "processed" / "risk_events_classified_v7.parquet"

ZERO8_TARGETS = {
    "7267.T":  "Honda Motor",
    "F":       "Ford Motor",
    "GLEN.L":  "Glencore",
    "ALB":     "Albemarle",
    "LCID":    "Lucid Motors",
    "BAS.DE":  "BASF",
    "3407.T":  "Asahi Kasei",
    "SQM":     "SQM Lithium",
}


def gini(x: np.ndarray) -> float:
    """Gini coefficient of array (non-negative)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or x.sum() == 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    cum = np.cumsum(x)
    return (2.0 * np.sum((np.arange(1, n + 1)) * x) - (n + 1) * cum[-1]) / (n * cum[-1])


def analyze(con: duckdb.DuckDBPyConnection, path: Path, label: str) -> dict:
    log.info(f"\n{'=' * 70}\n[{label}] {path.name}\n{'=' * 70}")
    if not path.exists():
        log.warning(f"  file not found: {path}")
        return {}

    # 기본 통계
    total = con.execute(f"SELECT COUNT(*) FROM read_parquet('{path}')").fetchone()[0]
    with_title = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{path}') "
        f"WHERE title IS NOT NULL AND LENGTH(TRIM(title)) >= 10"
    ).fetchone()[0]
    with_company = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{path}') "
        f"WHERE company_id IS NOT NULL AND company_id != ''"
    ).fetchone()[0]

    log.info(f"  total rows        : {total:>12,}")
    log.info(f"  rows with title   : {with_title:>12,}  ({100*with_title/total:.1f}%)")
    log.info(f"  rows with company : {with_company:>12,}  ({100*with_company/total:.1f}%)")

    # classify_source 분포
    log.info("\n  classify_source distribution:")
    cs = con.execute(
        f"SELECT COALESCE(classify_source, '(null)') AS cs, COUNT(*) AS n "
        f"FROM read_parquet('{path}') GROUP BY 1 ORDER BY n DESC"
    ).fetchall()
    for name, n in cs:
        log.info(f"    {name:<30s}: {n:>10,}")

    # 기업별 title-bearing count
    per_co = con.execute(f"""
        SELECT company_id, COUNT(*) AS n_title
        FROM read_parquet('{path}')
        WHERE company_id IS NOT NULL AND company_id != ''
          AND title IS NOT NULL AND LENGTH(TRIM(title)) >= 10
        GROUP BY company_id ORDER BY n_title DESC
    """).fetchdf()

    n_companies = len(per_co)
    scored_zero = con.execute(f"""
        WITH all_co AS (
            SELECT DISTINCT company_id FROM read_parquet('{path}')
            WHERE company_id IS NOT NULL AND company_id != ''
        ),
        titled AS (
            SELECT DISTINCT company_id FROM read_parquet('{path}')
            WHERE company_id IS NOT NULL AND company_id != ''
              AND title IS NOT NULL AND LENGTH(TRIM(title)) >= 10
        )
        SELECT COUNT(*) FROM all_co WHERE company_id NOT IN (SELECT company_id FROM titled)
    """).fetchone()[0]

    log.info(f"\n  companies with title         : {n_companies}")
    log.info(f"  companies with scored=0      : {scored_zero}")

    if len(per_co) > 0:
        counts = per_co["n_title"].to_numpy()
        g = gini(counts)
        log.info(f"  Gini (title-rows)            : {g:.4f}")
        log.info(f"  top 5 by title count:")
        for _, r in per_co.head(5).iterrows():
            log.info(f"    {r['company_id']:<12}: {int(r['n_title']):>8,}")
        log.info(f"  bottom 5 by title count:")
        for _, r in per_co.tail(5).iterrows():
            log.info(f"    {r['company_id']:<12}: {int(r['n_title']):>8,}")

    # zero8 target 기업 상태
    log.info("\n  zero8 target company status:")
    for cid, name in ZERO8_TARGETS.items():
        n = con.execute(f"""
            SELECT COUNT(*) FROM read_parquet('{path}')
            WHERE company_id = '{cid}'
              AND title IS NOT NULL AND LENGTH(TRIM(title)) >= 10
        """).fetchone()[0]
        n_finbert = con.execute(f"""
            SELECT COUNT(*) FROM read_parquet('{path}')
            WHERE company_id = '{cid}' AND finbert_score IS NOT NULL
        """).fetchone()[0]
        flag = "✓" if n > 0 else "✗"
        log.info(f"    {flag} {cid:<10} {name:<16}: title={n:>6,}  finbert={n_finbert:>6,}")

    return {
        "total": total,
        "with_title": with_title,
        "n_companies": n_companies,
        "scored_zero": scored_zero,
        "gini": gini(per_co["n_title"].to_numpy()) if len(per_co) > 0 else 0.0,
        "per_co": per_co,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v6-only", action="store_true")
    args = parser.parse_args()

    con = duckdb.connect(":memory:")
    con.execute("PRAGMA memory_limit='3GB'")

    v6_stat = analyze(con, V6, "V6 (before)")

    if args.v6_only or not V7.exists():
        if not V7.exists() and not args.v6_only:
            log.warning(f"\nV7 not found yet: {V7}")
            log.warning("run `python scripts/merge_v6_to_v7.py` first.")
        return

    v7_stat = analyze(con, V7, "V7 (after)")

    if v6_stat and v7_stat:
        log.info(f"\n{'=' * 70}\nDELTA (v6 → v7)\n{'=' * 70}")
        log.info(f"  total rows      : {v6_stat['total']:>12,} → {v7_stat['total']:>12,}  ({v7_stat['total']-v6_stat['total']:+,})")
        log.info(f"  with title      : {v6_stat['with_title']:>12,} → {v7_stat['with_title']:>12,}  ({v7_stat['with_title']-v6_stat['with_title']:+,})")
        log.info(f"  companies       : {v6_stat['n_companies']:>12,} → {v7_stat['n_companies']:>12,}")
        log.info(f"  scored=0        : {v6_stat['scored_zero']:>12,} → {v7_stat['scored_zero']:>12,}")
        log.info(f"  Gini            : {v6_stat['gini']:>12.4f} → {v7_stat['gini']:>12.4f}  ({v7_stat['gini']-v6_stat['gini']:+.4f})")


if __name__ == "__main__":
    main()
