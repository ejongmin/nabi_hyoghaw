#!/usr/bin/env python3
"""
diagnose_v8.py
──────────────
v8 (v7_final + Phase 1 columns) 진단. 검증 항목:

  1. Publisher 추출 커버리지
  2. Source tier 분포 (전체 + title-bearing + zero8)
  3. Language-stratified z-score 재검증 (각 모델이 μ≈0, σ≈1)
  4. Regime dummy 행 수 + 교차 집계
  5. Tier 1-only subset에서 scored=0 재검증 (bias 해소 여부)
  6. 기업별 tier mix

Usage:
    python scripts/diagnose_v8.py
"""
import logging
from pathlib import Path

import duckdb

log = logging.getLogger("diagnose_v8")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ROOT = Path(__file__).resolve().parent.parent
V8 = ROOT / "data" / "processed" / "risk_events_classified_v8.parquet"


def main():
    if not V8.exists():
        log.error(f"v8 not found: {V8}")
        return

    con = duckdb.connect(":memory:")
    con.execute("PRAGMA memory_limit='3GB'")

    log.info(f"Loading {V8.name}")
    total = con.execute(f"SELECT COUNT(*) FROM read_parquet('{V8}')").fetchone()[0]
    log.info(f"  total rows: {total:,}")

    # ─── 1. Publisher coverage ───
    log.info("\n[1] Publisher extraction coverage")
    rows = con.execute(f"""
        SELECT
            COUNT(*) FILTER (WHERE publisher IS NOT NULL AND publisher != '') AS n_with_pub,
            COUNT(*) FILTER (WHERE title IS NOT NULL AND LENGTH(TRIM(title)) >= 10) AS n_title,
            COUNT(*) FILTER (WHERE title IS NOT NULL AND LENGTH(TRIM(title)) >= 10
                             AND publisher IS NOT NULL AND publisher != '') AS n_title_with_pub
        FROM read_parquet('{V8}')
    """).fetchone()
    log.info(f"  with publisher           : {rows[0]:,} ({100*rows[0]/total:.1f}%)")
    log.info(f"  with title               : {rows[1]:,}")
    log.info(f"  title ∩ publisher        : {rows[2]:,}  ({100*rows[2]/max(rows[1],1):.1f}% of title-bearing)")

    # ─── 2. Tier distribution ───
    log.info("\n[2] Source tier distribution")
    tier_df = con.execute(f"""
        SELECT source_tier, COUNT(*) AS n_all,
               COUNT(*) FILTER (WHERE title IS NOT NULL AND LENGTH(TRIM(title)) >= 10) AS n_title,
               COUNT(*) FILTER (WHERE classify_source = 'google_news_zero8') AS n_zero8,
               COUNT(*) FILTER (WHERE finbert_score IS NOT NULL) AS n_finbert
        FROM read_parquet('{V8}')
        GROUP BY source_tier ORDER BY source_tier
    """).fetchdf()
    for _, r in tier_df.iterrows():
        log.info(
            f"  tier {int(r['source_tier'])}: "
            f"all={int(r['n_all']):>12,}  "
            f"title={int(r['n_title']):>8,}  "
            f"zero8={int(r['n_zero8']):>7,}  "
            f"finbert={int(r['n_finbert']):>7,}"
        )

    # ─── 3. Z-score re-check per model ───
    log.info("\n[3] Language-stratified z-score re-check (should be μ≈0, σ≈1 per model)")
    z_df = con.execute(f"""
        SELECT finbert_model,
               COUNT(*) AS n,
               AVG(finbert_z) AS mean_z,
               STDDEV_SAMP(finbert_z) AS std_z,
               AVG(finbert_score) AS mean_raw,
               STDDEV_SAMP(finbert_score) AS std_raw
        FROM read_parquet('{V8}')
        WHERE finbert_z IS NOT NULL
        GROUP BY finbert_model ORDER BY n DESC
    """).fetchdf()
    for _, r in z_df.iterrows():
        log.info(
            f"  {str(r['finbert_model']):<60s}  n={int(r['n']):>7,}  "
            f"raw μ={r['mean_raw']:+.3f} σ={r['std_raw']:.3f}  "
            f"z μ={r['mean_z']:+.3f} σ={r['std_z']:.3f}"
        )

    # ─── 4. Regime distribution ───
    log.info("\n[4] Regime dummy coverage")
    for col in ["regime_covid", "regime_ukraine", "regime_ira", "regime_china_export", "regime_post_covid"]:
        n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{V8}') WHERE {col} = TRUE").fetchone()[0]
        log.info(f"  {col:<22s}: {n:>12,}  ({100*n/total:.1f}%)")

    # ─── 5. Tier 1-only bias re-check ───
    log.info("\n[5] Tier 1-only subset: scored=0 companies (bias mitigation check)")
    scored_by_tier = con.execute(f"""
        WITH base AS (
            SELECT company_id, source_tier
            FROM read_parquet('{V8}')
            WHERE company_id IS NOT NULL AND company_id != ''
              AND title IS NOT NULL AND LENGTH(TRIM(title)) >= 10
              AND finbert_score IS NOT NULL
        )
        SELECT
            COUNT(DISTINCT CASE WHEN source_tier = 1 THEN company_id END) AS n_t1_co,
            COUNT(DISTINCT CASE WHEN source_tier <= 2 THEN company_id END) AS n_t12_co,
            COUNT(DISTINCT company_id) AS n_any_co,
            COUNT(*) FILTER (WHERE source_tier = 1) AS n_t1_rows,
            COUNT(*) FILTER (WHERE source_tier <= 2) AS n_t12_rows,
            COUNT(*) AS n_all_rows
        FROM base
    """).fetchone()
    log.info(f"  title+finbert rows (all tiers): {scored_by_tier[5]:>8,}  ({scored_by_tier[2]:>3} companies)")
    log.info(f"  title+finbert rows (T1+T2)    : {scored_by_tier[4]:>8,}  ({scored_by_tier[1]:>3} companies)")
    log.info(f"  title+finbert rows (T1 only)  : {scored_by_tier[3]:>8,}  ({scored_by_tier[0]:>3} companies)")

    # JA bias specific check
    log.info("\n[6] JA positive bias re-check (christian-phu model)")
    ja_stats = con.execute(f"""
        SELECT source_tier,
               COUNT(*) AS n,
               AVG(finbert_score) AS mean_raw,
               AVG(CASE WHEN finbert_score > 0.3 THEN 1.0 ELSE 0.0 END) AS frac_strong_pos,
               AVG(finbert_z) AS mean_z
        FROM read_parquet('{V8}')
        WHERE finbert_model = 'christian-phu/bert-finetuned-japanese-sentiment'
        GROUP BY source_tier ORDER BY source_tier
    """).fetchdf()
    for _, r in ja_stats.iterrows():
        log.info(
            f"  tier {int(r['source_tier'])}: n={int(r['n']):>5,}  "
            f"raw μ={r['mean_raw']:+.3f}  strong+={100*r['frac_strong_pos']:.1f}%  z μ={r['mean_z']:+.3f}"
        )

    # ─── 7. Per-company tier mix (top 10 companies by title count) ───
    log.info("\n[7] Per-company tier mix (top 10 by title count)")
    mix = con.execute(f"""
        WITH t AS (
            SELECT company_id, source_tier, COUNT(*) AS n
            FROM read_parquet('{V8}')
            WHERE company_id IS NOT NULL AND company_id != ''
              AND title IS NOT NULL AND LENGTH(TRIM(title)) >= 10
            GROUP BY 1, 2
        ),
        piv AS (
            SELECT company_id,
                   SUM(CASE WHEN source_tier = 1 THEN n ELSE 0 END) AS t1,
                   SUM(CASE WHEN source_tier = 2 THEN n ELSE 0 END) AS t2,
                   SUM(CASE WHEN source_tier = 3 THEN n ELSE 0 END) AS t3,
                   SUM(n) AS total
            FROM t GROUP BY company_id
        )
        SELECT * FROM piv ORDER BY total DESC LIMIT 10
    """).fetchdf()
    log.info(f"  {'company':<12} {'T1':>6} {'T2':>6} {'T3':>6} {'total':>7}  T1%")
    for _, r in mix.iterrows():
        total = int(r['total'])
        t1pct = 100 * int(r['t1']) / max(total, 1)
        log.info(f"  {r['company_id']:<12} {int(r['t1']):>6,} {int(r['t2']):>6,} {int(r['t3']):>6,} {total:>7,}  {t1pct:>5.1f}%")


if __name__ == "__main__":
    main()
