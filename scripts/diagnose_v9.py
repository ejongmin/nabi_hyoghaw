"""
Phase 2-F: v9 diagnostic — cross-lingual dedup 후 신뢰도/객관성 지표 재확인.

섹션:
  [1] cluster coverage    : cluster-matched / non-lead 비율
  [2] cluster size 분포    : singleton / 2~5 / 6~10 / 10+
  [3] 언어 교차 cluster     : 같은 cluster 내 서로 다른 finbert_model 개수
  [4] tier 승급 효과        : lead article 의 tier 분포 vs 전체
  [5] JA bias 재검증 (v9)   : dedup_weight 가중 평균
  [6] 기업별 dedup 감소율   : pre=n_title, post=cluster 개수
"""

import logging
from pathlib import Path
import duckdb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
V9 = ROOT / "data" / "processed" / "risk_events_classified_v9.parquet"


def main():
    assert V9.exists(), f"v9 not found: {V9}"
    con = duckdb.connect()
    log.info("Loading %s", V9.name)
    n_total = con.execute(f"SELECT COUNT(*) FROM read_parquet('{V9}')").fetchone()[0]
    log.info("  total rows: %d", n_total)

    log.info("\n[1] cluster coverage")
    r = con.execute(f"""
        SELECT
          SUM(CASE WHEN event_cluster_id IS NOT NULL THEN 1 ELSE 0 END) AS in_cluster,
          SUM(CASE WHEN is_cluster_lead THEN 1 ELSE 0 END) AS leads,
          SUM(CASE WHEN event_cluster_id IS NOT NULL AND NOT is_cluster_lead THEN 1 ELSE 0 END) AS non_leads,
          SUM(CASE WHEN dedup_weight > 0 THEN 1 ELSE 0 END) AS with_weight
        FROM read_parquet('{V9}')
    """).fetchone()
    log.info("  in_cluster : %d  (%.2f%%)", r[0], 100 * r[0] / n_total)
    log.info("  leads      : %d", r[1])
    log.info("  non_leads  : %d  (dedup_weight forced to 0)", r[2])
    log.info("  with_weight: %d  (%.2f%%)", r[3], 100 * r[3] / n_total)

    log.info("\n[2] cluster size distribution")
    for row in con.execute(f"""
        SELECT CASE
                 WHEN cluster_size = 1 THEN '1 (singleton)'
                 WHEN cluster_size BETWEEN 2 AND 5 THEN '2-5'
                 WHEN cluster_size BETWEEN 6 AND 10 THEN '6-10'
                 ELSE '10+'
               END AS bucket,
               COUNT(*) AS n
        FROM read_parquet('{V9}')
        WHERE event_cluster_id IS NOT NULL
        GROUP BY bucket ORDER BY bucket
    """).fetchall():
        log.info("  %-15s %d", row[0], row[1])

    log.info("\n[3] cross-lingual clusters (# of distinct finbert_model in same cluster)")
    for row in con.execute(f"""
        WITH c AS (
          SELECT event_cluster_id, COUNT(DISTINCT finbert_model) AS n_lang
          FROM read_parquet('{V9}')
          WHERE event_cluster_id IS NOT NULL
          GROUP BY event_cluster_id
        )
        SELECT n_lang, COUNT(*) AS n_clusters FROM c GROUP BY n_lang ORDER BY n_lang
    """).fetchall():
        log.info("  %d-lang clusters: %d", row[0], row[1])

    log.info("\n[4] lead-article source_tier distribution (vs all title+finbert)")
    for row in con.execute(f"""
        SELECT source_tier,
               SUM(CASE WHEN is_cluster_lead THEN 1 ELSE 0 END) AS leads,
               COUNT(*) AS total
        FROM read_parquet('{V9}')
        WHERE title IS NOT NULL AND finbert_score IS NOT NULL
        GROUP BY source_tier ORDER BY source_tier
    """).fetchall():
        pct_lead = 100 * row[1] / row[2] if row[2] else 0
        log.info("  tier %s  leads=%d  total=%d  lead%%=%.1f", row[0], row[1], row[2], pct_lead)

    log.info("\n[5] JA bias re-check under dedup_weight")
    for row in con.execute(f"""
        SELECT source_tier,
               COUNT(*) AS n_rows,
               SUM(CASE WHEN dedup_weight > 0 THEN 1 ELSE 0 END) AS n_active,
               SUM(finbert_z * dedup_weight) / NULLIF(SUM(dedup_weight), 0) AS wz_mean,
               SUM(CASE WHEN finbert_score > 0.5 AND dedup_weight > 0 THEN 1 ELSE 0 END) * 1.0
                 / NULLIF(SUM(CASE WHEN dedup_weight > 0 THEN 1 ELSE 0 END), 0) AS strong_pos_frac
        FROM read_parquet('{V9}')
        WHERE finbert_model = 'christian-phu/bert-finetuned-japanese-sentiment'
        GROUP BY source_tier ORDER BY source_tier
    """).fetchall():
        log.info("  tier %s  n=%d  active=%d  weighted_z_mean=%s  strong_pos=%s",
                 row[0], row[1], row[2],
                 f"{row[3]:+.3f}" if row[3] is not None else "NA",
                 f"{row[4]*100:.1f}%" if row[4] is not None else "NA")

    log.info("\n[6] per-company dedup reduction (top 10 by title count)")
    log.info("  %-12s %8s %8s %8s %6s", "company", "titles", "clusters", "leads", "dedup%")
    for row in con.execute(f"""
        SELECT company_id,
               COUNT(*) AS titles,
               COUNT(DISTINCT event_cluster_id) AS clusters,
               SUM(CASE WHEN is_cluster_lead THEN 1 ELSE 0 END) AS leads
        FROM read_parquet('{V9}')
        WHERE title IS NOT NULL AND finbert_score IS NOT NULL
          AND event_cluster_id IS NOT NULL
        GROUP BY company_id
        ORDER BY titles DESC LIMIT 10
    """).fetchall():
        cid, t, c, l = row
        dedup_pct = 100 * (1 - c / t) if t else 0
        log.info("  %-12s %8d %8d %8d  %5.1f%%", cid, t, c, l, dedup_pct)


if __name__ == "__main__":
    main()
