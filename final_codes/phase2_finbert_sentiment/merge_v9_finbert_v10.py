"""
Step 3a: v9 + finbert_v2 → v10 merge.

설계:
  - v9 는 건드리지 않고 v10 를 새로 만든다.
  - finbert_v2 (Colab 결과물) 를 v9.url 로 left join.
  - 기존 finbert_* 컬럼과 구분하기 위해 새 컬럼은 `finbert2_*` prefix 로 저장.
  - finbert2_z : 월별(first_date_raw YYYYMM) 평균/표준편차 기반 정규화.
    외부 기준은 월별이 가장 안정적 (EPU / CEPU 논문 관행).
  - finbert2_z 없는 row 는 NULL. 기존 finbert_z 는 그대로 보존.

입력:
  data/processed/risk_events_classified_v9.parquet
  data/processed/finbert_v2.parquet    ← Colab 결과물, 업로드 받은 후

출력:
  data/processed/risk_events_classified_v10.parquet
  reports/v10_merge_summary.txt
"""
from __future__ import annotations

import logging
from pathlib import Path

import duckdb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
V9 = ROOT / "data" / "processed" / "risk_events_classified_v9.parquet"
FB2 = ROOT / "data" / "processed" / "finbert_v2.parquet"
V10 = ROOT / "data" / "processed" / "risk_events_classified_v10.parquet"
REPORT = ROOT / "reports" / "v10_merge_summary.txt"


def main():
    assert V9.exists(), f"v9 missing: {V9}"
    assert FB2.exists(), f"finbert_v2 missing: {FB2} (Colab 결과물 업로드 필요)"
    V10.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='2GB'")
    con.execute("PRAGMA threads=2")
    con.execute("SET preserve_insertion_order=false")
    (ROOT / "tmp_duck").mkdir(exist_ok=True)
    con.execute(f"PRAGMA temp_directory='{ROOT / 'tmp_duck'}'")
    con.execute("PRAGMA max_temp_directory_size='8GiB'")

    log.info("Building v10 with finbert_v2 merge + monthly z ...")
    con.execute(f"""
        CREATE OR REPLACE TABLE fb2 AS
        SELECT
            url,
            finbert_pos  AS finbert2_pos,
            finbert_neg  AS finbert2_neg,
            finbert_neu  AS finbert2_neu,
            finbert_score AS finbert2_score,
            lid_en_conf AS finbert2_lid_conf,
            -- 월별 z-score (YYYYMM)
            (finbert_score - AVG(finbert_score)
                OVER (PARTITION BY SUBSTRING(first_date_raw, 1, 6)))
            / NULLIF(STDDEV_SAMP(finbert_score)
                OVER (PARTITION BY SUBSTRING(first_date_raw, 1, 6)), 0)
                AS finbert2_z_month
        FROM read_parquet('{FB2}')
    """)

    con.execute(f"""
        COPY (
            SELECT
                v9.*,
                fb2.finbert2_pos,
                fb2.finbert2_neg,
                fb2.finbert2_neu,
                fb2.finbert2_score,
                fb2.finbert2_z_month,
                fb2.finbert2_lid_conf
            FROM read_parquet('{V9}') v9
            LEFT JOIN fb2 ON v9.url = fb2.url
        )
        TO '{V10}' (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 500_000)
    """)
    log.info("Wrote %s", V10)

    # summary
    n_v10 = con.execute(f"SELECT COUNT(*) FROM read_parquet('{V10}')").fetchone()[0]
    n_fb1 = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{V10}') WHERE finbert_z IS NOT NULL"
    ).fetchone()[0]
    n_fb2 = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{V10}') WHERE finbert2_score IS NOT NULL"
    ).fetchone()[0]
    n_both = con.execute(f"""
        SELECT COUNT(*) FROM read_parquet('{V10}')
        WHERE finbert_z IS NOT NULL AND finbert2_score IS NOT NULL
    """).fetchone()[0]
    n_tone_nz = con.execute(f"""
        SELECT COUNT(*) FROM read_parquet('{V10}')
        WHERE tone IS NOT NULL AND tone != 0
    """).fetchone()[0]

    # fb2 ↔ tone corr (tone nonzero subset)
    corr_row = con.execute(f"""
        SELECT CORR(finbert2_score, tone)
        FROM read_parquet('{V10}')
        WHERE finbert2_score IS NOT NULL AND tone IS NOT NULL AND tone != 0
    """).fetchone()
    corr_fb2_tone = corr_row[0] if corr_row and corr_row[0] is not None else float("nan")

    # fb1 ↔ fb2 corr (overlap)
    corr_fb12 = con.execute(f"""
        SELECT CORR(finbert_score, finbert2_score)
        FROM read_parquet('{V10}')
        WHERE finbert_score IS NOT NULL AND finbert2_score IS NOT NULL
    """).fetchone()[0]

    lines = []
    lines.append("=" * 72)
    lines.append("Step 3a · v9 → v10 merge summary")
    lines.append("=" * 72)
    lines.append(f"v10 rows                  : {n_v10:>12,}")
    lines.append(f"  finbert (v1, 45K 원래)  : {n_fb1:>12,}")
    lines.append(f"  finbert2 (확장)         : {n_fb2:>12,}")
    lines.append(f"  fb1 ∩ fb2 overlap       : {n_both:>12,}")
    lines.append(f"  tone != 0               : {n_tone_nz:>12,}")
    lines.append("")
    lines.append("-- correlations --")
    lines.append(f"  corr(finbert2_score, tone)     = {corr_fb2_tone:+.4f}  "
                 "← V2Tone validator 가설 검증")
    lines.append(f"  corr(finbert_v1, finbert2)     = "
                 + (f"{corr_fb12:+.4f}" if corr_fb12 is not None else "N/A"))
    lines.append("")
    lines.append("Note:")
    lines.append("  - finbert2_z_month 는 YYYYMM 파티션 z-score.")
    lines.append("  - placebo 재실행은 scripts/placebo_v10.py 에서 finbert2_z_month 사용.")
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    log.info("Report -> %s", REPORT)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
