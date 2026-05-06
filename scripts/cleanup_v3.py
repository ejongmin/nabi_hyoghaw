#!/usr/bin/env python3
"""
나비효과 — V3 파일 Cleanup Pipeline
====================================
수행 작업:
  1. Severity 스케일 통일 (legacy/v2themes_aho/title_keyword: [1,5] → log 정규화 [0,1])
  2. 2018-2021 Tone 복구 (raw gdelt 파일에서 URL 기준 JOIN)
  3. country_ids 포맷 통일 (JSON array 문자열)

방법론 (severity):
  - gdelt_themes: 이미 log(1+raw)/log(1+30) 적용됨 [0,1] → 유지
  - 나머지 (raw [1,5]): log(1+sev)/log(1+5) 적용 → [0.387, 1.0]
  - 이유: 동일한 로그 정규화 방식으로 통계 분포 일관성 확보
  - 결과: 모든 행의 severity가 [0, 1] 범위로 통일

방법론 (tone):
  - 2018-2021 V3 legacy 행 (tone=0)에 대해서만 raw 파일 tone 값으로 업데이트
  - 2022+ 데이터는 기존 tone 유지
  - URL이 매칭 안 되는 행은 tone=NULL로 명시 (0 아님)

방법론 (country_ids):
  - 입력 4가지 포맷 → JSON array 문자열로 통일
    * '' or '[]' → '[]'
    * 'USA;China' → '["USA","China"]'
    * "['USA']" → '["USA"]'
    * "['China' 'USA']" → '["China","USA"]' (공백 구분자)
"""

import time
import math
import gc
from pathlib import Path
import duckdb

INPUT_V3 = '/sessions/fervent-tender-clarke/mnt/uploads/risk_events_classified_v3.parquet'
OUTPUT_V3 = '/sessions/fervent-tender-clarke/mnt/nabi_hyoghaw/data/processed/risk_events_classified_v3_cleaned.parquet'
TONE_LOOKUP = '/sessions/fervent-tender-clarke/mnt/nabi_hyoghaw/data/processed/_tone_lookup_2018_2021.parquet'

RAW_FILES = [
    '/sessions/fervent-tender-clarke/mnt/nabi_hyoghaw/data/raw/gdelt/gdelt_2018.parquet',
    '/sessions/fervent-tender-clarke/mnt/nabi_hyoghaw/data/raw/gdelt/gdelt_2018_2019.parquet',
    '/sessions/fervent-tender-clarke/mnt/nabi_hyoghaw/data/raw/gdelt/gdelt_gkg_2019~2020.parquet',
    '/sessions/fervent-tender-clarke/mnt/nabi_hyoghaw/data/raw/gdelt/gdelt_gkg_2021~2022.parquet',
]


def build_tone_lookup(con):
    """Raw gdelt 파일에서 url→tone lookup 생성 (중복 제거)"""
    print("\n[1/4] Tone lookup 생성 (raw 2018-2021 files)")
    t0 = time.time()

    files_str = ', '.join([f"'{f}'" for f in RAW_FILES])
    con.execute(f"""
    COPY (
        SELECT url, any_value(tone) AS tone
        FROM (
            SELECT article_url AS url, tone
            FROM read_parquet([{files_str}], union_by_name=true)
            WHERE article_url IS NOT NULL AND tone IS NOT NULL AND tone != 0
        )
        GROUP BY url
    ) TO '{TONE_LOOKUP}' (FORMAT PARQUET, COMPRESSION SNAPPY)
    """)

    n = con.execute(f"SELECT count(*) FROM read_parquet('{TONE_LOOKUP}')").fetchone()[0]
    sz = Path(TONE_LOOKUP).stat().st_size / 1024**2
    print(f"  ✓ {n:,} unique URLs, {sz:.0f} MB [{time.time()-t0:.0f}s]")
    return n


def run_cleanup(con):
    """V3 파일에 3가지 fix 적용하여 출력"""
    print("\n[2/4] Cleanup SQL 실행 (severity + tone + country_ids)")
    t0 = time.time()

    # Severity 정규화 공식 상수
    LOG_DENOM_5 = math.log(1 + 5.0)  # for legacy raw [1,5] scale

    # 완전한 cleanup SQL
    cleanup_sql = f"""
    COPY (
        SELECT
            e.event_id,
            e.event_time,
            e.url,
            e.title,
            e.risk_types,
            -- ═══ Severity 재정규화 ═══
            -- gdelt_themes: 유지 (이미 [0,1])
            -- 나머지: log(1+sev)/log(1+5) 적용
            CASE
                WHEN e.classify_source = 'gdelt_themes' THEN e.severity
                WHEN e.severity IS NULL THEN NULL
                WHEN e.severity <= 0 THEN 0.0
                ELSE ROUND(LEAST(LN(1 + e.severity) / {LOG_DENOM_5}, 1.0), 4)
            END AS severity,
            e.element,
            e.alias,
            e.company_id,
            e.score,
            -- FinBERT는 일단 그대로 (이번 cleanup 대상 아님)
            e.finbert,
            -- ═══ country_ids 포맷 통일 (JSON array 문자열) ═══
            CASE
                -- 빈 값
                WHEN e.country_ids IS NULL OR e.country_ids = '' OR e.country_ids = '[]'
                    THEN '[]'
                -- List 포맷 "['USA' 'China']" (공백 구분)
                WHEN e.country_ids LIKE '[%' THEN
                    '[' || list_aggregate(
                        list_transform(
                            regexp_extract_all(e.country_ids, '''([^'']+)''', 1),
                            x -> '"' || x || '"'
                        ),
                        'string_agg', ','
                    ) || ']'
                -- Plain 포맷 "USA;China"
                ELSE
                    '[' || list_aggregate(
                        list_transform(
                            list_filter(string_split(e.country_ids, ';'), x -> TRIM(x) != ''),
                            x -> '"' || TRIM(x) || '"'
                        ),
                        'string_agg', ','
                    ) || ']'
            END AS country_ids,
            -- ═══ Tone 복구 ═══
            -- 2018-2021이고 기존 tone=0이면 lookup 값으로 교체
            -- lookup 없으면 NULL (0과 구분)
            CASE
                WHEN EXTRACT(YEAR FROM e.event_time) BETWEEN 2018 AND 2021
                     AND (e.tone IS NULL OR e.tone = 0)
                THEN tl.tone  -- NULL이면 그대로 NULL (0 대신 명시적 missing)
                ELSE e.tone
            END AS tone,
            e.source_domain,
            e.collection_type,
            e.classify_source,
            e.classify_conf
        FROM read_parquet('{INPUT_V3}') e
        LEFT JOIN read_parquet('{TONE_LOOKUP}') tl ON e.url = tl.url
    ) TO '{OUTPUT_V3}' (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 100000)
    """

    con.execute(cleanup_sql)

    sz = Path(OUTPUT_V3).stat().st_size / 1024**2
    print(f"  ✓ {sz:.0f} MB [{time.time()-t0:.0f}s]")


def verify(con):
    """Before/After 통계 비교"""
    print("\n[3/4] 검증: Before vs After")
    print("=" * 70)

    # Severity
    print("\n▶ Severity")
    r = con.execute(f"""
    SELECT
      'BEFORE' AS tag,
      round(min(severity),3) mn, round(max(severity),3) mx,
      round(avg(severity),3) mean, round(stddev(severity),3) std,
      sum(CASE WHEN severity > 1.0 THEN 1 ELSE 0 END) over_1
    FROM read_parquet('{INPUT_V3}')
    UNION ALL
    SELECT
      'AFTER' AS tag,
      round(min(severity),3), round(max(severity),3),
      round(avg(severity),3), round(stddev(severity),3),
      sum(CASE WHEN severity > 1.0 THEN 1 ELSE 0 END)
    FROM read_parquet('{OUTPUT_V3}')
    """).fetchdf()
    print(r.to_string(index=False))

    print("\n  After: classify_source별 severity 분포")
    r = con.execute(f"""
    SELECT classify_source,
      round(min(severity),3) mn, round(max(severity),3) mx,
      round(avg(severity),3) mean, round(stddev(severity),3) std
    FROM read_parquet('{OUTPUT_V3}')
    GROUP BY 1 ORDER BY 1
    """).fetchdf()
    print(r.to_string(index=False))

    # Tone
    print("\n▶ Tone (2018-2021 초점)")
    r = con.execute(f"""
    SELECT
      'BEFORE' AS tag,
      sum(CASE WHEN tone IS NULL THEN 1 ELSE 0 END) null_n,
      sum(CASE WHEN tone = 0 THEN 1 ELSE 0 END) zero_n,
      round(avg(CASE WHEN tone != 0 THEN tone END),3) nonzero_mean,
      round(stddev(CASE WHEN tone != 0 THEN tone END),3) nonzero_std
    FROM read_parquet('{INPUT_V3}')
    WHERE EXTRACT(YEAR FROM event_time) BETWEEN 2018 AND 2021
    UNION ALL
    SELECT
      'AFTER' AS tag,
      sum(CASE WHEN tone IS NULL THEN 1 ELSE 0 END),
      sum(CASE WHEN tone = 0 THEN 1 ELSE 0 END),
      round(avg(CASE WHEN tone != 0 THEN tone END),3),
      round(stddev(CASE WHEN tone != 0 THEN tone END),3)
    FROM read_parquet('{OUTPUT_V3}')
    WHERE EXTRACT(YEAR FROM event_time) BETWEEN 2018 AND 2021
    """).fetchdf()
    print(r.to_string(index=False))

    # country_ids
    print("\n▶ country_ids 포맷")
    r = con.execute(f"""
    SELECT
      'BEFORE' AS tag,
      sum(CASE WHEN country_ids = '' THEN 1 ELSE 0 END) empty_str,
      sum(CASE WHEN country_ids = '[]' THEN 1 ELSE 0 END) empty_list,
      sum(CASE WHEN country_ids LIKE '[%' AND country_ids != '[]' THEN 1 ELSE 0 END) list_fmt,
      sum(CASE WHEN country_ids NOT LIKE '[%' AND country_ids != '' THEN 1 ELSE 0 END) plain_fmt
    FROM read_parquet('{INPUT_V3}')
    UNION ALL
    SELECT
      'AFTER' AS tag,
      sum(CASE WHEN country_ids = '' THEN 1 ELSE 0 END),
      sum(CASE WHEN country_ids = '[]' THEN 1 ELSE 0 END),
      sum(CASE WHEN country_ids LIKE '[%' AND country_ids != '[]' THEN 1 ELSE 0 END),
      sum(CASE WHEN country_ids NOT LIKE '[%' AND country_ids != '' THEN 1 ELSE 0 END)
    FROM read_parquet('{OUTPUT_V3}')
    """).fetchdf()
    print(r.to_string(index=False))

    print("\n  After: country_ids 샘플 TOP 10")
    r = con.execute(f"""
    SELECT country_ids, count(*) n FROM read_parquet('{OUTPUT_V3}')
    GROUP BY 1 ORDER BY 2 DESC LIMIT 10
    """).fetchdf()
    print(r.to_string(index=False))

    # Row count check
    before = con.execute(f"SELECT count(*) FROM read_parquet('{INPUT_V3}')").fetchone()[0]
    after = con.execute(f"SELECT count(*) FROM read_parquet('{OUTPUT_V3}')").fetchone()[0]
    print(f"\n▶ Row count: BEFORE={before:,} / AFTER={after:,} / 차이={after-before}")
    assert before == after, "Row count mismatch!"


def main():
    total_start = time.time()

    print("=" * 70)
    print("V3 Cleanup Pipeline — Severity + Tone + Country IDs")
    print("=" * 70)

    con = duckdb.connect('/sessions/fervent-tender-clarke/cleanup.duckdb')
    con.execute("SET memory_limit='4GB'")
    con.execute("SET temp_directory='/sessions/fervent-tender-clarke/duckdb_tmp'")
    con.execute("SET threads=4")

    # Step 1: Build tone lookup
    if not Path(TONE_LOOKUP).exists():
        build_tone_lookup(con)
    else:
        n = con.execute(f"SELECT count(*) FROM read_parquet('{TONE_LOOKUP}')").fetchone()[0]
        print(f"\n[1/4] ✓ Tone lookup 이미 존재: {n:,} URLs")

    # Step 2: Run cleanup
    run_cleanup(con)

    # Step 3: Verify
    verify(con)

    # Done
    elapsed = time.time() - total_start
    print(f"\n[4/4] ✓ Cleanup 완료: {elapsed/60:.1f}분")
    print(f"    출력: {OUTPUT_V3}")

    con.close()


if __name__ == '__main__':
    main()
