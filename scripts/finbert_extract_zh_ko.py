"""
finbert_extract_zh_ko.py
========================
v5_dedup에서 언어별로 finbert input 추출.

출력 (DISTINCT title 단위로 dedup — 같은 title은 1번만 점수화):
  - finbert_input_zh.parquet  (Chinese CJK)
  - finbert_input_ko.parquet  (Korean Hangul)

영어/Latin은 이미 v5_dedup에 finbert_score가 보존되어 있으므로 재실행 불필요.
일본어는 샘플에 거의 없어서 생략 (필요시 나중에 추가).

탐지 기준:
  - Chinese: [\u4e00-\u9fff] (CJK Unified Ideographs) 포함
  - Korean : [\uac00-\ud7af] (Hangul Syllables) 포함
  - 둘 다 있으면 Korean 우선 (한국 뉴스의 한자 혼용 케이스)
"""

import duckdb
import os
from pathlib import Path

ROOT = Path("/sessions/fervent-tender-clarke/mnt/nabi_hyoghaw")
SRC = ROOT / "data/processed/risk_events_classified_v5_dedup.parquet"
OUT_ZH = ROOT / "data/processed/finbert_input_zh.parquet"
OUT_KO = ROOT / "data/processed/finbert_input_ko.parquet"

con = duckdb.connect(":memory:")
con.execute("PRAGMA threads=4")
con.execute("PRAGMA memory_limit='4GB'")

# 언어별 추출. DISTINCT title로 중복 제거 (같은 헤드라인은 한 번만 스코어링)
# event_id는 나중 merge 시 title-기반으로 조인할 예정이므로 title만 들고감 + rep_event_id
con.execute(f"""
    COPY (
        SELECT DISTINCT ON (title) title, event_id AS rep_event_id
        FROM read_parquet('{SRC}')
        WHERE title IS NOT NULL
          AND LENGTH(TRIM(title)) >= 10
          AND regexp_matches(title, '[\uac00-\ud7af]')   -- Korean Hangul
    )
    TO '{OUT_KO}' (FORMAT PARQUET, COMPRESSION ZSTD)
""")

con.execute(f"""
    COPY (
        SELECT DISTINCT ON (title) title, event_id AS rep_event_id
        FROM read_parquet('{SRC}')
        WHERE title IS NOT NULL
          AND LENGTH(TRIM(title)) >= 10
          AND regexp_matches(title, '[\u4e00-\u9fff]')   -- CJK
          AND NOT regexp_matches(title, '[\uac00-\ud7af]') -- 한글 혼재 제외
    )
    TO '{OUT_ZH}' (FORMAT PARQUET, COMPRESSION ZSTD)
""")

for name, path in [("zh", OUT_ZH), ("ko", OUT_KO)]:
    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{path}')").fetchone()[0]
    sz = os.path.getsize(path) / 1024
    print(f"  {name}: {n:>7,} unique titles  ({sz:.1f} KB)  -> {path.name}")

# 참고: v5_dedup의 전체 언어별 title 행수
print()
print("=== Reference: v5_dedup language distribution (rows with title) ===")
rows = con.execute(f"""
    SELECT
        CASE
            WHEN regexp_matches(title, '[\uac00-\ud7af]') THEN 'ko'
            WHEN regexp_matches(title, '[\u4e00-\u9fff]') THEN 'zh'
            WHEN regexp_matches(title, '[\u3040-\u309f\u30a0-\u30ff]') THEN 'ja'
            ELSE 'en/other'
        END AS lang,
        COUNT(*) AS rows,
        COUNT(DISTINCT title) AS unique_titles
    FROM read_parquet('{SRC}')
    WHERE title IS NOT NULL AND LENGTH(TRIM(title)) >= 10
    GROUP BY 1 ORDER BY rows DESC
""").fetchall()
for r in rows:
    print(f"  {r[0]:<10} rows={r[1]:>8,}  unique titles={r[2]:>7,}")
