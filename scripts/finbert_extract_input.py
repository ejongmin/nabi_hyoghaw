"""
finbert_extract_input.py
========================
v4에서 실제 title이 있는 행만 추출하여 Colab 입력 파일 생성.

입력: risk_events_classified_v4.parquet (11.8M rows)
출력: finbert_input.parquet  (~41.6K rows, event_id + title)

Colab 업로드용이라 최대한 작게 (event_id, title 2컬럼만).
"""

import duckdb
from pathlib import Path

ROOT = Path("/sessions/fervent-tender-clarke/mnt/nabi_hyoghaw")
SRC = ROOT / "data/processed/risk_events_classified_v4.parquet"
DST = ROOT / "data/processed/finbert_input.parquet"

con = duckdb.connect(":memory:")
con.execute("PRAGMA threads=6")
con.execute("PRAGMA memory_limit='8GB'")

con.execute(f"""
    COPY (
        SELECT
            event_id,
            title
        FROM read_parquet('{SRC}')
        WHERE title IS NOT NULL
          AND LENGTH(TRIM(title)) >= 10     -- 10자 미만은 의미 없는 쓰레기
    )
    TO '{DST}' (FORMAT PARQUET, COMPRESSION ZSTD)
""")

row = con.execute(f"""
    SELECT
        COUNT(*),
        AVG(LENGTH(title)),
        MIN(LENGTH(title)),
        MAX(LENGTH(title))
    FROM read_parquet('{DST}')
""").fetchone()

size_kb = DST.stat().st_size / 1024
print(f"[finbert_extract_input] DONE")
print(f"  output      : {DST}")
print(f"  rows        : {row[0]:,}")
print(f"  title length: min={row[2]}, avg={row[1]:.0f}, max={row[3]}")
print(f"  file size   : {size_kb:.1f} KB")
