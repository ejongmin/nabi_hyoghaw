"""
finbert_merge_to_v5.py (streaming, OOM-safe)
============================================
Colab에서 생성된 finbert_results.parquet을 v4에 병합하여 v5 생성.

동작:
  1) v4의 기존 `finbert` 컬럼은 **모두 NULL로 리셋** (빈 title 기반 가짜 점수 제거)
  2) finbert_results의 41K행만 event_id로 조인하여 `finbert_score` 기록
  3) 추가로 `finbert_neg`, `finbert_neu`, `finbert_pos`, `finbert_score` 컬럼 신설
     - 기존 단일 `finbert` 컬럼은 제거 (이름 혼동 방지)

입력 :
  - data/processed/risk_events_classified_v4.parquet
  - data/processed/finbert_results.parquet   (Colab 결과)
출력 :
  - data/processed/risk_events_classified_v5.parquet
"""

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pandas as pd
import time
from pathlib import Path

ROOT = Path("/sessions/fervent-tender-clarke/mnt/nabi_hyoghaw")
V4    = ROOT / "data/processed/risk_events_classified_v4.parquet"
FB    = ROOT / "data/processed/finbert_results.parquet"
V5    = ROOT / "data/processed/risk_events_classified_v5.parquet"

assert V4.exists(), f"V4 not found: {V4}"
assert FB.exists(), f"FinBERT results not found: {FB}  (Colab 실행 후 Drive에서 가져왔는지 확인)"

print(f"[merge_v5] v4      : {V4}")
print(f"[merge_v5] finbert : {FB}")
print(f"[merge_v5] v5      : {V5}")

# --- 1. FinBERT 결과를 dict로 로드 (41K rows, 메모리 OK) ---
fb_df = pd.read_parquet(FB)
print(f"\n  finbert rows : {len(fb_df):,}")
print(f"  columns      : {fb_df.columns.tolist()}")
print(f"  score range  : [{fb_df['finbert_score'].min():.3f}, {fb_df['finbert_score'].max():.3f}]  mean={fb_df['finbert_score'].mean():.3f}")

# event_id -> (neg, neu, pos, score) dict
fb_map = {
    r.event_id: (r.finbert_neg, r.finbert_neu, r.finbert_pos, r.finbert_score)
    for r in fb_df.itertuples(index=False)
}
print(f"  map size     : {len(fb_map):,}")

# --- 2. v4 스키마 준비: 기존 finbert 컬럼 드랍, 신규 4컬럼 추가 ---
pf = pq.ParquetFile(V4)
src_fields = pf.schema_arrow
drop_names = {"finbert"}
base_names = [f for f in src_fields.names if f not in drop_names]

new_fields = [
    pa.field("finbert_neg",   pa.float32()),
    pa.field("finbert_neu",   pa.float32()),
    pa.field("finbert_pos",   pa.float32()),
    pa.field("finbert_score", pa.float32()),
]
out_schema = pa.schema([src_fields.field(n) for n in base_names] + new_fields)

print(f"\n  input cols   : {len(src_fields.names)}  (drop: {sorted(drop_names)})")
print(f"  output cols  : {len(out_schema.names)}  (add: finbert_neg/neu/pos/score)")
print(f"  row groups   : {pf.num_row_groups}")

writer = pq.ParquetWriter(V5, out_schema, compression="zstd", compression_level=3)

t0 = time.time()
total = 0
matched = 0

import numpy as np

for rg in range(pf.num_row_groups):
    tbl = pf.read_row_group(rg, columns=base_names)
    n = tbl.num_rows
    total += n

    eids = tbl.column("event_id").to_pylist()
    neg  = np.zeros(n, dtype=np.float32)
    neu  = np.zeros(n, dtype=np.float32)
    pos  = np.zeros(n, dtype=np.float32)
    score= np.zeros(n, dtype=np.float32)
    mask = np.ones(n, dtype=bool)  # True = NULL (not matched)

    for i, eid in enumerate(eids):
        v = fb_map.get(eid)
        if v is not None:
            neg[i], neu[i], pos[i], score[i] = v
            mask[i] = False
            matched += 1

    # mask=True means NULL in pyarrow
    tbl2 = tbl.append_column("finbert_neg",   pa.array(neg,   mask=mask, type=pa.float32()))
    tbl2 = tbl2.append_column("finbert_neu",  pa.array(neu,   mask=mask, type=pa.float32()))
    tbl2 = tbl2.append_column("finbert_pos",  pa.array(pos,   mask=mask, type=pa.float32()))
    tbl2 = tbl2.append_column("finbert_score",pa.array(score, mask=mask, type=pa.float32()))

    writer.write_table(tbl2)

    if (rg + 1) % 20 == 0 or rg == pf.num_row_groups - 1:
        print(f"  rg {rg+1:>3}/{pf.num_row_groups}  total={total:>11,}  matched={matched:>7,}")

writer.close()
print(f"\n  duration: {time.time()-t0:.1f}s")

# --- 3. 검증 ---
print("\n=== VERIFY ===")
import duckdb
con = duckdb.connect(":memory:")
con.execute("PRAGMA memory_limit='6GB'")
con.execute("PRAGMA threads=4")
row = con.execute(f"""
    SELECT
        COUNT(*)                                       AS total,
        SUM(CASE WHEN finbert_score IS NOT NULL THEN 1 ELSE 0 END) AS has_score,
        AVG(finbert_score)                             AS mean,
        STDDEV(finbert_score)                          AS std,
        MIN(finbert_score)                             AS mn,
        MAX(finbert_score)                             AS mx
    FROM read_parquet('{V5}')
""").fetchone()
print(f"  total rows     : {row[0]:,}")
print(f"  has finbert    : {row[1]:,}  ({row[1]/row[0]*100:.3f}%)")
print(f"  finbert mean   : {row[2]:.4f}")
print(f"  finbert std    : {row[3]:.4f}")
print(f"  finbert range  : [{row[4]:.3f}, {row[5]:.3f}]")
print(f"  matched (loop) : {matched:,}  (should == {len(fb_map):,})")

schema = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{V5}')").fetchall()
print(f"\n  schema ({len(schema)} cols):")
for r in schema:
    print(f"    - {r[0]:<20} {r[1]}")

sz = V5.stat().st_size / 1024 / 1024
print(f"\n  file size: {sz:.1f} MB")

print("\n[merge_v5] DONE")
