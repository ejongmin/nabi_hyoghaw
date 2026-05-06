"""
cleanup_v4.py (streaming, OOM-safe)
===================================
V3 cleaned -> V4 final cleanup via PyArrow row-group streaming.

변경 내역:
  5) event_time NULL (255행) DROP
  6) 2026 데이터 유지 (동적 KG용)
  7) element DROP (company_id와 100% 동일)
     score   DROP (collection_type='risk_only' 중복 인코딩)
     alias   유지 (provenance)
  FinBERT: 유지
"""

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import time
from pathlib import Path

ROOT = Path("/sessions/fervent-tender-clarke/mnt/nabi_hyoghaw")
INPUT  = ROOT / "data/processed/risk_events_classified_v3_cleaned.parquet"
OUTPUT = ROOT / "data/processed/risk_events_classified_v4.parquet"

DROP_COLS = {"element", "score"}

print(f"[cleanup_v4] input : {INPUT}")
print(f"[cleanup_v4] output: {OUTPUT}")

pf = pq.ParquetFile(INPUT)
src_schema = pf.schema_arrow
keep_names = [f for f in src_schema.names if f not in DROP_COLS]
out_schema = pa.schema([src_schema.field(n) for n in keep_names])

print(f"\n  input columns : {len(src_schema.names)}")
print(f"  output columns: {len(keep_names)}")
print(f"  dropped       : {sorted(DROP_COLS)}")
print(f"  row groups    : {pf.num_row_groups}")

writer = pq.ParquetWriter(
    OUTPUT,
    out_schema,
    compression="zstd",
    compression_level=3,
)

t0 = time.time()
total_in = 0
total_out = 0
null_dropped = 0

for rg_idx in range(pf.num_row_groups):
    tbl = pf.read_row_group(rg_idx, columns=keep_names)
    n_in = tbl.num_rows
    total_in += n_in

    # event_time NULL drop
    mask = pc.is_valid(tbl.column("event_time"))
    tbl2 = tbl.filter(mask)
    n_out = tbl2.num_rows
    null_dropped += (n_in - n_out)
    total_out += n_out

    writer.write_table(tbl2)

    if (rg_idx + 1) % 10 == 0 or rg_idx == pf.num_row_groups - 1:
        print(f"  rg {rg_idx+1:>3}/{pf.num_row_groups}  in={total_in:>11,}  out={total_out:>11,}  dropped={null_dropped}")

writer.close()
dt = time.time() - t0
print(f"\n  duration: {dt:.1f}s")

# --- 검증 ---
print("\n=== VERIFY ===")
pf2 = pq.ParquetFile(OUTPUT)
print(f"  total rows  : {pf2.metadata.num_rows:,}")
print(f"  row groups  : {pf2.num_row_groups}")
print(f"  dropped(time NULL): {null_dropped}")

# event_time range check via small scan
import duckdb
con = duckdb.connect(":memory:")
con.execute("PRAGMA threads=4")
con.execute("PRAGMA memory_limit='6GB'")
row = con.execute(f"""
    SELECT
        SUM(CASE WHEN event_time IS NULL THEN 1 ELSE 0 END) AS null_time,
        MIN(event_time) AS tmin,
        MAX(event_time) AS tmax
    FROM read_parquet('{OUTPUT}')
""").fetchone()
print(f"  event_time NULL after: {row[0]}")
print(f"  time range          : {row[1]} ~ {row[2]}")

print(f"\n  schema ({len(pf2.schema_arrow.names)} cols):")
for n in pf2.schema_arrow.names:
    print(f"    - {n}")

size_mb = OUTPUT.stat().st_size / 1024 / 1024
in_size_mb = INPUT.stat().st_size / 1024 / 1024
print(f"\n  file size: {size_mb:.1f} MB  (input {in_size_mb:.1f} MB, diff {size_mb - in_size_mb:+.1f} MB)")

print("\n[cleanup_v4] DONE")
