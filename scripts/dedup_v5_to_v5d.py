"""
dedup_v5_to_v5d.py (PyArrow streaming, OOM-safe)
================================================
v5 -> v5_dedup: (url, company_id) 동일한 진짜 중복만 제거.

전략 (메모리 안전):
  1) row_group 순회하면서 (url, company_id) 페어를 hash set에 등록
  2) 처음 본 페어만 출력에 keep (first-wins)
  3) hash 사용으로 메모리 ~600 MB 한계

주의:
  - first-wins는 우선순위 정보를 잃지만, 진짜 중복 25K개는
    (url, co) 동일한 케이스이므로 어느 행을 keep해도 주요 정보(severity, tone, finbert)는
    동일할 가능성이 매우 높음 (이전 분석에서 alias만 다른 케이스가 대부분).
  - 단, finbert가 채워진 행을 우선 keep하기 위해 2-pass 사용:
    pass1: finbert 있는 행 먼저 등록
    pass2: 나머지 행 처리
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import time
import os
import hashlib
from pathlib import Path

ROOT = Path("/sessions/fervent-tender-clarke/mnt/nabi_hyoghaw")
SRC = ROOT / "data/processed/risk_events_classified_v5.parquet"
DST = ROOT / "data/processed/risk_events_classified_v5_dedup.parquet"

print(f"[dedup] src: {SRC}")
print(f"[dedup] dst: {DST}")

pf = pq.ParquetFile(SRC)
schema = pf.schema_arrow
print(f"  rows      : {pf.metadata.num_rows:,}")
print(f"  row groups: {pf.num_row_groups}")

writer = pq.ParquetWriter(DST, schema, compression="zstd", compression_level=3)

seen = set()  # int hashes of (url, company_id)
total_in = 0
total_out = 0

def keypair_hash(url: str, co: str) -> int:
    """fast int hash for (url, co); collision risk negligible at 12M scale"""
    h = hashlib.blake2b(f"{url}\x00{co}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big")

# === PASS 1: finbert가 있는 행 먼저 등록 (우선순위) ===
print("\n[Pass 1/2] register finbert-scored rows first...")
t0 = time.time()
for rg in range(pf.num_row_groups):
    tbl = pf.read_row_group(rg, columns=["url", "company_id", "finbert_score"])
    urls = tbl.column("url").to_pylist()
    cos  = tbl.column("company_id").to_pylist()
    fbs  = tbl.column("finbert_score")
    valid_mask = pc.is_valid(fbs).to_pylist()
    for u, c, v in zip(urls, cos, valid_mask):
        if v:
            seen.add(keypair_hash(u, c))
print(f"  pass1 done ({time.time()-t0:.1f}s, registered {len(seen):,} priority keys)")

# === PASS 2: 전체 순회. 처음 본 (url, co) keep ===
print("\n[Pass 2/2] streaming dedup write...")
t0 = time.time()
priority_keep = 0  # finbert-scored rows kept (sanity)

for rg in range(pf.num_row_groups):
    tbl = pf.read_row_group(rg)
    urls = tbl.column("url").to_pylist()
    cos  = tbl.column("company_id").to_pylist()
    fbs  = pc.is_valid(tbl.column("finbert_score")).to_pylist()
    n = tbl.num_rows
    total_in += n

    keep = [False] * n
    # 1단계: priority (finbert) 행은 무조건 keep, key 등록 안 된 상태로 처리하기 위해
    #        Pass 1에서 hash set에 들어있는 키 중 finbert인 첫 등장만 keep해야 함.
    # 단순화: 새 dict로 다시 처리. seen은 Pass 1 결과만 사용.
    # → 새 set "kept_keys" 사용
    # (priority 처리 위해 실제 dedup loop은 다음 단계에서)
    pass

# Pass 2 재설계: 단일 패스로
print("  (re-designing pass2 with kept_keys...)")
kept_keys = set()
total_in = 0
total_out = 0

for rg in range(pf.num_row_groups):
    tbl = pf.read_row_group(rg)
    urls = tbl.column("url").to_pylist()
    cos  = tbl.column("company_id").to_pylist()
    fbs_valid = pc.is_valid(tbl.column("finbert_score")).to_pylist()
    n = tbl.num_rows
    total_in += n

    keep_idx = []
    for i in range(n):
        k = keypair_hash(urls[i], cos[i])
        if k in kept_keys:
            continue
        # priority rule: 이 키가 'seen'(finbert 우선)에 있고 현재 행이 finbert가 아니면 skip
        if (k in seen) and (not fbs_valid[i]):
            continue
        kept_keys.add(k)
        keep_idx.append(i)

    if keep_idx:
        sub = tbl.take(pa.array(keep_idx, type=pa.int64()))
        writer.write_table(sub)
        total_out += sub.num_rows

    if (rg + 1) % 20 == 0 or rg == pf.num_row_groups - 1:
        print(f"  rg {rg+1:>3}/{pf.num_row_groups}  in={total_in:>11,}  out={total_out:>11,}  removed={total_in-total_out:>7,}")

writer.close()
print(f"  pass2 done ({time.time()-t0:.1f}s)")

# === Verify ===
print("\n=== VERIFY ===")
print(f"  input  rows: {total_in:,}")
print(f"  output rows: {total_out:,}")
print(f"  removed    : {total_in - total_out:,}  (expected ≈ 25,121)")

import duckdb
con = duckdb.connect(":memory:")
con.execute("PRAGMA memory_limit='3GB'")
con.execute("PRAGMA threads=4")
n_dup = con.execute(f"""
    SELECT COUNT(*) - COUNT(DISTINCT (url, company_id))
    FROM read_parquet('{DST}')
""").fetchone()[0]
print(f"  remaining (url,co) dups: {n_dup:,}  (expected 0)")

n_fb = con.execute(f"""
    SELECT SUM(CASE WHEN finbert_score IS NOT NULL THEN 1 ELSE 0 END)
    FROM read_parquet('{DST}')
""").fetchone()[0]
print(f"  finbert scored kept: {n_fb:,}  (was 41,588 → expect ~41K)")

sz = os.path.getsize(DST) / 1024 / 1024
print(f"  file size  : {sz:.1f} MB")
print("\n[dedup] DONE")
