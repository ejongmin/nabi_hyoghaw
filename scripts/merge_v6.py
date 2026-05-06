"""
merge_v6.py (streaming, OOM-safe)
=================================
v5_dedup + finbert_results_zh + finbert_results_ko -> v6

기본 전제:
  - v5_dedup에는 이미 영어 FinBERT 점수가 28,997행에 보존되어 있음 (ProsusAI/finbert)
  - zh 결과: yiyanghkust/finbert-tone-chinese
  - ko 결과: snunlp/KR-FinBert-SC

전략:
  - 영어 점수는 유지 (ProsusAI/finbert 원본)
  - zh 결과는 title 기준으로 JOIN하여 업데이트 (같은 title의 모든 행에 동일 score 적용)
  - ko 결과는 title 기준으로 JOIN하여 업데이트
  - finbert_model 컬럼 추가: 'en', 'zh', 'ko', NULL (어느 모델로 점수화됐는지 추적)

PyArrow streaming + title→score dict 방식 (메모리 안전).
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pandas as pd
import numpy as np
import time
import os
from pathlib import Path

ROOT = Path("/sessions/fervent-tender-clarke/mnt/nabi_hyoghaw")
V5D  = ROOT / "data/processed/risk_events_classified_v5_dedup.parquet"
ZH   = ROOT / "data/processed/finbert_results_zh.parquet"
KO   = ROOT / "data/processed/finbert_results_ko.parquet"
V6   = ROOT / "data/processed/risk_events_classified_v6.parquet"

assert V5D.exists(), f"missing {V5D}"
assert ZH.exists(),  f"missing {ZH}  (Colab zh 실행 결과 필요)"
assert KO.exists(),  f"missing {KO}  (Colab ko 실행 결과 필요)"

print(f"[merge_v6] v5_dedup: {V5D}")
print(f"[merge_v6] zh     : {ZH}")
print(f"[merge_v6] ko     : {KO}")
print(f"[merge_v6] v6     : {V6}")

# --- 1. zh/ko 결과 → title → (neg, neu, pos, score) dict ---
def load_title_map(path):
    df = pd.read_parquet(path)
    print(f"  {path.name}: {len(df):,} rows, cols={list(df.columns)}")
    d = {}
    for r in df.itertuples(index=False):
        d[r.title] = (float(r.finbert_neg), float(r.finbert_neu), float(r.finbert_pos), float(r.finbert_score))
    return d

print("\n[1] loading zh/ko maps...")
zh_map = load_title_map(ZH)
ko_map = load_title_map(KO)
print(f"  zh titles: {len(zh_map):,}")
print(f"  ko titles: {len(ko_map):,}")

# --- 2. v5_dedup 스트리밍 처리 ---
pf = pq.ParquetFile(V5D)
src_schema = pf.schema_arrow
print(f"\n[2] streaming merge from v5_dedup ({pf.metadata.num_rows:,} rows, {pf.num_row_groups} groups)")

# 신규 컬럼 추가 (finbert_model)
out_fields = list(src_schema) + [pa.field("finbert_model", pa.string())]
out_schema = pa.schema(out_fields)

writer = pq.ParquetWriter(V6, out_schema, compression="zstd", compression_level=3)

t0 = time.time()
n_total = 0
n_en_kept = 0
n_zh_updated = 0
n_ko_updated = 0

for rg in range(pf.num_row_groups):
    tbl = pf.read_row_group(rg)
    n = tbl.num_rows
    n_total += n

    titles = tbl.column("title").to_pylist()
    old_neg = tbl.column("finbert_neg").to_pylist()
    old_neu = tbl.column("finbert_neu").to_pylist()
    old_pos = tbl.column("finbert_pos").to_pylist()
    old_sc  = tbl.column("finbert_score").to_pylist()

    new_neg = [None] * n
    new_neu = [None] * n
    new_pos = [None] * n
    new_sc  = [None] * n
    new_mdl = [None] * n

    for i in range(n):
        t = titles[i]
        # 우선순위: ko map > zh map > 기존 en (latin title만 의미 있음)
        # 이유: v5의 기존 finbert 컬럼에는 ProsusAI를 모든 title에 돌린 '잘못된 zh/ko 점수'도
        #       포함되어 있다. ko/zh map에 hit하면 무조건 새 전문 모델 점수로 덮어쓴다.
        matched = False
        if t is not None:
            v = ko_map.get(t)
            if v is not None:
                new_neg[i], new_neu[i], new_pos[i], new_sc[i] = v
                new_mdl[i] = "ko"
                n_ko_updated += 1
                matched = True
            else:
                v = zh_map.get(t)
                if v is not None:
                    new_neg[i], new_neu[i], new_pos[i], new_sc[i] = v
                    new_mdl[i] = "zh"
                    n_zh_updated += 1
                    matched = True
        if not matched and old_sc[i] is not None:
            # 기존 en 점수 유지 (단, latin title인 경우만 실제로 유효)
            new_neg[i] = old_neg[i]
            new_neu[i] = old_neu[i]
            new_pos[i] = old_pos[i]
            new_sc[i]  = old_sc[i]
            new_mdl[i] = "en"
            n_en_kept += 1

    # 원본 컬럼 교체 (drop + append)
    keep_cols = [c for c in tbl.column_names if c not in {"finbert_neg","finbert_neu","finbert_pos","finbert_score"}]
    tbl2 = tbl.select(keep_cols)
    tbl2 = tbl2.append_column("finbert_neg",   pa.array(new_neg,   type=pa.float32()))
    tbl2 = tbl2.append_column("finbert_neu",   pa.array(new_neu,   type=pa.float32()))
    tbl2 = tbl2.append_column("finbert_pos",   pa.array(new_pos,   type=pa.float32()))
    tbl2 = tbl2.append_column("finbert_score", pa.array(new_sc,    type=pa.float32()))
    tbl2 = tbl2.append_column("finbert_model", pa.array(new_mdl,   type=pa.string()))

    # 스키마 순서 맞추기 (out_schema 순서대로)
    tbl2 = tbl2.select(out_schema.names)

    writer.write_table(tbl2)

    if (rg + 1) % 20 == 0 or rg == pf.num_row_groups - 1:
        print(f"  rg {rg+1:>3}/{pf.num_row_groups}  total={n_total:>11,}  en={n_en_kept:>6,} zh={n_zh_updated:>6,} ko={n_ko_updated:>6,}")

writer.close()
print(f"\n  duration: {time.time()-t0:.1f}s")

# --- 3. 검증 ---
print("\n=== VERIFY ===")
import duckdb
con = duckdb.connect(":memory:")
con.execute("PRAGMA memory_limit='3GB'")
con.execute("PRAGMA threads=4")

row = con.execute(f"""
    SELECT
        COUNT(*) AS total,
        SUM(CASE WHEN finbert_score IS NOT NULL THEN 1 ELSE 0 END) AS has_score,
        SUM(CASE WHEN finbert_model='en' THEN 1 ELSE 0 END) AS en,
        SUM(CASE WHEN finbert_model='zh' THEN 1 ELSE 0 END) AS zh,
        SUM(CASE WHEN finbert_model='ko' THEN 1 ELSE 0 END) AS ko
    FROM read_parquet('{V6}')
""").fetchone()
print(f"  total   : {row[0]:,}")
print(f"  scored  : {row[1]:,}  ({row[1]/row[0]*100:.3f}%)")
print(f"  en rows : {row[2]:,}")
print(f"  zh rows : {row[3]:,}")
print(f"  ko rows : {row[4]:,}")

# 언어별 신호 강도
print("\n  [per-model signal strength]")
rows = con.execute(f"""
    SELECT
        finbert_model,
        COUNT(*) AS n,
        ROUND(AVG(CAST(finbert_score AS DOUBLE)), 4) AS mean,
        ROUND(STDDEV(CAST(finbert_score AS DOUBLE)), 4) AS std,
        SUM(CASE WHEN ABS(finbert_score) > 0.3 THEN 1 ELSE 0 END) AS strong,
        ROUND(100.0*SUM(CASE WHEN ABS(finbert_score) > 0.3 THEN 1 ELSE 0 END)/COUNT(*), 2) AS pct
    FROM read_parquet('{V6}')
    WHERE finbert_score IS NOT NULL
    GROUP BY 1
    ORDER BY 1
""").fetchall()
print(f"  {'model':<6} {'n':>8} {'mean':>9} {'std':>9} {'strong':>8} {'%strong':>9}")
for r in rows:
    print(f"  {r[0]:<6} {r[1]:>8,} {r[2]:>+9.4f} {r[3]:>9.4f} {r[4]:>8,} {r[5]:>8}%")

schema = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{V6}')").fetchall()
print(f"\n  schema ({len(schema)} cols):")
for r in schema:
    print(f"    - {r[0]:<20} {r[1]}")

sz = os.path.getsize(V6) / 1024 / 1024
print(f"\n  file size: {sz:.1f} MB")
print("\n[merge_v6] DONE")
