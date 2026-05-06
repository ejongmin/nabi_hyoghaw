#!/usr/bin/env python3
"""
merge_v6_to_v7.py
─────────────────
v6 (11.81M rows) + google_news_zero8.parquet → v7 (risk_events_classified_v7.parquet)

처리 단계:
  1. google_news_zero8.parquet 로드 + 스키마 정규화
  2. classify_risk_vectorized 로직 재사용하여 title 기반 risk_types + severity 계산
  3. v6 스키마 19개 컬럼으로 맞춤 (finbert 컬럼은 NULL, classify_source='google_news_zero8')
  4. v6과 concat, (url, company_id) 기준 dedup (zero8 우선 보존)
  5. PyArrow streaming으로 v7 저장 (OOM 방지)

Usage:
    python scripts/merge_v6_to_v7.py
    python scripts/merge_v6_to_v7.py --dry-run   # 통계만 확인
"""
import argparse
import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

log = logging.getLogger("merge_v7")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

ROOT = Path(__file__).resolve().parent.parent
V6 = ROOT / "data" / "processed" / "risk_events_classified_v6.parquet"
ZERO8 = ROOT / "data" / "raw" / "supplement" / "google_news_zero8.parquet"
V7 = ROOT / "data" / "processed" / "risk_events_classified_v7.parquet"
RISK_YAML = ROOT / "configs" / "risk_keywords.yaml"

V6_COLS = [
    "event_id", "event_time", "url", "title", "risk_types", "severity",
    "alias", "company_id", "country_ids", "tone", "source_domain",
    "collection_type", "classify_source", "classify_conf",
    "finbert_neg", "finbert_neu", "finbert_pos", "finbert_score", "finbert_model",
]


def build_mega_patterns(yaml_path: Path) -> dict:
    """risk_keywords.yaml → {rtype: (mega_regex, avg_weight)}"""
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    RTYPES = ["geopolitics", "logistics", "natural", "regulatory",
              "market", "labor", "technology", "esg", "pandemic"]
    out = {}
    for rtype in RTYPES:
        if rtype not in cfg:
            continue
        kws = cfg[rtype].get("keywords", [])
        patterns, total_w = [], 0
        for kw in kws:
            pat = kw.get("pattern", "")
            w = kw.get("weight", 1)
            if pat:
                patterns.append(pat)
                total_w += w
        if patterns:
            mega = "|".join(f"(?:{p})" for p in patterns)
            out[rtype] = {"mega": mega, "avg_w": total_w / len(patterns)}
            log.info(f"  {rtype:15s}: {len(patterns):3d} patterns")
    return out


def classify_titles(df: pd.DataFrame, risk_info: dict) -> pd.DataFrame:
    """title 벡터화 분류: risk_types + severity 채움."""
    titles = df["title"].fillna("").astype(str)
    n = len(titles)

    matches = {}
    for rtype, info in risk_info.items():
        t0 = time.time()
        hit = titles.str.contains(info["mega"], case=False, na=False, regex=True)
        matches[rtype] = hit
        log.info(f"  {rtype:15s}: {hit.sum():>6,} hits ({hit.sum()/n*100:.1f}%) in {time.time()-t0:.1f}s")

    match_df = pd.DataFrame(matches, index=df.index)
    severity_raw = pd.Series(0.0, index=df.index)
    for rtype, info in risk_info.items():
        severity_raw += match_df[rtype].astype(float) * info["avg_w"]
    S_MAX = 25.0
    severity = np.minimum(np.log(1 + severity_raw) / np.log(1 + S_MAX), 1.0).round(4)

    any_match = match_df.any(axis=1)
    def row_to_str(row):
        return ",".join([rt for rt in risk_info.keys() if row[rt]])
    risk_types = pd.Series("other", index=df.index)
    if any_match.any():
        mi = match_df.index[any_match]
        risk_types.loc[mi] = match_df.loc[mi].apply(row_to_str, axis=1)

    df["risk_types"] = risk_types.values
    df["severity"] = severity.values
    log.info(f"  matched: {any_match.sum():,} / {n:,} ({any_match.sum()/n*100:.1f}%)")
    return df


def normalize_zero8_schema(df: pd.DataFrame) -> pd.DataFrame:
    """zero8 DataFrame을 v6의 19컬럼 스키마로 정규화."""
    df = df.copy()

    # event_time: 타임존 제거 + NaT는 drop (FinBERT 단계에서 필터)
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    try:
        df["event_time"] = df["event_time"].dt.tz_localize(None)
    except Exception:
        pass
    before_nat = df["event_time"].isna().sum()
    df = df[df["event_time"].notna()].reset_index(drop=True)
    if before_nat > 0:
        log.info(f"  Dropped {before_nat:,} rows with invalid event_time")

    # 필수 컬럼 채우기
    df["classify_source"] = "google_news_zero8"
    df["classify_conf"] = 0.5
    df["finbert_neg"] = np.nan
    df["finbert_neu"] = np.nan
    df["finbert_pos"] = np.nan
    df["finbert_score"] = np.nan
    df["finbert_model"] = None

    # 누락된 필드 기본값
    for col, default in [
        ("country_ids", ""),
        ("tone", 0.0),
        ("source_domain", ""),
        ("alias", ""),
    ]:
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default)

    # collection_type 고정
    if "collection_type" not in df.columns or df["collection_type"].isna().all():
        df["collection_type"] = "google_news_zero8"

    # 컬럼 순서 맞추기
    missing = [c for c in V6_COLS if c not in df.columns]
    if missing:
        log.warning(f"Missing columns will be set to NULL: {missing}")
        for c in missing:
            df[c] = None

    return df[V6_COLS]


def to_arrow_with_nulls(df: pd.DataFrame) -> pa.Table:
    """pandas DataFrame → PyArrow Table with proper NULL handling for finbert columns."""
    arrays = []
    schema_fields = []
    # 타입 매핑 — v6 스키마와 동일하게
    type_map = {
        "event_id": pa.string(),
        "event_time": pa.timestamp("ns"),
        "url": pa.string(),
        "title": pa.string(),
        "risk_types": pa.string(),
        "severity": pa.float64(),
        "alias": pa.string(),
        "company_id": pa.string(),
        "country_ids": pa.string(),
        "tone": pa.float64(),
        "source_domain": pa.string(),
        "collection_type": pa.string(),
        "classify_source": pa.string(),
        "classify_conf": pa.float64(),
        "finbert_neg": pa.float32(),
        "finbert_neu": pa.float32(),
        "finbert_pos": pa.float32(),
        "finbert_score": pa.float32(),
        "finbert_model": pa.string(),
    }
    for col in V6_COLS:
        t = type_map[col]
        if col in ("finbert_neg", "finbert_neu", "finbert_pos", "finbert_score"):
            vals = df[col].to_numpy()
            mask = pd.isna(vals)  # True = NULL
            arr = pa.array(np.where(mask, 0.0, vals).astype(np.float32), mask=mask, type=t)
        elif col == "finbert_model":
            vals = df[col].tolist()
            arr = pa.array([None if (v is None or (isinstance(v, float) and math.isnan(v))) else str(v) for v in vals], type=t)
        else:
            arr = pa.array(df[col].tolist(), type=t)
        arrays.append(arr)
        schema_fields.append(pa.field(col, t))
    return pa.Table.from_arrays(arrays, schema=pa.schema(schema_fields))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not ZERO8.exists():
        log.error(f"zero8 file not found: {ZERO8}")
        log.error("먼저 Colab에서 Collect_GoogleNews_Zero8.ipynb 를 실행하세요.")
        return

    # 1) zero8 로드
    log.info(f"[1] Loading {ZERO8.name}")
    zdf = pd.read_parquet(ZERO8)
    log.info(f"    rows: {len(zdf):,}")
    log.info(f"    cols: {list(zdf.columns)}")

    # 2) title 기반 risk classify
    log.info("[2] Classifying risk_types from titles")
    risk_info = build_mega_patterns(RISK_YAML)
    zdf = classify_titles(zdf, risk_info)

    # 3) 스키마 정규화
    log.info("[3] Normalizing schema to v6 layout")
    zdf_norm = normalize_zero8_schema(zdf)
    log.info(f"    normalized rows: {len(zdf_norm):,}")

    # 기업별 통계
    log.info("\n=== Per-company recovery ===")
    for cid, n in zdf_norm.groupby("company_id").size().sort_values(ascending=False).items():
        log.info(f"  {cid:<12} {n:>6,}")

    if args.dry_run:
        log.info("[DRY-RUN] Skipping merge/save")
        return

    # 4) zero8 (url, company_id) 해시 집합 구축 (우선순위)
    log.info(f"\n[4] Building dedup key set from zero8 ({len(zdf_norm):,} rows)")
    import hashlib
    def mkkey(url: str, co: str) -> bytes:
        return hashlib.blake2b(f"{url}\x00{co}".encode("utf-8"), digest_size=8).digest()

    zero8_keys = set()
    for u, c in zip(zdf_norm["url"].tolist(), zdf_norm["company_id"].tolist()):
        zero8_keys.add(mkkey(u or "", c or ""))
    log.info(f"    zero8 unique keys: {len(zero8_keys):,}")

    # zero8을 arrow table로 변환 (한 번만)
    zarrow = to_arrow_with_nulls(zdf_norm)

    # 5) v6 스트리밍: row_group 단위로 읽고, zero8 key와 충돌하는 행만 필터링 후 write
    log.info("[5] Streaming v6 → writer, filtering duplicates")
    pf = pq.ParquetFile(V6)
    writer = None
    kept, dropped = 0, 0
    import pyarrow.compute as pc

    try:
        for rgi in range(pf.num_row_groups):
            tbl = pf.read_row_group(rgi)
            urls = tbl.column("url").to_pylist()
            cos = tbl.column("company_id").to_pylist()
            mask = np.ones(len(urls), dtype=bool)  # True = keep
            for i, (u, c) in enumerate(zip(urls, cos)):
                if mkkey(u or "", c or "") in zero8_keys:
                    mask[i] = False
            n_drop = (~mask).sum()
            if n_drop > 0:
                tbl = tbl.filter(pa.array(mask))
                dropped += int(n_drop)
            kept += tbl.num_rows

            if writer is None:
                writer = pq.ParquetWriter(V7, tbl.schema, compression="zstd")
            writer.write_table(tbl)
            if (rgi + 1) % 10 == 0:
                log.info(f"    row_group {rgi+1}/{pf.num_row_groups}  kept={kept:,}  dropped={dropped:,}")

        # 6) zero8 append
        log.info(f"[6] Appending zero8 ({zarrow.num_rows:,} rows)")
        # schema 맞추기 (v6 row_group과 동일한 field 순서)
        if writer is not None:
            zarrow_aligned = zarrow.cast(writer.schema) if zarrow.schema != writer.schema else zarrow
            writer.write_table(zarrow_aligned)
    finally:
        if writer is not None:
            writer.close()

    total = kept + zarrow.num_rows
    log.info(f"\n    v6 kept: {kept:,}  (dropped duplicates: {dropped:,})")
    log.info(f"    zero8 added: {zarrow.num_rows:,}")
    log.info(f"    v7 total: {total:,}")

    size_mb = V7.stat().st_size / 1024 / 1024
    log.info(f"    ✅ {V7.name}: {size_mb:.0f} MB")


if __name__ == "__main__":
    main()
