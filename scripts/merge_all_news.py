#!/usr/bin/env python
"""
전체 뉴스 소스 병합기
GDELT + BigKinds + 중국 뉴스 + 산업 RSS + 보완 데이터 → 통합 risk_events.parquet

사용법:
  python scripts/merge_all_news.py
  python scripts/merge_all_news.py --dry-run  # 병합 결과만 미리보기

입력 파일 (있는 것만 자동 감지):
  data/processed/risk_events_bq.parquet          ← GDELT BigQuery (메인)
  data/raw/supplement/supplement_news.parquet     ← Google News RSS 보완
  data/raw/bigkinds/bigkinds_news.parquet         ← BigKinds 한국 뉴스
  data/raw/chinese_news/chinese_news.parquet      ← 중국 뉴스
  data/raw/industry_rss/industry_news.parquet     ← 산업 RSS

출력:
  data/processed/risk_events.parquet              ← 통합 (이후 파이프라인 입력)
"""
import argparse
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger("merge_news")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

# ── 입력 소스 정의 ──
SOURCES = [
    ("GDELT BigQuery", "data/processed/risk_events_bq.parquet"),
    ("Supplement RSS", "data/raw/supplement/supplement_news.parquet"),
    ("BigKinds (KR)", "data/raw/bigkinds/bigkinds_news.parquet"),
    ("Chinese News", "data/raw/chinese_news/chinese_news.parquet"),
    ("Industry RSS", "data/raw/industry_rss/industry_news.parquet"),
]

# 통일 스키마
SCHEMA_COLS = [
    "event_id", "event_time", "url", "title", "risk_types",
    "severity", "entity_ids", "entity_scores", "finbert",
    "country_ids", "collection_type", "source_domain",
]


def _normalize_schema(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """각 소스의 스키마를 통일."""
    for col in SCHEMA_COLS:
        if col not in df.columns:
            if col in ("entity_ids", "entity_scores", "country_ids"):
                df[col] = [[] for _ in range(len(df))]
            elif col in ("severity", "finbert"):
                df[col] = 0.0
            else:
                df[col] = ""

    # event_time 통일 (datetime)
    if "event_time" in df.columns:
        df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce", utc=True)

    return df[SCHEMA_COLS]


def main():
    parser = argparse.ArgumentParser(description="전체 뉴스 소스 병합기")
    parser.add_argument("--dry-run", action="store_true", help="병합 결과 미리보기만")
    parser.add_argument("--output", default="data/processed/risk_events.parquet")
    args = parser.parse_args()

    root = Path(".")
    dfs = []

    log.info("=" * 60)
    log.info("뉴스 소스 병합기")
    log.info("=" * 60)

    for name, path_str in SOURCES:
        path = root / path_str
        if path.exists():
            df = pd.read_parquet(path)
            df = _normalize_schema(df, name)
            log.info(f"  ✅ {name:20s}: {len(df):>10,} rows  ({path_str})")
            dfs.append(df)
        else:
            log.info(f"  ⬜ {name:20s}: (not found)      ({path_str})")

    if not dfs:
        log.error("No source files found!")
        return

    # 병합
    merged = pd.concat(dfs, ignore_index=True)
    log.info(f"\n  Total before dedup: {len(merged):,}")

    # URL 기반 dedup
    n_before = len(merged)
    merged = merged.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
    n_dedup = n_before - len(merged)
    log.info(f"  Deduped: {n_dedup:,} (by URL)")
    log.info(f"  Total after dedup: {len(merged):,}")

    # 소스별 통계
    log.info("\n=== Per-source breakdown ===")
    for ct, group in merged.groupby("collection_type"):
        log.info(f"  {ct:30s}: {len(group):>10,}")

    # 기업별 통계 (entity_ids가 list인 경우)
    log.info("\n=== Per-company top 20 ===")
    try:
        all_entities = merged["entity_ids"].explode().dropna()
        entity_counts = all_entities.value_counts()
        for cid, cnt in entity_counts.head(20).items():
            log.info(f"  {str(cid):15s}: {cnt:>10,}")
    except Exception:
        pass

    if args.dry_run:
        log.info("\n[DRY-RUN] No files written.")
        return

    # 저장
    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)
    log.info(f"\n✅ Saved: {output_path} ({len(merged):,} rows)")


if __name__ == "__main__":
    main()
