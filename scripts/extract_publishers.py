#!/usr/bin/env python3
"""
extract_publishers.py
─────────────────────
v7_final의 title에서 Google News RSS ' - Publisher' 접미사를 파싱해
실제 발행사 이름을 복구하고 분포를 출력.

전제: Google News RSS는 거의 모든 title을 `<actual title> - <publisher>` 형식으로 제공.
source_domain 컬럼이 대부분 'news.google.com'으로 채워져 있어 무용지물이므로
title 접미사에서 발행사를 추출하는 것이 유일한 비용-0 방법.

출력:
    data/processed/publisher_distribution.csv
    (컬럼: publisher, n_rows, n_title_rows, n_zero8_rows)

Usage:
    python scripts/extract_publishers.py
"""
import re
import logging
from pathlib import Path

import duckdb
import pandas as pd

log = logging.getLogger("extract_publishers")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ROOT = Path(__file__).resolve().parent.parent
V7 = ROOT / "data" / "processed" / "risk_events_classified_v7_final.parquet"
OUT = ROOT / "data" / "processed" / "publisher_distribution.csv"

# 마지막 ' - Publisher' 또는 ' – Publisher'(long dash) 또는 ' — Publisher'(em dash) 캡처
SUFFIX_RE = re.compile(r"\s*[-–—]\s*([^-–—]+?)\s*$")


def extract_publisher(title: str) -> str:
    if not title or not isinstance(title, str):
        return ""
    m = SUFFIX_RE.search(title.strip())
    if not m:
        return ""
    pub = m.group(1).strip()
    # 너무 짧거나 너무 길면 publisher가 아닐 가능성
    if len(pub) < 2 or len(pub) > 80:
        return ""
    return pub


def main():
    if not V7.exists():
        log.error(f"v7_final not found: {V7}")
        return

    con = duckdb.connect(":memory:")
    con.execute("PRAGMA memory_limit='3GB'")

    log.info("[1] Loading title-bearing rows")
    df = con.execute(f"""
        SELECT title, classify_source, source_domain
        FROM read_parquet('{V7}')
        WHERE title IS NOT NULL AND LENGTH(TRIM(title)) >= 10
    """).fetchdf()
    log.info(f"  loaded: {len(df):,} rows")

    log.info("[2] Extracting publisher from title suffix")
    df["publisher"] = df["title"].apply(extract_publisher)

    extracted = (df["publisher"] != "").sum()
    log.info(f"  extracted: {extracted:,} / {len(df):,}  ({100*extracted/len(df):.1f}%)")

    # source_domain이 pretty-name인 경우는 그대로 사용 (legacy cn/kr/de collectors)
    mask_no_pub = (df["publisher"] == "") & (df["source_domain"] != "news.google.com")
    df.loc[mask_no_pub, "publisher"] = df.loc[mask_no_pub, "source_domain"]
    log.info(f"  fallback from source_domain: {int(mask_no_pub.sum()):,}")

    # Publisher 분포 집계
    log.info("[3] Aggregating publisher distribution")
    pub_df = (
        df[df["publisher"] != ""]
        .groupby("publisher")
        .size()
        .reset_index(name="n_title_rows")
        .sort_values("n_title_rows", ascending=False)
        .reset_index(drop=True)
    )

    zero8_pub = df[df["classify_source"] == "google_news_zero8"].groupby("publisher").size()
    pub_df["n_zero8_rows"] = pub_df["publisher"].map(zero8_pub).fillna(0).astype(int)

    pub_df.to_csv(OUT, index=False)
    log.info(f"  saved: {OUT}")
    log.info(f"  total unique publishers: {len(pub_df):,}")

    log.info("\n=== Top 80 publishers ===")
    for _, r in pub_df.head(80).iterrows():
        log.info(f"  {r['n_title_rows']:>6,}  {r['publisher']}")

    log.info("\n=== Rows without extracted publisher ===")
    missing = df[df["publisher"] == ""]
    log.info(f"  missing: {len(missing):,} ({100*len(missing)/len(df):.1f}%)")
    if len(missing) > 0:
        log.info("  sample titles:")
        for t in missing["title"].head(5).tolist():
            log.info(f"    {t[:110]}")


if __name__ == "__main__":
    main()
