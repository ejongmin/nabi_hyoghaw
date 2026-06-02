#!/usr/bin/env python3
"""
extract_zero8_finbert_input.py
──────────────────────────────
v7에서 google_news_zero8 행을 언어별로 분리해 FinBERT 입력 parquet을 생성.

출력:
    data/processed/finbert_input_zero8_en.parquet
    data/processed/finbert_input_zero8_ja.parquet
    data/processed/finbert_input_zero8_de.parquet
    data/processed/finbert_input_zero8_es.parquet

각 파일 스키마: rep_event_id, title  (언어별 unique title 기준, dedup)

Usage:
    python scripts/extract_zero8_finbert_input.py
"""
import logging
from pathlib import Path
import pandas as pd
import duckdb

log = logging.getLogger("extract_zero8")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ROOT = Path(__file__).resolve().parent.parent
V7 = ROOT / "data" / "processed" / "risk_events_classified_v7.parquet"
OUTDIR = ROOT / "data" / "processed"
ZERO8_RAW = ROOT / "data" / "raw" / "supplement" / "google_news_zero8.parquet"


def main():
    if not V7.exists():
        log.error(f"v7 not found: {V7}")
        return

    con = duckdb.connect(":memory:")
    con.execute("PRAGMA memory_limit='3GB'")

    # zero8 raw에 lang 컬럼이 있으니 우선 이걸로 split
    # v7에 없으면 raw에서 url 기준 join하여 lang 복원
    use_raw_lang = ZERO8_RAW.exists()

    if use_raw_lang:
        log.info("Using lang column from raw zero8 parquet (joined by url)")
        q = f"""
        WITH zero8 AS (
            SELECT DISTINCT url, lang
            FROM read_parquet('{ZERO8_RAW}')
        ),
        v7_zero8 AS (
            SELECT v.event_id, v.title, v.url, z.lang
            FROM read_parquet('{V7}') v
            INNER JOIN zero8 z ON v.url = z.url
            WHERE v.classify_source = 'google_news_zero8'
              AND v.title IS NOT NULL
              AND LENGTH(TRIM(v.title)) >= 10
        )
        SELECT lang, COUNT(*) AS n, COUNT(DISTINCT title) AS n_unique
        FROM v7_zero8 GROUP BY 1 ORDER BY n DESC
        """
    else:
        log.warning("raw zero8 not found, fallback: unicode-range detection")
        q = f"""
        WITH v7_zero8 AS (
            SELECT event_id, title, url,
              CASE
                WHEN REGEXP_MATCHES(title, '[\\u3040-\\u309F\\u30A0-\\u30FF]') THEN 'ja'
                WHEN REGEXP_MATCHES(title, '[\\u00E4\\u00F6\\u00FC\\u00DF]') THEN 'de'
                WHEN REGEXP_MATCHES(title, '[\\u00F1]') THEN 'es'
                ELSE 'en'
              END AS lang
            FROM read_parquet('{V7}')
            WHERE classify_source = 'google_news_zero8'
              AND title IS NOT NULL
              AND LENGTH(TRIM(title)) >= 10
        )
        SELECT lang, COUNT(*) AS n, COUNT(DISTINCT title) AS n_unique
        FROM v7_zero8 GROUP BY 1 ORDER BY n DESC
        """

    log.info("Language distribution:")
    for r in con.execute(q).fetchall():
        log.info(f"  {r[0]:<4}: {r[1]:>6,}  (unique titles: {r[2]:>6,})")

    # 언어별 extract  (raw parquet은 'jp'/'kr'/'cn' 코드, FinBERT 출력은 'ja'/'ko'/'zh' 코드로 통일)
    # lang_map: FinBERT 표준 → raw 코드
    lang_aliases = {"en": ["en"], "ja": ["ja", "jp"], "de": ["de"], "es": ["es"]}
    for lang in ["en", "ja", "de", "es"]:
        aliases = lang_aliases[lang]
        in_clause = ",".join(f"'{a}'" for a in aliases)
        if use_raw_lang:
            q_lang = f"""
            WITH zero8 AS (
                SELECT DISTINCT url, lang FROM read_parquet('{ZERO8_RAW}')
            ),
            labeled AS (
                SELECT v.event_id, v.title
                FROM read_parquet('{V7}') v
                INNER JOIN zero8 z ON v.url = z.url
                WHERE v.classify_source = 'google_news_zero8'
                  AND z.lang IN ({in_clause})
                  AND v.title IS NOT NULL
                  AND LENGTH(TRIM(v.title)) >= 10
            ),
            unique_titles AS (
                SELECT title, MIN(event_id) AS rep_event_id
                FROM labeled GROUP BY title
            )
            SELECT rep_event_id, title FROM unique_titles
            """
        else:
            q_lang = f"""
            WITH labeled AS (
                SELECT event_id, title,
                  CASE
                    WHEN REGEXP_MATCHES(title, '[\\u3040-\\u309F\\u30A0-\\u30FF]') THEN 'ja'
                    WHEN REGEXP_MATCHES(title, '[\\u00E4\\u00F6\\u00FC\\u00DF]') THEN 'de'
                    WHEN REGEXP_MATCHES(title, '[\\u00F1]') THEN 'es'
                    ELSE 'en'
                  END AS lang
                FROM read_parquet('{V7}')
                WHERE classify_source = 'google_news_zero8'
                  AND title IS NOT NULL
                  AND LENGTH(TRIM(title)) >= 10
            ),
            unique_titles AS (
                SELECT title, MIN(event_id) AS rep_event_id
                FROM labeled WHERE lang = '{lang}' GROUP BY title
            )
            SELECT rep_event_id, title FROM unique_titles
            """
        df = con.execute(q_lang).fetchdf()
        if len(df) == 0:
            log.info(f"  [{lang}] 0 rows, skipping")
            continue
        out = OUTDIR / f"finbert_input_zero8_{lang}.parquet"
        df.to_parquet(out, index=False, compression="zstd")
        log.info(f"  [{lang}] saved {len(df):,} unique titles → {out.name}")


if __name__ == "__main__":
    main()
