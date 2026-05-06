"""
Phase A-bis v3: pure-DuckDB reclass + 영어 휴리스틱 (high throughput).

설계:
  - lingua-py 는 row 당 ~1ms → 15.9M rows = 5+ 시간 → 폐기.
  - 대안 : DuckDB SQL 한 패스 + regex.
  - 영어 판정 휴리스틱 : slug_text 가 ASCII letter 우세이면서 영문 stopword
    (the/of/and/in/to/a/is/for/on/with/by/at/from/that/this/are/was/will/has) 가
    하나라도 포함되면 'en'. 그렇지 않으면 slug_lang_hint(TLD 기반) 유지.
    이 휴리스틱의 의도는 "FinBERT 에 안전하게 넣을 수 있는 영어 후보" 를 보수적으로
    추리는 것이며, false-negative 가 false-positive 보다 비용이 낮다.
  - 재분류 규칙(opaque 격하)도 SQL CASE 로 처리.

입력 : data/processed/gdelt_url_slugs.parquet     (15.9M rows)
출력 : data/processed/gdelt_url_slugs_abis.parquet (15.9M rows + 2 cols)
       reports/slug_abis_coverage.txt
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import duckdb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "data" / "processed" / "gdelt_url_slugs.parquet"
OUT = ROOT / "data" / "processed" / "gdelt_url_slugs_abis.parquet"
REPORT = ROOT / "reports" / "slug_abis_coverage.txt"

# CMS-id pattern
CMS_RE = (
    r"^(detail|article|doc|news|content|story|storys|post|topic|item|page|view|read|info)"
    r"\s+[0-9a-z]{5,}$"
)
HASH_RE = r"^[a-z0-9]{7,}$"
EN_STOP_RE = (
    r"\b(the|of|and|in|to|a|is|for|on|with|by|at|from|that|this|are|was|will|has|"
    r"have|been|after|before|over|new|over|out|up|how|what|why|which|its|their)\b"
)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='2GB'")
    con.execute("PRAGMA threads=2")
    (ROOT / "tmp_duck").mkdir(exist_ok=True)
    con.execute(f"PRAGMA temp_directory='{ROOT / 'tmp_duck'}'")

    log.info("Streaming reclass + lang heuristic via DuckDB COPY ...")
    t0 = time.time()
    con.execute(f"""
        COPY (
            WITH base AS (
                SELECT
                    article_url,
                    source_domain,
                    first_date_raw,
                    slug_raw,
                    slug_text,
                    slug_lang_hint,
                    slug_word_count,
                    slug_quality AS slug_quality_v1,
                    LOWER(COALESCE(slug_text, '')) AS slug_lower,
                    LENGTH(COALESCE(slug_text, '')) AS slug_len
                FROM read_parquet('{IN}')
            ),
            scored AS (
                SELECT
                    *,
                    -- letter-ish chars : keep latin, drop digits/punct/space
                    LENGTH(REGEXP_REPLACE(slug_lower, '[0-9 _\-+\.]', '', 'g')) AS alpha_len,
                    -- ASCII a-z only (English heuristic용)
                    LENGTH(REGEXP_REPLACE(slug_lower, '[^a-z]', '', 'g')) AS ascii_len
                FROM base
            ),
            recl AS (
                SELECT
                    *,
                    CASE
                        WHEN slug_quality_v1 IN ('empty','opaque') THEN slug_quality_v1
                        WHEN slug_text IS NULL OR slug_text = '' THEN 'empty'
                        WHEN REGEXP_MATCHES(slug_lower, '{CMS_RE}') THEN 'opaque'
                        WHEN slug_word_count = 1
                             AND REGEXP_MATCHES(slug_lower, '{HASH_RE}')
                             AND NOT REGEXP_MATCHES(slug_lower, '[aeiou]') THEN 'opaque'
                        WHEN slug_len > 0 AND alpha_len::DOUBLE / slug_len < 0.25 THEN 'opaque'
                        ELSE slug_quality_v1
                    END AS slug_quality
                FROM scored
            )
            SELECT
                article_url,
                source_domain,
                first_date_raw,
                slug_raw,
                slug_text,
                slug_quality,
                slug_lang_hint,
                slug_word_count,
                CASE
                    WHEN slug_text IS NULL OR slug_text = '' THEN slug_lang_hint
                    -- 영어 휴리스틱 : ASCII a-z 비율 ≥ 0.5 AND 영문 stopword 포함
                    WHEN slug_len > 0
                         AND ascii_len::DOUBLE / slug_len >= 0.5
                         AND REGEXP_MATCHES(slug_lower, '{EN_STOP_RE}')
                         THEN 'en'
                    ELSE slug_lang_hint
                END AS lang_detected
            FROM recl
        )
        TO '{OUT}' (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 500_000)
    """)
    log.info("COPY done in %.1fs", time.time() - t0)

    # ---------- aggregates ----------
    log.info("Aggregating ...")
    total_rows = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{OUT}')"
    ).fetchone()[0]
    uniq_urls = con.execute(
        f"SELECT COUNT(DISTINCT article_url) FROM read_parquet('{OUT}')"
    ).fetchone()[0]

    qual_dist = con.execute(f"""
        SELECT slug_quality,
               COUNT(*) AS n_rows,
               COUNT(DISTINCT article_url) AS n_urls
        FROM read_parquet('{OUT}')
        GROUP BY slug_quality
        ORDER BY n_urls DESC
    """).fetchdf()

    lang_dist = con.execute(f"""
        SELECT lang_detected,
               COUNT(*) AS n_rows,
               COUNT(DISTINCT article_url) AS n_urls
        FROM read_parquet('{OUT}')
        GROUP BY lang_detected
        ORDER BY n_urls DESC
        LIMIT 25
    """).fetchdf()

    en_good = con.execute(f"""
        SELECT
            COUNT(DISTINCT article_url) FILTER
                (WHERE slug_quality='good' AND lang_detected='en')         AS n_good_en,
            COUNT(DISTINCT article_url) FILTER
                (WHERE slug_quality IN ('good','fair') AND lang_detected='en') AS n_gf_en,
            COUNT(DISTINCT article_url) FILTER
                (WHERE slug_quality='opaque' AND lang_detected='en')       AS n_opq_en,
            COUNT(DISTINCT article_url) FILTER
                (WHERE slug_quality='good')                                AS n_good_all,
            COUNT(DISTINCT article_url) FILTER
                (WHERE slug_quality IN ('good','fair'))                    AS n_gf_all
        FROM read_parquet('{OUT}')
    """).fetchone()

    lines = []
    lines.append("=" * 78)
    lines.append("Phase A-bis · Reclassified slug + English heuristic (DuckDB SQL)")
    lines.append("=" * 78)
    lines.append(f"Total rows (with cross-file dup) : {total_rows:>14,}")
    lines.append(f"Unique article_url               : {uniq_urls:>14,}")
    lines.append("")
    lines.append("-- Reclassified slug_quality (unique URL basis) --")
    for _, r in qual_dist.iterrows():
        lines.append(f"  {r['slug_quality']:7s} : n_urls={int(r['n_urls']):>13,}  "
                     f"({int(r['n_urls'])/max(uniq_urls,1)*100:5.2f}%)")
    lines.append("")
    lines.append("-- lang_detected top 25 (unique URL basis) --")
    for _, r in lang_dist.iterrows():
        lang = str(r["lang_detected"])
        lines.append(f"  {lang:5s} : n_urls={int(r['n_urls']):>13,}  "
                     f"({int(r['n_urls'])/max(uniq_urls,1)*100:5.2f}%)")
    lines.append("")
    lines.append("-- FinBERT-eligible English candidates --")
    lines.append(f"  good (all langs)            : {int(en_good[3]):>13,}")
    lines.append(f"  good+fair (all langs)       : {int(en_good[4]):>13,}")
    lines.append(f"  good × en                   : {int(en_good[0]):>13,}")
    lines.append(f"  good+fair × en              : {int(en_good[1]):>13,}  <<< FinBERT 후보")
    lines.append(f"  opaque × en (Phase B 대상)  : {int(en_good[2]):>13,}")
    lines.append("")
    for qv in ["good", "fair", "opaque"]:
        lines.append(f"-- sample {qv} × en (n=15) --")
        samp = con.execute(f"""
            SELECT slug_text
            FROM read_parquet('{OUT}')
            WHERE slug_quality = '{qv}' AND lang_detected = 'en'
            LIMIT 15
        """).fetchdf()
        for _, r in samp.iterrows():
            st = str(r["slug_text"])[:115] if r["slug_text"] else ""
            lines.append(f"  {st}")
        lines.append("")

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    log.info("Report: %s", REPORT)
    print("\n".join(lines[:120]))


if __name__ == "__main__":
    main()
