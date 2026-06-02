"""
Step 1b: FinBERT 추론용 입력 parquet 추출 (Colab 업로드용).

v2 설계 (메모리 2GB 제약 하에서 OOM 회피):
  - 한 번의 COPY 로 filter + GROUP BY + LEFT JOIN 을 모두 하면 2.89GB RSS 까지
    튀어서 9분 넘게 stuck 되는 것을 확인 → 3 단계로 분리.
  - Stage 1: A-bis 필터만 (no group, no join) → tmp_cand_raw.parquet
      · WHERE slug_quality IN ('good','fair')
            AND lang_detected='en'
            AND slug_text IS NOT NULL AND LENGTH(slug_text)>=8
      · 필요한 6개 컬럼만 슬림하게. 결과 예상: 7-11M rows ≈ 1GB.
  - Stage 2: URL 단위 dedup (GROUP BY url) → tmp_cand_dedup.parquet
      · input 크기가 이미 작기 때문에 hash aggregation OK.
      · MIN(slug_quality) → 알파벳순으로 'fair' 보다 'good' 우선.
  - Stage 3: v9 URL SEMI join (flag in_v9) → finbert_input_candidates.parquet
      · v9 side 를 먼저 DISTINCT parquet 으로 내려놓고 LEFT JOIN.

출력:
  data/processed/finbert_input_candidates.parquet
  reports/finbert_input_summary.txt
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import duckdb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
ABIS = ROOT / "data" / "processed" / "gdelt_url_slugs_abis.parquet"
V9 = ROOT / "data" / "processed" / "risk_events_classified_v9.parquet"
OUT = ROOT / "data" / "processed" / "finbert_input_candidates.parquet"
REPORT = ROOT / "reports" / "finbert_input_summary.txt"

TMP_DIR = ROOT / "tmp_finbert"
TMP_CAND_RAW = TMP_DIR / "tmp_cand_raw.parquet"
TMP_CAND_DEDUP = TMP_DIR / "tmp_cand_dedup.parquet"
TMP_V9_URLS = TMP_DIR / "tmp_v9_urls.parquet"


def open_con():
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='2GB'")
    con.execute("PRAGMA threads=2")
    con.execute("SET preserve_insertion_order=false")
    (ROOT / "tmp_duck").mkdir(exist_ok=True)
    con.execute(f"PRAGMA temp_directory='{ROOT / 'tmp_duck'}'")
    con.execute("PRAGMA max_temp_directory_size='8GiB'")
    return con


def stage1_filter():
    log.info("[Stage 1] Streaming filter A-bis → %s", TMP_CAND_RAW.name)
    t0 = time.time()
    con = open_con()
    con.execute(f"""
        COPY (
            SELECT
                article_url AS url,
                slug_text,
                first_date_raw,
                source_domain,
                slug_quality
            FROM read_parquet('{ABIS}')
            WHERE slug_quality IN ('good','fair')
                  AND lang_detected = 'en'
                  AND slug_text IS NOT NULL
                  AND LENGTH(slug_text) >= 8
        )
        TO '{TMP_CAND_RAW}' (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 200_000)
    """)
    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{TMP_CAND_RAW}')").fetchone()[0]
    log.info("  Stage1 done in %.1fs, rows=%d", time.time() - t0, n)
    con.close()
    return n


def stage2_dedup():
    log.info("[Stage 2] URL dedup → %s", TMP_CAND_DEDUP.name)
    t0 = time.time()
    con = open_con()
    con.execute(f"""
        COPY (
            SELECT
                url,
                MAX(slug_text) AS slug_text,
                MIN(first_date_raw) AS first_date_raw,
                ANY_VALUE(source_domain) AS source_domain,
                MIN(slug_quality) AS slug_quality
            FROM read_parquet('{TMP_CAND_RAW}')
            GROUP BY url
        )
        TO '{TMP_CAND_DEDUP}' (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 200_000)
    """)
    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{TMP_CAND_DEDUP}')").fetchone()[0]
    log.info("  Stage2 done in %.1fs, unique URLs=%d", time.time() - t0, n)
    con.close()
    return n


def stage3_v9_urls():
    log.info("[Stage 3a] v9 distinct urls → %s", TMP_V9_URLS.name)
    t0 = time.time()
    con = open_con()
    con.execute(f"""
        COPY (
            SELECT DISTINCT url FROM read_parquet('{V9}') WHERE url IS NOT NULL
        )
        TO '{TMP_V9_URLS}' (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 200_000)
    """)
    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{TMP_V9_URLS}')").fetchone()[0]
    log.info("  Stage3a done in %.1fs, v9 distinct urls=%d", time.time() - t0, n)
    con.close()
    return n


def stage3_join():
    log.info("[Stage 3b] LEFT JOIN in_v9 flag → %s", OUT.name)
    t0 = time.time()
    con = open_con()
    con.execute(f"""
        COPY (
            SELECT c.url, c.slug_text, c.first_date_raw, c.source_domain, c.slug_quality,
                   (v9u.url IS NOT NULL) AS in_v9
            FROM read_parquet('{TMP_CAND_DEDUP}') c
            LEFT JOIN read_parquet('{TMP_V9_URLS}') v9u
                   ON c.url = v9u.url
        )
        TO '{OUT}' (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 200_000)
    """)
    log.info("  Stage3b done in %.1fs", time.time() - t0)
    con.close()


def summary():
    log.info("Summary ...")
    con = open_con()
    n_total = con.execute(f"SELECT COUNT(*) FROM read_parquet('{OUT}')").fetchone()[0]
    n_in_v9 = con.execute(f"SELECT COUNT(*) FROM read_parquet('{OUT}') WHERE in_v9").fetchone()[0]
    n_good = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{OUT}') WHERE slug_quality='good'"
    ).fetchone()[0]
    n_fair = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{OUT}') WHERE slug_quality='fair'"
    ).fetchone()[0]
    n_good_v9 = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{OUT}') WHERE slug_quality='good' AND in_v9"
    ).fetchone()[0]

    year_dist = con.execute(f"""
        SELECT SUBSTRING(first_date_raw, 1, 4) AS year, COUNT(*) AS n
        FROM read_parquet('{OUT}')
        GROUP BY year ORDER BY year
    """).fetchdf()

    lines = []
    lines.append("=" * 72)
    lines.append("Step 1b · FinBERT input candidates extraction")
    lines.append("=" * 72)
    lines.append(f"Output parquet : {OUT.name}")
    lines.append(f"Total unique URLs       : {n_total:>12,}")
    lines.append(f"  good quality          : {n_good:>12,}")
    lines.append(f"  fair quality          : {n_fair:>12,}")
    lines.append(f"  ∩ v9 (covered by v9)  : {n_in_v9:>12,}")
    lines.append(f"  good ∩ v9             : {n_good_v9:>12,}")
    lines.append("")
    lines.append("-- by year --")
    for _, r in year_dist.iterrows():
        lines.append(f"  {str(r['year']):6s} : {int(r['n']):>12,}")
    lines.append("")
    lines.append(
        "Note: 'in_v9' true 인 row 는 v9 의 기존 event record 와 바로 merge 가능.\n"
        "     false 인 row 는 raw GDELT 에는 있으나 v9 필터링(risk 매칭 등)으로\n"
        "     배제된 URL 로, v10 확장 여부 별도 결정."
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    log.info("Report -> %s", REPORT)
    print("\n".join(lines))
    con.close()


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    if not TMP_CAND_RAW.exists():
        stage1_filter()
    else:
        log.info("[Stage 1] skip (exists)")
    if not TMP_CAND_DEDUP.exists():
        stage2_dedup()
    else:
        log.info("[Stage 2] skip (exists)")
    if not TMP_V9_URLS.exists():
        stage3_v9_urls()
    else:
        log.info("[Stage 3a] skip (exists)")
    stage3_join()
    summary()

    # cleanup tmp (keep if you want to resume)
    for p in (TMP_CAND_RAW, TMP_CAND_DEDUP, TMP_V9_URLS):
        try:
            p.unlink()
        except Exception:
            pass
    try:
        TMP_DIR.rmdir()
    except Exception:
        pass


if __name__ == "__main__":
    main()
