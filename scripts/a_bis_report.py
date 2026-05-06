"""Phase A-bis aggregation report (lightweight, separate from COPY).

slug parquet 의 aggregate 를 작은 단일 쿼리로 쪼개서 실행한다.
COUNT(DISTINCT) 가 무거우면 APPROX_COUNT_DISTINCT 로 대체.
"""
from __future__ import annotations

import logging
from pathlib import Path

import duckdb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "gdelt_url_slugs_abis.parquet"
REPORT = ROOT / "reports" / "slug_abis_coverage.txt"

PARQ = f"read_parquet('{OUT}')"


def main():
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='2GB'")
    con.execute("PRAGMA threads=2")
    con.execute("SET preserve_insertion_order=false")
    (ROOT / "tmp_duck").mkdir(exist_ok=True)
    con.execute(f"PRAGMA temp_directory='{ROOT / 'tmp_duck'}'")
    con.execute("PRAGMA max_temp_directory_size='8GiB'")

    log.info("counts ...")
    total_rows = con.execute(f"SELECT COUNT(*) FROM {PARQ}").fetchone()[0]
    log.info("  rows = %s", f"{total_rows:,}")
    uniq_urls = con.execute(
        f"SELECT APPROX_COUNT_DISTINCT(article_url) FROM {PARQ}"
    ).fetchone()[0]
    log.info("  approx unique URL = %s", f"{uniq_urls:,}")

    log.info("quality dist ...")
    qual_rows = con.execute(f"""
        SELECT slug_quality, COUNT(*) AS n
        FROM {PARQ} GROUP BY slug_quality ORDER BY n DESC
    """).fetchdf()
    qual_uniq = con.execute(f"""
        SELECT slug_quality, APPROX_COUNT_DISTINCT(article_url) AS n
        FROM {PARQ} GROUP BY slug_quality ORDER BY n DESC
    """).fetchdf()

    log.info("lang dist ...")
    lang_rows = con.execute(f"""
        SELECT lang_detected, COUNT(*) AS n
        FROM {PARQ} GROUP BY lang_detected ORDER BY n DESC LIMIT 25
    """).fetchdf()
    lang_uniq = con.execute(f"""
        SELECT lang_detected, APPROX_COUNT_DISTINCT(article_url) AS n
        FROM {PARQ} GROUP BY lang_detected ORDER BY n DESC LIMIT 25
    """).fetchdf()

    log.info("english combinations ...")
    en_combo = con.execute(f"""
        SELECT slug_quality,
               COUNT(*) AS n_rows,
               APPROX_COUNT_DISTINCT(article_url) AS n_urls
        FROM {PARQ}
        WHERE lang_detected = 'en'
        GROUP BY slug_quality ORDER BY n_urls DESC
    """).fetchdf()

    log.info("samples ...")
    samples = {}
    for qv in ["good", "fair", "opaque"]:
        samples[qv] = con.execute(f"""
            SELECT slug_text FROM {PARQ}
            WHERE slug_quality='{qv}' AND lang_detected='en'
            LIMIT 15
        """).fetchdf()

    # year x quality (rows basis - 빠름)
    log.info("year x quality ...")
    yrqu = con.execute(f"""
        SELECT SUBSTRING(first_date_raw, 1, 4) AS year, slug_quality, COUNT(*) AS n
        FROM {PARQ}
        WHERE first_date_raw IS NOT NULL
        GROUP BY year, slug_quality
        ORDER BY year, slug_quality
    """).fetchdf()

    # ----- write report -----
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("=" * 78)
    lines.append("Phase A-bis · Reclass + English Heuristic (DuckDB SQL pipeline)")
    lines.append("=" * 78)
    lines.append(f"Total rows (cross-file dup 포함) : {total_rows:>14,}")
    lines.append(f"APPROX unique article_url       : {uniq_urls:>14,}")
    lines.append("")
    lines.append("-- slug_quality (rows) --")
    for _, r in qual_rows.iterrows():
        lines.append(f"  {r['slug_quality']:7s} : {int(r['n']):>14,}  "
                     f"({int(r['n'])/total_rows*100:5.2f}%)")
    lines.append("")
    lines.append("-- slug_quality (APPROX unique URL) --")
    for _, r in qual_uniq.iterrows():
        lines.append(f"  {r['slug_quality']:7s} : {int(r['n']):>14,}")
    lines.append("")
    lines.append("-- lang_detected top 25 (rows) --")
    for _, r in lang_rows.iterrows():
        lines.append(f"  {str(r['lang_detected']):5s} : {int(r['n']):>14,}  "
                     f"({int(r['n'])/total_rows*100:5.2f}%)")
    lines.append("")
    lines.append("-- lang_detected top 25 (APPROX unique URL) --")
    for _, r in lang_uniq.iterrows():
        lines.append(f"  {str(r['lang_detected']):5s} : {int(r['n']):>14,}")
    lines.append("")
    lines.append("-- English × slug_quality --")
    lines.append("  quality | n_rows         | n_urls (approx)")
    for _, r in en_combo.iterrows():
        lines.append(f"  {r['slug_quality']:7s} | {int(r['n_rows']):>14,} | "
                     f"{int(r['n_urls']):>14,}")
    lines.append("")
    lines.append("-- year × slug_quality (rows) --")
    pivot = yrqu.pivot_table(index="year", columns="slug_quality",
                              values="n", fill_value=0, aggfunc="sum")
    pivot = pivot.reindex(columns=["good", "fair", "opaque", "empty"], fill_value=0)
    pivot["total"] = pivot.sum(axis=1)
    lines.append(pivot.to_string())
    lines.append("")
    for qv in ["good", "fair", "opaque"]:
        lines.append(f"-- sample {qv} × en (n=15) --")
        for _, r in samples[qv].iterrows():
            st = str(r["slug_text"])[:115] if r["slug_text"] else ""
            lines.append(f"  {st}")
        lines.append("")
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    log.info("Report -> %s", REPORT)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
