"""
Step 1: A-bis 영어 휴리스틱 precision 측정.

50K 무작위 샘플 (lang_detected='en') 에 대해 langid.py (self-contained) 로
실제 언어를 재판정하고 precision / 언어 breakdown 을 계산한다.

정확도 향상을 위해 'good' 과 'good|fair' 서브셋에 각각 측정.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import duckdb
import langid

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "data" / "processed" / "gdelt_url_slugs_abis.parquet"
REPORT = ROOT / "reports" / "en_precision.txt"

SAMPLE_N = 50_000


def sample_and_classify(con, qualities, label):
    quality_in = "','".join(qualities)
    q = f"""
        SELECT slug_text
        FROM read_parquet('{IN}')
        WHERE slug_quality IN ('{quality_in}') AND lang_detected='en'
              AND slug_text IS NOT NULL AND LENGTH(slug_text) >= 8
        USING SAMPLE {SAMPLE_N} ROWS
    """
    t0 = time.time()
    df = con.execute(q).fetchdf()
    log.info("  [%s] sampled %d rows (%.1fs)", label, len(df), time.time() - t0)

    counts = {}
    n_en = 0
    unk = 0
    t0 = time.time()
    for i, txt in enumerate(df["slug_text"].tolist()):
        try:
            lang, _ = langid.classify(txt)
        except Exception:
            lang = "unk"
            unk += 1
        counts[lang] = counts.get(lang, 0) + 1
        if lang == "en":
            n_en += 1
        if (i + 1) % 10000 == 0:
            log.info("    [%s] %d/%d (%.0f/s)",
                     label, i + 1, len(df),
                     (i + 1) / max(time.time() - t0, 1))
    precision = n_en / max(len(df), 1)
    log.info("  [%s] precision(en) = %.4f (%d/%d)",
             label, precision, n_en, len(df))
    return precision, counts, len(df)


def main():
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='2GB'")
    con.execute("PRAGMA threads=2")

    log.info("Measuring English heuristic precision via langid.py ...")

    prec_good, cnt_good, n_good = sample_and_classify(con, ["good"], "good")
    prec_gf, cnt_gf, n_gf = sample_and_classify(con, ["good", "fair"], "good+fair")

    # convert to v9 - like extrapolation
    approx_good_urls = 4_880_920
    approx_gf_urls = 6_428_648

    lines = []
    lines.append("=" * 78)
    lines.append("Step 1 · English heuristic precision (langid.py, 50K sample each)")
    lines.append("=" * 78)
    lines.append(f"Sample size (good)      : {n_good:,}")
    lines.append(f"Sample size (good+fair) : {n_gf:,}")
    lines.append("")
    lines.append(f"precision(good × en)      = {prec_good:.4f}")
    lines.append(f"precision(good|fair × en) = {prec_gf:.4f}")
    lines.append("")
    lines.append("-- Extrapolated true-English URL counts --")
    lines.append(f"  good × en        : {int(approx_good_urls * prec_good):>12,} "
                 f"(of {approx_good_urls:,} approx)")
    lines.append(f"  good|fair × en   : {int(approx_gf_urls * prec_gf):>12,} "
                 f"(of {approx_gf_urls:,} approx)")
    lines.append("")
    lines.append("-- good × en langid breakdown top 15 --")
    for lang, n in sorted(cnt_good.items(), key=lambda x: -x[1])[:15]:
        lines.append(f"  {lang:5s} : {n:>6,}  ({n / n_good * 100:5.2f}%)")
    lines.append("")
    lines.append("-- good|fair × en langid breakdown top 15 --")
    for lang, n in sorted(cnt_gf.items(), key=lambda x: -x[1])[:15]:
        lines.append(f"  {lang:5s} : {n:>6,}  ({n / n_gf * 100:5.2f}%)")
    lines.append("")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    log.info("Report -> %s", REPORT)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
