"""
Phase A: URL 슬러그 파싱으로 title text 후보 추출 (row-group streaming 버전)

v2: 이전 버전은 DuckDB .fetchdf() 가 파일 전체를 한 번에 pandas 로 올려
    3.99M 행 파일에서 OOM kill. 이번엔 pyarrow ParquetFile 의 row_group 단위로
    필요한 컬럼(article_url, source_domain, date)만 읽어 chunk 별로 파싱한다.

Input  : data/raw/gdelt/gdelt_*.parquet  (8 files)
Output : data/processed/gdelt_url_slugs.parquet
         reports/slug_coverage.txt
"""
from __future__ import annotations

import gc
import logging
import re
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "gdelt"
OUT = ROOT / "data" / "processed" / "gdelt_url_slugs.parquet"
REPORT = ROOT / "reports" / "slug_coverage.txt"

RAW_FILES = [
    RAW_DIR / "gdelt_2018.parquet",
    RAW_DIR / "gdelt_2018_2019.parquet",
    RAW_DIR / "gdelt_gkg_2019~2020.parquet",
    RAW_DIR / "gdelt_gkg_2021~2022.parquet",
    RAW_DIR / "gdelt_gkg_2022.parquet",
    RAW_DIR / "gdelt_gkg_2023.parquet",
    RAW_DIR / "gdelt_gkg_2024.parquet",
    RAW_DIR / "gdelt_gkg_2025.parquet",
]

RE_NUMERIC = re.compile(r"^[0-9_\-]+$")
RE_HASH = re.compile(r"^[A-Za-z0-9]{8,}$")
RE_VOWEL = re.compile(r"[aeiouAEIOU가-힣一-龥あ-んア-ン]")
RE_EXT = re.compile(r"\.(html?|php|aspx?|jsp|cms|epi|amp|stm|shtml)$", re.I)
RE_SEP = re.compile(r"[-_+]+")
RE_WS = re.compile(r"\s+")

LANG_HINT_TLD = {
    "cn": "zh", "com.cn": "zh", "sina.com.cn": "zh", "163.com": "zh",
    "sohu.com": "zh", "ifeng.com": "zh", "baidu.com": "zh", "qq.com": "zh",
    "jp": "ja", "co.jp": "ja", "nikkei.com": "ja",
    "kr": "ko", "co.kr": "ko", "naver.com": "ko", "chosun.com": "ko",
    "ru": "ru", "su": "ru", "ua": "uk",
    "de": "de", "fr": "fr", "it": "it", "es": "es", "pt": "pt",
    "pl": "pl", "cz": "cs", "nl": "nl", "se": "sv", "no": "no",
    "vn": "vi", "th": "th", "id": "id", "com.id": "id",
    "ir": "fa", "sa": "ar", "eg": "ar", "ma": "ar", "lb": "ar", "iq": "ar",
    "in": "en", "co.in": "en", "uk": "en", "co.uk": "en",
    "au": "en", "com.au": "en", "us": "en", "ca": "en", "nz": "en",
    "br": "pt", "com.br": "pt", "mx": "es", "ar": "es", "cl": "es",
    "tr": "tr", "bd": "bn", "pk": "en", "ng": "en", "za": "en",
    "ro": "ro", "bg": "bg", "gr": "el", "hu": "hu", "fi": "fi",
    "rs": "sr", "hr": "hr", "si": "sl", "sk": "sk",
}


def detect_lang_hint(domain: str) -> str:
    if not domain:
        return "unk"
    d = domain.lower()
    if d.startswith("www."):
        d = d[4:]
    for k, v in LANG_HINT_TLD.items():
        if d == k or d.endswith("." + k):
            return v
    return "en"


def parse_one(url: str):
    if url is None:
        return (None, None, "empty", "unk", 0)
    try:
        p = urlparse(url)
    except Exception:
        return (None, None, "empty", "unk", 0)

    domain = p.netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    lang_hint = detect_lang_hint(domain)
    segs = [s for s in p.path.strip("/").split("/") if s]
    if not segs:
        return (None, None, "empty", lang_hint, 0)

    last = segs[-1]
    last = RE_EXT.sub("", last)
    if RE_NUMERIC.match(last) and len(segs) >= 2:
        last = segs[-2]
        last = RE_EXT.sub("", last)

    slug_raw = last
    try:
        decoded = unquote(last)
    except Exception:
        decoded = last

    text = RE_WS.sub(" ", RE_SEP.sub(" ", decoded)).strip()
    wc = len(text.split()) if text else 0

    if not text:
        qual = "empty"
    elif RE_NUMERIC.match(slug_raw):
        qual = "opaque"
    elif wc >= 4:
        alpha = 0
        for c in text:
            if (c.isalpha()
                or "\u4e00" <= c <= "\u9fff"
                or "\uac00" <= c <= "\ud7af"
                or "\u3040" <= c <= "\u30ff"):
                alpha += 1
        ratio = alpha / max(len(text), 1)
        qual = "good" if ratio >= 0.6 else ("fair" if ratio >= 0.3 else "opaque")
    elif wc >= 2:
        qual = "fair"
    else:
        if RE_HASH.match(slug_raw) and not RE_VOWEL.search(slug_raw):
            qual = "opaque"
        else:
            qual = "fair" if wc == 1 and len(text) >= 6 else "opaque"

    return (slug_raw, text, qual, lang_hint, wc)


def process_chunk(df: pd.DataFrame, quality_ct, lang_ct, year_ct) -> pa.Table:
    parsed = df["article_url"].map(parse_one)
    df[["slug_raw", "slug_text", "slug_quality", "slug_lang_hint", "slug_word_count"]] = pd.DataFrame(
        parsed.tolist(), index=df.index
    )
    for k, v in df["slug_quality"].value_counts().items():
        quality_ct[k] = quality_ct.get(k, 0) + int(v)
    for k, v in df["slug_lang_hint"].value_counts().items():
        lang_ct[k] = lang_ct.get(k, 0) + int(v)
    if "first_date_raw" in df.columns:
        try:
            yrs = df["first_date_raw"].astype(str).str[:4]
            for (y, q), n in df.groupby([yrs, "slug_quality"]).size().items():
                year_ct[(y, q)] = year_ct.get((y, q), 0) + int(n)
        except Exception:
            pass
    cols = ["article_url", "source_domain", "first_date_raw",
            "slug_raw", "slug_text", "slug_quality", "slug_lang_hint", "slug_word_count"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    # normalize first_date_raw to string so all files share same schema
    df["first_date_raw"] = df["first_date_raw"].astype("string")
    return pa.Table.from_pandas(df[cols], preserve_index=False)


def process_file(f: Path, writer_holder, quality_ct, lang_ct, year_ct, total):
    pf = pq.ParquetFile(f)
    all_names = set(pf.schema_arrow.names)
    wanted = ["article_url", "source_domain"]
    date_col = None
    for c in ("date", "gkg_date"):
        if c in all_names:
            date_col = c
            wanted.append(c)
            break
    log.info("  %s: row_groups=%d rows=%d cols_used=%s",
             f.name, pf.num_row_groups, pf.metadata.num_rows, wanted)
    file_rows = 0
    t0 = time.time()
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg, columns=wanted)
        df = tbl.to_pandas()
        del tbl
        df = df[df["article_url"].notna()].copy()
        df = df.drop_duplicates(subset=["article_url"])
        if date_col and date_col != "first_date_raw":
            df = df.rename(columns={date_col: "first_date_raw"})
        out_tbl = process_chunk(df, quality_ct, lang_ct, year_ct)
        if writer_holder[0] is None:
            writer_holder[0] = pq.ParquetWriter(OUT, out_tbl.schema, compression="snappy")
        writer_holder[0].write_table(out_tbl)
        file_rows += len(df)
        total[0] += len(df)
        del df, out_tbl
        gc.collect()
        log.info("    rg %d/%d: +%d (file=%d, total=%d, %.1fs)",
                 rg + 1, pf.num_row_groups, file_rows - (file_rows - len(df) if False else 0) if False else 0,
                 file_rows, total[0], time.time() - t0)
    log.info("  %s DONE: %d rows (%.1fs)", f.name, file_rows, time.time() - t0)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    writer_holder = [None]
    total = [0]
    quality_ct = {"good": 0, "fair": 0, "opaque": 0, "empty": 0}
    lang_ct = {}
    year_ct = {}

    log.info("Processing %d raw files (row-group streaming)...", len(RAW_FILES))
    for f in RAW_FILES:
        if not f.exists():
            log.warning("missing: %s", f)
            continue
        process_file(f, writer_holder, quality_ct, lang_ct, year_ct, total)

    if writer_holder[0] is not None:
        writer_holder[0].close()
    log.info("Wrote %s (%d rows total, cross-file dup possible)", OUT, total[0])

    lines = []
    lines.append("=" * 72)
    lines.append("Phase A · URL Slug Coverage Report")
    lines.append("=" * 72)
    lines.append(f"Total rows written     : {total[0]:>12,}")
    lines.append("")
    lines.append("-- Overall slug_quality --")
    for k in ["good", "fair", "opaque", "empty"]:
        n = quality_ct.get(k, 0)
        pct = n / max(total[0], 1) * 100
        lines.append(f"  {k:7s} : {n:>12,}  ({pct:5.2f}%)")
    lines.append("")
    lines.append("-- slug_lang_hint top 20 --")
    for lang, n in sorted(lang_ct.items(), key=lambda x: -x[1])[:20]:
        pct = n / max(total[0], 1) * 100
        lines.append(f"  {lang:5s} : {n:>12,}  ({pct:5.2f}%)")
    lines.append("")
    lines.append("-- slug_quality × year --")
    years = sorted({y for (y, _) in year_ct.keys()})
    qs = ["good", "fair", "opaque", "empty"]
    header = "year     " + "".join(f"{q:>12s}" for q in qs) + "        total"
    lines.append(header)
    for y in years:
        cells = []
        t = 0
        for q in qs:
            n = year_ct.get((y, q), 0)
            cells.append(f"{n:>12,}")
            t += n
        lines.append(y.ljust(9) + "".join(cells) + f"   {t:>12,}")
    lines.append("")

    import duckdb
    con2 = duckdb.connect()
    for qv in ["good", "fair", "opaque"]:
        lines.append(f"-- sample {qv} (n=20) --")
        samp = con2.execute(f"""
            SELECT slug_lang_hint, slug_text
            FROM read_parquet('{OUT}')
            WHERE slug_quality = '{qv}'
            LIMIT 20
        """).fetchdf()
        for _, r in samp.iterrows():
            st = str(r["slug_text"])[:100] if r["slug_text"] else ""
            lines.append(f"  [{r['slug_lang_hint']}] {st}")
        lines.append("")

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    log.info("Report written to %s", REPORT)
    print("\n".join(lines[:80]))


if __name__ == "__main__":
    main()
