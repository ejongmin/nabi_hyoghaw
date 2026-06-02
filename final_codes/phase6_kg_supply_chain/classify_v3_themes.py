#!/usr/bin/env python3
"""
classify_v3_themes.py  (V3 — hash-dedup, zero external dependency)
==================================================================
Phase 1: raw GKG → hash-based dedup → classify → lookup parquet
  - URL full string 대신 int hash로 dedup (1.1M entries = ~50MB vs ~200MB)
  - ParquetWriter 즉시 flush → 메모리 축적 없음
Phase 2: DuckDB LEFT JOIN (lookup + risk_events_final_v2) → output
Phase 3: 보충데이터 title 재분류
"""

import gc, math, re, time, shutil
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

BASE = Path(__file__).resolve().parent.parent
THEME_MAP = BASE / "configs" / "gdelt_theme_mapping.yaml"
KW_PATH   = BASE / "configs" / "risk_keywords.yaml"
INPUT     = BASE / "data" / "processed" / "risk_events_final_v2.parquet"
OUTPUT    = BASE / "data" / "processed" / "risk_events_classified_v3.parquet"
LOOKUP    = BASE / "data" / "processed" / "_theme_lookup_v3.parquet"
TMP_DIR   = Path("/sessions/fervent-tender-clarke/tmp")

RAW_GKG = {
    2022: BASE / "data" / "raw" / "gdelt" / "gdelt_gkg_2022.parquet",
    2023: BASE / "data" / "raw" / "gdelt" / "gdelt_gkg_2023.parquet",
    2024: BASE / "data" / "raw" / "gdelt" / "gdelt_gkg_2024.parquet",
    2025: BASE / "data" / "raw" / "gdelt" / "gdelt_gkg_2025.parquet",
}

S_MAX = 30.0; MIN_W = 2; MAX_CATS = 4; BATCH = 20_000
RTYPES = ["geopolitics","logistics","natural","regulatory",
          "market","labor","technology","esg","pandemic"]

LOOKUP_SCHEMA = pa.schema([
    ("url", pa.string()),
    ("risk_types_v3", pa.string()),
    ("severity_v3", pa.float64()),
    ("conf_v3", pa.float64()),
])


def load_theme_mapping():
    with open(THEME_MAP) as f:
        cfg = yaml.safe_load(f)
    m = {}
    for rt in RTYPES:
        if rt in cfg and "themes" in cfg[rt]:
            m[rt] = [(t["prefix"].upper(), t["weight"]) for t in cfg[rt]["themes"]]
        else:
            m[rt] = []
    return m


def classify_themes(themes_str, tmap):
    if not themes_str or pd.isna(themes_str):
        return None
    codes = set()
    for p in str(themes_str).split(";"):
        p = p.strip()
        if "," in p:
            c = p.rsplit(",", 1)[0].strip().upper()
            if c and len(c) > 1:
                codes.add(c)
    if not codes:
        return None
    cat_w = {}
    for rt, pfxs in tmap.items():
        w = 0
        for pfx, weight in pfxs:
            for c in codes:
                if c.startswith(pfx):
                    w += weight; break
        if w >= MIN_W:
            cat_w[rt] = w
    if not cat_w:
        return None
    top = sorted(cat_w.items(), key=lambda x: -x[1])[:MAX_CATS]
    rt = ",".join(sorted(c for c, _ in top))
    tw = sum(w for _, w in top)
    sev = round(min(math.log(1+tw)/math.log(1+S_MAX), 1.0), 4)
    conf = round(min(0.3 + 0.1*len(top) + 0.01*len(codes), 1.0), 3)
    return (rt, sev, conf)


# ═══════════════════════════════════════
# Phase 1: hash-dedup + classify → lookup
# ═══════════════════════════════════════
def build_lookup(tmap):
    """NO dedup in Python — 모든 분류 결과를 디스크에 즉시 flush.
    DuckDB JOIN 시 GROUP BY로 dedup 처리."""
    print("\n" + "="*60)
    print("Phase 1: stream classify → lookup (no Python set)")
    print("="*60)

    LOOKUP.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(str(LOOKUP), LOOKUP_SCHEMA, compression="snappy")
    total = 0; hit = 0

    for year, path in sorted(RAW_GKG.items()):
        if not path.exists():
            print(f"  [SKIP] {year}"); continue

        t0 = time.time()
        pf = pq.ParquetFile(path)
        yr_total = 0; yr_hit = 0
        buf_url, buf_rt, buf_sev, buf_conf = [], [], [], []

        for batch in pf.iter_batches(batch_size=BATCH,
                                     columns=["article_url", "v2_themes"]):
            urls_raw = batch.column("article_url").to_pylist()
            themes_raw = batch.column("v2_themes").to_pylist()

            for i in range(len(urls_raw)):
                u = urls_raw[i]
                if not u:
                    continue
                yr_total += 1
                result = classify_themes(themes_raw[i], tmap)
                if result is None:
                    continue
                rt, sev, conf = result
                buf_url.append(u)
                buf_rt.append(rt)
                buf_sev.append(sev)
                buf_conf.append(conf)
                yr_hit += 1

                # 10K마다 디스크 flush → 메모리 축적 0
                if len(buf_url) >= 10_000:
                    _flush(writer, buf_url, buf_rt, buf_sev, buf_conf)
                    buf_url, buf_rt, buf_sev, buf_conf = [], [], [], []

            del urls_raw, themes_raw
            gc.collect()

        if buf_url:
            _flush(writer, buf_url, buf_rt, buf_sev, buf_conf)
            buf_url, buf_rt, buf_sev, buf_conf = [], [], [], []
        gc.collect()

        pct = yr_hit / max(yr_total, 1) * 100
        print(f"  {year}: {yr_total:,} rows → {yr_hit:,} classified "
              f"({pct:.1f}%) [{time.time()-t0:.0f}s]")
        total += yr_total; hit += yr_hit

    writer.close()
    gc.collect()

    pct = hit / max(total, 1) * 100
    sz = LOOKUP.stat().st_size / 1024**2
    print(f"\n  Total: {hit:,} classified rows (with dupes) → {sz:.0f} MB")
    print(f"  DuckDB will dedup during JOIN")


def _flush(writer, urls, rts, sevs, confs):
    tbl = pa.table({
        "url": pa.array(urls, type=pa.string()),
        "risk_types_v3": pa.array(rts, type=pa.string()),
        "severity_v3": pa.array(sevs, type=pa.float64()),
        "conf_v3": pa.array(confs, type=pa.float64()),
    })
    writer.write_table(tbl)
    del tbl


# ═══════════════════════════════════════
# Phase 2: DuckDB JOIN
# ═══════════════════════════════════════
def duckdb_join():
    print("\n" + "="*60)
    print("Phase 2: DuckDB JOIN")
    print("="*60)

    TMP_DIR.mkdir(exist_ok=True)

    q = f"""
    SELECT
        e.event_id, e.event_time, e.url, e.title,
        CASE
            WHEN EXTRACT(YEAR FROM e.event_time)>=2022
                 AND e.collection_type IN ('company','risk_only','')
            THEN COALESCE(l.risk_types_v3, e.risk_types)
            ELSE e.risk_types
        END AS risk_types,
        CASE
            WHEN EXTRACT(YEAR FROM e.event_time)>=2022
                 AND e.collection_type IN ('company','risk_only','')
            THEN COALESCE(l.severity_v3, e.severity)
            ELSE e.severity
        END AS severity,
        e.element, e.alias, e.company_id, e.score,
        CAST(e.finbert AS DOUBLE) AS finbert,
        e.country_ids, e.tone, e.source_domain, e.collection_type,
        CASE
            WHEN EXTRACT(YEAR FROM e.event_time)>=2022
                 AND e.collection_type IN ('company','risk_only','')
                 AND l.risk_types_v3 IS NOT NULL THEN 'gdelt_themes'
            WHEN EXTRACT(YEAR FROM e.event_time)>=2022
                 AND e.collection_type IN ('company','risk_only','')
            THEN 'v2themes_aho'
            WHEN e.collection_type NOT IN ('company','risk_only','')
            THEN 'title_keyword'
            ELSE 'legacy'
        END AS classify_source,
        CASE
            WHEN EXTRACT(YEAR FROM e.event_time)>=2022
                 AND e.collection_type IN ('company','risk_only','')
                 AND l.conf_v3 IS NOT NULL THEN l.conf_v3
            WHEN EXTRACT(YEAR FROM e.event_time)>=2022
                 AND e.collection_type IN ('company','risk_only','')
            THEN CASE WHEN e.risk_types!='other' THEN 0.5 ELSE 0.0 END
            WHEN e.collection_type NOT IN ('company','risk_only','')
            THEN 0.3
            WHEN e.risk_types!='other' THEN 0.15
            ELSE 0.0
        END AS classify_conf
    FROM read_parquet('{INPUT}') e
    LEFT JOIN (
        SELECT url, FIRST(risk_types_v3) AS risk_types_v3,
               FIRST(severity_v3) AS severity_v3, FIRST(conf_v3) AS conf_v3
        FROM read_parquet('{LOOKUP}')
        GROUP BY url
    ) l ON e.url = l.url
    """

    con = duckdb.connect(str(TMP_DIR / "join.duckdb"))
    con.execute("SET threads=1")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET memory_limit='2GB'")
    con.execute(f"SET temp_directory='{TMP_DIR}'")

    t0 = time.time()
    print("  Running JOIN...")
    con.execute(f"COPY ({q}) TO '{OUTPUT}' (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 100000)")
    elapsed = time.time() - t0
    sz = OUTPUT.stat().st_size / 1024**2
    print(f"  Done in {elapsed:.0f}s ({sz:.0f} MB)")
    con.close()

    # cleanup duckdb file
    for f in TMP_DIR.glob("join.duckdb*"):
        f.unlink(missing_ok=True)


# ═══════════════════════════════════════
# Phase 3: 보충데이터 title 재분류
# ═══════════════════════════════════════
def reclassify_suppl():
    print("\n  Phase 3: supplementary title reclassification...")
    with open(KW_PATH) as f:
        cfg = yaml.safe_load(f)
    mega = {}; wts = {}
    for rt in RTYPES:
        if rt not in cfg or not isinstance(cfg[rt], dict):
            mega[rt] = None; wts[rt] = 0; continue
        kws = cfg[rt].get("keywords", [])
        ps = []; ws = []
        for kw in kws:
            p = kw.get("pattern", "")
            try: re.compile(p, re.I); ps.append(f"(?:{p})"); ws.append(kw.get("weight", 3))
            except: pass
        if ps:
            mega[rt] = re.compile("|".join(ps), re.I)
            wts[rt] = sum(ws) / len(ws)
        else:
            mega[rt] = None; wts[rt] = 0

    pf = pq.ParquetFile(OUTPUT)
    temp = OUTPUT.parent / "_temp_v3.parquet"
    writer = pq.ParquetWriter(str(temp), pf.schema_arrow, compression="snappy")
    n_hit = 0

    for batch in pf.iter_batches(batch_size=50_000):
        df = batch.to_pandas()
        mask = df["classify_source"] == "title_keyword"
        if mask.sum() > 0:
            idx = df.index[mask]
            titles = df.loc[idx, "title"].fillna("").astype(str)
            has_t = titles.str.len() >= 5
            for rt in RTYPES:
                if mega[rt] is None: continue
                hit = titles[has_t].str.contains(mega[rt], na=False)
                for i in hit.index[hit]:
                    cur = str(df.at[i, "risk_types"])
                    if cur == "other":
                        df.at[i, "risk_types"] = rt
                    elif rt not in cur:
                        cs = cur.split(",") + [rt]
                        df.at[i, "risk_types"] = ",".join(sorted(set(cs))[:MAX_CATS])
                    df.at[i, "severity"] = max(float(df.at[i, "severity"]),
                        round(min(math.log(1+wts[rt])/math.log(1+S_MAX), 1.0), 4))
                    df.at[i, "classify_conf"] = max(float(df.at[i, "classify_conf"]), 0.4)
                    n_hit += 1
        writer.write_table(pa.Table.from_pandas(df, schema=pf.schema_arrow,
                                                preserve_index=False))
        del df; gc.collect()
    writer.close()
    shutil.move(str(temp), str(OUTPUT))
    print(f"  Supplementary: {n_hit:,} keyword hits")


# ═══════════════════════════════════════
# Stats
# ═══════════════════════════════════════
def stats():
    print("\n" + "="*60)
    print("V3 Classification Results")
    print("="*60)

    con = duckdb.connect(str(TMP_DIR / "stats.duckdb"))
    con.execute("SET threads=1")
    con.execute(f"SET memory_limit='200MB'")
    con.execute(f"SET temp_directory='{TMP_DIR}'")

    o = str(OUTPUT)
    total = con.execute(f"SELECT count(*) FROM read_parquet('{o}')").fetchone()[0]
    other = con.execute(f"SELECT count(*) FROM read_parquet('{o}') WHERE risk_types='other'").fetchone()[0]
    cls = total - other
    print(f"\n  Total:      {total:,}")
    print(f"  Classified: {cls:,} ({cls/total*100:.1f}%)")
    print(f"  Other:      {other:,} ({other/total*100:.1f}%)")

    print(f"\n  Per-category:")
    for rt in RTYPES:
        n = con.execute(f"SELECT count(*) FROM read_parquet('{o}') WHERE risk_types LIKE '%{rt}%'").fetchone()[0]
        print(f"    {rt:15s}: {n:>10,} ({n/total*100:.2f}%)")

    print(f"\n  By source:")
    r = con.execute(f"""SELECT classify_source,count(*) n,
        sum(CASE WHEN risk_types!='other' THEN 1 ELSE 0 END) hit,
        round(avg(classify_conf),3) c
        FROM read_parquet('{o}') GROUP BY 1 ORDER BY n DESC""").fetchdf()
    for _, row in r.iterrows():
        s = row["classify_source"]; n = int(row["n"]); h = int(row["hit"]); c = row["c"]
        print(f"    {s:20s}: {n:>10,} | classified {h:,} ({h/max(n,1)*100:.1f}%) | conf={c:.3f}")

    print(f"\n  Top risk_types:")
    r2 = con.execute(f"SELECT risk_types,count(*) n FROM read_parquet('{o}') GROUP BY 1 ORDER BY n DESC LIMIT 15").fetchdf()
    for _, row in r2.iterrows():
        print(f"    {row['risk_types']:35s}: {int(row['n']):>10,}")

    con.close()
    for f in TMP_DIR.glob("stats.duckdb*"):
        f.unlink(missing_ok=True)


def main():
    t0 = time.time()
    print("="*60)
    print("V3 — GDELT Theme Code Mapping")
    print("="*60)
    tmap = load_theme_mapping()
    print(f"  Theme prefixes: {sum(len(v) for v in tmap.values())}")

    build_lookup(tmap)
    gc.collect()
    duckdb_join()
    gc.collect()
    reclassify_suppl()
    stats()

    if LOOKUP.exists():
        LOOKUP.unlink()
    print(f"\n✓ V3 complete in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
