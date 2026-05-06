#!/usr/bin/env python3
"""
recollect_2018_2021_themes.py
=============================
2018-2021 legacy 행(6.48M)의 V2Themes를 BigQuery GDELT GKG에서 수집 후 분류.

흐름:
  1) risk_events_classified_v3.parquet에서 legacy 2018-2021 URL 추출
  2) 반기별로 BQ 쿼리 → DocumentIdentifier+V2Themes 수집
  3) URL 매칭 + theme 분류 → lookup parquet 생성
  4) DuckDB JOIN으로 v3 파일 업데이트

사전 조건:
  gcloud auth application-default login
"""

import gc, math, time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from google.cloud import bigquery

BASE = Path(__file__).resolve().parent.parent
THEME_MAP = BASE / "configs" / "gdelt_theme_mapping.yaml"
V3_FILE   = BASE / "data" / "processed" / "risk_events_classified_v3.parquet"
LOOKUP    = BASE / "data" / "processed" / "_theme_lookup_2018_2021.parquet"
OUTPUT    = BASE / "data" / "processed" / "risk_events_classified_v3.parquet"  # overwrite
TMP_DIR   = BASE / "data" / "processed" / "_tmp"

S_MAX = 30.0; MIN_W = 2; MAX_CATS = 4
RTYPES = ["geopolitics","logistics","natural","regulatory",
          "market","labor","technology","esg","pandemic"]

LOOKUP_SCHEMA = pa.schema([
    ("url", pa.string()),
    ("risk_types_v3", pa.string()),
    ("severity_v3", pa.float64()),
    ("conf_v3", pa.float64()),
])

# 반기별 쿼리 범위 (2018 H1 ~ 2021 H2)
HALF_YEARS = [
    ("2018H1", "2018-01-01", "2018-06-30"),
    ("2018H2", "2018-07-01", "2018-12-31"),
    ("2019H1", "2019-01-01", "2019-06-30"),
    ("2019H2", "2019-07-01", "2019-12-31"),
    ("2020H1", "2020-01-01", "2020-06-30"),
    ("2020H2", "2020-07-01", "2020-12-31"),
    ("2021H1", "2021-01-01", "2021-06-30"),
    ("2021H2", "2021-07-01", "2021-12-31"),
]


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
    """V2Themes 문자열 → (risk_types, severity, conf) or None"""
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
# Step 1: Extract legacy 2018-2021 URLs
# ═══════════════════════════════════════
def extract_legacy_urls():
    print("\n" + "="*60)
    print("Step 1: Extract legacy 2018-2021 URLs")
    print("="*60)

    pf = pq.ParquetFile(V3_FILE)
    urls = set()
    for i in range(pf.metadata.num_row_groups):
        df = pf.read_row_group(i, columns=["event_time", "classify_source", "url"]).to_pandas()
        legacy = df[df["classify_source"] == "legacy"]
        yr = pd.to_datetime(legacy["event_time"]).dt.year
        mask = yr.isin([2018, 2019, 2020, 2021])
        for u in legacy.loc[mask, "url"].dropna():
            if len(str(u)) > 5:
                urls.add(str(u))
        del df, legacy
        gc.collect()

    print(f"  Unique legacy URLs: {len(urls):,}")
    return urls


# ═══════════════════════════════════════
# Step 2: Upload URLs to BQ temp table + query per half-year
# ═══════════════════════════════════════
def upload_urls_to_bq(client, urls):
    """URL을 BQ temp dataset에 올려서 서버사이드 JOIN 가능하게 함."""
    dataset_id = f"{client.project}.nabi_tmp"

    # temp dataset 생성 (없으면)
    from google.cloud.bigquery import Dataset
    ds = Dataset(dataset_id)
    ds.location = "US"
    ds.default_table_expiration_ms = 3600_000 * 24  # 24h 후 자동삭제
    try:
        client.create_dataset(ds, exists_ok=True)
    except Exception as e:
        print(f"    Dataset creation warning: {e}")

    table_id = f"{dataset_id}.legacy_urls"
    print(f"  Uploading {len(urls):,} URLs to {table_id}...")

    # DataFrame으로 변환 후 업로드
    df = pd.DataFrame({"url": list(urls)})
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        schema=[bigquery.SchemaField("url", "STRING")],
    )
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()  # wait
    print(f"  Uploaded: {job.output_rows:,} rows")
    del df
    gc.collect()
    return table_id


def query_bq_themes(urls, tmap):
    print("\n" + "="*60)
    print("Step 2: BigQuery V2Themes collection + classification")
    print("="*60)

    client = bigquery.Client()
    url_table = upload_urls_to_bq(client, urls)
    del urls  # 메모리 해제 — BQ에 올렸으니 더 이상 불필요
    gc.collect()

    LOOKUP.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(str(LOOKUP), LOOKUP_SCHEMA, compression="snappy")
    total_matched = 0; total_classified = 0

    for label, start, end in HALF_YEARS:
        t0 = time.time()
        print(f"\n  [{label}] {start} ~ {end}")

        # BQ 서버사이드 JOIN: 우리 URL과 매칭되는 것만 반환
        query = f"""
        SELECT g.DocumentIdentifier AS url, g.V2Themes AS themes
        FROM `gdelt-bq.gdeltv2.gkg_partitioned` g
        INNER JOIN `{url_table}` u ON g.DocumentIdentifier = u.url
        WHERE DATE(g._PARTITIONTIME) BETWEEN '{start}' AND '{end}'
          AND g.V2Themes IS NOT NULL
          AND g.V2Themes != ''
        """

        # dry-run으로 비용 확인
        job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        dry = client.query(query, job_config=job_config)
        gb = dry.total_bytes_processed / 1024**3
        print(f"    Scan estimate: {gb:.1f} GB")

        # 실제 실행
        job_config = bigquery.QueryJobConfig(use_query_cache=True)
        result = client.query(query, job_config=job_config)

        matched = 0; classified = 0
        buf_url, buf_rt, buf_sev, buf_conf = [], [], [], []

        for row in result:
            matched += 1
            res = classify_themes(row["themes"], tmap)
            if res is None:
                continue
            rt, sev, conf = res
            buf_url.append(row["url"])
            buf_rt.append(rt)
            buf_sev.append(sev)
            buf_conf.append(conf)
            classified += 1

            if len(buf_url) >= 10_000:
                _flush(writer, buf_url, buf_rt, buf_sev, buf_conf)
                buf_url, buf_rt, buf_sev, buf_conf = [], [], [], []

        if buf_url:
            _flush(writer, buf_url, buf_rt, buf_sev, buf_conf)
            buf_url, buf_rt, buf_sev, buf_conf = [], [], [], []

        elapsed = time.time() - t0
        pct = classified / max(matched, 1) * 100
        print(f"    Matched: {matched:,} | Classified: {classified:,} ({pct:.1f}%) | {elapsed:.0f}s")
        total_matched += matched
        total_classified += classified
        gc.collect()

    writer.close()

    # temp table 정리
    try:
        client.delete_table(url_table, not_found_ok=True)
        print(f"  Cleaned up: {url_table}")
    except:
        pass

    print(f"\n  Total: matched={total_matched:,} classified={total_classified:,}")
    sz = LOOKUP.stat().st_size / 1024**2
    print(f"  Lookup file: {sz:.0f} MB")


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
# Step 3: DuckDB JOIN to update V3 file
# ═══════════════════════════════════════
def duckdb_join():
    print("\n" + "="*60)
    print("Step 3: DuckDB JOIN — update legacy rows")
    print("="*60)

    import duckdb

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    temp_out = TMP_DIR / "v3_updated.parquet"

    q = f"""
    SELECT
        e.event_id, e.event_time, e.url, e.title,
        CASE
            WHEN e.classify_source = 'legacy'
                 AND EXTRACT(YEAR FROM e.event_time) BETWEEN 2018 AND 2021
                 AND l.risk_types_v3 IS NOT NULL
            THEN l.risk_types_v3
            ELSE e.risk_types
        END AS risk_types,
        CASE
            WHEN e.classify_source = 'legacy'
                 AND EXTRACT(YEAR FROM e.event_time) BETWEEN 2018 AND 2021
                 AND l.severity_v3 IS NOT NULL
            THEN l.severity_v3
            ELSE e.severity
        END AS severity,
        e.element, e.alias, e.company_id, e.score,
        CAST(e.finbert AS DOUBLE) AS finbert,
        e.country_ids, e.tone, e.source_domain, e.collection_type,
        CASE
            WHEN e.classify_source = 'legacy'
                 AND EXTRACT(YEAR FROM e.event_time) BETWEEN 2018 AND 2021
                 AND l.risk_types_v3 IS NOT NULL
            THEN 'bq_themes'
            ELSE e.classify_source
        END AS classify_source,
        CASE
            WHEN e.classify_source = 'legacy'
                 AND EXTRACT(YEAR FROM e.event_time) BETWEEN 2018 AND 2021
                 AND l.conf_v3 IS NOT NULL
            THEN l.conf_v3
            ELSE e.classify_conf
        END AS classify_conf
    FROM read_parquet('{V3_FILE}') e
    LEFT JOIN (
        SELECT url, FIRST(risk_types_v3) AS risk_types_v3,
               FIRST(severity_v3) AS severity_v3, FIRST(conf_v3) AS conf_v3
        FROM read_parquet('{LOOKUP}')
        GROUP BY url
    ) l ON e.url = l.url
    """

    con = duckdb.connect(str(TMP_DIR / "join.duckdb"))
    con.execute("SET threads=2")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET memory_limit='3GB'")
    con.execute(f"SET temp_directory='{TMP_DIR}'")

    t0 = time.time()
    print("  Running JOIN...")
    con.execute(f"COPY ({q}) TO '{temp_out}' (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 100000)")
    elapsed = time.time() - t0
    sz = temp_out.stat().st_size / 1024**2
    print(f"  Done in {elapsed:.0f}s ({sz:.0f} MB)")
    con.close()

    # Replace original
    import shutil
    backup = V3_FILE.with_suffix(".parquet.bak")
    if V3_FILE.exists():
        shutil.copy2(str(V3_FILE), str(backup))
        print(f"  Backup: {backup.name}")
    shutil.move(str(temp_out), str(OUTPUT))
    print(f"  Updated: {OUTPUT.name}")

    # Cleanup
    for f in TMP_DIR.glob("join.duckdb*"):
        f.unlink(missing_ok=True)


# ═══════════════════════════════════════
# Stats
# ═══════════════════════════════════════
def stats():
    print("\n" + "="*60)
    print("Updated V3 Results")
    print("="*60)

    import duckdb
    con = duckdb.connect()
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

    print(f"\n  By classify_source:")
    r = con.execute(f"""
        SELECT classify_source, count(*) n,
            sum(CASE WHEN risk_types!='other' THEN 1 ELSE 0 END) hit
        FROM read_parquet('{o}') GROUP BY 1 ORDER BY n DESC
    """).fetchdf()
    for _, row in r.iterrows():
        s = row["classify_source"]; n = int(row["n"]); h = int(row["hit"])
        print(f"    {s:20s}: {n:>10,} | classified {h:,} ({h/max(n,1)*100:.1f}%)")

    print(f"\n  By year:")
    r2 = con.execute(f"""
        SELECT EXTRACT(YEAR FROM event_time) y, count(*) n,
            sum(CASE WHEN risk_types!='other' THEN 1 ELSE 0 END) hit
        FROM read_parquet('{o}') GROUP BY 1 ORDER BY 1
    """).fetchdf()
    for _, row in r2.iterrows():
        y = int(row["y"]); n = int(row["n"]); h = int(row["hit"])
        print(f"    {y}: {n:>10,} | classified {h:,} ({h/max(n,1)*100:.1f}%)")

    con.close()


def main():
    t0 = time.time()
    print("="*60)
    print("Recollect 2018-2021 V2Themes from BigQuery")
    print("="*60)

    tmap = load_theme_mapping()
    print(f"  Theme prefixes: {sum(len(v) for v in tmap.values())}")

    urls = extract_legacy_urls()
    query_bq_themes(urls, tmap)
    gc.collect()
    duckdb_join()
    gc.collect()
    stats()

    # Cleanup lookup
    if LOOKUP.exists():
        LOOKUP.unlink()
        print(f"  Cleaned up: {LOOKUP.name}")

    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
