#!/usr/bin/env python3
"""GDELT BQ — AllNames 필드 확장 매칭 (전략 1)"""
import os, logging, hashlib
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\Users\john9\Downloads\nabi-490508-e7bc326096dc.json'
from google.cloud import bigquery
import pandas as pd
from pathlib import Path

log = logging.getLogger("allnames")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

# 뉴스 부족 기업 영문 별칭 (AllNames에서 매칭)
TARGETS = {
    # Chinese materials/cell (most underrepresented)
    "Putailai": "603659.SH", "Nuode": "600110.SH", "Jiayuan": "688388.SH",
    "CNGR": "300919.SZ", "Sunwoda": "300207.SZ", "Huayou Cobalt": "603799.SH",
    "Huayou": "603799.SH", "Capchem": "300037.SZ",
    "EVE Energy": "300014.SZ", "Gotion": "002074.SZ", "Guoxuan": "002074.SZ",
    "CALB": "3931.HK", "CATL": "300750.SZ", "Contemporary Amperex": "300750.SZ",
    "Ganfeng": "002460.SZ", "Ganfeng Lithium": "002460.SZ",
    "Tianqi": "002466.SZ", "Tianqi Lithium": "002466.SZ",
    "CMOC": "603993.SH", "China Molybdenum": "603993.SH",
    "Jiangxi Copper": "600362.SH", "BTR": "920185.BJ",
    "GEM Co": "002340.SZ",
    # Korean underrepresented
    "EcoPro": "247540.KQ", "EcoPro BM": "247540.KQ",
    "L&F": "066970.KS", "LnF": "066970.KS",
    "POSCO Future": "003670.KS", "POSCO Chemical": "003670.KS",
    # Others
    "Li-Cycle": "LICY", "Li Cycle": "LICY",
    "SQM": "SQM", "Sociedad Quimica": "SQM",
}

# Build regex
import re
aliases = sorted(TARGETS.keys(), key=len, reverse=True)
regex = "|".join([r"\b" + re.escape(a) + r"\b" for a in aliases])

client = bigquery.Client()
log.info(f"Connected: {client.project}")
log.info(f"Target aliases: {len(aliases)}")

all_dfs = []
for year in range(2019, 2026):
    for ds, de in [(f"{year}-01-01", f"{year}-07-01"), (f"{year}-07-01", f"{year+1}-01-01" if year < 2025 else "2025-12-31")]:
        query = f"""
        SELECT
            GKGRECORDID AS gkg_id,
            DATE AS gkg_date,
            SourceCommonName AS source_domain,
            DocumentIdentifier AS article_url,
            V2Organizations AS v2_organizations,
            AllNames AS all_names,
            V2Tone AS v2_tone,
            'allnames_boost' AS collection_type
        FROM `gdelt-bq.gdeltv2.gkg_partitioned`
        WHERE
            _PARTITIONTIME >= TIMESTAMP('{ds}')
            AND _PARTITIONTIME < TIMESTAMP('{de}')
            AND REGEXP_CONTAINS(
                IFNULL(AllNames, ''),
                r'(?i)({regex})'
            )
            AND NOT REGEXP_CONTAINS(
                IFNULL(V2Organizations, ''),
                r'(?i)({regex})'
            )
            AND NOT REGEXP_CONTAINS(
                LOWER(IFNULL(SourceCommonName, '')),
                r'reddit|twitter|x\.com|facebook|youtube|wikipedia|blogspot'
            )
        LIMIT 30000
        """
        log.info(f"  {ds}~{de}")
        try:
            job = client.query(query)
            df_q = job.to_dataframe()
            log.info(f"    → {len(df_q):,} rows")
            if not df_q.empty:
                all_dfs.append(df_q)
        except Exception as e:
            log.error(f"    Error: {e}")

if not all_dfs:
    log.warning("No results!")
    exit()

merged = pd.concat(all_dfs, ignore_index=True)
merged = merged.drop_duplicates(subset=["article_url"], keep="first")
log.info(f"\nTotal unique: {len(merged):,}")

# Extract entity_ids from AllNames
def extract_entities(all_names):
    if pd.isna(all_names): return []
    found = set()
    text = str(all_names).upper()
    for alias, cid in TARGETS.items():
        if alias.upper() in text:
            found.add(cid)
    return list(found)

merged["entity_ids"] = merged["all_names"].apply(extract_entities)
merged = merged[merged["entity_ids"].apply(len) > 0]
log.info(f"With entities: {len(merged):,}")

# Convert to risk_events format
def parse_tone(t):
    if pd.isna(t): return 2.0, 0.0
    parts = str(t).split(",")
    try:
        tone = float(parts[0])
        neg = float(parts[2]) if len(parts) > 2 else 0.0
        return min(5.0, 1.0 + neg * 0.5), max(-1.0, min(1.0, tone / 10.0))
    except: return 2.0, 0.0

sev = merged["v2_tone"].apply(parse_tone)
result = pd.DataFrame({
    "event_id": merged["article_url"].apply(lambda u: hashlib.md5(str(u).encode()).hexdigest()),
    "event_time": pd.to_datetime(merged["gkg_date"], format="%Y%m%d%H%M%S", errors="coerce"),
    "url": merged["article_url"],
    "title": "",
    "risk_types": "other",
    "severity": sev.apply(lambda x: x[0]),
    "entity_ids": merged["entity_ids"].values,
    "entity_scores": merged["entity_ids"].apply(lambda x: [1.0]*len(x)),
    "finbert": sev.apply(lambda x: x[1]),
    "country_ids": [["China"]] * len(merged),
    "collection_type": "gdelt_allnames_boost",
    "source_domain": merged["source_domain"],
})
result["event_time"] = result["event_time"].dt.tz_localize(None)

# Stats
exp = result.explode("entity_ids")
log.info("\n=== Per-Company ===")
for cid, cnt in exp.groupby("entity_ids").size().sort_values(ascending=False).items():
    log.info(f"  {str(cid):<14}: {cnt:>7,}")

out = Path("data/raw/chinese_news/gdelt_allnames_boost.parquet")
out.parent.mkdir(parents=True, exist_ok=True)
result.to_parquet(out, index=False)
log.info(f"\n✅ Saved: {out} ({len(result):,} rows)")
