#!/usr/bin/env python3
"""GDELT BQ — V2Organizations + .cn 도메인 확장 (전략 2)"""
import os, logging, hashlib
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\Users\john9\Downloads\nabi-490508-e7bc326096dc.json'
from google.cloud import bigquery
import pandas as pd
from pathlib import Path

log = logging.getLogger("cn_v2")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

CN_COMPANIES = {
    "300750.SZ": ["CATL","Contemporary Amperex"],
    "3931.HK": ["CALB","China Aviation Lithium"],
    "300014.SZ": ["EVE Energy"],
    "002074.SZ": ["Gotion","Guoxuan","Gotion High-Tech"],
    "300207.SZ": ["Sunwoda"],
    "300037.SZ": ["Capchem","Shenzhen Capchem"],
    "920185.BJ": ["BTR","BTR New Material"],
    "300919.SZ": ["CNGR","CNGR Advanced"],
    "002340.SZ": ["GEM Co"],
    "603799.SH": ["Huayou","Huayou Cobalt","Zhejiang Huayou"],
    "688388.SH": ["Jiayuan","Jiayuan Technology"],
    "600110.SH": ["Nuode"],
    "603659.SH": ["Putailai"],
    "002460.SZ": ["Ganfeng","Ganfeng Lithium"],
    "002466.SZ": ["Tianqi","Tianqi Lithium"],
    "603993.SH": ["CMOC","China Molybdenum","China Moly"],
    "600362.SH": ["Jiangxi Copper"],
    "1211.HK": ["BYD"],
    "0175.HK": ["Geely"],
    "000625.SZ": ["Changan"],
    "600006.SH": ["Dongfeng"],
    "601238.SH": ["GAC","Guangzhou Automobile"],
    "601633.SH": ["Great Wall Motor","GWM"],
    "600418.SH": ["JAC","Anhui Jianghuai"],
    "600733.SH": ["BAIC","Beijing Automotive"],
}

import re
all_aliases = []
alias_to_cid = {}
for cid, aliases in CN_COMPANIES.items():
    for a in aliases:
        all_aliases.append(a)
        alias_to_cid[a.upper()] = cid
all_aliases.sort(key=len, reverse=True)
regex = "|".join([r"\b" + re.escape(a) + r"\b" for a in all_aliases])

client = bigquery.Client()
log.info(f"Connected: {client.project}")

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
            V2Tone AS v2_tone,
            'cn_v2_org' AS collection_type
        FROM `gdelt-bq.gdeltv2.gkg_partitioned`
        WHERE
            _PARTITIONTIME >= TIMESTAMP('{ds}')
            AND _PARTITIONTIME < TIMESTAMP('{de}')
            AND REGEXP_CONTAINS(
                IFNULL(V2Organizations, ''),
                r'(?i)({regex})'
            )
            AND (
                REGEXP_CONTAINS(LOWER(IFNULL(SourceCommonName, '')), r'\.cn|sina|163\.com|qq\.com|sohu|eastmoney|caixin|yicai|thepaper|hexun|10jqka')
                OR REGEXP_CONTAINS(LOWER(IFNULL(DocumentIdentifier, '')), r'\.cn/')
            )
        LIMIT 50000
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
    log.warning("No results!"); exit()

merged = pd.concat(all_dfs, ignore_index=True)
merged = merged.drop_duplicates(subset=["article_url"], keep="first")
log.info(f"\nTotal unique: {len(merged):,}")

def extract_entities(v2orgs):
    if pd.isna(v2orgs): return []
    found = set()
    for entry in str(v2orgs).split(";"):
        name = entry.rsplit(",",1)[0].strip().upper() if "," in entry else entry.strip().upper()
        for alias, cid in alias_to_cid.items():
            if alias in name:
                found.add(cid)
    return list(found)

merged["entity_ids"] = merged["v2_organizations"].apply(extract_entities)
merged = merged[merged["entity_ids"].apply(len) > 0]
log.info(f"With entities: {len(merged):,}")

def parse_tone(t):
    if pd.isna(t): return 2.0, 0.0
    parts = str(t).split(",")
    try:
        tone = float(parts[0]); neg = float(parts[2]) if len(parts)>2 else 0.0
        return min(5.0, 1.0+neg*0.5), max(-1.0, min(1.0, tone/10.0))
    except: return 2.0, 0.0

sev = merged["v2_tone"].apply(parse_tone)
result = pd.DataFrame({
    "event_id": merged["article_url"].apply(lambda u: hashlib.md5(str(u).encode()).hexdigest()),
    "event_time": pd.to_datetime(merged["gkg_date"], format="%Y%m%d%H%M%S", errors="coerce"),
    "url": merged["article_url"], "title": "", "risk_types": "other",
    "severity": sev.apply(lambda x: x[0]),
    "entity_ids": merged["entity_ids"].values,
    "entity_scores": merged["entity_ids"].apply(lambda x: [1.0]*len(x)),
    "finbert": sev.apply(lambda x: x[1]),
    "country_ids": [["China"]]*len(merged),
    "collection_type": "gdelt_cn_v2_org",
    "source_domain": merged["source_domain"],
})
result["event_time"] = result["event_time"].dt.tz_localize(None)

exp = result.explode("entity_ids")
log.info("\n=== Per-Company ===")
for cid, cnt in exp.groupby("entity_ids").size().sort_values(ascending=False).items():
    log.info(f"  {str(cid):<14}: {cnt:>7,}")

out = Path("data/raw/chinese_news/gdelt_cn_v2_org.parquet")
result.to_parquet(out, index=False)
log.info(f"\n✅ Saved: {out} ({len(result):,} rows)")
