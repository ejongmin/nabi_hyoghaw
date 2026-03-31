#!/usr/bin/env python3
"""
GDELT BigQuery — 중국어 도메인(.cn) 뉴스 추가 수집
===================================================
기존 collect_gdelt_bigquery.py가 영문 V2Organizations 매칭에 의존하여
중국 기업 뉴스가 극히 적음 (CATL 2,868건 vs BMW 1,456,813건).

이 스크립트는:
1) source_domain LIKE '%.cn' 뉴스에서 중국어 기업명 매칭
2) 중국 주요 재경 매체(sina.com.cn, 163.com, qq.com 등) 필터
3) URL에 종목코드(300750, 002460 등) 포함된 기사 수집

Colab에서 실행: BQ 인증 필요
  python scripts/collect_gdelt_cn_domains.py --project-id nabi-effect-project

로컬 실행: gcloud auth 필요
  gcloud auth application-default login
  python scripts/collect_gdelt_cn_domains.py
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

try:
    from google.cloud import bigquery
except ImportError:
    print("pip install google-cloud-bigquery db-dtypes pyarrow")
    sys.exit(1)

log = logging.getLogger("gdelt_cn")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

# ── 중국 주요 재경 매체 ──
CN_FINANCE_DOMAINS = [
    "sina.com.cn", "finance.sina.com.cn",
    "163.com", "money.163.com",
    "qq.com", "finance.qq.com",
    "sohu.com", "business.sohu.com",
    "eastmoney.com",
    "10jqka.com.cn",
    "hexun.com",
    "caixin.com",
    "yicai.com",
    "cls.cn",
    "stcn.com",
    "cnstock.com",
    "securities.com",
    "nbd.com.cn",
    "jiemian.com",
    "36kr.com",
    "thepaper.cn",
    "bjnews.com.cn",
    "ce.cn",
    "people.com.cn",
    "xinhuanet.com",
    "chinadaily.com.cn",
    "cnevpost.com",
]

# ── 기업 종목코드 + 중문명 ──
CN_COMPANIES = {
    "300750.SZ": {"code": "300750", "cn": "宁德时代", "en": "CATL"},
    "3931.HK":   {"code": "3931", "cn": "中创新航", "en": "CALB"},
    "300014.SZ": {"code": "300014", "cn": "亿纬锂能", "en": "EVE Energy"},
    "002074.SZ": {"code": "002074", "cn": "国轩高科", "en": "Gotion"},
    "300207.SZ": {"code": "300207", "cn": "欣旺达", "en": "Sunwoda"},
    "300037.SZ": {"code": "300037", "cn": "新宙邦", "en": "Capchem"},
    "920185.BJ": {"code": "920185", "cn": "贝特瑞", "en": "BTR"},
    "300919.SZ": {"code": "300919", "cn": "中伟股份", "en": "CNGR"},
    "002340.SZ": {"code": "002340", "cn": "格林美", "en": "GEM"},
    "603799.SH": {"code": "603799", "cn": "华友钴业", "en": "Huayou"},
    "688388.SH": {"code": "688388", "cn": "嘉元科技", "en": "Jiayuan"},
    "600110.SH": {"code": "600110", "cn": "诺德股份", "en": "Nuode"},
    "603659.SH": {"code": "603659", "cn": "璞泰来", "en": "Putailai"},
    "002460.SZ": {"code": "002460", "cn": "赣锋锂业", "en": "Ganfeng"},
    "002466.SZ": {"code": "002466", "cn": "天齐锂业", "en": "Tianqi"},
    "603993.SH": {"code": "603993", "cn": "洛阳钼业", "en": "CMOC"},
    "600362.SH": {"code": "600362", "cn": "江西铜业", "en": "Jiangxi Copper"},
    "1211.HK":   {"code": "1211", "cn": "比亚迪", "en": "BYD"},
    "0175.HK":   {"code": "0175", "cn": "吉利汽车", "en": "Geely"},
    "000625.SZ": {"code": "000625", "cn": "长安汽车", "en": "Changan"},
    "600006.SH": {"code": "600006", "cn": "东风汽车", "en": "Dongfeng"},
    "601238.SH": {"code": "601238", "cn": "广汽集团", "en": "GAC"},
    "601633.SH": {"code": "601633", "cn": "长城汽车", "en": "GWM"},
    "600418.SH": {"code": "600418", "cn": "江淮汽车", "en": "JAC"},
    "600733.SH": {"code": "600733", "cn": "北汽蓝谷", "en": "BAIC"},
}


def build_stock_code_regex() -> str:
    """URL에서 종목코드 매칭을 위한 BigQuery regex"""
    codes = [info["code"] for info in CN_COMPANIES.values()]
    return "|".join(codes)


def build_cn_domain_filter() -> str:
    """중국 재경 매체 도메인 필터"""
    patterns = [f"'{d}'" for d in CN_FINANCE_DOMAINS]
    return ", ".join(patterns)


def build_query_stock_in_url(date_start: str, date_end: str, limit: int = 200000) -> str:
    """
    전략 1: URL에 종목코드가 포함된 .cn 도메인 기사.
    예: https://finance.sina.com.cn/stock/s/300750.html
    """
    stock_regex = build_stock_code_regex()
    return f"""
    SELECT
        GKGRECORDID AS gkg_id,
        DATE AS gkg_date,
        SourceCommonName AS source_domain,
        DocumentIdentifier AS article_url,
        V2Organizations AS v2_organizations,
        V2Themes AS v2_themes,
        V2Tone AS v2_tone,
        V2Locations AS v2_locations,
        'cn_stock_url' AS collection_type
    FROM `gdelt-bq.gdeltv2.gkg_partitioned`
    WHERE
        _PARTITIONTIME >= TIMESTAMP('{date_start}')
        AND _PARTITIONTIME < TIMESTAMP('{date_end}')
        -- URL에 종목코드 포함
        AND REGEXP_CONTAINS(
            IFNULL(DocumentIdentifier, ''),
            r'({stock_regex})'
        )
        -- .cn 도메인 또는 주요 중국 매체
        AND (
            REGEXP_CONTAINS(LOWER(IFNULL(SourceCommonName, '')), r'\\.cn$|\\.com\\.cn$')
            OR LOWER(SourceCommonName) IN ({build_cn_domain_filter()})
            OR REGEXP_CONTAINS(LOWER(IFNULL(DocumentIdentifier, '')), r'sina\\.com|163\\.com|qq\\.com|sohu\\.com|eastmoney|10jqka|hexun')
        )
    LIMIT {limit}
    """


def build_query_cn_org_match(date_start: str, date_end: str, limit: int = 200000) -> str:
    """
    전략 2: V2Organizations에서 중국 기업 영문명 매칭 + .cn 도메인 확장.
    기존 쿼리에서 .cn 도메인을 더 적극적으로 포함.
    AllNames 필드도 활용하여 매칭율 향상.
    """
    # Build regex for English names
    en_names = []
    for info in CN_COMPANIES.values():
        en_names.append(info["en"])
    en_regex = "|".join([f"\\\\b{n}\\\\b" for n in en_names])

    return f"""
    SELECT
        GKGRECORDID AS gkg_id,
        DATE AS gkg_date,
        SourceCommonName AS source_domain,
        DocumentIdentifier AS article_url,
        V2Organizations AS v2_organizations,
        V2Themes AS v2_themes,
        V2Tone AS v2_tone,
        V2Locations AS v2_locations,
        AllNames AS all_names,
        'cn_domain_org' AS collection_type
    FROM `gdelt-bq.gdeltv2.gkg_partitioned`
    WHERE
        _PARTITIONTIME >= TIMESTAMP('{date_start}')
        AND _PARTITIONTIME < TIMESTAMP('{date_end}')
        AND (
            -- V2Organizations에서 기업 매칭
            REGEXP_CONTAINS(
                IFNULL(V2Organizations, ''),
                r'(?i)({en_regex})'
            )
            OR
            -- AllNames에서 기업 매칭
            REGEXP_CONTAINS(
                IFNULL(AllNames, ''),
                r'(?i)({en_regex})'
            )
        )
        -- .cn 도메인 또는 중국어 소스
        AND (
            REGEXP_CONTAINS(LOWER(IFNULL(SourceCommonName, '')), r'\\.cn|sina|163\\.com|qq\\.com|sohu|eastmoney|caixin|yicai|thepaper')
            OR REGEXP_CONTAINS(LOWER(IFNULL(DocumentIdentifier, '')), r'\\.cn/|sina\\.com|163\\.com|qq\\.com|sohu\\.com')
        )
    LIMIT {limit}
    """


def build_query_battery_cn(date_start: str, date_end: str, limit: int = 100000) -> str:
    """
    전략 3: 중국 도메인에서 배터리 관련 키워드가 포함된 기사.
    V2Themes에서 battery, lithium, EV 키워드 매칭.
    """
    return f"""
    SELECT
        GKGRECORDID AS gkg_id,
        DATE AS gkg_date,
        SourceCommonName AS source_domain,
        DocumentIdentifier AS article_url,
        V2Organizations AS v2_organizations,
        V2Themes AS v2_themes,
        V2Tone AS v2_tone,
        V2Locations AS v2_locations,
        AllNames AS all_names,
        'cn_battery_theme' AS collection_type
    FROM `gdelt-bq.gdeltv2.gkg_partitioned`
    WHERE
        _PARTITIONTIME >= TIMESTAMP('{date_start}')
        AND _PARTITIONTIME < TIMESTAMP('{date_end}')
        -- 배터리/EV 관련 테마
        AND REGEXP_CONTAINS(
            CONCAT(IFNULL(V2Themes,''), ' ', IFNULL(DocumentIdentifier,'')),
            r'(?i)(lithium|battery|EV battery|electric vehicle|cathode|anode|electrolyte|separator|cobalt|nickel sulfate)'
        )
        -- 중국 소스
        AND (
            REGEXP_CONTAINS(LOWER(IFNULL(SourceCommonName, '')), r'\\.cn|sina|163|qq\\.com|sohu|eastmoney|caixin')
            OR REGEXP_CONTAINS(LOWER(IFNULL(DocumentIdentifier, '')), r'\\.cn/')
        )
    LIMIT {limit}
    """


def main():
    parser = argparse.ArgumentParser(description="GDELT BigQuery 중국 도메인 뉴스 수집")
    parser.add_argument("--project-id", default="nabi-effect-project")
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default="data/raw/chinese_news/gdelt_cn_domains.parquet")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("GDELT BigQuery — 중국 도메인 뉴스 수집")
    log.info(f"  Project: {args.project_id}")
    log.info(f"  Period: {args.start_year}~{args.end_year}")
    log.info(f"  Companies: {len(CN_COMPANIES)}")
    log.info("=" * 60)

    # Build queries for each half-year
    queries = []
    for year in range(args.start_year, args.end_year + 1):
        for half in [(f"{year}-01-01", f"{year}-07-01"), (f"{year}-07-01", f"{year+1}-01-01")]:
            ds, de = half
            if year == args.end_year and half[0].endswith("07-01"):
                de = f"{year}-12-31"

            queries.append(("stock_url", ds, de,
                          build_query_stock_in_url(ds, de)))
            queries.append(("cn_org", ds, de,
                          build_query_cn_org_match(ds, de)))
            queries.append(("battery_cn", ds, de,
                          build_query_battery_cn(ds, de)))

    log.info(f"Total queries to run: {len(queries)}")

    if args.dry_run:
        for qtype, ds, de, q in queries[:3]:
            log.info(f"\n--- {qtype} ({ds} ~ {de}) ---")
            log.info(q[:500] + "...")
        log.info(f"\n[DRY-RUN] Would run {len(queries)} queries")
        return

    # Execute queries
    client = bigquery.Client(project=args.project_id)
    all_dfs = []

    for i, (qtype, ds, de, query) in enumerate(queries):
        log.info(f"\n[{i+1}/{len(queries)}] {qtype} ({ds} ~ {de})")
        try:
            qjob = client.query(query)
            rows = qjob.result()
            df = rows.to_dataframe()
            log.info(f"  → {len(df):,} rows")
            if not df.empty:
                df["query_type"] = qtype
                df["period"] = f"{ds}~{de}"
                all_dfs.append(df)
        except Exception as e:
            log.error(f"  Error: {e}")
            continue

    if not all_dfs:
        log.warning("No results!")
        return

    merged = pd.concat(all_dfs, ignore_index=True)
    log.info(f"\nTotal before dedup: {len(merged):,}")

    # Dedup by article_url
    merged = merged.drop_duplicates(subset=["article_url"], keep="first").reset_index(drop=True)
    log.info(f"After dedup: {len(merged):,}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)
    log.info(f"\n✅ Saved: {output_path} ({len(merged):,} rows)")

    # Stats
    log.info("\n=== Query Type Breakdown ===")
    for qt, g in merged.groupby("query_type"):
        log.info(f"  {qt:<20}: {len(g):>8,}")

    log.info("\n=== Top 20 Domains ===")
    dom_counts = merged["source_domain"].value_counts().head(20)
    for d, c in dom_counts.items():
        log.info(f"  {d:<30}: {c:>8,}")


if __name__ == "__main__":
    main()
