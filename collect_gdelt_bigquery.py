#!/usr/bin/env python3
"""
나비효과 프로젝트 — GDELT BigQuery 수집기 v3
=============================================
GKG(V2Organizations) + Events(EventCode/GoldsteinScale) JOIN 방식
3년치(2022-2024) 뉴스를 수 분 만에 수집.

핵심 전략:
  1) GKG 파티셔닝 테이블에서 V2Organizations로 40개 기업 REGEXP 매칭 (기업 식별)
  2) Events 파티셔닝 테이블에서 SOURCEURL로 JOIN → EventCode, GoldsteinScale 확보
  3) V2Tone + GoldsteinScale + AvgTone → severity 계산 (FinBERT 불필요)
  4) V2Themes + risk_keywords.yaml → 리스크 분류
  5) V2Locations → 국가 매칭

BigQuery 비용 최적화:
  - gkg_partitioned / events_partitioned 사용 → 파티션 pruning으로 비용 최소화
  - 반기별 분할 쿼리 → 메모리/쿼타 안전
  - 3년 기준 예상: ~200-300GB (무료 1TB/월 이내)

사용법:
  # 전체 3년 수집 (2022-2024)
  python collect_gdelt_bigquery.py

  # 특정 기간만
  python collect_gdelt_bigquery.py --start-year 2024 --end-year 2024

  # dry-run (쿼리만 확인, 비용 추정)
  python collect_gdelt_bigquery.py --dry-run

  # 체크포인트에서 이어서
  python collect_gdelt_bigquery.py --resume

  # GKG만 수집 (Events JOIN 없이)
  python collect_gdelt_bigquery.py --table gkg

출력: data/raw/gdelt/gdelt_gkg_3y.parquet
"""

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml

try:
    from google.cloud import bigquery
except ImportError:
    print("google-cloud-bigquery 패키지가 필요합니다:")
    print("  pip install google-cloud-bigquery db-dtypes pyarrow")
    sys.exit(1)

# ──────────────────────── Config ────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_ID = "nabi-effect-project"  # GCP 프로젝트 ID (변경 필요 시 --project-id)
UNIVERSE_PATH = SCRIPT_DIR / "data" / "universe" / "univers_final.csv"
KEYWORDS_PATH = SCRIPT_DIR / "configs" / "risk_keywords.yaml"
OUTPUT_DIR = SCRIPT_DIR / "data" / "raw" / "gdelt"
CHECKPOINT_DIR = OUTPUT_DIR / "_checkpoints_bq"

DEFAULT_START_YEAR = 2022
DEFAULT_END_YEAR = 2024

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gdelt_bq")


# ──────────────────────── Company Aliases ────────────────────────

def build_company_aliases() -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """
    collect_gdelt_gkg.py의 166개 alias와 동일한 패턴을 BigQuery REGEXP용으로 구축.

    Returns:
        company_aliases: {company_id: [alias1, alias2, ...]}
        alias_to_cid: {alias_lower: company_id}
    """
    df = pd.read_csv(UNIVERSE_PATH)

    EXTRA_ALIASES: Dict[str, List[str]] = {
        # ── Cell/Pack ──
        "300750.SZ": ["CATL", "Contemporary Amperex", "Contemporary Amperex Technology",
                       "Contemporary Amperex Technology Co"],
        "373220.KS": ["LG Energy", "LGES", "LG Energy Solution", "LG Energy Solutions"],
        "1211.HK": ["BYD", "BYD Company", "BYD Auto", "BYD Co"],
        "051910.KS": ["LG Chem", "LG Chemical", "LG Chem Ltd"],
        "300014.SZ": ["EVE Energy", "EVE Energy Co"],
        "300207.SZ": ["Sunwoda", "Sunwoda Electronic", "Sunwoda Electronic Co"],
        "3931.HK": ["CALB", "CALB Group", "CALB Co",
                     "China Aviation Lithium Battery", "China Aviation Lithium"],
        "002074.SZ": ["Gotion", "Gotion High-Tech", "Guoxuan", "Guoxuan High-Tech"],

        # ── Materials ──
        "247540.KQ": ["EcoPro", "EcoPro BM", "EcoPro Co"],
        "066970.KS": ["L&F", "L and F", "LnF", "LF Co", "L\\&F"],
        "003670.KS": ["POSCO Future", "POSCO Future M", "POSCO Chemical", "POSCO Holdings"],
        "300919.SZ": ["CNGR", "CNGR Advanced Material", "CNGR Advanced"],
        "603659.SH": ["Putailai", "Putailai New Energy"],
        "300037.SZ": ["Capchem", "Shenzhen Capchem", "Capchem Technology"],
        "600110.SH": ["Nuode", "Nuode Investment", "Nuode New Material"],
        "688388.SH": ["Jiayuan", "Jiayuan International", "Jiayuan Technology"],
        "002340.SZ": ["GEM Co", "GEM Co Ltd"],
        "920185.BJ": ["BTR", "BTR New Material", "BTR New Energy"],
        "3407.T": ["Asahi Kasei", "Asahi Kasei Corp", "Asahi Kasei Corporation"],

        # ── Upstream/Refining ──
        "002460.SZ": ["Ganfeng", "Ganfeng Lithium", "Ganfeng Lithium Co",
                       "Jiangxi Ganfeng", "Jiangxi Ganfeng Lithium"],
        "002466.SZ": ["Tianqi", "Tianqi Lithium", "Tianqi Lithium Corp",
                       "Sichuan Tianqi", "Tianqi Lithium Industries"],
        "603799.SH": ["Huayou", "Huayou Cobalt", "Zhejiang Huayou",
                       "Huayou Cobalt Co", "Zhejiang Huayou Cobalt"],
        "603993.SH": ["CMOC", "CMOC Group", "China Moly", "China Molybdenum",
                       "CMOC Group Limited"],
        "600362.SH": ["Jiangxi Copper", "Jiangxi Copper Company",
                       "Jiangxi Copper Co", "Jiangxi Copper Corp"],
        "ALB": ["Albemarle", "Albemarle Corp", "Albemarle Corporation"],
        "SQM": ["SQM", "Sociedad Quimica y Minera", "Sociedad Quimica"],
        "GLEN.L": ["Glencore", "Glencore PLC", "Glencore International"],

        # ── OEM ──
        "BMW.DE": ["BMW", "Bayerische Motoren", "BMW Group", "BMW AG"],
        "VOW3.DE": ["Volkswagen", "VW", "Volkswagen AG", "Volkswagen Group", "VW Group"],
        "MBG.DE": ["Mercedes-Benz", "Mercedes Benz", "Daimler", "Mercedes-Benz Group"],
        "TSLA": ["Tesla", "Tesla Motors", "Tesla Inc"],
        "F": ["Ford Motor", "Ford Motor Company"],
        "GM": ["General Motors", "General Motors Company"],
        "LCID": ["Lucid Motors", "Lucid Group", "Lucid"],
        "RIVN": ["Rivian", "Rivian Automotive"],
        "7267.T": ["Honda Motor", "Honda", "Honda Motor Co"],
        "7203.T": ["Toyota Motor", "Toyota", "Toyota Motor Corp"],
        "0175.HK": ["Geely", "Geely Automobile", "Geely Auto", "Zhejiang Geely"],
        "600733.SH": ["BAIC", "BAIC BluePark", "BAIC Motor", "BAIC Group",
                       "Beijing Automotive"],
        "000625.SZ": ["Changan", "Changan Auto", "Changan Automobile",
                       "Chongqing Changan"],
        "600006.SH": ["Dongfeng", "Dongfeng Motor", "Dongfeng Motor Corp",
                       "Dongfeng Motor Group", "Dongfeng Auto"],
        "601238.SH": ["GAC", "GAC Group", "GAC Motor", "Guangzhou Automobile",
                       "GAC Aion", "Guangzhou Auto"],
        "601633.SH": ["Great Wall Motor", "Great Wall Motors", "GWM",
                       "Great Wall Motor Company"],
        "600418.SH": ["JAC Motors", "JAC", "Anhui Jianghuai", "JAC Automobile"],
        "LICY": ["Li-Cycle", "Li Cycle", "Li-Cycle Holdings"],

        # ── Chemical ──
        "BAS.DE": ["BASF", "BASF SE", "BASF AG", "BASF Corporation"],
    }

    company_aliases: Dict[str, List[str]] = {}
    alias_to_cid: Dict[str, str] = {}

    # Universe 기업 처리
    universe_ids = set()
    for _, row in df.iterrows():
        cid = str(row["company_id"]).strip()
        cname = str(row.get("canonical_name", "")).strip()
        universe_ids.add(cid)

        aliases = set()
        if cname and cname != "nan":
            aliases.add(cname)

        for a in EXTRA_ALIASES.get(cid, []):
            aliases.add(a)

        aliases = {a for a in aliases if len(a) >= 3}
        if aliases:
            company_aliases[cid] = sorted(aliases)
            for a in aliases:
                alias_to_cid[a.lower()] = cid

    # EXTRA_ALIASES에만 있는 기업 (TSLA, RIVN 등)
    for cid, alias_list in EXTRA_ALIASES.items():
        if cid not in universe_ids:
            valid = [a for a in alias_list if len(a) >= 3]
            if valid:
                company_aliases[cid] = valid
                for a in valid:
                    alias_to_cid[a.lower()] = cid

    total_aliases = sum(len(v) for v in company_aliases.values())
    log.info(f"Company aliases: {len(company_aliases)} companies, {total_aliases} patterns")
    return company_aliases, alias_to_cid


def build_bq_company_regex(company_aliases: Dict[str, List[str]]) -> str:
    """
    BigQuery REGEXP_CONTAINS용 패턴 생성.
    모든 alias를 word-boundary로 감싸서 OR 연결.
    """
    all_patterns = []
    for cid, aliases in company_aliases.items():
        for alias in aliases:
            # BigQuery RE2는 \b를 지원하므로 word-boundary 사용
            escaped = re.escape(alias)
            all_patterns.append(rf"\b{escaped}\b")

    # 긴 패턴부터 매칭 (greedy 방지)
    all_patterns.sort(key=len, reverse=True)
    return "|".join(all_patterns)


# ──────────────────────── Risk Keywords ────────────────────────

def build_bq_risk_regex() -> str:
    """risk_keywords.yaml에서 BigQuery REGEXP용 리스크 키워드 패턴 생성"""
    try:
        with open(KEYWORDS_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception:
        log.warning("risk_keywords.yaml 로드 실패, fallback 사용")
        return r"sanction|embargo|tariff|trade war|blockade|earthquake|tsunami|flood"

    all_patterns = []
    for risk_type in ["geopolitics", "logistics", "natural"]:
        section = data.get(risk_type, {})
        for kw in section.get("keywords", []):
            if isinstance(kw, dict) and "pattern" in kw:
                all_patterns.append(kw["pattern"])

    all_patterns.sort(key=len, reverse=True)
    log.info(f"Risk patterns: {len(all_patterns)}")
    return "|".join(all_patterns)


# ──────────────────────── BigQuery Queries ────────────────────────

# ── Type B: 순수 리스크 뉴스 (기업 언급 불필요) ──

def build_risk_only_query(
    company_regex: str,
    risk_regex: str,
    date_start: str,
    date_end: str,
    limit: int = 150000,
) -> str:
    """
    Type B 수집 (중복 제거 버전): 리스크 매칭 O + 기업 매칭 X인 기사만.
    Type A(기업+리스크)와 겹치지 않도록 기업 매칭을 NOT 조건으로.
    V2Tone의 부정적 톤이 강한 순서로 정렬하여 상위 limit건만 수집.
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
        TRUE AS risk_matched,
        'risk_only' AS collection_type
    FROM `gdelt-bq.gdeltv2.gkg_partitioned`
    WHERE
        _PARTITIONTIME >= TIMESTAMP('{date_start}')
        AND _PARTITIONTIME < TIMESTAMP('{date_end}')
        -- 리스크 키워드 매칭 (Themes + URL)
        AND REGEXP_CONTAINS(
            CONCAT(IFNULL(V2Themes,''), ' ', IFNULL(DocumentIdentifier,'')),
            r'(?i)({risk_regex})'
        )
        -- 기업 매칭 안 되는 것만 (Type A와 중복 방지)
        AND NOT REGEXP_CONTAINS(
            IFNULL(V2Organizations, ''),
            r'(?i)({company_regex})'
        )
        AND NOT REGEXP_CONTAINS(
            LOWER(IFNULL(SourceCommonName, '')),
            r'reddit\\.com|twitter\\.com|x\\.com|facebook\\.com|youtube\\.com|wikipedia\\.org|blogspot\\.com|wordpress\\.com'
        )
    ORDER BY SAFE_CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64) ASC
    LIMIT {limit}
    """


def build_gkg_only_query(
    company_regex: str,
    risk_regex: str,
    date_start: str,
    date_end: str,
) -> str:
    """
    GKG 파티셔닝 테이블에서 기업 매칭 기사 추출.
    V2Organizations + V2Tone + V2Themes + V2Locations 포함.
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
        AllNames AS all_names
    FROM `gdelt-bq.gdeltv2.gkg_partitioned`
    WHERE
        _PARTITIONTIME >= TIMESTAMP('{date_start}')
        AND _PARTITIONTIME < TIMESTAMP('{date_end}')
        AND REGEXP_CONTAINS(
            IFNULL(V2Organizations, ''),
            r'(?i)({company_regex})'
        )
        AND NOT REGEXP_CONTAINS(
            LOWER(IFNULL(SourceCommonName, '')),
            r'reddit\\.com|twitter\\.com|x\\.com|facebook\\.com|youtube\\.com|wikipedia\\.org|blogspot\\.com|wordpress\\.com'
        )
    """


def build_gkg_events_join_query(
    company_regex: str,
    risk_regex: str,
    date_start: str,
    date_end: str,
) -> str:
    """
    GKG + Events JOIN 쿼리.
    GKG의 V2Organizations로 기업 매칭 → Events의 EventCode/GoldsteinScale 확보.
    SOURCEURL = DocumentIdentifier로 JOIN.
    """
    return f"""
    WITH matched_gkg AS (
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
            -- 리스크 매칭
            CASE
                WHEN REGEXP_CONTAINS(
                    CONCAT(IFNULL(V2Themes,''), ' ', IFNULL(DocumentIdentifier,'')),
                    r'(?i)({risk_regex})'
                ) THEN TRUE ELSE FALSE
            END AS risk_matched
        FROM `gdelt-bq.gdeltv2.gkg_partitioned`
        WHERE
            _PARTITIONTIME >= TIMESTAMP('{date_start}')
            AND _PARTITIONTIME < TIMESTAMP('{date_end}')
            AND REGEXP_CONTAINS(
                IFNULL(V2Organizations, ''),
                r'(?i)({company_regex})'
            )
            AND NOT REGEXP_CONTAINS(
                LOWER(IFNULL(SourceCommonName, '')),
                r'reddit\\.com|twitter\\.com|x\\.com|facebook\\.com|youtube\\.com|wikipedia\\.org|blogspot\\.com|wordpress\\.com'
            )
    ),
    matched_events AS (
        SELECT
            SOURCEURL,
            ARRAY_AGG(STRUCT(
                EventCode,
                EventRootCode,
                QuadClass,
                GoldsteinScale,
                AvgTone AS event_avg_tone,
                NumArticles
            ) ORDER BY NumArticles DESC LIMIT 1)[OFFSET(0)] AS evt
        FROM `gdelt-bq.gdeltv2.events_partitioned`
        WHERE
            _PARTITIONTIME >= TIMESTAMP('{date_start}')
            AND _PARTITIONTIME < TIMESTAMP('{date_end}')
            AND SOURCEURL IS NOT NULL
            AND SOURCEURL != ''
        GROUP BY SOURCEURL
    )
    SELECT
        g.gkg_id,
        g.gkg_date,
        g.source_domain,
        g.article_url,
        g.v2_organizations,
        g.v2_themes,
        g.v2_tone,
        g.v2_locations,
        g.all_names,
        g.risk_matched,
        -- Events 데이터 (있으면)
        e.evt.EventCode AS event_code,
        e.evt.EventRootCode AS event_root_code,
        e.evt.QuadClass AS quad_class,
        e.evt.GoldsteinScale AS goldstein_scale,
        e.evt.event_avg_tone AS event_avg_tone,
        e.evt.NumArticles AS num_articles
    FROM matched_gkg g
    LEFT JOIN matched_events e
        ON g.article_url = e.SOURCEURL
    """


# ──────────────────────── Post-processing ────────────────────────

def post_process(
    df: pd.DataFrame,
    company_aliases: Dict[str, List[str]],
    alias_to_cid: Dict[str, str],
) -> pd.DataFrame:
    """
    BigQuery 결과 후처리:
    1) V2Organizations에서 company_ids 추출
    2) V2Tone 파싱 → tone, severity
    3) V2Locations에서 country_ids 추출
    4) risk_types 분류
    """
    log.info(f"Post-processing {len(df):,} rows...")

    # ── Company ID 추출 ──
    company_patterns = []
    for cid, aliases in company_aliases.items():
        for alias in aliases:
            if len(alias) >= 3:
                try:
                    pat = re.compile(r'\b' + re.escape(alias) + r'\b', re.IGNORECASE)
                    company_patterns.append((pat, cid))
                except re.error:
                    pass

    def extract_company_ids(v2orgs):
        if pd.isna(v2orgs) or not v2orgs:
            return ""
        # V2Organizations: "OrgName,offset;OrgName,offset;..."
        org_names = set()
        for entry in str(v2orgs).split(";"):
            if "," in entry:
                name = entry.rsplit(",", 1)[0].strip()
                if name:
                    org_names.add(name.upper())
        if not org_names:
            return ""
        text = " | ".join(org_names)
        hits = set()
        for pat, cid in company_patterns:
            if pat.search(text):
                hits.add(cid)
        return ";".join(sorted(hits))

    df["company_ids"] = df["v2_organizations"].apply(extract_company_ids)

    # ── V2Tone 파싱 ──
    def parse_tone(v2tone):
        if pd.isna(v2tone) or not v2tone:
            return 0.0
        try:
            return float(str(v2tone).split(",")[0])
        except (ValueError, IndexError):
            return 0.0

    df["tone"] = df["v2_tone"].apply(parse_tone)

    # ── Severity 계산 (GoldsteinScale + Tone 결합) ──
    import numpy as np

    def compute_severity(row):
        tone = row["tone"]
        gs = row.get("goldstein_scale", None)

        if pd.notna(gs):
            # Events JOIN 성공 → GoldsteinScale + Tone 결합
            norm_gs = np.clip((10 - float(gs)) / 20.0, 0, 1)
            norm_tone = np.clip((10 - tone) / 20.0, 0, 1)
            severity = 1.0 + norm_gs * 2.5 + norm_tone * 1.5
        else:
            # Events JOIN 실패 → Tone만으로 계산
            norm_tone = np.clip((10 - tone) / 20.0, 0, 1)
            severity = 1.0 + norm_tone * 4.0

        return round(float(np.clip(severity, 1.0, 5.0)), 3)

    df["severity"] = df.apply(compute_severity, axis=1)

    # ── Country 추출 (V2Locations) ──
    FIPS_MAP = {
        "CI": "Chile", "AS": "Australia", "ID": "Indonesia",
        "CG": "Congo", "CF": "Congo",
        "CH": "China", "KS": "South_Korea", "GM": "Germany",
        "JA": "Japan", "US": "USA", "UK": "UK", "CA": "Canada",
    }

    def extract_countries(v2loc):
        if pd.isna(v2loc) or not v2loc:
            return ""
        hits = set()
        for entry in str(v2loc).split(";"):
            if "," in entry:
                entry = entry.rsplit(",", 1)[0]
            parts = entry.split("#")
            if len(parts) >= 3:
                fips = parts[2].strip()
                if fips in FIPS_MAP:
                    hits.add(FIPS_MAP[fips])
        return ";".join(sorted(hits))

    df["country_ids"] = df["v2_locations"].apply(extract_countries)

    # ── Date 정리 ──
    df["date"] = df["gkg_date"].astype(str).str[:8]

    # ── 불필요 컬럼 정리 ──
    drop_cols = ["v2_organizations", "v2_themes", "v2_tone", "v2_locations",
                 "all_names", "gkg_date"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # company_ids가 비어있는 행 제거 (오탐 방지)
    df = df[df["company_ids"].str.strip() != ""]

    log.info(f"Post-processing complete: {len(df):,} rows retained")
    return df


def post_process_risk_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    Type B (리스크 전용) 기사 후처리.
    기업 매칭 없음 → company_ids = "" (빈값).
    Route Node 전파 시 V2Locations + risk_type으로 연결됨.
    """
    import numpy as np
    log.info(f"Post-processing Type B (risk-only) {len(df):,} rows...")

    # company_ids는 비어있음 (기업 언급 없는 기사)
    df["company_ids"] = ""

    # V2Tone 파싱
    def parse_tone(v2tone):
        if pd.isna(v2tone) or not v2tone:
            return 0.0
        try:
            return float(str(v2tone).split(",")[0])
        except (ValueError, IndexError):
            return 0.0

    df["tone"] = df["v2_tone"].apply(parse_tone)

    # Severity (Tone 기반만, Events JOIN 없으므로)
    df["severity"] = df["tone"].apply(
        lambda t: round(float(np.clip(1.0 + np.clip((10 - t) / 20.0, 0, 1) * 4.0, 1.0, 5.0)), 3)
    )

    # Country 추출
    FIPS_MAP = {
        "CI": "Chile", "AS": "Australia", "ID": "Indonesia",
        "CG": "Congo", "CF": "Congo",
        "CH": "China", "KS": "South_Korea", "GM": "Germany",
        "JA": "Japan", "US": "USA", "UK": "UK", "CA": "Canada",
    }

    def extract_countries(v2loc):
        if pd.isna(v2loc) or not v2loc:
            return ""
        hits = set()
        for entry in str(v2loc).split(";"):
            if "," in entry:
                entry = entry.rsplit(",", 1)[0]
            parts = entry.split("#")
            if len(parts) >= 3:
                fips = parts[2].strip()
                if fips in FIPS_MAP:
                    hits.add(FIPS_MAP[fips])
        return ";".join(sorted(hits))

    df["country_ids"] = df["v2_locations"].apply(extract_countries)

    # Date
    df["date"] = df["gkg_date"].astype(str).str[:8]

    # 컬럼 정리
    drop_cols = ["v2_organizations", "v2_themes", "v2_tone", "v2_locations",
                 "all_names", "gkg_date"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    log.info(f"Type B post-processing complete: {len(df):,} rows")
    return df


# ──────────────────────── Collection Logic ────────────────────────

def generate_date_ranges(start_year: int, end_year: int) -> List[Tuple[str, str, str]]:
    """분기별 날짜 범위 생성 (Q1~Q4)"""
    ranges = []
    today = datetime.now().strftime("%Y-%m-%d")

    for year in range(start_year, end_year + 1):
        quarters = [
            (f"{year}_Q1", f"{year}-01-01", f"{year}-04-01"),
            (f"{year}_Q2", f"{year}-04-01", f"{year}-07-01"),
            (f"{year}_Q3", f"{year}-07-01", f"{year}-10-01"),
            (f"{year}_Q4", f"{year}-10-01", f"{year + 1}-01-01"),
        ]
        for label, q_start, q_end in quarters:
            if q_start <= today:
                ranges.append((label, q_start, min(q_end, today)))

    return ranges


def _run_query(client, query, label, dry_run=False, timeout=1800):
    """단일 쿼리 실행 헬퍼 (timeout: 초, 기본 1800초=30분)"""
    if dry_run:
        job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        try:
            job = client.query(query, job_config=job_config)
            bytes_est = job.total_bytes_processed
            gb_est = bytes_est / (1024**3)
            log.info(f"  [{label}] Estimated: {gb_est:.1f} GB ({bytes_est:,} bytes)")
            return gb_est
        except Exception as e:
            log.error(f"  [{label}] Dry-run error: {e}")
            return 0.0
    else:
        job = client.query(query)
        job.result(timeout=timeout)  # 쿼리 완료 대기 (기본 1800초)
        # BigQuery Storage API 자동 사용 (google-cloud-bigquery-storage 설치 시)
        try:
            df = job.to_dataframe(create_bqstorage_client=True, progress_bar_type=None)
        except Exception:
            log.warning(f"  [{label}] BQ Storage API fallback → REST")
            df = job.to_dataframe(create_bqstorage_client=False)
        log.info(f"  [{label}] Raw rows: {len(df):,}")
        return df


def collect(
    client: bigquery.Client,
    company_regex: str,
    risk_regex: str,
    company_aliases: Dict[str, List[str]],
    alias_to_cid: Dict[str, str],
    date_ranges: List[Tuple[str, str, str]],
    use_join: bool = True,
    include_risk_only: bool = True,
    dry_run: bool = False,
    resume: bool = False,
):
    """메인 수집 루프.

    이중 수집 전략:
      Type A: 기업 매칭 기사 (GKG V2Organizations REGEXP)
      Type B: 순수 리스크 뉴스 (기업 언급 없음, 리스크 키워드만)
              → Route Node를 통한 간접 전파에 사용
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_company_dfs = []  # Type A
    all_risk_dfs = []     # Type B
    total_dry_gb = 0.0

    for label, date_start, date_end in date_ranges:
        # ── Type A: 기업 매칭 ──
        cp_file_a = CHECKPOINT_DIR / f"gkg_{label}.parquet"

        if resume and cp_file_a.exists():
            log.info(f"[SKIP] Type A {label} — checkpoint exists")
            all_company_dfs.append(pd.read_parquet(cp_file_a))
        else:
            if use_join:
                query_a = build_gkg_events_join_query(
                    company_regex, risk_regex, date_start, date_end
                )
            else:
                query_a = build_gkg_only_query(
                    company_regex, risk_regex, date_start, date_end
                )

            if dry_run:
                log.info(f"[DRY-RUN] Type A {label}: {date_start} ~ {date_end}")
                total_dry_gb += _run_query(client, query_a, f"A-{label}", dry_run=True)
            else:
                log.info(f"[QUERY] Type A {label}: {date_start} ~ {date_end}")
                try:
                    df_a = _run_query(client, query_a, f"A-{label}")
                    if not df_a.empty:
                        df_a["collection_type"] = "company"
                        df_a.to_parquet(cp_file_a, index=False)
                        all_company_dfs.append(df_a)
                        if "goldstein_scale" in df_a.columns:
                            join_rate = df_a["goldstein_scale"].notna().mean() * 100
                            log.info(f"  Events JOIN rate: {join_rate:.1f}%")
                except Exception as e:
                    log.error(f"  Type A query failed: {e}")

        # ── Type B: 리스크 전용 ──
        if include_risk_only:
            cp_file_b = CHECKPOINT_DIR / f"risk_{label}.parquet"

            if resume and cp_file_b.exists():
                log.info(f"[SKIP] Type B {label} — checkpoint exists")
                all_risk_dfs.append(pd.read_parquet(cp_file_b))
            else:
                query_b = build_risk_only_query(
                    company_regex, risk_regex, date_start, date_end
                )

                if dry_run:
                    log.info(f"[DRY-RUN] Type B {label}: {date_start} ~ {date_end}")
                    total_dry_gb += _run_query(client, query_b, f"B-{label}", dry_run=True)
                else:
                    log.info(f"[QUERY] Type B {label}: {date_start} ~ {date_end}")
                    try:
                        df_b = _run_query(client, query_b, f"B-{label}")
                        if not df_b.empty:
                            df_b.to_parquet(cp_file_b, index=False)
                            all_risk_dfs.append(df_b)
                    except Exception as e:
                        log.error(f"  Type B query failed: {e}")

    if dry_run:
        log.info(f"\n{'='*50}")
        log.info(f"Dry-run complete. Total estimated: {total_dry_gb:.1f} GB")
        log.info(f"Estimated cost: ${total_dry_gb * 6.25:.2f} (on-demand $6.25/TB)")
        log.info(f"Free tier: 1 TB/month")
        log.info(f"Remove --dry-run to execute.")
        return

    # ── Type A 합치기 + 후처리 ──
    if all_company_dfs:
        raw_a = pd.concat(all_company_dfs, ignore_index=True)
        before = len(raw_a)
        raw_a = raw_a.drop_duplicates(subset=["gkg_id"], keep="first")
        log.info(f"\nType A (company): {before:,} -> {len(raw_a):,} (dedup)")
        result_a = post_process(raw_a, company_aliases, alias_to_cid)
        result_a["collection_type"] = "company"
    else:
        result_a = pd.DataFrame()
        log.warning("No Type A (company) data collected!")

    # ── Type B 합치기 + 후처리 ──
    if all_risk_dfs:
        raw_b = pd.concat(all_risk_dfs, ignore_index=True)
        before = len(raw_b)
        raw_b = raw_b.drop_duplicates(subset=["gkg_id"], keep="first")
        log.info(f"Type B (risk-only): {before:,} -> {len(raw_b):,} (dedup)")
        result_b = post_process_risk_only(raw_b)
        result_b["collection_type"] = "risk_only"
    else:
        result_b = pd.DataFrame()
        log.info("No Type B (risk-only) data (--no-risk-only or no matches)")

    # ── 합쳐서 저장 ──
    parts = [df for df in [result_a, result_b] if not df.empty]
    if not parts:
        log.warning("No data collected at all!")
        return

    final_df = pd.concat(parts, ignore_index=True)

    # 최종 gkg_id 중복 제거
    final_df = final_df.drop_duplicates(subset=["gkg_id"], keep="first")

    output_path = OUTPUT_DIR / "gdelt_gkg_3y.parquet"
    final_df.to_parquet(output_path, engine="pyarrow", index=False)
    log.info(f"\nSaved: {output_path} ({len(final_df):,} rows)")
    log.info(f"  Type A (company): {len(result_a):,}")
    log.info(f"  Type B (risk-only): {len(result_b):,}")

    # ── 품질 리포트 ──
    print_report(final_df, company_aliases)


def print_report(df: pd.DataFrame, company_aliases: Dict[str, List[str]]):
    """수집 결과 품질 리포트"""
    from collections import Counter

    log.info("\n" + "=" * 60)
    log.info("QUALITY REPORT")
    log.info("=" * 60)
    log.info(f"Total articles: {len(df):,}")
    log.info(f"Date range: {df['date'].min()} ~ {df['date'].max()}")

    # Type A/B 분포
    if "collection_type" in df.columns:
        type_counts = df["collection_type"].value_counts()
        log.info(f"\nCollection type breakdown:")
        for ct, cnt in type_counts.items():
            log.info(f"  {ct}: {cnt:,} ({cnt/len(df)*100:.1f}%)")

    # 기업별 분포
    cc = Counter()
    for cids in df["company_ids"].dropna():
        for c in str(cids).split(";"):
            if c.strip():
                cc[c.strip()] += 1

    log.info(f"\nCompany distribution ({len(cc)} companies matched):")
    for cid, cnt in cc.most_common(20):
        log.info(f"  {cid:<15} {cnt:>6,}")

    # 미매칭 기업
    all_cids = set(company_aliases.keys())
    missing = all_cids - set(cc.keys())
    if missing:
        log.info(f"\nUnmatched companies ({len(missing)}):")
        for cid in sorted(missing):
            log.info(f"  {cid}")

    # Events JOIN 성공률
    if "goldstein_scale" in df.columns:
        join_rate = df["goldstein_scale"].notna().mean() * 100
        log.info(f"\nEvents JOIN rate: {join_rate:.1f}%")
        log.info(f"  With EventCode: {df['event_code'].notna().sum():,}")
        log.info(f"  With GoldsteinScale: {df['goldstein_scale'].notna().sum():,}")

    # 리스크 매칭
    if "risk_matched" in df.columns:
        risk_count = df["risk_matched"].sum()
        log.info(f"\nRisk matched: {risk_count:,} ({risk_count/len(df)*100:.1f}%)")

    # Severity 분포
    log.info(f"\nSeverity stats:")
    log.info(f"  mean={df['severity'].mean():.2f}, median={df['severity'].median():.2f}")
    log.info(f"  min={df['severity'].min():.2f}, max={df['severity'].max():.2f}")

    log.info("=" * 60)


# ──────────────────────── Main ────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GDELT BigQuery Collector v3 — 나비효과")
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument("--table", choices=["gkg", "join"], default="join",
                        help="gkg=GKG only, join=GKG+Events JOIN (default)")
    parser.add_argument("--dry-run", action="store_true", help="Estimate cost only")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints")
    parser.add_argument("--project-id", type=str, default=PROJECT_ID)
    parser.add_argument("--no-risk-only", action="store_true",
                        help="Type B (리스크 전용) 수집 비활성화")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("나비효과 — GDELT BigQuery Collector v3.1")
    log.info("  Type A: 기업 매칭 뉴스 (GKG V2Organizations)")
    log.info("  Type B: 리스크 전용 뉴스 (Route Node 전파용)")
    log.info("=" * 60)

    # 1. Aliases
    company_aliases, alias_to_cid = build_company_aliases()
    company_regex = build_bq_company_regex(company_aliases)
    risk_regex = build_bq_risk_regex()

    # 2. Date ranges
    date_ranges = generate_date_ranges(args.start_year, args.end_year)
    log.info(f"Collection: {args.start_year}-{args.end_year} ({len(date_ranges)} periods)")
    for lbl, ds, de in date_ranges:
        log.info(f"  {lbl}: {ds} ~ {de}")

    # 3. BigQuery client
    client = bigquery.Client(project=args.project_id)

    # 4. Collect
    collect(
        client=client,
        company_regex=company_regex,
        risk_regex=risk_regex,
        company_aliases=company_aliases,
        alias_to_cid=alias_to_cid,
        date_ranges=date_ranges,
        use_join=(args.table == "join"),
        include_risk_only=(not args.no_risk_only),
        dry_run=args.dry_run,
        resume=args.resume,
    )

    log.info("\nDone!")


if __name__ == "__main__":
    main()
