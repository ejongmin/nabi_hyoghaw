#!/usr/bin/env python3
"""
나비효과 프로젝트 — GDELT v2 GKG (Global Knowledge Graph) 수집기
================================================================
GDELT GKG 파일(15분 간격)을 다운로드하여
40개 기업 + 11개 국가 + 리스크 키워드에 매칭되는 기사만 추출.

핵심 차이 (vs Events Export 수집기):
  - Events Export: Actor1Name/Actor2Name에 기업명이 거의 없음 (0.1% 매칭)
  - GKG: V2Organizations에 기사 본문 NER 결과가 있어 기업 매칭률 대폭 상승

데이터 소스: http://data.gdeltproject.org/gdeltv2/{timestamp}.gkg.csv.zip
파일 포맷:  Tab-separated, 27 columns (GKG 2.1 schema)
갱신 주기:  15분 (하루 96파일, 1년 ~35,000파일)

사용법:
    python collect_gdelt_gkg.py --start 2018-01-01 --end 2025-12-31
    python collect_gdelt_gkg.py --resume
    python collect_gdelt_gkg.py --test           # 1일치 테스트
    python collect_gdelt_gkg.py --test --days 3   # 3일치 테스트

출력: data/raw/gdelt/gdelt_gkg_articles.parquet (증분 append)
"""

from __future__ import annotations
import argparse
import io
import json
import os
import re
import sys
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import Counter

import numpy as np
import pandas as pd
import requests
import yaml


# ──────────────────────────────────────────────
# GKG v2.1 Column Schema (27 columns)
# ──────────────────────────────────────────────
GKG_COLS = [
    "GKGRECORDID",              # 0
    "DATE",                      # 1  — YYYYMMDDHHMMSS
    "SourceCollectionIdentifier",# 2
    "SourceCommonName",          # 3  — domain
    "DocumentIdentifier",        # 4  — article URL
    "Counts",                    # 5
    "V2Counts",                  # 6
    "Themes",                    # 7
    "V2Themes",                  # 8  — THEME,offset;THEME,offset;...
    "Locations",                 # 9
    "V2Locations",               # 10 — type#name#CC#ADM1#lat#lon#FID,offset;...
    "Persons",                   # 11
    "V2Persons",                 # 12
    "Organizations",             # 13 — org1;org2;org3;...
    "V2Organizations",           # 14 — org1,offset;org2,offset;...
    "V2Tone",                    # 15 — tone,pos,neg,polarity,ard,sgrd,wc
    "DATES",                     # 16
    "GCAM",                      # 17
    "SharingImage",              # 18
    "RelatedImages",             # 19
    "SocialImageEmbeds",         # 20
    "SocialVideoEmbeds",         # 21
    "Quotations",                # 22
    "AllNames",                  # 23
    "Amounts",                   # 24
    "TranslationInfo",           # 25
    "Extras",                    # 26
]

# 메모리 절약: 필요한 컬럼만 로드
KEEP_COLS = [
    "GKGRECORDID", "DATE", "SourceCommonName", "DocumentIdentifier",
    "V2Themes", "V2Locations", "Organizations", "V2Organizations",
    "V2Tone", "AllNames"
]
KEEP_IDX = [GKG_COLS.index(c) for c in KEEP_COLS]


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
class Config:
    # GKG URL template (vs Events의 .export.CSV.zip)
    BASE_URL = "http://data.gdeltproject.org/gdeltv2/{ts}.gkg.csv.zip"

    INTERVAL_MIN = 15
    REQUEST_DELAY_SEC = 0.3
    MAX_RETRIES = 3
    RETRY_DELAY_SEC = 5

    # Paths
    PROJECT_ROOT = Path(__file__).resolve().parent
    UNIVERSE_CSV = PROJECT_ROOT / "data" / "universe" / "univers_final.csv"
    KEYWORDS_YAML = PROJECT_ROOT / "configs" / "risk_keywords.yaml"
    OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "gdelt"
    OUTPUT_PARQUET = OUTPUT_DIR / "gdelt_gkg_articles.parquet"
    CHECKPOINT_FILE = OUTPUT_DIR / "_checkpoint_gkg.json"

    # Source reputation (논문 C 참고)
    BLOCKED_DOMAINS = {
        "reddit.com", "twitter.com", "x.com", "facebook.com",
        "tiktok.com", "youtube.com", "instagram.com",
        "blogspot.com", "wordpress.com", "medium.com",
        "wikipedia.org",
    }

    # 배치 저장 간격
    BATCH_SAVE_FILES = 200  # 200 파일(~3시간)마다 중간 저장


# ──────────────────────────────────────────────
# GKG Organization Matcher
# ──────────────────────────────────────────────
class GKGEntityMatcher:
    """
    GKG V2Organizations 필드에서 기업명을 매칭.
    V2Organizations는 GDELT가 기사 본문에서 NER로 추출한 조직명이므로
    Events Actor 필드보다 기업 매칭률이 훨씬 높음.

    V2Organizations 형식: "Organization Name,charOffset;Another Org,charOffset;..."
    Organizations (v1) 형식: "Organization Name;Another Org;..."
    """

    def __init__(self, universe_csv: Path, keywords_yaml: Path):
        self.company_patterns: List[Tuple[re.Pattern, str]] = []
        self.risk_patterns: Dict[str, List[Tuple[re.Pattern, str, int]]] = {}

        # 국가→CountryCode 매핑 (GKG V2Locations용)
        self.country_codes: Dict[str, str] = {}

        self._build_company_patterns(universe_csv)
        self._build_country_code_map()
        self._build_risk_patterns(keywords_yaml)

    def _build_company_patterns(self, universe_csv: Path):
        """Universe 40개 기업의 매칭 패턴 구축 — GKG org 문자열에 최적화"""
        df = pd.read_csv(universe_csv)

        # 기업별 별칭 (GDELT V2Organizations에 나타나는 형태)
        # GKG는 기사 본문 NER이므로 정식 명칭이 더 잘 잡힘
        EXTRA_ALIASES: Dict[str, List[str]] = {
            # ── Cell/Pack 제조사 ──
            "300750.SZ": ["CATL", "CONTEMPORARY AMPEREX", "CONTEMPORARY AMPEREX TECHNOLOGY",
                          "CONTEMPORARY AMPEREX TECHNOLOGY CO"],
            "373220.KS": ["LG ENERGY", "LGES", "LG ENERGY SOLUTION", "LG ENERGY SOLUTIONS"],
            "1211.HK": ["BYD", "BYD COMPANY", "BYD AUTO", "BYD CO"],
            "051910.KS": ["LG CHEM", "LG CHEMICAL", "LG CHEM LTD"],
            "300014.SZ": ["EVE ENERGY", "EVE ENERGY CO"],
            "300207.SZ": ["SUNWODA", "SUNWODA ELECTRONIC", "SUNWODA ELECTRONIC CO"],
            "3931.HK": ["CALB", "CALB GROUP", "CALB CO",
                        "CHINA AVIATION LITHIUM BATTERY", "CHINA AVIATION LITHIUM"],
            "002074.SZ": ["GOTION", "GOTION HIGH-TECH", "GUOXUAN", "GUOXUAN HIGH-TECH"],

            # ── Materials (양극재/음극재/전해액/분리막) ──
            "247540.KQ": ["ECOPRO", "ECOPRO BM", "ECOPRO CO"],
            "066970.KS": ["L&F", "L AND F", "L&F CO"],
            "003670.KS": ["POSCO FUTURE", "POSCO FUTURE M", "POSCO CHEMICAL",
                          "POSCO HOLDINGS"],
            "300919.SZ": ["CNGR", "CNGR ADVANCED MATERIAL", "CNGR ADVANCED"],
            "603659.SH": ["PUTAILAI", "PUTAILAI NEW ENERGY"],
            "300037.SZ": ["CAPCHEM", "SHENZHEN CAPCHEM", "CAPCHEM TECHNOLOGY"],
            "600110.SH": ["NUODE", "NUODE INVESTMENT", "NUODE NEW MATERIAL"],
            "688388.SH": ["JIAYUAN", "JIAYUAN INTERNATIONAL", "JIAYUAN TECHNOLOGY"],
            "002340.SZ": ["GEM CO", "GEM CO LTD", "GEM CO LTD."],
            "920185.BJ": ["BTR", "BTR NEW MATERIAL", "BTR NEW ENERGY"],
            "3407.T": ["ASAHI KASEI", "ASAHI KASEI CORP", "ASAHI KASEI CORPORATION"],

            # ── Upstream/Refining (리튬/코발트/구리/니켈) ──
            "002460.SZ": ["GANFENG", "GANFENG LITHIUM", "GANFENG LITHIUM CO",
                          "JIANGXI GANFENG", "JIANGXI GANFENG LITHIUM"],
            "002466.SZ": ["TIANQI", "TIANQI LITHIUM", "TIANQI LITHIUM CORP",
                          "SICHUAN TIANQI", "TIANQI LITHIUM INDUSTRIES"],
            "603799.SH": ["HUAYOU", "HUAYOU COBALT", "ZHEJIANG HUAYOU",
                          "HUAYOU COBALT CO", "ZHEJIANG HUAYOU COBALT"],
            "603993.SH": ["CMOC", "CMOC GROUP", "CHINA MOLY", "CHINA MOLYBDENUM",
                          "CMOC GROUP LIMITED"],
            "600362.SH": ["JIANGXI COPPER", "JIANGXI COPPER COMPANY",
                          "JIANGXI COPPER CO", "JIANGXI COPPER CORP"],
            "ALB": ["ALBEMARLE", "ALBEMARLE CORP", "ALBEMARLE CORPORATION"],
            "SQM": ["SQM", "SOCIEDAD QUIMICA Y MINERA", "SOCIEDAD QUIMICA"],
            "GLEN.L": ["GLENCORE", "GLENCORE PLC", "GLENCORE INTERNATIONAL"],

            # ── OEM (완성차) ──
            "BMW.DE": ["BMW", "BAYERISCHE MOTOREN", "BMW GROUP", "BMW AG"],
            "VOW3.DE": ["VOLKSWAGEN", "VW", "VOLKSWAGEN AG", "VOLKSWAGEN GROUP",
                        "VW GROUP"],
            "MBG.DE": ["MERCEDES-BENZ", "MERCEDES BENZ", "DAIMLER",
                       "MERCEDES-BENZ GROUP"],
            "TSLA": ["TESLA", "TESLA MOTORS", "TESLA INC"],
            "F": ["FORD MOTOR", "FORD MOTOR COMPANY"],
            "GM": ["GENERAL MOTORS", "GENERAL MOTORS COMPANY"],
            "LCID": ["LUCID MOTORS", "LUCID GROUP", "LUCID"],
            "RIVN": ["RIVIAN", "RIVIAN AUTOMOTIVE"],
            "7267.T": ["HONDA MOTOR", "HONDA", "HONDA MOTOR CO"],
            "7203.T": ["TOYOTA MOTOR", "TOYOTA", "TOYOTA MOTOR CORP"],
            "0175.HK": ["GEELY", "GEELY AUTOMOBILE", "GEELY AUTO",
                        "ZHEJIANG GEELY"],
            "600733.SH": ["BAIC", "BAIC BLUEPARK", "BAIC MOTOR", "BAIC GROUP",
                          "BEIJING AUTOMOTIVE"],
            "000625.SZ": ["CHANGAN", "CHANGAN AUTO", "CHANGAN AUTOMOBILE",
                          "CHONGQING CHANGAN"],
            "600006.SH": ["DONGFENG", "DONGFENG MOTOR", "DONGFENG MOTOR CORP",
                          "DONGFENG MOTOR GROUP", "DONGFENG AUTO"],
            "601238.SH": ["GAC", "GAC GROUP", "GAC MOTOR", "GUANGZHOU AUTOMOBILE",
                          "GAC AION", "GUANGZHOU AUTO"],
            "601633.SH": ["GREAT WALL MOTOR", "GREAT WALL MOTORS", "GWM",
                          "GREAT WALL MOTOR COMPANY"],
            "600418.SH": ["JAC MOTORS", "JAC", "ANHUI JIANGHUAI",
                          "JAC AUTOMOBILE"],
            "LICY": ["LI-CYCLE", "LI CYCLE", "LI-CYCLE HOLDINGS"],

            # ── Chemical ──
            "BAS.DE": ["BASF", "BASF SE", "BASF AG", "BASF CORPORATION"],
        }

        # Universe CSV의 모든 기업 처리
        universe_ids = set()
        for _, row in df.iterrows():
            cid = str(row["company_id"]).strip()
            cname = str(row["canonical_name"]).strip()
            universe_ids.add(cid)

            aliases = set()
            if cname and cname != "nan":
                aliases.add(cname.upper())

            for a in EXTRA_ALIASES.get(cid, []):
                aliases.add(a.upper())

            for alias in aliases:
                if len(alias) < 3:
                    continue
                try:
                    pat = re.compile(r'\b' + re.escape(alias) + r'\b', re.IGNORECASE)
                    self.company_patterns.append((pat, cid))
                except re.error:
                    pass

        # EXTRA_ALIASES에만 있고 universe에 없는 기업도 추가
        # (TSLA, RIVN, VOW3.DE 등 — 관련 기업으로 뉴스 추적용)
        for cid, alias_list in EXTRA_ALIASES.items():
            if cid not in universe_ids:
                for a in alias_list:
                    if len(a) < 3:
                        continue
                    try:
                        pat = re.compile(r'\b' + re.escape(a) + r'\b', re.IGNORECASE)
                        self.company_patterns.append((pat, cid))
                    except re.error:
                        pass

        print(f"  Company patterns loaded: {len(self.company_patterns)}")

    def _build_country_code_map(self):
        """GKG V2Locations의 FIPS country code → 국가 노드 ID"""
        # V2Locations 형식: type#name#CountryCode#ADM1#lat#lon#FeatureID,offset
        # CountryCode는 FIPS 코드
        self.fips_to_country = {
            "CI": "Chile", "AS": "Australia", "ID": "Indonesia",
            "CG": "Congo", "CF": "Congo",
            "CH": "China", "KS": "South_Korea", "GM": "Germany",
            "JA": "Japan", "US": "USA", "UK": "UK", "CA": "Canada",
        }
        # 국가명 매칭도 (AllNames/V2Locations에 정식명칭이 나올 수 있음)
        self.country_name_map = {
            "CHILE": "Chile", "AUSTRALIA": "Australia", "INDONESIA": "Indonesia",
            "CONGO": "Congo", "DEMOCRATIC REPUBLIC OF THE CONGO": "Congo",
            "CHINA": "China", "SOUTH KOREA": "South_Korea", "KOREA": "South_Korea",
            "GERMANY": "Germany", "JAPAN": "Japan",
            "UNITED STATES": "USA", "UNITED KINGDOM": "UK", "CANADA": "Canada",
        }
        print(f"  Country mappings loaded: {len(self.fips_to_country)} FIPS + {len(self.country_name_map)} names")

    def _build_risk_patterns(self, keywords_yaml: Path):
        """risk_keywords.yaml에서 패턴 로드"""
        try:
            with open(keywords_yaml, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError:
            print("  [WARN] risk_keywords.yaml parse failed, using fallback")
            data = self._fallback_risk_keywords()

        total = 0
        for risk_type in ["geopolitics", "logistics", "natural"]:
            section = data.get(risk_type, {})
            patterns = []
            for kw in section.get("keywords", []):
                if isinstance(kw, dict) and "pattern" in kw:
                    try:
                        pat = re.compile(kw["pattern"], re.IGNORECASE)
                        weight = kw.get("weight", 3)
                        term = kw.get("term", "")
                        patterns.append((pat, term, weight))
                    except (re.error, TypeError):
                        pass
            self.risk_patterns[risk_type] = patterns
            total += len(patterns)
        print(f"  Risk patterns loaded: {total} across {len(self.risk_patterns)} types")

    @staticmethod
    def _fallback_risk_keywords() -> dict:
        """YAML 실패 시 핵심 키워드 fallback"""
        return {
            "geopolitics": {"keywords": [
                {"pattern": r"sanction[s]?", "weight": 4, "term": "sanction"},
                {"pattern": r"embargo", "weight": 4, "term": "embargo"},
                {"pattern": r"export control|export ban", "weight": 4, "term": "export control"},
                {"pattern": r"tariff[s]?", "weight": 3, "term": "tariff"},
                {"pattern": r"trade war", "weight": 3, "term": "trade war"},
                {"pattern": r"\bwar\b", "weight": 5, "term": "war"},
                {"pattern": r"invasion", "weight": 5, "term": "invasion"},
            ]},
            "logistics": {"keywords": [
                {"pattern": r"blockade", "weight": 5, "term": "blockade"},
                {"pattern": r"port strike|dock strike", "weight": 4, "term": "port strike"},
                {"pattern": r"supply disruption|supply chain disruption", "weight": 3, "term": "supply disruption"},
                {"pattern": r"Suez|Red Sea|Hormuz|Malacca", "weight": 4, "term": "chokepoint"},
            ]},
            "natural": {"keywords": [
                {"pattern": r"earthquake|seismic", "weight": 5, "term": "earthquake"},
                {"pattern": r"tsunami", "weight": 5, "term": "tsunami"},
                {"pattern": r"typhoon|hurricane|cyclone", "weight": 5, "term": "storm"},
                {"pattern": r"flood|flooding", "weight": 4, "term": "flood"},
                {"pattern": r"wildfire|forest fire", "weight": 4, "term": "wildfire"},
                {"pattern": r"drought|water shortage", "weight": 3, "term": "drought"},
            ]},
        }

    # ── Matching Methods ──

    # AllNames에서 인명/지명 오탐이 심한 패턴 → Organizations에서만 매칭
    # "FORD" → Harrison Ford, Gerald Ford / "JAC" → Jacob 등
    ALLNAMES_BLOCKED_PATTERNS = {
        r"\bFORD\b", r"\bJAC\b", r"\bGEM\b", r"\bGAC\b", r"\bBTR\b",
    }

    def match_organizations(self, v2orgs: str, v1orgs: str = "", allnames: str = "") -> List[str]:
        """
        GKG V2Organizations + V1Organizations + AllNames에서 기업 매칭.

        매칭 전략 (오탐 방지):
        - V2Organizations / V1Organizations: 모든 패턴으로 매칭 (NER이 조직만 추출)
        - AllNames: 긴 패턴(>6자)만 매칭 (인명/지명 혼재 → 짧은 패턴 오탐 위험)

        → company_id 리스트 반환
        """
        if not v2orgs and not v1orgs and not allnames:
            return []

        # ── Organizations 필드에서 조직명 추출 ──
        org_names = set()
        if v2orgs:
            for entry in str(v2orgs).split(";"):
                entry = entry.strip()
                if "," in entry:
                    name = entry.rsplit(",", 1)[0].strip()
                    if name:
                        org_names.add(name.upper())
                elif entry:
                    org_names.add(entry.upper())

        if v1orgs:
            for name in str(v1orgs).split(";"):
                name = name.strip()
                if name:
                    org_names.add(name.upper())

        # ── AllNames 필드에서 이름 추출 (별도 보관) ──
        all_name_set = set()
        if allnames:
            for entry in str(allnames).split(";"):
                entry = entry.strip()
                if "," in entry:
                    name = entry.rsplit(",", 1)[0].strip()
                    if name:
                        all_name_set.add(name.upper())

        if not org_names and not all_name_set:
            return []

        # ── 매칭: Organizations는 모든 패턴, AllNames는 긴 패턴만 ──
        org_text = " | ".join(org_names) if org_names else ""
        allnames_text = " | ".join(all_name_set) if all_name_set else ""

        hits = set()
        for pat, cid in self.company_patterns:
            # Organizations 필드에서 매칭 (모든 패턴 허용)
            if org_text and pat.search(org_text):
                hits.add(cid)
            # AllNames 필드에서 매칭 (블랙리스트 패턴 제외)
            elif allnames_text and pat.pattern not in self.ALLNAMES_BLOCKED_PATTERNS:
                if pat.search(allnames_text):
                    hits.add(cid)

        return list(hits)

    def match_countries_from_locations(self, v2locations: str) -> List[str]:
        """
        GKG V2Locations에서 국가 매칭.
        형식: type#fullname#CountryCode#ADM1#lat#lon#FeatureID,charOffset;...
        CountryCode는 FIPS 코드.
        """
        if not v2locations:
            return []

        hits = set()
        for entry in str(v2locations).split(";"):
            entry = entry.strip()
            # charOffset 제거
            if "," in entry:
                entry = entry.rsplit(",", 1)[0].strip()

            parts = entry.split("#")
            if len(parts) >= 3:
                fips_code = parts[2].strip()
                if fips_code in self.fips_to_country:
                    hits.add(self.fips_to_country[fips_code])

                # 전체 이름에서도 매칭 시도
                fullname = parts[1].strip().upper() if len(parts) > 1 else ""
                for cname, node_id in self.country_name_map.items():
                    if cname in fullname:
                        hits.add(node_id)

        return list(hits)

    def match_risk_from_themes(self, v2themes: str, url: str = "") -> List[Tuple[str, int]]:
        """
        GKG V2Themes + URL에서 리스크 키워드 매칭.
        V2Themes: "TAX_FNCACT,123;GENERAL_GOVERNMENT,456;WB_2024_TRADE_POLICY,789;..."

        GDELT 테마 중 리스크 관련 주요 테마:
        - TAX_*, TRADE, SANCTIONS, EMBARGO → geopolitics
        - NATURAL_DISASTER_*, EARTHQUAKE, FLOOD → natural
        - TRANSPORTATION, SHIPPING → logistics
        """
        results = {}
        combined_text = ""

        # V2Themes에서 테마명 추출
        if v2themes:
            theme_names = []
            for entry in str(v2themes).split(";"):
                entry = entry.strip()
                if "," in entry:
                    theme = entry.rsplit(",", 1)[0].strip()
                    theme_names.append(theme)
                elif entry:
                    theme_names.append(entry)
            combined_text = " ".join(theme_names)

        # URL도 포함 (기사 제목이 URL에 들어있는 경우 많음)
        if url:
            combined_text += " " + str(url)

        if not combined_text:
            return []

        for rtype, patterns in self.risk_patterns.items():
            max_w = 0
            for pat, term, weight in patterns:
                if pat.search(combined_text):
                    max_w = max(max_w, weight)
            if max_w > 0:
                results[rtype] = max_w

        return list(results.items())

    def extract_tone(self, v2tone: str) -> Tuple[float, float, float]:
        """
        V2Tone: "tone,pos_score,neg_score,polarity,activity_ref_density,self_group_ref_density,word_count"
        → (tone, pos_score, neg_score) 반환
        """
        if not v2tone:
            return (0.0, 0.0, 0.0)
        try:
            parts = str(v2tone).split(",")
            tone = float(parts[0]) if len(parts) > 0 else 0.0
            pos = float(parts[1]) if len(parts) > 1 else 0.0
            neg = float(parts[2]) if len(parts) > 2 else 0.0
            return (tone, pos, neg)
        except (ValueError, IndexError):
            return (0.0, 0.0, 0.0)


# ──────────────────────────────────────────────
# Severity Calculation
# ──────────────────────────────────────────────
def compute_severity_from_tone(tone: float) -> float:
    """
    GKG V2Tone 기반 severity 계산.
    tone: -100 ~ +100 (보통 -10 ~ +10 범위)
    severity: 1.0(약) ~ 5.0(강) — 부정적일수록 높음
    """
    # tone을 0~1 범위로 정규화 (부정일수록 1에 가까움)
    norm_tone = np.clip((10 - tone) / 20.0, 0, 1)
    severity = 1.0 + norm_tone * 4.0
    return round(float(np.clip(severity, 1.0, 5.0)), 3)


# ──────────────────────────────────────────────
# Source Reputation Filter
# ──────────────────────────────────────────────
def is_blocked_source(domain: str) -> bool:
    if not domain:
        return False
    domain_lower = domain.lower().replace("www.", "")
    return any(blocked in domain_lower for blocked in Config.BLOCKED_DOMAINS)


# ──────────────────────────────────────────────
# Timestamp Generation
# ──────────────────────────────────────────────
def generate_timestamps(start: datetime, end: datetime) -> List[str]:
    """GDELT v2 GKG 파일명 형식: YYYYMMDDHHmmSS"""
    timestamps = []
    current = start.replace(hour=0, minute=0, second=0, microsecond=0)
    while current <= end:
        ts = current.strftime("%Y%m%d%H%M%S")
        timestamps.append(ts)
        current += timedelta(minutes=Config.INTERVAL_MIN)
    return timestamps


# ──────────────────────────────────────────────
# Checkpoint System
# ──────────────────────────────────────────────
def load_checkpoint() -> dict:
    if Config.CHECKPOINT_FILE.exists():
        with open(Config.CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"last_timestamp": None, "total_matched": 0, "total_files": 0}


def save_checkpoint(data: dict):
    Config.CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(Config.CHECKPOINT_FILE, "w") as f:
        json.dump(data, f)


# ──────────────────────────────────────────────
# Main Collector
# ──────────────────────────────────────────────
def download_and_filter_gkg(
    ts: str,
    matcher: GKGEntityMatcher,
    session: requests.Session
) -> Optional[pd.DataFrame]:
    """
    단일 GKG 15분 파일 다운로드 → 필터링 → DataFrame 반환

    필터링 전략:
    1) V2Organizations에서 40개 기업 매칭 (핵심!)
    2) V2Locations에서 관심 국가 매칭
    3) V2Themes + URL에서 리스크 키워드 매칭
    4) blocked source 제거

    반환 조건: 기업이 1개 이상 매칭된 기사만 (국가만 매칭은 제외)
    """
    url = Config.BASE_URL.format(ts=ts)

    for attempt in range(Config.MAX_RETRIES):
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt == Config.MAX_RETRIES - 1:
                print(f"  [!] Failed: {ts} — {e}")
                return None
            time.sleep(Config.RETRY_DELAY_SEC * (attempt + 1))

    # Unzip
    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_name = zf.namelist()[0]
            raw_bytes = zf.read(csv_name)
    except (zipfile.BadZipFile, IndexError):
        return None

    # Parse TSV
    try:
        df = pd.read_csv(
            io.BytesIO(raw_bytes),
            sep="\t",
            header=None,
            names=GKG_COLS,
            usecols=KEEP_IDX,
            dtype=str,
            on_bad_lines="skip",
            encoding="utf-8",
            encoding_errors="replace"
        )
    except Exception as e:
        print(f"  [!] Parse error: {ts} — {e}")
        return None

    if df.empty:
        return None

    # ── Filter: Company must match ──
    matched_rows = []

    for idx, row in df.iterrows():
        domain = str(row.get("SourceCommonName", "") or "")
        article_url = str(row.get("DocumentIdentifier", "") or "")

        # Skip blocked sources
        if is_blocked_source(domain):
            continue

        v2orgs = str(row.get("V2Organizations", "") or "")
        v1orgs = str(row.get("Organizations", "") or "")
        allnames = str(row.get("AllNames", "") or "")
        v2locations = str(row.get("V2Locations", "") or "")
        v2themes = str(row.get("V2Themes", "") or "")
        v2tone = str(row.get("V2Tone", "") or "")
        date_str = str(row.get("DATE", "") or "")
        gkg_id = str(row.get("GKGRECORDID", "") or "")

        # 1) Company matching (핵심 — 기업 매칭 없으면 skip)
        company_hits = matcher.match_organizations(v2orgs, v1orgs, allnames)
        if not company_hits:
            continue  # ← Events 수집기와의 핵심 차이: 기업 필수

        # 2) Country matching
        country_hits = matcher.match_countries_from_locations(v2locations)

        # 3) Risk keyword matching
        risk_hits = matcher.match_risk_from_themes(v2themes, article_url)

        # 4) Tone & Severity
        tone, pos_score, neg_score = matcher.extract_tone(v2tone)
        severity = compute_severity_from_tone(tone)

        matched_rows.append({
            "gkg_id": gkg_id,
            "date": date_str[:8] if len(date_str) >= 8 else date_str,
            "datetime": date_str,
            "source_domain": domain,
            "article_url": article_url,
            "company_ids": ";".join(sorted(company_hits)),
            "country_ids": ";".join(sorted(country_hits)),
            "risk_types": ";".join(r for r, w in risk_hits) if risk_hits else "",
            "risk_max_weight": max((w for _, w in risk_hits), default=0),
            "tone": round(tone, 3),
            "pos_score": round(pos_score, 3),
            "neg_score": round(neg_score, 3),
            "severity": severity,
            "org_text": v2orgs[:200] if v2orgs else "",  # 디버깅용 (200자)
            "themes_text": v2themes[:150] if v2themes else "",  # 디버깅용 (150자)
        })

    if not matched_rows:
        return None

    return pd.DataFrame(matched_rows)


def _save_parquet(batch_df: pd.DataFrame):
    """Parquet에 증분 append"""
    output = Config.OUTPUT_PARQUET

    if output.exists():
        existing = pd.read_parquet(output)
        combined = pd.concat([existing, batch_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["gkg_id"], keep="last")
    else:
        combined = batch_df

    combined.to_parquet(output, engine="pyarrow", index=False)
    print(f"  → Saved {len(combined):,} articles to parquet "
          f"(+{len(batch_df):,} new)")


def collect(start_date: str, end_date: str, resume: bool = False):
    """메인 수집 루프"""
    print("=" * 60)
    print("나비효과 — GDELT GKG v2 Collector")
    print("=" * 60)

    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matcher = GKGEntityMatcher(Config.UNIVERSE_CSV, Config.KEYWORDS_YAML)

    # Timestamps
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1) - timedelta(seconds=1)
    all_ts = generate_timestamps(start, end)
    print(f"Date range: {start_date} → {end_date}")
    print(f"Total 15-min files: {len(all_ts):,}")

    # Checkpoint
    checkpoint = load_checkpoint() if resume else {
        "last_timestamp": None, "total_matched": 0, "total_files": 0
    }

    if resume and checkpoint["last_timestamp"]:
        skip_to = checkpoint["last_timestamp"]
        all_ts = [ts for ts in all_ts if ts > skip_to]
        print(f"Resuming from: {skip_to} ({len(all_ts):,} remaining)")

    # Session
    session = requests.Session()
    session.headers.update({"User-Agent": "NabiHyoghaw-Research/1.0"})

    all_frames: List[pd.DataFrame] = []
    total = len(all_ts)
    matched_total = checkpoint["total_matched"]
    files_processed = checkpoint["total_files"]
    company_counter = Counter()  # 기업별 매칭 수 실시간 추적

    for i, ts in enumerate(all_ts):
        # 하루 단위 진행률
        if i % 96 == 0:
            date_str = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}"
            pct = 100 * i / total if total > 0 else 0
            top5 = company_counter.most_common(5)
            top5_str = ", ".join(f"{k}:{v}" for k, v in top5) if top5 else "none yet"
            print(f"\n[{pct:5.1f}%] {date_str} | "
                  f"matched: {matched_total:,} | files: {files_processed:,}/{total:,}")
            print(f"        Top companies: {top5_str}")

        result = download_and_filter_gkg(ts, matcher, session)
        files_processed += 1

        if result is not None:
            all_frames.append(result)
            matched_total += len(result)

            # 기업별 카운터 업데이트
            for cids in result["company_ids"]:
                for cid in cids.split(";"):
                    if cid:
                        company_counter[cid] += 1

        # Rate limit
        time.sleep(Config.REQUEST_DELAY_SEC)

        # Batch save
        if len(all_frames) >= Config.BATCH_SAVE_FILES or (i == total - 1 and all_frames):
            batch_df = pd.concat(all_frames, ignore_index=True)
            _save_parquet(batch_df)
            all_frames.clear()

            save_checkpoint({
                "last_timestamp": ts,
                "total_matched": matched_total,
                "total_files": files_processed
            })

    # Final stats
    print(f"\n{'='*60}")
    print(f"COLLECTION COMPLETE")
    print(f"  Files processed: {files_processed:,}")
    print(f"  Matched articles: {matched_total:,}")
    print(f"  Output: {Config.OUTPUT_PARQUET}")
    print(f"\n  Company distribution (top 20):")
    for cid, count in company_counter.most_common(20):
        print(f"    {cid}: {count:,}")
    print(f"  Companies with ≥1 match: {len(company_counter)}/{40}")
    print(f"{'='*60}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="GDELT GKG v2 Collector for 나비효과")
    parser.add_argument("--start", default="2018-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--test", action="store_true", help="Test mode")
    parser.add_argument("--days", type=int, default=1, help="Days for test mode (default: 1)")
    args = parser.parse_args()

    if args.test:
        end_date = (datetime(2024, 1, 1) + timedelta(days=args.days - 1)).strftime("%Y-%m-%d")
        collect("2024-01-01", end_date, resume=False)
    else:
        collect(args.start, args.end, resume=args.resume)


if __name__ == "__main__":
    main()
