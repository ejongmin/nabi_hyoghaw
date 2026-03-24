#!/usr/bin/env python3
"""
DART (전자공시) 사업보고서 수집기
==================================
한국 상장 기업의 사업보고서에서 공급망 관계를 자동 추출.

대상 기업 (universe 중 한국 상장):
  - 373220.KS (LG Energy Solution)
  - 051910.KS (LG Chem)
  - 247540.KQ (EcoPro BM)
  - 066970.KS (L&F)
  - 003670.KS (POSCO Future M)

필요:
  - DART API Key (https://opendart.fss.or.kr 에서 발급)
  - 환경변수: DART_API_KEY 또는 --api-key 인자

수집 대상:
  - 사업보고서 (annual report) 중 "사업의 내용", "주요 거래처", "원재료" 섹션

출력:
  data/raw/filings/dart_edges.parquet — 공시 기반 공급망 엣지
  data/raw/filings/dart_supply_sentences.parquet — 원본 문장 (검증용)

사용:
  export DART_API_KEY=your_key_here
  python scripts/collect_dart_kr.py
  python scripts/collect_dart_kr.py --dry-run
  python scripts/collect_dart_kr.py --years 2022 2023 2024
"""
import argparse
import json
import logging
import os
import re
import time
import zipfile
import io
from pathlib import Path
from typing import Dict, List, Optional
from xml.etree import ElementTree as ET

import pandas as pd
import requests

log = logging.getLogger("dart_kr")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

# ── DART API config ──
DART_BASE = "https://opendart.fss.or.kr/api"
DART_REPORT_LIST = f"{DART_BASE}/list.json"
DART_DOCUMENT = f"{DART_BASE}/document.xml"

# ── 우리 Universe 중 한국 상장 기업 ──
KR_COMPANIES = {
    "373220": {"name": "LG에너지솔루션", "name_en": "LG Energy Solution", "company_id": "373220.KS"},
    "051910": {"name": "LG화학", "name_en": "LG Chem", "company_id": "051910.KS"},
    "247540": {"name": "에코프로비엠", "name_en": "EcoPro BM", "company_id": "247540.KQ"},
    "066970": {"name": "엘앤에프", "name_en": "L&F", "company_id": "066970.KS"},
    "003670": {"name": "포스코퓨처엠", "name_en": "POSCO Future M", "company_id": "003670.KS"},
}

# ── 공급망 키워드 (한국어) ──
SUPPLY_KEYWORDS_KR = [
    r"(?:주요\s*(?:거래처|고객|납품처|매출처|수요처))",
    r"(?:원재료|원자재|주요\s*자재)",
    r"(?:공급\s*(?:계약|협약|관계))",
    r"(?:매출\s*(?:비중|비율|구성))",
    r"(?:납품|공급|조달)",
    r"(?:리튬|코발트|니켈|흑연|구리|양극재|음극재|전해질|분리막)",
    r"(?:배터리\s*(?:셀|팩|모듈))",
    r"(?:CATL|BYD|파나소닉|삼성SDI|SK온|SK이노베이션)",
    r"(?:현대자동차|기아|GM|포드|테슬라|BMW|폭스바겐|도요타|혼다)",
    r"(?:합작|JV|파트너십|전략적\s*제휴|협력)",
]
SUPPLY_PATTERN_KR = re.compile("|".join(SUPPLY_KEYWORDS_KR))

# ── 알려진 기업 매핑 (한국어 → company_id) ──
KNOWN_COMPANIES_KR = {
    "CATL": "300750.SZ", "닝더스다이": "300750.SZ", "宁德时代": "300750.SZ",
    "LG에너지솔루션": "373220.KS", "LG에너지": "373220.KS",
    "LG화학": "051910.KS",
    "에코프로비엠": "247540.KQ", "에코프로": "247540.KQ",
    "엘앤에프": "066970.KS", "L&F": "066970.KS",
    "포스코퓨처엠": "003670.KS", "포스코케미칼": "003670.KS",
    "삼성SDI": "006400.KS",
    "SK온": "096770.KS", "SK이노베이션": "096770.KS",
    "BYD": "1211.HK", "비야디": "1211.HK",
    "파나소닉": "6752.T", "Panasonic": "6752.T",
    "현대자동차": "005380.KS", "현대차": "005380.KS",
    "기아": "000270.KS",
    "GM": "GM", "제너럴모터스": "GM",
    "포드": "F", "Ford": "F",
    "테슬라": "TSLA", "Tesla": "TSLA",
    "BMW": "BMW.DE",
    "폭스바겐": "VOW3.DE", "Volkswagen": "VOW3.DE",
    "도요타": "7203.T", "Toyota": "7203.T",
    "혼다": "7267.T", "Honda": "7267.T",
    "메르세데스": "MBG.DE", "벤츠": "MBG.DE",
    "Albemarle": "ALB", "알버말": "ALB",
    "간펑리튬": "002460.SZ", "Ganfeng": "002460.SZ",
    "톈치리튬": "002466.SZ", "Tianqi": "002466.SZ",
    "글렌코어": "GLEN.L", "Glencore": "GLEN.L",
    "BASF": "BAS.DE", "바스프": "BAS.DE",
    "화유코발트": "603799.SH", "Huayou": "603799.SH",
    "리비안": "RIVN", "Rivian": "RIVN",
}


def _dart_sleep():
    """DART API rate limit."""
    time.sleep(0.5)


def get_corp_code(api_key: str) -> Dict[str, str]:
    """
    DART 고유번호(corp_code) 조회.
    종목코드 → corp_code 매핑 반환.
    """
    url = f"{DART_BASE}/corpCode.xml"
    _dart_sleep()
    resp = requests.get(url, params={"crtfc_key": api_key}, timeout=30)
    resp.raise_for_status()

    # ZIP 파일 안에 XML
    z = zipfile.ZipFile(io.BytesIO(resp.content))
    xml_data = z.read(z.namelist()[0])

    root = ET.fromstring(xml_data)
    mapping = {}
    for item in root.findall("list"):
        stock_code = item.findtext("stock_code", "").strip()
        corp_code = item.findtext("corp_code", "").strip()
        if stock_code and corp_code:
            mapping[stock_code] = corp_code

    return mapping


def get_annual_reports(api_key: str, corp_code: str, years: List[int]) -> List[Dict]:
    """사업보고서 목록 조회."""
    reports = []

    for year in years:
        _dart_sleep()
        params = {
            "crtfc_key": api_key,
            "corp_code": corp_code,
            "bgn_de": f"{year}0101",
            "end_de": f"{year}1231",
            "pblntf_ty": "A",  # 정기공시
            "page_count": 100,
        }
        resp = requests.get(DART_REPORT_LIST, params=params, timeout=30)
        data = resp.json()

        if data.get("status") != "000":
            continue

        for item in data.get("list", []):
            report_nm = item.get("report_nm", "")
            # 사업보고서만 필터
            if "사업보고서" in report_nm and "분기" not in report_nm and "반기" not in report_nm:
                reports.append({
                    "rcept_no": item["rcept_no"],
                    "report_nm": report_nm,
                    "rcept_dt": item["rcept_dt"],
                    "fiscal_year": year,
                })

    return reports


def download_report_xml(api_key: str, rcept_no: str) -> str:
    """사업보고서 본문 XML 다운로드."""
    _dart_sleep()
    params = {
        "crtfc_key": api_key,
        "rcept_no": rcept_no,
    }
    resp = requests.get(DART_DOCUMENT, params=params, timeout=60)

    if resp.headers.get("Content-Type", "").startswith("application/json"):
        data = resp.json()
        if data.get("status") != "000":
            raise ValueError(f"DART error: {data.get('message', 'unknown')}")

    # ZIP 안에 XML 파일들
    try:
        z = zipfile.ZipFile(io.BytesIO(resp.content))
        texts = []
        for name in z.namelist():
            if name.endswith(".xml") or name.endswith(".html"):
                content = z.read(name).decode("utf-8", errors="ignore")
                # HTML 태그 제거
                clean = re.sub(r"<[^>]+>", " ", content)
                clean = re.sub(r"&nbsp;", " ", clean)
                clean = re.sub(r"\s+", " ", clean).strip()
                texts.append(clean)
        return "\n".join(texts)
    except zipfile.BadZipFile:
        # ZIP이 아니면 텍스트로 처리
        text = resp.text
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


def extract_supply_sentences_kr(text: str) -> List[Dict]:
    """한국어 텍스트에서 공급망 관련 문장 추출."""
    results = []
    # 문장 분리 (한국어)
    sentences = re.split(r"(?<=[.。다요])\s+", text)

    for sent in sentences:
        if SUPPLY_PATTERN_KR.search(sent):
            clean = sent.strip()[:500]
            if len(clean) > 20:
                results.append({"sentence": clean})

    return results


def parse_relations_kr(company_id: str, sentences: List[Dict], report_info: Dict) -> List[Dict]:
    """한국어 문장에서 관계 추출."""
    edges = []

    for item in sentences:
        sent = item["sentence"]

        # 알려진 기업명 매칭
        for name, cid in KNOWN_COMPANIES_KR.items():
            if cid == company_id:
                continue
            if name in sent:
                # 방향 결정
                rel = "SUPPLIES"
                if re.search(r"(?:조달|구매|수입|매입|공급받)", sent):
                    rel = "BUYS_FROM"
                if re.search(r"(?:합작|JV|파트너|제휴|협력|공동)", sent):
                    rel = "PARTNERS_WITH"

                edges.append({
                    "src_company_id": company_id if rel == "SUPPLIES" else cid,
                    "rel_type": rel,
                    "dst_company_id": cid if rel == "SUPPLIES" else company_id,
                    "dst_raw_name": name,
                    "confidence": 1.0,
                    "strength": 0.8,
                    "evidence_text": sent[:300],
                    "section": "사업보고서",
                    "filing_type": report_info.get("report_nm", "사업보고서"),
                    "filing_date": report_info.get("rcept_dt", ""),
                    "fiscal_year": report_info.get("fiscal_year", 0),
                    "filing_url": f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={report_info.get('rcept_no', '')}",
                })

        # 매출비중 패턴: "XX사 매출 비중 30%"
        pct_match = re.search(r"([\w가-힣]+(?:사|㈜)?)\s*(?:에\s*대한\s*)?매출\s*비(?:중|율)\s*[:=]?\s*(\d+)[%％]", sent)
        if pct_match:
            customer = pct_match.group(1).strip()
            pct = int(pct_match.group(2))
            if pct >= 10 and customer in KNOWN_COMPANIES_KR:
                edges.append({
                    "src_company_id": company_id,
                    "rel_type": "SUPPLIES",
                    "dst_company_id": KNOWN_COMPANIES_KR[customer],
                    "dst_raw_name": customer,
                    "confidence": 1.0,
                    "strength": min(pct / 100, 1.0),
                    "evidence_text": sent[:300],
                    "section": "매출비중",
                    "filing_type": report_info.get("report_nm", "사업보고서"),
                    "filing_date": report_info.get("rcept_dt", ""),
                    "fiscal_year": report_info.get("fiscal_year", 0),
                    "filing_url": f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={report_info.get('rcept_no', '')}",
                })

    return edges


def collect_all(api_key: str, years: List[int], dry_run: bool = False,
                out_dir: Path = None) -> pd.DataFrame:
    """모든 한국 기업의 사업보고서 수집 + 엣지 추출."""
    if out_dir is None:
        out_dir = Path("data/raw/filings")
    out_dir.mkdir(parents=True, exist_ok=True)

    # corp_code 매핑
    log.info("Loading DART corp_code mapping...")
    try:
        corp_codes = get_corp_code(api_key)
        log.info(f"  Loaded {len(corp_codes)} corp codes")
    except Exception as e:
        log.error(f"  Failed to load corp codes: {e}")
        log.info("  Falling back to manual corp_code lookup")
        corp_codes = {}

    all_edges = []
    all_sentences = []

    for stock_code, info in KR_COMPANIES.items():
        log.info(f"=== {stock_code} ({info['name']}) ===")
        company_id = info["company_id"]

        corp_code = corp_codes.get(stock_code)
        if not corp_code:
            log.warning(f"  No corp_code found for {stock_code}, skipping")
            continue

        try:
            reports = get_annual_reports(api_key, corp_code, years)
            log.info(f"  Found {len(reports)} annual reports")

            if dry_run:
                for r in reports:
                    log.info(f"  [DRY-RUN] {r['report_nm']} FY{r['fiscal_year']} ({r['rcept_dt']})")
                continue

            for report in reports:
                log.info(f"  Processing {report['report_nm']} FY{report['fiscal_year']}")

                try:
                    text = download_report_xml(api_key, report["rcept_no"])
                    log.info(f"    Downloaded: {len(text):,} chars")

                    sentences = extract_supply_sentences_kr(text)
                    log.info(f"    Supply-related sentences: {len(sentences)}")

                    for s in sentences:
                        s["company_id"] = company_id
                        s["stock_code"] = stock_code
                        s["fiscal_year"] = report["fiscal_year"]
                    all_sentences.extend(sentences)

                    edges = parse_relations_kr(company_id, sentences, report)
                    log.info(f"    Extracted edges: {len(edges)}")
                    all_edges.extend(edges)

                except Exception as e:
                    log.error(f"    Failed: {e}")

        except Exception as e:
            log.error(f"  Failed for {stock_code}: {e}")

    # Save
    if all_sentences:
        sent_df = pd.DataFrame(all_sentences)
        sent_path = out_dir / "dart_supply_sentences.parquet"
        sent_df.to_parquet(sent_path, index=False)
        log.info(f"Saved {len(sent_df)} supply sentences → {sent_path}")

    if all_edges:
        df = pd.DataFrame(all_edges)
        dedup_cols = ["src_company_id", "rel_type", "dst_company_id", "fiscal_year"]
        existing = [c for c in dedup_cols if c in df.columns]
        if existing:
            n_before = len(df)
            df = df.drop_duplicates(subset=existing, keep="first")
            log.info(f"Dedup: {n_before} → {len(df)} edges")

        edge_path = out_dir / "dart_edges.parquet"
        df.to_parquet(edge_path, index=False)
        log.info(f"Saved {len(df)} edges → {edge_path}")
        return df
    else:
        log.warning("No edges extracted")
        return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="DART 사업보고서 Supply Chain Edge Extractor")
    parser.add_argument("--api-key", type=str, default=os.environ.get("DART_API_KEY"),
                        help="DART API Key (or set DART_API_KEY env var)")
    parser.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024],
                        help="Fiscal years to collect (default: 2022 2023 2024)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List reports without downloading")
    parser.add_argument("--out-dir", type=str, default="data/raw/filings",
                        help="Output directory")
    args = parser.parse_args()

    if not args.api_key:
        log.error("DART API Key required. Set DART_API_KEY env var or use --api-key")
        log.error("Get your key at: https://opendart.fss.or.kr")
        return

    log.info(f"DART Collector — {len(KR_COMPANIES)} companies, years={args.years}")
    df = collect_all(args.api_key, args.years, args.dry_run, Path(args.out_dir))

    if not df.empty:
        log.info(f"\n=== Summary ===")
        log.info(f"Total edges: {len(df)}")
        if "rel_type" in df.columns:
            log.info(f"By relation:\n{df['rel_type'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
