#!/usr/bin/env python3
"""
SEC EDGAR 10-K Filing Collector
================================
미국 상장 기업의 10-K/10-F 공시에서 공급망 관계를 자동 추출.

대상 기업 (universe 중 미국 상장):
  - GM, F (Ford), TSLA, RIVN, LCID, ALB, LICY

수집 대상:
  - 10-K Item 1 (Business), Item 1A (Risk Factors), Item 7 (MD&A)
  - "Major Customers", "Supply Agreements", "Raw Material" 섹션

출력:
  data/raw/filings/sec_edges.parquet — 공시 기반 공급망 엣지

사용:
  python scripts/collect_sec_edgar.py
  python scripts/collect_sec_edgar.py --dry-run    # 수집만, 파싱 안 함
  python scripts/collect_sec_edgar.py --years 2022 2023 2024
"""
import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

log = logging.getLogger("sec_edgar")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

# ── SEC EDGAR API config ──
EDGAR_BASE = "https://efts.sec.gov/LATEST"
EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions"
EDGAR_FULL_TEXT = "https://efts.sec.gov/LATEST/search-index"
HEADERS = {
    "User-Agent": "NabiHyoghaw Research Project nabi@research.edu",
    "Accept": "application/json",
}

# ── 우리 Universe 중 미국 상장 기업 ──
US_COMPANIES = {
    "GM":   {"cik": "0001467858", "name": "General Motors", "company_id": "GM"},
    "F":    {"cik": "0000037996", "name": "Ford Motor", "company_id": "F"},
    "TSLA": {"cik": "0001318605", "name": "Tesla Inc", "company_id": "TSLA"},
    "RIVN": {"cik": "0001874178", "name": "Rivian Automotive", "company_id": "RIVN"},
    "LCID": {"cik": "0001811210", "name": "Lucid Group", "company_id": "LCID"},
    "ALB":  {"cik": "0000915913", "name": "Albemarle Corp", "company_id": "ALB"},
    "LICY": {"cik": "0001828522", "name": "Li-Cycle Holdings", "company_id": "LICY"},
}

# ── 공급망 키워드 ──
SUPPLY_KEYWORDS = [
    r"(?:major|significant|principal|key|largest)\s+(?:customer|client|buyer)s?",
    r"(?:supply|supplier|supplies|supplying)\s+(?:agreement|contract|arrangement)s?",
    r"(?:raw\s+material|lithium|cobalt|nickel|graphite|copper|cathode|anode|electrolyte|separator)",
    r"(?:battery\s+(?:cell|pack|module|supplier|supply|material))",
    r"accounted?\s+for\s+(?:approximately\s+)?\d+%",
    r"(?:purchase|procurement|sourcing)\s+(?:agreement|contract|obligation)s?",
    r"(?:joint\s+venture|partnership|strategic\s+alliance|collaboration)",
    r"(?:CATL|LG\s+Energy|BYD|Panasonic|Samsung\s+SDI|SK\s+Innovation|SK\s+On)",
    r"(?:Ganfeng|Tianqi|Albemarle|SQM|Glencore|BASF|Umicore)",
]
SUPPLY_PATTERN = re.compile("|".join(SUPPLY_KEYWORDS), re.IGNORECASE)

# ── 관계 추출 패턴 ──
RELATION_PATTERNS = [
    # "X supplies batteries to Y" / "X is a supplier to Y"
    (r"(?P<supplier>[\w\s&.]+?)\s+(?:supplies|supply|supplier\s+to|provides?\s+(?:batteries|materials|lithium|cobalt|cathode|anode|electrolyte))\s+(?:to\s+)?(?P<customer>[\w\s&.]+)",
     "SUPPLIES"),
    # "X sources from Y" / "X procures from Y"
    (r"(?P<customer>[\w\s&.]+?)\s+(?:sources?|procures?|purchases?|obtains?)\s+(?:from|through)\s+(?P<supplier>[\w\s&.]+)",
     "BUYS_FROM"),
    # "X accounted for N% of revenue"
    (r"(?P<customer>[\w\s&.]+?)\s+accounted?\s+for\s+(?:approximately\s+)?(?P<pct>\d+)%",
     "MAJOR_CUSTOMER"),
    # "joint venture with X" / "partnership with X"
    (r"(?:joint\s+venture|partnership|strategic\s+alliance|collaboration)\s+(?:with|between)\s+(?P<partner>[\w\s&.]+)",
     "PARTNERS_WITH"),
]


def _sec_sleep():
    """SEC EDGAR rate limit: 10 requests/second max."""
    time.sleep(0.15)


def get_filing_urls(cik: str, filing_type: str = "10-K", years: List[int] = None) -> List[Dict]:
    """
    SEC EDGAR API로 특정 기업의 10-K 파일 URL 목록 조회.
    """
    cik_padded = cik.lstrip("0").zfill(10)
    url = f"{EDGAR_SUBMISSIONS}/CIK{cik_padded}.json"

    _sec_sleep()
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    filings = []
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    for i, form in enumerate(forms):
        if form not in (filing_type, f"{filing_type}/A"):
            continue

        filing_date = dates[i]
        year = int(filing_date[:4])

        if years and year not in years:
            continue

        accession = accessions[i].replace("-", "")
        doc = primary_docs[i]
        doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession}/{doc}"

        filings.append({
            "form": form,
            "filing_date": filing_date,
            "fiscal_year": year - 1,  # 10-K filed in 2024 is for FY2023
            "accession": accessions[i],
            "url": doc_url,
        })

    return filings


def download_filing_text(url: str) -> str:
    """10-K HTML/text 다운로드."""
    _sec_sleep()
    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()

    text = resp.text
    # HTML 태그 제거 (간단한 처리)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_relevant_sections(full_text: str) -> Dict[str, str]:
    """
    10-K에서 Item 1, Item 1A, Item 7 섹션 추출.
    """
    sections = {}

    # Item 1 - Business
    m1 = re.search(r"(?:ITEM\s+1[\.\s]*[-—]?\s*BUSINESS)", full_text, re.IGNORECASE)
    m1a = re.search(r"(?:ITEM\s+1A[\.\s]*[-—]?\s*RISK\s+FACTORS)", full_text, re.IGNORECASE)
    m2 = re.search(r"(?:ITEM\s+2[\.\s]*[-—]?\s*PROPERTIES)", full_text, re.IGNORECASE)
    m7 = re.search(r"(?:ITEM\s+7[\.\s]*[-—]?\s*MANAGEMENT)", full_text, re.IGNORECASE)
    m7a = re.search(r"(?:ITEM\s+7A[\.\s]*[-—]?\s*QUANTITATIVE)", full_text, re.IGNORECASE)
    m8 = re.search(r"(?:ITEM\s+8[\.\s]*[-—]?\s*FINANCIAL\s+STATEMENTS)", full_text, re.IGNORECASE)

    if m1:
        end = m1a.start() if m1a else (m2.start() if m2 else m1.start() + 50000)
        sections["item_1_business"] = full_text[m1.start():end][:50000]

    if m1a:
        end = m2.start() if m2 else m1a.start() + 50000
        sections["item_1a_risk_factors"] = full_text[m1a.start():end][:50000]

    if m7:
        end = m7a.start() if m7a else (m8.start() if m8 else m7.start() + 50000)
        sections["item_7_mda"] = full_text[m7.start():end][:50000]

    return sections


def extract_supply_sentences(sections: Dict[str, str]) -> List[Dict]:
    """
    관련 섹션에서 공급망 키워드가 포함된 문장 추출.
    """
    results = []

    for section_name, text in sections.items():
        # 문장 분리
        sentences = re.split(r"(?<=[.!?])\s+", text)

        for sent in sentences:
            if SUPPLY_PATTERN.search(sent):
                # 문장이 너무 길면 자르기
                clean = sent.strip()[:500]
                if len(clean) > 30:  # 너무 짧은 건 스킵
                    results.append({
                        "section": section_name,
                        "sentence": clean,
                    })

    return results


def parse_relations(company_id: str, company_name: str,
                    sentences: List[Dict], filing_info: Dict) -> List[Dict]:
    """
    추출된 문장에서 관계(엣지) 파싱.
    간단한 패턴 매칭 + 키워드 기반.
    """
    edges = []

    for item in sentences:
        sent = item["sentence"]
        section = item["section"]

        # Major customer pattern: "X accounted for N% of revenue"
        pct_match = re.search(r"(\w[\w\s&.]*?)\s+accounted?\s+for\s+(?:approximately\s+)?(\d+)%", sent, re.IGNORECASE)
        if pct_match:
            customer_name = pct_match.group(1).strip()
            pct = int(pct_match.group(2))
            if pct >= 10:
                edges.append({
                    "src_company_id": company_id,
                    "rel_type": "SUPPLIES",
                    "dst_raw_name": customer_name,
                    "confidence": 1.0,
                    "strength": min(pct / 100, 1.0),
                    "evidence_text": sent[:300],
                    "section": section,
                    "filing_type": filing_info["form"],
                    "filing_date": filing_info["filing_date"],
                    "fiscal_year": filing_info["fiscal_year"],
                    "filing_url": filing_info["url"],
                })

        # Known battery/material company mentions
        known_companies = {
            "CATL": "300750.SZ", "Contemporary Amperex": "300750.SZ",
            "LG Energy": "373220.KS", "LG Chem": "051910.KS",
            "BYD": "1211.HK", "Panasonic": "6752.T",
            "Samsung SDI": "006400.KS", "SK Innovation": "096770.KS", "SK On": "096770.KS",
            "Ganfeng": "002460.SZ", "Tianqi": "002466.SZ",
            "Albemarle": "ALB", "SQM": "SQM", "Glencore": "GLEN.L",
            "BASF": "BAS.DE", "Umicore": "UMI.BR",
            "General Motors": "GM", "Ford": "F", "Tesla": "TSLA",
            "BMW": "BMW.DE", "Volkswagen": "VOW3.DE", "VW": "VOW3.DE",
            "Toyota": "7203.T", "Honda": "7267.T",
            "Mercedes": "MBG.DE", "Rivian": "RIVN", "Lucid": "LCID",
            "POSCO": "003670.KS", "EcoPro": "247540.KQ",
            "Huayou": "603799.SH", "CMOC": "603993.SH",
        }

        for name, cid in known_companies.items():
            if cid == company_id:
                continue
            if name.lower() in sent.lower():
                # 방향 결정: 이 기업이 supplier인지 customer인지
                rel = "SUPPLIES" if re.search(r"(?:suppli|provid|sell)", sent, re.IGNORECASE) else "BUYS_FROM"
                if re.search(r"(?:sourc|purchas|procur|buy|obtain)", sent, re.IGNORECASE):
                    rel = "BUYS_FROM"
                if re.search(r"(?:joint venture|partner|collaborat|alliance)", sent, re.IGNORECASE):
                    rel = "PARTNERS_WITH"

                edges.append({
                    "src_company_id": company_id if rel == "SUPPLIES" else cid,
                    "rel_type": rel,
                    "dst_company_id": cid if rel == "SUPPLIES" else company_id,
                    "dst_raw_name": name,
                    "confidence": 1.0,
                    "strength": 0.8,
                    "evidence_text": sent[:300],
                    "section": section,
                    "filing_type": filing_info["form"],
                    "filing_date": filing_info["filing_date"],
                    "fiscal_year": filing_info["fiscal_year"],
                    "filing_url": filing_info["url"],
                })

    return edges


def collect_all(years: List[int], dry_run: bool = False, out_dir: Path = None) -> pd.DataFrame:
    """
    모든 미국 기업의 10-K를 수집하고 공급망 엣지를 추출.
    """
    if out_dir is None:
        out_dir = Path("data/raw/filings")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_edges = []
    all_sentences = []

    for ticker, info in US_COMPANIES.items():
        log.info(f"=== {ticker} ({info['name']}) ===")
        cik = info["cik"]
        company_id = info["company_id"]

        try:
            filings = get_filing_urls(cik, "10-K", years)
            log.info(f"  Found {len(filings)} 10-K filings")

            if dry_run:
                for f in filings:
                    log.info(f"  [DRY-RUN] {f['form']} FY{f['fiscal_year']} ({f['filing_date']})")
                continue

            for filing in filings:
                log.info(f"  Processing {filing['form']} FY{filing['fiscal_year']} ({filing['filing_date']})")

                try:
                    text = download_filing_text(filing["url"])
                    log.info(f"    Downloaded: {len(text):,} chars")

                    sections = extract_relevant_sections(text)
                    log.info(f"    Sections found: {list(sections.keys())}")

                    sentences = extract_supply_sentences(sections)
                    log.info(f"    Supply-related sentences: {len(sentences)}")

                    for s in sentences:
                        s["company_id"] = company_id
                        s["ticker"] = ticker
                        s["fiscal_year"] = filing["fiscal_year"]
                    all_sentences.extend(sentences)

                    edges = parse_relations(company_id, info["name"], sentences, filing)
                    log.info(f"    Extracted edges: {len(edges)}")
                    all_edges.extend(edges)

                except Exception as e:
                    log.error(f"    Failed: {e}")

        except Exception as e:
            log.error(f"  Failed to get filings for {ticker}: {e}")

    # Save raw sentences for review
    if all_sentences:
        sent_df = pd.DataFrame(all_sentences)
        sent_path = out_dir / "sec_supply_sentences.parquet"
        sent_df.to_parquet(sent_path, index=False)
        log.info(f"Saved {len(sent_df)} supply sentences → {sent_path}")

    # Save edges
    if all_edges:
        df = pd.DataFrame(all_edges)

        # Dedup
        dedup_cols = ["src_company_id", "rel_type", "dst_company_id", "fiscal_year"]
        existing_cols = [c for c in dedup_cols if c in df.columns]
        if existing_cols:
            n_before = len(df)
            df = df.drop_duplicates(subset=existing_cols, keep="first")
            log.info(f"Dedup: {n_before} → {len(df)} edges")

        edge_path = out_dir / "sec_edges.parquet"
        df.to_parquet(edge_path, index=False)
        log.info(f"Saved {len(df)} edges → {edge_path}")
        return df
    else:
        log.warning("No edges extracted")
        return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="SEC EDGAR 10-K Supply Chain Edge Extractor")
    parser.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025],
                        help="Filing years to collect (default: 2023 2024 2025)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List filings without downloading")
    parser.add_argument("--out-dir", type=str, default="data/raw/filings",
                        help="Output directory")
    args = parser.parse_args()

    log.info(f"SEC EDGAR Collector — {len(US_COMPANIES)} companies, years={args.years}")
    df = collect_all(args.years, args.dry_run, Path(args.out_dir))

    if not df.empty:
        log.info(f"\n=== Summary ===")
        log.info(f"Total edges: {len(df)}")
        if "rel_type" in df.columns:
            log.info(f"By relation:\n{df['rel_type'].value_counts().to_string()}")
        if "fiscal_year" in df.columns:
            log.info(f"By fiscal year:\n{df['fiscal_year'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
