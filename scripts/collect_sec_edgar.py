#!/usr/bin/env python3
"""
SEC EDGAR 10-K Filing Collector v2
====================================
미국 상장 기업의 10-K 공시에서 공급망 관계를 자동 추출.

v2 개선:
  - BeautifulSoup HTML 파싱 (iXBRL/XBRL 지원)
  - Filing index에서 실제 본문 문서 자동 탐지
  - EDGAR Full-Text Search API fallback
  - 더 넓은 키워드 매칭

사용:
  python scripts/collect_sec_edgar.py
  python scripts/collect_sec_edgar.py --dry-run
  python scripts/collect_sec_edgar.py --years 2024 2025
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
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

log = logging.getLogger("sec_edgar")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

HEADERS = {"User-Agent": "NabiHyoghaw Research nabi@research.edu", "Accept": "*/*"}

# ── Universe 중 미국 상장 기업 ──
US_COMPANIES = {
    "GM":   {"cik": "1467858", "name": "General Motors", "company_id": "GM"},
    "F":    {"cik": "37996",   "name": "Ford Motor", "company_id": "F"},
    "TSLA": {"cik": "1318605", "name": "Tesla Inc", "company_id": "TSLA"},
    "RIVN": {"cik": "1874178", "name": "Rivian Automotive", "company_id": "RIVN"},
    "LCID": {"cik": "1811210", "name": "Lucid Group", "company_id": "LCID"},
    "ALB":  {"cik": "915913",  "name": "Albemarle Corp", "company_id": "ALB"},
    "LICY": {"cik": "1828522", "name": "Li-Cycle Holdings", "company_id": "LICY"},
}

# ── 알려진 기업명 → company_id 매핑 ──
KNOWN_ENTITIES = {
    "catl": "300750.SZ", "contemporary amperex": "300750.SZ",
    "lg energy": "373220.KS", "lg chem": "051910.KS",
    "byd": "1211.HK", "panasonic": "6752.T",
    "samsung sdi": "006400.KS", "sk on": "096770.KS", "sk innovation": "096770.KS",
    "ganfeng": "002460.SZ", "tianqi": "002466.SZ",
    "albemarle": "ALB", "sqm": "SQM", "glencore": "GLEN.L",
    "basf": "BAS.DE", "umicore": "UMI.BR",
    "general motors": "GM", "ford": "F", "tesla": "TSLA",
    "bmw": "BMW.DE", "volkswagen": "VOW3.DE",
    "toyota": "7203.T", "honda": "7267.T",
    "mercedes": "MBG.DE", "rivian": "RIVN", "lucid": "LCID",
    "posco": "003670.KS", "ecopro": "247540.KQ",
    "ultium": "373220.KS",  # Ultium Cells = LG-GM JV
    "huayou": "603799.SH", "cmoc": "603993.SH",
    "eve energy": "300014.SZ", "gotion": "002074.SZ",
    "li-cycle": "LICY", "redwood materials": "REDWOOD",
}


def _sleep(sec=0.15):
    time.sleep(sec)


def get_10k_filings(cik: str, years: List[int]) -> List[Dict]:
    """SEC submissions API로 10-K 파일 목록 조회."""
    url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
    _sleep()
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    filings = []
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])

    for i, form in enumerate(forms):
        if form != "10-K":  # Skip 10-K/A amendments
            continue
        year = int(dates[i][:4])
        if year not in years:
            continue

        filings.append({
            "form": form,
            "filing_date": dates[i],
            "fiscal_year": year - 1,
            "accession": accessions[i],
            "cik": cik,
        })

    return filings


def find_10k_document_url(cik: str, accession: str) -> Optional[str]:
    """Filing index에서 가장 큰 HTM/HTML 파일(=10-K 본문) URL 찾기."""
    acc_clean = accession.replace("-", "")
    idx_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_clean}/index.json"
    _sleep()

    try:
        resp = requests.get(idx_url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None

    best_doc = None
    best_size = 0

    for item in data.get("directory", {}).get("item", []):
        name = item.get("name", "")
        try:
            size = int(item.get("size", "0"))
        except (ValueError, TypeError):
            continue

        if name.lower().endswith((".htm", ".html")) and size > best_size:
            best_size = size
            best_doc = name

    if best_doc and best_size > 50000:
        return f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_clean}/{best_doc}"
    return None


def download_and_parse(url: str) -> str:
    """10-K HTML 다운로드 → BeautifulSoup으로 깨끗한 텍스트 추출."""
    _sleep(0.3)
    resp = requests.get(url, headers=HEADERS, timeout=120)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.content, "lxml")

    # script, style 제거
    for tag in soup(["script", "style", "meta", "link"]):
        tag.decompose()

    # 공백 구분으로 텍스트 추출 (iXBRL 줄바꿈 문제 방지)
    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text)
    return text


def extract_sections(text: str) -> Dict[str, str]:
    """10-K에서 Item 1, 1A, 7 섹션 추출."""
    sections = {}

    patterns = [
        ("item_1_business", r"(?:ITEM|Item)\s+1[\.\s]*[-—:]?\s*(?:BUSINESS|Business)\b"),
        ("item_1a_risk", r"(?:ITEM|Item)\s+1A[\.\s]*[-—:]?\s*(?:RISK|Risk)\s"),
        ("item_2_properties", r"(?:ITEM|Item)\s+2[\.\s]*[-—:]?\s*(?:PROPERTIES|Properties)\b"),
        ("item_7_mda", r"(?:ITEM|Item)\s+7[\.\s]*[-—:]?\s*(?:MANAGEMENT|Management)\b"),
        ("item_7a", r"(?:ITEM|Item)\s+7A[\.\s]*[-—:]?\s*(?:QUANTITATIVE|Quantitative)\b"),
        ("item_8_financial", r"(?:ITEM|Item)\s+8[\.\s]*[-—:]?\s*(?:FINANCIAL|Financial)\b"),
    ]

    # 모든 매칭을 찾고 마지막 것 사용 (목차가 아닌 본문)
    positions = {}
    for name, pat in patterns:
        matches = list(re.finditer(pat, text))
        if matches:
            # 마지막 매칭 = 본문 (앞쪽은 목차)
            positions[name] = matches[-1].start()

    # Item 1: Business
    if "item_1_business" in positions:
        start = positions["item_1_business"]
        end = positions.get("item_1a_risk", positions.get("item_2_properties", start + 80000))
        sections["item_1_business"] = text[start:end][:80000]

    # Item 1A: Risk Factors
    if "item_1a_risk" in positions:
        start = positions["item_1a_risk"]
        end = positions.get("item_2_properties", start + 80000)
        sections["item_1a_risk_factors"] = text[start:end][:80000]

    # Item 7: MD&A
    if "item_7_mda" in positions:
        start = positions["item_7_mda"]
        end = positions.get("item_7a", positions.get("item_8_financial", start + 80000))
        sections["item_7_mda"] = text[start:end][:80000]

    return sections


def find_supply_sentences(sections: Dict[str, str]) -> List[Dict]:
    """섹션에서 공급망 관련 문장 추출."""
    keywords = re.compile(
        r"(?:battery|lithium|cobalt|nickel|cathode|anode|electrolyte|separator|"
        r"cell\s+(?:supplier|supply|manufactur)|raw\s+material|"
        r"supply\s+(?:chain|agreement|contract)|"
        r"(?:major|significant|principal|key)\s+(?:customer|supplier|vendor)|"
        r"accounted?\s+for\s+(?:approximately\s+)?\d+\s*%|"
        r"purchase\s+(?:agreement|obligation|commitment)|"
        r"joint\s+venture|strategic\s+(?:partner|alliance)|"
        r"CATL|LG\s+Energy|Panasonic|Samsung\s+SDI|SK\s+(?:On|Innovation)|"
        r"Ganfeng|Tianqi|Albemarle|Glencore|BASF|Ultium|"
        r"EV\s+battery|electric\s+vehicle\s+battery)",
        re.IGNORECASE
    )

    results = []
    for section_name, text in sections.items():
        # 문장 분리: 마침표/느낌표/물음표 + 공백 + 대문자 시작
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 40 or len(sent) > 1500:
                continue
            if keywords.search(sent):
                results.append({
                    "section": section_name,
                    "sentence": sent[:600],
                })
    return results


def extract_edges(company_id: str, sentences: List[Dict], filing: Dict) -> List[Dict]:
    """문장에서 공급망 엣지 추출."""
    edges = []
    seen = set()

    for item in sentences:
        sent = item["sentence"]
        sent_lower = sent.lower()

        for entity_name, entity_id in KNOWN_ENTITIES.items():
            if entity_id == company_id:
                continue
            if entity_name not in sent_lower:
                continue

            # 관계 방향 결정
            rel = "MENTIONS"  # 기본
            if re.search(r"(?:suppli|provid|sell|deliver)", sent_lower):
                if re.search(r"(?:we|our|the\s+company)\s+(?:suppli|provid|sell|deliver)", sent_lower):
                    rel = "SUPPLIES"  # 우리(filing company)가 공급
                else:
                    rel = "BUYS_FROM"  # 상대가 공급 = 우리가 구매
            if re.search(r"(?:sourc|purchas|procur|buy|obtain|rely\s+on|depend)", sent_lower):
                rel = "BUYS_FROM"
            if re.search(r"(?:joint\s+venture|partner|collaborat|alliance|JV)", sent_lower):
                rel = "PARTNERS_WITH"

            # 방향 설정
            if rel == "SUPPLIES":
                src, dst = company_id, entity_id
            elif rel == "BUYS_FROM":
                src, dst = entity_id, company_id
            elif rel == "PARTNERS_WITH":
                src, dst = company_id, entity_id
            else:
                src, dst = company_id, entity_id

            # 중복 방지 (같은 filing, 같은 관계)
            edge_key = (src, rel, dst, filing["fiscal_year"])
            if edge_key in seen:
                continue
            seen.add(edge_key)

            # strength: % 매출 비중이 있으면 사용
            strength = 0.8
            pct_match = re.search(r"(\d+)\s*%", sent)
            if pct_match:
                pct = int(pct_match.group(1))
                if 5 <= pct <= 100:
                    strength = min(pct / 100, 1.0)

            edges.append({
                "src_company_id": src,
                "rel_type": rel if rel != "MENTIONS" else "SUPPLIES",
                "dst_company_id": dst,
                "confidence": 1.0,
                "strength": round(strength, 2),
                "evidence_text": sent[:400],
                "section": item["section"],
                "filing_type": filing["form"],
                "filing_date": filing["filing_date"],
                "fiscal_year": filing["fiscal_year"],
                "filing_url": filing.get("doc_url", ""),
            })

    return edges


def collect_all(years: List[int], dry_run: bool = False, out_dir: Path = None) -> pd.DataFrame:
    """전체 수집 + 엣지 추출."""
    if out_dir is None:
        out_dir = Path("data/raw/filings")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_edges = []
    all_sentences = []

    for ticker, info in US_COMPANIES.items():
        log.info(f"{'='*50}")
        log.info(f"{ticker} ({info['name']})")
        log.info(f"{'='*50}")

        try:
            filings = get_10k_filings(info["cik"], years)
            log.info(f"  10-K filings found: {len(filings)}")

            if dry_run:
                for f in filings:
                    log.info(f"  [DRY-RUN] FY{f['fiscal_year']} ({f['filing_date']})")
                continue

            for filing in filings:
                log.info(f"  FY{filing['fiscal_year']} ({filing['filing_date']})")

                # 실제 문서 URL 탐색
                doc_url = find_10k_document_url(info["cik"], filing["accession"])
                if not doc_url:
                    log.warning(f"    Could not find 10-K document URL")
                    continue

                filing["doc_url"] = doc_url
                log.info(f"    Document: {doc_url.split('/')[-1]}")

                try:
                    # 다운로드 + 파싱
                    text = download_and_parse(doc_url)
                    log.info(f"    Text: {len(text):,} chars")

                    # 섹션 추출
                    sections = extract_sections(text)
                    log.info(f"    Sections: {list(sections.keys())}")

                    if not sections:
                        # fallback: 전체 텍스트에서 검색
                        log.info(f"    No sections found, searching full text")
                        sections = {"full_text": text[:200000]}

                    # 공급망 문장 추출
                    sentences = find_supply_sentences(sections)
                    log.info(f"    Supply sentences: {len(sentences)}")

                    for s in sentences:
                        s["company_id"] = info["company_id"]
                        s["ticker"] = ticker
                        s["fiscal_year"] = filing["fiscal_year"]
                    all_sentences.extend(sentences)

                    # 엣지 추출
                    edges = extract_edges(info["company_id"], sentences, filing)
                    log.info(f"    Edges: {len(edges)}")
                    all_edges.extend(edges)

                except Exception as e:
                    log.error(f"    Error: {e}")

        except Exception as e:
            log.error(f"  Error: {e}")

    # Save sentences
    if all_sentences:
        df_sent = pd.DataFrame(all_sentences)
        path = out_dir / "sec_supply_sentences.parquet"
        df_sent.to_parquet(path, index=False)
        log.info(f"\nSaved {len(df_sent)} sentences → {path}")

    # Save edges
    if all_edges:
        df = pd.DataFrame(all_edges)
        n_before = len(df)
        df = df.drop_duplicates(
            subset=["src_company_id", "rel_type", "dst_company_id", "fiscal_year"],
            keep="first"
        )
        log.info(f"Dedup: {n_before} → {len(df)} edges")

        path = out_dir / "sec_edges.parquet"
        df.to_parquet(path, index=False)
        log.info(f"Saved {len(df)} edges → {path}")

        # Summary
        log.info(f"\n{'='*50}")
        log.info(f"SUMMARY: {len(df)} edges from {len(US_COMPANIES)} companies")
        if "rel_type" in df.columns:
            log.info(f"By type:\n{df['rel_type'].value_counts().to_string()}")
        if "fiscal_year" in df.columns:
            log.info(f"By year:\n{df['fiscal_year'].value_counts().to_string()}")
        return df
    else:
        log.warning("No edges extracted!")
        return pd.DataFrame()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SEC EDGAR 10-K Supply Chain Extractor v2")
    parser.add_argument("--years", type=int, nargs="+", default=[2024, 2025],
                        help="Filing years (default: 2024 2025)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=str, default="data/raw/filings")
    args = parser.parse_args()

    collect_all(args.years, args.dry_run, Path(args.out_dir))
