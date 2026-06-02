"""
collect_supplement_news.py
──────────────────────────
통합 보완 뉴스 수집기: Google News RSS + 영문 산업 뉴스 사이트.

Usage:
  python scripts/collect_supplement_news.py                    # 전체 수집
  python scripts/collect_supplement_news.py --dry-run          # 수집만, 병합 안 함
  python scripts/collect_supplement_news.py --source rss       # RSS만
  python scripts/collect_supplement_news.py --source sites     # 사이트만
  python scripts/collect_supplement_news.py --company 600110.SH  # 특정 기업만

대상: GDELT 매칭 부족 기업 9개
  600110.SH (Nuode), 066970.KS (L&F), 300919.SZ (CNGR),
  LICY (Li-Cycle), 920185.BJ (BTR), 688388.SH (Jiayuan),
  603659.SH (Putailai), 3931.HK (CALB), 300037.SZ (Capchem)

소스:
  1. Google News RSS — 각 기업별 맞춤 쿼리
  2. CnEVPost — 중국 EV 산업 영문 전문
  3. Battery-News.de — 유럽 배터리 뉴스
  4. SMM (Shanghai Metals Market) — 소재 산업
  5. Yicai Global — 중국 산업 뉴스 영문판
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote_plus, urlparse

import feedparser
import pandas as pd
import requests
import yaml
from bs4 import BeautifulSoup

log = logging.getLogger("supplement_collector")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

# ─── Paths ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
ALIASES_YAML = ROOT / "configs" / "company_aliases.yaml"
RISK_KEYWORDS_YAML = ROOT / "configs" / "risk_keywords.yaml"
OUTPUT_DIR = ROOT / "data" / "raw" / "supplement"
RISK_EVENTS_PATH = ROOT / "data" / "processed" / "risk_events.parquet"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}
REQUEST_DELAY = 2.0  # seconds between requests


# ─── Target Companies & Queries ──────────────────────────────────

TARGET_QUERIES: Dict[str, Dict] = {
    "600110.SH": {
        "name": "Nuode",
        "rss_queries": [
            '"Nuode" copper foil',
            '"Nuode Investment" battery',
            '"诺德股份" copper',
        ],
        "site_keywords": ["Nuode", "诺德"],
    },
    "066970.KS": {
        "name": "L&F",
        "rss_queries": [
            '"L&F Co" cathode',
            '"LF Co" battery cathode',
            '"엘앤에프" battery',
            '"L and F" cathode material',
        ],
        "site_keywords": ["L&F", "LF Co", "엘앤에프"],
    },
    "300919.SZ": {
        "name": "CNGR",
        "rss_queries": [
            '"CNGR Advanced Material"',
            '"CNGR" cathode precursor',
            '"中伟新材" battery',
        ],
        "site_keywords": ["CNGR", "中伟"],
    },
    "LICY": {
        "name": "Li-Cycle",
        "rss_queries": [
            '"Li-Cycle" battery recycling',
            '"Li-Cycle Holdings" lithium',
        ],
        "site_keywords": ["Li-Cycle"],
    },
    "920185.BJ": {
        "name": "BTR",
        "rss_queries": [
            '"BTR New Material" anode',
            '"BTR" battery anode',
            '"贝特瑞" battery',
        ],
        "site_keywords": ["BTR New Material", "贝特瑞"],
    },
    "688388.SH": {
        "name": "Jiayuan",
        "rss_queries": [
            '"Jiayuan Technology" copper foil',
            '"嘉元科技" copper',
            '"Jiayuan" electrolytic copper',
        ],
        "site_keywords": ["Jiayuan", "嘉元"],
    },
    "603659.SH": {
        "name": "Putailai",
        "rss_queries": [
            '"Putailai" anode material',
            '"Putailai New Energy"',
            '"璞泰来" battery',
        ],
        "site_keywords": ["Putailai", "璞泰来"],
    },
    "3931.HK": {
        "name": "CALB",
        "rss_queries": [
            '"CALB" battery',
            '"China Aviation Lithium Battery"',
            '"CALB Group" EV',
            '"中创新航" battery',
        ],
        "site_keywords": ["CALB", "中创新航"],
    },
    "300037.SZ": {
        "name": "Capchem",
        "rss_queries": [
            '"Capchem" electrolyte',
            '"Capchem Technology" battery',
            '"新宙邦" electrolyte',
        ],
        "site_keywords": ["Capchem", "新宙邦"],
    },
}

# ─── Industry News Sites ─────────────────────────────────────────

SITE_CONFIGS = {
    "cnevpost": {
        "name": "CnEVPost",
        "base_url": "https://cnevpost.com/?s={query}",
        "article_selector": "h2.entry-title a",
        "date_selector": "time.entry-date",
        "domain": "cnevpost.com",
    },
    "battery_news": {
        "name": "Battery-News.de",
        "base_url": "https://battery-news.de/?s={query}",
        "article_selector": "h2.entry-title a",
        "date_selector": "time.entry-date",
        "domain": "battery-news.de",
    },
    "yicai": {
        "name": "Yicai Global",
        "base_url": "https://www.yicaiglobal.com/search?keyword={query}",
        "article_selector": "a.search-result-title",
        "date_selector": "span.search-result-date",
        "domain": "yicaiglobal.com",
    },
}


# ─── Risk Keyword Matching ───────────────────────────────────────

def _load_risk_patterns(yaml_path: Path) -> List[Tuple[str, re.Pattern]]:
    """risk_keywords.yaml -> (risk_type, compiled_pattern) list."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    patterns = []
    for risk_type, section in data.items():
        if not isinstance(section, dict) or "keywords" not in section:
            continue
        for kw in section["keywords"]:
            pat_str = kw.get("pattern", kw.get("term", ""))
            if pat_str:
                try:
                    patterns.append((risk_type, re.compile(pat_str, re.IGNORECASE)))
                except re.error:
                    pass
    return patterns


def classify_risk(text: str, risk_patterns: List[Tuple[str, re.Pattern]]) -> str:
    """텍스트에서 risk_type 분류."""
    matched = set()
    for risk_type, pat in risk_patterns:
        if pat.search(text):
            matched.add(risk_type)
    return ",".join(sorted(matched)) if matched else "other"


def estimate_severity(text: str, risk_patterns: List[Tuple[str, re.Pattern]]) -> float:
    """매칭된 키워드 수 기반 severity 추정."""
    count = sum(1 for _, pat in risk_patterns if pat.search(text))
    if count == 0:
        return 1.5  # base severity
    return min(1.5 + count * 0.5, 5.0)


# ─── Event Builder ───────────────────────────────────────────────

def make_event_id(url: str, company_id: str) -> str:
    """Deterministic event ID."""
    return hashlib.sha256(f"{url}|{company_id}".encode()).hexdigest()[:16]


def build_event(
    title: str,
    url: str,
    pub_time: Optional[datetime],
    company_id: str,
    company_name: str,
    source_domain: str,
    collection_type: str,
    risk_patterns: List[Tuple[str, re.Pattern]],
) -> Optional[dict]:
    """Build one risk_event row."""
    if not url or not title:
        return None
    if pub_time is None:
        pub_time = datetime.now()

    risk_types = classify_risk(title, risk_patterns)
    severity = estimate_severity(title, risk_patterns)

    return {
        "event_id": make_event_id(url, company_id),
        "event_time": pd.Timestamp(pub_time),
        "url": url,
        "title": title,
        "risk_types": risk_types,
        "severity": severity,
        "entity_ids": [company_id],
        "entity_scores": [{"alias": company_name, "company_id": company_id, "score": 1.0}],
        "finbert": None,
        "country_ids": "",
        "tone": 0.0,
        "source_domain": source_domain,
        "collection_type": collection_type,
    }


# ─── Source 1: Google News RSS ───────────────────────────────────

RSS_BASE = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"


def collect_rss(
    targets: Dict[str, Dict],
    risk_patterns: List[Tuple[str, re.Pattern]],
) -> List[dict]:
    """Google News RSS 수집."""
    rows = []
    seen_urls = set()

    for cid, info in targets.items():
        name = info["name"]
        queries = info["rss_queries"]
        log.info(f"[RSS] {cid} ({name}) — {len(queries)} queries")

        count = 0
        for query in queries:
            url = RSS_BASE.format(query=quote_plus(query))
            try:
                feed = feedparser.parse(url)
            except Exception as e:
                log.warning(f"  RSS failed: {e}")
                continue

            for entry in feed.entries:
                link = entry.get("link", "")
                if link in seen_urls:
                    continue
                seen_urls.add(link)

                # Parse time
                pub_time = None
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    pub_time = datetime(*entry.published_parsed[:6])

                # Source domain
                source = ""
                if hasattr(entry, "source") and hasattr(entry.source, "title"):
                    source = entry.source.title
                else:
                    source = urlparse(link).netloc

                row = build_event(
                    title=entry.get("title", ""),
                    url=link,
                    pub_time=pub_time,
                    company_id=cid,
                    company_name=name,
                    source_domain=source,
                    collection_type="supplement_google_rss",
                    risk_patterns=risk_patterns,
                )
                if row:
                    rows.append(row)
                    count += 1

            time.sleep(REQUEST_DELAY)

        log.info(f"  → {count} articles")

    return rows


# ─── Source 2: Industry News Sites ───────────────────────────────

def scrape_site(
    site_key: str,
    query: str,
    company_id: str,
    company_name: str,
    risk_patterns: List[Tuple[str, re.Pattern]],
) -> List[dict]:
    """산업 뉴스 사이트 검색 결과 스크래핑."""
    config = SITE_CONFIGS[site_key]
    search_url = config["base_url"].format(query=quote_plus(query))

    try:
        resp = requests.get(search_url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        log.warning(f"  [{config['name']}] Request failed: {e}")
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    articles = soup.select(config["article_selector"])

    rows = []
    for a_tag in articles[:20]:  # max 20 per query
        title = a_tag.get_text(strip=True)
        url = a_tag.get("href", "")
        if not url.startswith("http"):
            url = f"https://{config['domain']}{url}"

        # Try to find date
        pub_time = None
        parent = a_tag.find_parent()
        if parent:
            time_tag = parent.find("time")
            if time_tag and time_tag.get("datetime"):
                try:
                    pub_time = datetime.fromisoformat(
                        time_tag["datetime"].replace("Z", "+00:00")
                    )
                except ValueError:
                    pass

        row = build_event(
            title=title,
            url=url,
            pub_time=pub_time,
            company_id=company_id,
            company_name=company_name,
            source_domain=config["domain"],
            collection_type=f"supplement_{site_key}",
            risk_patterns=risk_patterns,
        )
        if row:
            rows.append(row)

    return rows


def collect_sites(
    targets: Dict[str, Dict],
    risk_patterns: List[Tuple[str, re.Pattern]],
) -> List[dict]:
    """산업 뉴스 사이트 크롤링."""
    rows = []
    seen_urls = set()

    for cid, info in targets.items():
        name = info["name"]
        keywords = info.get("site_keywords", [name])
        log.info(f"[SITES] {cid} ({name}) — {len(keywords)} keywords × {len(SITE_CONFIGS)} sites")

        count = 0
        for kw in keywords:
            for site_key in SITE_CONFIGS:
                site_rows = scrape_site(site_key, kw, cid, name, risk_patterns)
                for r in site_rows:
                    if r["url"] not in seen_urls:
                        seen_urls.add(r["url"])
                        rows.append(r)
                        count += 1
                time.sleep(REQUEST_DELAY)

        log.info(f"  → {count} articles")

    return rows


# ─── Merge & Save ────────────────────────────────────────────────

def save_supplement(
    rows: List[dict],
    output_path: Path,
) -> pd.DataFrame:
    """보완 데이터 저장."""
    if not rows:
        log.warning("No data collected!")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("event_time").drop_duplicates(subset=["event_id"], keep="first")
    df = df.reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    log.info(f"Saved: {len(df):,} articles → {output_path}")

    # Summary
    log.info("=== Collection Summary ===")
    for cid in df["entity_ids"].apply(lambda x: x[0] if x else "").unique():
        sub = df[df["entity_ids"].apply(lambda x: x[0] if x else "") == cid]
        log.info(f"  {cid}: {len(sub)} articles")

    rt_counts = df["risk_types"].value_counts()
    log.info("Risk type distribution:")
    for rt, cnt in rt_counts.head(10).items():
        log.info(f"  {rt}: {cnt}")

    ct_counts = df["collection_type"].value_counts()
    log.info("Source distribution:")
    for ct, cnt in ct_counts.items():
        log.info(f"  {ct}: {cnt}")

    return df


def merge_into_risk_events(
    supplement_path: Path,
    risk_events_path: Path,
) -> None:
    """보완 데이터를 기존 risk_events.parquet에 병합."""
    if not supplement_path.exists():
        log.error(f"Supplement file not found: {supplement_path}")
        return

    supplement = pd.read_parquet(supplement_path)
    log.info(f"Supplement: {len(supplement):,} rows")

    if risk_events_path.exists():
        existing = pd.read_parquet(risk_events_path)
        log.info(f"Existing risk_events: {len(existing):,} rows")

        merged = pd.concat([existing, supplement], ignore_index=True)
        before = len(merged)
        merged = merged.drop_duplicates(subset=["url"], keep="first")
        log.info(f"Merged: {before:,} → {len(merged):,} (dedup {before - len(merged)})")
    else:
        merged = supplement
        log.info(f"No existing risk_events, using supplement only")

    merged.to_parquet(risk_events_path, index=False)
    log.info(f"Updated: {risk_events_path}")


# ─── CLI ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Supplement news collector (RSS + Sites)")
    parser.add_argument("--dry-run", action="store_true", help="Collect only, don't merge")
    parser.add_argument("--source", choices=["rss", "sites", "all"], default="all",
                        help="Source to collect from")
    parser.add_argument("--company", type=str, help="Specific company ID (e.g., 600110.SH)")
    parser.add_argument("--merge", action="store_true",
                        help="Merge supplement into risk_events.parquet")
    args = parser.parse_args()

    # Load risk patterns
    risk_patterns = _load_risk_patterns(RISK_KEYWORDS_YAML)
    log.info(f"Risk patterns: {len(risk_patterns)}")

    # Filter targets
    targets = TARGET_QUERIES.copy()
    if args.company:
        if args.company in targets:
            targets = {args.company: targets[args.company]}
        else:
            log.error(f"Unknown company: {args.company}")
            log.info(f"Available: {list(TARGET_QUERIES.keys())}")
            return

    # Collect
    all_rows = []

    if args.source in ("rss", "all"):
        log.info("=" * 60)
        log.info("Phase 1: Google News RSS")
        log.info("=" * 60)
        rss_rows = collect_rss(targets, risk_patterns)
        all_rows.extend(rss_rows)
        log.info(f"RSS total: {len(rss_rows)} articles")

    if args.source in ("sites", "all"):
        log.info("=" * 60)
        log.info("Phase 2: Industry News Sites")
        log.info("=" * 60)
        site_rows = collect_sites(targets, risk_patterns)
        all_rows.extend(site_rows)
        log.info(f"Sites total: {len(site_rows)} articles")

    # Save
    output_path = OUTPUT_DIR / "supplement_news.parquet"
    df = save_supplement(all_rows, output_path)

    if df.empty:
        return

    if args.dry_run:
        log.info("[DRY-RUN] Preview:")
        print(df[["event_time", "title", "risk_types", "severity", "collection_type"]]
              .head(30).to_string())
        return

    # Merge if requested
    if args.merge:
        merge_into_risk_events(output_path, RISK_EVENTS_PATH)

    log.info("Done!")


if __name__ == "__main__":
    main()
