"""
collect_google_news_rss.py
──────────────────────────
Google News RSS 자동 수집 → risk_events 스키마 변환 → 기존 데이터 병합.

Usage:
  python scripts/collect_google_news_rss.py                  # 전체 수집
  python scripts/collect_google_news_rss.py --dry-run        # 수집만, 병합 안 함
  python scripts/collect_google_news_rss.py --company 600110.SH  # 특정 기업만

대상: GDELT 매칭이 부족한 7개 기업
  600110.SH (Nuode), 300919.SZ (CNGR), LICY (Li-Cycle),
  688388.SH (Jiayuan), 603659.SH (Putailai), 3931.HK (CALB),
  300037.SZ (Capchem)
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import feedparser
import pandas as pd
import yaml

log = logging.getLogger("rss_collector")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

# ─── Config ────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
ALIASES_YAML = ROOT / "configs" / "company_aliases.yaml"
RISK_KEYWORDS_YAML = ROOT / "configs" / "risk_keywords.yaml"
CHECKPOINT_DIR = ROOT / "data" / "raw" / "gdelt" / "risk_events_checkpoint"
OUTPUT_PATH = ROOT / "data" / "raw" / "supplement" / "google_news_rss.parquet"

# 데이터 부족 기업 + 검색 쿼리 (영문)
TARGET_QUERIES: Dict[str, List[str]] = {
    "600110.SH": [
        '"Nuode" copper foil',
        '"Nuode Investment" battery',
        '"诺德股份" OR "诺德投资"',
    ],
    "300919.SZ": [
        '"CNGR Advanced Material"',
        '"CNGR" cathode precursor',
        '"中伟新材" OR "中伟股份"',
    ],
    "LICY": [
        '"Li-Cycle" battery recycling',
        '"Li-Cycle Holdings"',
    ],
    "688388.SH": [
        '"Jiayuan Technology" copper foil',
        '"嘉元科技"',
    ],
    "603659.SH": [
        '"Putailai" anode',
        '"Putailai New Energy"',
        '"璞泰来"',
    ],
    "3931.HK": [
        '"CALB" battery',
        '"China Aviation Lithium Battery"',
        '"CALB Group"',
        '"中创新航"',
    ],
    "300037.SZ": [
        '"Capchem" electrolyte',
        '"Capchem Technology"',
        '"新宙邦"',
    ],
}

RSS_BASE = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
REQUEST_DELAY = 2.0  # seconds between requests (rate limiting)


# ─── Risk keyword matching ─────────────────────────────────────────

def _load_risk_patterns(yaml_path: Path) -> List[Tuple[str, re.Pattern]]:
    """risk_keywords.yaml에서 (risk_type, compiled_pattern) 리스트 로드."""
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


def classify_risk(title: str, risk_patterns: List[Tuple[str, re.Pattern]]) -> str:
    """제목에서 risk_type 분류. 여러 개면 comma-separated."""
    matched = set()
    for risk_type, pat in risk_patterns:
        if pat.search(title):
            matched.add(risk_type)
    return ",".join(sorted(matched)) if matched else "other"


def estimate_severity(title: str, risk_patterns: List[Tuple[str, re.Pattern]]) -> float:
    """제목 기반 severity 추정 (0.0~1.0). 매칭된 키워드가 많을수록 높음."""
    import math
    total_weight = 0
    for risk_type, pat in risk_patterns:
        if pat.search(title):
            total_weight += 1
    if total_weight == 0:
        return 0.3  # default for unclassified
    return min(0.3 + total_weight * 0.15, 1.0)


# ─── RSS Fetching ──────────────────────────────────────────────────

def fetch_rss(query: str) -> List[dict]:
    """Google News RSS 피드에서 기사 목록 가져오기."""
    url = RSS_BASE.format(query=quote_plus(query))
    feed = feedparser.parse(url)

    articles = []
    for entry in feed.entries:
        pub_time = None
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            pub_time = datetime(*entry.published_parsed[:6])
        elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
            pub_time = datetime(*entry.updated_parsed[:6])

        # source 추출
        source = ""
        if hasattr(entry, "source") and hasattr(entry.source, "title"):
            source = entry.source.title
        elif hasattr(entry, "source"):
            source = getattr(entry.source, "href", "")

        articles.append({
            "title": entry.get("title", ""),
            "url": entry.get("link", ""),
            "published": pub_time,
            "source": source,
        })

    return articles


def make_event_id(url: str, company_id: str) -> str:
    """URL + company_id 기반 deterministic ID."""
    raw = f"{url}|{company_id}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def build_risk_event(
    article: dict,
    company_id: str,
    alias_matched: str,
    risk_patterns: List[Tuple[str, re.Pattern]],
) -> Optional[dict]:
    """RSS 기사 1건 → risk_events 스키마 row."""
    title = article["title"]
    url = article["url"]
    pub_time = article.get("published")

    if not url or not title:
        return None
    if pub_time is None:
        pub_time = datetime.now()

    risk_types = classify_risk(title, risk_patterns)
    severity = estimate_severity(title, risk_patterns)

    # source domain 추출
    source_domain = article.get("source", "")
    if not source_domain and url:
        from urllib.parse import urlparse
        source_domain = urlparse(url).netloc

    return {
        "event_id": make_event_id(url, company_id),
        "event_time": pd.Timestamp(pub_time),
        "url": url,
        "title": title,
        "risk_types": risk_types,
        "severity": severity,
        "entity_ids": [company_id],
        "entity_scores": [{"alias": alias_matched, "company_id": company_id, "score": 1.0}],
        "finbert": None,
        "country_ids": "",
        "tone": 0.0,
        "source_domain": source_domain,
        "collection_type": "supplement_google_news",
    }


# ─── Main Collection ──────────────────────────────────────────────

def collect_all(
    targets: Dict[str, List[str]],
    risk_patterns: List[Tuple[str, re.Pattern]],
    aliases: Dict[str, List[str]],
    dry_run: bool = False,
) -> pd.DataFrame:
    """전체 대상 기업에 대해 Google News RSS 수집."""
    all_rows = []
    seen_urls = set()

    for company_id, queries in targets.items():
        company_aliases = aliases.get(company_id, [company_id])
        alias_display = company_aliases[0] if company_aliases else company_id

        log.info(f"[{company_id}] {alias_display} — {len(queries)} queries")

        company_count = 0
        for query in queries:
            log.info(f"  Query: {query}")
            try:
                articles = fetch_rss(query)
            except Exception as e:
                log.warning(f"  RSS fetch failed: {e}")
                continue

            for article in articles:
                if article["url"] in seen_urls:
                    continue
                seen_urls.add(article["url"])

                row = build_risk_event(article, company_id, alias_display, risk_patterns)
                if row:
                    all_rows.append(row)
                    company_count += 1

            time.sleep(REQUEST_DELAY)

        log.info(f"  → {company_count} articles collected for {alias_display}")

    if not all_rows:
        log.warning("No articles collected!")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # 시간순 정렬 + 중복 제거
    df = df.sort_values("event_time").drop_duplicates(subset=["event_id"], keep="first")
    df = df.reset_index(drop=True)

    log.info(f"Total: {len(df)} unique articles from {len(targets)} companies")

    # risk_type 분포
    rt_counts = df["risk_types"].value_counts()
    for rt, cnt in rt_counts.head(10).items():
        log.info(f"  {rt}: {cnt}")

    return df


def merge_with_existing(
    supplement_df: pd.DataFrame,
    existing_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """기존 risk_events에 병합."""
    if existing_path.exists():
        existing = pd.read_parquet(existing_path)
        log.info(f"Existing: {len(existing):,} rows")
        merged = pd.concat([existing, supplement_df], ignore_index=True)
        before = len(merged)
        merged = merged.drop_duplicates(subset=["url"], keep="first")
        log.info(f"Merged: {before:,} → {len(merged):,} (dedup {before - len(merged)})")
    else:
        merged = supplement_df
        log.info(f"No existing file, saving {len(merged):,} rows as new")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)
    log.info(f"Saved: {output_path}")
    return merged


# ─── CLI ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Google News RSS → risk_events")
    parser.add_argument("--dry-run", action="store_true", help="수집만, 병합 안 함")
    parser.add_argument("--company", type=str, help="특정 기업만 수집 (e.g., 600110.SH)")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    args = parser.parse_args()

    # Load configs
    risk_patterns = _load_risk_patterns(RISK_KEYWORDS_YAML)
    log.info(f"Risk patterns: {len(risk_patterns)}")

    with open(ALIASES_YAML, "r", encoding="utf-8") as f:
        aliases = yaml.safe_load(f)
    log.info(f"Company aliases: {len(aliases)} companies")

    # Filter targets
    targets = TARGET_QUERIES.copy()
    if args.company:
        if args.company in targets:
            targets = {args.company: targets[args.company]}
        else:
            log.error(f"Unknown company: {args.company}")
            log.info(f"Available: {list(targets.keys())}")
            return

    # Collect
    df = collect_all(targets, risk_patterns, aliases, dry_run=args.dry_run)

    if df.empty:
        log.warning("No data collected!")
        return

    if args.dry_run:
        log.info("[DRY-RUN] Collected data preview:")
        print(df[["event_time", "title", "risk_types", "severity"]].head(20).to_string())
        return

    # Save / merge
    output_path = Path(args.output)
    merge_with_existing(df, output_path, output_path)

    log.info("Done!")


if __name__ == "__main__":
    main()
