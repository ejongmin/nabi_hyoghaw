#!/usr/bin/env python
"""
산업 전문 뉴스 RSS 수집기
CnEVPost, Yicai Global, Battery-News.de, Nikkei Asia, SMM, Electrive

사용법:
  python scripts/collect_industry_rss.py
  python scripts/collect_industry_rss.py --source cnevpost
  python scripts/collect_industry_rss.py --source all
"""
import argparse
import hashlib
import logging
import re
import time
from pathlib import Path
from typing import List

import feedparser
import pandas as pd
import yaml

log = logging.getLogger("industry_rss")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

# ── RSS 피드 목록 ──
RSS_FEEDS = {
    "cnevpost": {
        "name": "CnEVPost",
        "url": "https://cnevpost.com/feed/",
        "language": "en",
        "region": "CN",
        "description": "중국 EV 산업 영문 전문",
    },
    "electrive": {
        "name": "Electrive",
        "url": "https://www.electrive.com/feed/",
        "language": "en",
        "region": "GLOBAL",
        "description": "글로벌 전기차 산업 뉴스",
    },
    "battery_news_de": {
        "name": "Battery-News.de",
        "url": "https://battery-news.de/feed/",
        "language": "en",
        "region": "EU",
        "description": "유럽 배터리 산업",
    },
    "nikkei_asia": {
        "name": "Nikkei Asia",
        "url": "https://asia.nikkei.com/rss",
        "language": "en",
        "region": "JP",
        "description": "아시아 경제 영문 뉴스",
    },
    "yicai_global": {
        "name": "Yicai Global",
        "url": "https://www.yicaiglobal.com/rss",
        "language": "en",
        "region": "CN",
        "description": "제일재경 영문판",
    },
}

# 기업별 검색 키워드 (영문 — RSS 피드가 영문이므로)
COMPANY_KEYWORDS = {
    "300750.SZ": ["CATL", "Contemporary Amperex", "Ningde"],
    "373220.KS": ["LG Energy", "LGES"],
    "1211.HK": ["BYD"],
    "051910.KS": ["LG Chem"],
    "300014.SZ": ["EVE Energy"],
    "300207.SZ": ["Sunwoda"],
    "3931.HK": ["CALB", "China Aviation Lithium"],
    "002074.SZ": ["Gotion", "Guoxuan"],
    "247540.KQ": ["EcoPro"],
    "066970.KS": ["L&F", "LnF"],
    "003670.KS": ["POSCO Future", "POSCO Chemical"],
    "300919.SZ": ["CNGR"],
    "603659.SH": ["Putailai"],
    "300037.SZ": ["Capchem"],
    "600110.SH": ["Nuode"],
    "688388.SH": ["Jiayuan"],
    "002340.SZ": ["GEM Co"],
    "920185.BJ": ["BTR"],
    "3407.T": ["Asahi Kasei"],
    "002460.SZ": ["Ganfeng"],
    "002466.SZ": ["Tianqi"],
    "603799.SH": ["Huayou"],
    "603993.SH": ["CMOC", "China Moly"],
    "600362.SH": ["Jiangxi Copper"],
    "ALB": ["Albemarle"],
    "GLEN.L": ["Glencore"],
    "BMW.DE": ["BMW"],
    "VOW3.DE": ["Volkswagen", "VW"],
    "MBG.DE": ["Mercedes-Benz", "Mercedes"],
    "TSLA": ["Tesla"],
    "F": ["Ford Motor"],
    "GM": ["General Motors"],
    "LCID": ["Lucid"],
    "RIVN": ["Rivian"],
    "7267.T": ["Honda"],
    "7203.T": ["Toyota"],
    "0175.HK": ["Geely"],
    "LICY": ["Li-Cycle"],
    "BAS.DE": ["BASF"],
}

RISK_KEYWORDS_EN = {
    "geopolitics": ["sanction", "tariff", "trade war", "embargo", "export ban", "geopolit"],
    "logistics": ["shipping delay", "port", "logistics", "supply chain disrupt", "shortage"],
    "natural": ["earthquake", "flood", "fire", "wildfire", "typhoon", "hurricane"],
    "regulatory": ["regulat", "recall", "compliance", "penalty", "fine", "ban"],
    "market": ["price crash", "demand drop", "oversupply", "inventory", "loss", "downturn"],
    "labor": ["strike", "layoff", "workforce", "labor dispute"],
}


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _match_companies(title: str) -> List[str]:
    """제목에서 기업 매칭."""
    matched = []
    title_upper = title.upper()
    for cid, keywords in COMPANY_KEYWORDS.items():
        for kw in keywords:
            if kw.upper() in title_upper:
                matched.append(cid)
                break
    return matched


def _classify_risk_en(title: str) -> str:
    matched = []
    title_lower = title.lower()
    for rtype, keywords in RISK_KEYWORDS_EN.items():
        for kw in keywords:
            if kw in title_lower:
                matched.append(rtype)
                break
    return ",".join(sorted(set(matched))) if matched else "other"


def _severity_en(title: str) -> float:
    high = ["explosion", "fire", "sanction", "recall", "strike", "crash", "halt", "crisis"]
    mid = ["delay", "shortage", "decline", "risk", "concern", "warning"]
    title_lower = title.lower()
    score = 2.5
    for kw in high:
        if kw in title_lower: score += 0.5
    for kw in mid:
        if kw in title_lower: score += 0.3
    return min(score, 5.0)


def collect_feed(feed_key: str, feed_info: dict) -> List[dict]:
    """단일 RSS 피드 수집."""
    log.info(f"[{feed_info['name']}] Fetching {feed_info['url']}")

    try:
        feed = feedparser.parse(feed_info["url"])
    except Exception as e:
        log.error(f"  Feed error: {e}")
        return []

    rows = []
    for entry in feed.entries:
        title = entry.get("title", "")
        link = entry.get("link", "")
        pub = entry.get("published", entry.get("updated", ""))

        # 기업 매칭
        companies = _match_companies(title)
        if not companies:
            # 제목에서 매칭 안 되면 summary에서도 시도
            summary = entry.get("summary", "")
            companies = _match_companies(summary)

        # 기업 매칭 안 되면 스킵 (우리 universe 관련 뉴스만)
        if not companies:
            continue

        rows.append({
            "event_id": _md5(f"{link}|{feed_key}"),
            "event_time": pub,
            "url": link,
            "title": title,
            "risk_types": _classify_risk_en(title),
            "severity": _severity_en(title),
            "entity_ids": companies,
            "entity_scores": [1.0] * len(companies),
            "finbert": 0.0,
            "country_ids": [feed_info["region"]],
            "collection_type": f"rss_{feed_key}",
            "source_domain": feed_info["name"],
        })

    log.info(f"  → {len(rows)} relevant articles (from {len(feed.entries)} total)")
    return rows


# ── Google News 기업별 검색 (RSS, 과거 데이터 포함) ──

def collect_google_news_global() -> List[dict]:
    """Google News RSS로 기업별 영문 뉴스 수집."""
    all_rows = []

    # 뉴스가 부족한 기업 우선
    priority_companies = {
        "600110.SH": "Nuode copper foil battery",
        "300919.SZ": "CNGR precursor cathode battery",
        "688388.SH": "Jiayuan copper foil EV battery",
        "603659.SH": "Putailai anode material battery",
        "920185.BJ": "BTR anode material lithium",
        "300037.SZ": "Capchem electrolyte battery",
        "002340.SZ": "GEM Co recycling battery precursor",
        "3931.HK": "CALB battery China",
        "LICY": "Li-Cycle battery recycling",
        "3407.T": "Asahi Kasei separator battery",
    }

    for cid, query in priority_companies.items():
        url = f"https://news.google.com/rss/search?q={query}&hl=en&gl=US&ceid=US:en"
        log.info(f"[Google News] {cid}: {query}")

        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:50]:
                title = entry.get("title", "")
                link = entry.get("link", "")
                pub = entry.get("published", "")

                all_rows.append({
                    "event_id": _md5(f"{link}|google_global"),
                    "event_time": pub,
                    "url": link,
                    "title": title,
                    "risk_types": _classify_risk_en(title),
                    "severity": _severity_en(title),
                    "entity_ids": [cid],
                    "entity_scores": [1.0],
                    "finbert": 0.0,
                    "country_ids": ["GLOBAL"],
                    "collection_type": "google_news_global",
                    "source_domain": "news.google.com",
                })
        except Exception as e:
            log.error(f"  Error: {e}")

        time.sleep(3)

    return all_rows


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="산업 전문 뉴스 RSS 수집기")
    parser.add_argument("--source", choices=list(RSS_FEEDS.keys()) + ["google", "all"], default="all")
    parser.add_argument("--output", default="data/raw/industry_rss/industry_news.parquet")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []

    if args.source == "all":
        # 모든 RSS 피드
        for key, info in RSS_FEEDS.items():
            all_rows.extend(collect_feed(key, info))
            time.sleep(2)
        # + Google News 글로벌
        all_rows.extend(collect_google_news_global())
    elif args.source == "google":
        all_rows.extend(collect_google_news_global())
    else:
        info = RSS_FEEDS[args.source]
        all_rows.extend(collect_feed(args.source, info))

    if not all_rows:
        log.warning("No data collected")
        return

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["url"], keep="first")
    df.to_parquet(output_path, index=False)
    log.info(f"Saved {len(df):,} articles to {output_path}")

    # Per-source stats
    if "collection_type" in df.columns:
        log.info("Per-source breakdown:")
        for ct, group in df.groupby("collection_type"):
            log.info(f"  {ct}: {len(group):,}")


if __name__ == "__main__":
    main()
