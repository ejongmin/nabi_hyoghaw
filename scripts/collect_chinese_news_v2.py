#!/usr/bin/env python3
"""
중국 뉴스 대량 수집기 v2 — Google News CN 연도별 검색
=====================================================
Google News CN RSS는 검색당 최대 100건 반환.
25개 중국 기업 × 7년(2019~2025) × 다중 쿼리로 대량 수집.

전략:
  - 쿼리 1: 중문명 (예: "宁德时代")
  - 쿼리 2: 중문명 + "电池" (배터리 관련 뉴스 집중)
  - 쿼리 3: 중문명 + "供应链" (공급망 리스크 뉴스)
  - 쿼리 4: 영문명 (예: "CATL")

사용법:
  python scripts/collect_chinese_news_v2.py
  python scripts/collect_chinese_news_v2.py --dry-run
  python scripts/collect_chinese_news_v2.py --company CATL

출력: data/raw/chinese_news/chinese_news_v2.parquet
"""
import argparse
import hashlib
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from urllib.parse import quote_plus, urlparse
from xml.etree import ElementTree

import pandas as pd
import requests

log = logging.getLogger("cn_news_v2")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

# ── 중국 기업 ──
CHINESE_COMPANIES = {
    # Cell/Pack
    "300750.SZ": {"cn": "宁德时代", "en": "CATL"},
    "3931.HK":   {"cn": "中创新航", "en": "CALB"},
    "300014.SZ": {"cn": "亿纬锂能", "en": "EVE Energy"},
    "002074.SZ": {"cn": "国轩高科", "en": "Gotion"},
    "300207.SZ": {"cn": "欣旺达", "en": "Sunwoda"},
    "300037.SZ": {"cn": "新宙邦", "en": "Capchem"},
    # Materials
    "920185.BJ": {"cn": "贝特瑞", "en": "BTR"},
    "300919.SZ": {"cn": "中伟股份", "en": "CNGR"},
    "002340.SZ": {"cn": "格林美", "en": "GEM"},
    "603799.SH": {"cn": "华友钴业", "en": "Huayou Cobalt"},
    "688388.SH": {"cn": "嘉元科技", "en": "Jiayuan Technology"},
    "600110.SH": {"cn": "诺德股份", "en": "Nuode"},
    "603659.SH": {"cn": "璞泰来", "en": "Putailai"},
    # Upstream/Refining
    "002460.SZ": {"cn": "赣锋锂业", "en": "Ganfeng Lithium"},
    "002466.SZ": {"cn": "天齐锂业", "en": "Tianqi Lithium"},
    "603993.SH": {"cn": "洛阳钼业", "en": "CMOC"},
    "600362.SH": {"cn": "江西铜业", "en": "Jiangxi Copper"},
    # OEM
    "1211.HK":   {"cn": "比亚迪", "en": "BYD"},
    "0175.HK":   {"cn": "吉利汽车", "en": "Geely"},
    "000625.SZ": {"cn": "长安汽车", "en": "Changan"},
    "600006.SH": {"cn": "东风汽车", "en": "Dongfeng"},
    "601238.SH": {"cn": "广汽集团", "en": "GAC"},
    "601633.SH": {"cn": "长城汽车", "en": "Great Wall Motor"},
    "600418.SH": {"cn": "江淮汽车", "en": "JAC"},
    "600733.SH": {"cn": "北汽蓝谷", "en": "BAIC BluePark"},
}

# 검색어 접미사: 배터리/공급망 맥락
QUERY_SUFFIXES = [
    "",          # 기업명만
    " 电池",     # + 배터리
    " 供应链",   # + 공급망
]

YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]


def make_event_id(url: str, title: str) -> str:
    return hashlib.md5(f"{url}|{title}".encode()).hexdigest()


def extract_domain(url: str) -> str:
    try:
        d = urlparse(url).netloc.replace("www.", "")
        return d if d else "unknown"
    except Exception:
        return "unknown"


def fetch_google_news_rss(query: str, year: int,
                           hl: str = "zh-CN", gl: str = "CN",
                           ceid: str = "CN:zh-Hans") -> List[dict]:
    """Google News RSS 검색 — 1년 단위 날짜 필터."""
    after = f"{year}-01-01"
    before = f"{year}-12-31"
    full_query = f"{query} after:{after} before:{before}"
    encoded = quote_plus(full_query)
    url = f"https://news.google.com/rss/search?q={encoded}&hl={hl}&gl={gl}&ceid={ceid}"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return []
        root = ElementTree.fromstring(resp.content)
        items = []
        for item in root.findall(".//item"):
            title_el = item.find("title")
            link_el = item.find("link")
            pubdate_el = item.find("pubDate")

            title = title_el.text.strip() if title_el is not None and title_el.text else ""
            link = link_el.text.strip() if link_el is not None and link_el.text else ""
            pubdate = pubdate_el.text.strip() if pubdate_el is not None and pubdate_el.text else ""

            if not link or not title:
                continue

            event_time = None
            for fmt in ["%a, %d %b %Y %H:%M:%S %Z", "%a, %d %b %Y %H:%M:%S %z"]:
                try:
                    event_time = datetime.strptime(pubdate, fmt)
                    break
                except ValueError:
                    continue
            if event_time is None:
                event_time = datetime(year, 6, 15)

            items.append({
                "title": title,
                "url": link,
                "event_time": event_time,
                "source_domain": extract_domain(link),
            })
        return items
    except Exception:
        return []


def collect_company(company_id: str, cn_name: str, en_name: str) -> List[dict]:
    """한 기업의 전체 연도 × 전체 쿼리 수집."""
    all_items = []

    for year in YEARS:
        # 중문 쿼리들
        for suffix in QUERY_SUFFIXES:
            query = cn_name + suffix
            items = fetch_google_news_rss(query, year)
            for it in items:
                it["company_id"] = company_id
                it["query"] = query
                it["year"] = year
            all_items.extend(items)
            time.sleep(0.8)

        # 영문 쿼리 (중복 최소화를 위해 접미사 없이)
        en_items = fetch_google_news_rss(en_name, year)
        for it in en_items:
            it["company_id"] = company_id
            it["query"] = en_name
            it["year"] = year
        all_items.extend(en_items)
        time.sleep(0.8)

    return all_items


def main():
    parser = argparse.ArgumentParser(description="중국 뉴스 대량 수집기 v2")
    parser.add_argument("--company", type=str, help="특정 기업만 (e.g. CATL, BYD)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default="data/raw/chinese_news/chinese_news_v2.parquet")
    args = parser.parse_args()

    targets = CHINESE_COMPANIES
    if args.company:
        kw = args.company.lower()
        targets = {
            k: v for k, v in CHINESE_COMPANIES.items()
            if kw in v["en"].lower() or kw in v["cn"]
        }
        if not targets:
            log.error(f"Company '{args.company}' not found")
            return

    n_queries = len(targets) * len(YEARS) * (len(QUERY_SUFFIXES) + 1)  # +1 for English
    log.info("=" * 60)
    log.info("중국 뉴스 대량 수집기 v2 (Google News CN)")
    log.info(f"  대상: {len(targets)} companies × {len(YEARS)} years × {len(QUERY_SUFFIXES)+1} queries")
    log.info(f"  총 요청: ~{n_queries} RSS calls (max ~{n_queries*100:,} articles)")
    log.info(f"  예상 시간: ~{n_queries * 1.5 / 60:.0f} min")
    log.info("=" * 60)

    if args.dry_run:
        for cid, info in targets.items():
            log.info(f"  {cid:<14} {info['cn']:<10} ({info['en']})")
        return

    all_results = []
    for i, (cid, info) in enumerate(targets.items(), 1):
        cn_name = info["cn"]
        en_name = info["en"]
        log.info(f"\n[{i}/{len(targets)}] {cid} | {cn_name} ({en_name})")

        items = collect_company(cid, cn_name, en_name)
        log.info(f"  → {len(items)} articles (before dedup)")
        all_results.extend(items)

    if not all_results:
        log.warning("No articles collected!")
        return

    # Build DataFrame
    rows = []
    for it in all_results:
        rows.append({
            "event_id": make_event_id(it["url"], it["title"]),
            "event_time": it["event_time"],
            "url": it["url"],
            "title": it["title"],
            "risk_types": "other",
            "severity": 2.0,
            "entity_ids": [it["company_id"]],
            "entity_scores": [1.0],
            "finbert": 0.0,
            "country_ids": ["China"],
            "collection_type": "google_news_cn_v2",
            "source_domain": it["source_domain"],
        })

    df = pd.DataFrame(rows)

    # Dedup by URL
    n_before = len(df)
    df = df.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
    n_dedup = n_before - len(df)

    log.info(f"\n{'='*60}")
    log.info(f"Total collected: {n_before:,}")
    log.info(f"Deduped: {n_dedup:,}")
    log.info(f"Final: {len(df):,}")

    # Per-company breakdown
    log.info("\n=== Per-Company Breakdown ===")
    entity_counts = df.explode("entity_ids").groupby("entity_ids").size().sort_values(ascending=False)
    for cid, cnt in entity_counts.items():
        cn_name = CHINESE_COMPANIES.get(cid, {}).get("cn", "?")
        log.info(f"  {str(cid):<14} {cn_name:<12}: {cnt:>6}")

    # Year breakdown
    log.info("\n=== Per-Year Breakdown ===")
    df["_year"] = pd.to_datetime(df["event_time"], errors="coerce").dt.year
    for yr, g in df.groupby("_year"):
        log.info(f"  {yr}: {len(g):>6}")
    df.drop(columns=["_year"], inplace=True)

    # Top domains
    log.info("\n=== Top 15 Domains ===")
    for d, c in df["source_domain"].value_counts().head(15).items():
        log.info(f"  {d:<35}: {c:>5}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df["event_time"] = df["event_time"].dt.tz_localize(None)
    df.to_parquet(output_path, index=False)
    log.info(f"\n✅ Saved: {output_path} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
