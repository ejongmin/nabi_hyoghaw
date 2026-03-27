#!/usr/bin/env python
"""
BigKinds (빅카인즈) 한국 뉴스 수집기
https://www.bigkinds.or.kr/

사용법:
  1. https://www.bigkinds.or.kr/ 회원가입 (무료)
  2. BigKinds Open API 키 발급 (연구/학술용)
  3. 실행:
     python scripts/collect_bigkinds.py --api-key YOUR_KEY
     python scripts/collect_bigkinds.py --api-key YOUR_KEY --start 2018-01-01 --end 2025-12-31

참고: BigKinds API가 없으면 웹 검색 모드(--web-search)로 대체 가능
"""
import argparse
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
import yaml

log = logging.getLogger("bigkinds")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

# ── 한국 기업 검색어 (company_aliases.yaml에서 한국어 alias 추출) ──
KOREAN_QUERIES = {
    "373220.KS": ["LG에너지솔루션", "LG엔솔", "LG Energy Solution"],
    "051910.KS": ["LG화학", "LG Chem"],
    "247540.KQ": ["에코프로비엠", "에코프로", "EcoPro"],
    "066970.KS": ["엘앤에프", "L&F"],
    "003670.KS": ["포스코퓨처엠", "포스코케미칼", "POSCO Future M"],
}

# 리스크 키워드 (한국어)
RISK_KEYWORDS_KR = {
    "geopolitics": ["수출규제", "제재", "관세", "무역전쟁", "미중갈등", "수출통제", "경제안보"],
    "logistics": ["물류대란", "항만폐쇄", "선박지연", "컨테이너부족", "공급차질", "운송중단"],
    "natural": ["지진", "홍수", "태풍", "산불", "화재", "폭발", "자연재해"],
    "regulatory": ["규제강화", "환경규제", "배출규제", "인증", "리콜", "안전기준"],
    "market": ["가격폭락", "가격급등", "수요감소", "과잉공급", "재고급증", "실적부진"],
    "labor": ["파업", "노조", "인력부족", "구조조정", "정리해고"],
}


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _classify_risk(title: str) -> str:
    """한국어 제목에서 리스크 유형 분류."""
    matched = []
    for rtype, keywords in RISK_KEYWORDS_KR.items():
        for kw in keywords:
            if kw in title:
                matched.append(rtype)
                break
    return ",".join(sorted(set(matched))) if matched else "other"


def _severity_from_title(title: str) -> float:
    """키워드 밀도 기반 severity 추정 (V2Tone 없으므로)."""
    high_severity = ["폭발", "화재", "제재", "리콜", "파업", "폭락", "중단", "위기", "붕괴"]
    mid_severity = ["규제", "지연", "부족", "감소", "하락", "차질", "우려"]
    score = 2.5  # baseline
    for kw in high_severity:
        if kw in title:
            score += 0.5
    for kw in mid_severity:
        if kw in title:
            score += 0.3
    return min(score, 5.0)


# ── BigKinds API 수집 ──

def collect_bigkinds_api(api_key: str, start_date: str, end_date: str,
                         output_dir: Path) -> pd.DataFrame:
    """BigKinds Open API를 통한 뉴스 수집."""
    base_url = "https://tools.kinds.or.kr/search/news"
    all_rows = []

    for cid, queries in KOREAN_QUERIES.items():
        query_str = " OR ".join(f'"{q}"' for q in queries)
        log.info(f"[BigKinds API] {cid}: {query_str}")

        page = 0
        while True:
            params = {
                "access_key": api_key,
                "argument": json.dumps({
                    "query": query_str,
                    "published_at": {
                        "from": start_date.replace("-", ""),
                        "until": end_date.replace("-", ""),
                    },
                    "provider": [],  # 모든 언론사
                    "category": ["경제", "산업"],
                    "sort": {"date": "desc"},
                    "return_from": page * 100,
                    "return_size": 100,
                    "fields": ["title", "published_at", "provider_link_page", "provider_name"],
                }),
            }

            try:
                r = requests.post(base_url, json=params, timeout=30)
                data = r.json()
            except Exception as e:
                log.error(f"  API error: {e}")
                break

            documents = data.get("return_object", {}).get("documents", [])
            if not documents:
                break

            for doc in documents:
                title = doc.get("title", "")
                url = doc.get("provider_link_page", "")
                pub_date = doc.get("published_at", "")

                all_rows.append({
                    "event_id": _md5(f"{url}|bigkinds"),
                    "event_time": pub_date,
                    "url": url,
                    "title": title,
                    "risk_types": _classify_risk(title),
                    "severity": _severity_from_title(title),
                    "entity_ids": [cid],
                    "entity_scores": [1.0],
                    "finbert": 0.0,
                    "country_ids": ["KR"],
                    "collection_type": "bigkinds_kr",
                    "source_domain": doc.get("provider_name", "bigkinds"),
                })

            page += 1
            if page * 100 >= data.get("return_object", {}).get("total_hits", 0):
                break
            time.sleep(1)

        log.info(f"  {cid}: {sum(1 for r in all_rows if cid in str(r.get('entity_ids')))} articles")

    if not all_rows:
        log.warning("No articles collected from BigKinds API")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    return df


# ── 네이버 뉴스 RSS 대체 수집 (API 키 없을 때) ──

def collect_naver_rss(start_date: str, end_date: str) -> pd.DataFrame:
    """네이버 뉴스 RSS를 통한 대체 수집 (API 키 불필요)."""
    import feedparser

    all_rows = []
    for cid, queries in KOREAN_QUERIES.items():
        for query in queries:
            url = f"https://news.google.com/rss/search?q={query}+when:7d&hl=ko&gl=KR&ceid=KR:ko"
            log.info(f"[Google News KR] {cid}: {query}")

            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:50]:
                    title = entry.get("title", "")
                    link = entry.get("link", "")
                    pub = entry.get("published", "")

                    all_rows.append({
                        "event_id": _md5(f"{link}|google_kr"),
                        "event_time": pub,
                        "url": link,
                        "title": title,
                        "risk_types": _classify_risk(title),
                        "severity": _severity_from_title(title),
                        "entity_ids": [cid],
                        "entity_scores": [1.0],
                        "finbert": 0.0,
                        "country_ids": ["KR"],
                        "collection_type": "google_news_kr",
                        "source_domain": "news.google.com",
                    })
            except Exception as e:
                log.error(f"  RSS error: {e}")

            time.sleep(2)

    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="BigKinds 한국 뉴스 수집기")
    parser.add_argument("--api-key", help="BigKinds API key")
    parser.add_argument("--start", default="2018-01-01", help="시작일 (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-12-31", help="종료일 (YYYY-MM-DD)")
    parser.add_argument("--fallback-rss", action="store_true", help="API 없을 시 Google News KR RSS 사용")
    parser.add_argument("--output", default="data/raw/bigkinds/bigkinds_news.parquet")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.api_key:
        df = collect_bigkinds_api(args.api_key, args.start, args.end, output_path.parent)
    else:
        log.info("No API key provided, using Google News KR RSS fallback")
        df = collect_naver_rss(args.start, args.end)

    if df.empty:
        log.warning("No data collected")
        return

    df = df.drop_duplicates(subset=["url"], keep="first")
    df.to_parquet(output_path, index=False)
    log.info(f"Saved {len(df):,} articles to {output_path}")

    # Stats
    for cid in KOREAN_QUERIES:
        n = df[df["entity_ids"].apply(lambda x: cid in x if isinstance(x, list) else False)].shape[0]
        log.info(f"  {cid}: {n:,} articles")


if __name__ == "__main__":
    main()
