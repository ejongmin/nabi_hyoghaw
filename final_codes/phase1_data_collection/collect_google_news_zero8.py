#!/usr/bin/env python3
"""
collect_google_news_zero8.py
────────────────────────────
scored=0 8개 기업 (Honda/Ford/Glencore/Albemarle/Lucid/BASF/Asahi Kasei/SQM)
title-bearing 데이터를 Google News RSS로 복구.

기존 boost_google_news_half.py 구조를 그대로 재사용 (반기별 × 2018-2025).
Cowork VM에서는 news.google.com egress가 막혀 있어 Colab/로컬 실행 필수.

Usage (local or Colab):
    pip install pandas requests pyarrow
    python scripts/collect_google_news_zero8.py

Output:
    data/raw/supplement/google_news_zero8.parquet
"""
import hashlib
import logging
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus, urlparse
from xml.etree import ElementTree

import pandas as pd
import requests

log = logging.getLogger("gn_zero8")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# ─── 8 scored=0 target companies × 다국어 검색 쿼리 ────────────
# 각 기업의 domain 언어권에서 중요한 키워드 조합
# - en: 글로벌 영어 뉴스
# - 현지어(jp/de/es): 본국 언론 커버리지 추가
TARGETS = {
    # 일본 OEM — 영어 + 일본어
    "7267.T": {
        "name": "Honda Motor",
        "queries": {
            "en": ['"Honda Motor" battery EV', '"Honda" electric vehicle', '"Honda" EV supply chain'],
            "jp": ['ホンダ 電池', 'ホンダ EV', '本田技研 電動化'],
        },
        "locales": [("en", "US", "US:en"), ("ja", "JP", "JP:ja")],
    },
    # 미국 OEM — 영어만
    "F": {
        "name": "Ford Motor",
        "queries": {
            "en": ['"Ford Motor" battery', '"Ford" EV F-150 Lightning', '"Ford" battery plant', '"Ford" Mustang Mach-E'],
        },
        "locales": [("en", "US", "US:en")],
    },
    # 영국 원자재 — 영어
    "GLEN.L": {
        "name": "Glencore",
        "queries": {
            "en": ['"Glencore" cobalt', '"Glencore" lithium', '"Glencore" battery material', '"Glencore" nickel mining'],
        },
        "locales": [("en", "US", "US:en"), ("en", "GB", "GB:en")],
    },
    # 미국 리튬 — 영어
    "ALB": {
        "name": "Albemarle",
        "queries": {
            "en": ['"Albemarle" lithium', '"Albemarle Corp" battery', '"Albemarle" hydroxide', '"Albemarle" Chile Australia'],
        },
        "locales": [("en", "US", "US:en")],
    },
    # 미국 EV 신생 — 영어
    "LCID": {
        "name": "Lucid Motors",
        "queries": {
            "en": ['"Lucid Motors" battery', '"Lucid Group" EV', '"Lucid Air"', '"Lucid Motors" supply chain'],
        },
        "locales": [("en", "US", "US:en")],
    },
    # 독일 화학 — 영어 + 독일어
    "BAS.DE": {
        "name": "BASF",
        "queries": {
            "en": ['"BASF" battery material', '"BASF" cathode', '"BASF" lithium', '"BASF SE" EV'],
            "de": ['BASF Batterie', 'BASF Kathode', 'BASF Elektromobilität'],
        },
        "locales": [("en", "US", "US:en"), ("de", "DE", "DE:de")],
    },
    # 일본 분리막 — 영어 + 일본어
    "3407.T": {
        "name": "Asahi Kasei",
        "queries": {
            "en": ['"Asahi Kasei" separator battery', '"Asahi Kasei" lithium', '"Asahi Kasei" Hipore'],
            "jp": ['旭化成 セパレータ', '旭化成 電池', '旭化成 ハイポア'],
        },
        "locales": [("en", "US", "US:en"), ("ja", "JP", "JP:ja")],
    },
    # 칠레 리튬 — 영어 + 스페인어
    "SQM": {
        "name": "SQM Lithium",
        "queries": {
            "en": ['"SQM" lithium Chile', '"Sociedad Quimica" lithium', '"SQM" brine Atacama', '"SQM" battery'],
            "es": ['SQM litio Chile', 'Sociedad Quimica Minera litio', 'SQM Atacama'],
        },
        "locales": [("en", "US", "US:en"), ("es", "CL", "CL:es-419")],
    },
}

# 반기 구간 (2018-2025, Google News는 1년 이상 전 쿼리 결과가 제한적이지만 after:/before:로 확장 가능)
HALVES = []
for year in range(2018, 2026):
    HALVES.append((f"{year}-01-01", f"{year}-06-30"))
    HALVES.append((f"{year}-07-01", f"{year}-12-31"))


def fetch_rss(query: str, after: str, before: str, hl: str, gl: str, ceid: str, max_retry: int = 3):
    """반기 × 쿼리 × locale 단위 RSS fetch."""
    q = f"{query} after:{after} before:{before}"
    url = f"https://news.google.com/rss/search?q={quote_plus(q)}&hl={hl}&gl={gl}&ceid={ceid}"
    for attempt in range(max_retry):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code != 200:
                if r.status_code in (429, 503):
                    time.sleep(5 + attempt * 5)
                    continue
                return []
            root = ElementTree.fromstring(r.content)
            items = []
            for item in root.findall(".//item"):
                t = item.find("title")
                l = item.find("link")
                p = item.find("pubDate")
                title = t.text.strip() if t is not None and t.text else ""
                link = l.text.strip() if l is not None and l.text else ""
                pub = p.text.strip() if p is not None and p.text else ""
                if not link or not title:
                    continue
                event_time = None
                for fmt in ("%a, %d %b %Y %H:%M:%S %Z", "%a, %d %b %Y %H:%M:%S %z"):
                    try:
                        event_time = datetime.strptime(pub, fmt)
                        break
                    except Exception:
                        continue
                items.append({
                    "title": title,
                    "url": link,
                    "event_time": event_time,
                    "source_domain": urlparse(link).netloc.replace("www.", ""),
                })
            return items
        except Exception as e:
            if attempt == max_retry - 1:
                log.warning(f"fetch_rss failed (final): {e}")
            else:
                time.sleep(2 + attempt * 2)
    return []


def collect_one_company(cid: str, info: dict) -> list:
    """한 기업에 대해 모든 locale × 쿼리 × 반기 조합 수집.

    로직: 각 locale (hl, gl, ceid)에 대해, 그 locale 언어와 일치하는 쿼리 세트를 사용.
    언어 prefix 매칭: hl='en'→queries['en'], hl='ja'→queries['jp'],
                    hl='de'→queries['de'], hl='es'→queries['es']
    """
    lang_map = {"en": "en", "ja": "jp", "de": "de", "es": "es", "ko": "kr", "zh": "cn"}
    all_items = []
    for hl, gl, ceid in info["locales"]:
        lang_key = lang_map.get(hl[:2], hl[:2])
        if lang_key not in info["queries"]:
            continue
        queries = info["queries"][lang_key]
        log.info(f"    locale={hl}-{gl} ({lang_key}): {len(queries)} queries × {len(HALVES)} halves")
        for q in queries:
            for after, before in HALVES:
                items = fetch_rss(q, after, before, hl, gl, ceid)
                for it in items:
                    it["company_id"] = cid
                    it["lang"] = lang_key
                all_items.extend(items)
                time.sleep(0.8)  # rate limit
    return all_items


def main():
    all_rows = []
    for i, (cid, info) in enumerate(TARGETS.items(), 1):
        log.info(f"[{i}/{len(TARGETS)}] {cid} ({info['name']})")
        items = collect_one_company(cid, info)
        before = len(items)
        # 기업 내 중복 제거
        seen = set()
        uniq = []
        for it in items:
            if it["url"] in seen:
                continue
            seen.add(it["url"])
            uniq.append(it)
        log.info(f"  → {before:,} raw, {len(uniq):,} unique")
        all_rows.extend(uniq)

    if not all_rows:
        log.error("No rows collected!")
        return

    # Build DataFrame matching existing schema
    rows = []
    for it in all_rows:
        rows.append({
            "event_id": hashlib.md5(f"{it['url']}|{it.get('company_id','')}".encode()).hexdigest()[:16],
            "event_time": it["event_time"],
            "url": it["url"],
            "title": it["title"],
            "risk_types": "other",      # 후속 classify 단계에서 재계산
            "severity": 2.0,             # placeholder, classify 단계에서 재계산
            "alias": TARGETS[it["company_id"]]["name"],
            "company_id": it["company_id"],
            "country_ids": "",           # 후속 pipeline에서 채움
            "tone": 0.0,                 # GDELT tone 없음
            "source_domain": it["source_domain"],
            "collection_type": "google_news_zero8",
            "lang": it.get("lang", ""),
        })

    df = pd.DataFrame(rows)
    n_before = len(df)
    df = df.drop_duplicates(subset=["url", "company_id"], keep="first").reset_index(drop=True)
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    try:
        df["event_time"] = df["event_time"].dt.tz_localize(None)
    except Exception:
        pass
    log.info(f"\nTotal: {n_before:,} → dedup: {len(df):,}")

    # Per-company stats
    log.info("\n=== Per-Company Recovery ===")
    for cid in TARGETS.keys():
        n = (df["company_id"] == cid).sum()
        log.info(f"  {cid:<10} {TARGETS[cid]['name']:<15}: {n:>5,}")

    # Save
    out = Path("data/raw/supplement/google_news_zero8.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    log.info(f"\n✅ Saved: {out} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
