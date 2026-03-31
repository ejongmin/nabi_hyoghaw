#!/usr/bin/env python3
"""Google News 반기별 검색으로 중국+한국 뉴스 2배 확장 (전략 3+4)"""
import hashlib, logging, time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus, urlparse
from xml.etree import ElementTree
import pandas as pd, requests

log = logging.getLogger("gn_half")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0"}

# 중국 기업 (하위 뉴스량 위주)
CN_TARGETS = {
    "603659.SH": {"cn":"璞泰来","en":"Putailai"},
    "600110.SH": {"cn":"诺德股份","en":"Nuode"},
    "688388.SH": {"cn":"嘉元科技","en":"Jiayuan Technology"},
    "300919.SZ": {"cn":"中伟股份","en":"CNGR"},
    "300207.SZ": {"cn":"欣旺达","en":"Sunwoda"},
    "603799.SH": {"cn":"华友钴业","en":"Huayou Cobalt"},
    "300037.SZ": {"cn":"新宙邦","en":"Capchem"},
    "300014.SZ": {"cn":"亿纬锂能","en":"EVE Energy"},
    "002074.SZ": {"cn":"国轩高科","en":"Gotion"},
    "3931.HK":   {"cn":"中创新航","en":"CALB"},
    "300750.SZ": {"cn":"宁德时代","en":"CATL"},
    "002460.SZ": {"cn":"赣锋锂业","en":"Ganfeng Lithium"},
    "002466.SZ": {"cn":"天齐锂业","en":"Tianqi Lithium"},
    "603993.SH": {"cn":"洛阳钼业","en":"CMOC"},
    "601633.SH": {"cn":"长城汽车","en":"Great Wall Motor"},
    "600733.SH": {"cn":"北汽蓝谷","en":"BAIC"},
    "600362.SH": {"cn":"江西铜业","en":"Jiangxi Copper"},
    "920185.BJ": {"cn":"贝特瑞","en":"BTR"},
    "002340.SZ": {"cn":"格林美","en":"GEM"},
}

# 한국 기업
KR_TARGETS = {
    "247540.KQ": {"kr":"에코프로비엠","en":"EcoPro BM"},
    "066970.KS": {"kr":"엘앤에프","en":"L&F"},
    "003670.KS": {"kr":"포스코퓨처엠","en":"POSCO Future M"},
    "373220.KS": {"kr":"LG에너지솔루션","en":"LG Energy Solution"},
    "051910.KS": {"kr":"LG화학","en":"LG Chem"},
}

def fetch_rss(query, after, before, hl="zh-CN", gl="CN", ceid="CN:zh-Hans"):
    encoded = quote_plus(f"{query} after:{after} before:{before}")
    url = f"https://news.google.com/rss/search?q={encoded}&hl={hl}&gl={gl}&ceid={ceid}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200: return []
        root = ElementTree.fromstring(r.content)
        items = []
        for item in root.findall(".//item"):
            t = item.find("title"); l = item.find("link"); p = item.find("pubDate")
            title = t.text.strip() if t is not None and t.text else ""
            link = l.text.strip() if l is not None and l.text else ""
            pub = p.text.strip() if p is not None and p.text else ""
            if not link or not title: continue
            event_time = None
            for fmt in ["%a, %d %b %Y %H:%M:%S %Z", "%a, %d %b %Y %H:%M:%S %z"]:
                try: event_time = datetime.strptime(pub, fmt); break
                except: continue
            items.append({"title":title,"url":link,"event_time":event_time,"source_domain":urlparse(link).netloc.replace("www.","")})
        return items
    except: return []

def collect_half_yearly(cid, queries, hl, gl, ceid):
    """반기별 × 쿼리 수집"""
    all_items = []
    halves = []
    for year in range(2019, 2026):
        halves.append((f"{year}-01-01", f"{year}-06-30"))
        halves.append((f"{year}-07-01", f"{year}-12-31"))
    
    for query in queries:
        for after, before in halves:
            items = fetch_rss(query, after, before, hl, gl, ceid)
            for it in items:
                it["company_id"] = cid
            all_items.extend(items)
            time.sleep(0.7)
    return all_items

all_results = []

# Chinese companies
log.info(f"=== 중국 기업 ({len(CN_TARGETS)}) ===")
for i, (cid, info) in enumerate(CN_TARGETS.items(), 1):
    queries = [info["cn"], info["cn"]+" 电池", info["cn"]+" 供应链 风险", info["en"]]
    log.info(f"[{i}/{len(CN_TARGETS)}] {cid} {info['cn']} ({info['en']})")
    items = collect_half_yearly(cid, queries, "zh-CN", "CN", "CN:zh-Hans")
    log.info(f"  → {len(items)} articles")
    all_results.extend(items)

# Korean companies
log.info(f"\n=== 한국 기업 ({len(KR_TARGETS)}) ===")
for i, (cid, info) in enumerate(KR_TARGETS.items(), 1):
    queries_kr = [info["kr"], info["kr"]+" 배터리", info["kr"]+" 공급망"]
    queries_en = [info["en"]]
    log.info(f"[{i}/{len(KR_TARGETS)}] {cid} {info['kr']} ({info['en']})")
    # Korean RSS
    items_kr = collect_half_yearly(cid, queries_kr, "ko", "KR", "KR:ko")
    # English RSS for Korean companies
    items_en = collect_half_yearly(cid, queries_en, "en", "US", "US:en")
    items = items_kr + items_en
    log.info(f"  → {len(items)} articles (KR:{len(items_kr)}, EN:{len(items_en)})")
    all_results.extend(items)

if not all_results:
    log.warning("No results!"); exit()

# Build DataFrame
rows = []
for it in all_results:
    country = "Korea" if it["company_id"] in KR_TARGETS else "China"
    rows.append({
        "event_id": hashlib.md5(f"{it['url']}|{it['title']}".encode()).hexdigest(),
        "event_time": it["event_time"],
        "url": it["url"], "title": it["title"],
        "risk_types": "other", "severity": 2.0,
        "entity_ids": [it["company_id"]], "entity_scores": [1.0],
        "finbert": 0.0, "country_ids": [country],
        "collection_type": f"google_news_half_{'kr' if country=='Korea' else 'cn'}",
        "source_domain": it["source_domain"],
    })

df = pd.DataFrame(rows)
n_before = len(df)
df = df.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
log.info(f"\nTotal: {n_before:,} → Dedup: {len(df):,}")

# Stats
exp = df.explode("entity_ids")
log.info("\n=== Per-Company ===")
for cid, cnt in exp.groupby("entity_ids").size().sort_values(ascending=False).items():
    name = CN_TARGETS.get(cid,{}).get("cn","") or KR_TARGETS.get(cid,{}).get("kr","")
    log.info(f"  {str(cid):<14} {name:<12}: {cnt:>6}")

df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
df["event_time"] = df["event_time"].dt.tz_localize(None)

out = Path("data/raw/chinese_news/google_news_half_boost.parquet")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(out, index=False)
log.info(f"\n✅ Saved: {out} ({len(df):,} rows)")
