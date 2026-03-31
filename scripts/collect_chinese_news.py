#!/usr/bin/env python
"""
중국 뉴스 수집기 — 新浪财经 + 东方财富 + Google News CN
EV-배터리 공급망 중국 기업 뉴스 수집

사용법:
  python scripts/collect_chinese_news.py
  python scripts/collect_chinese_news.py --start 2018-01-01 --end 2025-12-31
  python scripts/collect_chinese_news.py --source sina     # 新浪财经만
  python scripts/collect_chinese_news.py --source eastmoney # 东方财富만
  python scripts/collect_chinese_news.py --source google   # Google News CN만
"""
import argparse
import hashlib
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import urllib.parse

import pandas as pd
import requests

log = logging.getLogger("chinese_news")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

# ── 중국 기업 목록 (stock code → company_id 매핑) ──
CHINESE_COMPANIES = {
    # code: (company_id, 중국어명, [검색키워드])
    "300750": ("300750.SZ", "宁德时代", ["宁德时代", "CATL"]),
    "002460": ("002460.SZ", "赣锋锂业", ["赣锋锂业", "赣锋"]),
    "002466": ("002466.SZ", "天齐锂业", ["天齐锂业", "天齐"]),
    "603799": ("603799.SH", "华友钴业", ["华友钴业", "华友"]),
    "603993": ("603993.SH", "洛阳钼业", ["洛阳钼业", "CMOC"]),
    "600362": ("600362.SH", "江西铜业", ["江西铜业"]),
    "300919": ("300919.SZ", "中伟股份", ["中伟新材", "中伟股份"]),
    "603659": ("603659.SH", "璞泰来", ["璞泰来"]),
    "300037": ("300037.SZ", "新宙邦", ["新宙邦", "Capchem"]),
    "600110": ("600110.SH", "诺德股份", ["诺德股份", "诺德投资"]),
    "688388": ("688388.SH", "嘉元科技", ["嘉元科技", "嘉元铜箔"]),
    "002340": ("002340.SZ", "格林美", ["格林美"]),
    "920185": ("920185.BJ", "贝特瑞", ["贝特瑞"]),
    "300014": ("300014.SZ", "亿纬锂能", ["亿纬锂能", "EVE Energy"]),
    "300207": ("300207.SZ", "欣旺达", ["欣旺达"]),
    "002074": ("002074.SZ", "国轩高科", ["国轩高科", "Gotion"]),
    "600733": ("600733.SH", "北汽蓝谷", ["北汽蓝谷", "北汽新能源"]),
    "000625": ("000625.SZ", "长安汽车", ["长安汽车"]),
    "600006": ("600006.SH", "东风汽车", ["东风汽车"]),
    "601238": ("601238.SH", "广汽集团", ["广汽集团", "广汽"]),
    "601633": ("601633.SH", "长城汽车", ["长城汽车", "GWM"]),
    "600418": ("600418.SH", "江淮汽车", ["江淮汽车", "JAC"]),
}

# 중국어 리스크 키워드
RISK_KEYWORDS_CN = {
    "geopolitics": ["制裁", "关税", "贸易战", "出口管制", "实体清单", "禁令", "封锁"],
    "logistics": ["物流", "航运", "港口", "运输中断", "供应链中断", "断供"],
    "natural": ["地震", "洪水", "台风", "火灾", "爆炸", "自然灾害"],
    "regulatory": ["监管", "环保", "召回", "处罚", "整改", "合规"],
    "market": ["暴跌", "暴涨", "价格崩盘", "需求下降", "产能过剩", "库存积压", "亏损"],
    "labor": ["罢工", "裁员", "用工荒", "停工", "欠薪"],
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json, text/html",
}


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _classify_risk_cn(title: str) -> str:
    matched = []
    for rtype, keywords in RISK_KEYWORDS_CN.items():
        for kw in keywords:
            if kw in title:
                matched.append(rtype)
                break
    return ",".join(sorted(set(matched))) if matched else "other"


def _severity_cn(title: str) -> float:
    high = ["爆炸", "火灾", "制裁", "召回", "罢工", "暴跌", "中断", "危机", "崩盘"]
    mid = ["监管", "延迟", "不足", "下降", "下滑", "风险", "担忧"]
    score = 2.5
    for kw in high:
        if kw in title: score += 0.5
    for kw in mid:
        if kw in title: score += 0.3
    return min(score, 5.0)


# ── 新浪财经 수집 ──

def collect_sina_finance(output_dir: Path) -> List[dict]:
    """新浪财经 개별 기업 뉴스 페이지 수집."""
    all_rows = []

    for code, (cid, cn_name, keywords) in CHINESE_COMPANIES.items():
        log.info(f"[新浪财经] {cid} ({cn_name})")
        # 新浪财经 기업별 뉴스 API
        url = f"https://vip.stock.finance.sina.com.cn/corp/go.php/vCB_AllNewsStock/symbol/{_sina_symbol(code)}.phtml"

        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.encoding = "gb2312"

            # HTML에서 뉴스 링크 추출
            pattern = r'<a href="(https?://finance\.sina\.com\.cn/[^"]+)"[^>]*>([^<]+)</a>\s*\((\d{4}-\d{2}-\d{2})'
            matches = re.findall(pattern, r.text)

            for link, title, date_str in matches[:100]:
                all_rows.append({
                    "event_id": _md5(f"{link}|sina"),
                    "event_time": f"{date_str} 00:00:00",
                    "url": link,
                    "title": title.strip(),
                    "risk_types": _classify_risk_cn(title),
                    "severity": _severity_cn(title),
                    "entity_ids": [cid],
                    "entity_scores": [1.0],
                    "finbert": 0.0,
                    "country_ids": ["CN"],
                    "collection_type": "sina_finance_cn",
                    "source_domain": "finance.sina.com.cn",
                })

            log.info(f"  → {len(matches)} articles")
        except Exception as e:
            log.error(f"  Error: {e}")

        time.sleep(2)

    return all_rows


def _sina_symbol(code: str) -> str:
    """주식 코드를 新浪 심볼로 변환."""
    if code.startswith("6"):
        return f"sh{code}"
    elif code.startswith(("0", "3")):
        return f"sz{code}"
    elif code.startswith("9"):
        return f"bj{code}"
    return f"sz{code}"


# ── 东方财富 수집 ──

def collect_eastmoney(output_dir: Path) -> List[dict]:
    """东方财富 기업별 뉴스 수집."""
    all_rows = []

    for code, (cid, cn_name, keywords) in CHINESE_COMPANIES.items():
        log.info(f"[东方财富] {cid} ({cn_name})")
        market = "SH" if code.startswith("6") else "SZ" if code.startswith(("0", "3")) else "BJ"
        symbol = f"{market}{code}"

        url = f"https://search-api-web.eastmoney.com/search/jsonp?cb=jQuery&param="
        param = {
            "uid": "",
            "keyword": cn_name,
            "type": ["cmsArticleWebOld"],
            "client": "web",
            "clientType": "web",
            "clientVersion": "curr",
            "param": {"cmsArticleWebOld": {"searchScope": "default", "sort": "default", "pageIndex": 1, "pageSize": 50}},
        }

        try:
            r = requests.get(
                "https://searchapi.eastmoney.com/api/suggest/get",
                params={"input": cn_name, "type": "14", "token": "D43BF722C8E33BDC906FB84D85E326E8", "count": "50"},
                headers=HEADERS, timeout=15,
            )

            # 东方财富 뉴스 리스트 API (simpler)
            news_url = f"https://emweb.securities.eastmoney.com/PC_HSF10/CompanyManagement/PageAjax?code={symbol}"
            r2 = requests.get(news_url, headers=HEADERS, timeout=15)
            data = r2.json() if r2.status_code == 200 else {}

            # 뉴스 관련 데이터가 있으면 수집
            for key in ["ggsc", "zyfw"]:
                items = data.get(key, [])
                if isinstance(items, list):
                    for item in items[:20]:
                        title = item.get("TITLE", item.get("BUSINESS_SCOPE", ""))
                        if title:
                            all_rows.append({
                                "event_id": _md5(f"{symbol}|{title}|eastmoney"),
                                "event_time": item.get("NOTICE_DATE", item.get("REPORT_DATE", ""))[:19],
                                "url": f"https://data.eastmoney.com/stockdata/{code}.html",
                                "title": title[:200],
                                "risk_types": _classify_risk_cn(title),
                                "severity": _severity_cn(title),
                                "entity_ids": [cid],
                                "entity_scores": [1.0],
                                "finbert": 0.0,
                                "country_ids": ["CN"],
                                "collection_type": "eastmoney_cn",
                                "source_domain": "eastmoney.com",
                            })

            log.info(f"  → {sum(1 for r in all_rows if cid in str(r.get('entity_ids')))} articles")
        except Exception as e:
            log.error(f"  Error: {e}")

        time.sleep(2)

    return all_rows


# ── Google News CN (중국어 검색) ──

def collect_google_news_cn(output_dir: Path) -> List[dict]:
    """Google News 중국어 검색으로 뉴스 수집."""
    import feedparser

    all_rows = []
    for code, (cid, cn_name, keywords) in CHINESE_COMPANIES.items():
        for kw in keywords[:1]:  # 첫 번째 키워드만
            query = f"{kw} 电池 供应链"
            url = f"https://news.google.com/rss/search?q={urllib.parse.quote_plus(query)}&hl=zh-CN&gl=CN&ceid=CN:zh-Hans"
            log.info(f"[Google CN] {cid}: {kw}")

            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:30]:
                    title = entry.get("title", "")
                    link = entry.get("link", "")
                    pub = entry.get("published", "")

                    all_rows.append({
                        "event_id": _md5(f"{link}|google_cn"),
                        "event_time": pub,
                        "url": link,
                        "title": title,
                        "risk_types": _classify_risk_cn(title),
                        "severity": _severity_cn(title),
                        "entity_ids": [cid],
                        "entity_scores": [1.0],
                        "finbert": 0.0,
                        "country_ids": ["CN"],
                        "collection_type": "google_news_cn",
                        "source_domain": "news.google.com",
                    })
            except Exception as e:
                log.error(f"  Error: {e}")

            time.sleep(3)

    return all_rows


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="중국 뉴스 수집기")
    parser.add_argument("--source", choices=["sina", "eastmoney", "google", "all"], default="all")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--output", default="data/raw/chinese_news/chinese_news.parquet")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []

    if args.source in ("sina", "all"):
        all_rows.extend(collect_sina_finance(output_path.parent))

    if args.source in ("eastmoney", "all"):
        all_rows.extend(collect_eastmoney(output_path.parent))

    if args.source in ("google", "all"):
        all_rows.extend(collect_google_news_cn(output_path.parent))

    if not all_rows:
        log.warning("No data collected")
        return

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["url"], keep="first")
    df.to_parquet(output_path, index=False)
    log.info(f"Saved {len(df):,} articles to {output_path}")

    # Per-company stats
    for code, (cid, cn_name, _) in CHINESE_COMPANIES.items():
        n = df[df["entity_ids"].apply(lambda x: cid in x if isinstance(x, list) else False)].shape[0]
        if n > 0:
            log.info(f"  {cid} ({cn_name}): {n:,}")


if __name__ == "__main__":
    main()
