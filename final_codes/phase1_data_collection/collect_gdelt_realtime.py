"""
collect_gdelt_realtime.py
─────────────────────────
GDELT DOC API 실시간 수집기 — Dynamic KG용 증분 파이프라인.

동적 KG의 핵심: 시간에 따라 변하는 엣지 가중치.
  → 매일/매시간 새 뉴스 수집 → risk_events 변환 → KG 엣지 업데이트

실행 모드:
  1. 단발성: python scripts/collect_gdelt_realtime.py --days 7
  2. 스케줄러: python scripts/collect_gdelt_realtime.py --daemon --interval 3600
  3. 테스트:  python scripts/collect_gdelt_realtime.py --days 1 --dry-run

데이터 흐름:
  GDELT DOC API → raw articles → Aho-Corasick 기업매칭 →
  risk_type 분류 → severity 계산 → risk_events_realtime.parquet (증분 append)

이 파일은 BigQuery 배치(과거)와 상호보완:
  BigQuery GKG  : 2022~2025 과거 데이터, 1회성 배치
  DOC API 실시간: 오늘~미래, 매일/매시간 증분
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import yaml

log = logging.getLogger("gdelt_realtime")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

# ─── Paths ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
ALIASES_YAML = ROOT / "configs" / "company_aliases.yaml"
RISK_KEYWORDS_YAML = ROOT / "configs" / "risk_keywords.yaml"
OUTPUT_DIR = ROOT / "data" / "raw" / "gdelt_realtime"
CHECKPOINT_FILE = OUTPUT_DIR / "_checkpoint_realtime.json"
MERGED_OUTPUT = ROOT / "data" / "processed" / "risk_events_realtime.parquet"

# GDELT DOC API
DOC_API_BASE = "https://api.gdeltproject.org/api/v2/doc/doc"
MAX_RECORDS = 250  # DOC API limit per request
REQUEST_DELAY = 10.0  # seconds between API calls (GDELT rate limit: ~6 req/min)


# ─── Company Alias Matching (Aho-Corasick) ───────────────────────

def _build_ac_matcher(aliases_yaml: Path) -> Tuple[Any, Dict[str, str]]:
    """Aho-Corasick 자동완성 트리 빌드."""
    try:
        import ahocorasick_rs
        use_ac = True
    except ImportError:
        use_ac = False
        log.warning("ahocorasick_rs not installed, using fallback regex matching")

    with open(aliases_yaml, "r", encoding="utf-8") as f:
        aliases = yaml.safe_load(f)

    # keyword → company_id 매핑
    keyword_to_cid: Dict[str, str] = {}
    all_keywords: List[str] = []

    # Common word blocklist
    blocklist = {"US", "IT", "A", "AI", "AN", "AM", "ARE", "BE", "BY", "DO",
                 "GO", "HE", "IN", "IS", "ME", "MY", "NO", "OF", "OK", "ON",
                 "OR", "SO", "TO", "UP", "WE"}

    for cid, alias_list in aliases.items():
        if not isinstance(alias_list, list):
            continue
        for alias in alias_list:
            alias_upper = alias.upper().strip()
            if alias_upper in blocklist or len(alias_upper) < 2:
                continue
            keyword_to_cid[alias_upper] = cid
            all_keywords.append(alias_upper)

    if use_ac:
        ac = ahocorasick_rs.AhoCorasick(all_keywords, matchkind=ahocorasick_rs.MATCHKIND_LEFTMOST_LONGEST)
        return ac, keyword_to_cid
    else:
        # Fallback: compile mega regex
        escaped = [re.escape(kw) for kw in all_keywords]
        pattern = re.compile("|".join(escaped), re.IGNORECASE)
        return pattern, keyword_to_cid


def match_companies(text: str, matcher, keyword_to_cid: Dict[str, str]) -> List[str]:
    """텍스트에서 기업 매칭."""
    if not text:
        return []

    text_upper = text.upper()
    matched_cids = set()

    try:
        import ahocorasick_rs
        # Aho-Corasick
        keywords = list(keyword_to_cid.keys())
        for match in matcher.find_matches_as_indexes(text_upper):
            kw = keywords[match[0]]
            matched_cids.add(keyword_to_cid[kw])
    except ImportError:
        # Regex fallback
        for m in matcher.finditer(text_upper):
            kw = m.group().upper()
            if kw in keyword_to_cid:
                matched_cids.add(keyword_to_cid[kw])

    return sorted(matched_cids)


# ─── Risk Classification ────────────────────────────────────────

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
    matched = set()
    for risk_type, pat in risk_patterns:
        if pat.search(text):
            matched.add(risk_type)
    return ",".join(sorted(matched)) if matched else "other"


def estimate_severity(text: str, risk_patterns: List[Tuple[str, re.Pattern]]) -> float:
    count = sum(1 for _, pat in risk_patterns if pat.search(text))
    if count == 0:
        return 1.5
    return min(1.5 + count * 0.5, 5.0)


# ─── GDELT DOC API ──────────────────────────────────────────────

def build_query_terms(aliases: Dict[str, List[str]]) -> List[str]:
    """기업 alias에서 DOC API 검색 쿼리 생성. 주요 기업 위주."""
    # 우선순위 기업 (데이터 부족 + 핵심 기업)
    priority_companies = [
        "CATL", "LG Energy Solution", "BYD", "Samsung SDI", "Panasonic",
        "BMW", "Toyota", "Tesla", "Volkswagen", "Mercedes-Benz",
        "CALB", "CNGR", "Nuode", "Li-Cycle", "BTR",
    ]
    # Supply chain risk 키워드
    risk_terms = [
        "battery supply chain",
        "EV battery risk",
        "lithium shortage",
        "cathode material disruption",
        "battery recall",
        "cobalt supply",
        "nickel shortage",
        "battery fire",
        "EV supply chain sanction",
        "battery factory",
    ]
    return priority_companies + risk_terms


def fetch_doc_api(
    query: str,
    start_date: str,
    end_date: str,
    max_records: int = MAX_RECORDS,
) -> pd.DataFrame:
    """GDELT DOC API에서 기사 검색."""
    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": str(max_records),
        "startdatetime": start_date.replace("-", "") + "000000",
        "enddatetime": end_date.replace("-", "") + "235959",
        "format": "json",
    }

    try:
        resp = requests.get(DOC_API_BASE, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.JSONDecodeError:
        log.warning(f"  JSON decode error for query: {query}")
        return pd.DataFrame()
    except Exception as e:
        log.warning(f"  API error for query '{query}': {e}")
        return pd.DataFrame()

    articles = data.get("articles", [])
    if not articles:
        return pd.DataFrame()

    rows = []
    for art in articles:
        rows.append({
            "url": art.get("url", ""),
            "title": art.get("title", ""),
            "seendate": art.get("seendate", ""),
            "source_domain": art.get("domain", ""),
            "language": art.get("language", ""),
            "tone": float(art.get("tone", 0)),
        })

    return pd.DataFrame(rows)


def collect_period(
    start_date: str,
    end_date: str,
    aliases: Dict[str, List[str]],
    matcher,
    keyword_to_cid: Dict[str, str],
    risk_patterns: List[Tuple[str, re.Pattern]],
) -> pd.DataFrame:
    """특정 기간의 뉴스 수집 + 변환."""
    queries = build_query_terms(aliases)
    all_rows = []
    seen_urls = set()

    log.info(f"Collecting {start_date} ~ {end_date} ({len(queries)} queries)")

    for i, query in enumerate(queries):
        log.info(f"  [{i+1}/{len(queries)}] {query}")

        df = fetch_doc_api(query, start_date, end_date)
        if df.empty:
            log.info(f"    → 0 articles")
            time.sleep(REQUEST_DELAY)
            continue

        count = 0
        for _, row in df.iterrows():
            url = row["url"]
            if url in seen_urls or not url:
                continue
            seen_urls.add(url)

            title = row.get("title", "")
            tone = row.get("tone", 0.0)

            # Company matching
            entity_ids = match_companies(title, matcher, keyword_to_cid)

            # Risk classification
            risk_types = classify_risk(title, risk_patterns)
            severity = estimate_severity(title, risk_patterns)

            # Tone-based severity adjustment
            if tone < -5:
                severity = min(severity + 1.0, 5.0)
            elif tone < -2:
                severity = min(severity + 0.5, 5.0)

            # Parse date
            seendate = row.get("seendate", "")
            try:
                event_time = pd.Timestamp(seendate)
            except Exception:
                event_time = pd.Timestamp(start_date)

            event_id = hashlib.sha256(f"{url}|rt".encode()).hexdigest()[:16]

            all_rows.append({
                "event_id": event_id,
                "event_time": event_time,
                "url": url,
                "title": title,
                "risk_types": risk_types,
                "severity": severity,
                "entity_ids": entity_ids if entity_ids else [],
                "entity_scores": [
                    {"alias": cid, "company_id": cid, "score": 1.0}
                    for cid in entity_ids
                ],
                "finbert": None,
                "country_ids": "",
                "tone": tone,
                "source_domain": row.get("source_domain", ""),
                "collection_type": "gdelt_doc_api_realtime",
            })
            count += 1

        log.info(f"    → {count} new articles")
        time.sleep(REQUEST_DELAY)

    if not all_rows:
        return pd.DataFrame()

    result = pd.DataFrame(all_rows)
    result = result.drop_duplicates(subset=["url"], keep="first")
    return result


# ─── Checkpoint ──────────────────────────────────────────────────

def load_checkpoint() -> Optional[str]:
    """마지막 수집 날짜 로드."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            data = json.load(f)
        return data.get("last_date")
    return None


def save_checkpoint(date_str: str):
    """수집 완료 날짜 저장."""
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"last_date": date_str, "updated": datetime.now().isoformat()}, f)


# ─── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GDELT DOC API 실시간 수집기")
    parser.add_argument("--days", type=int, default=7, help="수집할 일수 (default: 7)")
    parser.add_argument("--start-date", type=str, help="시작일 (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="종료일 (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true", help="수집만, 저장 안 함")
    parser.add_argument("--daemon", action="store_true", help="반복 실행 모드")
    parser.add_argument("--interval", type=int, default=3600,
                        help="반복 주기 (초, default: 3600 = 1시간)")
    parser.add_argument("--resume", action="store_true",
                        help="마지막 체크포인트부터 이어서 수집")
    args = parser.parse_args()

    # Load configs
    log.info("Loading configurations...")
    matcher, keyword_to_cid = _build_ac_matcher(ALIASES_YAML)
    log.info(f"Company matcher: {len(keyword_to_cid)} keywords")

    risk_patterns = _load_risk_patterns(RISK_KEYWORDS_YAML)
    log.info(f"Risk patterns: {len(risk_patterns)}")

    with open(ALIASES_YAML, "r", encoding="utf-8") as f:
        aliases = yaml.safe_load(f)

    # Determine date range
    if args.start_date and args.end_date:
        start = args.start_date
        end = args.end_date
    elif args.resume:
        last = load_checkpoint()
        if last:
            start = (datetime.strptime(last, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            start = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")
        end = datetime.now().strftime("%Y-%m-%d")
    else:
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")

    def run_once():
        nonlocal start, end
        log.info("=" * 60)
        log.info(f"GDELT DOC API 실시간 수집: {start} ~ {end}")
        log.info("=" * 60)

        # Collect in daily chunks (DOC API 제한)
        all_dfs = []
        current = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")

        while current <= end_dt:
            day_str = current.strftime("%Y-%m-%d")
            next_day = (current + timedelta(days=1)).strftime("%Y-%m-%d")

            df = collect_period(day_str, next_day, aliases, matcher, keyword_to_cid, risk_patterns)

            if not df.empty:
                all_dfs.append(df)
                log.info(f"  {day_str}: {len(df)} articles")
            else:
                log.info(f"  {day_str}: 0 articles")

            save_checkpoint(day_str)
            current += timedelta(days=1)

        if not all_dfs:
            log.warning("No articles collected!")
            return pd.DataFrame()

        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.drop_duplicates(subset=["url"], keep="first")

        # Summary
        n_with_entity = combined["entity_ids"].apply(len).gt(0).sum()
        log.info(f"\n=== Summary ===")
        log.info(f"Total: {len(combined):,} articles")
        log.info(f"With entity match: {n_with_entity:,} ({100*n_with_entity/len(combined):.1f}%)")

        rt_counts = combined["risk_types"].value_counts()
        log.info("Risk types:")
        for rt, cnt in rt_counts.head(10).items():
            log.info(f"  {rt}: {cnt}")

        if args.dry_run:
            log.info("[DRY-RUN] Preview:")
            print(combined[["event_time", "title", "risk_types", "entity_ids"]].head(20).to_string())
            return combined

        # Save
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        date_tag = datetime.now().strftime("%Y%m%d_%H%M")
        output_path = OUTPUT_DIR / f"realtime_{date_tag}.parquet"
        combined.to_parquet(output_path, index=False)
        log.info(f"Saved: {output_path}")

        # Merge into cumulative file
        if MERGED_OUTPUT.exists():
            existing = pd.read_parquet(MERGED_OUTPUT)
            merged = pd.concat([existing, combined], ignore_index=True)
            merged = merged.drop_duplicates(subset=["url"], keep="first")
        else:
            merged = combined

        MERGED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(MERGED_OUTPUT, index=False)
        log.info(f"Merged total: {len(merged):,} → {MERGED_OUTPUT}")

        return combined

    if args.daemon:
        log.info(f"Daemon mode: running every {args.interval}s")
        while True:
            try:
                run_once()
                # Next run: today only
                start = datetime.now().strftime("%Y-%m-%d")
                end = start
            except KeyboardInterrupt:
                log.info("Daemon stopped by user")
                break
            except Exception as e:
                log.error(f"Error: {e}")

            log.info(f"Sleeping {args.interval}s until next run...")
            time.sleep(args.interval)
    else:
        run_once()

    log.info("Done!")


if __name__ == "__main__":
    main()
