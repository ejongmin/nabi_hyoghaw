"""
supplement_news_template.py
───────────────────────────
Template script for supplementing news data for data-poor companies.

Usage:
  1. Collect news articles from alternative sources (see SOURCES section)
  2. Convert to the schema below
  3. Run this script to validate + merge into existing risk_events parquet

Targets (7 data-poor companies):
  - 600110.SH  (Nuode)        : 0 events   ← ZERO
  - 300919.SZ  (CNGR)         : 3 events   ← 극소
  - LICY       (Li-Cycle)     : 3 events   ← 극소
  - 688388.SH  (Jiayuan)      : 23 events  ← 극소
  - 603659.SH  (Putailai)     : 26 events  ← 극소
  - 3931.HK    (CALB)         : 105 events ← 부족
  - 300037.SZ  (Capchem)      : 109 events ← 부족
"""
from __future__ import annotations
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime
import hashlib
import json

# ─── Schema (MUST match _merged_risk_events.parquet) ─────────────────

REQUIRED_SCHEMA = pa.schema([
    ("event_id",        pa.string()),
    ("event_time",      pa.timestamp("ns")),
    ("url",             pa.string()),
    ("title",           pa.string()),
    ("risk_types",      pa.string()),         # comma-separated: "regulatory,market"
    ("severity",        pa.float64()),         # 0.0 ~ 1.0
    ("entity_ids",      pa.list_(pa.string())),
    ("entity_scores",   pa.list_(pa.struct([
        ("alias",       pa.string()),
        ("company_id",  pa.string()),
        ("score",       pa.float64()),
    ]))),
    ("finbert",         pa.null()),            # placeholder for FinBERT (null for now)
    ("country_ids",     pa.string()),          # comma-separated country codes
    ("tone",            pa.float64()),         # GDELT tone: -10 ~ +10
    ("source_domain",   pa.string()),
    ("collection_type", pa.string()),          # "gdelt_gkg" or "supplement_xxx"
])


def make_event_id(url: str, company_id: str, event_time: str) -> str:
    """Generate deterministic event_id to avoid duplicates."""
    raw = f"{url}|{company_id}|{event_time}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def build_supplement_row(
    url: str,
    title: str,
    event_time: str,           # ISO format: "2024-06-15T10:30:00"
    company_id: str,           # e.g., "600110.SH"
    alias_matched: str,        # e.g., "Nuode"
    risk_types: str = "",      # comma-separated risk categories
    severity: float = 0.5,
    tone: float = 0.0,
    country_ids: str = "CH",
    source_domain: str = "",
    collection_type: str = "supplement_manual",
) -> dict:
    """Build a single event row matching the risk_events schema."""
    eid = make_event_id(url, company_id, event_time)
    return {
        "event_id": eid,
        "event_time": pd.Timestamp(event_time),
        "url": url,
        "title": title,
        "risk_types": risk_types,
        "severity": severity,
        "entity_ids": [company_id],
        "entity_scores": [{"alias": alias_matched, "company_id": company_id, "score": 1.0}],
        "finbert": None,
        "country_ids": country_ids,
        "tone": tone,
        "source_domain": source_domain,
        "collection_type": collection_type,
    }


def validate_and_merge(
    supplement_df: pd.DataFrame,
    merged_path: Path = Path("data/raw/gdelt/risk_events_checkpoint/_merged_risk_events.parquet"),
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Validate supplement data and merge with existing risk_events."""
    # Validate required columns
    required_cols = [f.name for f in REQUIRED_SCHEMA]
    missing = set(required_cols) - set(supplement_df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Type coercion
    supplement_df["event_time"] = pd.to_datetime(supplement_df["event_time"], errors="coerce")
    supplement_df["severity"] = pd.to_numeric(supplement_df["severity"], errors="coerce").clip(0, 1)
    supplement_df["tone"] = pd.to_numeric(supplement_df["tone"], errors="coerce")

    # Load existing
    existing = pd.read_parquet(merged_path)
    print(f"Existing events: {len(existing):,}")
    print(f"Supplement events: {len(supplement_df):,}")

    # Deduplicate by event_id
    merged = pd.concat([existing, supplement_df], ignore_index=True)
    before = len(merged)
    merged = merged.drop_duplicates(subset=["event_id"], keep="first")
    print(f"After dedup: {len(merged):,} (removed {before - len(merged)} duplicates)")

    if output_path is None:
        output_path = merged_path
    merged.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")
    return merged


# ─── ALTERNATIVE DATA SOURCES ────────────────────────────────────────
#
# ┌──────────────┬─────────────────────────────────────────────────────────┐
# │ Source        │ Coverage & Notes                                       │
# ├──────────────┼─────────────────────────────────────────────────────────┤
# │ Yicai Global  │ https://www.yicaiglobal.com — English-language Chinese │
# │              │ business news. Good for Nuode, Jiayuan, CNGR.          │
# │              │ Has direct articles on Nuode Belgium plant, supply deals│
# ├──────────────┼─────────────────────────────────────────────────────────┤
# │ SMM (Shanghai │ https://news.metal.com — Shanghai Metals Market.       │
# │ Metals Market)│ English news on copper foil, anode, cathode precursors.│
# │              │ Covers Nuode, Jiayuan, Putailai directly.              │
# ├──────────────┼─────────────────────────────────────────────────────────┤
# │ CnEVPost     │ https://cnevpost.com — Chinese EV industry in English. │
# │              │ Good for CALB, CNGR, Capchem supply chain news.        │
# ├──────────────┼─────────────────────────────────────────────────────────┤
# │ Battery-News │ https://battery-news.de/en — European battery industry. │
# │              │ Covers CNGR JV (revomet), CALB Portugal factory.       │
# ├──────────────┼─────────────────────────────────────────────────────────┤
# │ EnergyTrend  │ https://www.energytrend.com — Battery market data.     │
# │              │ Covers CALB capacity expansion, market trends.         │
# ├──────────────┼─────────────────────────────────────────────────────────┤
# │ Kallanish    │ https://www.kallanish.com — Battery materials news.     │
# │              │ Has dedicated CNGR article feed.                       │
# ├──────────────┼─────────────────────────────────────────────────────────┤
# │ TYCORUN      │ https://www.tycorun.com — Battery industry analysis.   │
# │              │ Copper foil rankings covering Nuode, Jiayuan.          │
# ├──────────────┼─────────────────────────────────────────────────────────┤
# │ Li-Cycle IR  │ https://investors.li-cycle.com — Li-Cycle investor     │
# │              │ relations. Press releases, SEC filings, earnings.      │
# ├──────────────┼─────────────────────────────────────────────────────────┤
# │ GDELT (보완)  │ BigQuery에서 중국어 회사명으로 재검색:                     │
# │              │ "诺德股份"(Nuode), "中伟新材"(CNGR), "嘉元科技"(Jiayuan)    │
# │              │ "璞泰来"(Putailai), "中创新航"(CALB), "新宙邦"(Capchem)    │
# │              │ → company_aliases.yaml에 중문 alias 추가 후 재매칭      │
# └──────────────┴─────────────────────────────────────────────────────────┘
#
# ─── RECOMMENDED APPROACH (우선순위) ──────────────────────────────────
#
# 1순위: GDELT 중문 alias 추가 매칭 (무료, 대량, 기존 파이프라인 호환)
#   → company_aliases.yaml에 한자 alias 추가
#   → gdelt_bq.py 재실행 or 별도 BigQuery 쿼리
#
# 2순위: Yicai Global + SMM + CnEVPost 크롤링 (영문, 품질 높음)
#   → 각 사이트에서 기업명 검색 → 기사 수집 → 이 스크립트로 변환
#
# 3순위: Google News RSS (무료, 자동화 가능)
#   → https://news.google.com/rss/search?q="Nuode+copper+foil"
#   → RSS 파싱 → risk keyword 매칭 → 변환
# ────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    # Example: manually collected articles for Nuode
    rows = [
        build_supplement_row(
            url="https://www.yicaiglobal.com/news/chinas-nuode-to-build-usd553-million-copper-foil-plant-in-belgium",
            title="China's Nuode to Build USD553 Million Copper Foil Plant in Belgium",
            event_time="2023-05-18T08:00:00",
            company_id="600110.SH",
            alias_matched="Nuode",
            risk_types="regulatory",
            severity=0.4,
            tone=2.5,
            country_ids="BE,CH",
            source_domain="yicaiglobal.com",
            collection_type="supplement_yicai",
        ),
        build_supplement_row(
            url="https://www.yicaiglobal.com/news/chinas-nuode-becomes-copper-foil-supplier-of-unit-of-indias-biggest-storage-battery-maker",
            title="China's Nuode to Supply Copper Foil to Unit of India's Biggest Energy Storage Battery Maker",
            event_time="2024-03-15T08:00:00",
            company_id="600110.SH",
            alias_matched="Nuode",
            risk_types="market",
            severity=0.3,
            tone=3.0,
            country_ids="IN,CH",
            source_domain="yicaiglobal.com",
            collection_type="supplement_yicai",
        ),
    ]

    df = pd.DataFrame(rows)
    print("Sample supplement data:")
    print(df[["event_id","event_time","title","risk_types","collection_type"]].to_string(index=False))
    print(f"\nSchema columns: {list(df.columns)}")
    print("Ready to merge with: validate_and_merge(df)")
