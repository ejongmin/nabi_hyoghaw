#!/usr/bin/env python
"""
팀원이 수집한 parquet 파일 → risk_events 스키마로 변환

팀원 파일 스키마:
  gkg_id, source_domain, article_url, risk_matched, event_code,
  event_root_code, quad_class, goldstein_scale, event_avg_tone,
  num_articles, collection_type, company_ids, tone, severity,
  country_ids, date

우리 risk_events 스키마:
  event_id, event_time, url, title, risk_types, severity,
  entity_ids, entity_scores, finbert, country_ids,
  collection_type, source_domain

사용법:
  python scripts/convert_team_parquet.py
  python scripts/convert_team_parquet.py --input data/raw/gdelt/gdelt_gkg_2019~2020.parquet
"""
import argparse
import hashlib
import logging
import re
from pathlib import Path
from typing import List

import pandas as pd
import yaml

log = logging.getLogger("convert_team")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


def _load_risk_keywords(yaml_path: str = "configs/risk_keywords.yaml") -> dict:
    """risk_keywords.yaml에서 키워드 로드."""
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        patterns = {}
        for rtype, keywords in data.items():
            if isinstance(keywords, list):
                patterns[rtype] = [kw.lower() for kw in keywords]
        return patterns
    except Exception:
        # fallback
        return {
            "geopolitics": ["sanction", "tariff", "trade war", "embargo", "geopolit"],
            "logistics": ["shipping", "port", "logistics", "supply chain", "shortage"],
            "natural": ["earthquake", "flood", "fire", "wildfire", "typhoon"],
            "regulatory": ["regulat", "recall", "compliance", "penalty"],
            "market": ["price crash", "demand", "oversupply", "inventory"],
            "labor": ["strike", "layoff", "workforce"],
        }


def _classify_risk_from_url(url: str, risk_patterns: dict) -> str:
    """URL과 source_domain에서 risk_type 추정 (title이 없으므로)."""
    url_lower = url.lower()
    matched = []
    for rtype, keywords in risk_patterns.items():
        for kw in keywords:
            if kw in url_lower:
                matched.append(rtype)
                break
    return ",".join(sorted(set(matched))) if matched else "other"


def _classify_risk_from_event_code(event_code: str) -> str:
    """CAMEO event code에서 risk_type 추정."""
    if pd.isna(event_code) or not event_code:
        return ""

    code = str(event_code)[:2]
    mapping = {
        "01": "",  # Make public statement
        "02": "",  # Appeal
        "03": "",  # Express intent to cooperate
        "04": "",  # Consult
        "05": "",  # Diplomatic cooperation
        "06": "market",  # Material cooperation
        "07": "",  # Provide aid
        "08": "",  # Yield
        "09": "",  # Investigate
        "10": "geopolitics",  # Demand
        "11": "geopolitics",  # Disapprove
        "12": "geopolitics",  # Reject
        "13": "geopolitics",  # Threaten
        "14": "geopolitics",  # Protest
        "15": "labor",  # Exhibit force
        "16": "geopolitics",  # Reduce relations
        "17": "geopolitics",  # Coerce
        "18": "geopolitics",  # Assault
        "19": "geopolitics",  # Fight
        "20": "geopolitics",  # Use unconventional mass violence
    }
    return mapping.get(code, "")


def convert_file(input_path: Path, risk_patterns: dict) -> pd.DataFrame:
    """팀원 parquet → risk_events 스키마 변환."""
    log.info(f"Loading {input_path.name}...")
    df = pd.read_parquet(input_path)
    log.info(f"  Raw rows: {len(df):,}")

    # ── 컬럼 변환 ──

    # event_id: gkg_id + article_url 해시
    df["event_id"] = (df["gkg_id"].fillna("") + "|" + df["article_url"].fillna("")).apply(
        lambda x: hashlib.md5(x.encode("utf-8")).hexdigest()
    )

    # event_time: date (YYYYMMDD) → datetime
    df["event_time"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")

    # url
    df["url"] = df["article_url"].fillna("")

    # title: 없음 (팀원 파일에 title 컬럼 없음)
    df["title"] = ""

    # risk_types: event_code + URL 기반 추정
    def _get_risk(row):
        parts = []
        # event code에서
        ec_risk = _classify_risk_from_event_code(row.get("event_code"))
        if ec_risk:
            parts.append(ec_risk)
        # URL에서
        url_risk = _classify_risk_from_url(str(row.get("article_url", "")), risk_patterns)
        if url_risk and url_risk != "other":
            parts.extend(url_risk.split(","))
        # risk_matched flag
        if row.get("risk_matched"):
            if not parts:
                parts.append("other")
        return ",".join(sorted(set(parts))) if parts else "other"

    log.info("  Classifying risk types...")
    df["risk_types"] = df.apply(_get_risk, axis=1)

    # severity: 이미 있음
    df["severity"] = df["severity"].fillna(2.5).clip(1.0, 5.0)

    # entity_ids: company_ids (세미콜론 구분) → list
    def _parse_ids(x):
        if pd.isna(x) or not x:
            return []
        return [cid.strip() for cid in str(x).split(";") if cid.strip()]

    df["entity_ids"] = df["company_ids"].apply(_parse_ids)
    df["entity_scores"] = df["entity_ids"].apply(lambda x: [1.0] * len(x))

    # finbert
    df["finbert"] = 0.0

    # country_ids: 세미콜론 구분 → list
    df["country_ids"] = df["country_ids"].apply(
        lambda x: [c.strip() for c in str(x).split(";")] if pd.notna(x) and x else []
    )

    # collection_type: 유지
    df["collection_type"] = df["collection_type"].fillna("company")

    # source_domain: 유지
    df["source_domain"] = df["source_domain"].fillna("")

    # 출력 컬럼만
    out_cols = [
        "event_id", "event_time", "url", "title", "risk_types",
        "severity", "entity_ids", "entity_scores", "finbert",
        "country_ids", "collection_type", "source_domain",
    ]
    result = df[out_cols].copy()

    # dedup
    n_before = len(result)
    result = result.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
    log.info(f"  Dedup: {n_before:,} → {len(result):,} ({n_before - len(result):,} removed)")

    return result


def main():
    parser = argparse.ArgumentParser(description="팀원 parquet → risk_events 변환")
    parser.add_argument("--input", nargs="*", help="입력 파일 (없으면 자동 감지)")
    parser.add_argument("--output", default="data/processed/risk_events_team.parquet")
    args = parser.parse_args()

    # 리스크 키워드 로드
    risk_patterns = _load_risk_keywords()
    log.info(f"Risk patterns: {len(risk_patterns)} types")

    # 입력 파일 감지
    if args.input:
        files = [Path(f) for f in args.input]
    else:
        gdelt_dir = Path("data/raw/gdelt")
        files = sorted(gdelt_dir.glob("gdelt_gkg_*~*.parquet"))
        if not files:
            log.error("No team parquet files found (pattern: gdelt_gkg_*~*.parquet)")
            return

    log.info(f"Files to convert: {len(files)}")
    for f in files:
        log.info(f"  {f.name}")

    # 변환
    all_dfs = []
    for f in files:
        df = convert_file(f, risk_patterns)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        log.error("No data converted")
        return

    combined = pd.concat(all_dfs, ignore_index=True)

    # 전체 dedup
    n_before = len(combined)
    combined = combined.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
    log.info(f"\nTotal: {n_before:,} → {len(combined):,} after dedup")

    # 저장
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)
    log.info(f"Saved: {output_path} ({len(combined):,} rows)")

    # 통계
    log.info("\n=== Summary ===")
    log.info(f"  Date range: {combined['event_time'].min()} ~ {combined['event_time'].max()}")
    log.info(f"  collection_type:")
    for ct, n in combined["collection_type"].value_counts().items():
        log.info(f"    {ct}: {n:,}")

    # 기업별
    all_entities = combined["entity_ids"].explode().dropna()
    entity_counts = all_entities.value_counts()
    log.info(f"  Companies matched: {entity_counts.nunique()}")
    log.info(f"  Top 10:")
    for cid, cnt in entity_counts.head(10).items():
        log.info(f"    {cid}: {cnt:,}")


if __name__ == "__main__":
    main()
