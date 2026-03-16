"""GDELT v2 Raw File 실시간 수집기.

15분마다 갱신되는 GDELT v2 이벤트 파일을 다운로드하여
두 종류의 이벤트를 수집합니다:

1. company  — 유니버스 기업이 Actor로 등장하는 이벤트
               → 동적 공급망 관계 업데이트에 사용
2. risk     — 리스크 키워드가 포함된 이벤트 (제재, 지진, 해협 봉쇄 등)
               → 리스크 노출도 분석에 사용

event_category 필드로 각 이벤트의 매칭 유형을 구분합니다:
  - "company" : 기업 매칭만 된 이벤트
  - "risk"    : 리스크 키워드만 매칭된 이벤트
  - "both"    : 기업 + 리스크 키워드 모두 매칭된 이벤트

사용법:
    # 1회 수집
    python -m src.ingest.gdelt_realtime --once

    # 15분 주기 연속 수집
    python -m src.ingest.gdelt_realtime --interval 15
"""
from __future__ import annotations

import io
import hashlib
import logging
import zipfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yaml

from src.entity_link import _build_lookup, link_entities
from src.classify import classify_risk

# 짧은 검색어로 인한 오탐 방지 (e.g., 티커 "F"가 모든 텍스트에 매칭)
MIN_SEARCH_TERM_LENGTH = 3

logger = logging.getLogger("gdelt_realtime")

# GDELT v2 Raw File 엔드포인트
LASTUPDATE_URL = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"

# GDELT v2 Events export 컬럼 (61개 중 사용할 것만 정의)
EVENTS_COLUMNS = {
    0: "GLOBALEVENTID",
    1: "SQLDATE",
    6: "Actor1Name",
    16: "Actor2Name",
    26: "EventCode",
    30: "GoldsteinScale",
    32: "NumArticles",
    34: "AvgTone",
    60: "SOURCEURL",
}
TOTAL_COLUMNS = 61


@dataclass
class RealtimeConfig:
    """실시간 수집 설정."""
    universe_path: Path = Path("data/universe/univers_final.csv")
    keywords_yaml: Path = Path("configs/risk_keywords.yaml")
    output_dir: Path = Path("data/raw/gdelt/realtime")
    combined_csv: Path = Path("data/raw/gdelt/articles_realtime.csv")
    state_file: Path = Path("data/raw/gdelt/realtime/.last_fetched.txt")
    min_entity_score: int = 90
    max_entities: int = 3


@dataclass
class FetchResult:
    """1회 수집 결과."""
    timestamp: str = ""
    rows_downloaded: int = 0
    company_matched: int = 0
    risk_matched: int = 0
    both_matched: int = 0
    total_matched: int = 0
    new_events: int = 0
    skipped: bool = False


def _event_id(url: str, date: str) -> str:
    """SOURCEURL + SQLDATE 기반 이벤트 ID 생성."""
    raw = f"{url}_{date}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def _load_keyword_map(yaml_path: Path) -> dict:
    """risk_keywords.yaml를 로드."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_latest_export_url() -> tuple[str, str]:
    """lastupdate.txt에서 최신 이벤트 파일 URL과 타임스탬프 추출."""
    resp = requests.get(LASTUPDATE_URL, timeout=30)
    resp.raise_for_status()

    for line in resp.text.strip().split("\n"):
        parts = line.strip().split()
        if len(parts) >= 3 and ".export.CSV.zip" in parts[2]:
            url = parts[2]
            fname = url.split("/")[-1]
            timestamp = fname.split(".")[0]
            return url, timestamp

    raise RuntimeError("lastupdate.txt에서 export 파일 URL을 찾을 수 없습니다")


def _download_and_parse(url: str) -> pd.DataFrame:
    """zip 파일을 다운로드하여 DataFrame으로 파싱."""
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = [n for n in zf.namelist() if n.endswith(".CSV")][0]
        with zf.open(csv_name) as f:
            df = pd.read_csv(
                f,
                sep="\t",
                header=None,
                dtype=str,
                on_bad_lines="skip",
            )

    cols_to_keep = {idx: name for idx, name in EVENTS_COLUMNS.items() if idx < len(df.columns)}
    df = df[[idx for idx in cols_to_keep.keys()]].copy()
    df.columns = [cols_to_keep[idx] for idx in cols_to_keep.keys()]

    df["GoldsteinScale"] = pd.to_numeric(df["GoldsteinScale"], errors="coerce").fillna(0)
    df["AvgTone"] = pd.to_numeric(df["AvgTone"], errors="coerce").fillna(0)
    df["NumArticles"] = pd.to_numeric(df["NumArticles"], errors="coerce").fillna(1).astype(int)

    return df


def _filter_events(
    df: pd.DataFrame,
    lookup: list[dict],
    keyword_map: dict,
    min_fuzzy_score: int = 90,
    max_entities: int = 3,
) -> pd.DataFrame:
    """기업 매칭 OR 리스크 키워드 매칭으로 이벤트 필터링.

    조건:
      - company 매칭: Actor1Name 또는 Actor2Name이 유니버스 기업과 매칭
      - risk 매칭: SOURCEURL + Actor 이름에 리스크 키워드 포함

    하나라도 해당되면 수집 대상. event_category로 유형 구분.
    """
    results = []

    for _, row in df.iterrows():
        # --- Company 매칭: Actor 이름으로 기업 식별 ---
        all_entity_ids = []
        all_entity_mentions = []
        for col in ["Actor1Name", "Actor2Name"]:
            actor = row.get(col)
            if pd.notna(actor) and str(actor).strip():
                ids, mentions = link_entities(
                    str(actor), lookup, min_fuzzy_score, max_entities
                )
                all_entity_ids.extend(ids)
                all_entity_mentions.extend(mentions)

        # 중복 제거 (순서 유지)
        seen = set()
        unique_ids = []
        for eid in all_entity_ids:
            if eid not in seen:
                seen.add(eid)
                unique_ids.append(eid)
        unique_ids = unique_ids[:max_entities]

        has_company = len(unique_ids) > 0

        # --- Risk 키워드 매칭: SOURCEURL + Actor 이름에서 키워드 검색 ---
        text_parts = []
        for col in ["SOURCEURL", "Actor1Name", "Actor2Name"]:
            val = row.get(col)
            if pd.notna(val):
                text_parts.append(str(val))
        text = " ".join(text_parts)

        risk_types = classify_risk(text, keyword_map)
        has_risk = risk_types != ["other"]

        # --- OR 조건: 둘 중 하나라도 매칭되면 수집 ---
        if has_company or has_risk:
            if has_company and has_risk:
                category = "both"
            elif has_company:
                category = "company"
            else:
                category = "risk"

            result_row = row.copy()
            result_row["entity_ids"] = unique_ids
            result_row["risk_types"] = ",".join(risk_types) if has_risk else ""
            result_row["event_category"] = category
            results.append(result_row)

    if results:
        return pd.DataFrame(results)
    return pd.DataFrame()


def _compute_severity(df: pd.DataFrame) -> pd.DataFrame:
    """GoldsteinScale + AvgTone 기반 심각도 계산."""
    gs = df["GoldsteinScale"].values.astype(float)
    at = df["AvgTone"].values.astype(float)

    norm_gs = np.clip((10 - gs) / 20.0, 0, 1)
    norm_at = np.clip((10 - at) / 20.0, 0, 1)

    df["severity"] = np.clip(1.0 + (norm_gs * 2.5) + (norm_at * 1.5), 1.0, 5.0)
    return df


def _generate_event_ids(df: pd.DataFrame) -> pd.DataFrame:
    """이벤트 ID 생성."""
    df["event_id"] = [
        _event_id(str(u), str(d))
        for u, d in zip(df["SOURCEURL"], df["SQLDATE"])
    ]
    return df


def _read_last_fetched(state_file: Path) -> Optional[str]:
    """마지막으로 수집한 타임스탬프 읽기."""
    if state_file.exists():
        return state_file.read_text(encoding="utf-8").strip()
    return None


def _write_last_fetched(state_file: Path, timestamp: str):
    """마지막 수집 타임스탬프 저장."""
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(timestamp, encoding="utf-8")


def fetch_once(cfg: RealtimeConfig) -> FetchResult:
    """1회 수집 실행."""
    result = FetchResult()

    url, timestamp = _get_latest_export_url()
    result.timestamp = timestamp

    last = _read_last_fetched(cfg.state_file)
    if last == timestamp:
        result.skipped = True
        logger.info(f"[skip] {timestamp} already fetched")
        return result

    logger.info(f"[fetch] {timestamp} downloading...")

    df_raw = _download_and_parse(url)
    result.rows_downloaded = len(df_raw)
    logger.info(f"[fetch] {len(df_raw)} rows downloaded")

    # 유니버스 기업 룩업 테이블 구축 (짧은 검색어 제거)
    universe = pd.read_csv(cfg.universe_path, encoding="utf-8-sig")
    lookup = _build_lookup(universe)
    for entry in lookup:
        entry["terms"] = {t for t in entry["terms"] if len(t) >= MIN_SEARCH_TERM_LENGTH}
    lookup = [e for e in lookup if e["terms"]]
    logger.info(f"[filter] universe loaded: {len(lookup)} companies")

    # 리스크 키워드 맵 로드
    keyword_map = _load_keyword_map(cfg.keywords_yaml)
    logger.info(f"[filter] risk keywords loaded: {list(keyword_map.keys())}")

    # 이중 필터 적용 (company OR risk)
    df_matched = _filter_events(
        df_raw, lookup, keyword_map,
        min_fuzzy_score=cfg.min_entity_score,
        max_entities=cfg.max_entities,
    )

    if df_matched.empty:
        logger.info(f"[fetch] no matching events in {timestamp}")
        _write_last_fetched(cfg.state_file, timestamp)
        return result

    # 카테고리별 통계
    cat_counts = df_matched["event_category"].value_counts()
    result.company_matched = int(cat_counts.get("company", 0))
    result.risk_matched = int(cat_counts.get("risk", 0))
    result.both_matched = int(cat_counts.get("both", 0))
    result.total_matched = len(df_matched)

    logger.info(
        f"[filter] matched {len(df_matched)} events "
        f"(company={result.company_matched}, risk={result.risk_matched}, both={result.both_matched})"
    )

    # 심각도 계산
    df_matched = _compute_severity(df_matched)

    # 이벤트 ID 생성
    df_matched = _generate_event_ids(df_matched)

    # 날짜 변환
    df_matched["event_date"] = pd.to_datetime(
        df_matched["SQLDATE"], format="%Y%m%d", errors="coerce"
    )

    # 출력 컬럼 정리
    out_cols = [
        "GLOBALEVENTID", "event_id", "event_date", "event_category",
        "Actor1Name", "Actor2Name", "EventCode",
        "GoldsteinScale", "AvgTone", "NumArticles",
        "SOURCEURL", "entity_ids", "severity", "risk_types",
    ]
    out_df = df_matched[[c for c in out_cols if c in df_matched.columns]]

    # 청크 파일 저장
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = cfg.output_dir / f"{timestamp}.csv"
    out_df.to_csv(chunk_path, index=False, encoding="utf-8-sig")
    logger.info(f"[save] {chunk_path} ({len(out_df)} rows)")

    # 통합 CSV 업데이트
    cfg.combined_csv.parent.mkdir(parents=True, exist_ok=True)
    if cfg.combined_csv.exists():
        existing = pd.read_csv(cfg.combined_csv, dtype={"GLOBALEVENTID": str})
        combined = pd.concat([existing, out_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["GLOBALEVENTID"], keep="last")
    else:
        combined = out_df

    combined.to_csv(cfg.combined_csv, index=False, encoding="utf-8-sig")
    result.new_events = len(out_df)
    logger.info(f"[save] combined: {cfg.combined_csv} (total {len(combined)} rows)")

    _write_last_fetched(cfg.state_file, timestamp)

    return result


def run_realtime_loop(cfg: RealtimeConfig, interval_minutes: int = 15):
    """주기적 수집 루프."""
    logger.info(f"[start] realtime collector (interval={interval_minutes}min)")
    logger.info(f"[config] universe={cfg.universe_path}")
    logger.info(f"[config] keywords={cfg.keywords_yaml}")
    logger.info(f"[config] output={cfg.combined_csv}")

    while True:
        try:
            result = fetch_once(cfg)
            if not result.skipped:
                logger.info(
                    f"[result] ts={result.timestamp} "
                    f"downloaded={result.rows_downloaded} "
                    f"matched={result.total_matched} "
                    f"(company={result.company_matched} "
                    f"risk={result.risk_matched} "
                    f"both={result.both_matched}) "
                    f"new={result.new_events}"
                )
        except Exception as e:
            logger.error(f"[error] {e}")

        logger.info(f"[wait] next fetch in {interval_minutes} minutes...")
        time.sleep(interval_minutes * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GDELT v2 realtime collector")
    parser.add_argument("--once", action="store_true", help="1회만 수집")
    parser.add_argument("--interval", type=int, default=15, help="수집 주기 (분)")
    parser.add_argument("--universe", type=str, default="data/universe/univers_final.csv")
    parser.add_argument("--keywords", type=str, default="configs/risk_keywords.yaml")
    parser.add_argument("--output-dir", type=str, default="data/raw/gdelt/realtime")
    parser.add_argument("--combined", type=str, default="data/raw/gdelt/articles_realtime.csv")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    cfg = RealtimeConfig(
        universe_path=Path(args.universe),
        keywords_yaml=Path(args.keywords),
        output_dir=Path(args.output_dir),
        combined_csv=Path(args.combined),
    )

    if args.once:
        fetch_once(cfg)
    else:
        run_realtime_loop(cfg, interval_minutes=args.interval)
