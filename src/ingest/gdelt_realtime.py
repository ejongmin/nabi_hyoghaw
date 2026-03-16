"""GDELT v2 Raw File 실시간 수집기.

15분마다 갱신되는 GDELT v2 이벤트 파일을 다운로드하여
유니버스 기업과 관련된 이벤트만 필터링하고 저장합니다.

사용법:
    # 1회 수집
    python -m src.ingest.gdelt_realtime --once

    # 15분 주기 연속 수집
    python -m src.ingest.gdelt_realtime --interval 15
"""
from __future__ import annotations

import io
import hashlib
import zipfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from src.common.log import get_logger
from src.nlp.entity_linking import EntityLinker
from src.nlp.risk_scoring import RiskKeywordModel, _event_id

logger = get_logger("gdelt_realtime")

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
    rows_matched: int = 0
    new_events: int = 0
    skipped: bool = False


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


def _filter_by_universe(df: pd.DataFrame, linker: EntityLinker) -> pd.DataFrame:
    """유니버스 기업과 관련된 이벤트만 필터링."""

    def link_actors(row):
        hits = []
        for col in ["Actor1Name", "Actor2Name"]:
            actor = row.get(col)
            if pd.notna(actor) and str(actor).strip():
                res = linker.link_title(str(actor))
                hits.extend(res)
        best = {}
        for cid, score, alias in hits:
            if cid not in best or score > best[cid][0]:
                best[cid] = (score, alias)
        return list(best.keys())

    df["entity_ids"] = df.apply(link_actors, axis=1)
    matched = df[df["entity_ids"].apply(len) > 0].copy()
    return matched


def _compute_severity(df: pd.DataFrame) -> pd.DataFrame:
    """GoldsteinScale + AvgTone 기반 심각도 계산."""
    gs = df["GoldsteinScale"].values
    at = df["AvgTone"].values

    norm_gs = np.clip((10 - gs) / 20.0, 0, 1)
    norm_at = np.clip((10 - at) / 20.0, 0, 1)

    df["severity"] = np.clip(1.0 + (norm_gs * 2.5) + (norm_at * 1.5), 1.0, 5.0)
    return df


def _classify_risk(df: pd.DataFrame, risk_model: RiskKeywordModel) -> pd.DataFrame:
    """SOURCEURL + Actor 이름으로 리스크 유형 분류."""
    def classify_row(row):
        text_parts = []
        for col in ["SOURCEURL", "Actor1Name", "Actor2Name"]:
            val = row.get(col)
            if pd.notna(val):
                text_parts.append(str(val))
        text = " ".join(text_parts)
        types = risk_model.classify(text)
        return ",".join(types)

    df["risk_types"] = df.apply(classify_row, axis=1)
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

    universe = pd.read_csv(cfg.universe_path)
    linker = EntityLinker.from_universe(
        universe, min_score=cfg.min_entity_score, max_entities=cfg.max_entities
    )
    df_matched = _filter_by_universe(df_raw, linker)
    result.rows_matched = len(df_matched)

    if df_matched.empty:
        logger.info(f"[fetch] no matching events in {timestamp}")
        _write_last_fetched(cfg.state_file, timestamp)
        return result

    df_matched = _compute_severity(df_matched)

    risk_model = RiskKeywordModel.load(cfg.keywords_yaml)
    df_matched = _classify_risk(df_matched, risk_model)

    df_matched = _generate_event_ids(df_matched)

    df_matched["event_date"] = pd.to_datetime(df_matched["SQLDATE"], format="%Y%m%d", errors="coerce")

    out_cols = [
        "GLOBALEVENTID", "event_id", "event_date", "Actor1Name", "Actor2Name",
        "EventCode", "GoldsteinScale", "AvgTone", "NumArticles",
        "SOURCEURL", "entity_ids", "severity", "risk_types",
    ]
    out_df = df_matched[[c for c in out_cols if c in df_matched.columns]]

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = cfg.output_dir / f"{timestamp}.csv"
    out_df.to_csv(chunk_path, index=False, encoding="utf-8-sig")
    logger.info(f"[save] {chunk_path} ({len(out_df)} rows)")

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
    logger.info(f"[config] output={cfg.combined_csv}")

    while True:
        try:
            result = fetch_once(cfg)
            if not result.skipped:
                logger.info(
                    f"[result] ts={result.timestamp} "
                    f"downloaded={result.rows_downloaded} "
                    f"matched={result.rows_matched} "
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

    from src.common.log import setup_logging
    setup_logging("INFO")

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
