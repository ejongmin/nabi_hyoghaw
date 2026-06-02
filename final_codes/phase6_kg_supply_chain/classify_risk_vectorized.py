#!/usr/bin/env python3
"""
Risk 분류기 (벡터화 버전)
========================
기존 for-loop 방식 대비 ~50x 빠름.
핵심: 각 risk_type별 패턴을 하나의 mega-regex로 합쳐서
pd.Series.str.contains() 벡터 연산 (C 레벨) 활용.

사용법:
  python scripts/classify_risk_vectorized.py
  python scripts/classify_risk_vectorized.py --dry-run
  python scripts/classify_risk_vectorized.py --all   # 기존 분류도 재분류
"""
import argparse
import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger("risk_vec")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)


def load_risk_config(path: str = "configs/risk_keywords.yaml") -> dict:
    """risk_keywords.yaml 로드 → {risk_type: [(mega_regex, avg_weight), ...]}"""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    risk_info = {}  # {rtype: (mega_pattern_str, total_weight, n_patterns)}
    RTYPES = ["geopolitics", "logistics", "natural", "regulatory", "market", "labor", "technology", "esg", "pandemic"]

    for rtype in RTYPES:
        if rtype not in cfg:
            continue
        keywords = cfg[rtype].get("keywords", [])
        patterns = []
        total_weight = 0
        for kw in keywords:
            pat = kw.get("pattern", "")
            weight = kw.get("weight", 1)
            if pat:
                patterns.append(pat)
                total_weight += weight

        if patterns:
            # 각 패턴을 non-capturing group으로 감싸서 하나의 mega regex 생성
            mega = "|".join(f"(?:{p})" for p in patterns)
            avg_weight = total_weight / len(patterns)
            risk_info[rtype] = {
                "mega_pattern": mega,
                "avg_weight": avg_weight,
                "total_weight": total_weight,
                "n_patterns": len(patterns),
            }
            log.info(f"  {rtype:15s}: {len(patterns):3d} patterns → mega regex len={len(mega):,}")

    return risk_info


def classify_vectorized(df: pd.DataFrame, risk_info: dict, target_mask=None):
    """
    벡터화 분류: pd.Series.str.contains() 사용.
    각 risk_type별 1회 벡터 스캔 → boolean mask → 합산.
    """
    if target_mask is not None:
        idx = df.index[target_mask]
    else:
        idx = df.index

    n = len(idx)
    log.info(f"Classifying {n:,} events (vectorized)...")

    # title 우선, title이 비어있는 행만 url로 대체
    titles = df.loc[idx, "title"].fillna("").astype(str)
    urls = df.loc[idx, "url"].fillna("").astype(str)
    empty_title = (titles.str.len() < 5)
    text_series = titles.copy()
    text_series.loc[empty_title] = urls.loc[empty_title]
    n_url_fallback = empty_title.sum()
    log.info(f"  Text source: {n - n_url_fallback:,} from title, {n_url_fallback:,} from url fallback")

    # 각 risk_type별 벡터 매칭
    matches = {}  # {rtype: boolean Series}
    for rtype, info in risk_info.items():
        t0 = time.time()
        hit = text_series.str.contains(info["mega_pattern"], case=False, na=False, regex=True)
        elapsed = time.time() - t0
        n_hit = hit.sum()
        matches[rtype] = hit
        log.info(f"  {rtype:15s}: {n_hit:>10,} hits ({n_hit/n*100:.2f}%) in {elapsed:.1f}s")

    # risk_types 문자열 조합 + severity 계산
    log.info("Building risk_types and severity...")
    t0 = time.time()

    # DataFrame으로 만들어서 벡터 연산
    match_df = pd.DataFrame(matches, index=idx)

    # risk_types: 매칭된 타입들을 쉼표로 연결
    def row_to_risk_types(row):
        types = [rtype for rtype in risk_info.keys() if row[rtype]]
        return ",".join(types) if types else "other"

    # 벡터화된 severity: 매칭된 타입들의 avg_weight 합산
    severity_raw = pd.Series(0.0, index=idx)
    for rtype, info in risk_info.items():
        severity_raw += match_df[rtype].astype(float) * info["avg_weight"]

    S_MAX = 25.0
    severity = np.minimum(np.log(1 + severity_raw) / np.log(1 + S_MAX), 1.0)
    severity = severity.round(4)

    # any_match mask
    any_match = match_df.any(axis=1)

    # risk_types 조합 — 매칭 안 된 행은 빠르게 'other'
    # 매칭된 행만 문자열 조합 (이게 가장 비용 높지만 매칭 행 수는 적음)
    n_any = any_match.sum()
    log.info(f"  Matched rows: {n_any:,} / {n:,} ({n_any/n*100:.1f}%)")

    risk_types = pd.Series("other", index=idx)
    if n_any > 0:
        matched_idx = match_df.index[any_match]
        risk_types.loc[matched_idx] = match_df.loc[matched_idx].apply(row_to_risk_types, axis=1)

    elapsed = time.time() - t0
    log.info(f"  Combination done in {elapsed:.1f}s")

    return risk_types, severity


def main():
    parser = argparse.ArgumentParser(description="Risk 분류기 (벡터화)")
    parser.add_argument("--input", default="data/processed/risk_events.parquet")
    parser.add_argument("--output", default="data/processed/risk_events.parquet")
    parser.add_argument("--config", default="configs/risk_keywords.yaml")
    parser.add_argument("--all", action="store_true", help="모든 행 재분류 (기존 분류 포함)")
    parser.add_argument("--dry-run", action="store_true", help="결과만 보고 저장 안 함")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("Risk 분류기 (벡터화) v3")
    log.info("=" * 60)

    # 1) 패턴 로드
    risk_info = load_risk_config(args.config)
    log.info(f"Total risk types: {len(risk_info)}")

    # 2) 데이터 로드
    log.info("Loading parquet...")
    t_load = time.time()
    df = pd.read_parquet(args.input)
    log.info(f"Loaded: {len(df):,} rows in {time.time()-t_load:.1f}s")

    # 3) 대상 선택
    if args.all:
        target_mask = None
        log.info("Mode: ALL rows (re-classify everything)")
    else:
        target_mask = df["risk_types"] == "other"
        log.info(f"Mode: 'other' only → {target_mask.sum():,} rows")

    # 4) 벡터화 분류
    t_cls = time.time()
    new_risk_types, new_severity = classify_vectorized(df, risk_info, target_mask)
    log.info(f"Classification total: {time.time()-t_cls:.1f}s")

    # 5) 적용
    if target_mask is not None:
        idx = df.index[target_mask]
    else:
        idx = df.index
    df.loc[idx, "risk_types"] = new_risk_types
    df.loc[idx, "severity"] = new_severity

    # 6) 통계
    log.info("\n=== risk_types distribution ===")
    vc = df["risk_types"].value_counts()
    for rt, cnt in vc.head(25).items():
        log.info(f"  {rt:40s}: {cnt:>10,} ({cnt/len(df)*100:.1f}%)")

    n_classified = (df["risk_types"] != "other").sum()
    n_other = (df["risk_types"] == "other").sum()
    log.info(f"\n  Classified: {n_classified:,} ({n_classified/len(df)*100:.1f}%)")
    log.info(f"  Remaining other: {n_other:,} ({n_other/len(df)*100:.1f}%)")

    if args.dry_run:
        log.info("\n[DRY-RUN] No files written.")
        return

    # 7) 저장 (타입 정규화)
    log.info("Saving...")
    t_save = time.time()

    # entity_scores: 반드시 list[float]
    def _force_float_list(x):
        if isinstance(x, (list, tuple, np.ndarray)):
            return [
                float(v) if not isinstance(v, dict) else float(v.get("score", 0.0))
                for v in x
            ]
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return []
        return [float(x)] if isinstance(x, (int, float)) else []

    df["entity_scores"] = df["entity_scores"].apply(_force_float_list)

    # event_time: tz-naive
    if "event_time" in df.columns:
        df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce", utc=True)
        df["event_time"] = df["event_time"].dt.tz_localize(None)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    log.info(f"Saved: {output_path} ({len(df):,} rows) in {time.time()-t_save:.1f}s")
    log.info(f"\nTotal elapsed: {time.time()-t_cls:.1f}s")


if __name__ == "__main__":
    main()
