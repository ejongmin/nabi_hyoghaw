"""
Risk Events 집계 모듈
=====================
400만건 raw risk_events를 (날짜, 기업) 단위로 집계하여
compute_exposure / event_study에 적합한 크기로 줄입니다.

집계 전략:
  - 동일 날짜 + 동일 entity_ids 조합 → 1개 이벤트로 합산
  - severity = 해당 그룹의 max severity (가장 심각한 기사 대표)
  - tone = 해당 그룹의 mean tone
  - risk_types = 해당 그룹 전체의 union
  - url = 가장 severe한 기사의 URL (대표 기사)

기대 결과: 400만건 → ~5만~15만건 (90~97% 축소)
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

log = logging.getLogger("aggregate_events")


def _normalize_entity_key(entity_ids) -> str:
    """entity_ids (list/ndarray) → sorted, semicolon-joined string key."""
    if entity_ids is None:
        return ""
    try:
        ids = sorted(set(str(x) for x in entity_ids if x))
        return ";".join(ids)
    except (TypeError, ValueError):
        return ""


def aggregate_risk_events(
    risk_path: Path,
    out_path: Path,
    force: bool = False,
) -> pd.DataFrame:
    """
    Raw risk_events → 날짜+기업 집계된 risk_events.

    메모리 효율: PyArrow 배치로 필요 컬럼만 로드 → 집계.
    """
    if out_path.exists() and not force:
        log.info(f"Cached aggregated risk_events: {out_path}")
        return pd.read_parquet(out_path)

    log.info(f"Loading risk_events for aggregation: {risk_path}")

    # 필요 컬럼만 로드
    needed = ["event_id", "event_time", "url", "risk_types",
              "severity", "entity_ids", "tone", "country_ids"]
    pf = pq.ParquetFile(risk_path)
    available = set(pf.schema_arrow.names)
    use_cols = [c for c in needed if c in available]

    chunks = []
    for batch in pf.iter_batches(batch_size=500_000, columns=use_cols):
        chunks.append(batch.to_pandas())
    df = pd.concat(chunks, ignore_index=True)
    del chunks
    log.info(f"Loaded: {len(df):,} raw events")

    # entity_key 생성 (groupby용)
    df["entity_key"] = df["entity_ids"].apply(_normalize_entity_key)

    # 날짜를 date로 통일 (시간 제거)
    df["event_date"] = pd.to_datetime(df["event_time"], errors="coerce").dt.date

    # entity 없는 행 (Type B risk_only) 제거 — exposure/event_study에서 사용 불가
    n_before = len(df)
    df = df[df["entity_key"] != ""].copy()
    log.info(f"Filtered Type B (no entity): {n_before:,} → {len(df):,}")

    # (date, entity_key) 그룹별 집계
    log.info("Aggregating by (date, entity_key)...")

    def _agg_group(g):
        """그룹 내 가장 severe한 기사를 대표로, severity/tone은 통계값."""
        idx_max = g["severity"].idxmax()
        row = g.loc[idx_max]

        # risk_types union
        all_types = set()
        for rt in g["risk_types"].dropna():
            for t in str(rt).split(","):
                t = t.strip()
                if t:
                    all_types.add(t)

        return pd.Series({
            "event_id": row["event_id"],  # 대표 기사의 ID
            "event_time": row["event_time"],
            "url": row["url"],
            "severity": g["severity"].max(),
            "severity_mean": g["severity"].mean(),
            "tone": g["tone"].mean(),
            "risk_types": ",".join(sorted(all_types)) if all_types else "other",
            "entity_ids": row["entity_ids"],
            "country_ids": row.get("country_ids", ""),
            "n_articles": len(g),  # 집계된 기사 수
        })

    agg = df.groupby(["event_date", "entity_key"], sort=False).apply(_agg_group)
    agg = agg.reset_index(drop=True)

    # 파이프라인 호환 컬럼 추가
    agg["title"] = ""
    agg["finbert"] = None
    agg["entity_scores"] = agg["entity_ids"].apply(
        lambda ids: [{"company_id": c, "score": 100, "alias": "bq_agg"} for c in ids]
        if ids is not None and len(ids) > 0 else []
    )

    log.info(f"✅ Aggregated: {len(df):,} → {len(agg):,} events "
             f"({100*(1-len(agg)/len(df)):.1f}% reduction)")
    log.info(f"   Articles per event: mean={agg['n_articles'].mean():.1f}, "
             f"max={agg['n_articles'].max()}")

    # 저장
    out_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(out_path, index=False)
    log.info(f"✅ Saved: {out_path}")

    return agg
