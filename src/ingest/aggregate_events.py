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


def _normalize_company_bias(agg: pd.DataFrame) -> pd.DataFrame:
    """
    기업 편향 보정:

    문제: BMW 79만건 vs Nuode 8건 → 절대 severity가 대기업에 편중

    해결 1 — Z-score 정규화 (기업별 "평소 대비 이상치")
      BMW 평소 하루 severity 3.5 → 오늘 4.5 = z=1.5 (약간 이상)
      Nuode 평소 하루 severity 2.0 → 오늘 4.5 = z=3.0 (매우 이상)
      → 같은 severity 4.5여도 Nuode가 더 강한 시그널

    해결 2 — Inverse Frequency Weighting (희소 기업 시그널 증폭)
      BMW 이벤트 1건 가중치: 1/log(총이벤트수) → 낮음
      Nuode 이벤트 1건 가중치: 1/log(총이벤트수) → 높음
      → TF-IDF 원리: 희소할수록 정보 가치 높음

    출력: severity_zscore, severity_ifw 컬럼 추가 (원본 severity 보존)
    """
    log.info("  Normalizing company bias (z-score + inverse frequency)...")

    # entity_key에서 첫 번째 기업 ID 추출 (primary entity)
    def _get_primary(x):
        try:
            if x is not None and hasattr(x, '__len__') and len(x) > 0:
                return str(x[0])
        except (TypeError, IndexError, KeyError):
            pass
        return ""
    agg["_primary_entity"] = agg["entity_ids"].apply(_get_primary)

    # ── Z-score 정규화 ──
    # 기업별 severity 통계 (평소 baseline)
    company_stats = agg.groupby("_primary_entity")["severity"].agg(["mean", "std", "count"])
    company_stats["std"] = company_stats["std"].replace(0, 1.0)  # std=0 방지
    company_stats.columns = ["sev_mean", "sev_std", "event_count"]

    agg = agg.merge(
        company_stats[["sev_mean", "sev_std", "event_count"]],
        left_on="_primary_entity",
        right_index=True,
        how="left",
    )

    # z-score: (severity - company_mean) / company_std
    agg["severity_zscore"] = (agg["severity"] - agg["sev_mean"]) / agg["sev_std"]
    # clip to reasonable range
    agg["severity_zscore"] = agg["severity_zscore"].clip(-3.0, 5.0)

    # ── Inverse Frequency Weighting ──
    # log(total_events / company_events) → 희소 기업일수록 높은 가중치
    total_events = len(agg)
    agg["ifw"] = np.log1p(total_events / agg["event_count"].clip(lower=1))
    # normalize to [0, 1] range
    ifw_max = agg["ifw"].max()
    if ifw_max > 0:
        agg["ifw"] = agg["ifw"] / ifw_max

    # severity_ifw = severity * ifw (희소 기업의 severity 증폭)
    agg["severity_ifw"] = agg["severity"] * agg["ifw"]
    # re-scale to 1.0~5.0
    sev_min, sev_max = agg["severity_ifw"].min(), agg["severity_ifw"].max()
    if sev_max > sev_min:
        agg["severity_ifw"] = 1.0 + 4.0 * (agg["severity_ifw"] - sev_min) / (sev_max - sev_min)
    agg["severity_ifw"] = agg["severity_ifw"].round(3)

    # ── Combined score (z-score와 ifw를 blend) ──
    # severity_adjusted = 0.5 * severity_zscore_rescaled + 0.5 * severity_ifw
    zscore_rescaled = 1.0 + 4.0 * (agg["severity_zscore"] - agg["severity_zscore"].min()) / \
                      max(agg["severity_zscore"].max() - agg["severity_zscore"].min(), 1e-6)
    agg["severity_adjusted"] = (0.5 * zscore_rescaled + 0.5 * agg["severity_ifw"]).round(3)
    agg["severity_adjusted"] = agg["severity_adjusted"].clip(1.0, 5.0)

    # 로깅
    top5 = company_stats.nlargest(5, "event_count")
    bot5 = company_stats[company_stats["event_count"] > 0].nsmallest(5, "event_count")
    log.info(f"    Top-5 by events: {dict(zip(top5.index, top5['event_count']))}")
    log.info(f"    Bottom-5 by events: {dict(zip(bot5.index, bot5['event_count']))}")
    log.info(f"    IFW range: [{agg['ifw'].min():.3f}, {agg['ifw'].max():.3f}]")
    log.info(f"    severity_adjusted range: [{agg['severity_adjusted'].min():.3f}, {agg['severity_adjusted'].max():.3f}]")

    # 임시 컬럼 정리
    agg.drop(columns=["_primary_entity", "sev_mean", "sev_std", "event_count", "ifw"], inplace=True)

    log.info("  ✅ Bias normalization complete (added: severity_zscore, severity_ifw, severity_adjusted)")
    return agg


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

    # ── "other"-only 이벤트 severity 감쇠 ──
    # 구체적 risk_type(geopolitics, logistics, natural 등)이 없는 "other" 이벤트는
    # 노이즈일 가능성이 높으므로 severity를 50% 감쇠 + severity_threshold 미만 제거
    SEVERITY_THRESHOLD = 2.5  # 이 미만의 other-only 이벤트 제거
    OTHER_SEVERITY_DISCOUNT = 0.5  # other-only severity 감쇠 비율

    is_other_only = agg["risk_types"].isin(["other", ""])
    n_other = is_other_only.sum()

    # other-only severity 감쇠
    agg.loc[is_other_only, "severity"] = agg.loc[is_other_only, "severity"] * OTHER_SEVERITY_DISCOUNT

    # 감쇠 후 threshold 미만 제거
    n_before_filter = len(agg)
    agg = agg[~(is_other_only & (agg["severity"] < SEVERITY_THRESHOLD))].copy()
    n_removed = n_before_filter - len(agg)

    log.info(f"  'other'-only events: {n_other:,} / {n_before_filter:,} "
             f"({100*n_other/n_before_filter:.1f}%)")
    log.info(f"  Removed low-severity 'other': {n_removed:,} events")

    log.info(f"✅ Aggregated: {len(df):,} → {len(agg):,} events "
             f"({100*(1-len(agg)/len(df)):.1f}% reduction)")
    log.info(f"   Articles per event: mean={agg['n_articles'].mean():.1f}, "
             f"max={agg['n_articles'].max()}")

    # ── 기업 편향 보정: z-score 정규화 + inverse frequency weighting ──
    agg = _normalize_company_bias(agg)

    # 저장
    out_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(out_path, index=False)
    log.info(f"✅ Saved: {out_path}")

    return agg
