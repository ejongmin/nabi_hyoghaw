"""
finbert_v2 URL → risk_events_classified_v9 병합으로 v10 생성
=============================================================

상황:
  - finbert_v2.parquet: 3,393,773건 URL 키 FinBERT 스코어 (영어)
  - finbert_v2_multilingual.parquet: 447,696건 (다국어)
  - risk_events_classified_v9.parquet: 11,825,508건
  - 현재 v10: v9 + finbert_lang 컬럼만 있음 (finbert2 미병합)

목표:
  v9.url ←LEFT JOIN→ finbert_v2.url
  → finbert2_score, finbert2_neg, finbert2_pos, finbert2_z_month 추가
  → 출력: risk_events_classified_v10_new.parquet

처리 전략:
  1. finbert_v2 전체를 URL 키 dict으로 메모리 로드 (428MB → ~200MB)
  2. v9를 청크로 읽어 URL 기준 lookout join
  3. finbert2_z_month: 월(YYYYMM) 내 z-score 계산 후 추가
  4. 청크 단위 출력

예상 매칭률: ~30% (v9의 URL 중 finbert_v2에 있는 비율)
"""
from __future__ import annotations
import logging, sys
from pathlib import Path
import numpy as np, pandas as pd
import pyarrow as pa, pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

ROOT  = Path(__file__).resolve().parents[1]
V9    = ROOT/"data/processed/risk_events_classified_v9.parquet"
FB2   = ROOT/"data/processed/finbert_v2.parquet"
FB2M  = ROOT/"data/processed/finbert_v2_multilingual.parquet"
OUT   = ROOT/"data/processed/risk_events_classified_v10_new.parquet"
RPT   = ROOT/"reports/merge_finbert_v2_report.txt"

BATCH = 300_000


def load_finbert_v2_dict():
    """
    finbert_v2 (영어) + multilingual을 URL→score dict로 로드
    영어 우선, 없으면 multilingual로 fallback
    """
    log.info("finbert_v2 영어 로딩 (%s)...", FB2)
    pf = pq.ParquetFile(FB2)
    fb_dict = {}
    n_loaded = 0
    for batch in pf.iter_batches(batch_size=500_000,
                                  columns=["url","finbert_neg","finbert_neu",
                                           "finbert_pos","finbert_score"]):
        df = batch.to_pandas().dropna(subset=["url","finbert_score"])
        for _, r in df.iterrows():
            fb_dict[r["url"]] = dict(
                finbert2_neg  = float(r["finbert_neg"]),
                finbert2_neu  = float(r["finbert_neu"]),
                finbert2_pos  = float(r["finbert_pos"]),
                finbert2_score= float(r["finbert_score"]),
                lang          = "en",
            )
        n_loaded += len(df)
    log.info("  영어 로드: %d건", n_loaded)

    if FB2M.exists():
        log.info("finbert_v2_multilingual 로딩 (%s)...", FB2M)
        pf_m = pq.ParquetFile(FB2M)
        n_multi = 0
        for batch in pf_m.iter_batches(batch_size=200_000,
                                        columns=["url","finbert_neg","finbert_neu",
                                                 "finbert_pos","finbert_score","lang_ft"]):
            df = batch.to_pandas().dropna(subset=["url","finbert_score"])
            for _, r in df.iterrows():
                if r["url"] not in fb_dict:  # 영어 없는 경우만 추가
                    fb_dict[r["url"]] = dict(
                        finbert2_neg  = float(r["finbert_neg"]),
                        finbert2_neu  = float(r["finbert_neu"]),
                        finbert2_pos  = float(r["finbert_pos"]),
                        finbert2_score= float(r["finbert_score"]),
                        lang          = str(r.get("lang_ft","multi")),
                    )
                    n_multi += 1
        log.info("  다국어 추가: %d건 (영어 없는 URL만)", n_multi)

    log.info("총 URL dict: %d건", len(fb_dict))
    return fb_dict


def compute_monthly_z(scores: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """
    월(YYYYMM) 파티션별 z-score 계산
    dates: YYYYMMDD 형식 문자열 배열
    """
    result = np.full(len(scores), np.nan)
    months = np.array([str(d)[:6] if isinstance(d,str) else "" for d in dates])
    for m in np.unique(months):
        if not m: continue
        idx = months == m
        s = scores[idx]
        valid = ~np.isnan(s)
        if valid.sum() < 5: continue
        mu = s[valid].mean(); sigma = s[valid].std(ddof=1)
        if sigma < 1e-10: continue
        result[idx] = np.where(valid, (s - mu) / sigma, np.nan)
    return result


def main():
    log.info("finbert_v2 → v10 병합 시작")
    fb_dict = load_finbert_v2_dict()

    pf9 = pq.ParquetFile(V9)
    schema9 = pq.read_schema(V9)
    total_rows = pf9.metadata.num_rows

    # 출력 스키마: v9 컬럼 + finbert2_* 추가
    new_fields = [
        pa.field("finbert2_neg",   pa.float32()),
        pa.field("finbert2_neu",   pa.float32()),
        pa.field("finbert2_pos",   pa.float32()),
        pa.field("finbert2_score", pa.float32()),
        pa.field("finbert2_lang",  pa.string()),
        pa.field("finbert2_z_month", pa.float32()),
    ]
    out_schema = pa.schema(schema9.names + [f.name for f in new_fields],
                            metadata=schema9.metadata)

    writer = None
    n_written = 0; n_matched = 0; batch_num = 0

    # 월별 z 계산을 위해 전체를 처리 후 재계산 (메모리 절약 위해 2패스)
    # 1패스: URL 조인 + score 추가
    # 2패스: monthly z 계산
    # → 간단화: 청크별 z 계산 (정밀도 약간 손실 허용)

    log.info("1패스: URL 조인 시작 (총 %d행)...", total_rows)

    for batch in pf9.iter_batches(batch_size=BATCH):
        df = batch.to_pandas()

        # URL 기준 finbert2 조인
        fb2_neg   = np.full(len(df), np.nan, dtype=np.float32)
        fb2_neu   = np.full(len(df), np.nan, dtype=np.float32)
        fb2_pos   = np.full(len(df), np.nan, dtype=np.float32)
        fb2_score = np.full(len(df), np.nan, dtype=np.float32)
        fb2_lang  = np.full(len(df), "", dtype=object)

        matched_idx = []
        for i, url in enumerate(df["url"]):
            info = fb_dict.get(url)
            if info:
                fb2_neg[i]   = info["finbert2_neg"]
                fb2_neu[i]   = info["finbert2_neu"]
                fb2_pos[i]   = info["finbert2_pos"]
                fb2_score[i] = info["finbert2_score"]
                fb2_lang[i]  = info["lang"]
                matched_idx.append(i)

        n_matched += len(matched_idx)

        # monthly z (배치 내 기준 — 전체 기준보다 약간 부정확하나 실용적)
        dates = df["event_time"].astype(str).values
        fb2_z = compute_monthly_z(fb2_score, dates)

        # 컬럼 추가
        df["finbert2_neg"]    = fb2_neg
        df["finbert2_neu"]    = fb2_neu
        df["finbert2_pos"]    = fb2_pos
        df["finbert2_score"]  = fb2_score
        df["finbert2_lang"]   = fb2_lang
        df["finbert2_z_month"]= fb2_z.astype(np.float32)

        # Parquet 쓰기
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(OUT, table.schema,
                                      compression="snappy",
                                      row_group_size=500_000)
        writer.write_table(table)

        n_written += len(df)
        batch_num += 1

        if batch_num % 10 == 0:
            log.info("  진행: %d/%d행  매칭 %d건 (%.1f%%)",
                     n_written, total_rows, n_matched,
                     100*n_matched/max(1,n_written))

    if writer:
        writer.close()

    match_rate = n_matched / max(1, n_written)
    log.info("병합 완료: %d행, 매칭률=%.1f%%", n_written, match_rate*100)
    log.info("출력: %s", OUT)

    # 보고서
    lines = ["="*60, "finbert_v2 URL 병합 결과", "="*60, "",
             f"입력 (v9):     {total_rows:,}행",
             f"출력 (v10_new):{n_written:,}행",
             f"URL 매칭:      {n_matched:,}건  ({match_rate:.1%})",
             f"finbert_v2 dict 크기: {len(fb_dict):,}건",
             "",
             "추가된 컬럼:",
             "  finbert2_score : URL 기반 FinBERT 스코어 (영어 우선, 없으면 다국어)",
             "  finbert2_z_month: 월 내 z-score (감성 이상값 탐지용)",
             "  finbert2_lang  : 사용된 언어 모델 (en/ko/zh/multi 등)",
             "",
             "다음 단계:",
             "  1. tone_monthly_zscore 재생성 → finbert2_score 기반",
             "  2. Phase 4 CAR 재실행 → 더 정확한 감성 신호",
             "="*60]
    RPT.parent.mkdir(exist_ok=True)
    RPT.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
