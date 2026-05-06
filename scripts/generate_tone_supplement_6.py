"""
Phase 3 보완: 6개 누락 기업 tone_monthly_zscore 생성 후 최종 18개 통합

누락 기업: CALB(3931.HK), EVE(300014.SZ), Gotion(002074.SZ),
           EcoPro BM(247540.KQ), CNGR(300919.SZ), Huayou(603799.SH)

입력:
  data/processed/risk_events_classified_v9.parquet  (tone, finbert_z 포함)
  data/processed/finbert_v2.parquet                 (URL 키, finbert_score)
  data/processed/tone_monthly_zscore.parquet        (기존 12개 → 18개로 교체)

출력:
  data/processed/tone_monthly_zscore_18.parquet     (최종 18개 기업)
  reports/tone_supplement_6_summary.txt
"""
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]

MISSING_6 = ['3931.HK', '300014.SZ', '002074.SZ', '247540.KQ', '300919.SZ', '603799.SH']
FINAL_18 = {
    'TSLA','BMW.DE','VOW3.DE','MBG.DE','7267.T','1211.HK',
    '300750.SZ','373220.KS','3931.HK','300014.SZ','002074.SZ',
    '051910.KS','BAS.DE','247540.KQ','300919.SZ',
    'ALB','GLEN.L','603799.SH'
}

# 레짐 기간 정의 (기존 데이터와 동일)
REGIMES = {
    'regime_covid':        ('2020-01', '2022-12'),
    'regime_ukraine':      ('2022-02', '2099-12'),
    'regime_ira':          ('2022-08', '2099-12'),
    'regime_china_export': ('2023-10', '2099-12'),
}

# reliable: 월 기사 수 10건 이상
RELIABLE_MIN_TONE = 10
# finbert 우선 사용 임계값: 월 FinBERT 건수 >= 5
FB_MIN = 5
# neg/pos shock 임계값
SHOCK_THR = 2.0


def load_v9_missing(missing: list[str]) -> pd.DataFrame:
    """v9에서 누락 6개 기업 이벤트 로드 (tone, finbert_z, source_tier_weight 포함)"""
    log.info("v9 로딩 중 (6개 기업 필터)...")
    cols = ['event_id', 'event_time', 'url', 'company_id',
            'tone', 'finbert_score', 'finbert_z',
            'source_tier_weight', 'regime_covid', 'regime_ukraine',
            'regime_ira', 'regime_china_export']

    pf = pq.ParquetFile(ROOT / 'data/processed/risk_events_classified_v9.parquet')
    batches = []
    for batch in pf.iter_batches(batch_size=300_000, columns=cols):
        b = batch.to_pandas()
        b = b[b['company_id'].isin(missing)]
        if len(b):
            batches.append(b)

    df = pd.concat(batches, ignore_index=True)
    df['event_time'] = pd.to_datetime(df['event_time'], errors='coerce')
    df = df.dropna(subset=['event_time', 'company_id'])
    log.info("  로드 완료: %d건", len(df))
    return df


def load_finbert_v2_urls(urls: set[str]) -> pd.DataFrame:
    """finbert_v2에서 해당 URL의 스코어만 로드"""
    log.info("finbert_v2 로딩 중 (%d URLs)...", len(urls))
    pf = pq.ParquetFile(ROOT / 'data/processed/finbert_v2.parquet')
    batches = []
    for batch in pf.iter_batches(batch_size=500_000, columns=['url', 'finbert_score']):
        b = batch.to_pandas()
        b = b[b['url'].isin(urls)]
        if len(b):
            batches.append(b)
    if not batches:
        log.warning("  finbert_v2에서 매칭 URL 없음 → tone fallback 전량 사용")
        return pd.DataFrame(columns=['url', 'finbert2_score'])
    df = pd.concat(batches, ignore_index=True).rename(columns={'finbert_score': 'finbert2_score'})
    df = df.drop_duplicates('url')
    log.info("  매칭 완료: %d건", len(df))
    return df


def monthly_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """
    기업 × 월별 집계
    - tone_weighted_mean: source_tier_weight 가중 평균 GDELT tone
    - fb_mean: FinBERT 스코어 평균 (v2 우선, 없으면 v1 finbert_score)
    - n_tone, n_fb_total: 기사 수
    """
    df = df.copy()
    df['year_month'] = df['event_time'].dt.to_period('M').dt.to_timestamp()
    df['weight'] = df['source_tier_weight'].fillna(1.0)

    # 가중 tone
    df['tone_w'] = df['tone'].fillna(0) * df['weight']
    df['tone_valid'] = df['tone'].notna().astype(int)

    # finbert: v2 우선, 없으면 v1
    df['fb_score'] = df['finbert2_score'].fillna(df['finbert_score'])

    agg = df.groupby(['company_id', 'year_month']).agg(
        tone_w_sum=('tone_w', 'sum'),
        weight_sum=('weight', 'sum'),
        n_tone=('tone_valid', 'sum'),
        n_fb_total=('fb_score', lambda x: x.notna().sum()),
        fb_score_mean=('fb_score', 'mean'),
    ).reset_index()

    agg['tone_weighted_mean'] = np.where(
        agg['weight_sum'] > 0,
        agg['tone_w_sum'] / agg['weight_sum'],
        np.nan
    )
    agg = agg.drop(columns=['tone_w_sum', 'weight_sum'])
    return agg


def per_company_zscore(series: pd.Series) -> pd.Series:
    """시계열 내 per-company z-score (μ=0, σ=1)"""
    mu = series.mean()
    sigma = series.std(ddof=1)
    if sigma < 1e-10:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mu) / sigma


def build_tone_monthly(agg: pd.DataFrame) -> pd.DataFrame:
    """
    월별 집계 → tone_monthly_zscore 포맷 생성
    """
    rows = []
    for cid, g in agg.groupby('company_id'):
        g = g.sort_values('year_month').copy()

        # tone z-score
        tone_valid = g['n_tone'] >= RELIABLE_MIN_TONE
        g['tone_z'] = np.nan
        if tone_valid.sum() >= 12:
            g.loc[tone_valid, 'tone_z'] = per_company_zscore(
                g.loc[tone_valid, 'tone_weighted_mean']
            ).values

        # finbert z-score (n_fb >= FB_MIN 인 월만)
        fb_valid = g['n_fb_total'] >= FB_MIN
        g['fb_z_combined'] = np.nan
        if fb_valid.sum() >= 12:
            g.loc[fb_valid, 'fb_z_combined'] = per_company_zscore(
                g.loc[fb_valid, 'fb_score_mean']
            ).values

        # z_score 선택: finbert 우선, 부족하면 tone
        g['z_score'] = np.nan
        g['z_source'] = 'v2tone'
        fb_ok = g['fb_z_combined'].notna()
        g.loc[fb_ok, 'z_score'] = g.loc[fb_ok, 'fb_z_combined']
        g.loc[fb_ok, 'z_source'] = 'finbert'
        # tone fallback
        tone_ok = g['tone_z'].notna() & ~fb_ok
        g.loc[tone_ok, 'z_score'] = g.loc[tone_ok, 'tone_z']

        # 레짐 더미
        for col, (start, end) in REGIMES.items():
            s = pd.Period(start, 'M').to_timestamp()
            e = pd.Period(end, 'M').to_timestamp()
            g[col] = (g['year_month'] >= s) & (g['year_month'] <= e)

        # lag / delta / rolling
        g = g.sort_values('year_month').reset_index(drop=True)
        g['z_lag1'] = g['z_score'].shift(1)
        g['z_lag2'] = g['z_score'].shift(2)
        g['z_lag3'] = g['z_score'].shift(3)
        g['delta_z'] = g['z_score'] - g['z_lag1']
        g['rolling_3m'] = g['z_score'].rolling(3, min_periods=2).mean()

        # shock flags
        g['neg_shock'] = (g['z_score'] < -SHOCK_THR) & g['z_score'].notna()
        g['pos_shock'] = (g['z_score'] > SHOCK_THR) & g['z_score'].notna()

        # reliable
        g['reliable'] = g['n_tone'] >= RELIABLE_MIN_TONE

        # n_languages placeholder (다국어 정보 없음 → 1)
        g['n_languages'] = 1.0

        rows.append(g)

    result = pd.concat(rows, ignore_index=True)
    result['year_month'] = result['year_month'].dt.to_period('M').astype(str)
    return result


def main():
    out_path = ROOT / 'data/processed/tone_monthly_zscore_18.parquet'
    report_path = ROOT / 'reports/tone_supplement_6_summary.txt'

    # ── Step 1: 기존 parquet에서 최종 18개 기업 12개 추출 ──
    log.info("기존 tone_monthly_zscore 로딩...")
    existing = pd.read_parquet(ROOT / 'data/processed/tone_monthly_zscore.parquet')
    keep_12 = existing[existing['company_id'].isin(FINAL_18)].copy()
    log.info("  기존 12개 기업: %d행", len(keep_12))

    # ── Step 2: 누락 6개 기업 v9 로드 ──
    v9_miss = load_v9_missing(MISSING_6)

    # ── Step 3: finbert_v2 조인 ──
    urls = set(v9_miss['url'].dropna())
    fb2 = load_finbert_v2_urls(urls)
    if len(fb2):
        v9_miss = v9_miss.merge(fb2, on='url', how='left')
    else:
        v9_miss['finbert2_score'] = np.nan

    # ── Step 4: 월별 집계 ──
    log.info("월별 집계 중...")
    agg = monthly_aggregate(v9_miss)

    # ── Step 5: tone_monthly_zscore 포맷 생성 ──
    log.info("z-score 계산 중...")
    supplement = build_tone_monthly(agg)

    # 컬럼 순서 맞추기 (기존 포맷)
    target_cols = keep_12.columns.tolist()
    for c in target_cols:
        if c not in supplement.columns:
            supplement[c] = np.nan
    supplement = supplement[target_cols]

    # ── Step 6: 통합 ──
    combined = pd.concat([keep_12, supplement], ignore_index=True)
    combined = combined.sort_values(['company_id', 'year_month']).reset_index(drop=True)

    combined.to_parquet(out_path, index=False)
    log.info("저장 완료: %s (%d행, %d기업)", out_path, len(combined), combined['company_id'].nunique())

    # ── 요약 보고 ──
    lines = ["=" * 70,
             "Phase 3 보완 요약: 6개 누락 기업 → 최종 18개 통합",
             "=" * 70, ""]
    for cid in MISSING_6:
        sub = supplement[supplement['company_id'] == cid]
        n_neg = sub['neg_shock'].sum()
        n_pos = sub['pos_shock'].sum()
        n_rel = sub['reliable'].sum()
        src = sub['z_source'].mode().iloc[0] if len(sub) else 'N/A'
        lines.append(f"  {cid:12s}: {len(sub):3d}개월 | reliable={n_rel} "
                     f"| neg_shock={n_neg} pos_shock={n_pos} | z_source={src}")
    lines += ["",
              f"최종 통합: {combined['company_id'].nunique()}개 기업, {len(combined)}행",
              f"전체 neg_shock: {combined['neg_shock'].sum()}",
              f"전체 pos_shock: {combined['pos_shock'].sum()}"]

    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text("\n".join(lines), encoding='utf-8')
    print("\n".join(lines))


if __name__ == "__main__":
    main()
