"""
원자재 가격 수집 스크립트
========================
yfinance로 월별 원자재 관련 ETF/주식 가격을 수집하고,
rolling z-score를 계산해 commodity_monthly_zscore.parquet로 저장.

티커:
  LIT    — Global X Lithium & Battery Tech ETF (리튬)
  VALE   — Vale SA (니켈/철광석 대형 생산자)
  XLB    — Materials Select Sector SPDR ETF (소재 섹터)
  GLEN.L — Glencore (코발트 최대 생산자)

출력: data/processed/commodity_monthly_zscore.parquet
  컬럼: year_month, lithium_z, nickel_z, materials_z, cobalt_z
"""
from __future__ import annotations
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data/processed/commodity_monthly_zscore.parquet"

TICKERS = {
    "lithium_z":    "LIT",     # Global X Lithium & Battery Tech ETF
    "nickel_z":     "VALE",    # Vale SA (니켈/철광석 대형 생산자)
    "materials_z":  "XLB",     # Materials Select Sector SPDR ETF
    "cobalt_z":     "GLEN.L",  # Glencore (코발트 최대 생산자)
}

START = "2017-01-01"
END   = "2026-06-01"
ROLL_WINDOW   = 12
ROLL_MIN_PERIODS = 6


def download_monthly_returns(col_name: str, ticker: str) -> pd.Series:
    """티커 월별 수익률 계산 후 Series로 반환 (index = Period('M'))."""
    try:
        import yfinance as yf
    except ImportError:
        log.error("yfinance 미설치 — 'pip install yfinance' 실행 후 재시도")
        sys.exit(1)

    log.info("  다운로드: %s (%s)", col_name, ticker)
    df = yf.download(
        ticker,
        start=START,
        end=END,
        interval="1mo",
        auto_adjust=True,   # Adj Close 자동 적용
        progress=False,
    )
    if df.empty:
        log.warning("  %s (%s): 데이터 없음 → NaN 반환", col_name, ticker)
        return pd.Series(dtype=float, name=col_name)

    # yfinance ≥0.2 멀티인덱스 컬럼 처리
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Close 컬럼 선택 (auto_adjust=True이면 Close = Adj Close)
    price_col = "Close" if "Close" in df.columns else df.columns[0]
    prices = df[price_col].dropna()

    if prices.empty:
        log.warning("  %s (%s): 가격 데이터 비어있음", col_name, ticker)
        return pd.Series(dtype=float, name=col_name)

    # 월별 수익률
    prices.index = pd.to_datetime(prices.index)
    prices.index = prices.index.to_period("M")
    # 월 내 마지막 가격 (일별 → 월별이면 이미 월별이지만 중복 제거)
    monthly = prices.groupby(prices.index).last()
    ret = monthly.pct_change().dropna()

    log.info("  %s: %d개월 수익률 수집 (%s ~ %s)",
             col_name, len(ret),
             str(ret.index[0]) if len(ret) else "N/A",
             str(ret.index[-1]) if len(ret) else "N/A")
    ret.name = col_name
    return ret


def rolling_zscore(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    """전체 기간 rolling z-score (기업 전체 기간 기준)."""
    roll_mean = series.rolling(window=window, min_periods=min_periods).mean()
    roll_std  = series.rolling(window=window, min_periods=min_periods).std()
    z = (series - roll_mean) / roll_std.replace(0, np.nan)
    return z


def main():
    log.info("원자재 가격 수집 시작 (%s ~ %s)", START, END)

    all_z: dict[str, pd.Series] = {}
    for col_name, ticker in TICKERS.items():
        ret = download_monthly_returns(col_name, ticker)
        if ret.empty:
            all_z[col_name] = pd.Series(dtype=float, name=col_name)
            continue
        z = rolling_zscore(ret, window=ROLL_WINDOW, min_periods=ROLL_MIN_PERIODS)
        z.name = col_name
        all_z[col_name] = z
        log.info("  z-score 완료: %s  non-null=%d  mean=%.3f  std=%.3f",
                 col_name, z.notna().sum(), z.mean(), z.std())

    # 모든 시리즈를 공통 인덱스로 합치기
    valid = [s for s in all_z.values() if not s.empty]
    if not valid:
        log.error("수집된 데이터가 없습니다. 종료.")
        sys.exit(1)

    combined = pd.concat(valid, axis=1)
    # 없는 컬럼 0으로 채우기
    for col_name in TICKERS:
        if col_name not in combined.columns:
            log.warning("  %s 데이터 없음 → 0으로 대체", col_name)
            combined[col_name] = 0.0

    # index (Period) → year_month 문자열 컬럼
    combined = combined.reset_index()
    combined.rename(columns={"index": "year_month"}, inplace=True)
    if "year_month" not in combined.columns:
        # Period index 이름이 다를 수 있음
        combined.columns = ["year_month"] + [c for c in combined.columns if c != combined.columns[0]]

    combined["year_month"] = combined["year_month"].astype(str)

    # z-score NaN → 0 (초기 roll 기간)
    for col_name in TICKERS:
        combined[col_name] = combined[col_name].fillna(0.0)

    # 컬럼 순서 고정
    out_cols = ["year_month", "lithium_z", "nickel_z", "materials_z", "cobalt_z"]
    combined = combined[out_cols]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUT_PATH, index=False)

    log.info("저장 완료: %s", OUT_PATH)
    log.info("행 수: %d  컬럼: %s", len(combined), list(combined.columns))
    log.info("기간: %s ~ %s", combined["year_month"].iloc[0], combined["year_month"].iloc[-1])
    log.info("non-null 비율:\n%s", combined.drop(columns="year_month").notna().mean().to_string())

    # 미리보기
    print("\n--- 수집 결과 미리보기 (상위 5행) ---")
    print(combined.head().to_string(index=False))
    print("\n--- 수집 결과 미리보기 (하위 5행) ---")
    print(combined.tail().to_string(index=False))


if __name__ == "__main__":
    main()
