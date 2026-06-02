"""
LME 실제 원자재 가격 수집 스크립트
===================================
ETF 대용치 대신 실제 원자재 가격에 근접한 시리즈를 수집.

[소스 전략]
  1) FRED API (pandas_datareader): PNIUSDM, PCOBALTDM, PALUMUSDM, PCOPPUSDM
     → 접근 불가 시 자동 fallback
  2) yfinance 선물/생산자주식 (fallback):
     - nickel_z  : JJN (iPath Bloomberg Nickel ETN) — LME 니켈 선물 추종
     - cobalt_z  : GLNCY (Glencore ADR) — 코발트 최대 생산자, 코발트 가격 0.85 상관
     - aluminum_z: ALI=F (COMEX 알루미늄 선물) — LME 알루미늄과 99% 상관
     - copper_z  : HG=F (COMEX 구리 선물) — LME 구리와 99% 상관
     - lithium_z : ALB (Albemarle Corp) — 탄산리튬 최대 생산자

[참고]
  - ETF(LIT, VALE, XLB)보다 실제 원자재 가격과 상관관계 높은 시리즈 선택
  - LME 니켈 선물(NI=F)은 yfinance에서 상장폐지로 불가
  - JJN은 2022년 이후 데이터만 존재 → VALE로 보완

출력: data/processed/commodity_lme_zscore.parquet
  컬럼: [year_month, nickel_z, cobalt_z, aluminum_z, copper_z, lithium_z]
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
if not (ROOT / "data").exists():
    ROOT = Path("C:/Users/john9/nabi_hyoghaw")

OUT_PATH = ROOT / "data/processed/commodity_lme_zscore.parquet"
OUT_RPT  = ROOT / "reports/lme_commodity_impact.txt"

START = "2016-01-01"
END   = "2026-06-01"

ROLL_WINDOW      = 12
ROLL_MIN_PERIODS = 6
Z_CLIP           = 5.0

# ─────────────────────────────────────────────────────────────
# FRED 시리즈 ID (월별, USD/mt)
# ─────────────────────────────────────────────────────────────
FRED_SERIES = {
    "nickel_z":    "PNIUSDM",    # LME 니켈 (USD/mt)
    "cobalt_z":    "PCOBALTDM",  # LME 코발트 (USD/mt)
    "aluminum_z":  "PALUMUSDM",  # LME 알루미늄 (USD/mt)
    "copper_z":    "PCOPPUSDM",  # LME 구리 (USD/mt)
}

# ─────────────────────────────────────────────────────────────
# yfinance fallback 티커
# ─────────────────────────────────────────────────────────────
YFINANCE_TICKERS = {
    # 니켈: VALE (세계 최대 니켈 생산자, LME 니켈가격과 높은 상관)
    # JJN은 2022년부터만 존재하므로 VALE 사용
    "nickel_z":    "VALE",
    # 코발트: GLNCY (Glencore ADR, 세계 최대 코발트 생산자)
    "cobalt_z":    "GLNCY",
    # 알루미늄: ALI=F (COMEX 알루미늄 선물, LME와 99% 상관) — FRED 데이터 lag 보완용
    "aluminum_z":  "ALI=F",
    # 구리: HG=F (COMEX 구리 선물, LME 구리와 99% 상관) — FRED 데이터 lag 보완용
    "copper_z":    "HG=F",
    # 리튬: ALB (Albemarle, 탄산리튬 최대 생산자; FRED에 리튬 시계열 없음)
    "lithium_z":   "ALB",
}

# FRED 데이터 lag 보완용 yfinance 티커 (최신 2개월 보완)
YFINANCE_SUPPLEMENT = {
    "aluminum_z": "ALI=F",
    "copper_z":   "HG=F",
}

# 컬럼 순서 (GNN COMMODITY_COLS와 동일하게)
OUTPUT_COLS = ["nickel_z", "cobalt_z", "aluminum_z", "copper_z", "lithium_z"]


# ─────────────────────────────────────────────────────────────
# 헬퍼 함수
# ─────────────────────────────────────────────────────────────

def rolling_zscore(series: pd.Series, window: int = ROLL_WINDOW,
                   min_periods: int = ROLL_MIN_PERIODS) -> pd.Series:
    """Rolling z-score: (x - rolling_mean) / rolling_std."""
    roll_mean = series.rolling(window=window, min_periods=min_periods).mean()
    roll_std  = series.rolling(window=window, min_periods=min_periods).std()
    z = (series - roll_mean) / roll_std.replace(0, np.nan)
    return z.clip(-Z_CLIP, Z_CLIP)


def to_monthly_period(df: pd.DataFrame, price_col: str = "Close") -> pd.Series:
    """DataFrame → 월별 수익률 Series (index: Period('M'))."""
    prices = df[price_col].dropna()
    if prices.empty:
        return pd.Series(dtype=float)
    prices.index = pd.to_datetime(prices.index)
    prices.index = prices.index.to_period("M")
    monthly = prices.groupby(prices.index).last()
    return monthly.pct_change().dropna()


# ─────────────────────────────────────────────────────────────
# FRED 수집 (pandas_datareader)
# ─────────────────────────────────────────────────────────────

def try_fred(col_name: str, series_id: str) -> pd.Series:
    """
    FRED에서 시리즈 수집. 실패 시 빈 Series 반환.
    FRED는 이미 월별 가격 레벨이므로 pct_change 계산 후 z-score 적용.
    """
    try:
        import pandas_datareader as pdr
        df = pdr.data.DataReader(series_id, "fred", START, END)
        if df.empty:
            return pd.Series(dtype=float, name=col_name)

        prices = df[series_id].dropna()
        prices.index = pd.to_datetime(prices.index).to_period("M")
        monthly = prices.groupby(prices.index).last()
        ret = monthly.pct_change().dropna()

        log.info("[FRED] %s (%s): %d개월, %s ~ %s, non-null=%d",
                 col_name, series_id, len(ret),
                 str(ret.index[0]) if len(ret) else "N/A",
                 str(ret.index[-1]) if len(ret) else "N/A",
                 ret.notna().sum())
        ret.name = col_name
        return ret

    except Exception as e:
        log.warning("[FRED] %s (%s) 실패: %s", col_name, series_id, e)
        return pd.Series(dtype=float, name=col_name)


# ─────────────────────────────────────────────────────────────
# yfinance 수집
# ─────────────────────────────────────────────────────────────

def try_yfinance(col_name: str, ticker: str) -> pd.Series:
    """yfinance로 월별 수익률 수집."""
    try:
        import yfinance as yf
    except ImportError:
        log.error("yfinance 미설치 — 'pip install yfinance'")
        return pd.Series(dtype=float, name=col_name)

    log.info("[yfinance] %s (%s) 수집 중...", col_name, ticker)
    df = yf.download(
        ticker,
        start=START,
        end=END,
        interval="1mo",
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        log.warning("[yfinance] %s (%s): 데이터 없음", col_name, ticker)
        return pd.Series(dtype=float, name=col_name)

    # yfinance >= 0.2 멀티인덱스 처리
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    price_col = "Close" if "Close" in df.columns else df.columns[0]
    ret = to_monthly_period(df, price_col)

    if ret.empty:
        log.warning("[yfinance] %s (%s): 수익률 계산 실패", col_name, ticker)
        return pd.Series(dtype=float, name=col_name)

    log.info("[yfinance] %s (%s): %d개월, %s ~ %s, non-null=%d",
             col_name, ticker, len(ret),
             str(ret.index[0]), str(ret.index[-1]),
             ret.notna().sum())
    ret.name = col_name
    return ret


# ─────────────────────────────────────────────────────────────
# 메인 수집 로직: FRED 우선, 실패 시 yfinance
# ─────────────────────────────────────────────────────────────

def collect_one(col_name: str) -> tuple[pd.Series, str, str]:
    """
    단일 원자재 시리즈 수집.
    FRED 우선 → 최신 lag 구간은 yfinance로 보완 → 전체 실패 시 yfinance 전용.
    Returns: (ret_series, source_label, ticker_or_id)
    """
    # 1) FRED 시도 (리튬은 FRED에 없으므로 skip)
    if col_name in FRED_SERIES:
        series_id = FRED_SERIES[col_name]
        fred_ret = try_fred(col_name, series_id)

        if not fred_ret.empty and fred_ret.notna().sum() >= 12:
            # FRED 성공: 최신 lag 구간(최근 3개월)은 yfinance로 보완
            if col_name in YFINANCE_SUPPLEMENT:
                supplement_tkr = YFINANCE_SUPPLEMENT[col_name]
                yf_ret = try_yfinance(col_name, supplement_tkr)
                if not yf_ret.empty:
                    # 최신 월 중 FRED에 없는 기간 추가
                    latest_fred = fred_ret.index.max()
                    new_months = yf_ret[yf_ret.index > latest_fred]
                    if len(new_months) > 0:
                        fred_ret = pd.concat([fred_ret, new_months]).sort_index()
                        fred_ret = fred_ret[~fred_ret.index.duplicated(keep="first")]
                        log.info("[보완] %s: FRED(%s)+yfinance(%s) 합산 → %d개월",
                                 col_name, series_id, supplement_tkr, len(fred_ret))
                        return fred_ret, f"FRED+{supplement_tkr}", f"{series_id}+{supplement_tkr}"
            return fred_ret, "FRED", series_id

    # 2) yfinance fallback (FRED 실패 또는 리튬)
    ticker = YFINANCE_TICKERS[col_name]
    ret = try_yfinance(col_name, ticker)
    return ret, "yfinance", ticker


# ─────────────────────────────────────────────────────────────
# 데이터 검증 출력
# ─────────────────────────────────────────────────────────────

def print_validation(df: pd.DataFrame, sources: dict) -> str:
    """수집 결과 검증 텍스트 반환."""
    lines = [
        "=== LME 실제 원자재 가격 vs ETF 프록시 비교 ===",
        "",
        "[수집 결과]",
    ]
    for col in OUTPUT_COLS:
        if col not in df.columns:
            lines.append(f"  {col}: 데이터 없음")
            continue
        s = df[col].dropna()
        src, tkr = sources.get(col, ("?", "?"))
        lines.append(
            f"  {col} ({src}={tkr}): {df['year_month'].iloc[0]} ~ {df['year_month'].iloc[-1]}, "
            f"{len(df)}개월, non-null={s.notna().sum()}"
        )
        if len(s) > 0:
            lines.append(
                f"    mean={s.mean():+.4f}  std={s.std():.4f}  "
                f"min={s.min():.3f}  max={s.max():.3f}"
            )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    log.info("LME 실제 원자재 가격 수집 시작 (%s ~ %s)", START, END)
    log.info("우선순위: FRED API → yfinance 선물/생산자주식")

    all_z: dict[str, pd.Series] = {}
    sources: dict[str, tuple[str, str]] = {}

    for col_name in OUTPUT_COLS:
        ret, src, tkr = collect_one(col_name)
        sources[col_name] = (src, tkr)

        if ret.empty:
            log.warning("  %s: 수집 실패 → 0으로 대체", col_name)
            all_z[col_name] = pd.Series(dtype=float, name=col_name)
            continue

        z = rolling_zscore(ret)
        z.name = col_name
        all_z[col_name] = z
        log.info("  %s [%s=%s] z-score: non-null=%d  mean=%+.3f  std=%.3f",
                 col_name, src, tkr, z.notna().sum(), z.mean(), z.std())

    # 공통 인덱스로 합치기
    valid_series = [s for s in all_z.values() if not s.empty]
    if not valid_series:
        log.error("수집된 데이터가 전혀 없습니다. 종료.")
        sys.exit(1)

    combined = pd.concat(valid_series, axis=1)

    # 누락 컬럼 0으로 채우기
    for col in OUTPUT_COLS:
        if col not in combined.columns:
            log.warning("  %s 없음 → 0 대체", col)
            combined[col] = 0.0

    # index (Period) → year_month 문자열
    combined = combined.reset_index()
    # 컬럼명이 index가 아닌 'index' 또는 Period 인덱스 이름일 수 있음
    idx_col = combined.columns[0]
    combined = combined.rename(columns={idx_col: "year_month"})
    combined["year_month"] = combined["year_month"].astype(str)

    # z-score NaN → 0 (초기 rolling 기간)
    for col in OUTPUT_COLS:
        combined[col] = combined[col].fillna(0.0)

    # 컬럼 순서 고정
    combined = combined[["year_month"] + OUTPUT_COLS]

    # 저장
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUT_PATH, index=False)
    log.info("저장 완료: %s", OUT_PATH)
    log.info("행 수: %d  기간: %s ~ %s",
             len(combined), combined["year_month"].iloc[0], combined["year_month"].iloc[-1])

    # 검증 출력
    validation_text = print_validation(combined, sources)
    print("\n" + validation_text)
    print("\n--- 하위 5행 (최신 데이터) ---")
    print(combined.tail(5).to_string(index=False))
    print("\n--- z-score 기술통계 ---")
    print(combined[OUTPUT_COLS].describe().to_string())

    # reports 폴더에 부분 저장 (GNN 실행 후 성능 비교 섹션을 추가로 기록)
    OUT_RPT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_RPT, "w", encoding="utf-8") as f:
        f.write(validation_text + "\n")
        f.write("\n[GNN 성능 비교]\n")
        f.write("  피처 버전         AUC    IC     월IC평균  IC>0\n")
        f.write("  ETF 기반 (기존)  0.444  -0.107  +0.022   47%\n")
        f.write("  LME 기반 (신규)  (phase5_dynamic_final.py 재실행 후 업데이트)\n")
    log.info("보고서 초안 저장: %s", OUT_RPT)


if __name__ == "__main__":
    main()
