#!/usr/bin/env python3
"""
LICY + SQM 두 종목 일별 시세를 prices_daily.parquet 에 추가하는 로컬 실행 스크립트.

[배경]
- universe_final = 46사 (2026-04-08 SQM 추가)
- prices_daily.parquet = 44사 (LICY, SQM 누락)
- VM은 yfinance egress가 막혀 있어 사용자 로컬 환경에서 실행 필요.

[주의]
- LICY (Li-Cycle, Toronto/NYSE): 2024~2025년 Chapter 11 / 구조조정 이슈로
  거래정지·상장폐지 가능성이 있음. yfinance에서 빈 응답이 올 수 있음.
  그 경우 .TO 접미사 또는 OTC 티커로 재시도 필요.
- SQM (NYSE ADR): 정상 다운로드 가능할 것으로 추정.

[사용법]
    cd <repo_root>
    pip install yfinance pandas pyarrow
    python scripts/fetch_licy_sqm_prices.py
"""
from pathlib import Path
import time
import sys

import pandas as pd
import yfinance as yf

PRICES_PATH = Path("data/processed/prices_daily.parquet")
TARGETS = ["LICY", "SQM"]


def fetch_one(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    print(f"[{ticker}] downloading {start} ~ {end} ...")
    df = yf.download(ticker, start=start, end=end,
                     auto_adjust=False, progress=False)
    if df is None or df.empty:
        print(f"  -> empty response")
        return None
    sub = df.reset_index()
    if isinstance(sub.columns, pd.MultiIndex):
        sub.columns = [c[0] if isinstance(c, tuple) else c for c in sub.columns]
    sub = sub.rename(columns={
        "Date": "date", "Close": "close",
        "Adj Close": "adj_close", "Volume": "volume",
    })
    sub["company_id"] = ticker
    sub["provider_ticker"] = ticker
    keep = ["date", "company_id", "provider_ticker", "close", "adj_close", "volume"]
    for c in keep:
        if c not in sub.columns:
            sub[c] = pd.NA
    sub = sub[keep]
    sub["date"] = pd.to_datetime(sub["date"])
    print(f"  -> {len(sub):,} rows ({sub['date'].min().date()} ~ {sub['date'].max().date()})")
    return sub


def main():
    if not PRICES_PATH.exists():
        sys.exit(f"prices file not found: {PRICES_PATH}")

    existing = pd.read_parquet(PRICES_PATH)
    existing_tickers = set(existing["company_id"].unique())
    print(f"existing prices: {len(existing):,} rows / {len(existing_tickers)} firms")

    start_date = str(existing["date"].min().date())
    end_date = str(existing["date"].max().date())

    already = [t for t in TARGETS if t in existing_tickers]
    if already:
        print(f"already present (skip): {already}")

    todo = [t for t in TARGETS if t not in existing_tickers]
    if not todo:
        print("nothing to fetch.")
        return

    print(f"to fetch: {todo}")

    new_frames = []
    failed = []
    for tkr in todo:
        try:
            df = fetch_one(tkr, start_date, end_date)
            if df is None or df.empty:
                failed.append(tkr)
            else:
                new_frames.append(df)
            time.sleep(0.6)
        except Exception as e:
            print(f"  ERROR {tkr}: {e}")
            failed.append(tkr)

    if not new_frames:
        print("no new data fetched.")
        if failed:
            print(f"failed: {failed}")
            print("LICY 빈응답인 경우 .TO / OTC 티커 (LICYF 등) 재시도 필요.")
        return

    backup = PRICES_PATH.with_name(PRICES_PATH.stem + "_backup_pre_licy_sqm.parquet")
    if not backup.exists():
        existing.to_parquet(backup, index=False)
        print(f"backup written: {backup}")

    new_data = pd.concat(new_frames, ignore_index=True)
    combined = pd.concat([existing, new_data], ignore_index=True)
    combined = combined.sort_values(["company_id", "date"]).reset_index(drop=True)
    combined.to_parquet(PRICES_PATH, index=False)
    print(
        f"updated: {len(existing):,} -> {len(combined):,} rows, "
        f"{existing['company_id'].nunique()} -> {combined['company_id'].nunique()} firms"
    )

    if failed:
        print(f"\n[WARN] still failed: {failed}")
        print("  - LICY: Li-Cycle 2024~25 Ch.11. yfinance에서 .TO 또는 LICYF 재시도 권장.")


if __name__ == "__main__":
    main()
