#!/usr/bin/env python3
"""
누락된 7개 티커 가격 데이터를 기존 prices_daily.parquet에 추가하는 스크립트.
로컬에서 실행: python scripts/fix_missing_prices.py
"""
import pandas as pd
import yfinance as yf
import time
from pathlib import Path

PRICES_PATH = Path("data/processed/prices_daily.parquet")

def main():
    existing = pd.read_parquet(PRICES_PATH)
    start_date = str(existing['date'].min().date())
    end_date = str(existing['date'].max().date())
    
    existing_tickers = set(existing['company_id'].unique())
    universe = pd.read_csv("data/processed/universe_normalized.csv")
    all_cids = set(universe['company_id'].tolist())
    missing = sorted(all_cids - existing_tickers)
    
    if not missing:
        print("All tickers already have price data!")
        return
    
    print(f"Missing tickers: {missing}")
    print(f"Date range: {start_date} ~ {end_date}")
    
    all_new = []
    failed = []
    
    for cid in missing:
        try:
            df = yf.download(cid, start=start_date, end=end_date,
                            auto_adjust=False, progress=False)
            if df.empty:
                print(f"  {cid}: No data")
                failed.append(cid)
                continue
            
            sub = df.reset_index()
            if isinstance(sub.columns, pd.MultiIndex):
                sub.columns = [c[0] if isinstance(c, tuple) else c for c in sub.columns]
            
            sub = sub.rename(columns={
                'Date': 'date', 'Close': 'close',
                'Adj Close': 'adj_close', 'Volume': 'volume'
            })
            sub['company_id'] = cid
            sub['provider_ticker'] = cid
            
            keep = ['date', 'company_id', 'provider_ticker', 'close', 'adj_close', 'volume']
            for c in keep:
                if c not in sub.columns:
                    sub[c] = pd.NA
            sub = sub[keep]
            sub['date'] = pd.to_datetime(sub['date'])
            
            all_new.append(sub)
            print(f"  {cid}: {len(sub)} rows")
            time.sleep(0.5)
        except Exception as e:
            print(f"  {cid}: ERROR - {e}")
            failed.append(cid)
    
    if all_new:
        new_data = pd.concat(all_new, ignore_index=True)
        combined = pd.concat([existing, new_data], ignore_index=True)
        combined = combined.sort_values(['company_id', 'date']).reset_index(drop=True)
        combined.to_parquet(PRICES_PATH, index=False)
        print(f"\nUpdated: {len(existing):,} → {len(combined):,} rows, "
              f"{existing['company_id'].nunique()} → {combined['company_id'].nunique()} tickers")
    
    if failed:
        print(f"\nFailed (need manual check): {failed}")
        print("Note: 920185.BJ (BTR) may not be on yfinance. LICY may be delisted.")

if __name__ == "__main__":
    main()
