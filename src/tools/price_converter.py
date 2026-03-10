import pandas as pd
from pathlib import Path

def convert_raw_to_parquet(csv_in: Path, univ_in: Path, parquet_out: Path):
    print(f"[*] Converting {csv_in} to {parquet_out}...")
    
    # 1. Load data
    df = pd.read_csv(csv_in)
    uni = pd.read_csv(univ_in)
    
    # 2. Map ticker to company_id
    # Note: ticker in CSV might need cleaning (e.g. .SS instead of .SH)
    # Let's check the ticker mapping logic
    ticker_to_cid = {}
    for _, row in uni.iterrows():
        cid = row['company_id']
        pt = row.get('provider_ticker_yf')
        if pd.notna(pt):
            ticker_to_cid[str(pt).strip()] = cid
            
        # Fallback for tickers field (separated by ;)
        ts = str(row.get('tickers', '')).split(';')
        for t in ts:
            t = t.strip()
            if t:
                ticker_to_cid[t] = cid

    # 3. Apply Mapping
    df['company_id'] = df['ticker'].map(ticker_to_cid)
    
    # Check for unmapped tickers
    unmapped = df[df['company_id'].isna()]['ticker'].unique()
    if len(unmapped) > 0:
        print(f"[!] Warning: Unmapped tickers: {unmapped}")
        # Try some heuristics (e.g. .SH -> .SS)
        for t in unmapped:
            if '.SH' in t:
                alt = t.replace('.SH', '.SS')
                if alt in ticker_to_cid:
                    df.loc[df['ticker'] == t, 'company_id'] = ticker_to_cid[alt]
            elif '.SS' in t:
                alt = t.replace('.SS', '.SH')
                if alt in ticker_to_cid:
                    df.loc[df['ticker'] == t, 'company_id'] = ticker_to_cid[alt]
    
    # Final check
    df = df.dropna(subset=['company_id'])
    
    # 4. Cleanup and Save
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['company_id', 'date'])
    
    # Select final columns to match project schema
    # Required: company_id, date, open, high, low, close, volume, adj_close
    final_df = df[['company_id', 'date', 'open', 'high', 'low', 'close', 'volume', 'adj_close']]
    
    final_df.to_parquet(parquet_out, index=False)
    print(f"[*] Done. Saved {len(final_df)} rows for {df['company_id'].nunique()} companies.")
    print(f"[*] Period: {final_df['date'].min()} to {final_df['date'].max()}")

if __name__ == "__main__":
    csv_in = Path("/mnt/c/Users/john9/Desktop/prices_daily_raw.csv")
    univ_in = Path("/mnt/c/Users/john9/nabi_hyoghaw/data/universe/univers_final.csv")
    parquet_out = Path("/mnt/c/Users/john9/nabi_hyoghaw/data/processed/prices_daily.parquet")
    
    convert_raw_to_parquet(csv_in, univ_in, parquet_out)
