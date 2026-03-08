from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def make_mock_articles(out_csv: Path) -> pd.DataFrame:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"seendate":"2026-03-10", "title":"BYD faces export ban rumor as sanctions tighten", "url":"http://example.com/a1"},
        {"seendate":"2026-03-18", "title":"CATL logistics disruption due to port strike", "url":"http://example.com/a2"},
        {"seendate":"2026-04-05", "title":"Tesla supplier disruption: earthquake impacts battery materials", "url":"http://example.com/a3"},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return df

def make_mock_prices(company_ids: list[str], out_parquet: Path, start: str, end: str) -> pd.DataFrame:
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    dates = pd.date_range(start_dt, end_dt, freq="B")  # business days
    all_rows = []
    rng = np.random.default_rng(42)
    for cid in company_ids[:10]:  # limit for smoke test
        price = 100 + np.cumsum(rng.normal(0, 1, size=len(dates)))
        for d, p in zip(dates, price):
            all_rows.append({"date": d, "company_id": cid, "provider_ticker": cid, "close": float(p), "adj_close": float(p), "volume": int(abs(rng.normal(1e6, 2e5)))})
    df = pd.DataFrame(all_rows)
    df.to_parquet(out_parquet, index=False)
    return df
