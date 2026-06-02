from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class PricesQC:
    tickers: int
    rows: int
    missing_by_company: pd.DataFrame
    outlier_by_company: pd.DataFrame

def run_prices_qc(prices: pd.DataFrame, missing_rate_max: float = 0.03) -> PricesQC:
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values(["company_id","date"])

    miss = df.groupby("company_id")["close"].apply(lambda s: float(s.isna().mean())).reset_index(name="missing_rate")
    miss["pass"] = miss["missing_rate"] <= missing_rate_max

    df["ret"] = df.groupby("company_id")["close"].pct_change()
    outlier = df.groupby("company_id")["ret"].apply(lambda s: float((s.abs() > 0.2).mean())).reset_index(name="outlier_rate(|ret|>0.2)")

    return PricesQC(
        tickers=int(df["company_id"].nunique()),
        rows=int(len(df)),
        missing_by_company=miss.sort_values("missing_rate", ascending=False),
        outlier_by_company=outlier.sort_values("outlier_rate(|ret|>0.2)", ascending=False),
    )
