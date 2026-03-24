from __future__ import annotations
from dataclasses import dataclass
from typing import List
import pandas as pd

@dataclass
class NewsQC:
    rows_before: int
    rows_after: int
    dedup_ratio: float

def dedup(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    out = df.copy()
    keys2 = [k for k in keys if k in out.columns]
    if keys2:
        out = out.drop_duplicates(subset=keys2)
    return out

def run_news_qc(df: pd.DataFrame, dedup_keys: List[str]) -> tuple[NewsQC, pd.DataFrame]:
    before = len(df)
    after_df = dedup(df, dedup_keys)
    after = len(after_df)
    ratio = 0.0 if before == 0 else float((before - after) / before)
    return NewsQC(before, after, ratio), after_df
