from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np

def to_timestamp(x) -> Optional[pd.Timestamp]:
    ts = pd.to_datetime(x, errors="coerce", utc=False)
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts)

def next_trading_day(trading_days: np.ndarray, event_day: pd.Timestamp) -> Optional[pd.Timestamp]:
    """Given sorted unique trading_days (datetime64), return first day >= event_day."""
    if trading_days.size == 0:
        return None
    # normalize to day (event_day may have time)
    ed = np.datetime64(pd.Timestamp(event_day).normalize())
    pos = np.searchsorted(trading_days, ed, side="left")
    if pos >= trading_days.size:
        return None
    return pd.Timestamp(trading_days[pos])
