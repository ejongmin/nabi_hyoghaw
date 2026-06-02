from __future__ import annotations
import pandas as pd
import numpy as np

def max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return float(dd.min())

def annualized_return(equity: pd.Series, periods_per_year: int = 252) -> float:
    if len(equity) < 2:
        return 0.0
    total = float(equity.iloc[-1] / equity.iloc[0])
    years = (len(equity) - 1) / periods_per_year
    if years <= 0:
        return 0.0
    return total ** (1/years) - 1.0

def annualized_vol(returns: pd.Series, periods_per_year: int = 252) -> float:
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))

def sharpe(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    ex = returns - rf/periods_per_year
    vol = ex.std(ddof=1)
    if vol == 0 or np.isnan(vol):
        return 0.0
    return float(ex.mean() / vol * np.sqrt(periods_per_year))
