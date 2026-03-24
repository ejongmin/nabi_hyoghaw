from __future__ import annotations
import pandas as pd

def df_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    """Return a markdown table if possible; otherwise fall back to CSV text.

    pandas.DataFrame.to_markdown requires `tabulate`. We include tabulate in requirements,
    but this fallback prevents hard-fail if the dependency is missing.
    """
    if df is None:
        return ""
    df2 = df.head(max_rows).copy()
    try:
        return df2.to_markdown(index=False)
    except Exception:
        return df2.to_csv(index=False)
