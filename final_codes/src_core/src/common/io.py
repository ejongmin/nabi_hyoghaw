from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd

def read_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    suf = path.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf == ".csv":
        return pd.read_csv(path)
    if suf == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suf in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {suf}")

def write_df(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suf = path.suffix.lower()
    if suf == ".parquet":
        df.to_parquet(path, index=index)
        return
    if suf == ".csv":
        df.to_csv(path, index=index, encoding="utf-8-sig")
        return
    if suf == ".tsv":
        df.to_csv(path, index=index, sep="\t", encoding="utf-8")
        return
    raise ValueError(f"Unsupported output type: {suf}")
