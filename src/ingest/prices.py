from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from pathlib import Path
import time
import pandas as pd

from src.ingest.tickers import normalize_for_yfinance

def _download_batch(tickers: List[str], start: str, end: str, retries: int = 3, backoff: float = 2.0) -> pd.DataFrame:
    try:
        import yfinance as yf
    except Exception as e:
        raise RuntimeError("yfinance 미설치: pip install yfinance") from e

    last_err: Optional[Exception] = None
    for i in range(retries):
        try:
            df = yf.download(
                tickers=tickers,
                start=start,
                end=end,
                group_by="ticker",
                auto_adjust=False,
                threads=True,
                progress=False,
            )
            return df
        except Exception as e:
            last_err = e
            time.sleep(backoff * (2 ** i))
    raise RuntimeError(f"가격 수집 실패(재시도 {retries}회). 마지막 에러: {last_err}")

def ingest_prices_from_universe(universe: pd.DataFrame,
                               rules: Dict[str,str],
                               out_parquet: Path,
                               start: str,
                               end: str,
                               retries: int = 3,
                               backoff: float = 2.0,
                               force: bool = False,
                               batch_size: int = 30) -> pd.DataFrame:
    """Download prices via yfinance using provider tickers derived from universe company_id."""
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    if out_parquet.exists() and not force:
        return pd.read_parquet(out_parquet)

    # Build provider tickers mapping
    mapping = []
    for r in universe.itertuples(index=False):
        cid = str(getattr(r, "company_id"))
        if cid == "nan" or cid.strip() == "":
            continue
        prov = normalize_for_yfinance(cid, rules)
        mapping.append((cid, prov))
    map_df = pd.DataFrame(mapping, columns=["company_id","provider_ticker"]).drop_duplicates()
    provider_tickers = map_df["provider_ticker"].dropna().unique().tolist()

    all_rows = []
    missing = []
    # Download in batches
    for i in range(0, len(provider_tickers), batch_size):
        batch = provider_tickers[i:i+batch_size]
        raw = _download_batch(batch, start, end, retries=retries, backoff=backoff)

        # Convert to tidy
        if raw.empty:
            missing.extend(batch)
            continue

        if isinstance(raw.columns, pd.MultiIndex):
            for t in batch:
                if t not in raw.columns.get_level_values(0):
                    missing.append(t)
                    continue
                sub = raw[t].copy()
                sub["provider_ticker"] = t
                sub = sub.reset_index().rename(columns={"Date": "date"})
                # Standardize columns
                rename = {c: c.lower().replace(" ", "_") for c in sub.columns}
                sub = sub.rename(columns=rename)
                keep_cols = ["date","provider_ticker","close","adj_close","volume"]
                for c in keep_cols:
                    if c not in sub.columns:
                        sub[c] = pd.NA
                all_rows.append(sub[keep_cols])
        else:
            # single ticker
            t = batch[0]
            sub = raw.copy().reset_index().rename(columns={"Date":"date"})
            sub["provider_ticker"] = t
            sub = sub.rename(columns={"Close":"close","Adj Close":"adj_close","Volume":"volume"})[["date","provider_ticker","close","adj_close","volume"]]
            all_rows.append(sub)

    tidy = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame(columns=["date","provider_ticker","close","adj_close","volume"])
    tidy["date"] = pd.to_datetime(tidy["date"], errors="coerce")
    tidy = tidy.dropna(subset=["date"])

    # Map provider_ticker back to company_id
    tidy = tidy.merge(map_df, on="provider_ticker", how="left")
    # If multiple company_id map to same provider ticker (unlikely), keep as is; user can disambiguate.
    tidy = tidy[["date","company_id","provider_ticker","close","adj_close","volume"]].dropna(subset=["company_id"])

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    tidy.to_parquet(out_parquet, index=False)

    # Save missing tickers list (non-fatal but important)
    if missing:
        miss_path = out_parquet.with_suffix(".missing_tickers.txt")
        miss_path.write_text("\n".join(sorted(set(missing))) + "\n", encoding="utf-8")

    return tidy
