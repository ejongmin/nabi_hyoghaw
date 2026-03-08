from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
from pathlib import Path
import time
import pandas as pd

def ingest_gdelt(cfg: Dict[str, Any], out_csv: Path, force: bool = False) -> pd.DataFrame:
    """Fetch articles from GDELT DOC API via gdeltdoc. Writes CSV."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists() and not force:
        return pd.read_csv(out_csv)

    try:
        from gdeltdoc import GdeltDoc, Filters
    except Exception as e:
        raise RuntimeError("gdeltdoc 미설치: pip install gdeltdoc") from e

    gcfg = cfg.get("gdelt", {})
    keyword = gcfg.get("keyword_query")
    start_date = cfg.get("project", {}).get("start_date")
    end_date = cfg.get("project", {}).get("end_date")
    language = gcfg.get("language", "English")
    max_records = int(gcfg.get("max_records", 2000))
    retries = int(gcfg.get("retries", 3))
    backoff = float(gcfg.get("retry_backoff_sec", 2.0))

    gd = GdeltDoc()
    f = Filters(keyword=keyword, start_date=start_date, end_date=end_date, language=language)

    last_err: Optional[Exception] = None
    for i in range(retries):
        try:
            df = gd.article_search(f)
            if df is None:
                df = pd.DataFrame()
            if len(df) > max_records:
                df = df.head(max_records).copy()
            df.to_csv(out_csv, index=False, encoding="utf-8-sig")
            return df
        except Exception as e:
            last_err = e
            time.sleep(backoff * (2 ** i))

    raise RuntimeError(f"GDELT 수집 실패(재시도 {retries}회). 네트워크/쿼리를 확인하세요. 마지막 에러: {last_err}")
