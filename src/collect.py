"""Step 2: Collect articles from GDELT DOC API (chunked by date + risk type)."""

import time
from datetime import datetime, timedelta

import pandas as pd
from gdeltdoc import GdeltDoc, Filters

from .config import ARTICLES_CSV, RAW_DIR, load_risk_keywords


def _date_chunks(start: str, end: str, months: int = 3) -> list[tuple[str, str]]:
    """Split a date range into chunks of N months."""
    fmt = "%Y-%m-%d"
    s = datetime.strptime(start, fmt)
    e = datetime.strptime(end, fmt)
    chunks = []
    while s < e:
        chunk_end = s + timedelta(days=months * 30)
        if chunk_end > e:
            chunk_end = e
        chunks.append((s.strftime(fmt), chunk_end.strftime(fmt)))
        s = chunk_end
    return chunks


def _split_keyword_list(keywords: list[str], max_per_group: int = 8) -> list[list[str]]:
    """Split a keyword list into smaller groups to avoid query length limits."""
    groups = []
    for i in range(0, len(keywords), max_per_group):
        groups.append(keywords[i : i + max_per_group])
    return groups


def collect_articles(
    start_date: str,
    end_date: str,
    language: str = "English",
    query_override: str | None = None,
    chunk_months: int = 3,
    sleep_sec: float = 5.0,
) -> pd.DataFrame:
    """
    Fetch articles from GDELT DOC API.

    Splits by risk_type keyword groups and by date in 3-month chunks.
    Keywords are passed as a list so gdeltdoc wraps them properly.
    """
    keyword_map = load_risk_keywords()
    chunks = _date_chunks(start_date, end_date, chunk_months)

    # Build (label, keyword_list) pairs
    if query_override:
        query_groups = [("custom", query_override.split(" OR "))]
    else:
        query_groups = []
        for rtype, cfg in keyword_map.items():
            for gi, group in enumerate(_split_keyword_list(cfg["keywords"])):
                label = f"{rtype}_{gi}" if len(cfg["keywords"]) > 8 else rtype
                query_groups.append((label, group))

    total_calls = len(chunks) * len(query_groups)
    print(f"[collect] {len(query_groups)} keyword groups × {len(chunks)} date chunks = {total_calls} API calls")
    print(f"[collect] Estimated time: ~{total_calls * sleep_sec / 60:.1f} min")

    gd = GdeltDoc()
    all_dfs = []
    call_num = 0

    for label, kw_list in query_groups:
        print(f"\n[collect] === {label} === ({len(kw_list)} keywords)")
        for i, (cs, ce) in enumerate(chunks):
            call_num += 1
            print(f"  [{call_num}/{total_calls}] {cs} ~ {ce} ...", end=" ", flush=True)
            try:
                filters = Filters(
                    keyword=kw_list,
                    start_date=cs,
                    end_date=ce,
                    language=language,
                )
                chunk_df = gd.article_search(filters)
                if len(chunk_df) > 0:
                    chunk_df["_risk_query"] = label
                    all_dfs.append(chunk_df)
                    print(f"{len(chunk_df)} articles")
                else:
                    print("0 articles")
            except Exception as e:
                print(f"ERROR: {e}")

            if call_num < total_calls:
                time.sleep(sleep_sec)

    if all_dfs:
        articles = pd.concat(all_dfs, ignore_index=True)
        articles = articles.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
    else:
        articles = pd.DataFrame(columns=["url", "title", "seendate", "domain"])

    print(f"\n[collect] Total unique articles: {len(articles)}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    articles.to_csv(ARTICLES_CSV, index=False, encoding="utf-8-sig")
    print(f"[collect] Saved to {ARTICLES_CSV}")

    return articles


if __name__ == "__main__":
    collect_articles(start_date="2023-01-01", end_date="2026-01-01")
