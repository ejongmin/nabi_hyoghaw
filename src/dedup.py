"""Step 3: Deduplicate articles by URL and title similarity."""

import pandas as pd
from rapidfuzz import fuzz


def dedup_articles(df: pd.DataFrame, title_threshold: int = 85) -> pd.DataFrame:
    """
    Remove duplicate articles.

    1) Exact URL dedup
    2) Near-duplicate title dedup (fuzzy match)

    Parameters
    ----------
    df : DataFrame with at least 'url' and 'title' columns.
    title_threshold : int  Minimum fuzz ratio to consider titles as duplicates.

    Returns
    -------
    Deduplicated DataFrame.
    """
    if df.empty:
        print("[dedup] Empty DataFrame - skipping")
        return df

    before = len(df)

    # --- 1차: URL 중복 제거 ---
    if "url" not in df.columns:
        print("[dedup] No 'url' column - skipping URL dedup")
    else:
        df = df.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
    after_url = len(df)
    print(f"[dedup] URL dedup: {before} → {after_url} (removed {before - after_url})")

    # --- 2차: Title 유사도 기반 중복 제거 ---
    if "title" not in df.columns:
        print("[dedup] No 'title' column - skipping title dedup")
        return df

    keep_mask = [True] * len(df)
    titles = df["title"].fillna("").tolist()

    for i in range(len(titles)):
        if not keep_mask[i]:
            continue
        for j in range(i + 1, len(titles)):
            if not keep_mask[j]:
                continue
            score = fuzz.ratio(titles[i].lower(), titles[j].lower())
            if score >= title_threshold:
                keep_mask[j] = False

    df = df[keep_mask].reset_index(drop=True)
    after_title = len(df)
    print(
        f"[dedup] Title dedup: {after_url} → {after_title} "
        f"(removed {after_url - after_title})"
    )

    return df
