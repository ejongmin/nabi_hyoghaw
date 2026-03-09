"""Step 6: Link articles to company entities from the universe master."""

import pandas as pd
from pathlib import Path
from rapidfuzz import fuzz

from .config import UNIVERSE_DIR


def load_universe() -> pd.DataFrame:
    """
    Load company universe master from data/universe/.

    Looks for universe_final.csv first, then any *_combined.csv.
    Expected columns: company_id, canonical_name, tickers (semicolon-separated),
                      aliases (semicolon-separated, optional).
    """
    final = UNIVERSE_DIR / "universe_final.csv"
    if final.exists():
        return pd.read_csv(final, encoding="utf-8-sig")

    combined = sorted(UNIVERSE_DIR.glob("*_combined.csv"))
    if combined:
        return pd.read_csv(combined[0], encoding="utf-8-sig")

    raise FileNotFoundError(
        f"No universe file found in {UNIVERSE_DIR}. "
        "Place universe_final.csv or *_combined.csv there."
    )


def _build_lookup(universe_df: pd.DataFrame) -> list[dict]:
    """Pre-build a lookup list of (company_id, search_terms) for speed."""
    lookup = []
    for _, row in universe_df.iterrows():
        cid = row["company_id"]
        terms = set()

        # canonical_name
        cn = str(row.get("canonical_name", "")).strip()
        if cn:
            terms.add(cn.lower())

        # tickers
        tickers_raw = str(row.get("tickers", ""))
        for t in tickers_raw.split(";"):
            t = t.strip()
            if t:
                terms.add(t.lower())

        # aliases
        aliases_raw = str(row.get("aliases", ""))
        for a in aliases_raw.split(";"):
            a = a.strip()
            if a:
                terms.add(a.lower())

        lookup.append({"company_id": cid, "terms": terms, "canonical_name": cn})

    return lookup


def link_entities(
    title: str,
    lookup: list[dict],
    min_fuzzy_score: int = 90,
    max_entities: int = 3,
) -> tuple[list[str], list[str]]:
    """
    Link a single article title to company entities.

    Returns (entity_ids, entity_mentions).
    """
    text = str(title).lower()
    hits = []
    mentions = []

    for entry in lookup:
        for term in entry["terms"]:
            # exact substring match
            if term in text:
                hits.append(entry["company_id"])
                mentions.append(term)
                break
            # fuzzy match for longer names (>=4 chars) to avoid false positives
            if len(term) >= 4:
                score = fuzz.partial_ratio(term, text)
                if score >= min_fuzzy_score:
                    hits.append(entry["company_id"])
                    mentions.append(term)
                    break

    # deduplicate, keep order, cap at max_entities
    seen = set()
    unique_ids = []
    unique_mentions = []
    for eid, mention in zip(hits, mentions):
        if eid not in seen:
            seen.add(eid)
            unique_ids.append(eid)
            unique_mentions.append(mention)
        if len(unique_ids) >= max_entities:
            break

    return unique_ids, unique_mentions


def link_all_entities(df: pd.DataFrame) -> pd.DataFrame:
    """Add entity_ids and entity_mentions columns to the articles DataFrame."""
    try:
        universe_df = load_universe()
    except FileNotFoundError as e:
        print(f"[entity_link] WARNING: {e}")
        print("[entity_link] Skipping entity linking - no universe file.")
        df["entity_ids"] = [[] for _ in range(len(df))]
        df["entity_mentions"] = [[] for _ in range(len(df))]
        return df

    lookup = _build_lookup(universe_df)
    print(f"[entity_link] Universe loaded: {len(lookup)} companies")

    ids_col = []
    mentions_col = []
    for _, row in df.iterrows():
        ids, mentions = link_entities(str(row.get("title", "")), lookup)
        ids_col.append(ids)
        mentions_col.append(mentions)

    df["entity_ids"] = ids_col
    df["entity_mentions"] = mentions_col

    linked = sum(1 for ids in ids_col if len(ids) > 0)
    print(f"[entity_link] Linked {linked}/{len(df)} articles ({linked/len(df)*100:.1f}%)")

    return df
