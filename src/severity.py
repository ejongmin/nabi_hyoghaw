"""Step 5: Compute severity scores (v0 rule-based, optional v1 FinBERT)."""

import math
import pandas as pd

from .config import load_risk_keywords


# ---------------------------------------------------------------------------
# v0: Rule-based severity
# ---------------------------------------------------------------------------

def _keyword_hit_count(text: str, keyword_map: dict) -> int:
    """Count total keyword hits across all risk types."""
    text_lower = text.lower()
    count = 0
    for cfg in keyword_map.values():
        for kw in cfg["keywords"]:
            if kw.lower() in text_lower:
                count += 1
    return count


def compute_severity_v0(
    df: pd.DataFrame,
    dup_count_col: str | None = None,
) -> pd.DataFrame:
    """
    Rule-based severity (v0).

    severity = 1.0
             + 0.4 * keyword_hit_count
             + 0.3 * is_in_title  (always 1 since we match on title)
             + 0.2 * log(1 + duplicate_article_count)
    capped at 5.0.
    """
    keyword_map = load_risk_keywords()

    severities = []
    for _, row in df.iterrows():
        title = str(row.get("title", ""))
        hits = _keyword_hit_count(title, keyword_map)
        dup_count = int(row[dup_count_col]) if dup_count_col and dup_count_col in df.columns else 0

        s = 1.0
        s += 0.4 * hits
        s += 0.3  # keyword matched in title
        s += 0.2 * math.log(1 + dup_count)
        s = min(s, 5.0)
        severities.append(round(s, 3))

    df["severity"] = severities
    print(f"[severity-v0] mean={sum(severities)/len(severities):.2f}, "
          f"max={max(severities):.2f}, min={min(severities):.2f}")
    return df


# ---------------------------------------------------------------------------
# v1: FinBERT augmented (optional)
# ---------------------------------------------------------------------------

def compute_severity_v1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Augment severity with FinBERT sentiment.

    severity_v1 = severity + 0.2 * max(0, neg - pos)

    Requires 'transformers' and 'torch'. If unavailable, skips silently.
    """
    try:
        from transformers import pipeline
    except ImportError:
        print("[severity-v1] transformers not installed - skipping FinBERT")
        return df

    print("[severity-v1] Loading FinBERT pipeline...")
    finbert = pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        truncation=True,
        max_length=512,
    )

    titles = df["title"].fillna("").tolist()
    results = finbert(titles, batch_size=32)

    sentiments = []
    for r in results:
        label = r["label"].lower()
        score = r["score"]
        sentiments.append({"label": label, "score": score})

    neg_scores = []
    pos_scores = []
    for s in sentiments:
        if s["label"] == "negative":
            neg_scores.append(s["score"])
            pos_scores.append(0.0)
        elif s["label"] == "positive":
            neg_scores.append(0.0)
            pos_scores.append(s["score"])
        else:
            neg_scores.append(0.0)
            pos_scores.append(0.0)

    df["finbert_negative"] = neg_scores
    df["finbert_positive"] = pos_scores
    df["finbert_label"] = [s["label"] for s in sentiments]

    df["severity"] = df.apply(
        lambda row: min(
            5.0,
            row["severity"] + 0.2 * max(0, row["finbert_negative"] - row["finbert_positive"]),
        ),
        axis=1,
    ).round(3)

    print("[severity-v1] FinBERT augmentation complete")
    return df
