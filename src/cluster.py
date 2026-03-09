"""Step 7: Cluster deduplicated articles into events and build risk_events."""

import hashlib
import pandas as pd
from rapidfuzz import fuzz

from .config import PROCESSED_DIR, RISK_EVENTS_PARQUET


def _date_bucket(dt_str: str) -> str:
    """Extract YYYY-MM-DD from a datetime string."""
    return str(dt_str)[:10]


def cluster_into_events(
    df: pd.DataFrame,
    title_sim_threshold: int = 70,
) -> pd.DataFrame:
    """
    Cluster articles into events based on:
    - same date bucket (1 day)
    - overlapping risk_types
    - overlapping entity_ids
    - high title similarity

    Each cluster becomes one event row.
    """
    # Ensure required columns
    if "seendate" in df.columns:
        date_col = "seendate"
    elif "date" in df.columns:
        date_col = "date"
    else:
        date_col = df.columns[0]  # fallback

    df = df.copy()
    df["_date_bucket"] = df[date_col].astype(str).apply(_date_bucket)

    # Assign cluster IDs via greedy merging within each date bucket
    df["_cluster"] = -1
    cluster_id = 0

    for bucket, group in df.groupby("_date_bucket"):
        indices = group.index.tolist()
        assigned = {}  # index -> cluster_id

        for i, idx in enumerate(indices):
            if idx in assigned:
                continue
            assigned[idx] = cluster_id
            row_i = df.loc[idx]
            rt_i = set(row_i.get("risk_types", []))
            eid_i = set(row_i.get("entity_ids", []))
            title_i = str(row_i.get("title", "")).lower()

            for j in range(i + 1, len(indices)):
                jdx = indices[j]
                if jdx in assigned:
                    continue
                row_j = df.loc[jdx]
                rt_j = set(row_j.get("risk_types", []))
                eid_j = set(row_j.get("entity_ids", []))
                title_j = str(row_j.get("title", "")).lower()

                # Check overlap conditions
                rt_overlap = bool(rt_i & rt_j)
                eid_overlap = bool(eid_i & eid_j) if (eid_i and eid_j) else False
                title_sim = fuzz.ratio(title_i, title_j)

                if rt_overlap and (eid_overlap or title_sim >= title_sim_threshold):
                    assigned[jdx] = cluster_id

            cluster_id += 1

        for idx, cid in assigned.items():
            df.at[idx, "_cluster"] = cid

    # Handle any unassigned (shouldn't happen, but safety)
    unassigned = df["_cluster"] == -1
    if unassigned.any():
        for idx in df[unassigned].index:
            df.at[idx, "_cluster"] = cluster_id
            cluster_id += 1

    print(f"[cluster] {len(df)} articles → {df['_cluster'].nunique()} event clusters")
    return df


def build_risk_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate article clusters into the final risk_events schema.

    Schema:
        event_id, event_time, risk_types, severity,
        entity_mentions, entity_ids, source, url, evidence_text
    """
    events = []

    for cid, group in df.groupby("_cluster"):
        # Pick representative article (highest severity)
        rep = group.sort_values("severity", ascending=False).iloc[0]

        # Merge all entity info
        all_eids = []
        all_mentions = []
        for _, row in group.iterrows():
            all_eids.extend(row.get("entity_ids", []))
            all_mentions.extend(row.get("entity_mentions", []))
        # deduplicate
        all_eids = list(dict.fromkeys(all_eids))
        all_mentions = list(dict.fromkeys(all_mentions))

        # Merge risk_types
        all_rt = []
        for _, row in group.iterrows():
            all_rt.extend(row.get("risk_types", []))
        all_rt = list(dict.fromkeys(all_rt))

        # event_id: hash of cluster content
        date_col = "seendate" if "seendate" in group.columns else "date"
        date_val = str(rep.get(date_col, rep.get("_date_bucket", "")))
        hash_input = f"{date_val}|{'|'.join(all_rt)}|{rep.get('title', '')}"
        event_id = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

        # Max severity in cluster
        max_severity = group["severity"].max()

        # Source info
        source = str(rep.get("domain", rep.get("source", "")))
        url = str(rep.get("url", ""))

        events.append({
            "event_id": event_id,
            "event_time": date_val,
            "risk_types": all_rt,
            "severity": round(max_severity, 3),
            "entity_mentions": all_mentions,
            "entity_ids": all_eids,
            "source": source,
            "url": url,
            "evidence_text": str(rep.get("title", "")),
            "article_count": len(group),
        })

    events_df = pd.DataFrame(events)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    events_df.to_parquet(RISK_EVENTS_PARQUET, index=False)
    print(f"[cluster] Saved {len(events_df)} events to {RISK_EVENTS_PARQUET}")

    return events_df
