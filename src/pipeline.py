"""
Main pipeline: Orchestrate all steps from collection to risk_events.parquet.

Usage:
    # Full pipeline (requires GDELT API access):
    python -m src.pipeline --start 2026-03-01 --end 2026-06-01

    # From existing articles.csv (skip collection):
    python -m src.pipeline --skip-collect

    # With FinBERT v1 severity:
    python -m src.pipeline --skip-collect --finbert
"""

import argparse
import pandas as pd

from .config import ARTICLES_CSV
from .collect import collect_articles
from .dedup import dedup_articles
from .classify import classify_articles
from .severity import compute_severity_v0, compute_severity_v1
from .entity_link import link_all_entities
from .cluster import cluster_into_events, build_risk_events
from .dq_gate import generate_dq_report


def run_pipeline(
    start_date: str = "2026-03-01",
    end_date: str = "2026-06-01",
    skip_collect: bool = False,
    use_finbert: bool = False,
    query_override: str | None = None,
):
    """Run the full news → risk_events pipeline."""

    # ---- Step 2: Collect ----
    if skip_collect:
        print(f"\n=== Step 2: SKIP (loading {ARTICLES_CSV}) ===")
        df = pd.read_csv(ARTICLES_CSV, encoding="utf-8-sig")
        print(f"[pipeline] Loaded {len(df)} articles from CSV")
    else:
        print("\n=== Step 2: Collect articles from GDELT ===")
        df = collect_articles(
            start_date=start_date,
            end_date=end_date,
            query_override=query_override,
        )

    raw_count = len(df)

    if df.empty:
        print("\n[pipeline] ERROR: No articles collected. Check GDELT API or provide articles.csv manually.")
        return pd.DataFrame()

    # ---- Step 3: Dedup ----
    print("\n=== Step 3: Deduplicate ===")
    df = dedup_articles(df)
    deduped_count = len(df)

    # ---- Step 4: Classify risk_type ----
    print("\n=== Step 4: Classify risk types ===")
    df = classify_articles(df)

    # ---- Step 5: Severity ----
    print("\n=== Step 5: Compute severity (v0) ===")
    df = compute_severity_v0(df)

    if use_finbert:
        print("\n=== Step 5b: Augment severity with FinBERT (v1) ===")
        df = compute_severity_v1(df)

    # ---- Step 6: Entity linking ----
    print("\n=== Step 6: Entity linking ===")
    df = link_all_entities(df)

    # ---- Step 7: Cluster → risk_events.parquet ----
    print("\n=== Step 7: Cluster into events ===")
    df = cluster_into_events(df)
    events_df = build_risk_events(df)

    # ---- Step 8: DQ Gate ----
    print("\n=== Step 8: Data Quality Gate ===")
    report = generate_dq_report(raw_count, deduped_count, events_df)
    print(report)

    print("\n=== Pipeline complete ===")
    print(f"Output: data/processed/risk_events.parquet ({len(events_df)} events)")

    return events_df


def main():
    parser = argparse.ArgumentParser(description="News → Risk Events pipeline")
    parser.add_argument("--start", default="2026-03-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-06-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--skip-collect", action="store_true", help="Skip GDELT collection, use existing CSV")
    parser.add_argument("--finbert", action="store_true", help="Use FinBERT v1 severity augmentation")
    parser.add_argument("--query", default=None, help="Override GDELT query string")
    args = parser.parse_args()

    run_pipeline(
        start_date=args.start,
        end_date=args.end,
        skip_collect=args.skip_collect,
        use_finbert=args.finbert,
        query_override=args.query,
    )


if __name__ == "__main__":
    main()
