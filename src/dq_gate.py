"""Step 8: Data Quality Gate - generate DQ report."""

import pandas as pd
from pathlib import Path

from .config import REPORTS_DIR


def generate_dq_report(
    articles_raw_count: int,
    articles_deduped_count: int,
    events_df: pd.DataFrame,
    output_path: Path | None = None,
) -> str:
    """
    Generate a data quality report for the news/risk events pipeline.

    Returns the report as a string and writes to reports/data_quality_report.md.
    """
    if output_path is None:
        output_path = REPORTS_DIR / "data_quality_report.md"

    n_events = len(events_df)

    # risk_type distribution
    rt_counts: dict[str, int] = {}
    for rts in events_df["risk_types"]:
        if isinstance(rts, list):
            for rt in rts:
                rt_counts[rt] = rt_counts.get(rt, 0) + 1
        else:
            rt_counts[str(rts)] = rt_counts.get(str(rts), 0) + 1
    rt_sorted = sorted(rt_counts.items(), key=lambda x: -x[1])

    # entity link success rate
    linked = sum(
        1 for eids in events_df["entity_ids"]
        if isinstance(eids, list) and len(eids) > 0
    )
    link_rate = linked / n_events if n_events else 0

    # evidence_text coverage
    evidence_missing = sum(
        1 for t in events_df["evidence_text"]
        if not t or str(t).strip() == "" or str(t) == "nan"
    )
    evidence_rate = 1 - (evidence_missing / n_events) if n_events else 0

    # severity stats
    sev_mean = events_df["severity"].mean()
    sev_max = events_df["severity"].max()
    sev_min = events_df["severity"].min()

    # Build report
    rt_lines = ", ".join(f"{k}({v})" for k, v in rt_sorted)

    report = f"""# Data Quality Report - News / Risk Events

## Collection
- articles_raw: {articles_raw_count}
- articles_after_dedup: {articles_deduped_count}
- events_final: {n_events}
- dedup_ratio: {1 - articles_deduped_count / articles_raw_count:.2%} removed

## Risk Type Distribution
- {rt_lines}

## Entity Linking
- entity_link_success_rate: {link_rate:.2%}

## Evidence Text
- evidence_text_coverage: {evidence_rate:.2%}

## Severity
- mean: {sev_mean:.2f}
- max: {sev_max:.2f}
- min: {sev_min:.2f}

## DQ Gate Status
- risk_type consistency: {'PASS' if n_events >= 10 else 'WARN (< 10 events)'}
- entity_link_rate ≥ 0.90: {'PASS' if link_rate >= 0.90 else 'FAIL'}
- evidence_text_coverage ≥ 0.95: {'PASS' if evidence_rate >= 0.95 else 'FAIL'}
"""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"[dq_gate] Report saved to {output_path}")

    return report
