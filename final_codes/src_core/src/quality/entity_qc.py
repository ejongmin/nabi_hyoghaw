from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

@dataclass
class EntityQC:
    rows: int
    linked_rows: int
    success_rate: float
    top_failed_titles: pd.DataFrame

def run_entity_qc(risk_events: pd.DataFrame, topn: int = 10) -> EntityQC:
    df = risk_events.copy()
    df["has_entity"] = df["entity_ids"].apply(lambda x: x is not None and len(x) > 0)
    rows = len(df)
    linked = int(df["has_entity"].sum())
    rate = 0.0 if rows == 0 else linked / rows
    cols = [c for c in ["event_id", "title", "url", "Actor1Name"] if c in df.columns]
    failed = df[~df["has_entity"]][cols].head(topn)
    return EntityQC(rows=rows, linked_rows=linked, success_rate=float(rate), top_failed_titles=failed)
