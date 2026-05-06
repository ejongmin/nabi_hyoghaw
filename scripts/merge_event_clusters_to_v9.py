"""
Phase 2-D: v8 + event_clusters.parquet → v9
Row-group 스트리밍 + dict-lookup merge (4GB VM 최적화)
"""

import gc
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
V8 = ROOT / "data" / "processed" / "risk_events_classified_v8.parquet"
CLUSTERS = ROOT / "data" / "processed" / "event_clusters.parquet"
V9 = ROOT / "data" / "processed" / "risk_events_classified_v9.parquet"

CHUNK = 25_000


def build_lookup_dict():
    log.info("Loading cluster lookup")
    lk = pd.read_parquet(CLUSTERS)
    lk = lk.sort_values(
        ["event_id", "is_cluster_lead", "cluster_size"],
        ascending=[True, False, False],
    ).drop_duplicates(subset="event_id", keep="first")
    log.info("  unique event_ids: %d", len(lk))

    # 4 parallel arrays keyed by event_id
    eid = lk["event_id"].to_numpy()
    cid = lk["event_cluster_id"].to_numpy(dtype="int64")
    csz = lk["cluster_size"].to_numpy(dtype="int32")
    lead = lk["is_cluster_lead"].to_numpy(dtype=bool)
    ltier = lk["cluster_lead_tier"].to_numpy(dtype="int8")

    # single dict: eid → (cid, csz, lead, ltier) as int64 packed is overkill;
    # use 4 separate dicts or one dict with tuple. Dict of tuples is fine for 42K.
    d = {e: (int(cid[i]), int(csz[i]), bool(lead[i]), int(ltier[i]))
         for i, e in enumerate(eid)}
    del lk, eid, cid, csz, lead, ltier
    gc.collect()
    return d


def main():
    lookup = build_lookup_dict()
    log.info("lookup dict size: %d keys", len(lookup))

    pf_in = pq.ParquetFile(V8)
    schema_in = pf_in.schema_arrow
    log.info("v8: %d cols, %d row_groups, %d rows",
             len(schema_in), pf_in.num_row_groups, pf_in.metadata.num_rows)

    new_fields = [
        pa.field("event_cluster_id", pa.int64(), nullable=True),
        pa.field("cluster_size", pa.int32()),
        pa.field("is_cluster_lead", pa.bool_()),
        pa.field("cluster_lead_tier", pa.int8(), nullable=True),
        pa.field("dedup_weight", pa.float32()),
    ]
    schema_out = pa.schema(list(schema_in) + new_fields)
    col_order = [f.name for f in schema_out]

    writer = pq.ParquetWriter(V9, schema_out, compression="snappy")
    total = matched = non_leads = 0

    # Iterate in small chunks (not full row_groups)
    batch_iter = pf_in.iter_batches(batch_size=CHUNK)
    for b_idx, batch in enumerate(batch_iter):
        df = batch.to_pandas()
        n = len(df)

        # vectorized dict lookup
        eids = df["event_id"].to_numpy()
        cids = np.full(n, -1, dtype="int64")  # sentinel
        sizes = np.ones(n, dtype="int32")
        leads = np.ones(n, dtype=bool)  # default: ambient → treated as own lead
        ltiers = np.full(n, -1, dtype="int8")  # sentinel
        has_c = np.zeros(n, dtype=bool)

        for i in range(n):
            ent = lookup.get(eids[i])
            if ent is None:
                continue
            cids[i], sizes[i], leads[i], ltiers[i] = ent
            has_c[i] = True

        n_in = int(has_c.sum())
        matched += n_in
        n_nl = int((has_c & (~leads)).sum())
        non_leads += n_nl

        # nullable Int64/Int8
        cid_series = pd.Series(cids, dtype="Int64")
        cid_series[~has_c] = pd.NA
        ltier_series = pd.Series(ltiers, dtype="Int8")
        ltier_series[~has_c] = pd.NA

        df["event_cluster_id"] = cid_series
        df["cluster_size"] = sizes
        df["is_cluster_lead"] = leads
        df["cluster_lead_tier"] = ltier_series

        base_w = df["source_tier_weight"].to_numpy(dtype="float32")
        is_non_lead = has_c & (~leads)
        dedup_w = base_w.copy()
        dedup_w[is_non_lead] = 0.0
        df["dedup_weight"] = dedup_w.astype("float32")

        df = df[col_order]
        out_tbl = pa.Table.from_pandas(df, schema=schema_out, preserve_index=False)
        writer.write_table(out_tbl, row_group_size=CHUNK)

        total += n
        if (b_idx + 1) % 40 == 0:
            log.info("  batch %d  total=%d matched=%d non_lead=%d",
                     b_idx + 1, total, matched, non_leads)

        del df, out_tbl, cid_series, ltier_series
        if b_idx % 20 == 0:
            gc.collect()

    writer.close()
    log.info("DONE  %s", V9)
    log.info("  total           : %d", total)
    log.info("  cluster-matched : %d (%.2f%%)", matched, 100 * matched / total)
    log.info("  non-lead (w=0)  : %d (%.2f%%)", non_leads, 100 * non_leads / total)


if __name__ == "__main__":
    main()
