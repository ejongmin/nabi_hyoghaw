#!/usr/bin/env python3
"""
merge_zero8_finbert.py
──────────────────────
Colab에서 생성한 4개 finbert_results_zero8_{en,ja,de,es}.parquet을
v7 parquet의 finbert_* 컬럼에 채워 넣어 v7_final 생성.

매칭 key: title (언어별 unique, rep_event_id 기준으로 v7 모든 zero8 행에 broadcast)

  → 즉, Colab이 언어별 unique title에 대해 돌았기 때문에
    동일 title을 가진 여러 zero8 row에 같은 score를 복제해서 채움.

출력:
    data/processed/risk_events_classified_v7_final.parquet

Usage:
    python scripts/merge_zero8_finbert.py
    python scripts/merge_zero8_finbert.py --dry-run
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger("merge_zero8_finbert")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ROOT = Path(__file__).resolve().parent.parent
V7 = ROOT / "data" / "processed" / "risk_events_classified_v7.parquet"
V7_FINAL = ROOT / "data" / "processed" / "risk_events_classified_v7_final.parquet"
PROC = ROOT / "data" / "processed"
LANGS = ["en", "ja", "de", "es"]


def load_finbert_results() -> pd.DataFrame:
    """4개 언어 finbert 결과 concat. 컬럼: title, finbert_neg/neu/pos/score/model"""
    frames = []
    for lang in LANGS:
        p = PROC / f"finbert_results_zero8_{lang}.parquet"
        if not p.exists():
            log.warning(f"  missing: {p.name}, skipping")
            continue
        df = pd.read_parquet(p)
        # 컬럼 표준화
        if "finbert_model" not in df.columns:
            df["finbert_model"] = f"zero8_{lang}"
        df["_lang"] = lang
        log.info(f"  {lang}: {len(df):,} rows  (model={df['finbert_model'].iloc[0] if len(df) > 0 else '?'})")
        frames.append(df[["title", "finbert_neg", "finbert_neu", "finbert_pos", "finbert_score", "finbert_model", "_lang"]])
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # title별 unique (만에 하나 언어간 겹침 있으면 첫 언어 우선)
    out = out.drop_duplicates(subset=["title"], keep="first").reset_index(drop=True)
    log.info(f"  merged unique titles: {len(out):,}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not V7.exists():
        log.error(f"v7 not found: {V7}")
        return

    log.info(f"[1] Loading FinBERT results ({len(LANGS)} languages)")
    fb = load_finbert_results()
    if fb.empty:
        log.error("No FinBERT result files found!")
        return

    # title → finbert_* 룩업 (pandas merge용)
    fb_lookup = fb.set_index("title")[["finbert_neg", "finbert_neu", "finbert_pos", "finbert_score", "finbert_model"]]
    log.info(f"  lookup size: {len(fb_lookup):,} unique titles")

    log.info(f"\n[2] Streaming v7 {V7.name}")
    pf = pq.ParquetFile(V7)
    schema = pf.schema_arrow
    log.info(f"  num_row_groups: {pf.num_row_groups}")

    if args.dry_run:
        # 처음 한 row_group만 확인
        tbl = pf.read_row_group(0)
        df0 = tbl.to_pandas()
        zero8_mask = df0["classify_source"] == "google_news_zero8"
        n_z = zero8_mask.sum()
        log.info(f"  [dry-run] row_group 0: {len(df0):,} rows, zero8={n_z:,}")
        if n_z > 0:
            hit = df0.loc[zero8_mask, "title"].isin(fb_lookup.index).sum()
            log.info(f"  [dry-run] zero8 title hit rate: {hit}/{n_z} ({100*hit/max(n_z,1):.1f}%)")
        return

    writer = None
    total_rg = pf.num_row_groups
    stats = {"updated": 0, "zero8_total": 0}

    try:
        for rgi in range(total_rg):
            tbl = pf.read_row_group(rgi)
            df = tbl.to_pandas()

            mask = (df["classify_source"] == "google_news_zero8")
            n_z = int(mask.sum())
            stats["zero8_total"] += n_z

            if n_z > 0:
                # title 기반 lookup
                titles = df.loc[mask, "title"]
                matched = titles.map(fb_lookup["finbert_neg"])
                ok = matched.notna()
                idx = titles.index[ok]
                if len(idx) > 0:
                    df.loc[idx, "finbert_neg"] = titles.loc[idx].map(fb_lookup["finbert_neg"]).astype(np.float32)
                    df.loc[idx, "finbert_neu"] = titles.loc[idx].map(fb_lookup["finbert_neu"]).astype(np.float32)
                    df.loc[idx, "finbert_pos"] = titles.loc[idx].map(fb_lookup["finbert_pos"]).astype(np.float32)
                    df.loc[idx, "finbert_score"] = titles.loc[idx].map(fb_lookup["finbert_score"]).astype(np.float32)
                    df.loc[idx, "finbert_model"] = titles.loc[idx].map(fb_lookup["finbert_model"]).astype(str)
                    stats["updated"] += len(idx)

            new_tbl = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(V7_FINAL, schema, compression="zstd")
            writer.write_table(new_tbl)

            if (rgi + 1) % 10 == 0 or rgi + 1 == total_rg:
                log.info(f"  row_group {rgi+1}/{total_rg}  updated={stats['updated']:,}  zero8={stats['zero8_total']:,}")
    finally:
        if writer is not None:
            writer.close()

    log.info(f"\n  ✅ {V7_FINAL.name}")
    log.info(f"     zero8 total rows    : {stats['zero8_total']:,}")
    log.info(f"     finbert updated rows: {stats['updated']:,}  ({100*stats['updated']/max(stats['zero8_total'],1):.1f}%)")

    size_mb = V7_FINAL.stat().st_size / 1024 / 1024
    log.info(f"     size                : {size_mb:.0f} MB")


if __name__ == "__main__":
    main()
