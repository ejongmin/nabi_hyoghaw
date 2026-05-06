#!/usr/bin/env python3
"""
build_v8.py
───────────
v7_final → v8: Phase 1 신뢰도/객관성 보강 컬럼 추가.

추가 컬럼 (원본 19컬럼은 그대로 유지):
  1) publisher           : title 접미사 " - Publisher"에서 추출
  2) source_tier         : 1/2/3  (configs/source_tiers.yaml 기반)
  3) source_tier_weight  : 1.0 / 0.5 / 0.15
  4) finbert_z           : 언어(finbert_model)별 z-score 정규화
  5) regime_covid        : 2020-03-01 ~ 2022-06-30
  6) regime_ukraine      : 2022-02-24 ~ (ongoing)
  7) regime_ira          : 2022-08-16 ~ (IRA 시행일)
  8) regime_china_export : 2023-01-01 ~ 2023-12-31  ★ 중국 EV 수출 급증기
  9) regime_post_covid   : 2022-07-01 ~ 2023-12-31  ★

v7_final은 read-only로 유지. 출력: data/processed/risk_events_classified_v8.parquet

Usage:
    python scripts/build_v8.py
    python scripts/build_v8.py --dry-run
"""
import argparse
import logging
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

log = logging.getLogger("build_v8")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ROOT = Path(__file__).resolve().parent.parent
V7 = ROOT / "data" / "processed" / "risk_events_classified_v7_final.parquet"
V8 = ROOT / "data" / "processed" / "risk_events_classified_v8.parquet"
TIERS = ROOT / "configs" / "source_tiers.yaml"

# Regime 날짜 경계 (추론 포함 ★)
# 근거:
#   COVID   : WHO PHEIC 2020-01-30, 주요 lockdown 2020-03, Omicron 하락 2022-06
#   Ukraine : 2022-02-24 (Russian invasion start) — ongoing
#   IRA     : 2022-08-16 (signed by Biden)
#   China export : 2022 Q4 ~ 2023 Q4 중국 EV/BEV 수출 급증 (CPCA, SAIC 월간 데이터 기준) ★
#   Post-COVID  : 2022-07-01 이후 경제 정상화 시작 ★
REGIMES = {
    "regime_covid":        (datetime(2020, 3,  1), datetime(2022, 6, 30)),
    "regime_ukraine":      (datetime(2022, 2, 24), None),
    "regime_ira":          (datetime(2022, 8, 16), None),
    "regime_china_export": (datetime(2023, 1,  1), datetime(2023, 12, 31)),
    "regime_post_covid":   (datetime(2022, 7,  1), datetime(2023, 12, 31)),
}

SUFFIX_RE = re.compile(r"\s*[-–—]\s*([^-–—]+?)\s*$")


def extract_publisher(title):
    if not title or not isinstance(title, str):
        return ""
    m = SUFFIX_RE.search(title.strip())
    if not m:
        return ""
    pub = m.group(1).strip()
    if len(pub) < 2 or len(pub) > 80:
        return ""
    return pub


def build_tier_lookup() -> tuple[dict, dict]:
    with open(TIERS, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    weights = cfg["weights"]
    lookup = {}
    for name in cfg.get("tier_1", []):
        lookup[name.strip().lower()] = 1
    for name in cfg.get("tier_2", []):
        lookup[name.strip().lower()] = 2
    log.info(f"  tier_1: {sum(1 for v in lookup.values() if v==1)}  publishers")
    log.info(f"  tier_2: {sum(1 for v in lookup.values() if v==2)}  publishers")
    return lookup, weights


def assign_tier(pub: str, lookup: dict) -> int:
    if not pub:
        return 3
    return lookup.get(pub.strip().lower(), 3)


def compute_regime_mask(times: pd.Series, start, end) -> np.ndarray:
    t = pd.to_datetime(times, errors="coerce")
    mask = t >= start
    if end is not None:
        mask &= t <= end
    return mask.fillna(False).to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Only process first row_group")
    args = ap.parse_args()

    if not V7.exists():
        log.error(f"v7_final not found: {V7}")
        return

    log.info("[1] Loading tier lookup from source_tiers.yaml")
    tier_lookup, weights = build_tier_lookup()
    weights_arr = {1: weights[1], 2: weights[2], 3: weights[3]}

    log.info(f"[2] Streaming {V7.name}")
    pf = pq.ParquetFile(V7)
    orig_schema = pf.schema_arrow
    log.info(f"  num_row_groups: {pf.num_row_groups}")
    log.info(f"  orig columns  : {len(orig_schema.names)}")

    # 새 컬럼 스키마
    new_fields = [
        pa.field("publisher",          pa.string()),
        pa.field("source_tier",        pa.int8()),
        pa.field("source_tier_weight", pa.float32()),
        pa.field("finbert_z",          pa.float32()),
        pa.field("regime_covid",       pa.bool_()),
        pa.field("regime_ukraine",     pa.bool_()),
        pa.field("regime_ira",         pa.bool_()),
        pa.field("regime_china_export", pa.bool_()),
        pa.field("regime_post_covid",  pa.bool_()),
    ]
    out_schema = pa.schema(list(orig_schema) + new_fields)

    # ─── Pass 1: 언어별(finbert_model별) μ, σ 수집 ──────────────────
    log.info("[3] Pass 1: computing per-model (language) mean/std for z-score")
    # 메모리 아끼려고 Welford 없이 2-pass: row_group 별로 누적
    model_sum = {}
    model_sqsum = {}
    model_cnt = {}

    n_rg = pf.num_row_groups
    rg_range = range(min(1, n_rg) if args.dry_run else n_rg)
    for rgi in rg_range:
        tbl = pf.read_row_group(rgi, columns=["finbert_model", "finbert_score"])
        df = tbl.to_pandas()
        df = df[df["finbert_score"].notna()]
        if df.empty:
            continue
        for model, grp in df.groupby("finbert_model"):
            s = grp["finbert_score"].to_numpy(dtype=np.float64)
            model_sum[model] = model_sum.get(model, 0.0) + s.sum()
            model_sqsum[model] = model_sqsum.get(model, 0.0) + (s * s).sum()
            model_cnt[model] = model_cnt.get(model, 0) + len(s)

    model_stats = {}
    for m in sorted(model_cnt.keys()):
        n = model_cnt[m]
        mu = model_sum[m] / n
        var = model_sqsum[m] / n - mu * mu
        sd = float(np.sqrt(max(var, 1e-12)))
        model_stats[m] = (mu, sd)
        log.info(f"  {m:<60s}  n={n:>8,}  μ={mu:+.4f}  σ={sd:.4f}")

    # ─── Pass 2: streaming rewrite with added columns ───────────────
    log.info("[4] Pass 2: rewriting with added columns")
    writer = None
    stats = {"total": 0, "tier1": 0, "tier2": 0, "tier3": 0, "pub_extracted": 0}

    try:
        for rgi in rg_range:
            tbl = pf.read_row_group(rgi)
            df = tbl.to_pandas()
            n_rows = len(df)
            stats["total"] += n_rows

            # publisher 추출
            pub = df["title"].apply(extract_publisher)
            mask_no_pub = (pub == "") & (df["source_domain"] != "news.google.com") & df["source_domain"].notna()
            pub = pub.where(~mask_no_pub, df["source_domain"])
            stats["pub_extracted"] += int((pub != "").sum())

            # tier
            tier = pub.apply(lambda p: assign_tier(p, tier_lookup)).astype(np.int8)
            tier_w = tier.map(weights_arr).astype(np.float32)
            stats["tier1"] += int((tier == 1).sum())
            stats["tier2"] += int((tier == 2).sum())
            stats["tier3"] += int((tier == 3).sum())

            # finbert_z: 모델별 z-score
            z = np.full(n_rows, np.nan, dtype=np.float32)
            has_fb = df["finbert_score"].notna() & df["finbert_model"].notna()
            if has_fb.any():
                for model, (mu, sd) in model_stats.items():
                    m_mask = has_fb & (df["finbert_model"] == model)
                    if m_mask.any():
                        raw = df.loc[m_mask, "finbert_score"].to_numpy(dtype=np.float64)
                        z[m_mask.to_numpy()] = ((raw - mu) / sd).astype(np.float32)

            # regime flags
            regime_cols = {}
            for col, (start, end) in REGIMES.items():
                regime_cols[col] = compute_regime_mask(df["event_time"], start, end)

            # attach
            df_out = df.copy()
            df_out["publisher"] = pub.astype(str).fillna("")
            df_out["source_tier"] = tier
            df_out["source_tier_weight"] = tier_w
            df_out["finbert_z"] = z
            for col in regime_cols:
                df_out[col] = regime_cols[col]

            new_tbl = pa.Table.from_pandas(df_out, schema=out_schema, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(V8, out_schema, compression="zstd")
            writer.write_table(new_tbl)

            if (rgi + 1) % 10 == 0 or rgi + 1 == n_rg:
                log.info(
                    f"  row_group {rgi+1}/{n_rg}  "
                    f"total={stats['total']:,}  "
                    f"T1={stats['tier1']:,}  T2={stats['tier2']:,}  T3={stats['tier3']:,}"
                )
    finally:
        if writer is not None:
            writer.close()

    log.info(f"\n[5] Summary")
    log.info(f"  ✅ {V8.name}")
    log.info(f"     total rows         : {stats['total']:,}")
    log.info(f"     publishers extracted: {stats['pub_extracted']:,}")
    log.info(f"     tier 1             : {stats['tier1']:,}  ({100*stats['tier1']/max(stats['total'],1):.2f}%)")
    log.info(f"     tier 2             : {stats['tier2']:,}  ({100*stats['tier2']/max(stats['total'],1):.2f}%)")
    log.info(f"     tier 3             : {stats['tier3']:,}  ({100*stats['tier3']/max(stats['total'],1):.2f}%)")
    if V8.exists():
        log.info(f"     size               : {V8.stat().st_size / 1024 / 1024:.0f} MB")


if __name__ == "__main__":
    main()
