#!/usr/bin/env python3
"""
BigQuery Parquet → risk_events 변환 스크립트.
로컬 터미널에서 실행:  python run_bq_bridge.py

메모리 요구: ~2GB (PyArrow 배치 처리)
소요 시간: ~10~20분 (4개 파일, 490만행)
"""
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

import pandas as pd
import yaml

from src.ingest.gdelt_bq import ingest_gdelt_bq


def main():
    # Config
    with open("configs/base.yaml") as f:
        cfg = yaml.safe_load(f)

    # Universe
    uni = pd.read_csv("data/universe/univers_final.csv")
    print(f"Universe: {len(uni)} companies")

    # Output paths
    bq_out = Path("data/processed/risk_events_bq.parquet")
    main_out = Path("data/processed/risk_events.parquet")

    # Run bridge (returns Path, not DataFrame)
    import shutil
    result_path = ingest_gdelt_bq(cfg, uni, bq_out, force=True)

    # Copy to main pipeline path
    shutil.copy2(str(bq_out), str(main_out))

    # Summary (메모리 효율적: 청크로 통계)
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(bq_out)
    total_rows = pf.metadata.num_rows

    print(f"\n{'='*60}")
    print(f"COMPLETE")
    print(f"   risk_events_bq.parquet: {total_rows:,} rows")
    print(f"   risk_events.parquet: copied (pipeline ready)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
