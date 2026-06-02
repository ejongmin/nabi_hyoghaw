"""
03b_build_tone_all44_v2.py
===========================
tone_monthly_zscore_all44_v2.parquet 공식 생성 스크립트.

구성:
  - 기존 18개 기업: data/processed/tone_monthly_zscore_18_v3.parquet
  - 신규 26개 기업: experiments/universe_44/data/processed/tone_new26_multilingual.parquet
    (영어 + 중국어 + 다국어 FinBERT 통합 버전)

우선순위: 18개 기업 데이터가 26개 기업 버전보다 우선 (18개 기업이 더 신뢰도 높음)

출력: experiments/universe_44/data/processed/tone_monthly_zscore_all44_v2.parquet
"""
from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── 경로 설정 ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]   # nabi_hyoghaw/
EXP  = ROOT / "experiments/universe_44"

PATH_TONE18  = ROOT / "data/processed/tone_monthly_zscore_18_v3.parquet"
PATH_TONE26  = EXP / "data/processed/tone_new26_multilingual.parquet"
OUT_PATH     = EXP / "data/processed/tone_monthly_zscore_all44_v2.parquet"

EXPECTED_TONE18_ROWS = 1727


def main() -> None:
    log.info("=== 03b_build_tone_all44_v2.py 시작 ===")

    # ── 입력 파일 검증 ─────────────────────────────────────────
    if not PATH_TONE18.exists():
        log.error("기존 18개 기업 tone 없음: %s", PATH_TONE18)
        sys.exit(1)
    if not PATH_TONE26.exists():
        log.error("신규 26개 기업 다국어 tone 없음: %s", PATH_TONE26)
        log.error("먼저 02b_merge_multilingual.py 를 실행하세요.")
        sys.exit(1)

    # ── STEP 1: tone18 읽기 ────────────────────────────────────
    tone18 = pd.read_parquet(PATH_TONE18)
    log.info("tone18 로드: %d행 | 기업 수: %d | 기간: %s ~ %s",
             len(tone18),
             tone18["company_id"].nunique(),
             tone18["year_month"].min(),
             tone18["year_month"].max())
    if len(tone18) != EXPECTED_TONE18_ROWS:
        log.warning("tone18 행 수 불일치: 기대=%d, 실제=%d",
                    EXPECTED_TONE18_ROWS, len(tone18))

    # ── STEP 2: tone26_multi 읽기 ──────────────────────────────
    tone26 = pd.read_parquet(PATH_TONE26)
    log.info("tone26_multi 로드: %d행 | 기업 수: %d | 기간: %s ~ %s",
             len(tone26),
             tone26["company_id"].nunique(),
             tone26["year_month"].min(),
             tone26["year_month"].max())

    # ── STEP 3: 18개 기업 ID 목록 추출 ────────────────────────
    cids_18 = set(tone18["company_id"].unique())
    log.info("18개 기업 ID: %d개 (%s ...)",
             len(cids_18), sorted(cids_18)[:3])

    # ── STEP 4: tone26에서 18개 기업 제거 (중복 방지) ─────────
    overlap = tone26[tone26["company_id"].isin(cids_18)]
    if len(overlap) > 0:
        log.warning("tone26에서 18개 기업 중복 %d행 제거 (tone18 우선)", len(overlap))
    tone26_new = tone26[~tone26["company_id"].isin(cids_18)].copy()
    log.info("tone26 중복 제거 후: %d행 (기업 %d개)",
             len(tone26_new), tone26_new["company_id"].nunique())

    # ── STEP 5: 공통 컬럼만 사용해서 concat ───────────────────
    cols18 = set(tone18.columns)
    cols26 = set(tone26_new.columns)
    common_cols = sorted(cols18 & cols26)
    only_18     = sorted(cols18 - cols26)
    only_26     = sorted(cols26 - cols18)

    log.info("공통 컬럼: %d개 | tone18 전용: %d개 | tone26 전용: %d개",
             len(common_cols), len(only_18), len(only_26))
    if only_18:
        log.info("  tone18 전용 컬럼 (제외): %s", only_18)
    if only_26:
        log.info("  tone26 전용 컬럼 (제외): %s", only_26)

    combined = pd.concat(
        [tone18[common_cols], tone26_new[common_cols]],
        ignore_index=True,
    )

    # ── STEP 6: company_id, year_month 기준 정렬 ──────────────
    combined = combined.sort_values(
        ["company_id", "year_month"]
    ).reset_index(drop=True)

    # 혹시 남은 (company_id, year_month) 중복 확인 및 제거 (tone18 우선이므로 이미 없어야 함)
    n_before = len(combined)
    combined = combined.drop_duplicates(
        subset=["company_id", "year_month"], keep="first"
    )
    if len(combined) < n_before:
        log.warning("추가 중복 제거: %d → %d행", n_before, len(combined))

    # ── STEP 7: 저장 ──────────────────────────────────────────
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUT_PATH, index=False)
    log.info("저장 완료: %s", OUT_PATH)

    # ── 저장 후 검증 ───────────────────────────────────────────
    log.info("\n%s", "=" * 60)
    log.info("■ 검증 1: 전체 행/기업 수")
    log.info("  총 행 수: %d", len(combined))
    log.info("  기업 수:  %d", combined["company_id"].nunique())
    log.info("  기간:     %s ~ %s",
             combined["year_month"].min(),
             combined["year_month"].max())

    log.info("\n■ 검증 2: 18개 기업 z_score_v3 일치 확인 (5개 기업 × 5개월)")
    if "z_score_v3" in tone18.columns and "z_score_v3" in combined.columns:
        sample_cids = sorted(cids_18)[:5]
        for cid in sample_cids:
            orig_rows = tone18[tone18["company_id"] == cid].set_index("year_month")
            new_rows  = combined[combined["company_id"] == cid].set_index("year_month")
            common_ym = sorted(set(orig_rows.index) & set(new_rows.index))[:5]
            mismatch_count = 0
            for ym in common_ym:
                v_orig = orig_rows.loc[ym, "z_score_v3"]
                v_new  = new_rows.loc[ym, "z_score_v3"]
                # NaN vs NaN은 일치로 처리
                both_nan = pd.isna(v_orig) and pd.isna(v_new)
                if not both_nan and not np.isclose(
                    float(v_orig) if not pd.isna(v_orig) else np.nan,
                    float(v_new)  if not pd.isna(v_new)  else np.nan,
                    equal_nan=True,
                ):
                    mismatch_count += 1
                    log.warning("    [불일치] %s %s: orig=%.4f new=%.4f",
                                cid, ym, v_orig, v_new)
            status = "OK" if mismatch_count == 0 else f"MISMATCH {mismatch_count}건"
            log.info("  %-18s  체크 %d개월 → %s", cid, len(common_ym), status)
    else:
        log.warning("  z_score_v3 컬럼 없음 — 검증 건너뜀")

    log.info("\n■ 검증 3: reliable_v3 분포")
    if "reliable_v3" in combined.columns:
        rel_total = int(combined["reliable_v3"].sum())
        rel_pct   = 100.0 * rel_total / len(combined)
        log.info("  reliable_v3=True: %d/%d행 (%.1f%%)",
                 rel_total, len(combined), rel_pct)

        # 18개 vs 26개 기업 분리 통계
        mask18 = combined["company_id"].isin(cids_18)
        rel18  = int(combined.loc[mask18,  "reliable_v3"].sum())
        rel26  = int(combined.loc[~mask18, "reliable_v3"].sum())
        n18    = mask18.sum()
        n26    = (~mask18).sum()
        log.info("  기존 18개 기업: reliable %d/%d (%.1f%%)",
                 rel18, n18, 100.0 * rel18 / max(1, n18))
        log.info("  신규 26개 기업: reliable %d/%d (%.1f%%)",
                 rel26, n26, 100.0 * rel26 / max(1, n26))
    else:
        log.warning("  reliable_v3 컬럼 없음 — 검증 건너뜀")

    log.info("\n%s", "=" * 60)
    log.info("=== 완료 ===")


if __name__ == "__main__":
    main()
