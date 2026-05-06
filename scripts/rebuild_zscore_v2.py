"""
tone_monthly_zscore 개선판 (v2) 생성
======================================

현행 v1의 한계점 4가지를 동시에 해결:

  [한계 1] 공급망 무관 기사 포함
    → 개선: supply_z (natural/logistics/geopolitics 특화) 추가
    → 일반 감성(general_z)과 분리해서 두 신호 모두 제공

  [한계 2] z_source 불일치 (finbert vs v2tone 혼용)
    → 개선: finbert_z_strict (n_fb >= 10인 달만 사용)
    → v2tone은 보조 fallback으로만 사용, z_source_v2 명시

  [한계 3] reliable 기준 너무 관대 (n_tone >= 10)
    → 개선: reliable_v2 (finbert 기준 n_fb >= 10 OR tone 기준 n_tone >= 20)
    → CNGR처럼 데이터 희박 기업은 reliable_v2=False 더 많이 받음

  [한계 4] 횡단면(Cross-sectional) 정보 부재
    → 개선: z_cross (같은 달 18개 기업 내에서 상대적 위치)
    → "이번 달 CATL이 다른 기업들에 비해 얼마나 부정적인가"

출력:
  data/processed/tone_monthly_zscore_18_v2.parquet
  reports/zscore_v2_improvement_report.txt
"""
from __future__ import annotations
import logging, sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

ROOT   = Path(__file__).resolve().parents[1]
V1     = ROOT / "data/processed/tone_monthly_zscore_18.parquet"
V9     = ROOT / "data/processed/risk_events_classified_v9.parquet"
C18    = ROOT / "data/universe/companies_18.csv"
OUT    = ROOT / "data/processed/tone_monthly_zscore_18_v2.parquet"
REPORT = ROOT / "reports/zscore_v2_improvement_report.txt"

FINAL_18 = ['TSLA','BMW.DE','VOW3.DE','MBG.DE','7267.T','1211.HK',
            '300750.SZ','373220.KS','3931.HK','300014.SZ','002074.SZ',
            '051910.KS','BAS.DE','247540.KQ','300919.SZ','ALB','GLEN.L','603799.SH']

# ── 개선된 임계값 ──
FB_STRICT   = 10   # finbert_z_strict: 이 이상일 때만 finbert 사용
TONE_STRICT = 20   # reliable_v2 fallback 기준
SHOCK_THR   = 2.0  # neg/pos shock 임계값


# ─────────────────────────────────────────────────────────────
# Step 1: v9에서 supply-chain 특화 감성 집계
# ─────────────────────────────────────────────────────────────

def build_supply_signal(v9_path: Path) -> pd.DataFrame:
    """
    v9에서 공급망 특화 리스크 유형(natural, logistics, geopolitics) 기사만 필터링
    → 기업 × 월별 가중 평균 tone 계산

    공급망 특화 기사: "공장 화재", "광산 파업", "항만 폐쇄" 등 직접적 공급 충격
    geopolitics 포함 이유: 무역 제재, 광물 수출 통제 등 공급망 직접 영향
    """
    log.info("v9에서 공급망 특화 신호 추출 중...")

    SUPPLY_TYPES = {"natural", "logistics", "geopolitics"}

    pf = pq.ParquetFile(v9_path)
    batches = []

    for batch in pf.iter_batches(
        batch_size=300_000,
        columns=["company_id", "event_time", "risk_types",
                 "tone", "source_tier_weight", "finbert_score"]
    ):
        b = batch.to_pandas()
        b = b[b["company_id"].isin(FINAL_18)].copy()
        if len(b) == 0:
            continue

        # 공급망 특화 리스크 유형 필터
        def is_supply_specific(rt: str) -> bool:
            if not isinstance(rt, str):
                return False
            types = set(t.strip() for t in rt.split(","))
            # other만 있는 경우 제외, supply 관련 타입 하나 이상 포함
            return bool(types & SUPPLY_TYPES) and types != {"other"}

        b["is_supply"] = b["risk_types"].apply(is_supply_specific)
        b = b[b["is_supply"]].copy()

        if len(b) == 0:
            continue

        b["event_time"] = pd.to_datetime(b["event_time"], errors="coerce")
        b = b.dropna(subset=["event_time"])
        b["ym"] = b["event_time"].dt.to_period("M")
        b["weight"] = b["source_tier_weight"].fillna(1.0)
        b["tone_w"] = b["tone"].fillna(0) * b["weight"]
        b["tone_valid"] = b["tone"].notna().astype(int)
        b["fb_valid"] = b["finbert_score"].notna().astype(int)
        b["fb_w"] = b["finbert_score"].fillna(0) * b["weight"]

        batches.append(b[["company_id", "ym", "tone_w", "weight",
                           "tone_valid", "fb_valid", "fb_w"]])

    if not batches:
        log.warning("공급망 특화 이벤트 없음")
        return pd.DataFrame()

    df = pd.concat(batches, ignore_index=True)

    agg = df.groupby(["company_id", "ym"]).agg(
        supply_tone_wsum=("tone_w",     "sum"),
        supply_weight=   ("weight",     "sum"),
        supply_n_tone=   ("tone_valid", "sum"),
        supply_n_fb=     ("fb_valid",   "sum"),
        supply_fb_wsum=  ("fb_w",       "sum"),
    ).reset_index()

    agg["supply_tone_mean"] = np.where(
        agg["supply_weight"] > 0,
        agg["supply_tone_wsum"] / agg["supply_weight"], np.nan
    )
    agg["supply_fb_mean"] = np.where(
        agg["supply_n_fb"] > 0,
        agg["supply_fb_wsum"] / agg["supply_n_fb"].replace(0, np.nan), np.nan
    )

    log.info("  공급망 특화 집계: %d기업-월 행", len(agg))
    return agg.drop(columns=["supply_tone_wsum", "supply_weight",
                              "supply_fb_wsum"])


# ─────────────────────────────────────────────────────────────
# Step 2: per-company z-score 계산 (strict 기준 적용)
# ─────────────────────────────────────────────────────────────

def per_company_z(series: pd.Series, min_obs: int = 12) -> pd.Series:
    """유효 관측치 min_obs 이상인 경우에만 z-score 계산"""
    valid = series.dropna()
    if len(valid) < min_obs:
        return pd.Series(np.nan, index=series.index)
    mu  = valid.mean()
    sig = valid.std(ddof=1)
    if sig < 1e-10:
        return pd.Series(0.0, index=series.index)
    return (series - mu) / sig


def add_zscore_columns(panel: pd.DataFrame) -> pd.DataFrame:
    """
    v1 컬럼을 유지하면서 개선된 컬럼 추가:
    - finbert_z_strict: n_fb >= 10인 달만 사용한 finbert z-score
    - z_source_v2: 엄격한 우선순위 (finbert_strict > tone_strict > NaN)
    - z_score_v2: 개선된 z-score
    - reliable_v2: 더 엄격한 신뢰성 플래그
    - neg_shock_v2, pos_shock_v2: 개선된 shock 플래그
    """
    panel = panel.sort_values(["company_id", "ym"]).copy()
    rows = []

    for cid, g in panel.groupby("company_id"):
        g = g.copy()

        # ── finbert_z_strict: n_fb >= FB_STRICT인 달만 ──
        fb_mask = g["n_fb_total"] >= FB_STRICT
        g["fb_strict_raw"] = np.where(fb_mask, g["fb_z_combined"], np.nan)
        g["finbert_z_strict"] = per_company_z(g["fb_strict_raw"])

        # ── tone_z_strict: n_tone >= TONE_STRICT인 달만 ──
        tone_mask = g["n_tone"] >= TONE_STRICT
        g["tone_strict_raw"] = np.where(tone_mask, g["tone_weighted_mean"], np.nan)
        g["tone_z_strict"] = per_company_z(g["tone_strict_raw"])

        # ── z_score_v2: finbert_strict 우선, 없으면 tone_strict ──
        g["z_score_v2"] = g["finbert_z_strict"]
        g["z_source_v2"] = np.where(g["finbert_z_strict"].notna(),
                                     "finbert_strict", "v2tone_strict")
        fallback_mask = g["finbert_z_strict"].isna() & g["tone_z_strict"].notna()
        g.loc[fallback_mask, "z_score_v2"] = g.loc[fallback_mask, "tone_z_strict"]

        # ── reliable_v2: finbert >= 10 OR tone >= 20 ──
        g["reliable_v2"] = fb_mask | tone_mask

        # ── lag / delta / rolling (v2 기준) ──
        g["z_lag1_v2"]    = g["z_score_v2"].shift(1)
        g["z_lag2_v2"]    = g["z_score_v2"].shift(2)
        g["delta_z_v2"]   = g["z_score_v2"] - g["z_lag1_v2"]
        g["rolling_3m_v2"]= g["z_score_v2"].rolling(3, min_periods=2).mean()

        # ── shock flags v2 ──
        g["neg_shock_v2"] = (g["z_score_v2"] < -SHOCK_THR) & g["reliable_v2"] & g["z_score_v2"].notna()
        g["pos_shock_v2"] = (g["z_score_v2"] > SHOCK_THR)  & g["reliable_v2"] & g["z_score_v2"].notna()

        rows.append(g)

    return pd.concat(rows, ignore_index=True)


def add_cross_sectional_z(panel: pd.DataFrame) -> pd.DataFrame:
    """
    횡단면 z-score 추가:
    같은 달 18개 기업 중 상대적 위치
    → "이번 달 CATL이 다른 기업들에 비해 얼마나 부정적인가"
    → 공급망 레이어 내 비교에 더 적합
    """
    panel = panel.copy()

    # 전체 18개 기업 기준 횡단면 z
    def cross_z(group):
        vals = group["z_score_v2"].dropna()
        if len(vals) < 5:
            group["z_cross_all"] = np.nan
            return group
        mu  = vals.mean()
        sig = vals.std(ddof=1)
        if sig < 1e-10:
            group["z_cross_all"] = 0.0
        else:
            group["z_cross_all"] = (group["z_score_v2"] - mu) / sig
        return group

    panel = panel.groupby("ym", group_keys=False).apply(cross_z)

    # 동일 레이어 내 횡단면 z (같은 Cell/Pack끼리, OEM끼리 비교)
    c18 = pd.read_csv(C18)
    stage_map = dict(zip(c18["company_id"], c18["stage_bucket"]))
    panel["stage"] = panel["company_id"].map(stage_map)

    def cross_z_stage(group):
        vals = group["z_score_v2"].dropna()
        if len(vals) < 3:
            group["z_cross_stage"] = np.nan
            return group
        mu  = vals.mean()
        sig = vals.std(ddof=1)
        if sig < 1e-10:
            group["z_cross_stage"] = 0.0
        else:
            group["z_cross_stage"] = (group["z_score_v2"] - mu) / sig
        return group

    panel = panel.groupby(["ym", "stage"], group_keys=False).apply(cross_z_stage)

    return panel.drop(columns=["stage"])


def add_supply_signal_columns(panel: pd.DataFrame,
                               supply_agg: pd.DataFrame) -> pd.DataFrame:
    """
    공급망 특화 z-score 추가:
    - supply_z: natural/logistics/geopolitics 기사 기반 z-score
    - neg_shock_supply / pos_shock_supply
    """
    if supply_agg.empty:
        panel["supply_z"] = np.nan
        panel["neg_shock_supply"] = False
        panel["pos_shock_supply"] = False
        return panel

    panel = panel.merge(
        supply_agg[["company_id", "ym", "supply_tone_mean",
                    "supply_n_tone", "supply_n_fb"]],
        on=["company_id", "ym"], how="left"
    )

    # supply_z: per-company z of supply_tone_mean
    rows = []
    for cid, g in panel.groupby("company_id"):
        g = g.copy()
        g["supply_z"] = per_company_z(g["supply_tone_mean"], min_obs=6)
        rows.append(g)
    panel = pd.concat(rows, ignore_index=True)

    panel["neg_shock_supply"] = (
        (panel["supply_z"] < -SHOCK_THR) &
        (panel["supply_n_tone"].fillna(0) >= 5) &
        panel["supply_z"].notna()
    )
    panel["pos_shock_supply"] = (
        (panel["supply_z"] > SHOCK_THR) &
        (panel["supply_n_tone"].fillna(0) >= 5) &
        panel["supply_z"].notna()
    )

    return panel


# ─────────────────────────────────────────────────────────────
# Step 3: 개선 효과 리포트
# ─────────────────────────────────────────────────────────────

def build_report(v1: pd.DataFrame, v2: pd.DataFrame) -> str:
    c18 = pd.read_csv(C18)
    nm  = dict(zip(c18["company_id"], c18["canonical_name"]))
    lines = ["=" * 72,
             "z-score v2 개선 리포트",
             "=" * 72, ""]

    lines += ["■ 핵심 변경사항",
              "  1. finbert_z_strict: n_fb >= 10인 달만 사용 (이전: 기준 없음)",
              "  2. reliable_v2: n_fb>=10 OR n_tone>=20 (이전: n_tone>=10)",
              "  3. z_cross_all: 전체 18개 기업 내 횡단면 z-score (신규)",
              "  4. z_cross_stage: 동일 레이어 내 횡단면 z-score (신규)",
              "  5. supply_z: natural/logistics/geopolitics 특화 신호 (신규)",
              ""]

    lines.append("■ reliable 월 수 비교 (v1 vs v2)")
    lines.append(f"  {'기업명':<25} {'v1 reliable':>12} {'v2 reliable':>12} {'변화':>8}")
    lines.append("  " + "-" * 60)
    for cid in FINAL_18:
        n = nm.get(cid, cid)
        r1 = int(v1[v1["company_id"] == cid]["reliable"].sum())
        r2 = int(v2[v2["company_id"] == cid]["reliable_v2"].sum())
        delta = r2 - r1
        flag = " ↓" if delta < -5 else (" ↑" if delta > 5 else "")
        lines.append(f"  {n:<25} {r1:>12} {r2:>12} {delta:>+7}{flag}")

    lines += [""]
    lines.append("■ neg/pos shock 수 비교 (v1 vs v2)")
    neg_v1 = int(v1["neg_shock"].sum())
    pos_v1 = int(v1["pos_shock"].sum())
    neg_v2 = int(v2["neg_shock_v2"].sum())
    pos_v2 = int(v2["pos_shock_v2"].sum())
    neg_sup = int(v2["neg_shock_supply"].sum()) if "neg_shock_supply" in v2.columns else 0
    pos_sup = int(v2["pos_shock_supply"].sum()) if "pos_shock_supply" in v2.columns else 0
    lines.append(f"  v1 neg={neg_v1}, pos={pos_v1}  →  "
                 f"v2 neg={neg_v2}, pos={pos_v2}  |  "
                 f"supply neg={neg_sup}, pos={pos_sup}")

    lines += [""]
    lines.append("■ supply_z 충격 발생 기업-월 (공급망 특화)")
    if "neg_shock_supply" in v2.columns:
        sup_events = v2[v2["neg_shock_supply"] | v2["pos_shock_supply"]].copy()
        for _, r in sup_events.sort_values("company_id").iterrows():
            typ = "NEG" if r["neg_shock_supply"] else "POS"
            z   = r.get("supply_z", np.nan)
            ym_str = r.get("year_month", r.get("ym", "?"))
            lines.append(f"  {nm.get(r['company_id'],'?'):25s} | {ym_str} | "
                         f"{typ} | supply_z={z:+.2f}")
    lines += ["", "=" * 72]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    log.info("z-score v2 재구축 시작")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    # v1 로드
    v1 = pd.read_parquet(V1)
    v1["ym"] = pd.to_datetime(v1["year_month"]).dt.to_period("M")
    bool_cols = ["regime_covid","regime_ukraine","regime_ira","regime_china_export"]
    for c in bool_cols:
        if c in v1.columns and v1[c].dtype == bool:
            v1[c] = v1[c].astype("float32")

    log.info("v1 로드: %d행", len(v1))

    # Step 1: 공급망 특화 신호
    supply_agg = build_supply_signal(V9)

    # Step 2: strict z-score + reliable_v2
    log.info("strict z-score 계산 중...")
    v2 = add_zscore_columns(v1)

    # Step 3: 횡단면 z-score
    log.info("횡단면 z-score 계산 중...")
    v2 = add_cross_sectional_z(v2)

    # Step 4: supply z-score 추가
    log.info("공급망 특화 z-score 추가 중...")
    v2 = add_supply_signal_columns(v2, supply_agg)

    # ym → str로 되돌리기 (저장 형식 유지)
    v2["year_month"] = v2["ym"].dt.strftime("%Y-%m")
    v2 = v2.drop(columns=["ym"])
    if "supply_tone_mean" in v2.columns:
        pass  # 유지

    # 저장
    v2.to_parquet(OUT, index=False)
    log.info("저장: %s (%d행, %d컬럼)", OUT, len(v2), len(v2.columns))

    # 리포트
    v1["ym"] = pd.to_datetime(v1["year_month"]).dt.to_period("M")
    report = build_report(v1, v2)
    REPORT.write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
