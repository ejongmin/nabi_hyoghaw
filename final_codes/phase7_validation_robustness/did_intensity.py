"""
did_intensity.py
================
Phase 4 CAR 데이터로 충격 강도(z_score)별 CAAR 분석 및 DID 검정

입력:
  - data/processed/car_panel_phase4.parquet
  - data/processed/tone_monthly_zscore_18_v3.parquet  (참조용)

출력:
  - experiments/hypothesis/walk_forward_cv/reports/did_intensity.txt
"""
from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

ROOT    = Path("C:/Users/john9/nabi_hyoghaw")
HYP     = ROOT / "experiments/hypothesis/walk_forward_cv"
OUT_RPT = HYP / "reports/did_intensity.txt"

PATH_CAR  = ROOT / "data/processed/car_panel_phase4.parquet"
PATH_TONE = ROOT / "data/processed/tone_monthly_zscore_18_v3.parquet"


# ─────────────────────────────────────────────────────────────
# 1. 데이터 로딩 및 필터링
# ─────────────────────────────────────────────────────────────

def load_and_filter(path: Path) -> pd.DataFrame:
    """
    분석 대상: window_label=='0_21', quality=='full',
               confounding_self==False, relationship=='DIRECT'
    """
    df = pd.read_parquet(path)
    print(f"[로딩] 전체 행 수: {len(df)}")
    print(f"  컬럼: {df.columns.tolist()}")

    # relationship 컬럼 이름 확인 (rel_type 또는 relationship)
    rel_col = None
    for c in ["relationship", "rel_type"]:
        if c in df.columns:
            rel_col = c
            break

    if rel_col is None:
        raise ValueError(f"DIRECT 관계 컬럼을 찾을 수 없음. 컬럼 목록: {df.columns.tolist()}")

    # 필터 적용
    mask = (
        (df["window_label"] == "0_21") &
        (df["quality"] == "full") &
        (df["confounding_self"] == False) &
        (df[rel_col] == "DIRECT")
    )
    filtered = df[mask].copy()
    print(f"  필터 후 (window=0_21, quality=full, no_confound, DIRECT): {len(filtered)}")

    if len(filtered) == 0:
        # 디버그: 각 필터 조건별 카운트
        print("\n  [디버그] 각 조건별 통과 수:")
        print(f"    window_label=='0_21': {(df['window_label']=='0_21').sum()}")
        print(f"    quality=='full': {(df['quality']=='full').sum()}")
        print(f"    confounding_self==False: {(df['confounding_self']==False).sum()}")
        print(f"    {rel_col}=='DIRECT': {(df[rel_col]=='DIRECT').sum()}")
        print(f"\n  window_label 유니크: {df['window_label'].unique()}")
        print(f"  quality 유니크: {df['quality'].unique()}")
        print(f"  {rel_col} 유니크: {df[rel_col].unique()}")
        raise ValueError("필터 조건 후 데이터 없음 — 위 디버그 정보 확인 필요")

    return filtered


# ─────────────────────────────────────────────────────────────
# 2. 그룹별 CAAR 통계 계산
# ─────────────────────────────────────────────────────────────

def caar_stats(sub: pd.DataFrame, label: str) -> dict:
    """
    sub: 해당 그룹 DataFrame (CAR, t_stat_patell 컬럼 포함)
    반환: N, CAAR, t, p, CI_lo, CI_hi
    """
    n = len(sub)
    if n < 2:
        return dict(label=label, N=n, CAAR=np.nan, t=np.nan, p=np.nan,
                    CI_lo=np.nan, CI_hi=np.nan)

    cars   = sub["CAR"].values
    caar   = cars.mean()

    # Patell t-stat: cross-sectional mean of individual t-stats
    # (표준 이벤트 스터디 관행: t = mean(t_patell) / sqrt(1/N) × sqrt(N) = sum/sqrt(N))
    # 여기서는 mean(CAR) / (std(CAR)/sqrt(N)) 방식 사용 (표본 t-검정)
    se   = cars.std(ddof=1) / np.sqrt(n)
    t_cs = caar / se if se > 0 else np.nan

    # p-value (양측, df=N-1)
    if not np.isnan(t_cs):
        p_val = 2 * stats.t.sf(abs(t_cs), df=n - 1)
    else:
        p_val = np.nan

    # 95% CI
    t_crit = stats.t.ppf(0.975, df=n - 1)
    ci_lo  = caar - t_crit * se
    ci_hi  = caar + t_crit * se

    return dict(label=label, N=n, CAAR=caar, t=t_cs, p=p_val,
                CI_lo=ci_lo, CI_hi=ci_hi)


# ─────────────────────────────────────────────────────────────
# 3. DID 검정
# ─────────────────────────────────────────────────────────────

def did_test(treatment: pd.Series, control: pd.Series) -> dict:
    """
    treatment, control: CAR 시리즈
    DID = mean(treatment) - mean(control)
    Welch's t-test
    """
    if len(treatment) < 2 or len(control) < 2:
        return dict(DID=np.nan, t=np.nan, p=np.nan,
                    N_treat=len(treatment), N_ctrl=len(control))

    did = treatment.mean() - control.mean()
    t_val, p_val = stats.ttest_ind(treatment, control, equal_var=False)

    return dict(DID=did, t=t_val, p=p_val,
                N_treat=len(treatment), N_ctrl=len(control))


# ─────────────────────────────────────────────────────────────
# 4. Main
# ─────────────────────────────────────────────────────────────

def main():
    OUT_RPT.parent.mkdir(parents=True, exist_ok=True)

    # 데이터 로딩
    df = load_and_filter(PATH_CAR)

    # z_score 컬럼 확인
    if "z_score" not in df.columns:
        raise ValueError(f"z_score 컬럼 없음. 컬럼: {df.columns.tolist()}")

    z = df["z_score"]
    print(f"\n  z_score 통계: min={z.min():.3f}, max={z.max():.3f}, "
          f"mean={z.mean():.3f}, median={z.median():.3f}")
    print(f"  z < -1.0: {(z < -1.0).sum()}, z < -1.5: {(z < -1.5).sum()}, "
          f"z < -2.0: {(z < -2.0).sum()}, z < -2.5: {(z < -2.5).sum()}")

    # ── 임계값 그룹 정의 ──────────────────────────────────────
    cumulative_bins = {
        "z < -1.0": df[z < -1.0],
        "z < -1.5": df[z < -1.5],
        "z < -2.0": df[z < -2.0],
        "z < -2.5": df[z < -2.5],
    }
    interval_bins = {
        "-1.5 ≤ z < -1.0": df[(z >= -1.5) & (z < -1.0)],
        "-2.0 ≤ z < -1.5": df[(z >= -2.0) & (z < -1.5)],
    }

    # ── CAAR 통계 계산 ────────────────────────────────────────
    cum_stats    = [caar_stats(sub, lbl) for lbl, sub in cumulative_bins.items()]
    interval_stats = [caar_stats(sub, lbl) for lbl, sub in interval_bins.items()]

    # ── DID 분석 ──────────────────────────────────────────────
    # 각 임계값에 대해 treatment vs control 정의:
    #   treatment = z < threshold
    #   control   = threshold ≤ z < 0  (약한 충격/중립)
    did_thresholds = [
        (-1.5, "z < -1.5 vs  -1.5 ≤ z < 0"),
        (-2.0, "z < -2.0 vs  -2.0 ≤ z < 0"),
        (-2.5, "z < -2.5 vs  -2.5 ≤ z < 0"),
    ]
    did_results = []
    for thr, label in did_thresholds:
        treat_cars   = df[z < thr]["CAR"]
        control_cars = df[(z >= thr) & (z < 0)]["CAR"]
        d = did_test(treat_cars, control_cars)
        d["threshold"] = thr
        d["label"]     = label
        did_results.append(d)

    # ── 추가: 전체 DIRECT 이벤트 기준 ────────────────────────
    overall = caar_stats(df, "전체 DIRECT (필터 후)")

    # ── 보고서 작성 ───────────────────────────────────────────
    lines = [
        "=" * 78,
        "충격 강도별 CAAR 분석 — Phase 4 DID (Diff-in-Differences)",
        "=" * 78,
        "",
        f"데이터: {PATH_CAR.name}",
        f"필터: window_label='0_21', quality='full', confounding_self=False, rel=DIRECT",
        f"전체 DIRECT 이벤트 수: {len(df)}",
        "",
    ]

    # ── 섹션 1: 전체 기준 ────────────────────────────────────
    r = overall
    lines += [
        "─" * 78,
        f"전체 DIRECT  N={r['N']:4d}  "
        f"CAAR={r['CAAR']*100:+.3f}%  "
        f"t={r['t']:+.3f}  p={r['p']:.4f}  "
        f"95%CI=[{r['CI_lo']*100:+.3f}%, {r['CI_hi']*100:+.3f}%]",
        "",
    ]

    # ── 섹션 2: 누적 임계값별 CAAR ───────────────────────────
    lines += [
        "=== 충격 강도별 CAAR 분석 ===",
        f"  {'그룹':<22} {'N':>5}  {'CAAR':>8}  {'t':>7}  {'p':>7}  {'95% CI':>22}",
        "  " + "-" * 72,
    ]
    for r in cum_stats:
        if not np.isnan(r["CAAR"]):
            ci_str = f"[{r['CI_lo']*100:+.2f}%, {r['CI_hi']*100:+.2f}%]"
            lines.append(
                f"  {r['label']:<22} {r['N']:>5}  "
                f"{r['CAAR']*100:>+7.3f}%  "
                f"{r['t']:>7.3f}  "
                f"{r['p']:>7.4f}  "
                f"{ci_str:>22}"
            )
        else:
            lines.append(f"  {r['label']:<22} {r['N']:>5}  (데이터 부족)")

    lines += [""]

    # ── 섹션 3: 구간별 CAAR ──────────────────────────────────
    lines += [
        "=== 구간별 CAAR (비교) ===",
        f"  {'그룹':<26} {'N':>5}  {'CAAR':>8}  {'t':>7}  {'p':>7}  {'95% CI':>22}",
        "  " + "-" * 76,
    ]
    for r in interval_stats:
        if not np.isnan(r["CAAR"]):
            ci_str = f"[{r['CI_lo']*100:+.2f}%, {r['CI_hi']*100:+.2f}%]"
            lines.append(
                f"  {r['label']:<26} {r['N']:>5}  "
                f"{r['CAAR']*100:>+7.3f}%  "
                f"{r['t']:>7.3f}  "
                f"{r['p']:>7.4f}  "
                f"{ci_str:>22}"
            )
        else:
            lines.append(f"  {r['label']:<26} {r['N']:>5}  (데이터 부족)")

    lines += [""]

    # ── 섹션 4: DID ──────────────────────────────────────────
    lines += [
        "=== DID: 강한 충격 vs 약한 충격 ===",
        f"  {'임계값':>6}  {'N_treat':>8}  {'N_ctrl':>8}  "
        f"{'DID':>8}  {'t':>7}  {'p':>7}  설명",
        "  " + "-" * 72,
    ]
    for d in did_results:
        if not np.isnan(d["DID"]):
            sig = "***" if d["p"] < 0.01 else ("**" if d["p"] < 0.05 else ("*" if d["p"] < 0.1 else ""))
            lines.append(
                f"  {d['threshold']:>6.1f}  {d['N_treat']:>8}  {d['N_ctrl']:>8}  "
                f"{d['DID']*100:>+7.3f}%  "
                f"{d['t']:>7.3f}  "
                f"{d['p']:>7.4f}  {sig}"
            )
        else:
            lines.append(f"  {d['threshold']:>6.1f}  (데이터 부족 — Welch t-test 불가)")

    lines += [
        "",
        "  유의성: *** p<0.01, ** p<0.05, * p<0.10",
        "  DID = CAAR(강한 충격) - CAAR(약한 충격/중립)",
        "  t-검정: Welch's t-test (등분산 가정 없음)",
        "",
    ]

    # ── 섹션 5: 해석 ─────────────────────────────────────────
    lines += [
        "=== 해석 가이드 ===",
        "  1. CAAR이 음수 + 유의 → 충격이 강할수록 주가 하락 폭 커짐",
        "  2. DID 유의 → 강한 충격과 약한 충격 간 시장 반응 차이 실증",
        "  3. z < -2.0 이하에서도 유의하면 임계값 효과 존재",
        "  4. 표본 소규모(N<10) 해석 주의",
        "",
        "  활용: z_score 임계값을 GNN 이벤트 필터 기준으로 사용 가능",
        "        (예: z < -1.5만 신호로 채택 → False Positive 감소)",
    ]

    lines += ["", "=" * 78]
    rpt = "\n".join(lines)
    OUT_RPT.write_text(rpt, encoding="utf-8")
    print("\n" + rpt)
    print(f"\n결과 저장: {OUT_RPT}")


if __name__ == "__main__":
    main()
