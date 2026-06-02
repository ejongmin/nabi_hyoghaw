"""
Phase 5 Permutation Test
목적: IC=+0.068이 우연인지 검증 (year_month 그룹 내 레이블 셔플)
출력: reports/phase5_permutation_test.txt
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "phase5_final_predictions.parquet"
OUT_PATH = ROOT / "reports" / "phase5_permutation_test.txt"

B = 2000
RNG = np.random.default_rng(42)


def spearman_ic(df: pd.DataFrame) -> float:
    corr, _ = spearmanr(df["prob"], df["actual_ar"])
    return float(corr)


def permutation_pvalue(df: pd.DataFrame, observed: float, b: int) -> tuple[float, np.ndarray]:
    """year_month 그룹 내에서 actual_ar을 셔플해 permutation IC 분포를 생성."""
    shuffle_ics = np.empty(b)
    for i in range(b):
        shuffled = df.copy()
        shuffled["actual_ar"] = (
            shuffled.groupby("year_month")["actual_ar"]
            .transform(lambda x: RNG.permutation(x.values))
        )
        shuffle_ics[i] = spearman_ic(shuffled)
    pval = (np.abs(shuffle_ics) >= np.abs(observed)).sum() / b
    return pval, shuffle_ics


def main():
    df = pd.read_parquet(DATA_PATH)
    test_df = df[(df["split"] == "test") & (df["valid"] == True)].reset_index(drop=True)

    # ── 전체 IC 검정 ──────────────────────────────────────────────────
    overall_ic = spearman_ic(test_df)
    pval_overall, shuffle_dist = permutation_pvalue(test_df, overall_ic, B)

    # ── 월별 IC 검정 ──────────────────────────────────────────────────
    monthly_results = []
    for ym, grp in test_df.groupby("year_month"):
        if len(grp) < 5:
            monthly_results.append((ym, len(grp), np.nan, np.nan, np.nan))
            continue
        obs = spearman_ic(grp)
        # 월별은 그룹이 1개이므로 전체 셔플(=동일 그룹 내 셔플)
        shuf_ics = np.empty(B)
        for i in range(B):
            grp2 = grp.copy()
            grp2["actual_ar"] = RNG.permutation(grp2["actual_ar"].values)
            shuf_ics[i] = spearman_ic(grp2)
        pv = (np.abs(shuf_ics) >= np.abs(obs)).sum() / B
        monthly_results.append((ym, len(grp), obs, pv, np.std(shuf_ics)))

    # ── 출력 ──────────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 60)
    lines.append("Phase 5 Permutation Test 결과")
    lines.append("=" * 60)
    lines.append(f"\n데이터: split=='test' & valid==True  (N={len(test_df)})")
    lines.append(f"셔플 횟수: B={B}  (year_month 그룹 내 actual_ar 셔플)")
    lines.append(f"RNG seed: 42\n")

    lines.append("■ 전체 IC 검정")
    lines.append(f"  실제 IC (Spearman)  = {overall_ic:+.4f}")
    lines.append(f"  셔플 IC 평균        = {shuffle_dist.mean():+.4f}")
    lines.append(f"  셔플 IC 표준편차    = {shuffle_dist.std():.4f}")
    lines.append(f"  p-value (양측)      = {pval_overall:.4f}")
    sig = "**유의 (p<0.05)**" if pval_overall < 0.05 else "비유의 (p>=0.05)"
    lines.append(f"  판정               → {sig}\n")

    lines.append("■ 월별 IC 검정")
    lines.append(f"  {'year_month':<12} {'N':>4} {'IC':>8} {'p-value':>9} {'셔플 SD':>9}")
    lines.append("  " + "-" * 48)
    for ym, n, ic, pv, sd in monthly_results:
        if np.isnan(ic):
            lines.append(f"  {ym:<12} {n:>4}  (N<5, skip)")
        else:
            star = "*" if pv < 0.05 else ""
            lines.append(f"  {ym:<12} {n:>4} {ic:>+8.4f} {pv:>9.4f}{star:1}  {sd:>8.4f}")

    # ── IC>0 비율 / 평균 요약
    valid_months = [(ym, ic, pv) for ym, n, ic, pv, sd in monthly_results
                    if not np.isnan(ic) and pv is not None and not np.isnan(pv)]
    ic_vals  = [ic for _, ic, _ in valid_months]
    ic_pos   = sum(1 for ic in ic_vals if ic > 0)
    sig_vals = [(ym, ic) for ym, ic, pv in valid_months if pv < 0.05]

    lines.append("")
    lines.append("■ 요약 통계")
    lines.append(f"  월별 IC 평균   = {np.mean(ic_vals):+.4f}")
    lines.append(f"  월별 IC std    = {np.std(ic_vals):.4f}")
    lines.append(f"  IC>0 비율      = {ic_pos}/{len(ic_vals)} ({100*ic_pos/len(ic_vals):.1f}%)")
    lines.append(f"  p<0.05 달      = {len(sig_vals)}개 {[ym for ym, _ in sig_vals]}")

    # ── 다중 비교 보정 (Bonferroni & BH FDR) ─────────────────────
    monthly_pvals = [pv for _, _, pv in valid_months]
    n_tests = len(monthly_pvals)

    # Bonferroni
    alpha_bonf = 0.05 / n_tests if n_tests > 0 else 0.05
    bonf_sig = [(ym, ic) for ym, ic, pv in valid_months if pv <= alpha_bonf]

    # Benjamini-Hochberg FDR (q=0.05)
    sorted_pvals_idx = sorted(enumerate(monthly_pvals), key=lambda x: x[1])
    bh_threshold     = [(i + 1) / n_tests * 0.05 for i in range(n_tests)]
    bh_significant   = [p <= bh_threshold[i] for i, (_, p) in enumerate(sorted_pvals_idx)]
    bh_sig_orig_idx  = {sorted_pvals_idx[i][0] for i, sig in enumerate(bh_significant) if sig}
    bh_sig_vals      = [(ym, ic) for j, (ym, ic, _) in enumerate(valid_months)
                        if j in bh_sig_orig_idx]

    lines.append("")
    lines.append(f"■ Multiple Testing Correction (월별 {n_tests}개 검정)")
    lines.append(f"  Bonferroni α' = 0.05/{n_tests} = {alpha_bonf:.4f}")
    bonf_label = f"  {[ym for ym, _ in bonf_sig]}" if bonf_sig else ""
    lines.append(f"  Bonferroni 유의 달: {len(bonf_sig)}개{bonf_label}")
    lines.append("")
    lines.append("  BH FDR correction (q=0.05):")
    bh_label = f"  {[ym for ym, _ in bh_sig_vals]}" if bh_sig_vals else ""
    lines.append(f"  BH 유의 달: {len(bh_sig_vals)}개{bh_label}")
    lines.append("")
    if len(bonf_sig) == 0 and len(bh_sig_vals) == 0:
        lines.append("  결론: 다중 비교 보정 후 유의한 달 없음")
    else:
        lines.append(f"  결론: Bonferroni {len(bonf_sig)}개, BH {len(bh_sig_vals)}개 달 유의")

    # ── 전체 IC 단측 p-value (H1: IC > 0) ─────────────────────────
    pval_one_sided = (shuffle_dist >= overall_ic).sum() / B
    lines.append("")
    lines.append("■ 전체 IC 단측 검정 (H1: IC > 0)")
    lines.append(f"  p-value (단측, 우측)  = {pval_one_sided:.4f}")
    sig_label = "**유의 (p<0.05)**" if pval_one_sided < 0.05 else "비유의 (p>=0.05)"
    lines.append(f"  판정                 → {sig_label}")

    lines.append("\n" + "=" * 60)

    report = "\n".join(lines)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n저장 완료: {OUT_PATH}")


if __name__ == "__main__":
    main()
