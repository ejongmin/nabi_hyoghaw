"""
Phase 5 Permutation Test — universe_44 (44개 기업)
목적: IC가 우연인지 검증 (year_month 그룹 내 레이블 셔플, 양측 검정)
출력: experiments/universe_44/reports/phase5_permutation_test_44.txt
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT     = Path(__file__).resolve().parents[3]   # nabi_hyoghaw/
EXP      = ROOT / "experiments/universe_44"
DATA_PATH = EXP / "data/processed/phase5_predictions_44.parquet"
OUT_PATH  = EXP / "reports/phase5_permutation_test_44.txt"

B   = 10000
RNG = np.random.default_rng(42)


def spearman_ic(df: pd.DataFrame) -> float:
    corr, _ = spearmanr(df["prob"], df["actual_ar"])
    return float(corr)


def permutation_pvalue(df: pd.DataFrame, observed: float, b: int):
    """year_month 그룹 내 actual_ar 셔플 → 양측 검정."""
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

    print(f"데이터: {DATA_PATH.name}")
    print(f"test valid N={len(test_df)}, 기업 수={test_df['company_id'].nunique()}, "
          f"기간={test_df['year_month'].min()} ~ {test_df['year_month'].max()}")

    # ── 전체 IC 검정
    overall_ic = spearman_ic(test_df)
    pval_overall, shuffle_dist = permutation_pvalue(test_df, overall_ic, B)

    # ── 월별 IC 검정
    monthly_results = []
    for ym, grp in test_df.groupby("year_month"):
        if len(grp) < 5:
            monthly_results.append((ym, len(grp), np.nan, np.nan, np.nan))
            continue
        obs = spearman_ic(grp)
        shuf = np.empty(B)
        for i in range(B):
            g2 = grp.copy()
            g2["actual_ar"] = RNG.permutation(g2["actual_ar"].values)
            shuf[i] = spearman_ic(g2)
        pv = (np.abs(shuf) >= np.abs(obs)).sum() / B
        monthly_results.append((ym, len(grp), obs, pv, float(np.std(shuf))))

    # ── 출력
    lines = ["=" * 64,
             "Phase 5 Permutation Test — universe_44 (44개 기업)",
             "=" * 64,
             f"\n데이터: {DATA_PATH.name}  (N={len(test_df)}, "
             f"기업={test_df['company_id'].nunique()})",
             f"셔플 횟수: B={B}  (year_month 그룹 내 actual_ar 셔플, 양측)",
             f"RNG seed: 42\n",
             "■ 전체 IC 검정",
             f"  실제 IC (Spearman)  = {overall_ic:+.4f}",
             f"  셔플 IC 평균        = {shuffle_dist.mean():+.4f}",
             f"  셔플 IC 표준편차    = {shuffle_dist.std():.4f}",
             f"  p-value (양측)      = {pval_overall:.4f}",
             f"  판정               → {'**유의 (p<0.05)**' if pval_overall < 0.05 else '비유의 (p>=0.05)'}\n",
             "■ 월별 IC 검정",
             f"  {'year_month':<12} {'N':>4} {'IC':>8} {'p-value':>9} {'셔플 SD':>9}",
             "  " + "-" * 48]

    for ym, n, ic, pv, sd in monthly_results:
        if np.isnan(ic):
            lines.append(f"  {ym:<12} {n:>4}  (N<5, skip)")
        else:
            star = "*" if pv < 0.05 else ""
            lines.append(f"  {ym:<12} {n:>4} {ic:>+8.4f} {pv:>9.4f}{star:<1}  {sd:>8.4f}")

    # ── IC>0 비율 / 평균 요약
    valid_months = [(ym, ic, pv) for ym, n, ic, pv, sd in monthly_results
                    if not np.isnan(ic) and pv is not None and not np.isnan(pv)]
    ic_vals  = [ic for _, ic, _ in valid_months]
    ic_pos   = sum(1 for ic in ic_vals if ic > 0)
    sig_vals = [(ym, ic) for ym, ic, pv in valid_months if pv < 0.05]

    lines += [
        "",
        "■ 요약 통계",
        f"  월별 IC 평균   = {np.mean(ic_vals):+.4f}",
        f"  월별 IC std    = {np.std(ic_vals):.4f}",
        f"  IC>0 비율      = {ic_pos}/{len(ic_vals)} ({100*ic_pos/len(ic_vals):.1f}%)",
        f"  p<0.05 달      = {len(sig_vals)}개 {[ym for ym,_ in sig_vals]}",
    ]

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
    # BH 유의 인덱스를 원래 valid_months 인덱스로 역매핑
    bh_sig_orig_idx  = {sorted_pvals_idx[i][0] for i, sig in enumerate(bh_significant) if sig}
    bh_sig_vals      = [(ym, ic) for j, (ym, ic, _) in enumerate(valid_months)
                        if j in bh_sig_orig_idx]

    lines += [
        "",
        f"■ Multiple Testing Correction (월별 {n_tests}개 검정)",
        f"  Bonferroni α' = 0.05/{n_tests} = {alpha_bonf:.4f}",
        f"  Bonferroni 유의 달: {len(bonf_sig)}개"
        + (f"  {[ym for ym, _ in bonf_sig]}" if bonf_sig else ""),
        "",
        f"  BH FDR correction (q=0.05):",
        f"  BH 유의 달: {len(bh_sig_vals)}개"
        + (f"  {[ym for ym, _ in bh_sig_vals]}" if bh_sig_vals else ""),
        "",
    ]
    if len(bonf_sig) == 0 and len(bh_sig_vals) == 0:
        lines.append("  결론: 다중 비교 보정 후 유의한 달 없음")
    else:
        lines.append(f"  결론: Bonferroni {len(bonf_sig)}개, BH {len(bh_sig_vals)}개 달 유의")

    # ── 전체 IC 단측 p-value (H1: IC > 0) ─────────────────────────
    pval_one_sided = (shuffle_dist >= overall_ic).sum() / B
    lines += [
        "",
        "■ 전체 IC 단측 검정 (H1: IC > 0)",
        f"  p-value (단측, 우측)  = {pval_one_sided:.4f}",
        f"  판정                 → "
        + ("**유의 (p<0.05)**" if pval_one_sided < 0.05 else "비유의 (p>=0.05)"),
        "\n" + "=" * 64,
    ]

    report = "\n".join(lines)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\n저장 완료: {OUT_PATH}")


if __name__ == "__main__":
    main()
