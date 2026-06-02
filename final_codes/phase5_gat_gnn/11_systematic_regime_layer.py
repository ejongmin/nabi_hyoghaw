"""
11_systematic_regime_layer.py
==============================
레짐 × 레이어 전체 조합 체계적 검정 + 다중 비교 보정

편향 해결:
  - 관세×OEM+Cell만 보고하는 post-hoc selection bias 제거
  - 모든 가능한 레짐×레이어 조합을 동시에 검정
  - Bonferroni & BH FDR 보정 적용
  - 어떤 조합이 보정 후에도 유의한지 보고

Phase 5 (test set 2024-09 ~ 2025-11):
  레짐: Pre-tariff / Trump Tariff / IRA (분리) / China Export (분리)
  레이어: OEM+Cell / Materials+Upstream / 전체

Phase 4 (전체 기간):
  레짐: Pre-regime / COVID / Ukraine / IRA / China Export / Trump Tariff
  레이어: 전체 / OEM / Cell / Materials / Upstream
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, ttest_1samp
from itertools import product

ROOT = Path("C:/Users/john9/nabi_hyoghaw")
EXP  = ROOT / "experiments/universe_44"
OUT  = EXP / "reports/systematic_regime_layer.txt"
SEP  = "=" * 72

OEM_CELL_STAGES      = {"OEM", "Cell/Pack"}
MAT_UPS_STAGES       = {"Materials", "Upstream/Refining"}

P5_REGIMES = {
    "Pre-tariff (2024-09~12)": ("2024-09", "2024-12"),
    "Trump Tariff (2025-01~11)": ("2025-01", "2025-11"),
    "IRA only (2024-09~12)":   ("2024-09", "2024-12"),   # IRA but pre-tariff
    "IRA+Tariff (2025-01~11)": ("2025-01", "2025-11"),   # IRA+Tariff overlap
}

P4_REGIMES = {
    "Pre-regime (2018~2019)":    ("2018-01", "2019-12"),
    "COVID (2020~2021)":         ("2020-01", "2021-12"),
    "Ukraine (2022.02~2023.06)": ("2022-02", "2023-06"),
    "IRA (2022.08~2025.12)":     ("2022-08", "2025-12"),
    "China Export (2023.08~)":   ("2023-08", "2025-12"),
    "Trump Tariff (2025.01~)":   ("2025-01", "2026-12"),
}

P4_LAYERS = {
    "전체": None,
    "OEM": {"OEM"},
    "Cell/Pack": {"Cell/Pack"},
    "OEM+Cell": OEM_CELL_STAGES,
    "Materials": {"Materials"},
    "Upstream": {"Upstream/Refining"},
    "Mat+Upstream": MAT_UPS_STAGES,
}


def bh_correct(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR correction."""
    n = len(pvals)
    if n == 0:
        return []
    sorted_idx = np.argsort(pvals)
    thresholds = [(i + 1) / n * alpha for i in range(n)]
    bh_pass = [False] * n
    for i, (orig_idx, p) in enumerate(zip(sorted_idx, [pvals[j] for j in sorted_idx])):
        if p <= thresholds[i]:
            bh_pass[orig_idx] = True
    return bh_pass


def sig(p, bonf=None):
    if pd.isna(p): return ""
    s = ""
    if p < 0.001: s = "***"
    elif p < 0.01: s = "**"
    elif p < 0.05: s = "*"
    elif p < 0.10: s = "†"
    if bonf is not None and p < bonf: s += " [B]"
    return s


def main():
    lines = [SEP,
             "레짐 × 레이어 전체 조합 체계적 검정 (사후선택 편향 해소)",
             SEP, ""]

    c44 = pd.read_csv(EXP / "data/universe/companies_44.csv")
    stage_map = dict(zip(c44["company_id"], c44["stage_bucket"]))

    # ── Phase 5: 레짐 × 레이어 전체 조합 ─────────────────────────
    lines += ["■ Phase 5 — 레짐 × 레이어 IC (test set 2024-09~2025-11)", ""]

    pred = pd.read_parquet(EXP / "data/processed/phase5_predictions_44.parquet")
    test = pred[(pred["split"] == "test") & (pred["valid"] == True)].copy()
    test["stage"] = test["company_id"].map(stage_map)
    test["ym_dt"] = pd.to_datetime(test["year_month"] + "-01")

    p5_layers = {
        "전체":         None,
        "OEM+Cell":    OEM_CELL_STAGES,
        "Mat+Upstream": MAT_UPS_STAGES,
    }

    p5_regime_defs = {
        "Pre-tariff (2024-09~12)": ("2024-09", "2024-12"),
        "Trump Tariff (2025-01~11)": ("2025-01", "2025-11"),
    }

    p5_results = []
    for (reg_label, (start, end)), (lay_label, stages) in product(
            p5_regime_defs.items(), p5_layers.items()):
        sub = test[
            (test["ym_dt"] >= pd.to_datetime(start + "-01")) &
            (test["ym_dt"] <= pd.to_datetime(end + "-01"))
        ]
        if stages:
            sub = sub[sub["stage"].isin(stages)]
        if len(sub) < 20:
            continue
        ic, _ = spearmanr(sub["prob"], sub["actual_ar"])
        # permutation test (B=2000)
        rng = np.random.default_rng(42)
        shuf_ics = []
        for _ in range(2000):
            s = sub.copy()
            s["actual_ar"] = rng.permutation(s["actual_ar"].values)
            r, _ = spearmanr(s["prob"], s["actual_ar"])
            shuf_ics.append(r)
        pval = (np.abs(shuf_ics) >= np.abs(ic)).sum() / 2000
        p5_results.append({
            "regime": reg_label, "layer": lay_label,
            "N": len(sub), "IC": float(ic), "p": float(pval)
        })

    p5_df = pd.DataFrame(p5_results)
    n_tests_p5 = len(p5_df)
    bonf_p5 = 0.05 / n_tests_p5 if n_tests_p5 > 0 else 0.05
    p5_df["bh_pass"] = bh_correct(p5_df["p"].tolist())

    lines += [f"  총 {n_tests_p5}개 조합 검정",
              f"  Bonferroni α' = 0.05/{n_tests_p5} = {bonf_p5:.4f}",
              f"  BH FDR correction (q=0.05)",
              "",
              f"  {'레짐':<28} {'레이어':<14} {'N':>5} {'IC':>8} {'p':>7} {'보정후'}",
              "  " + "-"*70]

    for _, r in p5_df.sort_values("p").iterrows():
        bonf_mark = "[B]" if r["p"] < bonf_p5 else ""
        bh_mark   = "[FDR]" if r["bh_pass"] else ""
        lines.append(f"  {r['regime']:<28} {r['layer']:<14} {r['N']:>5} "
                     f"{r['IC']:>+7.4f}  {r['p']:>6.4f}{sig(r['p'])}  {bonf_mark}{bh_mark}")

    surv_bonf = p5_df[p5_df["p"] < bonf_p5]
    surv_bh   = p5_df[p5_df["bh_pass"]]
    lines += ["",
              f"  Bonferroni 통과: {len(surv_bonf)}개",
              f"  BH FDR 통과:    {len(surv_bh)}개",
              f"  결론: {'신호 없음' if len(surv_bh)==0 else '유의 신호 존재'}"]

    # ── Phase 4: 레짐 × 레이어 전체 조합 ─────────────────────────
    lines += ["", SEP[:72], "",
              "■ Phase 4 — 레짐 × 레이어 CAAR ([0,+21], DIRECT)", ""]

    cp = pd.read_parquet(ROOT / "data/processed/car_panel_phase4_44.parquet")
    direct = cp[
        (cp["relationship"] == "DIRECT") &
        (cp["quality"] == "full") &
        (~cp["confounding_self"].fillna(False)) &
        (cp["window_label"] == "0_21")
    ].copy()
    direct["shock_month_dt"] = pd.to_datetime(direct["shock_month"] + "-01")

    # stage 매핑 (shock_company 기준)
    direct["shock_stage"] = direct["shock_company"].map(stage_map)

    p4_results = []
    for (reg_label, (start, end)), (lay_label, stages) in product(
            P4_REGIMES.items(), P4_LAYERS.items()):
        sub = direct[
            (direct["shock_month_dt"] >= pd.to_datetime(start + "-01")) &
            (direct["shock_month_dt"] <= pd.to_datetime(end + "-01"))
        ]
        if stages:
            sub = sub[sub["shock_stage"].isin(stages)]
        if len(sub) < 5:
            continue
        t, p = ttest_1samp(sub["CAR"], 0)
        p4_results.append({
            "regime": reg_label, "layer": lay_label,
            "N": len(sub), "CAAR": sub["CAR"].mean(),
            "t": t, "p": float(p)
        })

    p4_df = pd.DataFrame(p4_results)
    n_tests_p4 = len(p4_df)
    bonf_p4 = 0.05 / n_tests_p4 if n_tests_p4 > 0 else 0.05
    p4_df["bh_pass"] = bh_correct(p4_df["p"].tolist())

    lines += [f"  총 {n_tests_p4}개 조합 검정",
              f"  Bonferroni α' = 0.05/{n_tests_p4} = {bonf_p4:.4f}",
              "",
              f"  {'레짐':<28} {'레이어':<14} {'N':>5} {'CAAR':>8} {'p':>7} {'보정후'}",
              "  " + "-"*70]

    for _, r in p4_df.sort_values("p").head(25).iterrows():
        bonf_mark = "[B]" if r["p"] < bonf_p4 else ""
        bh_mark   = "[FDR]" if r["bh_pass"] else ""
        lines.append(f"  {r['regime']:<28} {r['layer']:<14} {r['N']:>5} "
                     f"{r['CAAR']*100:>+7.2f}%  {r['p']:>6.4f}{sig(r['p'])}  "
                     f"{bonf_mark}{bh_mark}")

    surv_bonf4 = p4_df[p4_df["p"] < bonf_p4]
    surv_bh4   = p4_df[p4_df["bh_pass"]]
    lines += [f"  ... (상위 25개만 표시, 총 {n_tests_p4}개)",
              "",
              f"  Bonferroni 통과: {len(surv_bonf4)}개",
              f"  BH FDR 통과:    {len(surv_bh4)}개"]

    if len(surv_bh4) > 0:
        lines += ["", "  BH FDR 통과 조합:"]
        for _, r in surv_bh4.sort_values("p").iterrows():
            lines.append(f"    {r['regime']} × {r['layer']}: "
                         f"CAAR={r['CAAR']*100:+.2f}% p={r['p']:.4f}")

    lines += ["", SEP]
    report = "\n".join(lines)
    OUT.write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
