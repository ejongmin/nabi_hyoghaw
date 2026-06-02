"""
10_all_regime_analysis.py
==========================
모든 레짐 기간별 Phase 4 + Phase 5 신호 강도 비교

레짐:
  regime_covid        : 2020-01 ~ 2021-12
  regime_ukraine      : 2022-02 ~ 2023-06
  regime_ira          : 2022-08 ~ 2025-12
  regime_china_export : 2023-08 ~ 2025-12
  regime_trump_tariff : 2025-01 ~ 2027-12

Phase 4: 레짐별 CAAR (직접 효과, window [0,+21])
Phase 5: 레짐별 IC (GNN 예측 기반)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, ttest_1samp

ROOT = Path("C:/Users/john9/nabi_hyoghaw")
EXP  = ROOT / "experiments/universe_44"

OUT = EXP / "reports/all_regime_analysis.txt"

REGIMES = {
    "Pre-regime (2018~2019)":   ("2018-01", "2019-12"),
    "COVID (2020~2021)":        ("2020-01", "2021-12"),
    "Ukraine (2022.02~2023.06)":("2022-02", "2023-06"),
    "IRA (2022.08~2025.12)":    ("2022-08", "2025-12"),
    "China Export (2023.08~)":  ("2023-08", "2025-12"),
    "Trump Tariff (2025.01~)":  ("2025-01", "2026-12"),
}

SEP = "=" * 72


# ── Phase 4: 레짐별 CAAR ─────────────────────────────────────────
def analyze_phase4_by_regime():
    cp = pd.read_parquet(ROOT / "data/processed/car_panel_phase4_44.parquet")
    direct = cp[
        (cp["relationship"] == "DIRECT") &
        (cp["quality"] == "full") &
        (~cp["confounding_self"].fillna(False)) &
        (cp["window_label"] == "0_21")
    ].copy()

    direct["shock_month_dt"] = pd.to_datetime(direct["shock_month"] + "-01")

    results = []
    for label, (start, end) in REGIMES.items():
        sub = direct[
            (direct["shock_month_dt"] >= pd.to_datetime(start)) &
            (direct["shock_month_dt"] <= pd.to_datetime(end))
        ]
        if len(sub) < 3:
            results.append({"regime": label, "N": len(sub),
                             "CAAR": np.nan, "t": np.nan, "p": np.nan})
            continue
        t, p = ttest_1samp(sub["CAR"], 0)
        results.append({
            "regime": label, "N": len(sub),
            "CAAR": sub["CAR"].mean(),
            "t": t, "p": p,
        })

    # NEG vs POS 분리도
    for label, (start, end) in REGIMES.items():
        sub = direct[
            (direct["shock_month_dt"] >= pd.to_datetime(start)) &
            (direct["shock_month_dt"] <= pd.to_datetime(end))
        ]
        for shock in ["NEG", "POS"]:
            s = sub[sub["shock_type"] == shock]
            if len(s) < 3:
                continue

    return pd.DataFrame(results)


# ── Phase 4: 전파 레짐별 비교 ─────────────────────────────────────
def analyze_contagion_by_regime():
    cp = pd.read_parquet(ROOT / "data/processed/car_panel_phase4_44.parquet")
    # UPSTREAM_2 (Bonferroni-robust 신호)
    upstream2 = cp[
        (cp["relationship"] == "UPSTREAM_2_2") &
        (cp["quality"] == "full") &
        (cp["window_label"] == "0_21")
    ].copy()
    upstream2["shock_month_dt"] = pd.to_datetime(upstream2["shock_month"] + "-01")

    results = []
    for label, (start, end) in REGIMES.items():
        sub = upstream2[
            (upstream2["shock_month_dt"] >= pd.to_datetime(start)) &
            (upstream2["shock_month_dt"] <= pd.to_datetime(end))
        ]
        if len(sub) < 5:
            results.append({"regime": label, "N": len(sub),
                             "CAAR": np.nan, "p": np.nan})
            continue
        t, p = ttest_1samp(sub["CAR"], 0)
        results.append({"regime": label, "N": len(sub),
                         "CAAR": sub["CAR"].mean(), "p": p})
    return pd.DataFrame(results)


# ── Phase 5: 레짐별 IC ───────────────────────────────────────────
def analyze_phase5_by_regime():
    pred = pd.read_parquet(EXP / "data/processed/phase5_predictions_44.parquet")
    test = pred[(pred["split"] == "test") & (pred["valid"] == True)].copy()
    test["ym_dt"] = pd.to_datetime(test["year_month"] + "-01")

    results = []
    for label, (start, end) in REGIMES.items():
        sub = test[
            (test["ym_dt"] >= pd.to_datetime(start)) &
            (test["ym_dt"] <= pd.to_datetime(end))
        ]
        if len(sub) < 10:
            results.append({"regime": label, "N": len(sub),
                             "IC": np.nan, "IC>0%": np.nan})
            continue
        ic, _ = spearmanr(sub["prob"], sub["actual_ar"])

        # 월별 IC
        monthly_ics = []
        for ym, g in sub.groupby("year_month"):
            if len(g) >= 5:
                r, _ = spearmanr(g["prob"], g["actual_ar"])
                if not np.isnan(r):
                    monthly_ics.append(r)
        pos_pct = 100 * sum(1 for v in monthly_ics if v > 0) / len(monthly_ics) \
                  if monthly_ics else np.nan

        results.append({
            "regime": label, "N": len(sub),
            "IC": float(ic),
            "monthly_IC_mean": np.mean(monthly_ics) if monthly_ics else np.nan,
            "IC>0%": pos_pct,
            "n_months": len(monthly_ics),
        })
    return pd.DataFrame(results)


# ── 실행 ────────────────────────────────────────────────────────
p4_direct = analyze_phase4_by_regime()
p4_upstream2 = analyze_contagion_by_regime()
p5 = analyze_phase5_by_regime()


def sig(p):
    if pd.isna(p): return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.10:  return "†"
    return ""


lines = [SEP,
"전체 레짐별 신호 강도 비교 — Phase 4 & Phase 5",
SEP, "",
"■ Phase 4: 직접 효과 CAAR (window [0,+21], DIRECT, quality=full)", "",
f"  {'레짐':<30} {'N':>5} {'CAAR':>8} {'t':>7} {'p':>7}",
"  " + "-"*56]

for _, r in p4_direct.iterrows():
    if pd.isna(r["CAAR"]):
        lines.append(f"  {r['regime']:<30} {r['N']:>5}  (N<3, skip)")
    else:
        lines.append(f"  {r['regime']:<30} {r['N']:>5} {r['CAAR']*100:>+7.2f}%  "
                     f"{r['t']:>+6.3f}  {r['p']:>6.4f}{sig(r['p'])}")

lines += ["",
"■ Phase 4: UPSTREAM_2 hop2 CAAR (Bonferroni-robust 신호)", "",
f"  {'레짐':<30} {'N':>5} {'CAAR':>8} {'p':>7}",
"  " + "-"*44]

for _, r in p4_upstream2.iterrows():
    if pd.isna(r["CAAR"]):
        lines.append(f"  {r['regime']:<30} {r['N']:>5}  (N<5, skip)")
    else:
        lines.append(f"  {r['regime']:<30} {r['N']:>5} {r['CAAR']*100:>+7.2f}%  "
                     f"{r['p']:>6.4f}{sig(r['p'])}")

lines += ["",
"■ Phase 5: GNN IC 레짐별 비교 (test set 한정)", "",
f"  {'레짐':<30} {'N':>5} {'IC':>8} {'월IC평균':>9} {'IC>0%':>7} {'n월':>4}",
"  " + "-"*64]

for _, r in p5.iterrows():
    if pd.isna(r["IC"]):
        lines.append(f"  {r['regime']:<30} {r['N']:>5}  (N<10, skip)")
    else:
        lines.append(f"  {r['regime']:<30} {r['N']:>5} {r['IC']:>+7.4f}  "
                     f"{r['monthly_IC_mean']:>+8.4f}  {r['IC>0%']:>6.1f}%  {r['n_months']:>4}")

# 종합 해석
lines += ["",
"■ 종합 해석", "",
"  레짐별 신호 강도 순위 (Phase 4 CAAR 기준):"]

p4_sorted = p4_direct.dropna(subset=["CAAR"]).sort_values("CAAR", ascending=False)
for i, (_, r) in enumerate(p4_sorted.iterrows(), 1):
    lines.append(f"  {i}위: {r['regime']:<30} CAAR={r['CAAR']*100:+.2f}% (p={r['p']:.3f}{sig(r['p'])})")

lines += ["",
"  Phase 5 IC 강도 순위:"]
p5_sorted = p5.dropna(subset=["IC"]).sort_values("IC", ascending=False)
for i, (_, r) in enumerate(p5_sorted.iterrows(), 1):
    lines.append(f"  {i}위: {r['regime']:<30} IC={r['IC']:+.4f} IC>0={r['IC>0%']:.0f}%")

lines += ["", SEP]
report = "\n".join(lines)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(report, encoding="utf-8")
print("\n" + report)
print(f"\n저장: {OUT}")
