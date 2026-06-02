"""
08_regime_conditional.py
=========================
Phase 5 레짐 조건부 IC 분석

트럼프 관세 레짐(2025-01~) × 레이어(OEM+Cell) 조건에서
GNN 신호가 유의해지는가를 검증.

분석:
  1. 전체 test set IC (baseline)
  2. 관세 레짐 기간(2025-01~) 조건부 IC
  3. OEM+Cell 기업 조건부 IC
  4. 관세 레짐 × OEM+Cell 교차 조건부 IC
  5. 각 조건별 Permutation test (B=5000)
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path("C:/Users/john9/nabi_hyoghaw")
EXP  = ROOT / "experiments/universe_44"

PRED_PATH = EXP  / "data/processed/phase5_predictions_44.parquet"
C44_PATH  = EXP  / "data/universe/companies_44.csv"
OUT_PATH  = EXP  / "reports/phase5_regime_conditional.txt"

B   = 5000
RNG = np.random.default_rng(42)
SEP = "=" * 64

OEM_CELL_STAGES = {"OEM", "Cell/Pack"}
TARIFF_START    = "2025-01"


def spearman_ic(df: pd.DataFrame) -> float:
    if len(df) < 5:
        return np.nan
    corr, _ = spearmanr(df["prob"], df["actual_ar"])
    return float(corr)


def perm_pval(df: pd.DataFrame, obs: float, b: int) -> tuple:
    if np.isnan(obs) or len(df) < 5:
        return np.nan, np.array([])
    shuf = np.empty(b)
    for i in range(b):
        s = df.copy()
        s["actual_ar"] = (
            s.groupby("year_month")["actual_ar"]
            .transform(lambda x: RNG.permutation(x.values))
        )
        shuf[i] = spearman_ic(s)
    # 양측 검정
    pval = (np.abs(shuf) >= np.abs(obs)).sum() / b
    return float(pval), shuf


def monthly_ic(df: pd.DataFrame) -> dict:
    rows = []
    for ym, g in df.groupby("year_month"):
        ic = spearman_ic(g)
        rows.append({"ym": ym, "IC": ic, "N": len(g)})
    vals = [r["IC"] for r in rows if not np.isnan(r["IC"])]
    return {
        "mean": np.mean(vals) if vals else np.nan,
        "std":  np.std(vals)  if vals else np.nan,
        "pos":  sum(1 for v in vals if v > 0),
        "n":    len(vals),
        "monthly": rows,
    }


def analyze(df: pd.DataFrame, label: str, run_perm: bool = True):
    ic = spearman_ic(df)
    mi = monthly_ic(df)

    if run_perm and not np.isnan(ic):
        pval, shuf = perm_pval(df, ic, B)
    else:
        pval, shuf = np.nan, np.array([])

    result = {
        "label": label, "N": len(df),
        "IC": ic, "pval": pval,
        "monthly_mean": mi["mean"], "monthly_std": mi["std"],
        "IC_pos_pct": 100 * mi["pos"] / mi["n"] if mi["n"] > 0 else np.nan,
        "n_months": mi["n"],
        "shuf_mean": shuf.mean() if len(shuf) > 0 else np.nan,
        "shuf_std":  shuf.std()  if len(shuf) > 0 else np.nan,
    }
    return result


def main():
    # ── 데이터 로드 ──────────────────────────────────────────────
    pred = pd.read_parquet(PRED_PATH)
    c44  = pd.read_csv(C44_PATH)
    stage_map = dict(zip(c44["company_id"], c44["stage_bucket"]))

    test = pred[(pred["split"] == "test") & (pred["valid"] == True)].copy()
    test["stage"] = test["company_id"].map(stage_map)
    test["is_tariff"] = test["year_month"] >= TARIFF_START
    test["is_oem_cell"] = test["stage"].isin(OEM_CELL_STAGES)

    print(f"전체 test valid: N={len(test)}, 기업={test['company_id'].nunique()}")
    print(f"관세 레짐(≥{TARIFF_START}): N={test['is_tariff'].sum()}")
    print(f"OEM+Cell: N={test['is_oem_cell'].sum()}")
    print(f"관세×OEM+Cell: N={(test['is_tariff'] & test['is_oem_cell']).sum()}")
    print()

    # ── 5가지 조건 분석 ─────────────────────────────────────────
    conditions = [
        ("전체 test (baseline)",
         test),
        (f"관세 레짐({TARIFF_START}~) 기간만",
         test[test["is_tariff"]]),
        ("Pre-tariff(~2024-12) 기간만",
         test[~test["is_tariff"]]),
        ("OEM+Cell 기업만",
         test[test["is_oem_cell"]]),
        (f"관세 레짐 × OEM+Cell (교차)",
         test[test["is_tariff"] & test["is_oem_cell"]]),
        ("Materials+Upstream 기업만",
         test[~test["is_oem_cell"]]),
    ]

    results = []
    for label, df_sub in conditions:
        print(f"  분석 중: {label} (N={len(df_sub)})...")
        r = analyze(df_sub, label, run_perm=(len(df_sub) >= 20))
        results.append(r)

    # ── 보고서 ───────────────────────────────────────────────────
    def sig(p):
        if np.isnan(p): return ""
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        if p < 0.10:  return "†"
        return ""

    lines = [
        SEP,
        "Phase 5 — 레짐 조건부 IC 분석 (44개 기업, bias-free v4)",
        SEP, "",
        f"  예측 파일: {PRED_PATH.name}",
        f"  관세 레짐 시작: {TARIFF_START}",
        f"  OEM+Cell 기업: {sorted(test[test['is_oem_cell']]['company_id'].unique())}",
        f"  Permutation B={B} (양측, year_month 그룹 내 셔플)", "",
        "■ 조건별 IC 비교",
        "",
        f"  {'조건':<28} {'N':>5} {'IC':>8} {'p(양측)':>9} {'월IC평균':>9} {'IC>0%':>7}",
        "  " + "-" * 68,
    ]

    for r in results:
        pv_str = f"{r['pval']:.4f}{sig(r['pval'])}" if not np.isnan(r['pval']) else "  N/A "
        lines.append(
            f"  {r['label']:<28} {r['N']:>5} {r['IC']:>+7.4f}  "
            f"{pv_str:>10}  {r['monthly_mean']:>+8.4f}  {r['IC_pos_pct']:>6.1f}%"
        )

    # Shuffled distribution for baseline
    base = results[0]
    lines += [
        "",
        "■ Shuffled IC 분포 (baseline 기준)",
        f"  셔플 IC 평균 = {base['shuf_mean']:+.4f}",
        f"  셔플 IC 표준편차 = {base['shuf_std']:.4f}",
        "",
        "■ 해석",
    ]

    # 관세 레짐 효과
    tariff = results[1]
    pretariff = results[2]
    if not np.isnan(tariff["IC"]) and not np.isnan(pretariff["IC"]):
        diff = tariff["IC"] - pretariff["IC"]
        lines.append(f"  관세 레짐 IC ({tariff['IC']:+.4f}) vs Pre-tariff ({pretariff['IC']:+.4f}) "
                     f"= 차이 {diff:+.4f}")
        lines.append(f"  → 관세 레짐 {'강화' if diff > 0 else '약화'} "
                     f"({'유의' if tariff['pval'] < 0.05 else '비유의'})")

    # OEM+Cell 효과
    oem_cell = results[3]
    mat_ups   = results[5]
    if not np.isnan(oem_cell["IC"]) and not np.isnan(mat_ups["IC"]):
        lines.append(f"  OEM+Cell IC ({oem_cell['IC']:+.4f}) vs Materials+Upstream ({mat_ups['IC']:+.4f})")
        lines.append(f"  → OEM+Cell {'우월' if oem_cell['IC'] > mat_ups['IC'] else '열세'}")

    # 교차 효과
    cross = results[4]
    lines.append(f"  교차 조건(관세×OEM+Cell) IC = {cross['IC']:+.4f} "
                 f"({'유의 ' if cross['pval'] < 0.05 else '비유의 '}p={cross['pval']:.4f}{sig(cross['pval'])})")

    lines += ["", SEP]
    report = "\n".join(lines)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\n저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
