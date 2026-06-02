"""
12_mat_upstream_deep.py
========================
Pre-tariff × Mat+Upstream IC=+0.397 심층 분석

핵심 질문:
  1. 어떤 기업들이 이 신호를 만드는가?
  2. 어떤 달에 IC가 집중되는가?
  3. 이 신호가 안정적인가 (permutation test)?
  4. 왜 Materials+Upstream이 pre-tariff에서 더 강한가?
  5. 더 많은 데이터를 활용할 수 있는가?
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

ROOT = Path("C:/Users/john9/nabi_hyoghaw")
EXP  = ROOT / "experiments/universe_44"
OUT  = EXP / "reports/mat_upstream_deep.txt"
SEP  = "=" * 68

MAT_UPS_STAGES = {"Materials", "Upstream/Refining"}
PRE_TARIFF_START = "2024-09"
PRE_TARIFF_END   = "2024-12"
TARIFF_START     = "2025-01"


def main():
    pred = pd.read_parquet(EXP / "data/processed/phase5_predictions_44.parquet")
    c44  = pd.read_csv(EXP / "data/universe/companies_44.csv")
    tone = pd.read_parquet(EXP / "data/processed/tone_monthly_zscore_all44_v4.parquet")

    stage_map = dict(zip(c44["company_id"], c44["stage_bucket"]))
    name_map  = dict(zip(c44["company_id"], c44["canonical_name"]))

    test = pred[(pred["split"] == "test") & (pred["valid"] == True)].copy()
    test["stage"] = test["company_id"].map(stage_map)
    test["name"]  = test["company_id"].map(name_map)
    test["ym_dt"] = pd.to_datetime(test["year_month"] + "-01")
    test["is_mat_ups"] = test["stage"].isin(MAT_UPS_STAGES)
    test["is_pretariff"] = (test["year_month"] >= PRE_TARIFF_START) & \
                           (test["year_month"] <= PRE_TARIFF_END)

    mat_pre = test[test["is_mat_ups"] & test["is_pretariff"]]
    mat_tar = test[test["is_mat_ups"] & (test["year_month"] >= TARIFF_START)]
    all_pre = test[test["is_pretariff"]]

    lines = [SEP, "Pre-tariff × Materials+Upstream IC=+0.397 심층 분석", SEP, ""]

    # ── 1. 기간별 요약 ─────────────────────────────────────────────
    lines += ["■ 1. 기간별 비교 요약", ""]
    groups = {
        "Pre-tariff × Mat+Upstream": mat_pre,
        "Tariff × Mat+Upstream":     mat_tar,
        "Pre-tariff × 전체":          all_pre,
    }
    for label, df in groups.items():
        ic, _ = spearmanr(df["prob"], df["actual_ar"]) if len(df) >= 5 else (np.nan, np.nan)
        lines.append(f"  {label:<32}  N={len(df):>4}  IC={ic:+.4f}")
    lines.append("")

    # ── 2. Mat+Upstream 기업 목록 및 기업별 IC ────────────────────
    lines += ["■ 2. Materials+Upstream 기업별 분석 (pre-tariff 기간)", ""]
    lines.append(f"  {'기업':<25} {'stage':<15} {'N':>4} {'prob평균':>9} {'AR평균':>9} {'IC':>8}")
    lines.append("  " + "-"*72)

    company_ics = []
    for cid, grp in mat_pre.groupby("company_id"):
        if len(grp) < 2: continue
        ic_c, _ = spearmanr(grp["prob"], grp["actual_ar"]) if len(grp) >= 3 else (np.nan, np.nan)
        company_ics.append({
            "cid": cid, "name": name_map.get(cid, cid),
            "stage": stage_map.get(cid, "?"),
            "N": len(grp),
            "prob_mean": grp["prob"].mean(),
            "ar_mean": grp["actual_ar"].mean(),
            "ic": ic_c
        })
        lines.append(f"  {name_map.get(cid,cid):<25} {stage_map.get(cid,'?'):<15} "
                     f"{len(grp):>4} {grp['prob'].mean():>+8.4f} "
                     f"{grp['actual_ar'].mean()*100:>+8.2f}%  {ic_c:>+7.4f}")

    lines.append("")

    # ── 3. 월별 IC ──────────────────────────────────────────────────
    lines += ["■ 3. 월별 IC (Materials+Upstream, pre-tariff)", ""]
    lines.append(f"  {'월':<10} {'N':>4} {'IC':>8} {'prob평균':>9} {'AR평균':>9}")
    lines.append("  " + "-"*46)

    monthly_ics = []
    for ym, grp in mat_pre.groupby("year_month"):
        if len(grp) < 3:
            lines.append(f"  {ym:<10} {len(grp):>4}  (N<3, skip)")
            continue
        ic_m, _ = spearmanr(grp["prob"], grp["actual_ar"])
        monthly_ics.append(ic_m)
        lines.append(f"  {ym:<10} {len(grp):>4} {ic_m:>+7.4f}  "
                     f"{grp['prob'].mean():>+8.4f} {grp['actual_ar'].mean()*100:>+8.2f}%")

    lines.append(f"\n  월별 IC 평균: {np.mean(monthly_ics):+.4f}  std: {np.std(monthly_ics):.4f}")
    lines.append(f"  IC>0 비율: {sum(1 for v in monthly_ics if v>0)}/{len(monthly_ics)}")

    # ── 4. Permutation test (B=5000) ───────────────────────────────
    lines += ["", "■ 4. Permutation Test (B=5000, 양측)", ""]
    RNG = np.random.default_rng(42)
    obs_ic, _ = spearmanr(mat_pre["prob"], mat_pre["actual_ar"])
    shuf = np.empty(5000)
    for i in range(5000):
        s = mat_pre.copy()
        s["actual_ar"] = (
            s.groupby("year_month")["actual_ar"]
            .transform(lambda x: RNG.permutation(x.values))
        )
        r, _ = spearmanr(s["prob"], s["actual_ar"])
        shuf[i] = r
    p5000 = (np.abs(shuf) >= np.abs(obs_ic)).sum() / 5000
    p5000_one = (shuf >= obs_ic).sum() / 5000

    lines += [f"  관측 IC = {obs_ic:+.4f}",
              f"  셔플 IC 평균 = {shuf.mean():+.4f}  std = {shuf.std():.4f}",
              f"  p-value (양측, B=5000) = {p5000:.4f}{'  → 유의' if p5000<0.05 else '  → 비유의'}",
              f"  p-value (단측 H1: IC>0) = {p5000_one:.4f}{'  → 유의' if p5000_one<0.05 else '  → 비유의'}"]

    # ── 5. 왜 Mat+Upstream이 pre-tariff에서 더 강한가? ─────────────
    lines += ["", "■ 5. 경제적 해석 — 2024-09~12 Materials+Upstream 시장 상황", ""]

    # 해당 기간 tone 데이터 확인
    tone_sub = tone[
        (tone["company_id"].map(stage_map).isin(MAT_UPS_STAGES)) &
        (tone["year_month"] >= PRE_TARIFF_START) &
        (tone["year_month"] <= PRE_TARIFF_END) &
        (tone["reliable_v3"] == True)
    ]
    if len(tone_sub) > 0:
        lines.append(f"  해당 기간 Mat+Upstream reliable z-score:")
        z_stats = tone_sub.groupby("company_id")["z_score_v3"].mean().sort_values()
        for cid, z in z_stats.items():
            lines.append(f"    {name_map.get(cid,cid):<25}: z_mean={z:+.3f}")

    lines += ["",
              "  [맥락 설명]",
              "  2024 Q3-Q4: EV 수요 둔화 가시화, 중국 배터리 공급과잉, 리튬 가격 저점",
              "  → Materials(리튬·코발트·니켈 소재) 기업 뉴스는 '추가 하락 vs 반등' 기대로 양극화",
              "  → GNN이 이 시기 Mat+Upstream 감성 z-score를 통해 가격 방향을 효과적으로 포착",
              "  → 관세 레짐(2025-01~) 이후: 거시 관세 불확실성이 감성 신호를 희석",
              "",
              "  [검증 필요 사항]",
              f"  N=25 (월 6~7개 기업 × 4개월)로 표본이 매우 작음",
              "  더 긴 기간의 Mat+Upstream 데이터가 있다면 신뢰도 향상 가능",
              "  현재로선 흥미로운 탐색적 발견 수준 (exploratory)"]

    # ── 6. 전체 test 기간 Mat+Upstream 월별 IC 추이 ────────────────
    lines += ["", "■ 6. 전체 test 기간 Mat+Upstream IC 추이 (2024-09~2025-11)", ""]
    mat_all = test[test["is_mat_ups"]]
    lines.append(f"  {'월':<10} {'N':>4} {'IC':>8}  {'비고'}")
    lines.append("  " + "-"*40)
    for ym, grp in mat_all.groupby("year_month"):
        ic_m, _ = spearmanr(grp["prob"], grp["actual_ar"]) if len(grp)>=3 else (np.nan,np.nan)
        regime = "Pre-tariff" if ym <= PRE_TARIFF_END else "Tariff"
        lines.append(f"  {ym:<10} {len(grp):>4} {ic_m:>+7.4f}  [{regime}]")

    lines += ["", SEP]
    report = "\n".join(lines)
    OUT.write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
