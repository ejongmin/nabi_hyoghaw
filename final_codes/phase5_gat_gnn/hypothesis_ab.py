"""
hypothesis_ab.py
=================
가설 A: 충격 비대칭성 — NEG 충격 전파가 POS보다 빠르고 강하다
가설 B: 홉 거리 감쇠 — CAAR = α × exp(-β × hop)

car_panel_phase4_44.parquet (44개 기업, expanding window) 사용
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

ROOT = Path("C:/Users/john9/nabi_hyoghaw")
PANEL_PATH = ROOT / "data/processed/car_panel_phase4_44.parquet"
OUT = ROOT / "reports/hypothesis_ab.txt"

df = pd.read_parquet(PANEL_PATH)
SEP = "=" * 68


# ── 공통 필터 ─────────────────────────────────────────────────
base = df[
    (df["quality"] == "full") &
    (~df["confounding_self"].fillna(False))
].copy()

# hop 거리 매핑
HOP_MAP = {
    "DIRECT": 0,
    "DOWNSTREAM_1": 1, "UPSTREAM_1": 1, "PEER_1": 1,
    "DOWNSTREAM_2_2": 2, "UPSTREAM_2_2": 2, "MIXED_2_2": 2, "PEER_2_2": 2,
    # 18개 기업 버전 호환
    "DOWNSTREAM": 1, "UPSTREAM": 1, "PEER": 1,
    "DOWNSTREAM_2": 2, "UPSTREAM_2": 2, "MIXED_2": 2, "PEER_2": 2,
}
base["hop"] = base["relationship"].map(HOP_MAP)


# ── 가설 A: 충격 비대칭성 ────────────────────────────────────
def test_hypothesis_a(panel):
    """NEG vs POS CAAR 비교 (창별, DIRECT 효과)"""
    direct = panel[panel["relationship"] == "DIRECT"].copy()

    windows = ["0_1", "0_5", "0_10", "0_21"]
    results = []

    for w in windows:
        sub = direct[direct["window_label"] == w]
        for shock in ["NEG", "POS"]:
            s = sub[sub["shock_type"] == shock]["CAR"]
            if len(s) < 3:
                continue
            t, p = stats.ttest_1samp(s, 0)
            results.append({
                "window": w, "shock": shock, "N": len(s),
                "CAAR": s.mean(), "t": t, "p": p,
            })

    # 동일 창에서 NEG vs POS 차이 검정 (Welch t)
    diff_results = []
    for w in windows:
        sub = direct[direct["window_label"] == w]
        neg = sub[sub["shock_type"] == "NEG"]["CAR"]
        pos = sub[sub["shock_type"] == "POS"]["CAR"]
        if len(neg) < 3 or len(pos) < 3:
            continue
        t, p = stats.ttest_ind(neg, pos, equal_var=False)
        diff_results.append({"window": w, "NEG_CAAR": neg.mean(),
                              "POS_CAAR": pos.mean(),
                              "DIFF": neg.mean()-pos.mean(),
                              "t": t, "p": p, "N_neg": len(neg), "N_pos": len(pos)})

    # NEG의 단기 집중도 (조기 수렴 지표)
    # 단기 [0,+5] 대비 장기 [0,+21] 비율로 "전파 속도" 측정
    def concentration_ratio(shock_type):
        s5  = direct[(direct["window_label"]=="0_5")  & (direct["shock_type"]==shock_type)]["CAR"].mean()
        s21 = direct[(direct["window_label"]=="0_21") & (direct["shock_type"]==shock_type)]["CAR"].mean()
        return abs(s5) / max(abs(s21), 1e-10) if abs(s21) > 1e-10 else np.nan

    neg_conc = concentration_ratio("NEG")
    pos_conc = concentration_ratio("POS")

    return pd.DataFrame(results), pd.DataFrame(diff_results), neg_conc, pos_conc


# ── 가설 B: 홉 거리 지수감쇠 ─────────────────────────────────
def test_hypothesis_b(panel):
    """hop별 |CAAR| 및 지수감쇠 회귀"""
    win21 = panel[panel["window_label"] == "0_21"].copy()
    win21 = win21.dropna(subset=["hop"])

    # hop별 CAAR 통계
    hop_stats = []
    for hop in [0, 1, 2]:
        sub = win21[win21["hop"] == hop]["CAR"]
        if len(sub) < 3:
            continue
        t, p = stats.ttest_1samp(sub, 0)
        hop_stats.append({
            "hop": hop, "N": len(sub),
            "CAAR": sub.mean(), "|CAAR|": sub.abs().mean(),
            "t": t, "p": p,
        })

    hop_df = pd.DataFrame(hop_stats)

    # 로그 선형 회귀: ln(|CAAR|) ~ hop
    valid = hop_df[hop_df["|CAAR|"] > 1e-6].copy()
    if len(valid) >= 2:
        valid["log_abs_caar"] = np.log(valid["|CAAR|"])
        res_lr = stats.linregress(valid["hop"].values, valid["log_abs_caar"].values)
        slope, intercept, r_corr, p_reg, se = [float(x) for x in res_lr]
        alpha = float(np.exp(intercept))
        beta  = float(-slope)
    else:
        slope = intercept = r_corr = p_reg = se = alpha = beta = np.nan

    # hop별 NEG vs POS 분리
    breakdown = []
    for hop in [0, 1, 2]:
        for shock in ["NEG", "POS"]:
            sub = win21[(win21["hop"]==hop) & (win21["shock_type"]==shock)]["CAR"]
            if len(sub) < 3: continue
            t, p = stats.ttest_1samp(sub, 0)
            breakdown.append({"hop": hop, "shock": shock,
                               "N": len(sub), "CAAR": sub.mean(),
                               "t": t, "p": p})

    return hop_df, slope, intercept, r_corr, p_reg, alpha, beta, pd.DataFrame(breakdown)


# ── 실행 ─────────────────────────────────────────────────────
res_a, diff_a, neg_conc, pos_conc = test_hypothesis_a(base)
hop_df, slope, intercept, r_corr, p_reg, alpha, beta, breakdown = test_hypothesis_b(base)


def sig(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.10:  return "†"
    return ""


lines = [
    SEP,
    "가설 A·B 검증 — 44개 기업, Expanding Window",
    SEP, "",
    "■ 가설 A: 충격 비대칭성 (NEG 전파가 POS보다 빠르고 강하다)", "",
    "  [A-1] DIRECT 효과 창별 CAAR",
    f"  {'창':<8} {'충격':<5} {'N':>4} {'CAAR':>8} {'t':>7} {'p':>7}",
    "  " + "-"*44,
]
for _, r in res_a.iterrows():
    lines.append(f"  {r['window']:<8} {r['shock']:<5} {r['N']:>4} "
                 f"{r['CAAR']*100:>+7.2f}%  {r['t']:>+6.3f}  "
                 f"{r['p']:>6.4f}{sig(r['p'])}")

lines += ["",
    "  [A-2] 동일 창에서 NEG vs POS 차이 검정 (Welch t)",
    f"  {'창':<8} {'NEG_CAAR':>9} {'POS_CAAR':>9} {'차이':>9} {'t':>7} {'p':>7}",
    "  " + "-"*56,
]
for _, r in diff_a.iterrows():
    lines.append(f"  {r['window']:<8} {r['NEG_CAAR']*100:>+8.2f}%  "
                 f"{r['POS_CAAR']*100:>+8.2f}%  {r['DIFF']*100:>+8.2f}%  "
                 f"{r['t']:>+6.3f}  {r['p']:>6.4f}{sig(r['p'])}")

lines += ["",
    f"  [A-3] 조기 수렴 지표 (|CAAR[0,+5]| / |CAAR[0,+21]|)",
    f"  NEG 조기수렴 비율: {neg_conc:.3f}  ({'NEG가 조기 집중' if neg_conc > pos_conc else 'POS가 조기 집중'})",
    f"  POS 조기수렴 비율: {pos_conc:.3f}",
    "",
    "  해석:",
]
neg_r = res_a[res_a["shock"]=="NEG"].set_index("window")
pos_r = res_a[res_a["shock"]=="POS"].set_index("window")
if "0_5" in neg_r.index and "0_5" in pos_r.index:
    if abs(neg_r.loc["0_5","CAAR"]) > abs(pos_r.loc["0_5","CAAR"]):
        lines.append("  → NEG 충격이 단기([0,+5])에 더 강한 반응 — 가설 A 부분 지지")
    else:
        lines.append("  → POS 충격이 단기([0,+5])에 더 강한 반응 — 가설 A 기각")
lines.append("")

lines += [
    SEP[:68], "",
    "■ 가설 B: 홉 거리 지수감쇠 (CAAR = α × exp(-β × hop))", "",
    "  [B-1] 홉별 CAAR (window [0,+21], all shocks)",
    f"  {'hop':<5} {'N':>5} {'CAAR':>8} {'|CAAR|':>8} {'t':>7} {'p':>7}",
    "  " + "-"*46,
]
for _, r in hop_df.iterrows():
    lines.append(f"  hop={int(r['hop'])}  {r['N']:>5} {r['CAAR']*100:>+7.2f}%  "
                 f"{r['|CAAR|']*100:>7.2f}%  {r['t']:>+6.3f}  {r['p']:>6.4f}{sig(r['p'])}")

lines += ["",
    "  [B-2] 로그 선형 회귀: ln(|CAAR|) ~ hop",
    f"  slope (β): {slope:+.4f}  intercept: {intercept:+.4f}",
    f"  감쇠계수 β: {beta:+.4f}  {'(양수→감쇠)' if beta>0 else '(음수→증폭)'}",
    f"  α (hop=0 추정): {alpha*100:.2f}%",
    f"  R²: {r_corr**2:.4f}   p: {p_reg:.4f}{sig(p_reg)}",
    "",
    "  해석:",
]
if not np.isnan(beta):
    if beta > 0 and p_reg < 0.10:
        lines.append(f"  → β={beta:.3f} > 0 (p={p_reg:.4f}{sig(p_reg)}) → 홉 거리 증가 시 |CAAR| 감쇠 — 가설 B 지지")
    elif beta > 0:
        lines.append(f"  → β={beta:.3f} > 0이지만 비유의 (p={p_reg:.4f}) → 감쇠 경향 있으나 통계적 확신 부족")
    else:
        lines.append(f"  → β={beta:.3f} < 0 → 오히려 hop2에서 증폭 — 가설 B 기각")

lines += ["",
    "  [B-3] 홉 × 충격유형 교차 분석",
    f"  {'hop':<5} {'충격':<5} {'N':>5} {'CAAR':>8} {'t':>7} {'p':>7}",
    "  " + "-"*46,
]
for _, r in breakdown.iterrows():
    lines.append(f"  hop={int(r['hop'])}  {r['shock']:<5} {r['N']:>5} "
                 f"{r['CAAR']*100:>+7.2f}%  {r['t']:>+6.3f}  "
                 f"{r['p']:>6.4f}{sig(r['p'])}")

lines += ["", SEP]
report = "\n".join(lines)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(report, encoding="utf-8")
print("\n" + report)
print(f"\n저장: {OUT}")
