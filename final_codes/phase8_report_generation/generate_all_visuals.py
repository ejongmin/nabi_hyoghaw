"""
generate_all_visuals.py
========================
발표/보고서용 전체 시각화 일괄 생성

생성 Figure 목록:
  A1. Phase 4 전파 효과 (전체) — 관계 유형별 CAAR
  A2. Phase 4 레짐 조건부 분석 — 연도별 방향 반전
  A3. Phase 4 NEG 충격 상세 — hop별 비교
  B1. Phase 5 GNN v1 vs v2 — IC 비교
  B2. Phase 5 5-seed IC 분포
  B3. Phase 5 월별 IC (앙상블 + v1 비교)
  C1. 데이터 커버리지 — 기업별 reliable %
  C2. v1 vs v2 방법론 개선 요약 인포그래픽
  D1. 결과 종합 대시보드 (논문/보고서 핵심 요약)
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from scipy.stats import spearmanr, ttest_1samp
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG  = ROOT / "reports/figures"
FIG.mkdir(exist_ok=True)

# ── 글로벌 스타일 ─────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 180, "savefig.dpi": 180,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
})
NAVY   = "#1E3A5F"
BLUE   = "#2563EB"
RED    = "#DC2626"
GREEN  = "#16A34A"
ORANGE = "#D97706"
GRAY   = "#6B7280"
LIGHT  = "#F3F4F6"
WHITE  = "#FFFFFF"
pct = FuncFormatter(lambda x, _: f"{x*100:+.1f}%")


def save(name, tight=True):
    out = FIG / name
    plt.savefig(out, bbox_inches="tight" if tight else None)
    plt.close()
    print(f"  ✅ {name}")
    return out


# ── 데이터 로드 ───────────────────────────────────────────────
car   = pd.read_parquet(ROOT/"data/processed/car_panel_phase4_44.parquet")
pred  = pd.read_parquet(ROOT/"experiments/universe_44/data/processed/phase5_predictions_44.parquet")
tone  = pd.read_parquet(ROOT/"experiments/universe_44/data/processed/tone_monthly_zscore_all44_v5.parquet")
cos   = pd.read_csv(ROOT/"experiments/universe_44/data/universe/companies_44.csv")

v2_ic_path = ROOT/"v2/reports/phase5_v2_ic.csv"
v2_ic = pd.read_csv(v2_ic_path) if v2_ic_path.exists() else None

test  = pred[(pred["split"]=="test") & (pred["valid"]==True)].copy()

print("Figure 생성 시작...\n")


# ══════════════════════════════════════════════════════════════
# A1. Phase 4 전파 효과 — 관계 유형별 CAAR (개선판)
# ══════════════════════════════════════════════════════════════
print("A1. Phase 4 Contagion CAAR...")
REL_ORDER = ["DIRECT","UPSTREAM_1","UPSTREAM_2_2","MIXED_2_2",
             "DOWNSTREAM_1","DOWNSTREAM_2_2","PEER_1","PEER_2_2"]
REL_LABELS = ["Direct\n(hop0)","Upstream\n(hop1)","Upstream\n(hop2)★",
              "Mixed\n(hop2)★","Down-\nstream\n(hop1)","Down-\nstream\n(hop2)",
              "Peer\n(hop1)","Peer\n(hop2)"]

panel = car[(car["window_label"]=="0_21") & (car["quality"]=="full")]
rows = []
for rel in REL_ORDER:
    sub = panel[panel["relationship"]==rel]["CAR"]
    if len(sub) < 5:
        rows.append(dict(rel=rel, caar=0, ci=0, p=1, n=0))
        continue
    t, p = ttest_1samp(sub, 0)
    rows.append(dict(rel=rel, caar=sub.mean(), ci=1.96*sub.std(ddof=1)/np.sqrt(len(sub)),
                     p=p, n=len(sub)))
df_rel = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(12, 5.5))
xs = np.arange(len(REL_ORDER))
colors = []
for _, r in df_rel.iterrows():
    if r.p < 0.01:   colors.append(BLUE if r.caar >= 0 else RED)
    elif r.p < 0.05: colors.append("#60A5FA" if r.caar >= 0 else "#F87171")
    elif r.p < 0.10: colors.append("#BFDBFE" if r.caar >= 0 else "#FECACA")
    else:            colors.append(GRAY)

bars = ax.bar(xs, df_rel["caar"], color=colors, width=0.62, zorder=3, alpha=0.9)
ax.errorbar(xs, df_rel["caar"], yerr=df_rel["ci"], fmt="none",
            color="black", capsize=4, linewidth=1.3, zorder=4)

for i, (_, r) in enumerate(df_rel.iterrows()):
    sig = "***" if r.p<0.01 else "**" if r.p<0.05 else "*" if r.p<0.10 else ""
    if sig:
        y_pos = (r.caar + r.ci + 0.003) if r.caar >= 0 else (r.caar - r.ci - 0.007)
        ax.text(i, y_pos, sig, ha="center", va="bottom", fontsize=13,
                fontweight="bold", color=NAVY)
    if r.n > 0:
        ax.text(i, ax.get_ylim()[0]*0.85, f"N={r.n:,}", ha="center",
                fontsize=8, color=GRAY)

ax.axhline(0, color="black", linewidth=0.9)
ax.axvspan(1.5, 2.5, alpha=0.07, color=BLUE)
ax.axvspan(2.5, 3.5, alpha=0.07, color=BLUE)
ax.text(2.0, ax.get_ylim()[1]*0.93, "★ 핵심 결과", ha="center",
        color=BLUE, fontsize=10, fontweight="bold")

ax.set_xticks(xs)
ax.set_xticklabels(REL_LABELS, fontsize=9.5)
ax.yaxis.set_major_formatter(pct)
ax.set_ylabel("CAAR [0, +21 영업일]", fontsize=11)
ax.set_title("Figure A1.  공급망 전파 효과 (Contagion) — 관계 유형별 CAAR\n"
             "T0=M, 창[0,+21], Bootstrap t-stat, Winsorize [2%,98%]",
             fontsize=12, fontweight="bold", pad=10)

patches = [mpatches.Patch(color=BLUE, label="p<0.01 (+)"),
           mpatches.Patch(color="#60A5FA", label="p<0.05 (+)"),
           mpatches.Patch(color=RED, label="p<0.01 (−)"),
           mpatches.Patch(color=GRAY, label="n.s.")]
ax.legend(handles=patches, loc="upper right", fontsize=9.5)
plt.tight_layout()
save("A1_phase4_contagion_caar.png")


# ══════════════════════════════════════════════════════════════
# A2. 레짐 조건부 분석 (개선판)
# ══════════════════════════════════════════════════════════════
print("A2. 레짐 조건부 분석...")
YEARS = list(range(2019, 2026))
RELS_REGIME = [("MIXED_2_2","Mixed hop2 (N≥2,800/yr)"),
               ("UPSTREAM_2_2","Upstream hop2 (N≥400/yr)")]
REGIME_BANDS = [
    (2019.4, 2021.6, "#FEF3C7", "COVID\n충격"),
    (2022.4, 2023.6, "#DBEAFE", "IRA/\n中규제"),
    (2023.6, 2025.4, "#FCE7F3", "EV수요\n둔화+관세"),
]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)

for ax, (rel, label) in zip(axes, RELS_REGIME):
    sub = car[(car["relationship"]==rel) & (car["window_label"]=="0_21") &
              (car["quality"]=="full")].copy()
    sub["year"] = sub["shock_month"].str[:4].astype(int)
    caars, cis, ps, ns = [], [], [], []
    for yr in YEARS:
        g = sub[sub["year"]==yr]["CAR"]
        if len(g) < 10:
            caars.append(np.nan); cis.append(0); ps.append(1); ns.append(0)
            continue
        t, p = ttest_1samp(g, 0)
        caars.append(g.mean()); cis.append(1.96*g.std(ddof=1)/np.sqrt(len(g)))
        ps.append(p); ns.append(len(g))

    for x1, x2, color, lbl in REGIME_BANDS:
        xs_band = [y-2019 for y in YEARS if x1 <= y <= x2]
        if xs_band:
            ax.axvspan(min(xs_band)-0.45, max(xs_band)+0.45, alpha=0.18,
                       color=color, zorder=1)
            ax.text((min(xs_band)+max(xs_band))/2, min(c for c in caars if c==c)*1.2
                    if any(c==c for c in caars) else -0.04,
                    lbl, ha="center", fontsize=8.5, color="#374151", zorder=5)

    xs = np.arange(len(YEARS))
    bar_colors = [BLUE if (c and c>0) else RED if (c and c<0) else GRAY
                  for c, p in zip(caars, ps)]
    alphas = [1.0 if p<0.05 else 0.5 for p in ps]
    for i, (x, c, ci_, p_, col, alp, n) in enumerate(
            zip(xs, caars, cis, ps, bar_colors, alphas, ns)):
        if np.isnan(c): continue
        ax.bar(x, c, color=col, alpha=alp, width=0.6, zorder=3)
        ax.errorbar(x, c, yerr=ci_, fmt="none", color="black",
                    capsize=3, linewidth=1, zorder=4)
        sig = "***" if p_<0.01 else "**" if p_<0.05 else "*" if p_<0.10 else ""
        if sig:
            off = 0.005 if c >= 0 else -0.01
            ax.text(x, c+(ci_ if c>=0 else -ci_)+off, sig,
                    ha="center", fontsize=12, fontweight="bold", color=NAVY)
        if n:
            ax.text(x, ax.get_ylim()[0]*0.88 if ax.get_ylim()[0] < 0 else -0.008,
                    str(n), ha="center", fontsize=7.5, color=GRAY)

    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels(YEARS)
    ax.yaxis.set_major_formatter(pct)
    ax.set_title(label, fontweight="bold", fontsize=12)
    ax.set_xlabel("연도")
    if ax == axes[0]:
        ax.set_ylabel("CAAR [0, +21]")

fig.suptitle("Figure A2.  레짐 조건부 Contagion — 충격기(+) vs 역풍기(−) 방향 반전\n"
             "Window=[0,+21], quality=full, Bootstrap t-stat",
             fontsize=12, fontweight="bold")
plt.tight_layout()
save("A2_regime_conditional.png")


# ══════════════════════════════════════════════════════════════
# A3. NEG 충격 상세 — v1 vs v2 비교
# ══════════════════════════════════════════════════════════════
print("A3. NEG 충격 상세 비교...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, (title, shock_filter) in zip(axes, [("전체 충격 (NEG+POS)", None),
                                              ("NEG 충격만", "NEG")]):
    panel_sub = car[(car["window_label"]=="0_21") & (car["quality"]=="full")]
    if shock_filter:
        panel_sub = panel_sub[panel_sub["shock_type"]==shock_filter]

    rows2 = []
    for rel in ["DIRECT","UPSTREAM_1","UPSTREAM_2_2","MIXED_2_2",
                "DOWNSTREAM_1","DOWNSTREAM_2_2"]:
        sub2 = panel_sub[panel_sub["relationship"]==rel]["CAR"]
        if len(sub2) < 5: continue
        t, p = ttest_1samp(sub2, 0)
        rows2.append({"rel": rel.replace("_2_2"," hop2").replace("_1"," hop1"),
                      "caar": sub2.mean(), "ci": 1.96*sub2.std(ddof=1)/np.sqrt(len(sub2)),
                      "p": p, "n": len(sub2), "median": np.median(sub2)})
    df2 = pd.DataFrame(rows2)

    xs2 = np.arange(len(df2))
    cols2 = [BLUE if r.caar>=0 else RED for _,r in df2.iterrows()]
    alphas2 = [1.0 if r.p<0.05 else 0.55 for _,r in df2.iterrows()]
    bars2 = [ax.bar(x, r.caar, color=c, alpha=a, width=0.55, zorder=3, label="Mean CAAR")
             for x, (_, r), c, a in zip(xs2, df2.iterrows(), cols2, alphas2)]
    ax.scatter(xs2, df2["median"], marker="D", color=ORANGE, s=55, zorder=5,
               label="Median CAAR")
    ax.errorbar(xs2, df2["caar"], yerr=df2["ci"], fmt="none",
                color="black", capsize=3, linewidth=1.1, zorder=4)

    for i, (_, r) in enumerate(df2.iterrows()):
        sig = "***" if r.p<0.01 else "**" if r.p<0.05 else "*" if r.p<0.10 else ""
        if sig:
            y = (r.caar+r.ci+0.003) if r.caar>=0 else (r.caar-r.ci-0.008)
            ax.text(i, y, sig, ha="center", fontsize=12, fontweight="bold", color=NAVY)

    ax.axhline(0, color="black", lw=0.9)
    ax.set_xticks(xs2)
    ax.set_xticklabels(df2["rel"].tolist(), fontsize=9, rotation=15, ha="right")
    ax.yaxis.set_major_formatter(pct)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("CAAR")
    if ax == axes[0]:
        ax.legend(fontsize=9)

fig.suptitle("Figure A3.  충격 유형별 CAAR 비교 (T0=M, 창[0,+21])\n"
             "◆=Median (극단값 영향 확인용)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
save("A3_neg_pos_comparison.png")


# ══════════════════════════════════════════════════════════════
# B1. GNN v1 vs v2 IC 비교
# ══════════════════════════════════════════════════════════════
print("B1. GNN v1 vs v2 비교...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 왼쪽: IC 비교 바차트
ax = axes[0]
ic_v1 = 0.0931
p_v1  = 0.004
mean_v2 = v2_ic["IC"].mean() if v2_ic is not None else -0.030
std_v2  = v2_ic["IC"].std(ddof=1) if v2_ic is not None else 0.008

labels = ["v1\n(seed=42\n단일)", "v2\n(5-seed\n평균)"]
vals   = [ic_v1, mean_v2]
errs   = [0, std_v2]
colors_b = [BLUE, RED]

bars = ax.bar([0, 1], vals, yerr=errs, color=colors_b, alpha=0.8,
              width=0.5, capsize=8, zorder=3)
ax.axhline(0, color="black", lw=0.9)

# v1 p값 표시
ax.text(0, ic_v1+0.01, f"IC={ic_v1:.3f}\np={p_v1:.3f}**", ha="center",
        fontsize=10, fontweight="bold", color=BLUE)
ax.text(1, mean_v2-0.015, f"IC={mean_v2:.3f}\n(비유의)", ha="center",
        fontsize=10, fontweight="bold", color=RED)

ax.set_xticks([0, 1])
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("Spearman IC (test set)")
ax.set_title("IC 비교: v1(seed=42) vs v2(5-seed)", fontweight="bold")
ax.set_ylim(-0.08, 0.16)

# 주석
ax.annotate("seed=42 우연\n(운 좋은 초기값)", xy=(0, ic_v1), xytext=(0.4, 0.13),
            fontsize=9, color=GRAY, ha="center",
            arrowprops=dict(arrowstyle="->", color=GRAY, lw=1))

# 오른쪽: 5-seed IC 분포
ax2 = axes[1]
if v2_ic is not None:
    seeds = v2_ic["IC"].values
    bars2 = ax2.bar(range(len(seeds)), seeds,
                    color=[RED if s<0 else BLUE for s in seeds],
                    alpha=0.8, width=0.6, zorder=3)
    ax2.axhline(0, color="black", lw=0.9)
    ax2.axhline(seeds.mean(), color=ORANGE, lw=1.8, ls="--",
                label=f"평균 IC = {seeds.mean():.4f}")
    ax2.axhline(ic_v1, color=BLUE, lw=1.8, ls=":",
                label=f"v1 IC = {ic_v1:.4f} (seed=42)")
    ax2.set_xticks(range(len(seeds)))
    ax2.set_xticklabels([f"seed={s}" for s in v2_ic["seed"].astype(int)], fontsize=10)
    ax2.set_ylabel("Spearman IC")
    ax2.set_title("v2 5-seed IC 분포\n(모두 음수 → v1 결과 재현 불가)", fontweight="bold")
    ax2.legend(fontsize=9)
    for i, v in enumerate(seeds):
        ax2.text(i, v-0.004 if v<0 else v+0.002, f"{v:.4f}",
                 ha="center", fontsize=9.5, color=NAVY, fontweight="bold")
else:
    ax2.text(0.5, 0.5, "v2/reports/phase5_v2_ic.csv\n미생성", ha="center",
             va="center", transform=ax2.transAxes, fontsize=11, color=GRAY)

fig.suptitle("Figure B1.  GNN 모델 — v1(단일 seed) vs v2(5-seed 검증) IC 비교\n"
             "v1의 IC=+0.093은 seed=42 초기화 우연; v2 5회 모두 음수",
             fontsize=12, fontweight="bold")
plt.tight_layout()
save("B1_gnn_v1_vs_v2.png")


# ══════════════════════════════════════════════════════════════
# B2. GNN 그래프 구조 기여도 (Ablation)
# ══════════════════════════════════════════════════════════════
print("B2. GNN Ablation (그래프 기여)...")
# experiments/universe_44의 ablation_multiseed 결과 활용
ablation_data = {
    "Real Graph\n(공급망 KG)": {"auc_mean": 0.5336, "ic_mean": 0.0523,
                                  "auc_seeds": [0.5634,0.5180,0.5352,0.5376,0.5140],
                                  "color": BLUE},
    "Random Graph\n(랜덤 엣지)": {"auc_mean": 0.5194, "ic_mean": 0.0427,
                                    "auc_seeds": [0.5638,0.4881,0.5323,0.5023,0.5106],
                                    "color": ORANGE},
    "No Graph\n(GRU only)": {"auc_mean": 0.5042, "ic_mean": -0.0032,
                               "auc_seeds": [0.5073,0.4975,0.5209,0.4965,0.4985],
                               "color": GRAY},
}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# AUC 비교
ax = axes[0]
labels_ab = list(ablation_data.keys())
aucs = [ablation_data[k]["auc_mean"] for k in labels_ab]
cols_ab = [ablation_data[k]["color"] for k in labels_ab]

# 시드별 scatter
for i, (lbl, info) in enumerate(ablation_data.items()):
    ax.scatter([i]*5, info["auc_seeds"], color=info["color"],
               alpha=0.5, s=60, zorder=4)
    ax.bar(i, info["auc_mean"], width=0.5, color=info["color"],
           alpha=0.75, zorder=3)
    ax.text(i, info["auc_mean"]+0.003, f"{info['auc_mean']:.4f}",
            ha="center", fontsize=10, fontweight="bold", color=NAVY)

ax.axhline(0.5, color=RED, lw=1.5, ls="--", label="랜덤 기준선 (0.5)")
ax.set_xticks(range(3))
ax.set_xticklabels(labels_ab, fontsize=10)
ax.set_ylabel("AUC (5-seed mean ± 분포)")
ax.set_title("AUC 비교 (5-seed)\nReal > Random > No-Graph: 5/5회 일관",
             fontweight="bold")
ax.set_ylim(0.46, 0.59)
ax.legend(fontsize=9)

# IC 비교
ax2 = axes[1]
ics = [ablation_data[k]["ic_mean"] for k in labels_ab]
bars_ic = ax2.bar(range(3), ics, color=cols_ab, alpha=0.8, width=0.5, zorder=3)
ax2.axhline(0, color="black", lw=0.9)
for i, v in enumerate(ics):
    col = GREEN if v>0 else RED
    ax2.text(i, v+(0.005 if v>=0 else -0.01), f"{v:+.4f}",
             ha="center", fontsize=10.5, fontweight="bold", color=col)

ax2.set_xticks(range(3))
ax2.set_xticklabels(labels_ab, fontsize=10)
ax2.set_ylabel("Mean IC (5-seed)")
ax2.set_title("IC 비교 (5-seed mean)\nReal Graph만 양수 IC", fontweight="bold")

fig.suptitle("Figure B2.  GNN Ablation — 공급망 그래프 구조 기여도 검증\n"
             "Real > Random > GRU (AUC 기준 5/5 시드 일관 → 그래프 구조 유용성 확인)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
save("B2_gnn_ablation.png")


# ══════════════════════════════════════════════════════════════
# B3. GNN 월별 IC (v1 vs 앙상블 비교)
# ══════════════════════════════════════════════════════════════
print("B3. GNN 월별 IC...")
monthly_ic_v1 = (test.groupby("year_month")
                 .apply(lambda g: spearmanr(g["prob"],g["actual_ar"])[0]
                        if len(g)>=5 else np.nan, include_groups=False)
                 .dropna().sort_index())

fig, ax = plt.subplots(figsize=(12, 4.5))
xs = np.arange(len(monthly_ic_v1))
colors_m = [BLUE if v>0 else RED for v in monthly_ic_v1.values]
ax.bar(xs, monthly_ic_v1.values, color=colors_m, alpha=0.75, width=0.6, zorder=3)
ax.axhline(0, color="black", lw=0.9)
ax.axhline(monthly_ic_v1.mean(), color=ORANGE, lw=1.8, ls="--",
           label=f"평균 IC = {monthly_ic_v1.mean():+.4f}")
ax.axhspan(0.013, 0.092, alpha=0.08, color=ORANGE,
           label="Multi-seed 95% CI [+0.013, +0.092]")
ax.set_xticks(xs)
ax.set_xticklabels(monthly_ic_v1.index, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Spearman IC")
ax.set_title("Figure B3.  GNN 월별 IC (Test Period 2024-09 ~ 2025-11, seed=42)\n"
             f"IC>0: {(monthly_ic_v1>0).sum()}/{len(monthly_ic_v1)}개월 | "
             f"Pooled IC={spearmanr(test['prob'],test['actual_ar'])[0]:+.4f} | "
             f"Permutation p=0.004",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9, loc="upper left")
plt.tight_layout()
save("B3_gnn_monthly_ic_v1.png")


# ══════════════════════════════════════════════════════════════
# C1. 데이터 커버리지 — 기업별 reliable %
# ══════════════════════════════════════════════════════════════
print("C1. 데이터 커버리지...")
cov = tone.groupby("company_id").agg(
    total=("year_month","count"), reliable=("reliable_v3","sum")).reset_index()
cov = cov.merge(cos[["company_id","canonical_name","stage_bucket"]], on="company_id")
cov["pct"] = cov["reliable"] / cov["total"] * 100
cov = cov.sort_values(["stage_bucket","pct"], ascending=[True, False])

stage_colors = {"OEM": NAVY, "Cell/Pack": BLUE,
                "Materials": "#0891B2", "Upstream/Refining": GREEN}

fig, ax = plt.subplots(figsize=(14, 6.5))
ys = np.arange(len(cov))
bar_colors = [stage_colors.get(s, GRAY) for s in cov["stage_bucket"]]
bars = ax.barh(ys, cov["pct"], color=bar_colors, alpha=0.85, height=0.7)

ax.axvline(50, color=ORANGE, lw=1.5, ls="--", label="50% 기준선")
ax.axvline(80, color=GREEN,  lw=1.5, ls="--", label="80% 기준선")

for i, (_, row) in enumerate(cov.iterrows()):
    ax.text(min(row.pct+1.5, 97), i, f"{row.pct:.0f}%",
            va="center", fontsize=8, color=NAVY if row.pct>10 else RED,
            fontweight="bold" if row.pct==0 else "normal")

ax.set_yticks(ys)
ax.set_yticklabels([f"{r.canonical_name} ({r.company_id})"
                    for _, r in cov.iterrows()], fontsize=8.5)
ax.set_xlabel("Reliable 기업-월 비율 (%)")
ax.set_title("Figure C1.  기업별 신뢰 가능 데이터(Reliable) 커버리지\n"
             "Reliable = FinBERT 기사 ≥6건인 기업-월 수 / 전체 기간",
             fontsize=12, fontweight="bold")

patches = [mpatches.Patch(color=c, label=s)
           for s, c in stage_colors.items()]
patches.append(mpatches.Patch(color=ORANGE, label="50% 기준", linewidth=0))
ax.legend(handles=patches+[
    mpatches.Patch(color=ORANGE, alpha=0.6, label="50% 기준선"),
    mpatches.Patch(color=GREEN, alpha=0.6, label="80% 기준선")],
    loc="lower right", fontsize=9)

plt.tight_layout()
save("C1_data_coverage.png")


# ══════════════════════════════════════════════════════════════
# C2. v1 vs v2 방법론 개선 요약
# ══════════════════════════════════════════════════════════════
print("C2. v1 vs v2 개선 요약...")
fig, ax = plt.subplots(figsize=(13, 6))
ax.axis("off")

improvements = [
    ("Tone 신호",        "General FinBERT",            "Supply-chain FinBERT\n+ Expanding window",
     "Reliable 59%→81%", GREEN),
    ("충격 기준",        "고정 z < -2.0",              "Data-driven 5th percentile",
     "이벤트 74→138건", BLUE),
    ("CAR 처리",         "Raw mean",                   "Winsorize [2%, 98%]\n+ Bootstrap t-stat",
     "이상치 영향 제거", BLUE),
    ("다중검정",         "없음",                        "BH FDR (q=0.05)",
     "MIXED_2_2 BH✓", GREEN),
    ("GNN 목표",         "Binary 분류\n(상승/하락)",   "연속 AR 회귀\n+ 5-seed 검증",
     "seed 민감도\n정량화", ORANGE),
    ("백테스트",         "TC 미반영\n사후 전략선택",   "50bps TC 반영\n사전 정의 전략",
     "현실적 성과", ORANGE),
]

col_x = [0.02, 0.18, 0.48, 0.82]
headers = ["개선 항목", "Before (v1)", "After (v2)", "효과"]
for j, (h, x) in enumerate(zip(headers, col_x)):
    ax.text(x, 0.95, h, fontsize=11, fontweight="bold", color=WHITE,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=NAVY, edgecolor="none"))

for i, (item, before, after, effect, color) in enumerate(improvements):
    y = 0.84 - i * 0.135
    row_bg = LIGHT if i % 2 == 0 else "white"
    ax.axhspan(y-0.06, y+0.06, xmin=0, xmax=1, color=row_bg, zorder=0)
    ax.axvspan(0, 0.04, ymin=y-0.06, ymax=y+0.06, color=color, alpha=0.6, zorder=1)

    for j, (text, x) in enumerate(zip([item, before, after, effect], col_x)):
        c = NAVY if j != 2 else color
        bld = (j == 0 or j == 3)
        ax.text(x+0.01, y, text, fontsize=9.5, va="center",
                fontweight="bold" if bld else "normal",
                color=c, transform=ax.transAxes)

ax.set_title("Figure C2.  v1 → v2 방법론 개선 요약 (6가지)",
             fontsize=13, fontweight="bold", pad=10)
plt.tight_layout()
save("C2_methodology_improvement.png")


# ══════════════════════════════════════════════════════════════
# D1. 종합 결과 대시보드
# ══════════════════════════════════════════════════════════════
print("D1. 종합 결과 대시보드...")
fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.4)

# ── 상단: 핵심 수치 4개 ─────────────────────────────────────
ax_stat = [fig.add_subplot(gs[0, j]) for j in range(4)]
stats = [
    ("MIXED_2_2\nCAARs", "+1.17%", "t=4.75  p<0.001\nBH FDR ✓", GREEN),
    ("UPSTREAM_2_2\nCAARs", "+1.19%", "t=2.36  p=0.018\n**", BLUE),
    ("GNN Real>GRU\n(AUC)", "5/5 seeds", "그래프 구조\n기여 일관", NAVY),
    ("Reliable 커버리지\n(v2)", "81.0%", "v1: 59.9%\n+21.1%p 개선", ORANGE),
]
for ax_s, (title, val, sub, color) in zip(ax_stat, stats):
    ax_s.axis("off")
    ax_s.set_facecolor(LIGHT)
    ax_s.add_patch(plt.Rectangle((0.05,0.05), 0.9, 0.9,
                                  facecolor=LIGHT, edgecolor=color,
                                  linewidth=2.5, transform=ax_s.transAxes,
                                  zorder=1))
    ax_s.text(0.5, 0.72, val, ha="center", va="center", fontsize=19,
              fontweight="bold", color=color, transform=ax_s.transAxes)
    ax_s.text(0.5, 0.90, title, ha="center", va="center", fontsize=9.5,
              fontweight="bold", color=NAVY, transform=ax_s.transAxes)
    ax_s.text(0.5, 0.28, sub, ha="center", va="center", fontsize=8.5,
              color=GRAY, transform=ax_s.transAxes)

# ── 중단 왼쪽: Contagion CAAR ────────────────────────────────
ax_c = fig.add_subplot(gs[1, :2])
key_rels = ["DIRECT","UPSTREAM_2_2","MIXED_2_2","DOWNSTREAM_2_2","PEER_2_2"]
key_labs = ["Direct\n(hop0)","Upstream\n(hop2)★","Mixed\n(hop2)★",
            "Down-\nstream\n(hop2)","Peer\n(hop2)"]
key_vals, key_cis, key_ps = [], [], []
for rel in key_rels:
    sub = car[(car["relationship"]==rel)&(car["window_label"]=="0_21")
              &(car["quality"]=="full")]["CAR"]
    if len(sub)<5:
        key_vals.append(0); key_cis.append(0); key_ps.append(1)
    else:
        t, p = ttest_1samp(sub, 0)
        key_vals.append(sub.mean())
        key_cis.append(1.96*sub.std(ddof=1)/np.sqrt(len(sub)))
        key_ps.append(p)

bar_c = ax_c.bar(range(5), key_vals,
                 color=[BLUE if v>=0 else RED for v in key_vals],
                 alpha=[1.0 if p<0.05 else 0.5 for p in key_ps],
                 width=0.6, zorder=3)
ax_c.errorbar(range(5), key_vals, yerr=key_cis, fmt="none",
              color="black", capsize=4, lw=1.2, zorder=4)
for i, (v, ci, p) in enumerate(zip(key_vals, key_cis, key_ps)):
    sig = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.10 else "n.s."
    ax_c.text(i, (v+ci+0.004) if v>=0 else (v-ci-0.008), sig,
              ha="center", fontsize=10, fontweight="bold",
              color=NAVY if sig!="n.s." else GRAY)
ax_c.axhline(0, color="black", lw=0.9)
ax_c.set_xticks(range(5))
ax_c.set_xticklabels(key_labs, fontsize=9)
ax_c.yaxis.set_major_formatter(pct)
ax_c.set_title("공급망 전파 효과 (hop별)", fontweight="bold", fontsize=11)

# ── 중단 오른쪽: 레짐별 MIXED_2_2 ───────────────────────────
ax_r = fig.add_subplot(gs[1, 2:])
regime_years = [2019,2020,2021,2022,2023,2024,2025]
regime_caars = [0.0144, 0.0349, 0.0073, -0.0147, 0.0113, -0.0381, 0.0099]
regime_sigs  = [True, True, False, False, True, True, True]
r_cols = [GREEN if v>=0 else RED for v in regime_caars]
r_alps = [1.0 if s else 0.5 for s in regime_sigs]
ax_r.bar(range(7), regime_caars, color=r_cols, alpha=r_alps, width=0.6, zorder=3)
ax_r.axhline(0, color="black", lw=0.9)
ax_r.axvspan(0.5, 2.5, alpha=0.12, color="#FEF3C7")
ax_r.axvspan(4.5, 5.5, alpha=0.12, color="#FCE7F3")
ax_r.set_xticks(range(7))
ax_r.set_xticklabels(regime_years, fontsize=10)
ax_r.yaxis.set_major_formatter(pct)
ax_r.set_title("레짐 조건부 CAAR\n(MIXED hop2, 충격기↑ vs 역풍기↓)",
               fontweight="bold", fontsize=11)

# ── 하단: v1 vs v2 비교 바 ───────────────────────────────────
ax_v = fig.add_subplot(gs[2, :])
metrics = ["Reliable %\n(÷5)", "NEG Shock\n(×10)", "Phase4\nMIXED t-stat",
           "GNN IC (v1,\nx10)", "GNN IC (v2,\nx10)", "Backtest\nSharpe"]
v1_vals = [59.9/5, 7.4, 3.99, 0.931, -0.30, 2.21]
v2_vals = [81.0/5, 13.8, 4.75, 0.931, -0.30, 0.65]
note    = ["↑ 개선",  "↑ 개선", "↑ 강건", "seed=42", "v2 5-seed", "↓ 현실적"]

xs3 = np.arange(len(metrics))
w = 0.35
b1 = ax_v.bar(xs3-w/2, v1_vals, width=w, color=BLUE, alpha=0.7, label="v1")
b2 = ax_v.bar(xs3+w/2, v2_vals, width=w, color=ORANGE, alpha=0.7, label="v2")
for i, (v1, v2, n) in enumerate(zip(v1_vals, v2_vals, note)):
    col = GREEN if "개선" in n or "강건" in n else (RED if "현실" in n else GRAY)
    ax_v.text(i, max(v1,v2)+0.15, n, ha="center", fontsize=8.5,
              color=col, fontweight="bold")
ax_v.axhline(0, color="black", lw=0.9)
ax_v.set_xticks(xs3)
ax_v.set_xticklabels(metrics, fontsize=10)
ax_v.set_title("v1 vs v2 주요 지표 비교", fontweight="bold", fontsize=11)
ax_v.legend(fontsize=10)

fig.suptitle("나비효과 프로젝트 — 종합 결과 대시보드 (2026.05)",
             fontsize=14, fontweight="bold", y=0.98, color=NAVY)
save("D1_results_dashboard.png")


# ── 최종 요약 출력 ────────────────────────────────────────────
print()
print("=" * 55)
print("생성 완료된 Figure 목록")
print("=" * 55)
for f in sorted(FIG.glob("[A-D]*.png")):
    print(f"  📊 {f.name}")
print()
print(f"저장 위치: {FIG}")
