"""C2, D1 Figure 생성"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from scipy.stats import ttest_1samp
from pathlib import Path

ROOT = Path("C:/Users/john9/nabi_hyoghaw")
FIG  = ROOT / "reports/figures"

NAVY,BLUE,RED,GREEN,ORANGE,GRAY,LIGHT,WHITE = (
    "#1E3A5F","#2563EB","#DC2626","#16A34A",
    "#D97706","#6B7280","#F3F4F6","#FFFFFF")
plt.rcParams.update({
    "font.family":"DejaVu Sans","font.size":11,
    "axes.spines.top":False,"axes.spines.right":False,
    "figure.dpi":180,"savefig.dpi":180,
    "axes.grid":True,"grid.alpha":0.3,
})
pct = FuncFormatter(lambda x, _: f"{x*100:+.1f}%")

car  = pd.read_parquet(ROOT/"data/processed/car_panel_phase4_44.parquet")
pred = pd.read_parquet(ROOT/"experiments/universe_44/data/processed/phase5_predictions_44.parquet")
v2ic = pd.read_csv(ROOT/"v2/reports/phase5_v2_ic.csv")

# ── C2: 방법론 개선 ──────────────────────────────────────────
improvements = [
    ("Tone 신호",    "General FinBERT",       "Supply-chain FinBERT\n+ Expanding window",  "Reliable 59%→81%",   GREEN),
    ("충격 기준",    "고정 z < -2.0",          "Data-driven 5th percentile",               "이벤트 74→138건",     BLUE),
    ("CAR 처리",     "Raw mean",               "Winsorize [2%,98%]\n+ Bootstrap t-stat",   "이상치 영향 제거",    BLUE),
    ("다중검정",     "없음",                    "BH FDR (q=0.05)",                          "MIXED_2_2 BH✓",      GREEN),
    ("GNN 목표",     "Binary 분류\n(상승/하락)","연속 AR 회귀\n+ 5-seed 검증",              "seed 민감도\n정량화", ORANGE),
    ("백테스트",     "TC 미반영\n사후 선택",   "50bps TC 반영\n사전 정의 전략",             "현실적 성과",         ORANGE),
]
fig, ax = plt.subplots(figsize=(13, 6))
ax.axis("off")
col_x = [0.02, 0.18, 0.48, 0.82]
for h, x in zip(["개선 항목","Before (v1)","After (v2)","효과"], col_x):
    ax.text(x, 0.95, h, fontsize=11, fontweight="bold", color=WHITE,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=NAVY, edgecolor="none"))
for i, (item, before, after, effect, color) in enumerate(improvements):
    y = 0.84 - i * 0.135
    ax.axhspan(y-0.06, y+0.06, xmin=0, xmax=1,
               color=LIGHT if i%2==0 else "white", zorder=0)
    ax.axvspan(0, 0.04, ymin=y-0.055, ymax=y+0.055,
               color=color, alpha=0.6, zorder=1)
    for j, (text, x) in enumerate(zip([item, before, after, effect], col_x)):
        ax.text(x+0.01, y, text, fontsize=9.5, va="center",
                fontweight="bold" if j in (0, 3) else "normal",
                color=NAVY if j != 2 else color, transform=ax.transAxes)
ax.set_title("Figure C2.  v1 → v2 방법론 개선 요약 (6가지)",
             fontsize=13, fontweight="bold", pad=10)
plt.tight_layout()
plt.savefig(FIG/"C2_methodology_improvement.png", bbox_inches="tight")
plt.close()
print("✅ C2_methodology_improvement.png")

# ── D1: 종합 대시보드 ─────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.45)

# 상단 지표
ax_stat = [fig.add_subplot(gs[0, j]) for j in range(4)]
stats = [
    ("MIXED_2_2\nCAARs",    "+1.17%", "t=4.75  p<0.001\nBH FDR ✓",   GREEN),
    ("UPSTREAM_2_2\nCAARs", "+1.19%", "t=2.36  p=0.018\n**",          BLUE),
    ("GNN Real>GRU\n(AUC)", "5/5 seeds","그래프 구조\n기여 일관",      NAVY),
    ("Reliable 커버리지\n(v2)","81.0%","v1: 59.9%\n+21.1%p 개선",     ORANGE),
]
for ax_s, (title, val, sub, color) in zip(ax_stat, stats):
    ax_s.axis("off")
    ax_s.add_patch(plt.Rectangle(
        (0.05,0.05), 0.9, 0.9, facecolor=LIGHT, edgecolor=color,
        linewidth=2.5, transform=ax_s.transAxes, zorder=1))
    ax_s.text(0.5,0.72, val, ha="center", va="center",
              fontsize=19, fontweight="bold", color=color, transform=ax_s.transAxes)
    ax_s.text(0.5,0.90, title, ha="center", va="center",
              fontsize=9.5, fontweight="bold", color=NAVY, transform=ax_s.transAxes)
    ax_s.text(0.5,0.28, sub, ha="center", va="center",
              fontsize=8.5, color=GRAY, transform=ax_s.transAxes)

# 중단 왼쪽: Contagion
ax_c = fig.add_subplot(gs[1, :2])
key_rels  = ["DIRECT","UPSTREAM_2_2","MIXED_2_2","DOWNSTREAM_2_2","PEER_2_2"]
key_labs  = ["Direct\n(hop0)","Upstream\n(hop2)★","Mixed\n(hop2)★",
             "Downstream\n(hop2)","Peer\n(hop2)"]
panel = car[(car["window_label"]=="0_21") & (car["quality"]=="full")]
kv, kc, kp = [], [], []
for rel in key_rels:
    sub = panel[panel["relationship"]==rel]["CAR"]
    if len(sub)<5: kv.append(0); kc.append(0); kp.append(1)
    else:
        t,p = ttest_1samp(sub,0)
        kv.append(sub.mean()); kc.append(1.96*sub.std(ddof=1)/np.sqrt(len(sub))); kp.append(p)
for i5,(v5,ci5,p5) in enumerate(zip(kv,kc,kp)):
    ax_c.bar(i5, v5, color=BLUE if v5>=0 else RED,
             alpha=1.0 if p5<0.05 else 0.5, width=0.6, zorder=3)
ax_c.errorbar(range(5), kv, yerr=kc, fmt="none", color="black", capsize=4, lw=1.2, zorder=4)
for i,(v,ci,p) in enumerate(zip(kv,kc,kp)):
    sig = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.10 else "n.s."
    ax_c.text(i, (v+ci+0.004) if v>=0 else (v-ci-0.008), sig,
              ha="center", fontsize=10, fontweight="bold",
              color=NAVY if sig!="n.s." else GRAY)
ax_c.axhline(0, color="black", lw=0.9)
ax_c.set_xticks(range(5)); ax_c.set_xticklabels(key_labs, fontsize=9)
ax_c.yaxis.set_major_formatter(pct)
ax_c.set_title("공급망 전파 효과 (hop별)", fontweight="bold", fontsize=11)

# 중단 오른쪽: 레짐 조건부
ax_r = fig.add_subplot(gs[1, 2:])
ry = [2019,2020,2021,2022,2023,2024,2025]
rv = [0.0144,0.0349,0.0073,-0.0147,0.0113,-0.0381,0.0099]
rs = [True,True,False,False,True,True,True]
for i7,(v7,s7) in enumerate(zip(rv,rs)):
    ax_r.bar(i7, v7, color=GREEN if v7>=0 else RED,
             alpha=1.0 if s7 else 0.5, width=0.6, zorder=3)
ax_r.axhline(0, color="black", lw=0.9)
ax_r.axvspan(0.5,2.5,alpha=0.12,color="#FEF3C7")
ax_r.axvspan(4.5,5.5,alpha=0.12,color="#FCE7F3")
ax_r.set_xticks(range(7)); ax_r.set_xticklabels(ry, fontsize=10)
ax_r.yaxis.set_major_formatter(pct)
ax_r.set_title("레짐 조건부 CAAR (MIXED hop2)", fontweight="bold", fontsize=11)

# 하단: v1 vs v2 비교
ax_v = fig.add_subplot(gs[2, :])
met = ["Reliable %\n(÷5)","NEG Shock\n(×10)","Phase4\nMIXED t","GNN IC(v1\n×10)","GNN IC(v2\n×10)","Backtest\nSharpe"]
v1f = [59.9/5, 7.4, 3.99, 0.931, -0.30, 2.21]
v2f = [81.0/5, 13.8, 4.75, 0.931, -0.30, 0.65]
notes = ["↑ 개선","↑ 개선","↑ 강건","seed=42","v2 5-seed","↓ 현실적"]
xs3 = np.arange(len(met))
ax_v.bar(xs3-0.18, v1f, width=0.35, color=BLUE, alpha=0.75, label="v1")
ax_v.bar(xs3+0.18, v2f, width=0.35, color=ORANGE, alpha=0.75, label="v2")
for i,(a,b,n) in enumerate(zip(v1f,v2f,notes)):
    c = GREEN if "개선" in n or "강건" in n else (RED if "현실" in n else GRAY)
    ax_v.text(i, max(a,b)+0.1, n, ha="center", fontsize=8.5, color=c, fontweight="bold")
ax_v.axhline(0, color="black", lw=0.9)
ax_v.set_xticks(xs3); ax_v.set_xticklabels(met, fontsize=10)
ax_v.set_title("v1 vs v2 주요 지표 비교", fontweight="bold", fontsize=11)
ax_v.legend(fontsize=10)

fig.suptitle("나비효과 프로젝트 — 종합 결과 대시보드 (2026.05)",
             fontsize=14, fontweight="bold", y=0.99, color=NAVY)
plt.savefig(FIG/"D1_results_dashboard.png", bbox_inches="tight")
plt.close()
print("✅ D1_results_dashboard.png")

print("\n생성된 전체 Figure:")
for f in sorted(FIG.glob("[A-D]*.png")):
    print(f"  {f.name}")
