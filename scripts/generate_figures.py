"""
나비효과 프로젝트 — 전체 시각화 생성
출력: reports/figures/01_kg_supply_chain.png
      reports/figures/02_phase4_car_results.png
      reports/figures/03_phase5_gat_results.png
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import networkx as nx
from pathlib import Path

matplotlib.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]

OUT = Path("reports/figures")
OUT.mkdir(parents=True, exist_ok=True)

# ── 공통 데이터 로드 ──
c18   = pd.read_csv("data/universe/companies_18.csv")
edges = pd.read_csv("data/seed/seed_edges_18_internal.csv")
car   = pd.read_parquet("data/processed/car_panel_phase4.parquet")
attn  = pd.read_csv("reports/phase5_attention_weights.csv")
ic    = pd.read_csv("reports/phase5_ic_by_month.csv")

nm    = dict(zip(c18["company_id"], c18["canonical_name"]))
stage = dict(zip(c18["company_id"], c18["stage_bucket"]))

STAGE_COLOR = {
    "Upstream/Refining": "#D62728",
    "Materials":          "#FF7F0E",
    "Cell/Pack":          "#1F77B4",
    "OEM":                "#2CA02C",
}
BG    = "#0D1117"
PANEL = "#161B22"
GRID  = "#30363D"
WHITE = "#E6EDF3"


# ═══════════════════════════════════════════════════════════════
# Figure 1: 공급망 지식 그래프
# ═══════════════════════════════════════════════════════════════
print("Figure 1 생성 중...")
fig, ax = plt.subplots(figsize=(18, 11))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

G = nx.DiGraph()
for cid in c18["company_id"]:
    G.add_node(cid)

REL_COLOR = {
    "SUPPLIES":      "#58A6FF",
    "PARTNERS_WITH": "#3FB950",
    "COMPETES_WITH": "#F85149",
    "BUYS_FROM":     "#D2A8FF",
    "OWNS":          "#FFA657",
}

for _, r in edges.iterrows():
    s, d = r["src_company_id"], r["dst_company_id"]
    if s in G and d in G:
        G.add_edge(s, d, rel_type=r["rel_type"],
                   weight=float(r.get("strength", 0.5)))

# 레이아웃: 단계별 Y축 고정
stage_y = {"Upstream/Refining": 3.0, "Materials": 2.0, "Cell/Pack": 1.0, "OEM": 0.0}
pos = {}
for s, y in stage_y.items():
    nodes_s = [cid for cid in c18["company_id"] if stage.get(cid) == s]
    n = len(nodes_s)
    for i, cid in enumerate(sorted(nodes_s)):
        pos[cid] = ((i - (n - 1) / 2) * 2.5, y)

# 엣지 그리기 (rel_type별)
for rel, color in REL_COLOR.items():
    el = [(u, v) for u, v, d in G.edges(data=True) if d["rel_type"] == rel]
    if not el:
        continue
    nx.draw_networkx_edges(
        G, pos, edgelist=el, ax=ax,
        edge_color=color, alpha=0.75, arrows=True,
        arrowsize=14, arrowstyle="-|>", width=1.5,
        node_size=2000, connectionstyle="arc3,rad=0.06",
    )

# 노드
for cid in G.nodes():
    col = STAGE_COLOR.get(stage.get(cid, "OEM"), "#888")
    nx.draw_networkx_nodes(G, pos, nodelist=[cid], ax=ax,
                           node_color=col, node_size=2000, alpha=0.95)

# 라벨
short = {}
for cid in G.nodes():
    name = nm.get(cid, cid)
    name = (name.replace("Advanced Material", "Adv.Mat.")
                .replace("Energy Solution", "Eng.Sol.")
                .replace("High-Tech", "H-T")
                .replace("Mercedes-Benz", "Mercedes"))
    short[cid] = name if len(name) <= 10 else name[:10] + ".."

nx.draw_networkx_labels(G, pos, labels=short, ax=ax,
                        font_size=7, font_color="white", font_weight="bold")

# 단계 레이블
for s, y in stage_y.items():
    col = STAGE_COLOR.get(s, WHITE)
    ax.text(-15, y, s, fontsize=10, color=col, fontweight="bold",
            va="center", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc=PANEL, ec=col, lw=1.5))

# 범례 (엣지 유형)
handles = [mpatches.Patch(color=c, label=r) for r, c in REL_COLOR.items()]
ax.legend(handles=handles, loc="lower right", fontsize=9,
          facecolor=PANEL, edgecolor=GRID, labelcolor=WHITE,
          title="Edge Type", title_fontsize=9)

ax.set_title("Nabi Effect — EV Battery Supply Chain Knowledge Graph  |  18 Companies, 65 Edges",
             fontsize=13, color=WHITE, fontweight="bold", pad=12)
ax.axis("off")
plt.tight_layout()
plt.savefig(OUT / "01_kg_supply_chain.png", dpi=150,
            bbox_inches="tight", facecolor=BG)
plt.close()
print("  저장: 01_kg_supply_chain.png")


# ═══════════════════════════════════════════════════════════════
# Figure 2: Phase 4 CAR Event Study (2x2 패널)
# ═══════════════════════════════════════════════════════════════
print("Figure 2 생성 중...")
fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor(BG)
gs2 = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

direct_full = car[
    (car["relationship"] == "DIRECT") &
    (car["quality"] == "full") &
    (~car["confounding_self"])
].copy()

# ─ (A) 창별 CAAR ─
ax = fig.add_subplot(gs2[0, 0])
ax.set_facecolor(PANEL)
win_order  = ["m1_1", "0_1", "0_5", "0_10", "0_21", "m5_5"]
win_labels = ["[-1,+1]", "[0,+1]", "[0,+5]", "[0,+10]", "[0,+21]", "[-5,+5]"]
caars, cis, ns = [], [], []
for w in win_order:
    sub = direct_full[direct_full["window_label"] == w]["CAR"]
    n = len(sub)
    ns.append(n)
    caars.append(sub.mean() * 100 if n > 0 else 0)
    cis.append(1.96 * sub.std() / np.sqrt(n) * 100 if n > 1 else 0)

cols_a = ["#F85149" if c < 0 else "#3FB950" for c in caars]
bars_a = ax.bar(win_labels, caars, color=cols_a, alpha=0.82, width=0.58,
                edgecolor=GRID, linewidth=0.8)
ax.errorbar(win_labels, caars, yerr=cis, fmt="none",
            color=WHITE, capsize=5, linewidth=1.8)
for b, n in zip(bars_a, ns):
    y_pos = b.get_height() + (max(cis) * 0.15 if caars[bars_a.index(b)] >= 0 else -max(cis)*0.15)
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.4,
            f"N={n}", ha="center", va="bottom", fontsize=8, color="#8B949E")
ax.axhline(0, color=GRID, linewidth=0.9)
ax.set_ylabel("CAAR (%)", color=WHITE, fontsize=9)
ax.set_title("(A) CAAR by Event Window\n(Direct, Full Quality)", color=WHITE,
             fontsize=10, fontweight="bold")
ax.tick_params(colors=WHITE, labelsize=8)
for sp in ax.spines.values():
    sp.set_color(GRID)
plt.setp(ax.get_xticklabels(), rotation=25, ha="right")

# ─ (B) NEG vs POS 분포 ─
ax = fig.add_subplot(gs2[0, 1])
ax.set_facecolor(PANEL)
sub_main = direct_full[direct_full["window_label"] == "0_21"]
for shock_type, color, label in [
    ("NEG", "#F85149", "Negative Shock"),
    ("POS", "#3FB950", "Positive Shock"),
]:
    g = sub_main[sub_main["shock_type"] == shock_type]["CAR"]
    if len(g) < 2:
        continue
    ax.hist(g * 100, bins=10, color=color, alpha=0.6,
            label=f"{label}  N={len(g)}  CAAR={g.mean()*100:+.1f}%",
            edgecolor="white", linewidth=0.5)
    ax.axvline(g.mean() * 100, color=color, linewidth=2.2, linestyle="--")
ax.axvline(0, color=WHITE, linewidth=0.8, alpha=0.4)
ax.set_xlabel("CAR (%)", color=WHITE, fontsize=9)
ax.set_ylabel("Count", color=WHITE, fontsize=9)
ax.set_title("(B) CAR Distribution by Shock Type  [0,+21]",
             color=WHITE, fontsize=10, fontweight="bold")
ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=WHITE)
ax.tick_params(colors=WHITE, labelsize=8)
for sp in ax.spines.values():
    sp.set_color(GRID)

# ─ (C) 공급망 전파 ─
ax = fig.add_subplot(gs2[1, 0])
ax.set_facecolor(PANEL)
contagion_df = car[
    (car["window_label"] == "0_21") &
    (car["quality"] == "full") &
    (~car["confounding_target"])
].copy()

def norm_rel(r):
    if r == "DIRECT":
        return "DIRECT"
    parts = r.split("_")
    hop = parts[-1]
    direction = "_".join(parts[:-1])
    return f"{direction} (hop{hop})" if hop in ("1", "2") else r

contagion_df["rel_group"] = contagion_df["relationship"].apply(norm_rel)
rel_stats = (contagion_df.groupby("rel_group")["CAR"]
             .agg(["mean", "count"]).reset_index())
rel_stats = rel_stats[rel_stats["count"] >= 5].sort_values("mean", ascending=True)

col_c = ["#F85149" if m < 0 else "#58A6FF" for m in rel_stats["mean"]]
bars_c = ax.barh(rel_stats["rel_group"], rel_stats["mean"] * 100,
                  color=col_c, alpha=0.82, height=0.6,
                  edgecolor=GRID, linewidth=0.8)
for b, n in zip(bars_c, rel_stats["count"]):
    ax.text(b.get_width() + 0.1, b.get_y() + b.get_height() / 2,
            f"N={n}", va="center", fontsize=8, color="#8B949E")
ax.axvline(0, color=GRID, linewidth=0.9)
ax.set_xlabel("CAAR (%)", color=WHITE, fontsize=9)
ax.set_title("(C) Supply Chain Contagion — CAAR by Relationship [0,+21]",
             color=WHITE, fontsize=10, fontweight="bold")
ax.tick_params(colors=WHITE, labelsize=8)
for sp in ax.spines.values():
    sp.set_color(GRID)

# ─ (D) 단계별 CAAR ─
ax = fig.add_subplot(gs2[1, 1])
ax.set_facecolor(PANEL)
stage_df = sub_main.copy()
stage_df["shock_stage"] = stage_df["shock_company"].map(stage)
sg = (stage_df.groupby("shock_stage")["CAR"]
     .agg(["mean", "std", "count"]).reset_index())
sg = sg[sg["count"] >= 2]
sg["ci"] = 1.96 * sg["std"] / np.sqrt(sg["count"])
sg = sg.sort_values("mean", ascending=False)

col_d = [STAGE_COLOR.get(s, "#888") for s in sg["shock_stage"]]
bars_d = ax.bar(sg["shock_stage"], sg["mean"] * 100,
                 color=col_d, alpha=0.85, width=0.52,
                 edgecolor=GRID, linewidth=0.8)
ax.errorbar(sg["shock_stage"], sg["mean"] * 100,
            yerr=sg["ci"] * 100, fmt="none",
            color=WHITE, capsize=5, linewidth=1.8)
for b, n in zip(bars_d, sg["count"]):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.4,
            f"N={n}", ha="center", va="bottom", fontsize=8, color="#8B949E")
ax.axhline(0, color=GRID, linewidth=0.9)
ax.set_ylabel("CAAR (%)", color=WHITE, fontsize=9)
ax.set_title("(D) CAAR by Value Chain Stage  [0,+21]",
             color=WHITE, fontsize=10, fontweight="bold")
ax.tick_params(colors=WHITE, labelsize=8)
for sp in ax.spines.values():
    sp.set_color(GRID)
plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

fig.suptitle("Nabi Effect — Phase 4: CAR Event Study Results",
             fontsize=14, color=WHITE, fontweight="bold", y=1.01)
plt.savefig(OUT / "02_phase4_car_results.png", dpi=150,
            bbox_inches="tight", facecolor=BG)
plt.close()
print("  저장: 02_phase4_car_results.png")


# ═══════════════════════════════════════════════════════════════
# Figure 3: Phase 5 GAT (어텐션 + IC)
# ═══════════════════════════════════════════════════════════════
print("Figure 3 생성 중...")
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.patch.set_facecolor(BG)

# ─ 왼쪽: 어텐션 상위 엣지 ─
ax = axes[0]
ax.set_facecolor(PANEL)

attn_top = attn.head(15).copy()
attn_top["src_short"] = attn_top["src_company_id"].map(
    lambda x: nm.get(x, x)[:12])
attn_top["dst_short"] = attn_top["dst_company_id"].map(
    lambda x: nm.get(x, x)[:12])
attn_top["label"] = attn_top["src_short"] + " →\n" + attn_top["dst_short"]
attn_top = attn_top.sort_values("attn_combined", ascending=True)

vals = attn_top["attn_combined"].values
norm_v = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
colors5 = plt.cm.YlOrRd(norm_v * 0.85 + 0.1)

bars5 = ax.barh(range(len(attn_top)), attn_top["attn_combined"],
                 color=colors5, alpha=0.88, height=0.7,
                 edgecolor=GRID, linewidth=0.5)
ax.set_yticks(range(len(attn_top)))
ax.set_yticklabels(attn_top["label"].values, fontsize=8, color=WHITE)
for b, val in zip(bars5, attn_top["attn_combined"]):
    ax.text(b.get_width() + 0.002, b.get_y() + b.get_height()/2,
            f"{val:.4f}", va="center", fontsize=7, color="#8B949E")
ax.set_xlabel("Attention Weight (avg over test period)", color=WHITE, fontsize=9)
ax.set_title("(A) GAT Attention — Top 15 Supply Chain Edges",
             color=WHITE, fontsize=11, fontweight="bold")
ax.tick_params(axis="x", colors=WHITE, labelsize=8)
for sp in ax.spines.values():
    sp.set_color(GRID)

# ─ 오른쪽: IC 시계열 ─
ax = axes[1]
ax.set_facecolor(PANEL)

ic_gat  = ic[ic["model"] == "TemporalGAT"].reset_index(drop=True)
ic_base = ic[ic["model"] == "BaselineGRU"].reset_index(drop=True)

x = range(len(ic_gat))
if len(ic_gat) > 0:
    ax.fill_between(x, ic_gat["IC"], 0,
                    where=ic_gat["IC"] >= 0, alpha=0.25, color="#3FB950",
                    label="_nolegend_")
    ax.fill_between(x, ic_gat["IC"], 0,
                    where=ic_gat["IC"] < 0, alpha=0.25, color="#F85149",
                    label="_nolegend_")
    ax.plot(x, ic_gat["IC"], color="#58A6FF", lw=2.2, marker="o",
            markersize=6, zorder=5,
            label=f"TemporalGAT  mean IC={ic_gat['IC'].mean():+.3f}")

if len(ic_base) > 0:
    xb = range(len(ic_base))
    ax.plot(xb, ic_base["IC"], color="#FF7F0E", lw=2.0, linestyle="--",
            marker="s", markersize=5, zorder=4,
            label=f"BaselineGRU  mean IC={ic_base['IC'].mean():+.3f}")

ax.axhline(0,    color=WHITE,    lw=0.8, alpha=0.4)
ax.axhline(0.05, color="#3FB950", lw=0.9, linestyle=":", alpha=0.6, label="IC=+0.05")
ax.axhline(-0.05,color="#F85149", lw=0.9, linestyle=":", alpha=0.6, label="IC=-0.05")

if len(ic_gat) > 0:
    step = max(1, len(ic_gat) // 6)
    xticks = list(range(0, len(ic_gat), step))
    ax.set_xticks(xticks)
    ax.set_xticklabels([ic_gat["year_month"].iloc[i] for i in xticks],
                        rotation=30, ha="right", fontsize=8)

ax.set_ylabel("IC (Spearman ρ)", color=WHITE, fontsize=10)
ax.set_title("(B) Monthly IC Time Series (Test Period)",
             color=WHITE, fontsize=11, fontweight="bold")
ax.legend(fontsize=9, facecolor=PANEL, edgecolor=GRID, labelcolor=WHITE,
          loc="upper left")
ax.tick_params(axis="y", colors=WHITE, labelsize=8)
for sp in ax.spines.values():
    sp.set_color(GRID)

fig.suptitle("Nabi Effect — Phase 5: Temporal GAT Results",
             fontsize=14, color=WHITE, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT / "03_phase5_gat_results.png", dpi=150,
            bbox_inches="tight", facecolor=BG)
plt.close()
print("  저장: 03_phase5_gat_results.png")

print(f"\n모든 그래프 저장 완료: {OUT.resolve()}")
