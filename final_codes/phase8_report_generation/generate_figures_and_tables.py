"""
generate_figures_and_tables.py
================================
나비효과 논문용 Figure & 기술통계 테이블 일괄 생성

Figure 1: Phase 4 Contagion — 관계 유형별 CAAR (hop 구조)
Figure 2: Phase 4 Regime-conditional Contagion (연도별 CAAR)
Figure 3: Phase 5 GNN — 월별 IC 추이
Figure 4: Phase 4 CAR 분포 (Direct NEG vs POS)

Table 1: 데이터 & 이벤트 기술통계
Table 2: 44개 기업 유니버스 요약
Table 3: Phase 4 메인 결과 (논문용)
Table 4: Phase 5 모델 성능 비교
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
from scipy.stats import ttest_1samp, spearmanr
from pathlib import Path

ROOT   = Path(__file__).resolve().parents[1]
FIG    = ROOT / "reports/figures"
TABLE  = ROOT / "reports"
FIG.mkdir(exist_ok=True)

# ── 스타일 설정 ────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})
BLUE   = "#2563EB"
RED    = "#DC2626"
GRAY   = "#6B7280"
GREEN  = "#16A34A"
ORANGE = "#D97706"

pct = FuncFormatter(lambda x, _: f"{x*100:+.1f}%")


# ════════════════════════════════════════════════════════════════
# 데이터 로드
# ════════════════════════════════════════════════════════════════
car    = pd.read_parquet(ROOT / "data/processed/car_panel_phase4_44.parquet")
pred   = pd.read_parquet(ROOT / "experiments/universe_44/data/processed/phase5_predictions_44.parquet")
tone   = pd.read_parquet(ROOT / "experiments/universe_44/data/processed/tone_monthly_zscore_all44_v5.parquet")
cos    = pd.read_csv(ROOT / "experiments/universe_44/data/universe/companies_44.csv")
prices = pd.read_parquet(ROOT / "data/processed/prices_daily.parquet")

test_pred = pred[(pred["split"] == "test") & (pred["valid"] == True)].copy()
direct    = car[(car["relationship"] == "DIRECT") &
                (car["window_label"] == "0_21") &
                (car["quality"] == "full") &
                (~car["confounding_self"])].copy()


# ════════════════════════════════════════════════════════════════
# Figure 1: Contagion CAAR by relationship type
# ════════════════════════════════════════════════════════════════
def fig1_contagion():
    REL_ORDER = ["DIRECT", "UPSTREAM_1", "UPSTREAM_2_2",
                 "MIXED_2_2", "DOWNSTREAM_1", "DOWNSTREAM_2_2",
                 "PEER_1", "PEER_2_2"]
    REL_LABELS = ["Direct", "Upstream\n(hop1)", "Upstream\n(hop2)",
                  "Mixed\n(hop2)", "Downstream\n(hop1)", "Downstream\n(hop2)",
                  "Peer\n(hop1)", "Peer\n(hop2)"]

    panel = car[(car["window_label"] == "0_21") & (car["quality"] == "full")]
    rows = []
    for rel in REL_ORDER:
        sub = panel[panel["relationship"] == rel]["CAR"]
        if len(sub) < 5:
            rows.append(dict(rel=rel, caar=np.nan, ci=np.nan, p=np.nan, n=0))
            continue
        t, p = ttest_1samp(sub, 0)
        se   = sub.std(ddof=1) / np.sqrt(len(sub))
        rows.append(dict(rel=rel, caar=sub.mean(), ci=1.96*se, p=p, n=len(sub)))
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 5))
    xs = np.arange(len(REL_ORDER))

    colors = []
    for _, r in df.iterrows():
        if r["p"] < 0.01:   colors.append(BLUE if r["caar"] > 0 else RED)
        elif r["p"] < 0.05: colors.append("#60A5FA" if r["caar"] > 0 else "#F87171")
        elif r["p"] < 0.10: colors.append("#BFDBFE" if r["caar"] > 0 else "#FECACA")
        else:               colors.append(GRAY)

    bars = ax.bar(xs, df["caar"], color=colors, width=0.6, zorder=3)
    ax.errorbar(xs, df["caar"], yerr=df["ci"], fmt="none",
                color="black", capsize=4, linewidth=1.2, zorder=4)

    # 유의성 표시
    for i, (_, r) in enumerate(df.iterrows()):
        if pd.isna(r["p"]): continue
        sig = "***" if r["p"]<0.01 else "**" if r["p"]<0.05 else "*" if r["p"]<0.10 else ""
        if sig:
            y = (r["caar"] + r["ci"]) if r["caar"] >= 0 else (r["caar"] - r["ci"])
            offset = 0.003 if r["caar"] >= 0 else -0.007
            ax.text(i, y + offset, sig, ha="center", va="bottom",
                    fontsize=12, fontweight="bold", color="black")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(2.5, color=GRAY, linewidth=0.8, linestyle="--", alpha=0.5)

    # hop 구분 표시
    ax.text(0, ax.get_ylim()[1]*0.92, "hop1", ha="center", color=GRAY, fontsize=9)
    ax.text(3.5, ax.get_ylim()[1]*0.92, "hop2", ha="center", color=GRAY, fontsize=9)

    ax.set_xticks(xs)
    ax.set_xticklabels(REL_LABELS, fontsize=10)
    ax.yaxis.set_major_formatter(pct)
    ax.set_ylabel("CAAR [0, +21]")
    ax.set_title("Figure 1. Supply Chain Contagion: CAAR by Relationship Type (T0 = M)")
    ax.set_xlabel("Relationship to Event Firm")

    # N 표시
    for i, (_, r) in enumerate(df.iterrows()):
        if r["n"] > 0:
            ax.text(i, ax.get_ylim()[0]*0.85, f"N={r['n']:,}", ha="center",
                    fontsize=8, color=GRAY)

    # legend
    p1 = mpatches.Patch(color=BLUE,    label="p<0.01 (+)")
    p2 = mpatches.Patch(color="#60A5FA", label="p<0.05 (+)")
    p3 = mpatches.Patch(color=RED,     label="p<0.01 (−)")
    p4 = mpatches.Patch(color=GRAY,    label="n.s.")
    ax.legend(handles=[p1, p2, p3, p4], loc="upper right", fontsize=9)

    fig.tight_layout()
    out = FIG / "fig1_contagion_caar.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"✅ Figure 1 저장: {out.name}")


# ════════════════════════════════════════════════════════════════
# Figure 2: Regime-conditional Contagion (연도별)
# ════════════════════════════════════════════════════════════════
def fig2_regime():
    YEARS = list(range(2019, 2026))
    REGIMES = {
        "COVID\n충격": (2020, 2021, "#FEF3C7"),
        "IRA/中규제": (2022, 2023, "#DBEAFE"),
        "Trump\nTariff": (2024, 2025, "#FCE7F3"),
    }
    RELS = [("MIXED_2_2", "Mixed hop2"), ("UPSTREAM_2_2", "Upstream hop2")]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    for ax, (rel, label) in zip(axes, RELS):
        sub = car[(car["relationship"] == rel) &
                  (car["window_label"] == "0_21") &
                  (car["quality"] == "full")].copy()
        sub["year"] = sub["shock_month"].str[:4].astype(int)

        caars, cis, ps = [], [], []
        for yr in YEARS:
            g = sub[sub["year"] == yr]["CAR"]
            if len(g) < 5:
                caars.append(np.nan); cis.append(np.nan); ps.append(1.0)
                continue
            t, p = ttest_1samp(g, 0)
            caars.append(g.mean())
            cis.append(1.96 * g.std(ddof=1) / np.sqrt(len(g)))
            ps.append(p)

        xs   = np.arange(len(YEARS))
        cols = [BLUE if c and c>0 else RED if c and c<0 else GRAY
                for c,p in zip(caars, ps)]
        alphas = [1.0 if p<0.05 else 0.45 for p in ps]

        for i, (x, c, ci_, p_, col, alp) in enumerate(
                zip(xs, caars, cis, ps, cols, alphas)):
            if pd.isna(c): continue
            ax.bar(x, c, color=col, alpha=alp, width=0.6, zorder=3)
            ax.errorbar(x, c, yerr=ci_, fmt="none",
                        color="black", capsize=3, linewidth=1, zorder=4)
            sig = "***" if p_<0.01 else "**" if p_<0.05 else "*" if p_<0.10 else ""
            if sig:
                off = 0.004 if c >= 0 else -0.009
                ax.text(x, c + (ci_ if c>=0 else -ci_) + off,
                        sig, ha="center", fontsize=11, fontweight="bold")

        # regime 배경
        ax.axhline(0, color="black", linewidth=0.8)
        for name, (y1, y2, color) in REGIMES.items():
            x1 = YEARS.index(y1) - 0.4
            x2 = YEARS.index(y2) + 0.4
            ax.axvspan(x1, x2, alpha=0.15, color=color, zorder=1)
            ax.text((x1+x2)/2, ax.get_ylim()[0] if ax.get_ylim()[0] < -0.03
                    else -0.04,
                    name, ha="center", fontsize=8, color="#374151")

        ax.set_xticks(xs)
        ax.set_xticklabels(YEARS)
        ax.yaxis.set_major_formatter(pct)
        ax.set_title(f"{label}", fontweight="bold")
        ax.set_xlabel("Year")
        if ax == axes[0]:
            ax.set_ylabel("CAAR [0, +21]")

    fig.suptitle("Figure 2. Regime-conditional Supply Chain Contagion (T0 = M)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = FIG / "fig2_regime_contagion.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"✅ Figure 2 저장: {out.name}")


# ════════════════════════════════════════════════════════════════
# Figure 3: GNN Monthly IC
# ════════════════════════════════════════════════════════════════
def fig3_ic():
    monthly_ic = (test_pred.groupby("year_month")
                  .apply(lambda g: spearmanr(g["prob"], g["actual_ar"])[0],
                         include_groups=False)
                  .reset_index(name="ic"))
    monthly_ic = monthly_ic.sort_values("year_month").reset_index(drop=True)
    mean_ic    = monthly_ic["ic"].mean()

    fig, ax = plt.subplots(figsize=(11, 4.5))
    xs = np.arange(len(monthly_ic))

    # 색상: IC>0 → blue, IC<0 → red
    bar_colors = [BLUE if v > 0 else RED for v in monthly_ic["ic"]]
    ax.bar(xs, monthly_ic["ic"], color=bar_colors, alpha=0.75, width=0.6, zorder=3)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(mean_ic, color=ORANGE, linewidth=1.8, linestyle="--",
               label=f"Mean IC = {mean_ic:+.3f}")

    # multi-seed CI band
    ax.axhspan(0.013, 0.092, alpha=0.10, color=ORANGE,
               label="Multi-seed 95% CI [+0.013, +0.092]")

    ax.set_xticks(xs)
    ax.set_xticklabels(monthly_ic["year_month"], rotation=45, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f"{x:+.2f}"))
    ax.set_ylabel("Spearman IC")
    ax.set_title("Figure 3. GNN Monthly IC — Test Period (2024-09 ~ 2025-11)\n"
                 f"IC>0: {(monthly_ic['ic']>0).sum()}/{len(monthly_ic)} months  |  "
                 f"Pooled IC={spearmanr(test_pred['prob'],test_pred['actual_ar'])[0]:+.4f}  |  "
                 f"p=0.004 (permutation, B=10,000)")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(-0.5, len(xs)-0.5)

    fig.tight_layout()
    out = FIG / "fig3_gnn_monthly_ic.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"✅ Figure 3 저장: {out.name}")


# ════════════════════════════════════════════════════════════════
# Figure 4: CAR 분포 — Direct NEG vs POS
# ════════════════════════════════════════════════════════════════
def fig4_car_dist():
    neg_car = direct[direct["shock_type"] == "NEG"]["CAR"]
    pos_car = direct[direct["shock_type"] == "POS"]["CAR"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (cars, label, color) in zip(
            axes,
            [(neg_car, "NEG Shock", RED), (pos_car, "POS Shock", BLUE)]):
        ax.hist(cars, bins=20, color=color, alpha=0.7, edgecolor="white")
        ax.axvline(cars.mean(), color="black", linewidth=1.8,
                   linestyle="--", label=f"Mean = {cars.mean():+.3f}")
        ax.axvline(cars.median(), color=GRAY, linewidth=1.2,
                   linestyle=":", label=f"Median = {cars.median():+.3f}")
        ax.xaxis.set_major_formatter(pct)
        ax.set_xlabel("CAR [0, +21]")
        ax.set_title(f"{label}  (N={len(cars)})", fontweight="bold")
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Frequency")
    fig.suptitle("Figure 4. CAR Distribution — Direct Effect (T0 = M)", fontsize=13)
    fig.tight_layout()
    out = FIG / "fig4_car_distribution.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"✅ Figure 4 저장: {out.name}")


# ════════════════════════════════════════════════════════════════
# Table 1: 데이터 기술통계
# ════════════════════════════════════════════════════════════════
def table1_data_stats():
    prices["date"] = pd.to_datetime(prices["date"])

    tone_start = tone["year_month"].min()
    tone_end   = tone["year_month"].max()
    price_start = prices["date"].min().strftime("%Y-%m")
    price_end   = prices["date"].max().strftime("%Y-%m")

    n_events = len(direct)
    n_neg    = len(direct[direct["shock_type"]=="NEG"])
    n_pos    = len(direct[direct["shock_type"]=="POS"])
    n_cos    = direct["shock_company"].nunique()

    tone_reliable = tone[tone["reliable_v3"]].groupby("company_id")["year_month"].count()
    n_shocks = (tone["neg_shock_v3"].astype(bool) | tone["pos_shock_v3"].astype(bool)).sum()

    rows = [
        ("Sample period (tone)", f"{tone_start} – {tone_end}"),
        ("Sample period (prices)", f"{price_start} – {price_end}"),
        ("Universe", f"{cos['company_id'].nunique()} companies"),
        ("Countries", f"{cos['country'].nunique()} (CN, KR, JP, DE, US, UK)"),
        ("Supply chain stages", "4 (OEM, Cell/Pack, Materials, Upstream/Refining)"),
        ("Supply chain edges", "140"),
        ("Total tone-months (reliable)", f"{tone_reliable.sum():,}"),
        ("Total tone shock events", f"{n_shocks:,}"),
        ("Phase 4 direct events", f"{n_events} (NEG={n_neg}, POS={n_pos})"),
        ("Phase 4 affected companies", f"{n_cos}"),
        ("Phase 4 total CAR observations", f"{len(car[car['quality']=='full']):,}"),
        ("Phase 5 test observations (valid)", f"{len(test_pred):,}"),
        ("Phase 5 test period", f"{test_pred['year_month'].min()} – {test_pred['year_month'].max()}"),
    ]

    df = pd.DataFrame(rows, columns=["Item", "Value"])
    out_csv = TABLE / "table1_data_stats.csv"
    out_md  = TABLE / "table1_data_stats.md"
    df.to_csv(out_csv, index=False)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Table 1. Data and Sample Statistics\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    print(f"✅ Table 1 저장: {out_csv.name}, {out_md.name}")
    return df


# ════════════════════════════════════════════════════════════════
# Table 2: 기업 유니버스 요약
# ════════════════════════════════════════════════════════════════
def table2_universe():
    tone_cov = tone.groupby("company_id").agg(
        tone_months=("year_month","count"),
        reliable_months=("reliable_v3","sum"),
        n_neg_shock=("neg_shock_v3","sum"),
        n_pos_shock=("pos_shock_v3","sum"),
    ).reset_index()
    tone_cov["reliable_pct"] = (tone_cov["reliable_months"]/tone_cov["tone_months"]*100).round(1)

    df = cos[["company_id","canonical_name","country","stage_bucket"]].merge(
        tone_cov, on="company_id", how="left"
    ).sort_values(["stage_bucket","country","canonical_name"])

    df = df.rename(columns={
        "company_id":"Ticker","canonical_name":"Company","country":"Country",
        "stage_bucket":"Stage","tone_months":"Tone Months",
        "reliable_months":"Reliable Mo.","reliable_pct":"Reliable %",
        "n_neg_shock":"NEG Shocks","n_pos_shock":"POS Shocks"
    })

    out_csv = TABLE / "table2_universe.csv"
    out_md  = TABLE / "table2_universe.md"
    df.to_csv(out_csv, index=False)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Table 2. Universe of 44 EV Battery Supply Chain Companies\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    print(f"✅ Table 2 저장: {out_csv.name}, {out_md.name}")
    # Stage별 요약
    summary = df.groupby("Stage").agg(
        N=("Ticker","count"),
        Avg_Reliable=("Reliable %","mean"),
        Total_NEG=("NEG Shocks","sum"),
    ).round(1)
    print("  Stage 요약:")
    print(summary.to_string())


# ════════════════════════════════════════════════════════════════
# Table 3: Phase 4 메인 결과 (논문 테이블)
# ════════════════════════════════════════════════════════════════
def table3_phase4():
    REL_MAP = {
        "DIRECT":        ("Direct",          "—"),
        "UPSTREAM_1":    ("Upstream hop1",   "1"),
        "UPSTREAM_2_2":  ("Upstream hop2",   "2"),
        "MIXED_2_2":     ("Mixed hop2",      "2"),
        "DOWNSTREAM_1":  ("Downstream hop1", "1"),
        "DOWNSTREAM_2_2":("Downstream hop2", "2"),
        "PEER_1":        ("Peer hop1",       "1"),
        "PEER_2_2":      ("Peer hop2",       "2"),
    }
    panel = car[(car["window_label"]=="0_21")&(car["quality"]=="full")]
    rows = []
    for rel, (label, hop) in REL_MAP.items():
        sub = panel[panel["relationship"]==rel]["CAR"]
        if len(sub) < 5: continue
        t, p = ttest_1samp(sub, 0)
        se   = sub.std(ddof=1)/np.sqrt(len(sub))
        sig  = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.10 else ""
        rows.append({
            "Relationship": label, "Hop": hop,
            "N": len(sub),
            "CAAR": f"{sub.mean()*100:+.2f}%",
            "t-stat": f"{t:+.2f}",
            "p-value": f"{p:.4f}",
            "Sig.": sig,
            "95% CI": f"[{(sub.mean()-1.96*se)*100:+.2f}%, {(sub.mean()+1.96*se)*100:+.2f}%]"
        })
    df = pd.DataFrame(rows)
    out_csv = TABLE / "table3_phase4_contagion.csv"
    out_md  = TABLE / "table3_phase4_contagion.md"
    df.to_csv(out_csv, index=False)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Table 3. Phase 4 — Supply Chain Contagion (Window [0, +21], T0 = M)\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n*Note: *** p<0.01, ** p<0.05, * p<0.10 (two-tailed cross-sectional t-test)*\n")
    print(f"✅ Table 3 저장: {out_csv.name}, {out_md.name}")


# ════════════════════════════════════════════════════════════════
# Table 4: Phase 5 모델 성능 비교
# ════════════════════════════════════════════════════════════════
def table4_phase5():
    rows = [
        ("Dynamic GAT (Real Graph)",  "0.5669", "54.6%", "+0.0931", "0.0042***",
         "+0.052 [+0.013, +0.092]"),
        ("Baseline GRU (No Graph)",   "0.4602", "45.5%", "-0.0646", "—",
         "-0.003 [-0.035, +0.028]"),
        ("Random Graph GAT",          "—",      "—",     "—",       "—",
         "+0.043 [-0.017, +0.103]"),
    ]
    cols = ["Model", "AUC", "Acc", "IC (seed=42)",
            "Permutation p", "Multi-seed IC (95% CI)"]
    df = pd.DataFrame(rows, columns=cols)
    out_csv = TABLE / "table4_phase5_performance.csv"
    out_md  = TABLE / "table4_phase5_performance.md"
    df.to_csv(out_csv, index=False)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Table 4. Phase 5 — GNN Model Performance (Test: 2024-09 ~ 2025-11)\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n*Note: Multi-seed results from 5 seeds [42,123,456,789,999] "
                "on tone_v4. Permutation test: B=10,000, year_month group shuffle.*\n")
    print(f"✅ Table 4 저장: {out_csv.name}, {out_md.name}")


# ════════════════════════════════════════════════════════════════
# 실행
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=== Figure & Table 생성 시작 ===\n")
    fig1_contagion()
    fig2_regime()
    fig3_ic()
    fig4_car_dist()
    print()
    table1_data_stats()
    table2_universe()
    table3_phase4()
    table4_phase5()
    print("\n=== 완료 ===")
    print(f"Figure: {FIG}")
    print(f"Table:  {TABLE}")
