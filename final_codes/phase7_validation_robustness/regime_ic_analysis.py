"""
regime_ic_analysis.py
트럼프 관세 레짐이 공급망 감성 신호의 예측력을 활성화한다 — 가설 검증

분석 1: 레짐별 IC 비교 (Permutation 기반)
분석 2: 레짐별 supply_z 신호 강도 비교
분석 3: 기업 레이어별 레짐 효과

출력:
  experiments/hypothesis/walk_forward_cv/reports/regime_ic_analysis.txt
  experiments/hypothesis/walk_forward_cv/reports/regime_ic_chart.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr, ttest_ind
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
ROOT = Path("C:/Users/john9/nabi_hyoghaw")
PRED_PATH  = ROOT / "experiments/universe_44/data/processed/phase5_predictions_44.parquet"
TONE_PATH  = ROOT / "experiments/universe_44/data/processed/tone_monthly_zscore_all44_v2.parquet"
COMP_PATH  = ROOT / "experiments/universe_44/data/universe/companies_44.csv"
REPORT_DIR = ROOT / "experiments/hypothesis/walk_forward_cv/reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_TXT = REPORT_DIR / "regime_ic_analysis.txt"
REPORT_PNG = REPORT_DIR / "regime_ic_chart.png"

SEED = 42
B    = 5000   # permutation 반복 횟수

# ── 데이터 로드 ────────────────────────────────────────────────────────────────
pred = pd.read_parquet(PRED_PATH)
tone = pd.read_parquet(TONE_PATH)
comp = pd.read_csv(COMP_PATH)

# test split, valid==True 필터
test = pred[(pred["split"] == "test") & (pred["valid"] == True)].copy()

# 레짐 구분 (2025-01 이상 = tariff)
test["regime"] = test["year_month"].apply(
    lambda m: "tariff" if m >= "2025-01" else "pre_tariff"
)

print(f"Test set shape: {test.shape}")
print(f"  pre_tariff N={len(test[test['regime']=='pre_tariff'])}, "
      f"tariff N={len(test[test['regime']=='tariff'])}")

# ── 함수 ──────────────────────────────────────────────────────────────────────

def spearman_ic(df: pd.DataFrame) -> tuple[float, float, int]:
    """(IC, p_value, N) — prob vs actual_ar Spearman rho"""
    if len(df) < 4:
        return np.nan, np.nan, len(df)
    ic, p = spearmanr(df["prob"], df["actual_ar"])
    return ic, p, len(df)


def permutation_pvalue(obs_ic: float, x: np.ndarray, y: np.ndarray,
                       B: int = 5000, seed: int = 42) -> float:
    """양측 permutation p-value"""
    rng = np.random.default_rng(seed)
    null = np.array([
        spearmanr(rng.permutation(x), y)[0] for _ in range(B)
    ])
    return float(np.mean(np.abs(null) >= abs(obs_ic)))


def monthly_ic(df: pd.DataFrame) -> pd.DataFrame:
    """월별 IC 계산"""
    rows = []
    for ym, grp in df.groupby("year_month"):
        ic, p, n = spearman_ic(grp)
        regime = "tariff" if ym >= "2025-01" else "pre_tariff"
        rows.append({"year_month": ym, "IC": ic, "p": p, "N": n, "regime": regime})
    return pd.DataFrame(rows).sort_values("year_month").reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 분석 1: 레짐별 IC 비교
# ═══════════════════════════════════════════════════════════════════════════════

lines = []
lines.append("=" * 60)
lines.append("  트럼프 관세 레짐 IC 심화 분석")
lines.append("=" * 60)

regime_results = {}
for regime in ["pre_tariff", "tariff"]:
    sub = test[test["regime"] == regime]
    ic, p_param, n = spearman_ic(sub)
    # permutation p-value
    p_perm = permutation_pvalue(
        ic,
        sub["prob"].values,
        sub["actual_ar"].values,
        B=B, seed=SEED
    )
    # 95% CI via bootstrap
    rng = np.random.default_rng(SEED)
    boot = []
    idx = np.arange(len(sub))
    for _ in range(B):
        s = rng.choice(idx, size=len(idx), replace=True)
        tmp = sub.iloc[s]
        if tmp["prob"].std() > 0 and tmp["actual_ar"].std() > 0:
            boot.append(spearmanr(tmp["prob"], tmp["actual_ar"])[0])
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan)
    regime_results[regime] = {
        "N": n, "IC": ic, "p_param": p_param, "p_perm": p_perm,
        "ci_lo": ci_lo, "ci_hi": ci_hi
    }

pre = regime_results["pre_tariff"]
tar = regime_results["tariff"]
ic_diff = tar["IC"] - pre["IC"]

lines.append("")
lines.append("[구간별 IC]")
lines.append(f"  Pre-tariff (2024-09~2024-12):  N={pre['N']:>4d},  IC={pre['IC']:+.4f},  "
             f"p_param={pre['p_param']:.4f},  p_perm={pre['p_perm']:.4f},  "
             f"95%CI=[{pre['ci_lo']:+.4f}, {pre['ci_hi']:+.4f}]")
lines.append(f"  Tariff     (2025-01~2025-11):  N={tar['N']:>4d},  IC={tar['IC']:+.4f},  "
             f"p_param={tar['p_param']:.4f},  p_perm={tar['p_perm']:.4f},  "
             f"95%CI=[{tar['ci_lo']:+.4f}, {tar['ci_hi']:+.4f}]")
lines.append(f"  IC 차이: {ic_diff:+.4f}  (tariff - pre_tariff)")

# 월별 IC
monthly_df = monthly_ic(test)

lines.append("")
lines.append("[월별 IC]")
lines.append(f"  {'year_month':<12} {'IC':>8} {'N':>5} {'regime':<12}")
lines.append("  " + "-" * 42)
for _, row in monthly_df.iterrows():
    ic_str = f"{row['IC']:+.4f}" if not np.isnan(row["IC"]) else "   nan"
    lines.append(f"  {row['year_month']:<12} {ic_str:>8} {int(row['N']):>5}  {row['regime']:<12}")


# ═══════════════════════════════════════════════════════════════════════════════
# 분석 2: 레짐별 supply_z 신호 강도 비교
# ═══════════════════════════════════════════════════════════════════════════════

# pre-tariff: 2024-07~2024-12 / tariff: 2025-01~2025-11
tone_pre = tone[
    (tone["year_month"] >= "2024-07") & (tone["year_month"] <= "2024-12") &
    tone["supply_z_v3"].notna()
]["supply_z_v3"]

tone_tar = tone[
    (tone["year_month"] >= "2025-01") & (tone["year_month"] <= "2025-11") &
    tone["supply_z_v3"].notna()
]["supply_z_v3"]

t_stat, t_p = ttest_ind(tone_pre, tone_tar, equal_var=False)  # Welch's t-test

lines.append("")
lines.append("[Supply_z 신호 강도]")
lines.append(f"  Pre-tariff avg supply_z (2024-07~12): {tone_pre.mean():+.4f}  "
             f"(std={tone_pre.std():.4f},  N={len(tone_pre)})")
lines.append(f"  Tariff     avg supply_z (2025-01~11): {tone_tar.mean():+.4f}  "
             f"(std={tone_tar.std():.4f},  N={len(tone_tar)})")
lines.append(f"  Welch t-statistic: {t_stat:+.4f},  p={t_p:.4f}")
lines.append(f"  해석: {'tariff 기간 supply_z 유의하게 다름' if t_p < 0.05 else 'tariff 기간 supply_z 유의한 차이 없음'} (α=0.05)")


# ═══════════════════════════════════════════════════════════════════════════════
# 분석 3: 기업 레이어별 레짐 효과
# ═══════════════════════════════════════════════════════════════════════════════

test_comp = test.merge(comp[["company_id", "stage_bucket"]], on="company_id", how="left")

layer_rows = []
for layer in ["OEM", "Cell/Pack", "Materials", "Upstream/Refining"]:
    sub_layer = test_comp[test_comp["stage_bucket"] == layer]
    for regime in ["pre_tariff", "tariff"]:
        sub = sub_layer[sub_layer["regime"] == regime]
        ic, p, n = spearman_ic(sub)
        layer_rows.append({"layer": layer, "regime": regime, "IC": ic, "p": p, "N": n})

layer_df = pd.DataFrame(layer_rows)

lines.append("")
lines.append("[레이어별 레짐 효과]")
header = f"  {'레이어':<20} {'pre_IC':>8} {'pre_N':>6} {'tariff_IC':>10} {'tariff_N':>8} {'개선폭':>8} {'유의(tariff)':>14}"
lines.append(header)
lines.append("  " + "-" * 78)

for layer in ["OEM", "Cell/Pack", "Materials", "Upstream/Refining"]:
    pre_row = layer_df[(layer_df["layer"] == layer) & (layer_df["regime"] == "pre_tariff")]
    tar_row = layer_df[(layer_df["layer"] == layer) & (layer_df["regime"] == "tariff")]
    if pre_row.empty or tar_row.empty:
        continue
    pre_ic = pre_row.iloc[0]["IC"]
    tar_ic = tar_row.iloc[0]["IC"]
    tar_p  = tar_row.iloc[0]["p"]
    pre_n  = int(pre_row.iloc[0]["N"])
    tar_n  = int(tar_row.iloc[0]["N"])
    diff   = tar_ic - pre_ic if not (np.isnan(tar_ic) or np.isnan(pre_ic)) else np.nan
    sig    = "*(p<0.05)" if (not np.isnan(tar_p) and tar_p < 0.05) else ""
    pre_s  = f"{pre_ic:+.4f}" if not np.isnan(pre_ic) else "   nan"
    tar_s  = f"{tar_ic:+.4f}" if not np.isnan(tar_ic) else "   nan"
    diff_s = f"{diff:+.4f}" if not np.isnan(diff) else "   nan"
    lines.append(f"  {layer:<20} {pre_s:>8} {pre_n:>6} {tar_s:>10} {tar_n:>8} {diff_s:>8}  {sig:<14}")


# ═══════════════════════════════════════════════════════════════════════════════
# 가설 판정 요약
# ═══════════════════════════════════════════════════════════════════════════════

lines.append("")
lines.append("[가설 판정]")
lines.append(f"  H0: 레짐 전후 IC 차이 없음")
lines.append(f"  IC 차이 = {ic_diff:+.4f}")

# 차이에 대한 bootstrap 검정 (pre-tariff vs tariff 월별 IC 분포 비교)
pre_monthly = monthly_df[monthly_df["regime"] == "pre_tariff"]["IC"].dropna()
tar_monthly = monthly_df[monthly_df["regime"] == "tariff"]["IC"].dropna()

if len(pre_monthly) > 1 and len(tar_monthly) > 1:
    t_ic, p_ic = ttest_ind(pre_monthly, tar_monthly, equal_var=False)
    lines.append(f"  월별 IC 분포 Welch t-test: t={t_ic:+.4f}, p={p_ic:.4f}")
    verdict = "채택 (IC 개선 유의)" if (ic_diff > 0 and p_ic < 0.1) else "기각 불충분"
    lines.append(f"  판정: {verdict}")
else:
    lines.append("  월별 IC 샘플 부족으로 검정 불가")

lines.append("")
lines.append("=" * 60)

report_text = "\n".join(lines)
print(report_text)

# 텍스트 보고서 저장
with open(REPORT_TXT, "w", encoding="utf-8") as f:
    f.write(report_text)
print(f"\n텍스트 보고서 저장: {REPORT_TXT}")


# ═══════════════════════════════════════════════════════════════════════════════
# 시각화
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#0f1117")

# ── 차트 1: 월별 IC 추이 + 레짐 배경 ──────────────────────────────────────────
ax1 = axes[0]
ax1.set_facecolor("#1a1d27")

x_labels = monthly_df["year_month"].tolist()
x_pos = range(len(x_labels))
ics = monthly_df["IC"].tolist()
regimes = monthly_df["regime"].tolist()

# 레짐 배경 (tariff 구간 빨간색)
tariff_start = None
for i, r in enumerate(regimes):
    if r == "tariff" and tariff_start is None:
        tariff_start = i - 0.5
if tariff_start is not None:
    ax1.axvspan(tariff_start, len(x_labels) - 0.5,
                color="#ff4444", alpha=0.15, label="Trump Tariff Regime")

# IC=0 기준선
ax1.axhline(0, color="#888888", linewidth=0.8, linestyle="--", alpha=0.7)

# 월별 IC 라인
colors = ["#ff6666" if r == "tariff" else "#6699ff" for r in regimes]
ax1.plot(list(x_pos), ics, color="#dddddd", linewidth=1.2, alpha=0.6, zorder=2)
for i, (ic_val, c) in enumerate(zip(ics, colors)):
    if not np.isnan(ic_val):
        ax1.scatter(i, ic_val, color=c, s=55, zorder=3, edgecolors="white", linewidths=0.5)

ax1.set_xticks(list(x_pos))
ax1.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=7.5, color="#cccccc")
ax1.set_ylabel("Monthly IC (Spearman rho)", color="#cccccc", fontsize=9)
ax1.set_title("월별 IC 추이 (레짐 구분)", color="white", fontsize=11, pad=10)
ax1.tick_params(colors="#aaaaaa")
for spine in ax1.spines.values():
    spine.set_edgecolor("#444444")

patch_pre = mpatches.Patch(color="#6699ff", label="Pre-tariff")
patch_tar = mpatches.Patch(color="#ff6666", label="Tariff (2025-01~)")
ax1.legend(handles=[patch_pre, patch_tar], fontsize=8,
           facecolor="#1a1d27", edgecolor="#555555", labelcolor="white")

# ── 차트 2: 레짐별 IC 분포 boxplot ────────────────────────────────────────────
ax2 = axes[1]
ax2.set_facecolor("#1a1d27")

pre_ics = monthly_df[monthly_df["regime"] == "pre_tariff"]["IC"].dropna().tolist()
tar_ics = monthly_df[monthly_df["regime"] == "tariff"]["IC"].dropna().tolist()

data_groups = [pre_ics, tar_ics]
labels_box  = ["Pre-tariff\n(2024-09~12)", "Tariff\n(2025-01~11)"]
box_colors  = ["#6699ff", "#ff6666"]

bp = ax2.boxplot(data_groups, patch_artist=True, widths=0.4,
                 medianprops=dict(color="white", linewidth=2))

for patch, color in zip(bp["boxes"], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
for whisker in bp["whiskers"]:
    whisker.set_color("#aaaaaa")
for cap in bp["caps"]:
    cap.set_color("#aaaaaa")
for flier in bp["fliers"]:
    flier.set(marker="o", color="#aaaaaa", alpha=0.5)

# 개별 점 오버레이
rng_jitter = np.random.default_rng(SEED)
for idx_g, (grp, color) in enumerate(zip(data_groups, box_colors), start=1):
    jitter = rng_jitter.uniform(-0.08, 0.08, size=len(grp))
    ax2.scatter([idx_g + j for j in jitter], grp,
                color=color, s=45, alpha=0.9, zorder=5,
                edgecolors="white", linewidths=0.5)

ax2.axhline(0, color="#888888", linewidth=0.8, linestyle="--", alpha=0.7)

# p-value 주석
if len(pre_ics) > 1 and len(tar_ics) > 1:
    _, p_box = ttest_ind(pre_ics, tar_ics, equal_var=False)
    ax2.text(1.5, max(max(pre_ics, default=0), max(tar_ics, default=0)) * 1.05,
             f"p={p_box:.3f}", ha="center", fontsize=9, color="#ffcc44")

ax2.set_xticks([1, 2])
ax2.set_xticklabels(labels_box, color="#cccccc", fontsize=9)
ax2.set_ylabel("Monthly IC", color="#cccccc", fontsize=9)
ax2.set_title("레짐별 IC 분포", color="white", fontsize=11, pad=10)
ax2.tick_params(colors="#aaaaaa")
for spine in ax2.spines.values():
    spine.set_edgecolor("#444444")

plt.tight_layout(pad=2.0)
plt.savefig(REPORT_PNG, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"차트 저장: {REPORT_PNG}")
