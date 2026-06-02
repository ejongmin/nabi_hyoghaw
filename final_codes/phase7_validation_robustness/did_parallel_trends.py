"""
did_parallel_trends.py
======================
DID(Difference-in-Differences)의 핵심 가정인 "처치 전 평행 추세(parallel trends)"를 검정.

데이터: data/processed/car_panel_phase4.parquet
  - Treatment group: DIRECT가 아닌 기업 (DOWNSTREAM/PEER/UPSTREAM 등 — 간접 노출)
  - Control group: DIRECT 기업 (충격 기업 자신 — 연결되지 않은 나머지로 해석)

  ※ 설계 근거:
    car_panel_phase4에서 relationship='DIRECT'는 충격 기업 자신(self)을 가리킴.
    DID 맥락에서 처치(treatment) = 충격을 외부에서 받는 간접 연결 기업,
    통제(control) = 충격 기업 본인(직접 노출). 사전 기간에서 두 그룹의 CAR 추이가
    평행한지 검정.

  사전 기간: win_start < 0 인 window (m1_1, m5_5)
  처치 후 기간: win_start >= 0 인 window (0_1, 0_5, 0_10, 0_21)

검정:
  - Wilcoxon rank-sum test (Mann–Whitney U) — 비모수
  - 귀무가설: 사전 기간 treatment CAR 분포 = control CAR 분포
  - p > 0.05 → 평행 추세 가정 지지

출력:
  - experiments/hypothesis/walk_forward_cv/reports/did_parallel_trends.txt
  - experiments/hypothesis/walk_forward_cv/reports/did_parallel_trends.png
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ranksums

warnings.filterwarnings("ignore")

ROOT    = Path("C:/Users/john9/nabi_hyoghaw")
HYP     = ROOT / "experiments/hypothesis/walk_forward_cv"
OUT_RPT = HYP / "reports/did_parallel_trends.txt"
OUT_PNG = HYP / "reports/did_parallel_trends.png"
PATH_CAR = ROOT / "data/processed/car_panel_phase4.parquet"


# ─────────────────────────────────────────────────────────────
# 1. 데이터 로딩 및 그룹 분류
# ─────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    df = pd.read_parquet(PATH_CAR)
    # quality 필터 — marginal 제외
    df = df[df["quality"] == "full"].copy()
    # confounding 제외 (자기 혼재)
    df = df[df["confounding_self"] == False].copy()
    return df


def assign_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Treatment: DIRECT가 아닌 관계 (간접 노출 기업)
    Control:   DIRECT (충격 기업 자신)
    """
    df = df.copy()
    df["group"] = np.where(df["relationship"] == "DIRECT", "Control", "Treatment")
    return df


# ─────────────────────────────────────────────────────────────
# 2. 평행 추세 검정
# ─────────────────────────────────────────────────────────────

def parallel_trends_test(df: pd.DataFrame) -> dict:
    """
    사전 기간(win_start < 0) window별 treatment vs control 비교.
    Wilcoxon rank-sum (Mann–Whitney U) 사용.
    """
    pre  = df[df["win_start"] < 0].copy()
    post = df[df["win_start"] >= 0].copy()

    treat_pre  = pre[pre["group"] == "Treatment"]["CAR"]
    ctrl_pre   = pre[pre["group"] == "Control"]["CAR"]
    treat_post = post[post["group"] == "Treatment"]["CAR"]
    ctrl_post  = post[post["group"] == "Control"]["CAR"]

    # Wilcoxon rank-sum (= Mann–Whitney U, two-sided)
    if len(treat_pre) >= 1 and len(ctrl_pre) >= 1:
        stat, p_pre = ranksums(treat_pre, ctrl_pre)
    else:
        stat, p_pre = np.nan, np.nan

    # 사전 기간 평균 차이
    diff_pre = treat_pre.mean() - ctrl_pre.mean()

    # 사후 기간 (참고용)
    if len(treat_post) >= 1 and len(ctrl_post) >= 1:
        _, p_post = ranksums(treat_post, ctrl_post)
    else:
        p_post = np.nan
    diff_post = treat_post.mean() - ctrl_post.mean()

    # window별 세부 비교 (사전 기간)
    window_results = []
    for wlabel in sorted(pre["window_label"].unique()):
        sub = pre[pre["window_label"] == wlabel]
        t_sub = sub[sub["group"] == "Treatment"]["CAR"]
        c_sub = sub[sub["group"] == "Control"]["CAR"]
        if len(t_sub) >= 1 and len(c_sub) >= 1:
            _, wp = ranksums(t_sub, c_sub)
        else:
            wp = np.nan
        window_results.append({
            "window": wlabel,
            "N_treat": len(t_sub),
            "N_ctrl": len(c_sub),
            "mean_treat": t_sub.mean(),
            "mean_ctrl": c_sub.mean(),
            "diff": t_sub.mean() - c_sub.mean(),
            "p": wp,
        })

    return {
        "treat_pre_mean": treat_pre.mean(),
        "ctrl_pre_mean": ctrl_pre.mean(),
        "treat_pre_n": len(treat_pre),
        "ctrl_pre_n": len(ctrl_pre),
        "diff_pre": diff_pre,
        "stat_pre": stat,
        "p_pre": p_pre,
        "treat_post_mean": treat_post.mean(),
        "ctrl_post_mean": ctrl_post.mean(),
        "treat_post_n": len(treat_post),
        "ctrl_post_n": len(ctrl_post),
        "diff_post": diff_post,
        "p_post": p_post,
        "window_results": window_results,
    }


# ─────────────────────────────────────────────────────────────
# 3. 누적 CAR 추이 시각화
# ─────────────────────────────────────────────────────────────

def plot_car_trend(df: pd.DataFrame, out_path: Path) -> None:
    """
    각 window별 treatment/control 평균 CAR을 꺾은선으로 시각화.
    x축: window_label (win_start 기준 정렬)
    """
    # window_label → win_start 매핑 (정렬용)
    wmap = df.groupby("window_label")["win_start"].first().to_dict()
    win_order = sorted(wmap.keys(), key=lambda w: (wmap[w], w))

    grp_mean = (
        df.groupby(["window_label", "group"])["CAR"]
        .agg(["mean", "sem", "count"])
        .reset_index()
    )
    grp_mean.columns = ["window_label", "group", "mean", "sem", "n"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"Treatment": "#e74c3c", "Control": "#2980b9"}
    markers = {"Treatment": "o", "Control": "s"}

    # ── 왼쪽: 평균 CAR per window ──────────────────────────────
    ax = axes[0]
    for grp in ["Treatment", "Control"]:
        sub = grp_mean[grp_mean["group"] == grp].copy()
        sub["x_pos"] = [win_order.index(w) for w in sub["window_label"]]
        sub = sub.sort_values("x_pos")
        ax.errorbar(
            sub["x_pos"], sub["mean"] * 100,
            yerr=sub["sem"] * 100 * 1.96,
            label=grp, color=colors[grp],
            marker=markers[grp], linewidth=2, markersize=7,
            capsize=4, capthick=1.5,
        )

    # 사전/사후 기간 구분선 (win_start<0 마지막 x_pos와 0 첫 x_pos 사이)
    pre_indices  = [i for i, w in enumerate(win_order) if wmap[w] < 0]
    post_indices = [i for i, w in enumerate(win_order) if wmap[w] >= 0]
    if pre_indices and post_indices:
        vline_x = (pre_indices[-1] + post_indices[0]) / 2
        ax.axvline(vline_x, color="gray", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.text(vline_x + 0.05, ax.get_ylim()[0] if ax.get_ylim()[0] != ax.get_ylim()[1] else 0,
                "처치시점 →", fontsize=8, color="gray", va="bottom")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xticks(range(len(win_order)))
    ax.set_xticklabels(win_order, rotation=30, ha="right", fontsize=9)
    ax.set_xlabel("Event Window", fontsize=10)
    ax.set_ylabel("평균 CAR (%)", fontsize=10)
    ax.set_title("Treatment vs Control: 평균 CAR (95% CI)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # 사전/사후 배경 색상
    if pre_indices:
        ax.axvspan(-0.5, vline_x, alpha=0.06, color="blue")
    if post_indices:
        ax.axvspan(vline_x, len(win_order) - 0.5, alpha=0.06, color="red")

    # ── 오른쪽: 사전 기간 CAR 분포 (박스플롯) ─────────────────
    ax2 = axes[1]
    pre_df = df[df["win_start"] < 0]
    treat_cars = pre_df[pre_df["group"] == "Treatment"]["CAR"].values * 100
    ctrl_cars  = pre_df[pre_df["group"] == "Control"]["CAR"].values * 100

    bp = ax2.boxplot(
        [treat_cars, ctrl_cars],
        labels=["Treatment\n(간접 노출)", "Control\n(충격 기업 본인)"],
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        boxprops=dict(linewidth=1.2),
        flierprops=dict(marker=".", markersize=3, alpha=0.4),
        notch=False,
    )
    bp["boxes"][0].set_facecolor("#e74c3c"); bp["boxes"][0].set_alpha(0.4)
    bp["boxes"][1].set_facecolor("#2980b9"); bp["boxes"][1].set_alpha(0.4)

    ax2.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax2.set_ylabel("CAR (%)", fontsize=10)
    ax2.set_title("사전 기간 CAR 분포\n(Parallel Trends 검정 대상)", fontsize=11)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    # 평균 표시
    for i, (vals, col) in enumerate([(treat_cars, "#e74c3c"), (ctrl_cars, "#2980b9")], 1):
        ax2.scatter([i], [np.mean(vals)], color=col, zorder=5, s=50, marker="D",
                    label=f"mean={np.mean(vals):.2f}%")
    ax2.legend(fontsize=8)

    plt.suptitle("DID Parallel Trends 검정 — car_panel_phase4", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[차트 저장] {out_path}")


# ─────────────────────────────────────────────────────────────
# 4. 보고서 작성
# ─────────────────────────────────────────────────────────────

def write_report(res: dict, out_path: Path) -> str:
    r = res
    p_pre = r["p_pre"]

    if np.isnan(p_pre):
        judgment = "판정 불가 (데이터 부족)"
    elif p_pre > 0.05:
        judgment = "평행 추세 가정 충족 (p > 0.05)"
    elif p_pre > 0.10:
        judgment = "평행 추세 가정 충족 (p > 0.10)"
    else:
        judgment = "평행 추세 위반 가능성 (p ≤ 0.05)"

    SEP = "=" * 64
    lines = [
        SEP,
        "=== DID Parallel Trends 검정 ===",
        SEP,
        "",
        "그룹 정의:",
        "  Treatment: relationship != 'DIRECT' (간접 노출 기업 — DOWNSTREAM/PEER/UPSTREAM/MIXED)",
        "  Control:   relationship == 'DIRECT'  (충격 기업 자신)",
        "  필터: quality='full', confounding_self=False",
        "",
        "사전 기간: win_start < 0 (window: m1_1, m5_5)",
        "처치 후 기간: win_start >= 0 (window: 0_1, 0_5, 0_10, 0_21)",
        "",
        "─" * 64,
        "■ 사전 기간 (Pre-treatment) 비교",
        "─" * 64,
        f"  사전 기간 Treatment CAR 평균: {r['treat_pre_mean']*100:+.4f}%  (N={r['treat_pre_n']})",
        f"  사전 기간 Control   CAR 평균: {r['ctrl_pre_mean']*100:+.4f}%  (N={r['ctrl_pre_n']})",
        f"  사전 차이 (Treat - Ctrl):     {r['diff_pre']*100:+.4f}%",
        f"  Wilcoxon rank-sum stat:       {r['stat_pre']:+.4f}" if not np.isnan(r['stat_pre']) else "  Wilcoxon rank-sum stat: N/A",
        f"  Wilcoxon p-value:             {p_pre:.4f}" if not np.isnan(p_pre) else "  Wilcoxon p-value: N/A",
        f"  판정: {judgment}",
        "",
        "─" * 64,
        "■ Window별 세부 비교 (사전 기간)",
        "─" * 64,
        f"  {'Window':<10} {'N_treat':>8} {'N_ctrl':>8} {'Treat_mean':>12} {'Ctrl_mean':>12} {'Diff':>10} {'p':>8}",
        "  " + "-" * 62,
    ]

    for wr in r["window_results"]:
        p_str = f"{wr['p']:.4f}" if not np.isnan(wr["p"]) else "  N/A"
        sig = ""
        if not np.isnan(wr["p"]):
            sig = "***" if wr["p"] < 0.01 else ("**" if wr["p"] < 0.05 else ("*" if wr["p"] < 0.10 else ""))
        lines.append(
            f"  {wr['window']:<10} {wr['N_treat']:>8} {wr['N_ctrl']:>8} "
            f"{wr['mean_treat']*100:>+11.4f}% {wr['mean_ctrl']*100:>+11.4f}% "
            f"{wr['diff']*100:>+9.4f}% {p_str:>8} {sig}"
        )

    lines += [
        "  유의성: *** p<0.01, ** p<0.05, * p<0.10",
        "",
        "─" * 64,
        "■ 처치 후 기간 (Post-treatment) 참고",
        "─" * 64,
        f"  사후 기간 Treatment CAR 평균: {r['treat_post_mean']*100:+.4f}%  (N={r['treat_post_n']})",
        f"  사후 기간 Control   CAR 평균: {r['ctrl_post_mean']*100:+.4f}%  (N={r['ctrl_post_n']})",
        f"  사후 차이 (Treat - Ctrl):     {r['diff_post']*100:+.4f}%",
        f"  Wilcoxon p-value (사후):      {r['p_post']:.4f}" if not np.isnan(r["p_post"]) else "  Wilcoxon p-value (사후): N/A",
        "",
        "─" * 64,
        "■ 해석",
        "─" * 64,
        "  DID 유효성 조건:",
        "    1) 사전 기간 p > 0.05 → 두 그룹 사전 추이 통계적으로 동일",
        "    2) 사후 기간 p < 0.05 → 처치 효과 존재",
        "",
        "  [참고] 본 데이터에서 'Control' = 충격 기업 자신(DIRECT)이므로",
        "  전통적 DID 설계와 다를 수 있음. 평행 추세 충족 여부는",
        "  사전 기간 두 그룹의 CAR 분포 동질성으로 해석.",
        "",
        "  차트: did_parallel_trends.png 참조",
        SEP,
    ]

    rpt = "\n".join(lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rpt, encoding="utf-8")
    print("\n" + rpt)
    return rpt


# ─────────────────────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────────────────────

def main():
    print("[1] 데이터 로딩...")
    df = load_data()
    df = assign_groups(df)

    print(f"  전체 (quality=full, no_confound): {len(df)}행")
    print(f"  Treatment (non-DIRECT): {(df['group']=='Treatment').sum()}")
    print(f"  Control   (DIRECT):     {(df['group']=='Control').sum()}")
    print(f"  window_label 종류: {sorted(df['window_label'].unique())}")

    print("\n[2] 평행 추세 검정...")
    res = parallel_trends_test(df)

    print("\n[3] 시각화...")
    plot_car_trend(df, OUT_PNG)

    print("\n[4] 보고서 작성...")
    write_report(res, OUT_RPT)
    print(f"\n[완료] 보고서: {OUT_RPT}")
    print(f"[완료] 차트:   {OUT_PNG}")


if __name__ == "__main__":
    main()
