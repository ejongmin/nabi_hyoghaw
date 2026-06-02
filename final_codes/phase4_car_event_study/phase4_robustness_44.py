"""
Phase 4 Robustness Checks — 44개 기업 확장판
==============================================
입력: data/processed/car_panel_phase4_44.parquet  (CAR 패널, 이미 계산됨)
출력: reports/phase4_robustness_44.txt

분석 내용:
  [1] 충격 임계값 민감도 (z_score 기준, NEG)
  [2] 충격 임계값 민감도 (z_score 기준, POS)
  [3] Cluster SE (이벤트-월 클러스터 부트스트랩, B=500)
  [4] 레이어별 임계값 분석 (stage_bucket 기준)

필터 조건 (사용자 명세):
  - quality == 'full'
  - confounding_self == False
  - relationship == 'DIRECT'
  - window_label == '0_21'
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

ROOT = Path("C:/Users/john9/nabi_hyoghaw")
PATH_PANEL = ROOT / "data/processed/car_panel_phase4_44.parquet"
PATH_UNIV  = ROOT / "data/universe/univers_final.csv"
OUT_RPT    = ROOT / "reports/phase4_robustness_44.txt"

# 분석 임계값 목록
NEG_THRESHOLDS = [-1.0, -1.5, -2.0, -2.5]
POS_THRESHOLDS = [ 1.0,  1.5,  2.0,  2.5]
BOOTSTRAP_B    = 500
BOOTSTRAP_SEED = 42

# 레이어별 분석에 쓸 임계값
LAYER_THR_NEG = -1.5
LAYER_THR_POS =  1.5


# ── 공통 유틸 ───────────────────────────────────────────────────

def sig_stars(p: float) -> str:
    """유의성 별표"""
    if np.isnan(p):
        return "   "
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else "   "


def caar_stats(cars: np.ndarray) -> dict:
    """CAAR, t-stat, p-value, 95% CI 계산"""
    n = len(cars)
    if n == 0:
        return dict(N=0, CAAR=np.nan, t=np.nan, p=np.nan, ci_lo=np.nan, ci_hi=np.nan)
    mu = float(cars.mean())
    if n == 1:
        return dict(N=1, CAAR=mu, t=np.nan, p=np.nan, ci_lo=np.nan, ci_hi=np.nan)
    sd = float(cars.std(ddof=1))
    se = sd / np.sqrt(n)
    if se < 1e-15:
        return dict(N=n, CAAR=mu, t=np.nan, p=np.nan, ci_lo=np.nan, ci_hi=np.nan)
    t = mu / se
    df = n - 1
    p = float(2 * stats.t.sf(abs(t), df=df))
    t_crit = float(stats.t.ppf(0.975, df=df))
    ci_lo = mu - t_crit * se
    ci_hi = mu + t_crit * se
    return dict(N=n, CAAR=mu, t=float(t), p=p, ci_lo=ci_lo, ci_hi=ci_hi)


def clustered_caar_bootstrap(
    df_sub: pd.DataFrame,
    B: int = 500,
    seed: int = 42,
) -> dict:
    """
    이벤트-월(shock_month) 클러스터 단위 비모수 부트스트랩 SE.

    df_sub 필요 컬럼: shock_month, CAR
    """
    rng = np.random.default_rng(seed)
    clusters = df_sub["shock_month"].unique()
    nc = len(clusters)

    caar_real = float(df_sub["CAR"].mean())

    if nc < 5:
        log.warning("클러스터 수 %d개 < 5, 부트스트랩 신뢰성 낮음", nc)

    # 클러스터 단위 리샘플링
    boot_caars = np.empty(B)
    for b in range(B):
        chosen = rng.choice(clusters, size=nc, replace=True)
        rows = pd.concat(
            [df_sub[df_sub["shock_month"] == c] for c in chosen],
            ignore_index=True,
        )
        boot_caars[b] = rows["CAR"].mean()

    se = float(np.std(boot_caars, ddof=1))
    ci_lo = float(np.percentile(boot_caars, 2.5))
    ci_hi = float(np.percentile(boot_caars, 97.5))
    # 양측 p: 귀무 CAAR=0 기준 부트스트랩 분포에서의 비율
    p = float(2 * min((boot_caars >= 0).mean(), (boot_caars <= 0).mean()))
    p = max(p, 1.0 / B)  # 0 방지

    return dict(
        N=len(df_sub),
        CAAR=caar_real,
        se_cluster=se,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        p_boot=p,
        n_cluster=nc,
    )


# ── 데이터 로드 및 필터 ──────────────────────────────────────────

def load_base(panel_path: Path, univ_path: Path):
    """기본 필터 적용 후 패널 반환"""
    log.info("패널 로드: %s", panel_path)
    df = pd.read_parquet(panel_path)
    log.info("  전체 건수: %d", len(df))

    # 기본 필터
    mask = (
        (df["quality"] == "full")
        & (df["confounding_self"] == False)
        & (df["relationship"] == "DIRECT")
        & (df["window_label"] == "0_21")
    )
    base = df[mask].copy()
    log.info("  필터 후 건수: %d (NEG=%d, POS=%d)",
             len(base),
             (base["shock_type"] == "NEG").sum(),
             (base["shock_type"] == "POS").sum())

    # 유니버스(stage_bucket) 조인
    univ = pd.read_csv(univ_path, usecols=["company_id", "stage_bucket"])
    base = base.merge(
        univ.rename(columns={"company_id": "shock_company", "stage_bucket": "layer"}),
        on="shock_company",
        how="left",
    )
    missing_layer = base["layer"].isna().sum()
    if missing_layer > 0:
        log.warning("  layer 매핑 실패: %d건 (NaN)", missing_layer)

    return base


# ── [1][2] 임계값 민감도 (NEG / POS) ────────────────────────────

def run_threshold_sensitivity(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    NEG: z_score < threshold (threshold < 0)
    POS: z_score > threshold (threshold > 0)
    """
    rows_neg, rows_pos = [], []

    for thr in NEG_THRESHOLDS:
        sub = base[base["z_score"] < thr]
        s = caar_stats(sub["CAR"].values)
        rows_neg.append(dict(threshold=thr, **s))
        log.info("  NEG z<%.1f  N=%d  CAAR=%+.4f  p=%s",
                 thr, s["N"],
                 s["CAAR"] if not np.isnan(s["CAAR"]) else 0,
                 f"{s['p']:.4f}" if not np.isnan(s["p"]) else "N/A")

    for thr in POS_THRESHOLDS:
        sub = base[base["z_score"] > thr]
        s = caar_stats(sub["CAR"].values)
        rows_pos.append(dict(threshold=thr, **s))
        log.info("  POS z>%.1f  N=%d  CAAR=%+.4f  p=%s",
                 thr, s["N"],
                 s["CAAR"] if not np.isnan(s["CAAR"]) else 0,
                 f"{s['p']:.4f}" if not np.isnan(s["p"]) else "N/A")

    return pd.DataFrame(rows_neg), pd.DataFrame(rows_pos)


# ── [3] Cluster SE ──────────────────────────────────────────────

def run_cluster_se(base: pd.DataFrame) -> dict:
    """NEG shock만 대상으로 Cluster SE 계산"""
    sub = base[base["shock_type"] == "NEG"][["shock_month", "CAR"]].dropna()
    log.info("  Cluster SE 대상: %d건, %d개 고유 이벤트-월",
             len(sub), sub["shock_month"].nunique())
    return clustered_caar_bootstrap(sub, B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED)


# ── [4] 레이어별 임계값 분석 ────────────────────────────────────

def run_layer_analysis(base: pd.DataFrame) -> pd.DataFrame:
    """
    각 stage_bucket에서 NEG z < LAYER_THR_NEG, POS z > LAYER_THR_POS 분석
    """
    # 알려진 레이어 순서
    known_layers = ["Upstream/Refining", "Materials", "Cell/Pack", "OEM"]
    layers = [l for l in known_layers if l in base["layer"].dropna().unique()]
    # 혹시 모르는 레이어 추가
    for l in base["layer"].dropna().unique():
        if l not in layers:
            layers.append(l)

    rows = []
    for layer in layers:
        layer_df = base[base["layer"] == layer]
        for shock_type, thr, direction in [
            ("NEG", LAYER_THR_NEG, "neg"),
            ("POS", LAYER_THR_POS, "pos"),
        ]:
            if direction == "neg":
                sub = layer_df[layer_df["z_score"] < thr]
            else:
                sub = layer_df[layer_df["z_score"] > thr]
            s = caar_stats(sub["CAR"].values)
            rows.append(dict(layer=layer, shock_type=shock_type,
                             threshold=thr, **s))
            log.info("  [레이어] %s / %s / z%s%.1f: N=%d  CAAR=%+.4f  p=%s",
                     layer, shock_type,
                     "<" if direction == "neg" else ">", abs(thr),
                     s["N"],
                     s["CAAR"] if not np.isnan(s["CAAR"]) else 0,
                     f"{s['p']:.4f}" if not np.isnan(s["p"]) else "N/A")

    return pd.DataFrame(rows)


# ── 보고서 출력 ──────────────────────────────────────────────────

def build_report(
    base: pd.DataFrame,
    neg_df: pd.DataFrame,
    pos_df: pd.DataFrame,
    clust: dict,
    layer_df: pd.DataFrame,
) -> str:
    SEP = "=" * 68
    lines = [
        SEP,
        "Phase 4 Robustness Checks 결과 (44개 기업)",
        SEP,
        "",
        f"  분석 기준: quality=full, confounding_self=False,",
        f"            relationship=DIRECT, window=0_21",
        f"  전체 이벤트: NEG {(base['shock_type']=='NEG').sum()}건  "
        f"POS {(base['shock_type']=='POS').sum()}건",
        "",
    ]

    # [1] NEG 임계값 민감도
    lines += [
        "■ [1] 충격 임계값 민감도 (직접 효과, [0,+21], NEG)",
        "",
        f"  {'임계값':10s}  {'N':>4}  {'CAAR':>9}  {'t':>7}  {'p':>8}  {'95% CI':>24}",
        "  " + "-" * 70,
    ]
    for _, r in neg_df.iterrows():
        p = r["p"]
        ci = (f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"
              if not np.isnan(r["ci_lo"]) else "N/A")
        pstr = f"{p:.4f}{sig_stars(p)}" if not np.isnan(p) else "  N/A   "
        lines.append(
            f"  z < {r['threshold']:4.1f}    "
            f"{int(r['N']):4d}  "
            f"{r['CAAR']:+9.4f}  "
            f"{r['t']:+7.3f}  "
            f"{pstr}  {ci}"
            if not np.isnan(r["t"])
            else f"  z < {r['threshold']:4.1f}    "
                 f"{int(r['N']):4d}  "
                 f"{r['CAAR']:+9.4f}  {'N/A':>7}  {'N/A':>8}  N/A"
        )
    lines.append("")

    # [2] POS 임계값 민감도
    lines += [
        "■ [2] 충격 임계값 민감도 (직접 효과, [0,+21], POS)",
        "",
        f"  {'임계값':10s}  {'N':>4}  {'CAAR':>9}  {'t':>7}  {'p':>8}  {'95% CI':>24}",
        "  " + "-" * 70,
    ]
    for _, r in pos_df.iterrows():
        p = r["p"]
        ci = (f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"
              if not np.isnan(r["ci_lo"]) else "N/A")
        pstr = f"{p:.4f}{sig_stars(p)}" if not np.isnan(p) else "  N/A   "
        lines.append(
            f"  z > {r['threshold']:4.1f}    "
            f"{int(r['N']):4d}  "
            f"{r['CAAR']:+9.4f}  "
            f"{r['t']:+7.3f}  "
            f"{pstr}  {ci}"
            if not np.isnan(r["t"])
            else f"  z > {r['threshold']:4.1f}    "
                 f"{int(r['N']):4d}  "
                 f"{r['CAAR']:+9.4f}  {'N/A':>7}  {'N/A':>8}  N/A"
        )
    lines.append("")

    # [3] Cluster SE
    lines += [
        f"■ [3] Cluster SE (이벤트-월 클러스터 부트스트랩, B={BOOTSTRAP_B})",
        "",
    ]
    if clust:
        contains_zero = clust["ci_lo"] <= 0 <= clust["ci_hi"]
        lines += [
            f"  N (이벤트)     = {clust['N']}건",
            f"  CAAR           = {clust['CAAR']:+.4f}",
            f"  SE (클러스터)  = {clust['se_cluster']:.4f}",
            f"  95% CI         = [{clust['ci_lo']:+.4f}, {clust['ci_hi']:+.4f}]",
            f"  p (부트스트랩) = {clust['p_boot']:.4f}{sig_stars(clust['p_boot'])}",
            f"  클러스터 수    = {clust['n_cluster']}개 고유 이벤트-월",
            "",
            f"  → 95% CI {'0 포함 (통계 비유의)' if contains_zero else '0 미포함 (통계 유의)'}:"
            f" 클러스터링 후에도 {'결론 약화' if contains_zero else '결론 유지'}",
        ]
    else:
        lines.append("  [Cluster SE 결과 없음]")
    lines.append("")

    # [4] 레이어별 분석
    lines += [
        "■ [4] 레이어별 임계값 분석",
        f"  (NEG: z<{LAYER_THR_NEG}, POS: z>{LAYER_THR_POS}, [0,+21], DIRECT)",
        "",
        f"  {'레이어':20s}  {'유형':5s}  {'N':>4}  {'CAAR':>9}  {'t':>7}  {'p':>8}",
        "  " + "-" * 60,
    ]

    for _, r in layer_df.iterrows():
        p = r["p"]
        pstr = f"{p:.4f}{sig_stars(p)}" if not np.isnan(p) else "  N/A   "
        tstr = f"{r['t']:+7.3f}" if not np.isnan(r["t"]) else "    N/A"
        lines.append(
            f"  {r['layer']:20s}  {r['shock_type']:5s}  "
            f"{int(r['N']):4d}  {r['CAAR']:+9.4f}  {tstr}  {pstr}"
            if not np.isnan(r["CAAR"])
            else f"  {r['layer']:20s}  {r['shock_type']:5s}  "
                 f"{int(r['N']):4d}  {'N/A':>9}  {'N/A':>7}  N/A"
        )

    lines += [
        "",
        "주: *** p<0.01  ** p<0.05  * p<0.10",
        SEP,
    ]

    return "\n".join(lines)


# ── Main ────────────────────────────────────────────────────────

def main():
    log.info("Phase 4 Robustness Checks 44개 기업 시작")

    # 데이터 로드
    base = load_base(PATH_PANEL, PATH_UNIV)

    # [1][2] 임계값 민감도
    log.info("[1][2] 임계값 민감도 분석 (NEG / POS)...")
    neg_df, pos_df = run_threshold_sensitivity(base)

    # [3] Cluster SE
    log.info("[3] Cluster SE (B=%d)...", BOOTSTRAP_B)
    clust = run_cluster_se(base)

    # [4] 레이어별 분석
    log.info("[4] 레이어별 분석...")
    layer_df = run_layer_analysis(base)

    # 보고서 생성
    report = build_report(base, neg_df, pos_df, clust, layer_df)
    OUT_RPT.write_text(report, encoding="utf-8")
    log.info("보고서 저장: %s", OUT_RPT)

    print("\n" + report)

    # 상세 CSV 저장
    detail_path = ROOT / "reports/phase4_robustness_44_detail.csv"
    pd.concat(
        [
            neg_df.assign(section="threshold_NEG"),
            pos_df.assign(section="threshold_POS"),
            layer_df.assign(section="layer"),
        ],
        ignore_index=True,
    ).to_csv(detail_path, index=False, encoding="utf-8-sig")
    log.info("상세 CSV 저장: %s", detail_path)


if __name__ == "__main__":
    main()
