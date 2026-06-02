"""
02_zscore_expanding.py
======================
26개 신규 기업의 tone monthly z-score 계산 — Expanding Window 방식.

기존 02_zscore_from_existing.py 의 look-ahead bias 수정:
  - 기존: 전체 기간(2017~2026) 통계로 모든 월 정규화  →  미래 데이터 누설
  - 수정: 월 t의 z-score는 t-1까지 reliable 데이터의 mean/std만 사용

출력: experiments/universe_44/data/processed/tone_new26_expanding.parquet
"""
from __future__ import annotations
import logging, sys
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]   # nabi_hyoghaw/
EXP  = ROOT / "experiments/universe_44"
OUT  = EXP  / "data/processed/tone_new26_expanding.parquet"

NEW_COS = [
    '300037.SZ','300207.SZ','920185.BJ','002340.SZ','688388.SH',
    '600110.SH','603659.SH','600733.SH','000625.SZ','600006.SH',
    '601238.SH','0175.HK','601633.SH','600418.SH','603993.SH',
    '002460.SZ','600362.SH','002466.SZ','3407.T','066970.KS',
    '003670.KS','F','GM','LCID','7203.T','RIVN'
]

SUPPLY_KW = {
    "supply","supplier","shortage","disruption","delay","delivery",
    "shipment","contract","mine","mining","refin","lithium","cobalt",
    "nickel","cathode","anode","precursor","production halt","factory",
    "recall","tariff","sanction","export ban","import","procurement",
}

REGIME_DATES = {
    "regime_covid":        ("2020-01", "2021-12"),
    "regime_ukraine":      ("2022-02", "2023-06"),
    "regime_ira":          ("2022-08", "2025-12"),
    "regime_china_export": ("2023-08", "2025-12"),
    "regime_trump_tariff": ("2025-01", "2027-12"),
}

FB2_STRICT   = 10
SUP_STRICT   = 5
MIN_HIST     = 12   # expanding window: t-1까지 최소 12개 reliable 월 필요
MIN_SUP_HIST = 6
Z_CLIP       = 5.0


def regime_flags(ym: str) -> dict:
    return {col: 1.0 if s <= ym <= e else 0.0
            for col, (s, e) in REGIME_DATES.items()}


def is_supply(title: str) -> bool:
    tl = str(title).lower()
    return any(kw in tl for kw in SUPPLY_KW)


def compute_company_zscores(cid_agg: pd.DataFrame) -> list[dict]:
    """
    Expanding window z-score 계산.
    월 t의 z는 t-1까지의 reliable 데이터로만 계산 (look-ahead bias 없음).
    """
    rows: list[dict] = []
    running_reliable_fb2: list[float] = []   # t-1까지 누적 (FB2)
    running_reliable_sup: list[float] = []   # t-1까지 누적 (Supply)
    running_z: list[float] = []              # lag 계산용

    for _, row in cid_agg.sort_values("ym").iterrows():
        n_fb2    = int(row.get("n_fb2", 0) or 0)
        fb2_mean = row.get("fb2_mean", np.nan)
        sup_n    = int(row.get("sup_n_fb2", 0) or 0)
        sup_mean = row.get("sup_fb2_mean", np.nan)
        ym_str   = str(row["ym"])

        # ── FB2 z-score (t-1까지 expanding 통계) ──────────────────
        has_hist = len(running_reliable_fb2) >= MIN_HIST
        current_reliable = (n_fb2 >= FB2_STRICT and not pd.isna(fb2_mean))

        if has_hist and current_reliable:
            mu_fb2  = np.mean(running_reliable_fb2)
            sig_fb2 = np.std(running_reliable_fb2, ddof=1) if len(running_reliable_fb2) > 1 else np.nan
            if sig_fb2 is not None and not np.isnan(sig_fb2) and sig_fb2 > 1e-10:
                z = float(np.clip((fb2_mean - mu_fb2) / sig_fb2, -Z_CLIP, Z_CLIP))
            else:
                z = np.nan
        else:
            z = np.nan

        # ── Supply z-score (t-1까지 expanding 통계) ────────────────
        has_sup_hist = len(running_reliable_sup) >= MIN_SUP_HIST
        current_sup_reliable = (sup_n >= SUP_STRICT and not pd.isna(sup_mean))

        if has_sup_hist and current_sup_reliable:
            mu_sup  = np.mean(running_reliable_sup)
            sig_sup = np.std(running_reliable_sup, ddof=1) if len(running_reliable_sup) > 1 else np.nan
            if sig_sup is not None and not np.isnan(sig_sup) and sig_sup > 1e-10:
                sz = float(np.clip((sup_mean - mu_sup) / sig_sup, -Z_CLIP, Z_CLIP))
            else:
                sz = np.nan
        else:
            sz = np.nan

        # ── Lag / rolling 피처 (running_z 기반) ────────────────────
        z_lag1  = running_z[-1] if len(running_z) >= 1 else np.nan
        z_lag2  = running_z[-2] if len(running_z) >= 2 else np.nan
        recent  = running_z[-2:] + ([z] if not np.isnan(z) else [])
        rolling = float(np.mean(recent[-3:])) if len(recent) >= 2 else np.nan
        delta   = float(z - z_lag1) if (not np.isnan(z) and not np.isnan(z_lag1)) else np.nan

        reliable = bool(
            not np.isnan(z) and has_hist and current_reliable
        )

        rows.append({
            "company_id":    row.get("company_id", ""),
            "year_month":    ym_str,
            "n_fb2":         n_fb2,
            "fb2_mean":      fb2_mean,
            "sup_n_fb2":     sup_n,
            "sup_fb2_mean":  sup_mean,
            "fb2_z_v3":      z,
            "z_score_v3":    z,
            "z_source_v3":   "finbert2_expanding" if reliable else "insufficient",
            "reliable_v3":   reliable,
            "supply_z_v3":   sz,
            "z_lag1_v3":     z_lag1,
            "z_lag2_v3":     z_lag2,
            "rolling_3m_v3": rolling,
            "delta_z_v3":    delta,
            "neg_shock_v3":  bool(not np.isnan(z) and z < -2.0 and reliable),
            "pos_shock_v3":  bool(not np.isnan(z) and z >  2.0 and reliable),
            **regime_flags(ym_str),
            "tone_wsum": np.nan, "wsum": np.nan, "n_tone": 0,
            "tone_mean": np.nan, "tone_z_v3": np.nan,
            "sup_n_tone": 0, "sup_tone_mean": np.nan,
            "neg_shock_sup_v3": bool(not np.isnan(sz) and sz < -2.0
                                     and current_sup_reliable),
            "pos_shock_sup_v3": bool(not np.isnan(sz) and sz >  2.0
                                     and current_sup_reliable),
            "z_cross_v3":   np.nan,
            "z_cross_stage": np.nan,
            # expanding window 메타데이터
            "expanding_n_hist": len(running_reliable_fb2),
        })

        # ── 현재 월 데이터를 running list에 추가 (다음 달부터 사용) ──
        if current_reliable:
            running_reliable_fb2.append(float(fb2_mean))
        if current_sup_reliable:
            running_reliable_sup.append(float(sup_mean))
        if not np.isnan(z):
            running_z.append(z)

    return rows


def add_cross_z(df_: pd.DataFrame) -> pd.DataFrame:
    """월별 기업간 교차 정규화 (z_cross_v3)."""
    def cx(g):
        v = g["z_score_v3"].dropna()
        if len(v) < 5:
            return g
        mu, sig = v.mean(), v.std(ddof=1)
        if sig > 1e-10:
            g = g.copy()
            g["z_cross_v3"] = (g["z_score_v3"] - mu) / sig
        return g
    return df_.groupby("year_month", group_keys=False).apply(cx)


def main():
    log.info("=== Expanding Window z-score: 26개 신규 기업 ===")
    log.info("출력: %s", OUT)

    # ── 1. 기존 이벤트 + FinBERT 로드 ─────────────────────────────
    log.info("1. risk_events_v10 로드 (신규 26개 기업만 필터)")
    ev = pd.read_parquet(
        ROOT / "data/processed/risk_events_classified_v10_new.parquet",
        columns=["url", "title", "company_id", "event_time"],
    )
    ev = ev[ev["company_id"].isin(NEW_COS)].copy()
    log.info("   신규 기업 이벤트: %d건", len(ev))

    log.info("2. finbert_v2 로드 및 조인")
    fb = pd.read_parquet(
        ROOT / "data/processed/finbert_v2.parquet",
        columns=["url", "finbert_score"],
    )
    merged = ev.merge(fb, on="url", how="inner")
    log.info("   조인 성공: %d건 (%.1f%%)", len(merged), 100 * len(merged) / max(1, len(ev)))

    # ── 2. 월별 집계 ───────────────────────────────────────────────
    log.info("3. 월별 집계")
    merged["event_time"] = pd.to_datetime(merged["event_time"], errors="coerce")
    merged = merged.dropna(subset=["event_time"])
    merged["ym"]       = merged["event_time"].dt.to_period("M").astype(str)
    merged["is_supply"] = merged["title"].apply(is_supply)
    merged["fb2_score"] = merged["finbert_score"].astype(float)

    agg = (
        merged.groupby(["company_id", "ym"])
        .agg(fb2_wsum=("fb2_score", "sum"), n_fb2=("fb2_score", "count"))
        .reset_index()
    )
    agg["fb2_mean"] = agg["fb2_wsum"] / agg["n_fb2"].replace(0, np.nan)

    sup = (
        merged[merged["is_supply"]].groupby(["company_id", "ym"])
        .agg(sup_n_fb2=("fb2_score", "count"), sup_fb2_wsum=("fb2_score", "sum"))
        .reset_index()
    )
    sup["sup_fb2_mean"] = sup["sup_fb2_wsum"] / sup["sup_n_fb2"].replace(0, np.nan)

    agg = agg.merge(
        sup[["company_id", "ym", "sup_n_fb2", "sup_fb2_mean"]],
        on=["company_id", "ym"],
        how="left",
    )
    agg["sup_n_fb2"] = agg["sup_n_fb2"].fillna(0).astype(int)
    log.info("   집계 완료: %d 기업-월 조합", len(agg))

    # ── 3. 커버리지 요약 ───────────────────────────────────────────
    log.info("\n=== 기업별 월 커버리지 ===")
    for cid in NEW_COS:
        sub = agg[agg["company_id"] == cid]
        reliable = sub[sub["n_fb2"] >= FB2_STRICT]
        log.info(
            "  %-15s  총%3d월  reliable(n>=%d):%3d월  fb2_mean평균=%.4f",
            cid, len(sub), FB2_STRICT, len(reliable),
            reliable["fb2_mean"].mean() if len(reliable) else float("nan"),
        )

    # ── 4. Expanding Window z-score 계산 ──────────────────────────
    log.info("\n4. Expanding Window z-score 계산 (MIN_HIST=%d개월)", MIN_HIST)
    all_rows: list[dict] = []
    for cid in NEW_COS:
        cid_agg = agg[agg["company_id"] == cid].copy()
        cid_agg["company_id"] = cid
        if cid_agg.empty:
            log.warning("   %s: 데이터 없음, 건너뜀", cid)
            continue
        rows = compute_company_zscores(cid_agg)
        all_rows.extend(rows)

    new_df = pd.DataFrame(all_rows)

    # ── 5. 월별 교차 z-score ──────────────────────────────────────
    new_df = add_cross_z(new_df)

    # ── 6. 저장 ────────────────────────────────────────────────────
    (EXP / "data/processed").mkdir(parents=True, exist_ok=True)
    new_df.to_parquet(OUT, index=False)
    log.info("\n저장: %s (%d행)", OUT.name, len(new_df))

    # ── 7. z-score 품질 요약 ──────────────────────────────────────
    log.info("\n=== z-score 품질 요약 (Expanding Window) ===")
    for cid in NEW_COS:
        sub = new_df[new_df["company_id"] == cid]
        rel = sub[sub["reliable_v3"] == True]
        log.info(
            "  %-15s  총%3d월  reliable:%3d월  z=[%.2f, %.2f]  mean=%.4f",
            cid, len(sub), len(rel),
            sub["z_score_v3"].min() if not sub["z_score_v3"].isna().all() else float("nan"),
            sub["z_score_v3"].max() if not sub["z_score_v3"].isna().all() else float("nan"),
            sub["z_score_v3"].mean() if not sub["z_score_v3"].isna().all() else float("nan"),
        )

    # ── 8. 기존(full-period) vs 수정(expanding) 비교 ─────────────
    old_path = EXP / "data/processed/tone_new26_from_existing.parquet"
    if old_path.exists():
        log.info("\n=== Expanding Window vs Full-Period 비교 ===")
        old_df = pd.read_parquet(old_path)
        log.info("  %-16s  %12s  %13s  %8s  %10s  %10s",
                 "기업", "full_mean_z", "expand_mean_z", "차이",
                 "full_rel월", "expand_rel월")
        log.info("  " + "-" * 80)

        for cid in NEW_COS:
            old_sub = old_df[old_df["company_id"] == cid]
            new_sub = new_df[new_df["company_id"] == cid]
            full_z   = old_sub["z_score_v3"].mean()
            exp_z    = new_sub["z_score_v3"].mean()
            diff     = exp_z - full_z if (not pd.isna(full_z) and not pd.isna(exp_z)) else float("nan")
            full_rel = int(old_sub["reliable_v3"].sum()) if "reliable_v3" in old_sub.columns else 0
            exp_rel  = int(new_sub["reliable_v3"].sum())
            log.info(
                "  %-16s  %+12.4f  %+13.4f  %+8.4f  %10d  %10d",
                cid,
                full_z if not pd.isna(full_z) else float("nan"),
                exp_z  if not pd.isna(exp_z)  else float("nan"),
                diff,
                full_rel,
                exp_rel,
            )
    else:
        log.info("기존 full-period 파일 없음 — 비교 생략")

    log.info("\n완료.")


if __name__ == "__main__":
    main()
