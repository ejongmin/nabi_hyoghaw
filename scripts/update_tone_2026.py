"""
2026년 1-5월 tone_monthly_zscore_18_v3 증분 업데이트
====================================================

전체 v10_new(1.25GB) 재처리 없이 신규 달만 추가하는 경량 파이프라인:

  [1] GDELT DOC API로 2026-01~05 기사 수집 (월별, 18개 기업 통합쿼리)
  [2] 회사명 매칭 (company_aliases.yaml 정규식)
  [3] FinBERT 감성 스코어 (ProsusAI/finbert, CPU)
  [4] 월별 집계 (fb2_mean, n_fb2, sup_fb2_mean, sup_n_fb2)
  [5] 기존 historical 통계로 z-score 계산 (mean/std from existing fb2_mean)
  [6] 파생 피처 (z_lag, delta, rolling, z_cross, regime, shock)
  [7] tone_monthly_zscore_18_v3.parquet에 append 후 재저장
"""
from __future__ import annotations
import logging, sys, time, re, warnings, calendar
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

ROOT    = Path(__file__).resolve().parents[1]
# 워크트리에서 실행 시 메인 레포 데이터 경로로 폴백
if not (ROOT / "data").exists():
    ROOT = Path("C:/Users/john9/nabi_hyoghaw")

C18     = ROOT / "data/universe/companies_18.csv"
TONE_V3 = ROOT / "data/processed/tone_monthly_zscore_18_v3.parquet"
ALIAS   = ROOT / "configs/company_aliases.yaml"
RAW_DIR = ROOT / "data/raw/gdelt_2026"

TARGET_MONTHS = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05"]

SUPPLY_KW = {
    "supply", "supplier", "shortage", "disruption", "delay", "delivery",
    "shipment", "contract", "mine", "mining", "refin", "lithium", "cobalt",
    "nickel", "cathode", "anode", "precursor", "production halt", "factory",
    "recall", "tariff", "sanction", "export ban", "import", "procurement",
}

REGIME_DATES = {
    "regime_covid":        ("2020-01", "2021-12"),
    "regime_ukraine":      ("2022-02", "2023-06"),
    "regime_ira":          ("2022-08", "2025-12"),
    "regime_china_export": ("2023-08", "2025-12"),
    "regime_trump_tariff": ("2025-01", "2027-12"),
}


# ─────────────────────────────────────────────────────────────
# 0. 유틸
# ─────────────────────────────────────────────────────────────

def load_aliases(path: Path) -> Dict[str, List[str]]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    result: Dict[str, List[str]] = {}
    if isinstance(data, dict):
        for company_id, info in data.items():
            if isinstance(info, dict):
                aliases = info.get("aliases", [])
            elif isinstance(info, list):
                aliases = info
            else:
                continue
            result[company_id] = [str(a) for a in aliases if a and len(str(a)) > 1]
    return result


def month_to_date_range(ym: str) -> Tuple[str, str]:
    y, m = int(ym[:4]), int(ym[5:7])
    last_day = calendar.monthrange(y, m)[1]
    return f"{ym}-01", f"{ym}-{last_day:02d}"


def is_supply_article(title: str) -> bool:
    tl = title.lower()
    return any(kw in tl for kw in SUPPLY_KW)


def regime_flags(ym: str) -> Dict[str, float]:
    return {col: 1.0 if s <= ym <= e else 0.0
            for col, (s, e) in REGIME_DATES.items()}


# ─────────────────────────────────────────────────────────────
# 1. GDELT DOC 수집
# ─────────────────────────────────────────────────────────────

# 기업별 GDELT 검색 키워드 (단일 단어, 5자 이상, 고유 명칭)
COMPANY_SEARCH_TERM: Dict[str, str] = {
    "ALB":        "Albemarle",
    "GLEN.L":     "Glencore",
    "BAS.DE":     "BASF",
    "300919.SZ":  "CNGR",
    "247540.KQ":  "EcoPro",
    "603799.SH":  "Huayou",
    "051910.KS":  "LGChem",
    "3931.HK":    "CALB",
    "300750.SZ":  "Amperex",       # CATL = Contemporary Amperex
    "300014.SZ":  "EVEEnergy",
    "002074.SZ":  "Gotion",
    "373220.KS":  "LGEnergy",
    "BMW.DE":     "BMW",
    "1211.HK":    "BYD",
    "7267.T":     "Honda",
    "MBG.DE":     "Mercedes",
    "TSLA":       "Tesla",
    "VOW3.DE":    "Volkswagen",
}


def _gdelt_query_one(keyword: str, start: str, end: str,
                     retries: int = 3) -> pd.DataFrame:
    """단일 키워드로 GDELT 쿼리. 실패 시 빈 DF 반환."""
    try:
        from gdeltdoc import GdeltDoc, Filters
    except ImportError:
        raise RuntimeError("pip install gdeltdoc 필요")

    gd = GdeltDoc()
    f  = Filters(keyword=keyword, start_date=start, end_date=end, language="English")
    for attempt in range(retries):
        try:
            df = gd.article_search(f)
            return df if (df is not None and len(df) > 0) else pd.DataFrame()
        except Exception as e:
            if attempt < retries - 1:
                wait = 3 * (2 ** attempt)
                log.debug("    재시도 %d (%.0fs): %s", attempt+1, wait, e)
                time.sleep(wait)
    return pd.DataFrame()


def collect_month(month: str, companies: List[str],
                  raw_dir: Path, force: bool = False) -> pd.DataFrame:
    """
    기업별 단일 키워드 쿼리로 한 달치 기사 수집.
    GDELT API 복합 OR 쿼리 이슈를 우회하기 위해 기업당 1 쿼리씩 실행.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache = raw_dir / f"gdelt_{month}.csv"
    if cache.exists() and not force:
        cached = pd.read_csv(cache)
        log.info("캐시 로드: %s (%d건)", cache.name, len(cached))
        return cached

    start, end = month_to_date_range(month)
    all_dfs: List[pd.DataFrame] = []

    for cid in companies:
        keyword = COMPANY_SEARCH_TERM.get(cid)
        if not keyword:
            continue
        df = _gdelt_query_one(keyword, start, end)
        if not df.empty:
            df["_matched_cid"] = cid
            all_dfs.append(df)
            log.info("  [%s] %s: %d건", cid, keyword, len(df))
        time.sleep(1.2)   # API rate limit

    if not all_dfs:
        log.warning("%s: 수집 결과 없음", month)
        result = pd.DataFrame(columns=["url","title","seendate","domain","_matched_cid"])
    else:
        result = pd.concat(all_dfs, ignore_index=True)
        if "url" in result.columns:
            result = result.drop_duplicates(subset=["url"])
        log.info("  %s: 총 %d건 수집 (중복 제거 후)", month, len(result))

    result.to_csv(cache, index=False, encoding="utf-8-sig")
    return result


# ─────────────────────────────────────────────────────────────
# 2. 회사 매칭
# ─────────────────────────────────────────────────────────────

def match_articles_to_companies(df: pd.DataFrame,
                                 aliases: Dict[str, List[str]],
                                 companies: List[str]) -> pd.DataFrame:
    if df.empty or "title" not in df.columns:
        return pd.DataFrame()

    patterns: Dict[str, re.Pattern] = {}
    for cid in companies:
        terms = sorted(set(aliases.get(cid, []) + [cid]), key=len, reverse=True)
        pat = "|".join(re.escape(t) for t in terms if len(t) > 1)
        if pat:
            patterns[cid] = re.compile(pat, re.IGNORECASE)

    rows = []
    for _, art in df.iterrows():
        title  = str(art.get("title", "") or "")
        seen   = str(art.get("seendate", "") or "")
        url    = str(art.get("url", "") or "")
        supply = is_supply_article(title)
        for cid, pat in patterns.items():
            if pat.search(title):
                rows.append({"company_id": cid, "title": title,
                             "seendate": seen, "url": url, "is_supply": supply})

    log.info("  매칭: %d 기사→기업 쌍 (%d개 원기사에서)", len(rows), len(df))
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# 3. FinBERT
# ─────────────────────────────────────────────────────────────

def run_finbert(titles: List[str], batch_size: int = 32) -> List[float]:
    if not titles:
        return []
    try:
        from transformers import pipeline
    except ImportError:
        raise RuntimeError("pip install transformers torch 필요")

    log.info("FinBERT 로딩 (ProsusAI/finbert)...")
    nlp = pipeline("text-classification", model="ProsusAI/finbert",
                   tokenizer="ProsusAI/finbert", top_k=None,
                   device=-1, truncation=True, max_length=512)

    scores = []
    total  = len(titles)
    for i in range(0, total, batch_size):
        batch = titles[i: i + batch_size]
        try:
            results = nlp(batch)
        except Exception as e:
            log.warning("  배치 오류 (%d~%d): %s", i, i+len(batch), e)
            scores.extend([0.0] * len(batch))
            continue
        for item in results:
            lm = {r["label"].lower(): r["score"] for r in item}
            scores.append(float(lm.get("positive", 0.0) - lm.get("negative", 0.0)))
        if (i // batch_size) % 5 == 0:
            log.info("  FinBERT: %d/%d", min(i + batch_size, total), total)
    return scores


# ─────────────────────────────────────────────────────────────
# 4. 월별 집계
# ─────────────────────────────────────────────────────────────

def aggregate_monthly(matched_df: pd.DataFrame) -> pd.DataFrame:
    if matched_df.empty:
        return pd.DataFrame()
    df = matched_df.dropna(subset=["fb2_score"]).copy()

    agg = (df.groupby(["company_id", "ym"])
             .agg(fb2_wsum=("fb2_score","sum"), n_fb2=("fb2_score","count"))
             .reset_index())
    agg["fb2_mean"] = agg["fb2_wsum"] / agg["n_fb2"].replace(0, np.nan)

    sup = (df[df["is_supply"]].groupby(["company_id","ym"])
             .agg(sup_fb2_wsum=("fb2_score","sum"), sup_n_fb2=("fb2_score","count"))
             .reset_index())
    sup["sup_fb2_mean"] = sup["sup_fb2_wsum"] / sup["sup_n_fb2"].replace(0, np.nan)

    result = agg.merge(sup[["company_id","ym","sup_n_fb2","sup_fb2_mean"]],
                       on=["company_id","ym"], how="left")
    result["sup_n_fb2"]    = result["sup_n_fb2"].fillna(0).astype(int)
    result["sup_fb2_mean"] = result["sup_fb2_mean"].fillna(np.nan)
    return result


# ─────────────────────────────────────────────────────────────
# 5. 증분 z-score 계산
# ─────────────────────────────────────────────────────────────

def compute_incremental_zscores(new_agg: pd.DataFrame, existing: pd.DataFrame,
                                 companies: List[str],
                                 fb2_strict: int = 10, sup_strict: int = 5,
                                 shock_thr: float = 2.0) -> pd.DataFrame:
    rows = []
    for cid in companies:
        new_cid = new_agg[new_agg["company_id"] == cid].copy()
        if new_cid.empty:
            continue
        ex_cid = existing[existing["company_id"] == cid].sort_values("year_month")

        hist_fb2 = ex_cid.loc[ex_cid["reliable_v3"] == True, "fb2_mean"].dropna()
        mu_fb2   = hist_fb2.mean() if len(hist_fb2) >= 12 else 0.0
        sig_fb2  = hist_fb2.std(ddof=1) if len(hist_fb2) >= 12 else 1.0

        hist_sup = ex_cid.loc[ex_cid["sup_n_fb2"].fillna(0) >= sup_strict, "sup_fb2_mean"].dropna()
        mu_sup   = hist_sup.mean() if len(hist_sup) >= 6 else 0.0
        sig_sup  = hist_sup.std(ddof=1) if len(hist_sup) >= 6 else 1.0

        for _, row in new_cid.iterrows():
            fb2_mean     = row.get("fb2_mean", np.nan)
            n_fb2        = int(row.get("n_fb2", 0))
            sup_fb2_mean = row.get("sup_fb2_mean", np.nan)
            sup_n_fb2    = int(row.get("sup_n_fb2", 0))
            ym_str       = str(row["ym"])
            reliable     = n_fb2 >= fb2_strict

            z        = ((fb2_mean - mu_fb2) / sig_fb2 if sig_fb2 > 1e-10 else 0.0) \
                       if (reliable and not np.isnan(fb2_mean)) else np.nan
            z_source = "finbert2" if reliable and not np.isnan(fb2_mean) else "insufficient"
            sz       = ((sup_fb2_mean - mu_sup) / sig_sup if sig_sup > 1e-10 else 0.0) \
                       if (sup_n_fb2 >= sup_strict and not np.isnan(sup_fb2_mean)) else np.nan

            prev_z      = ex_cid["z_score_v3"].dropna()
            z_lag1      = prev_z.iloc[-1] if len(prev_z) >= 1 else np.nan
            z_lag2      = prev_z.iloc[-2] if len(prev_z) >= 2 else np.nan
            recent      = list(prev_z.tail(2))
            if not np.isnan(z): recent.append(z)
            rolling     = float(np.mean(recent[-3:])) if len(recent) >= 2 else np.nan
            delta       = float(z - z_lag1) if not np.isnan(z) and not np.isnan(z_lag1) else np.nan

            rows.append({
                "company_id":       cid,
                "tone_wsum":        np.nan, "wsum": np.nan, "n_tone": 0,
                "n_fb2":            n_fb2,
                "fb2_wsum":         fb2_mean * n_fb2 if not np.isnan(fb2_mean) else np.nan,
                "tone_mean":        np.nan, "fb2_mean": fb2_mean,
                "sup_n_tone":       0, "sup_n_fb2": sup_n_fb2,
                "sup_tone_mean":    np.nan, "sup_fb2_mean": sup_fb2_mean,
                "fb2_z_v3":         z, "tone_z_v3": np.nan,
                "z_score_v3":       z, "z_source_v3": z_source,
                "reliable_v3":      reliable,
                "supply_z_v3":      sz,
                "z_lag1_v3":        z_lag1, "z_lag2_v3": z_lag2,
                "delta_z_v3":       delta,  "rolling_3m_v3": rolling,
                "neg_shock_v3":     bool(not np.isnan(z) and z < -shock_thr and reliable),
                "pos_shock_v3":     bool(not np.isnan(z) and z > shock_thr  and reliable),
                "neg_shock_sup_v3": bool(not np.isnan(sz) and sz < -shock_thr and sup_n_fb2 >= sup_strict),
                "pos_shock_sup_v3": bool(not np.isnan(sz) and sz > shock_thr  and sup_n_fb2 >= sup_strict),
                "z_cross_v3":       np.nan,
                **regime_flags(ym_str),
                "z_cross_stage":    np.nan,
                "year_month":       ym_str,
            })
    return pd.DataFrame(rows)


def add_cross_z(df: pd.DataFrame) -> pd.DataFrame:
    def cx(g):
        v = g["z_score_v3"].dropna()
        if len(v) < 5:
            return g
        mu, sig = v.mean(), v.std(ddof=1)
        if sig > 1e-10:
            g["z_cross_v3"] = (g["z_score_v3"] - mu) / sig
        return g
    return df.groupby("year_month", group_keys=False).apply(cx)


# ─────────────────────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────────────────────

def main():
    log.info("=== 2026년 tone 증분 업데이트 시작 ===")

    c18       = pd.read_csv(C18)
    companies = c18["company_id"].tolist()
    aliases   = load_aliases(ALIAS)
    existing  = pd.read_parquet(TONE_V3)
    log.info("기존 tone: %d행 (%s ~ %s)",
             len(existing), existing["year_month"].min(), existing["year_month"].max())

    existing_months = set(existing["year_month"].unique())
    new_months = [m for m in TARGET_MONTHS if m not in existing_months]
    if not new_months:
        log.info("모든 대상 달이 이미 존재합니다.")
        return
    log.info("추가할 달: %s", new_months)

    log.info("검색 키워드: %s", {c: COMPANY_SEARCH_TERM.get(c,'?') for c in companies})

    all_agg: List[pd.DataFrame] = []

    for month in new_months:
        log.info("── %s 처리 ──", month)

        raw_df  = collect_month(month, companies, RAW_DIR)
        if raw_df.empty:
            continue

        matched = match_articles_to_companies(raw_df, aliases, companies)
        if matched.empty:
            continue

        matched["ym"] = month
        log.info("  FinBERT 실행: %d건", len(matched))
        matched["fb2_score"] = run_finbert(matched["title"].tolist())

        agg_m = aggregate_monthly(matched)
        if not agg_m.empty:
            all_agg.append(agg_m)
            log.info("  집계: %d 기업-월", len(agg_m))
        time.sleep(2)

    if not all_agg:
        log.error("수집된 데이터 없음.")
        return

    new_agg  = pd.concat(all_agg, ignore_index=True)
    new_rows = compute_incremental_zscores(new_agg, existing, companies)
    new_rows = add_cross_z(new_rows)

    combined = pd.concat(
        [existing.drop(columns=["ym"], errors="ignore"), new_rows],
        ignore_index=True
    ).sort_values(["company_id", "year_month"]).reset_index(drop=True)

    combined.to_parquet(TONE_V3, index=False)
    log.info("저장 완료: %s (%d행, %d ~ %s)",
             TONE_V3.name, len(combined),
             combined["year_month"].min(), combined["year_month"].max())

    log.info("\n=== 업데이트 요약 ===")
    for ym in sorted(new_rows["year_month"].unique()):
        sub = new_rows[new_rows["year_month"] == ym]
        log.info("  %s: %d기업, reliable=%d, mean_z=%.3f",
                 ym, len(sub), int(sub["reliable_v3"].sum()),
                 sub["z_score_v3"].mean() if sub["reliable_v3"].any() else float("nan"))


if __name__ == "__main__":
    main()
