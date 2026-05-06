#!/usr/bin/env python3
"""
Phase 4: CAR Event Study (나비효과 프로젝트)
============================================
감성 충격(neg/pos shock) → 시장모형 CAR → 공급망 전파(contagion) 분석

■ 이벤트 정의
  - 출처: tone_monthly_zscore_18.parquet (neg_shock=True or pos_shock=True, reliable=True)
  - T=0  : 충격월 M 의 다음달(M+1) 첫 거래일
           (월별 감성은 월말에 확정 → 월내 선행편의 방지)

■ 시장 모형
  - 추정기간: T=0 기준 [-250, -20] 거래일 (실사용 가능 일수 min 60td)
  - 정상수익률: α_hat + β_hat × R_mkt
  - 기업별 지역 벤치마크: CSI300(중국/홍콩), KOSPI(한국), NIKKEI225(일본),
                           SP500(미국), STOXX600(유럽)

■ CAR 계산 & 통계 검정
  - 이벤트 창: [0,+1], [0,+5], [0,+10], [0,+21], [-1,+1], [-5,+5] (거래일)
  - Patell (1976) 표준화 t-통계량
  - 교차단면 t-검정 (CAAR / SE_cross)
  - 비모수 부호 검정 (Generalized Sign Test)

■ 공급망 전파 (Contagion)
  - 충격 기업 C → 내부 엣지로 연결된 기업의 CAR 분석
  - 관계 방향: DOWNSTREAM(C→neighbor, C가 공급자),
               UPSTREAM(neighbor→C, C가 수요자),
               PEER(COMPETES_WITH)
  - 1홉 직접 + 2홉 간접 구분

■ 출력
  - data/processed/car_panel_phase4.parquet   : 전체 CAR 패널
  - reports/phase4_caar_summary.csv           : CAAR 요약 통계
  - reports/phase4_heterogeneity.csv          : 이질성 분석
  - reports/phase4_contagion.csv              : 공급망 전파 분석
  - reports/phase4_summary.txt               : 전체 실행 요약 보고서

참고 문헌:
  Patell (1976), J. Accounting Research
  Boehmer, Musumeci & Poulsen (1991), J. Financial Economics
  MacKinlay (1997), J. Economic Literature
  Brown & Warner (1985), J. Financial Economics
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
# 로깅 설정
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 경로 & 상수
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]

PATH_TONE      = ROOT / "data/processed/tone_monthly_zscore_18.parquet"
PATH_PRICES    = ROOT / "data/processed/prices_daily.parquet"
PATH_MKT       = ROOT / "data/processed/market_proxies.parquet"
PATH_MKT_MAP   = ROOT / "data/processed/market_proxy_mapping.yaml"
PATH_EDGES_INT = ROOT / "data/seed/seed_edges_18_internal.csv"
PATH_C18       = ROOT / "data/universe/companies_18.csv"

OUT_PANEL      = ROOT / "data/processed/car_panel_phase4.parquet"
OUT_CAAR       = ROOT / "reports/phase4_caar_summary.csv"
OUT_HETERO     = ROOT / "reports/phase4_heterogeneity.csv"
OUT_CONTAGION  = ROOT / "reports/phase4_contagion.csv"
OUT_REPORT     = ROOT / "reports/phase4_summary.txt"

# 이벤트 창 정의: (시작 offset, 끝 offset, 창 이름)
# offset 0 = T=0 (첫 거래일)
EVENT_WINDOWS: List[Tuple[int, int, str]] = [
    (0,  1,  "0_1"),
    (0,  5,  "0_5"),
    (0,  10, "0_10"),
    (0,  21, "0_21"),   # ★ 주 분석 창 (1개월 전진)
    (-1, 1,  "m1_1"),
    (-5, 5,  "m5_5"),
]

# 추정 기간 (T=0 기준 거래일)
EST_WIN_START = -250   # T-250
EST_WIN_END   = -20    # T-20
MIN_EST_DAYS  = 60     # 절대 하한 (Brown & Warner 1985 권장: 100+; 여기선 상장 초기 기업 포용)
PREFERRED_EST = 120    # 이 미만이면 'marginal' 품질 플래그

# 충격 임계값
SHOCK_THR = 2.0

# 공급망 contagion: 최대 홉 수
MAX_HOPS = 2

# 재현성
REPORT_LINES: List[str] = []


# ─────────────────────────────────────────────────────────────────────────────
# 데이터 클래스
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CompanyData:
    """기업별 정렬된 가격·시장 수익률 배열"""
    company_id: str
    dates: np.ndarray       # datetime64[D], 거래일 배열 (오름차순)
    ret: np.ndarray         # float64, 일별 수익률
    mkt_ret: np.ndarray     # float64, 시장 수익률 (동일 인덱스)
    date_to_idx: Dict       # np.datetime64 → int 인덱스
    mkt_index: str          # 사용된 시장 인덱스명
    list_date: np.datetime64  # 최초 거래일


@dataclass
class Event:
    """단일 이벤트 (충격 기업 × 충격 월)"""
    event_id: str
    shock_company: str
    shock_month: str        # "YYYY-MM" 형식
    shock_type: str         # "NEG" | "POS"
    z_score: float
    z_source: str           # "finbert" | "v2tone"


@dataclass
class CARResult:
    """단일 CAR 계산 결과"""
    event_id: str
    shock_company: str
    shock_month: str
    shock_type: str
    z_score: float
    target_company: str
    relationship: str       # "DIRECT" | "DOWNSTREAM_1" | "UPSTREAM_1" | ... | "PEER_1"
    rel_type: str           # 원본 엣지 관계명 (DIRECT면 "SELF")
    window_label: str
    win_start: int
    win_end: int
    t0_date: str            # YYYY-MM-DD
    CAR: float
    t_stat_patell: float
    p_patell: float
    alpha: float
    beta: float
    sigma_ar: float
    est_len: int
    quality: str            # "full" | "marginal"
    confounding_self: bool  # 동일 기업 다른 이벤트가 창 내 존재
    confounding_target: bool  # 대상 기업이 창 내 자체 이벤트 보유


# ─────────────────────────────────────────────────────────────────────────────
# 1. 데이터 로딩 & 검증
# ─────────────────────────────────────────────────────────────────────────────

def load_events() -> List[Event]:
    """tone_monthly_zscore_18에서 이벤트 목록 로드"""
    log.info("이벤트 로딩: %s", PATH_TONE)
    df = pd.read_parquet(PATH_TONE)

    required_cols = {"company_id", "year_month", "neg_shock", "pos_shock",
                     "reliable", "z_score", "z_source"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"tone_monthly_zscore에 컬럼 없음: {missing}")

    # reliable=True이고 neg 또는 pos 충격인 행만
    mask = df["reliable"] & (df["neg_shock"] | df["pos_shock"])
    df_ev = df[mask].copy()

    events: List[Event] = []
    seen: set = set()
    for _, r in df_ev.iterrows():
        eid = f"{r['company_id']}_{r['year_month']}"
        if eid in seen:
            log.warning("중복 이벤트 ID: %s → 건너뜀", eid)
            continue
        seen.add(eid)

        shock_type = "NEG" if r["neg_shock"] else "POS"
        # pos이면서 neg도 True인 경우 (|z|>2인데 양쪽 모두 True는 논리적으로 불가능하나 방어)
        if r["neg_shock"] and r["pos_shock"]:
            shock_type = "NEG" if r["z_score"] < 0 else "POS"

        events.append(Event(
            event_id=eid,
            shock_company=str(r["company_id"]),
            shock_month=str(r["year_month"]),
            shock_type=shock_type,
            z_score=float(r["z_score"]),
            z_source=str(r["z_source"]),
        ))

    log.info("  총 이벤트: %d건 (NEG=%d, POS=%d)",
             len(events),
             sum(1 for e in events if e.shock_type == "NEG"),
             sum(1 for e in events if e.shock_type == "POS"))
    return events


def load_prices_and_market() -> Tuple[pd.DataFrame, Dict[str, pd.Series], Dict[str, str]]:
    """
    prices_daily, market_proxies, market_proxy_mapping 로드
    Returns: (prices_df, index_returns, company_to_index)
    """
    log.info("주가 데이터 로딩: %s", PATH_PRICES)
    prices = pd.read_parquet(PATH_PRICES)
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["company_id", "date"]).reset_index(drop=True)

    # adj_close 우선, 없으면 close
    prices["price"] = prices["adj_close"].fillna(prices["close"])
    nan_price = prices["price"].isna().sum()
    if nan_price > 0:
        log.warning("  가격 NaN %d건 → ffill 처리", nan_price)
        prices["price"] = prices.groupby("company_id")["price"].transform(
            lambda s: s.ffill().bfill()
        )

    log.info("  주가 로드 완료: %d행, %d기업",
             len(prices), prices["company_id"].nunique())

    log.info("시장 인덱스 로딩: %s", PATH_MKT)
    mkt_df = pd.read_parquet(PATH_MKT)
    mkt_df["date"] = pd.to_datetime(mkt_df["date"])

    # 컬럼명 통일 (index_id 또는 index_name)
    idx_col = "index_id" if "index_id" in mkt_df.columns else "index_name"
    mkt_df = mkt_df.sort_values([idx_col, "date"])

    index_price_col = "adj_close" if "adj_close" in mkt_df.columns else "close"
    mkt_df["mkt_ret"] = mkt_df.groupby(idx_col)[index_price_col].pct_change()
    mkt_df = mkt_df.dropna(subset=["mkt_ret"])

    index_returns: Dict[str, pd.Series] = {}
    for iname, g in mkt_df.groupby(idx_col):
        s = g.set_index("date")["mkt_ret"].sort_index()
        s.name = "mkt_ret"
        index_returns[str(iname)] = s

    log.info("  인덱스 로드: %s", list(index_returns.keys()))

    log.info("마켓 프록시 매핑 로딩: %s", PATH_MKT_MAP)
    with open(PATH_MKT_MAP, "r", encoding="utf-8") as f:
        mapping = yaml.safe_load(f)
    company_to_index: Dict[str, str] = mapping.get("company_to_market_proxy", {})
    log.info("  매핑 기업 수: %d", len(company_to_index))

    return prices, index_returns, company_to_index


def build_company_data(
    prices: pd.DataFrame,
    index_returns: Dict[str, pd.Series],
    company_to_index: Dict[str, str],
    target_companies: List[str],
) -> Dict[str, CompanyData]:
    """
    기업별 (ret, mkt_ret) 정렬 배열 생성.
    시장 프록시가 없는 기업은 equal-weight fallback(내부 18개 기업 평균).
    """
    # equal-weight fallback: 내부 18개 기업 평균 수익률
    ew_fallback = (
        prices[prices["company_id"].isin(target_companies)]
        .assign(ret=lambda d: d.groupby("company_id")["price"].transform(
            lambda s: s.pct_change()))
        .dropna(subset=["ret"])
        .groupby("date")["ret"]
        .mean()
        .rename("mkt_ret")
    )

    company_data: Dict[str, CompanyData] = {}
    n_regional, n_fallback = 0, 0

    for cid, g in prices[prices["company_id"].isin(target_companies)].groupby("company_id"):
        g = g.sort_values("date").copy()
        g["ret"] = g["price"].pct_change()
        g = g.dropna(subset=["ret"])

        if len(g) < 30:
            log.warning("  %s: 가격 데이터 부족(%d일) → 제외", cid, len(g))
            continue

        # 시장 프록시 선택
        idx_name = company_to_index.get(cid)
        if idx_name and idx_name in index_returns:
            mkt_s = index_returns[idx_name]
            n_regional += 1
        else:
            log.warning("  %s: 마켓 프록시 없음 → equal-weight fallback", cid)
            mkt_s = ew_fallback
            idx_name = "EW_FALLBACK"
            n_fallback += 1

        # 날짜 inner join
        merged = g.set_index("date")[["ret"]].join(mkt_s.to_frame(), how="inner").dropna()
        if len(merged) < 30:
            log.warning("  %s: 조인 후 데이터 부족(%d일) → 제외", cid, len(merged))
            continue

        dates = merged.index.values.astype("datetime64[D]")
        company_data[cid] = CompanyData(
            company_id=cid,
            dates=dates,
            ret=merged["ret"].values.astype(np.float64),
            mkt_ret=merged["mkt_ret"].values.astype(np.float64),
            date_to_idx={d: i for i, d in enumerate(dates)},
            mkt_index=str(idx_name),
            list_date=dates[0],
        )

    log.info("CompanyData 구축: %d기업 (regional=%d, fallback=%d)",
             len(company_data), n_regional, n_fallback)
    return company_data


# ─────────────────────────────────────────────────────────────────────────────
# 2. KG 엣지 & Contagion 구조 구축
# ─────────────────────────────────────────────────────────────────────────────

def load_internal_edges() -> pd.DataFrame:
    """seed_edges_18_internal.csv 로드 및 검증"""
    log.info("내부 엣지 로딩: %s", PATH_EDGES_INT)
    edges = pd.read_csv(PATH_EDGES_INT)
    required = {"src_company_id", "rel_type", "dst_company_id", "confidence_plink", "strength"}
    missing = required - set(edges.columns)
    if missing:
        raise ValueError(f"seed_edges_18_internal에 컬럼 없음: {missing}")
    log.info("  엣지: %d건, rel_type: %s",
             len(edges), edges["rel_type"].value_counts().to_dict())
    return edges


def build_adjacency(edges: pd.DataFrame) -> Dict[str, List[dict]]:
    """
    {company_id: [{neighbor, rel_type, direction, weight, hops}, ...]}
    direction:
      - 'DOWNSTREAM': src=shock_company, dst=neighbor (C→neighbor: C가 공급자)
      - 'UPSTREAM':   src=neighbor, dst=shock_company (neighbor→C: C가 수요자)
      - 'PEER':       COMPETES_WITH / PARTNERS_WITH
    """
    adj: Dict[str, List[dict]] = defaultdict(list)

    for _, r in edges.iterrows():
        src = r["src_company_id"]
        dst = r["dst_company_id"]
        rtype = r["rel_type"]
        weight = float(r.get("confidence_plink", 1.0)) * float(r.get("strength", 1.0))

        if rtype in ("SUPPLIES", "BUYS_FROM", "SOURCES_FROM", "OWNS", "DEPENDS_ON_ROUTE"):
            # src → dst (src가 공급자)
            adj[src].append({"neighbor": dst, "rel_type": rtype,
                             "direction": "DOWNSTREAM", "weight": weight, "hops": 1})
            adj[dst].append({"neighbor": src, "rel_type": rtype,
                             "direction": "UPSTREAM", "weight": weight, "hops": 1})
        elif rtype in ("COMPETES_WITH", "PARTNERS_WITH"):
            adj[src].append({"neighbor": dst, "rel_type": rtype,
                             "direction": "PEER", "weight": weight, "hops": 1})
            adj[dst].append({"neighbor": src, "rel_type": rtype,
                             "direction": "PEER", "weight": weight, "hops": 1})
        else:
            # 기타 관계는 양방향 PEER로 처리
            adj[src].append({"neighbor": dst, "rel_type": rtype,
                             "direction": "PEER", "weight": weight, "hops": 1})
            adj[dst].append({"neighbor": src, "rel_type": rtype,
                             "direction": "PEER", "weight": weight, "hops": 1})

    # 2홉 탐색
    for center in list(adj.keys()):
        one_hop = {entry["neighbor"] for entry in adj[center]}
        for entry1 in list(adj[center]):
            n1 = entry1["neighbor"]
            for entry2 in adj.get(n1, []):
                n2 = entry2["neighbor"]
                if n2 == center or n2 in one_hop:
                    continue
                # 2홉 방향: entry1.direction + entry2.direction
                if entry1["direction"] == entry2["direction"]:
                    dir2 = entry1["direction"] + "_2"
                else:
                    dir2 = "MIXED_2"
                adj[center].append({
                    "neighbor": n2,
                    "rel_type": f"{entry1['rel_type']}->{entry2['rel_type']}",
                    "direction": dir2,
                    "weight": entry1["weight"] * entry2["weight"],
                    "hops": 2,
                })

    return dict(adj)


# ─────────────────────────────────────────────────────────────────────────────
# 3. OLS 시장 모형 + Patell 통계
# ─────────────────────────────────────────────────────────────────────────────

def ols_car(
    ret: np.ndarray,
    mkt: np.ndarray,
    est_start: int,
    est_end: int,
    evt_start: int,
    evt_end: int,
) -> Optional[dict]:
    """
    OLS 시장 모형 → CAR + Patell (1976) 표준화 t-통계량

    Parameters
    ----------
    ret, mkt  : 동일 길이의 수익률 배열 (already aligned)
    est_start, est_end  : 추정기간 슬라이스 인덱스 [est_start, est_end)
    evt_start, evt_end  : 이벤트 창 슬라이스 인덱스 [evt_start, evt_end)

    Returns
    -------
    dict with keys: CAR, t_stat_patell, p_patell, alpha, beta,
                    sigma_ar, est_len, evt_len
    None if data insufficient
    """
    est_len = est_end - est_start
    if est_len < MIN_EST_DAYS:
        return None
    if est_start < 0 or est_end > len(ret):
        return None
    if evt_start < 0 or evt_end > len(ret):
        return None

    est_r = ret[est_start:est_end]
    est_m = mkt[est_start:est_end]
    evt_r = ret[evt_start:evt_end]
    evt_m = mkt[evt_start:evt_end]
    evt_len = len(evt_r)

    if evt_len == 0:
        return None
    # 이벤트 창 80% 이상 데이터 요구
    expected_len = evt_end - evt_start
    if evt_len < max(1, int(expected_len * 0.8)):
        return None

    # ── OLS (numpy 수치 안정성 보장) ──
    n = est_len
    sum_m   = est_m.sum()
    sum_r   = est_r.sum()
    sum_mm  = (est_m * est_m).sum()
    sum_mr  = (est_m * est_r).sum()

    denom = n * sum_mm - sum_m * sum_m
    if abs(denom) < 1e-15:
        return None

    beta  = (n * sum_mr  - sum_m * sum_r) / denom
    alpha = (sum_r - beta * sum_m) / n

    # ── 추정기간 잔차 → σ_AR ──
    expected_est = alpha + beta * est_m
    residuals    = est_r - expected_est
    # df = n-2 (alpha, beta 추정)
    sigma_ar = float(np.sqrt(np.sum(residuals ** 2) / max(1, n - 2)))
    if sigma_ar < 1e-10:
        return None

    # ── 이벤트 창 AR & CAR ──
    expected_evt = alpha + beta * evt_m
    ar  = evt_r - expected_evt
    car = float(ar.sum())

    # ── Patell (1976) 표준화 t-통계량 ──
    # Var(CAR) ≈ L2 × σ²_AR × (1 + 1/L1 + (mkt_mean_evt - mkt_mean_est)² / S²_mkt_est)
    # 여기서 L2=이벤트 창 길이, L1=추정 기간 길이
    # prediction error 보정 항 포함 (Brown & Warner 1985, eq. 6)
    mkt_bar_est = est_m.mean()
    S2_mkt_est  = np.sum((est_m - mkt_bar_est) ** 2)

    mkt_mean_evt = evt_m.mean() if evt_len > 0 else 0.0
    correction = 1.0 + 1.0 / n
    if S2_mkt_est > 1e-15:
        correction += (mkt_mean_evt - mkt_bar_est) ** 2 / S2_mkt_est

    # 표준화된 SAR = CAR / (σ_AR × sqrt(L2 × correction))
    denom_patell = sigma_ar * np.sqrt(evt_len * correction)
    if denom_patell < 1e-15:
        return None

    t_stat = car / denom_patell
    # p-value: two-tailed t-distribution, df = est_len - 2
    p_value = float(2.0 * stats.t.sf(abs(t_stat), df=max(1, n - 2)))

    return {
        "CAR":          car,
        "t_stat_patell": float(t_stat),
        "p_patell":     p_value,
        "alpha":        float(alpha),
        "beta":         float(beta),
        "sigma_ar":     float(sigma_ar),
        "est_len":      int(est_len),
        "evt_len":      int(evt_len),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. 이벤트 날짜 → T=0 인덱스 결정
# ─────────────────────────────────────────────────────────────────────────────

def find_t0_idx(cd: CompanyData, shock_month: str) -> Optional[int]:
    """
    T=0 = 충격월 M+1의 첫 거래일 인덱스 (company_data 기준)

    충격월 M의 월별 감성은 월말에 확정되므로,
    T=0을 M+1 첫 거래일로 설정하면 월내 선행편의(look-ahead bias)를 완전히 방지.

    Returns
    -------
    int or None (데이터 범위 밖이면 None)
    """
    # shock_month "YYYY-MM" → 다음달 1일
    try:
        t0_month_start = (
            pd.Period(shock_month, "M") + 1
        ).to_timestamp()  # M+1 의 1일 (자정)
    except Exception:
        return None

    t0_ts64 = np.datetime64(t0_month_start.date(), "D")

    # cd.dates에서 t0_ts64 이상인 첫 번째 거래일 탐색
    pos = np.searchsorted(cd.dates, t0_ts64, side="left")
    if pos >= len(cd.dates):
        return None
    # M+1 의 첫 거래일이 M+2 이상이면 (예: 휴장 연속) 허용 — 단 M+2 초과는 경고
    actual_date = cd.dates[pos]
    actual_ts = pd.Timestamp(str(actual_date))
    t0_expected_end = t0_month_start + pd.offsets.MonthEnd(1)
    if actual_ts > t0_expected_end:
        log.debug("  T=0이 M+1 범위 초과: %s shock_month=%s → actual=%s",
                  cd.company_id, shock_month, actual_ts.date())
    return int(pos)


def get_estimation_indices(
    cd: CompanyData, t0_idx: int
) -> Tuple[int, int, str]:
    """
    추정 기간 인덱스 계산.

    Returns
    -------
    (est_start, est_end, quality)
    quality: "full" | "marginal" | "insufficient"
    """
    est_end   = t0_idx + EST_WIN_END    # T-20
    est_start = t0_idx + EST_WIN_START  # T-250

    # 배열 경계 클램핑
    est_start = max(0, est_start)
    est_end   = min(len(cd.dates), est_end)

    actual_len = est_end - est_start
    if actual_len < MIN_EST_DAYS:
        return est_start, est_end, "insufficient"
    elif actual_len < PREFERRED_EST:
        return est_start, est_end, "marginal"
    else:
        return est_start, est_end, "full"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Confounding 이벤트 감지
# ─────────────────────────────────────────────────────────────────────────────

def build_event_date_map(events: List[Event]) -> Dict[str, List[Tuple[str, np.datetime64]]]:
    """
    기업별 (event_id, T0_근사날짜) 목록 (confounding 감지용)
    T0_근사 = 충격월 M+1 의 1일 (월 첫날이 비거래일이면 ±5일 범위로 검색)
    """
    result: Dict[str, List[Tuple[str, np.datetime64]]] = defaultdict(list)
    for ev in events:
        try:
            t0_approx = np.datetime64(
                (pd.Period(ev.shock_month, "M") + 1).to_timestamp().date(), "D"
            )
            result[ev.shock_company].append((ev.event_id, t0_approx))
        except Exception:
            pass
    return dict(result)


def has_confounding(
    company_id: str,
    t0_idx: int,
    win_start: int,
    win_end: int,
    event_date_map: Dict[str, List[Tuple[str, np.datetime64]]],
    cd: CompanyData,
    current_event_id: str,
) -> bool:
    """
    target company_id 의 이벤트 창 [t0+win_start, t0+win_end] 내에
    현재 이벤트와 다른 이벤트가 존재하는지 확인.

    T0_근사는 월 1일 기준이므로 월 1일이 비거래일일 경우를 대비해
    window_dates 를 ±5일 패딩하여 검색.
    """
    ev_pairs = event_date_map.get(company_id, [])
    if not ev_pairs:
        return False

    evt_s = t0_idx + win_start
    evt_e = t0_idx + win_end + 1  # 슬라이스 exclusive
    # 배열 경계 클램핑
    s_clamped = max(0, evt_s)
    e_clamped = min(len(cd.dates), evt_e)
    if s_clamped >= e_clamped:
        return False

    # 창 내 실제 거래일 집합 (±5일 패딩으로 월 1일 비거래일 커버)
    pad = 5
    s_pad = max(0, s_clamped - pad)
    e_pad = min(len(cd.dates), e_clamped + pad)
    window_dates = set(cd.dates[s_pad:e_pad])

    for (eid, ed) in ev_pairs:
        if eid == current_event_id:
            continue  # 현재 이벤트 자신은 제외
        if ed in window_dates:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# 6. 메인 CAR 계산 루프
# ─────────────────────────────────────────────────────────────────────────────

def run_car_study(
    events: List[Event],
    company_data: Dict[str, CompanyData],
    adjacency: Dict[str, List[dict]],
    event_date_map: Dict[str, List[np.datetime64]],
) -> List[CARResult]:
    """
    전체 이벤트 × (직접 + contagion) × 창 루프
    """
    results: List[CARResult] = []
    n_skip_nodata   = 0
    n_skip_insuff   = 0
    n_skip_listdate = 0
    n_computed      = 0

    for ev in events:
        # ── 충격 기업의 T=0 결정 ──
        cd_shock = company_data.get(ev.shock_company)
        if cd_shock is None:
            log.debug("  가격 데이터 없음: %s", ev.shock_company)
            n_skip_nodata += 1
            continue

        t0_idx_shock = find_t0_idx(cd_shock, ev.shock_month)
        if t0_idx_shock is None:
            log.debug("  T=0 결정 불가: %s %s", ev.shock_company, ev.shock_month)
            n_skip_listdate += 1
            continue

        t0_date_shock = str(cd_shock.dates[t0_idx_shock])

        # ── 대상 기업 목록 구성 ──
        # (1) 직접: 충격 기업 자신
        targets: List[dict] = [{"company_id": ev.shock_company,
                                 "relationship": "DIRECT",
                                 "rel_type": "SELF",
                                 "weight": 1.0,
                                 "hops": 0}]
        # (2) contagion: 인접 기업 (1홉 + 2홉, 중복 제거)
        seen_neighbors: set = {ev.shock_company}
        for entry in adjacency.get(ev.shock_company, []):
            nb = entry["neighbor"]
            if nb in seen_neighbors:
                continue
            seen_neighbors.add(nb)
            rel_label = f"{entry['direction']}_{entry['hops']}"
            targets.append({
                "company_id": nb,
                "relationship": rel_label,
                "rel_type": entry["rel_type"],
                "weight": entry["weight"],
                "hops": entry["hops"],
            })

        # ── 각 대상 기업 × 이벤트 창 CAR 계산 ──
        for tgt in targets:
            cid = tgt["company_id"]
            cd  = company_data.get(cid)
            if cd is None:
                n_skip_nodata += 1
                continue

            # 대상 기업의 T=0 (DIRECT면 충격 기업과 동일 인덱스 의미를 공유하되,
            # 실제 날짜는 각 기업의 거래일 기준으로 독립 탐색)
            t0_idx = find_t0_idx(cd, ev.shock_month)
            if t0_idx is None:
                n_skip_listdate += 1
                continue

            est_start, est_end, quality = get_estimation_indices(cd, t0_idx)
            if quality == "insufficient":
                n_skip_insuff += 1
                continue

            t0_date = str(cd.dates[t0_idx])

            for (win_s, win_e, win_label) in EVENT_WINDOWS:
                evt_s = t0_idx + win_s
                evt_e = t0_idx + win_e + 1  # +1: 슬라이스는 exclusive

                # 경계 확인
                if evt_s < 0 or evt_e > len(cd.dates):
                    continue

                res = ols_car(cd.ret, cd.mkt_ret,
                              est_start, est_end, evt_s, evt_e)
                if res is None:
                    continue

                # Confounding 감지
                confounding_self = has_confounding(
                    ev.shock_company, t0_idx_shock, win_s, win_e + 1,
                    event_date_map, cd_shock, ev.event_id
                )
                confounding_target = (
                    has_confounding(
                        cid, t0_idx, win_s, win_e + 1,
                        event_date_map, cd, ev.event_id
                    )
                    if cid != ev.shock_company else confounding_self
                )

                results.append(CARResult(
                    event_id=ev.event_id,
                    shock_company=ev.shock_company,
                    shock_month=ev.shock_month,
                    shock_type=ev.shock_type,
                    z_score=ev.z_score,
                    target_company=cid,
                    relationship=tgt["relationship"],
                    rel_type=tgt["rel_type"],
                    window_label=win_label,
                    win_start=win_s,
                    win_end=win_e,
                    t0_date=t0_date,
                    CAR=res["CAR"],
                    t_stat_patell=res["t_stat_patell"],
                    p_patell=res["p_patell"],
                    alpha=res["alpha"],
                    beta=res["beta"],
                    sigma_ar=res["sigma_ar"],
                    est_len=res["est_len"],
                    quality=quality,
                    confounding_self=confounding_self,
                    confounding_target=confounding_target,
                ))
                n_computed += 1

        if n_computed % 500 == 0 and n_computed > 0:
            log.info("  진행: %d CAR 계산됨 ...", n_computed)

    log.info("CAR 계산 완료: %d건", n_computed)
    log.info("  스킵 (가격 없음=%d, 추정 부족=%d, 상장 전=%d)",
             n_skip_nodata, n_skip_insuff, n_skip_listdate)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 7. 통계 집계 함수
# ─────────────────────────────────────────────────────────────────────────────

def caar_stats(car_values: np.ndarray, label: str = "") -> dict:
    """
    N개 CAR 값으로 CAAR 통계 계산.

    - t_cross: 교차단면 t-검정 (CAAR / SE_cross)
    - t_patell_agg: Patell aggregate (sum of standardized residuals / sqrt(N))
    - sign_test_p: 비모수 부호 검정 (Binomial, two-tailed)
    - pct_negative: 음의 CAR 비율
    """
    n = len(car_values)
    if n == 0:
        return {"N": 0, "CAAR": np.nan, "std": np.nan,
                "t_cross": np.nan, "p_cross": np.nan,
                "sign_test_p": np.nan, "pct_negative": np.nan}

    caar = car_values.mean()
    std  = car_values.std(ddof=1) if n > 1 else np.nan

    # 교차단면 t-검정
    if n > 1 and std > 1e-15:
        se_cross = std / np.sqrt(n)
        t_cross  = caar / se_cross
        p_cross  = float(2.0 * stats.t.sf(abs(t_cross), df=n - 1))
    else:
        t_cross, p_cross = np.nan, np.nan

    # 부호 검정 (two-tailed Binomial)
    n_neg = int((car_values < 0).sum())
    p_sign = float(2 * min(
        stats.binom.cdf(n_neg, n, 0.5),
        1 - stats.binom.cdf(n_neg - 1, n, 0.5)
    ))

    pct_neg = n_neg / n if n > 0 else np.nan

    return {
        "N":            n,
        "CAAR":         float(caar),
        "std":          float(std) if not np.isnan(std) else np.nan,
        "t_cross":      float(t_cross) if not np.isnan(t_cross) else np.nan,
        "p_cross":      float(p_cross) if not np.isnan(p_cross) else np.nan,
        "sign_test_p":  float(p_sign),
        "pct_negative": float(pct_neg),
    }


def summarize_group(
    df: pd.DataFrame,
    group_cols: List[str],
    window: str = "0_21",
    quality_filter: bool = True,
    confounding_filter: bool = True,
) -> pd.DataFrame:
    """
    그룹별 CAAR 통계 집계.

    Parameters
    ----------
    quality_filter     : True면 quality='full'만 사용
    confounding_filter : True면 confounding_self=False만 사용
    """
    sub = df[df["window_label"] == window].copy()

    if quality_filter:
        sub = sub[sub["quality"] == "full"]
    if confounding_filter:
        sub = sub[~sub["confounding_self"]]

    rows = []
    for keys, g in sub.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        stat = caar_stats(g["CAR"].values)
        row = dict(zip(group_cols, keys))
        row.update(stat)
        rows.append(row)

    return pd.DataFrame(rows).sort_values("CAAR" if "CAAR" in (rows[0] if rows else {}) else group_cols[0])


# ─────────────────────────────────────────────────────────────────────────────
# 8. 이질성 & Contagion 분석
# ─────────────────────────────────────────────────────────────────────────────

def run_heterogeneity(
    panel: pd.DataFrame,
    c18: pd.DataFrame,
) -> pd.DataFrame:
    """
    DIRECT 이벤트 기준 이질성 분석

    차원:
    1. 충격 유형 (NEG vs POS)
    2. 충격 기업 공급망 단계 (stage_bucket)
    3. 레짐 기간
    4. 전체 (pooled)
    """
    # stage_bucket 조인
    stage_map = dict(zip(c18["company_id"], c18["stage_bucket"]))
    panel = panel.copy()
    panel["shock_stage"] = panel["shock_company"].map(stage_map)

    # DIRECT만, 주 창(0_21), full quality, non-confounding
    direct = panel[
        (panel["relationship"] == "DIRECT") &
        (panel["window_label"] == "0_21") &
        (panel["quality"] == "full") &
        (~panel["confounding_self"])
    ].copy()

    results = []

    # (A) 전체 pooled
    stat = caar_stats(direct["CAR"].values)
    results.append({"analysis": "pooled", "group": "ALL", **stat})

    # (B) 충격 유형
    for stype, g in direct.groupby("shock_type"):
        stat = caar_stats(g["CAR"].values)
        results.append({"analysis": "shock_type", "group": stype, **stat})

    # (C) 공급망 단계
    for stage, g in direct.groupby("shock_stage"):
        stat = caar_stats(g["CAR"].values)
        results.append({"analysis": "stage_bucket", "group": str(stage), **stat})

    # (D) 레짐 기간
    direct["year"] = pd.to_datetime(direct["t0_date"]).dt.year
    direct["regime"] = pd.cut(
        direct["year"],
        bins=[0, 2019, 2021, 2023, 9999],
        labels=["pre_covid(≤2019)", "covid(2020-21)", "post_covid(2022-23)", "recent(≥2024)"],
    )
    for regime, g in direct.groupby("regime", observed=True):
        if len(g) == 0:
            continue
        stat = caar_stats(g["CAR"].values)
        results.append({"analysis": "regime", "group": str(regime), **stat})

    # (E) NEG 충격 유형 × 공급망 단계 (교차)
    for (stype, stage), g in direct.groupby(["shock_type", "shock_stage"]):
        if len(g) < 3:
            continue
        stat = caar_stats(g["CAR"].values)
        results.append({
            "analysis": "shock_type_x_stage",
            "group": f"{stype}|{stage}", **stat
        })

    # (F) 이벤트 창 비교 (pooled)
    for win, g in panel[
        (panel["relationship"] == "DIRECT") &
        (panel["quality"] == "full") &
        (~panel["confounding_self"])
    ].groupby("window_label"):
        stat = caar_stats(g["CAR"].values)
        results.append({"analysis": "event_window", "group": win, **stat})

    return pd.DataFrame(results)


def run_contagion_analysis(
    panel: pd.DataFrame,
    c18: pd.DataFrame,
    window: str = "0_21",
) -> pd.DataFrame:
    """
    공급망 전파(contagion) 분석

    비교:
    - DIRECT vs DOWNSTREAM_1 vs UPSTREAM_1 vs PEER_1 vs *_2
    - NEG 충격 시 전파 방향별 CAAR
    """
    stage_map = dict(zip(c18["company_id"], c18["stage_bucket"]))
    panel = panel.copy()
    panel["target_stage"] = panel["target_company"].map(stage_map)

    sub = panel[
        (panel["window_label"] == window) &
        (panel["quality"] == "full") &
        (~panel["confounding_target"])
    ].copy()

    results = []

    # (A) 관계 유형별 (DIRECT / DOWNSTREAM_1 / UPSTREAM_1 / PEER_1 / *_2)
    def normalize_rel(r: str) -> str:
        if r == "DIRECT":
            return "DIRECT"
        hops = r.split("_")[-1]
        direction = "_".join(r.split("_")[:-1])
        if hops in ("1", "2"):
            return f"{direction} (hop{hops})"
        return r

    sub["rel_group"] = sub["relationship"].apply(normalize_rel)

    for rg, g in sub.groupby("rel_group"):
        stat = caar_stats(g["CAR"].values)
        results.append({"dimension": "relationship", "group": str(rg), **stat})

    # (B) NEG 충격만 → 관계별 CAAR
    neg = sub[sub["shock_type"] == "NEG"]
    for rg, g in neg.groupby("rel_group"):
        stat = caar_stats(g["CAR"].values)
        results.append({"dimension": "neg_shock_relationship", "group": str(rg), **stat})

    # (C) DOWNSTREAM_1 의 대상 기업 공급망 단계별
    ds1 = sub[sub["relationship"] == "DOWNSTREAM_1"]
    for stage, g in ds1.groupby("target_stage"):
        if len(g) < 3:
            continue
        stat = caar_stats(g["CAR"].values)
        results.append({"dimension": "downstream1_target_stage", "group": str(stage), **stat})

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# 9. 보고서 생성
# ─────────────────────────────────────────────────────────────────────────────

def format_stat_row(row: dict) -> str:
    def fmt(v):
        if isinstance(v, float):
            if np.isnan(v):
                return "   N/A"
            return f"{v:+7.4f}"
        return str(v)
    n     = int(row.get("N", 0))
    caar  = fmt(row.get("CAAR", np.nan))
    t_c   = fmt(row.get("t_cross", np.nan))
    p_c   = fmt(row.get("p_cross", np.nan))
    psign = fmt(row.get("sign_test_p", np.nan))
    pneg  = fmt(row.get("pct_negative", np.nan))
    return (f"N={n:3d}  CAAR={caar}  t={t_c}  p_cross={p_c}  "
            f"p_sign={psign}  pct_neg={pneg}")


def build_report(
    events: List[Event],
    panel: pd.DataFrame,
    hetero: pd.DataFrame,
    contagion: pd.DataFrame,
    company_data: Dict[str, CompanyData],
) -> str:
    lines = []
    sep = "=" * 76

    lines += [sep, "나비효과 Phase 4 — CAR Event Study 결과 보고서", sep, ""]

    # ── 데이터 요약 ──
    lines.append("■ 이벤트 및 데이터 요약")
    lines.append(f"  총 이벤트(raw):         {len(events)}건")
    direct_full = panel[
        (panel["relationship"] == "DIRECT") &
        (panel["quality"] == "full") &
        (panel["window_label"] == "0_21")
    ]
    direct_all_win = panel[(panel["relationship"] == "DIRECT")]
    lines.append(f"  유효 direct 이벤트:     {direct_full['event_id'].nunique()}건 "
                 f"(quality=full, window=0_21, non-confounding 전)")
    lines.append(f"  전체 CAR 관측치:        {len(panel)}건 "
                 f"(모든 기업×창 조합)")
    lines.append(f"  충격 유형: NEG={sum(1 for e in events if e.shock_type=='NEG')}, "
                 f"POS={sum(1 for e in events if e.shock_type=='POS')}")
    lines.append("")

    # ── 창별 CAAR (DIRECT, full, non-confounding) ──
    lines.append("■ 이벤트 창별 CAAR (DIRECT 효과, quality=full, confounding 제외)")
    lines.append(f"  {'Window':<10} {'N':>4}  {'CAAR':>8}  {'t_cross':>8}  "
                 f"{'p_cross':>8}  {'p_sign':>8}  {'%neg':>6}")
    lines.append("  " + "-" * 68)

    win_order = ["m1_1", "0_1", "0_5", "0_10", "0_21", "m5_5"]
    win_stats = (
        hetero[hetero["analysis"] == "event_window"]
        .set_index("group")
        .reindex(win_order)
        .dropna(how="all")
    )
    for win, row in win_stats.iterrows():
        n    = int(row.get("N", 0))
        caar = row.get("CAAR", np.nan)
        tc   = row.get("t_cross", np.nan)
        pc   = row.get("p_cross", np.nan)
        ps   = row.get("sign_test_p", np.nan)
        pn   = row.get("pct_negative", np.nan)
        sig  = ("***" if (not np.isnan(pc) and pc < 0.01) else
                "**"  if (not np.isnan(pc) and pc < 0.05) else
                "*"   if (not np.isnan(pc) and pc < 0.10) else "")
        lines.append(
            f"  [{win:<8}]  {n:3d}  {caar:+8.4f}  {tc:+8.3f}  {pc:8.4f}  "
            f"{ps:8.4f}  {pn:6.2%}  {sig}"
        )
    lines.append("")

    # ── 이질성 분석 ──
    lines.append("■ 이질성 분석 (주 창 [0,+21], DIRECT, full, non-confounding)")
    for analysis in ["shock_type", "stage_bucket", "regime"]:
        sub = hetero[hetero["analysis"] == analysis]
        if sub.empty:
            continue
        lines.append(f"\n  [{analysis}]")
        lines.append(f"  {'Group':<30}  {'N':>4}  {'CAAR':>8}  {'t_cross':>8}  {'p_cross':>8}")
        lines.append("  " + "-" * 62)
        for _, row in sub.iterrows():
            n  = int(row.get("N", 0))
            if n == 0:
                continue
            g  = str(row["group"])[:28]
            c  = row.get("CAAR", np.nan)
            t  = row.get("t_cross", np.nan)
            p  = row.get("p_cross", np.nan)
            sig = ("***" if (not np.isnan(p) and p < 0.01) else
                   "**"  if (not np.isnan(p) and p < 0.05) else
                   "*"   if (not np.isnan(p) and p < 0.10) else "")
            lines.append(
                f"  {g:<30}  {n:4d}  {c:+8.4f}  "
                f"{t:+8.3f}  {p:8.4f}  {sig}"
            )
    lines.append("")

    # ── 공급망 전파 분석 ──
    lines.append("■ 공급망 전파 (Contagion) 분석 (창 [0,+21], quality=full)")
    lines.append(f"  {'Relationship':<35}  {'N':>4}  {'CAAR':>8}  "
                 f"{'t_cross':>8}  {'p_cross':>8}")
    lines.append("  " + "-" * 68)
    for dim in ["relationship", "neg_shock_relationship"]:
        sub = contagion[contagion["dimension"] == dim]
        if sub.empty:
            continue
        lines.append(f"\n  [dim={dim}]")
        for _, row in sub.iterrows():
            n  = int(row.get("N", 0))
            if n == 0:
                continue
            g  = str(row["group"])[:33]
            c  = row.get("CAAR", np.nan)
            t  = row.get("t_cross", np.nan)
            p  = row.get("p_cross", np.nan)
            sig = ("***" if (not np.isnan(p) and p < 0.01) else
                   "**"  if (not np.isnan(p) and p < 0.05) else
                   "*"   if (not np.isnan(p) and p < 0.10) else "")
            lines.append(
                f"  {g:<35}  {n:4d}  {c:+8.4f}  "
                f"{t:+8.3f}  {p:8.4f}  {sig}"
            )
    lines.append("")

    # ── 품질 플래그 요약 ──
    lines.append("■ 품질 플래그 요약 (모든 창 합산, DIRECT)")
    d_all = panel[panel["relationship"] == "DIRECT"]
    for q in ["full", "marginal", "insufficient"]:
        n = len(d_all[d_all["quality"] == q])
        lines.append(f"  quality={q:<13}: {n}건")
    n_conf = d_all["confounding_self"].sum()
    lines.append(f"  confounding_self=True  : {n_conf}건 (분석에서 제외됨)")
    lines.append("")

    lines += [sep, "주: *** p<0.01, ** p<0.05, * p<0.10 (two-tailed)", sep]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 10. 메인
# ─────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    # 출력 디렉토리 생성
    OUT_PANEL.parent.mkdir(parents=True, exist_ok=True)
    OUT_CAAR.parent.mkdir(parents=True, exist_ok=True)

    log.info("Phase 4 CAR Event Study 시작")
    log.info("T=0 정의: 충격월 M+1 의 첫 거래일 (월내 선행편의 방지)")
    log.info("추정기간: [T-%d, T-%d] (최소 %dtd, 권장 %d+td)",
             abs(EST_WIN_START), abs(EST_WIN_END), MIN_EST_DAYS, PREFERRED_EST)

    # ── 데이터 로딩 ──
    events = load_events()
    prices, index_returns, company_to_index = load_prices_and_market()
    edges = load_internal_edges()
    c18   = pd.read_csv(PATH_C18)

    target_companies = list(c18["company_id"].unique())

    # ── CompanyData 구축 ──
    company_data = build_company_data(
        prices, index_returns, company_to_index, target_companies
    )

    # ── KG adjacency ──
    adjacency = build_adjacency(edges)
    log.info("KG adjacency: %d기업, 평균 %.1f 연결",
             len(adjacency),
             np.mean([len(v) for v in adjacency.values()]) if adjacency else 0)

    # ── 이벤트 날짜 맵 (confounding 감지) ──
    event_date_map = build_event_date_map(events)

    # ── CAR 계산 ──
    log.info("CAR 계산 시작 (이벤트×기업×창 조합)...")
    car_results = run_car_study(events, company_data, adjacency, event_date_map)

    if not car_results:
        log.error("CAR 결과가 없습니다. 입력 데이터를 확인하세요.")
        sys.exit(1)

    # ── DataFrame 변환 ──
    panel = pd.DataFrame([vars(r) for r in car_results])
    log.info("패널 크기: %d행", len(panel))

    # ── 통계 분석 ──
    log.info("이질성 분석 실행...")
    hetero = run_heterogeneity(panel, c18)

    log.info("Contagion 분석 실행...")
    contagion = run_contagion_analysis(panel, c18, window="0_21")

    # ── CAAR 요약 (주 창 [0,+21]) ──
    caar_summary = summarize_group(
        panel, ["relationship", "shock_type"],
        window="0_21", quality_filter=True, confounding_filter=True
    )

    # ── 보고서 생성 ──
    report = build_report(events, panel, hetero, contagion, company_data)

    # ── 저장 ──
    log.info("저장 중...")
    panel.to_parquet(OUT_PANEL, index=False)
    caar_summary.to_csv(OUT_CAAR, index=False, encoding="utf-8-sig")
    hetero.to_csv(OUT_HETERO, index=False, encoding="utf-8-sig")
    contagion.to_csv(OUT_CONTAGION, index=False, encoding="utf-8-sig")
    OUT_REPORT.write_text(report, encoding="utf-8")

    log.info("저장 완료:")
    log.info("  %s", OUT_PANEL)
    log.info("  %s", OUT_CAAR)
    log.info("  %s", OUT_HETERO)
    log.info("  %s", OUT_CONTAGION)
    log.info("  %s", OUT_REPORT)

    # 콘솔 출력
    print("\n" + report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4: CAR Event Study")
    parser.add_argument(
        "--no-quality-filter", action="store_true",
        help="marginal 품질 이벤트도 분석에 포함"
    )
    parser.add_argument(
        "--no-confounding-filter", action="store_true",
        help="confounding 이벤트도 분석에 포함"
    )
    args = parser.parse_args()
    main(args)
