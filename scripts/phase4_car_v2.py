"""
Phase 4 CAR — z-score v2 재분석
================================
v1(기존) / v2(strict) / supply_z(공급망 특화) 세 가지 이벤트 정의로
CAR을 비교해서 어느 신호가 공급망 전파를 가장 잘 잡는지 확인

핵심 질문:
  "supply_z 이벤트로 보면 Upstream(ALB·GLEN·Huayou) 충격이
   Downstream(Cell·OEM)에 유의미하게 전달되는가?"
"""
from __future__ import annotations
import logging, sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yaml
from scipy import stats

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]

PATH_TONE_V2  = ROOT / "data/processed/tone_monthly_zscore_18_v2.parquet"
PATH_PRICES   = ROOT / "data/processed/prices_daily.parquet"
PATH_MKT      = ROOT / "data/processed/market_proxies.parquet"
PATH_MKT_MAP  = ROOT / "data/processed/market_proxy_mapping.yaml"
PATH_EDGES    = ROOT / "data/seed/seed_edges_18_internal.csv"
PATH_C18      = ROOT / "data/universe/companies_18.csv"

OUT_DIR = ROOT / "reports"
OUT_DIR.mkdir(exist_ok=True)

EST_WIN_START = -250
EST_WIN_END   = -20
MIN_EST_DAYS  = 60
PREFERRED_EST = 120
SHOCK_THR     = 2.0

EVENT_WINDOWS = [
    (0,  1,  "0_1"),
    (0,  5,  "0_5"),
    (0,  10, "0_10"),
    (0,  21, "0_21"),
    (-1, 1,  "m1_1"),
]


# ─── 이벤트 로드 (세 가지 버전) ───────────────────────────────

def load_events_v2(tone: pd.DataFrame, shock_col_neg: str, shock_col_pos: str,
                   reliable_col: str) -> List[dict]:
    mask = tone[reliable_col] & (tone[shock_col_neg] | tone[shock_col_pos])
    rows = []
    seen = set()
    for _, r in tone[mask].iterrows():
        eid = f"{r['company_id']}_{r['year_month']}"
        if eid in seen: continue
        seen.add(eid)
        z_col = "z_score_v2" if "z_score_v2" in tone.columns else "z_score"
        if "supply" in shock_col_neg:
            z_col = "supply_z"
        rows.append(dict(
            event_id    = eid,
            shock_company = r["company_id"],
            shock_month = r["year_month"],
            shock_type  = "NEG" if r[shock_col_neg] else "POS",
            z_score     = float(r.get(z_col, 0) or 0),
            version     = "supply" if "supply" in shock_col_neg else "v2",
        ))
    return rows


# ─── 가격·시장 데이터 (phase4와 동일 로직) ──────────────────

def build_company_data(company_ids):
    prices = pd.read_parquet(PATH_PRICES)
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices[prices["company_id"].isin(company_ids)].sort_values(["company_id","date"])
    prices["price"] = prices["adj_close"].fillna(prices["close"])

    mkt_df = pd.read_parquet(PATH_MKT)
    mkt_df["date"] = pd.to_datetime(mkt_df["date"])
    idx_col = "index_id" if "index_id" in mkt_df.columns else "index_name"
    mkt_df = mkt_df.sort_values([idx_col, "date"])
    mkt_df["mkt_ret"] = mkt_df.groupby(idx_col)["adj_close"].pct_change()
    mkt_df = mkt_df.dropna(subset=["mkt_ret"])
    index_returns = {}
    for iname, g in mkt_df.groupby(idx_col):
        s = g.set_index("date")["mkt_ret"]; s.name = "mkt_ret"
        index_returns[str(iname)] = s

    with open(PATH_MKT_MAP) as f:
        mapping = yaml.safe_load(f)
    c2i = mapping.get("company_to_market_proxy", {})

    ew = (prices.assign(ret=lambda d: d.groupby("company_id")["price"].transform(
              lambda s: s.pct_change()))
          .dropna(subset=["ret"]).groupby("date")["ret"].mean().rename("mkt_ret"))

    company_data = {}
    for cid, g in prices.groupby("company_id"):
        g = g.sort_values("date").copy()
        g["ret"] = g["price"].pct_change()
        g = g.dropna(subset=["ret"])
        if len(g) < 30: continue
        mkt_s = index_returns.get(c2i.get(cid, ""), ew)
        df = g.set_index("date")[["ret"]].join(mkt_s.to_frame(), how="inner").dropna()
        if len(df) < 30: continue
        dates = df.index.values.astype("datetime64[D]")
        company_data[cid] = dict(
            dates=dates, ret=df["ret"].values.astype(np.float64),
            mkt=df["mkt_ret"].values.astype(np.float64),
            d2i={d: i for i, d in enumerate(dates)}
        )
    return company_data


def ols_car(ret, mkt, es, ee, xs, xe):
    n = ee - es
    if n < MIN_EST_DAYS or es < 0 or xe > len(ret) or xs < 0: return None
    er, em = ret[es:ee], mkt[es:ee]
    sm, sr = em.sum(), er.sum(); smm = (em*em).sum(); smr = (em*er).sum()
    denom = n*smm - sm*sm
    if abs(denom) < 1e-15: return None
    beta  = (n*smr - sm*sr) / denom
    alpha = (sr - beta*sm) / n
    resid = er - (alpha + beta*em)
    sigma = float(np.sqrt(np.sum(resid**2) / max(1, n-2)))
    if sigma < 1e-10: return None
    vr, vm = ret[xs:xe], mkt[xs:xe]
    el = len(vr)
    if el < max(1, int((xe-xs)*0.8)): return None
    car = float((vr - (alpha + beta*vm)).sum())
    mbar = em.mean(); S2m = np.sum((em-mbar)**2)
    corr = 1 + 1/n + ((vm.mean()-mbar)**2 / S2m if S2m > 1e-15 else 0)
    denom_p = sigma * np.sqrt(el * corr)
    if denom_p < 1e-15: return None
    t = car / denom_p
    p = float(2.0 * stats.t.sf(abs(t), df=max(1, n-2)))
    q = "full" if n >= PREFERRED_EST else "marginal"
    return dict(CAR=car, t=t, p=p, quality=q, beta=float(beta))


def find_t0(cd, shock_month):
    try:
        t0 = np.datetime64((pd.Period(shock_month,"M")+1).to_timestamp().date(), "D")
    except: return None
    pos = np.searchsorted(cd["dates"], t0, side="left")
    return int(pos) if pos < len(cd["dates"]) else None


# ─── adjacency ────────────────────────────────────────────────

def build_adjacency(edges_df, company_ids):
    from collections import defaultdict
    adj = defaultdict(list)
    ci = set(company_ids)
    for _, r in edges_df.iterrows():
        s, d = r["src_company_id"], r["dst_company_id"]
        if s not in ci or d not in ci: continue
        w = float(r.get("confidence_plink",0.8))*float(r.get("strength",0.8))
        rt = r["rel_type"]
        if rt in ("SUPPLIES","BUYS_FROM","SOURCES_FROM","OWNS"):
            adj[s].append(dict(nb=d, dir="DOWNSTREAM_1", rt=rt, w=w))
            adj[d].append(dict(nb=s, dir="UPSTREAM_1",   rt=rt, w=w))
        elif rt in ("COMPETES_WITH","PARTNERS_WITH"):
            adj[s].append(dict(nb=d, dir="PEER_1", rt=rt, w=w))
            adj[d].append(dict(nb=s, dir="PEER_1", rt=rt, w=w))
    return dict(adj)


# ─── CAR 루프 ─────────────────────────────────────────────────

def run_car(events, company_data, adjacency):
    rows = []
    for ev in events:
        cd_sh = company_data.get(ev["shock_company"])
        if cd_sh is None: continue
        t0_sh = find_t0(cd_sh, ev["shock_month"])
        if t0_sh is None: continue

        # 대상 기업 목록: 직접 + 1홉
        targets = [dict(cid=ev["shock_company"], rel="DIRECT", rt="SELF")]
        seen = {ev["shock_company"]}
        for entry in adjacency.get(ev["shock_company"], []):
            if entry["nb"] not in seen:
                seen.add(entry["nb"])
                targets.append(dict(cid=entry["nb"], rel=entry["dir"], rt=entry["rt"]))

        for tgt in targets:
            cd = company_data.get(tgt["cid"])
            if cd is None: continue
            t0 = find_t0(cd, ev["shock_month"])
            if t0 is None: continue
            es = max(0, t0 + EST_WIN_START)
            ee = t0 + EST_WIN_END
            if ee - es < MIN_EST_DAYS: continue

            for ws, we, wl in EVENT_WINDOWS:
                xs, xe = t0+ws, t0+we+1
                res = ols_car(cd["ret"], cd["mkt"], es, ee, xs, xe)
                if res is None: continue
                rows.append(dict(
                    version       = ev["version"],
                    event_id      = ev["event_id"],
                    shock_company = ev["shock_company"],
                    shock_month   = ev["shock_month"],
                    shock_type    = ev["shock_type"],
                    z_score       = ev["z_score"],
                    target_company= tgt["cid"],
                    relationship  = tgt["rel"],
                    rel_type      = tgt["rt"],
                    window        = wl,
                    CAR           = res["CAR"],
                    t_stat        = res["t"],
                    p_val         = res["p"],
                    quality       = res["quality"],
                    beta          = res["beta"],
                ))
    return pd.DataFrame(rows)


# ─── 통계 집계 ────────────────────────────────────────────────

def caar_stats(vals):
    n = len(vals)
    if n == 0: return dict(N=0, CAAR=np.nan, t=np.nan, p=np.nan, p_sign=np.nan)
    caar = vals.mean()
    std  = vals.std(ddof=1) if n > 1 else np.nan
    t, p = (np.nan, np.nan)
    if n > 1 and std > 1e-15:
        t = caar / (std / np.sqrt(n))
        p = float(2 * stats.t.sf(abs(t), df=n-1))
    nneg  = int((vals < 0).sum())
    p_sign = float(2 * min(stats.binom.cdf(nneg,n,0.5),
                           1-stats.binom.cdf(nneg-1,n,0.5)))
    return dict(N=n, CAAR=float(caar), t=float(t) if not np.isnan(t) else np.nan,
                p=float(p) if not np.isnan(p) else np.nan, p_sign=float(p_sign))


def summarize(panel: pd.DataFrame, version: str, window: str = "0_21") -> pd.DataFrame:
    sub = panel[(panel["version"]==version) &
                (panel["window"]==window) &
                (panel["quality"]=="full")]
    rows = []
    for rel, g in sub.groupby("relationship"):
        s = caar_stats(g["CAR"].values)
        rows.append({"relationship": rel, **s})
    return pd.DataFrame(rows).sort_values("CAAR")


# ─── 메인 ─────────────────────────────────────────────────────

def main():
    log.info("Phase 4 CAR v2 분석 시작")
    tone = pd.read_parquet(PATH_TONE_V2)

    c18     = pd.read_csv(PATH_C18)
    nm      = dict(zip(c18["company_id"], c18["canonical_name"]))
    stage   = dict(zip(c18["company_id"], c18["stage_bucket"]))
    company_ids = c18["company_id"].tolist()
    edges   = pd.read_csv(PATH_EDGES)
    adj     = build_adjacency(edges, company_ids)
    cd_map  = build_company_data(company_ids)
    log.info("CompanyData: %d기업 로드", len(cd_map))

    # 세 가지 이벤트 셋
    ev_v2     = load_events_v2(tone, "neg_shock_v2",    "pos_shock_v2",    "reliable_v2")
    ev_supply = load_events_v2(tone, "neg_shock_supply","pos_shock_supply", "reliable_v2")
    # v1 (비교용) — 원본 컬럼 사용
    ev_v1     = load_events_v2(tone, "neg_shock",       "pos_shock",       "reliable")
    for e in ev_v1: e["version"] = "v1"

    log.info("이벤트 수: v1=%d, v2=%d, supply=%d",
             len(ev_v1), len(ev_v2), len(ev_supply))

    all_events = ev_v1 + ev_v2 + ev_supply
    log.info("CAR 계산 시작...")
    panel = run_car(all_events, cd_map, adj)
    log.info("CAR 패널: %d행", len(panel))

    # 저장
    panel.to_parquet(ROOT/"data/processed/car_panel_v2_comparison.parquet", index=False)

    # ── 결과 출력 ──────────────────────────────────────────────
    SEP = "=" * 72
    print(f"\n{SEP}")
    print("Phase 4 CAR — v1 / v2 / supply_z 비교 결과 (창 [0,+21])")
    print(SEP)

    # 이벤트 수 요약
    for ver, evs in [("v1", ev_v1), ("v2", ev_v2), ("supply", ev_supply)]:
        neg = sum(1 for e in evs if e["shock_type"]=="NEG")
        pos = sum(1 for e in evs if e["shock_type"]=="POS")
        print(f"  {ver:8s}: 총 {len(evs):>3}건  (NEG={neg}, POS={pos})")
    print()

    # 버전별 관계 유형 CAAR 비교 표
    print(f"{'관계':22s}  {'v1 CAAR':>8}  {'v1 p':>6}  "
          f"{'v2 CAAR':>8}  {'v2 p':>6}  "
          f"{'supply CAAR':>11}  {'p':>6}  {'N':>4}")
    print("-" * 80)

    rels_order = ["DIRECT","DOWNSTREAM_1","UPSTREAM_1","PEER_1"]
    stats_all = {}
    for ver in ["v1","v2","supply"]:
        sub = panel[(panel["version"]==ver) &
                    (panel["window"]=="0_21") &
                    (panel["quality"]=="full")]
        stats_all[ver] = {}
        for rel in rels_order:
            g = sub[sub["relationship"]==rel]["CAR"].values
            stats_all[ver][rel] = caar_stats(g)

    for rel in rels_order:
        v1s = stats_all["v1"].get(rel, {})
        v2s = stats_all["v2"].get(rel, {})
        ss  = stats_all["supply"].get(rel, {})
        def fmt(s, k): return f"{s.get(k,np.nan):+7.3f}" if s else "    N/A"
        def fmtp(s):
            p = s.get("p", np.nan)
            if np.isnan(p): return "   N/A"
            sig = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.10 else ""
            return f"{p:.3f}{sig:3s}"
        n_sup = ss.get("N",0)
        print(f"  {rel:20s}  {fmt(v1s,'CAAR'):>8}  {fmtp(v1s):>9}  "
              f"{fmt(v2s,'CAAR'):>8}  {fmtp(v2s):>9}  "
              f"{fmt(ss,'CAAR'):>11}  {fmtp(ss):>9}  {n_sup:>4}")

    print()
    # ── supply_z 심층 분석: Upstream 충격 → Downstream 반응 ──
    print("■ supply_z 핵심 분석: Upstream 충격 → 공급망 단계별 CAR")
    print()
    sup_neg = panel[
        (panel["version"]=="supply") &
        (panel["shock_type"]=="NEG") &
        (panel["window"]=="0_21") &
        (panel["quality"]=="full")
    ].copy()
    sup_neg["shock_stage"]  = sup_neg["shock_company"].map(stage)
    sup_neg["target_stage"] = sup_neg["target_company"].map(stage)

    # Upstream 기업이 충격 → 다른 레이어 반응
    upstream_shock = sup_neg[sup_neg["shock_stage"]=="Upstream/Refining"]
    if len(upstream_shock) > 0:
        print("  [Upstream 기업(ALB·GLEN·Huayou) 음의 충격 시 각 레이어 반응]")
        print(f"  {'관계':20s}  {'대상 레이어':20s}  {'N':>4}  {'CAAR':>8}  {'p':>8}")
        print("  " + "-" * 68)
        for (rel, tstage), g in upstream_shock.groupby(["relationship","target_stage"]):
            s = caar_stats(g["CAR"].values)
            if s["N"] < 2: continue
            p = s.get("p", np.nan)
            sig = "***" if not np.isnan(p) and p<0.01 else \
                  "**"  if not np.isnan(p) and p<0.05 else \
                  "*"   if not np.isnan(p) and p<0.10 else ""
            print(f"  {rel:20s}  {tstage:20s}  {s['N']:>4}  "
                  f"{s['CAAR']:+8.4f}  "
                  f"{p:7.4f}{sig:3s}" if not np.isnan(p) else
                  f"{s['CAAR']:+8.4f}  {'N/A':>10}")
    else:
        print("  Upstream 충격 이벤트 없음 (supply_z threshold 미달)")

    print()
    # ── 상위 supply 이벤트 상세 ──
    print("■ supply_z 이벤트 상세 (NEG, 충격 크기 순)")
    sup_direct = panel[
        (panel["version"]=="supply") &
        (panel["relationship"]=="DIRECT") &
        (panel["window"]=="0_21") &
        (panel["quality"]=="full") &
        (panel["shock_type"]=="NEG")
    ].copy()
    sup_direct["shock_stage"] = sup_direct["shock_company"].map(stage)
    sup_direct["name"] = sup_direct["shock_company"].map(nm)
    sup_direct = sup_direct.sort_values("z_score")
    print(f"  {'기업':22s}  {'레이어':20s}  {'월':>8}  {'z_score':>8}  {'CAR[0,+21]':>11}  {'p':>7}")
    print("  " + "-" * 84)
    for _, r in sup_direct.iterrows():
        p = r["p_val"]
        sig = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.10 else ""
        print(f"  {r['name']:22s}  {r['shock_stage']:20s}  "
              f"{r['shock_month']:>8}  {r['z_score']:+8.2f}  "
              f"{r['CAR']:+11.4f}  {p:6.4f}{sig:3s}")

    print(f"\n{SEP}")
    print("저장: data/processed/car_panel_v2_comparison.parquet")
    print(SEP)


if __name__ == "__main__":
    main()
