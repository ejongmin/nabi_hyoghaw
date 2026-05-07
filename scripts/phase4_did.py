"""
Phase 4 DID (Difference-in-Differences) 분석
==============================================
약점 해결: "PEER 전파가 공급망 효과인지, 단순 섹터 효과인지 구분"

설계:
  Treatment group: 충격 기업과 공급망으로 직접 연결된 기업
  Control group:   같은 레이어(섹터)이지만 공급망 미연결 기업

  DID = CAAR(treatment, event) - CAAR(control, event)
      - CAAR(treatment, non-event) + CAAR(control, non-event)

  interpretation:
  DID > 0: 공급망 연결 자체가 주가에 추가 영향 → 진짜 공급망 효과
  DID ≈ 0: 섹터 효과 (연결 여부 무관)

추가 분석:
  - Upstream shock → Downstream 반응 (공급망 방향)
  - supply_z 이벤트만 사용 (공급망 특화)
"""
from __future__ import annotations
import logging, sys
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd, yaml
from scipy import stats

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]

PATH_TONE  = ROOT/"data/processed/tone_monthly_zscore_18_v2.parquet"
PATH_PRICES= ROOT/"data/processed/prices_daily.parquet"
PATH_MKT   = ROOT/"data/processed/market_proxies.parquet"
PATH_MKTMAP= ROOT/"data/processed/market_proxy_mapping.yaml"
PATH_EDGES = ROOT/"data/seed/seed_edges_18_internal_v3.csv"
PATH_C18   = ROOT/"data/universe/companies_18.csv"
OUT_RPT    = ROOT/"reports/phase4_did.txt"
OUT_CSV    = ROOT/"reports/phase4_did_detail.csv"

EST_WIN_START=-250; EST_WIN_END=-20; MIN_EST=60; PREF_EST=120
WS, WE = 0, 21  # 주 창


def build_company_data(cids):
    prices = pd.read_parquet(PATH_PRICES)
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices[prices["company_id"].isin(cids)].sort_values(["company_id","date"])
    prices["price"] = prices["adj_close"].fillna(prices["close"])
    mkt_df = pd.read_parquet(PATH_MKT)
    mkt_df["date"] = pd.to_datetime(mkt_df["date"])
    idx_col = "index_id" if "index_id" in mkt_df.columns else "index_name"
    mkt_df["mkt_ret"] = mkt_df.groupby(idx_col)["adj_close"].pct_change()
    ir = {}
    for iname, g in mkt_df.groupby(idx_col):
        s = g.set_index("date")["mkt_ret"]; s.name="mkt_ret"
        ir[str(iname)] = s
    with open(PATH_MKTMAP) as f:
        c2i = yaml.safe_load(f).get("company_to_market_proxy",{})
    ew = (prices.assign(ret=lambda d: d.groupby("company_id")["price"]
                        .transform(lambda s: s.pct_change()))
          .dropna(subset=["ret"]).groupby("date")["ret"].mean().rename("mkt_ret"))
    cd = {}
    for cid, g in prices.groupby("company_id"):
        g = g.sort_values("date").copy(); g["ret"] = g["price"].pct_change()
        g = g.dropna(subset=["ret"])
        if len(g)<30: continue
        mkt_s = ir.get(c2i.get(cid,""), ew)
        df = g.set_index("date")[["ret"]].join(mkt_s.to_frame(), how="inner").dropna()
        if len(df)<30: continue
        dates = df.index.values.astype("datetime64[D]")
        cd[cid] = dict(dates=dates, ret=df["ret"].values.astype(np.float64),
                       mkt=df["mkt_ret"].values.astype(np.float64),
                       d2i={d:i for i,d in enumerate(dates)})
    return cd


def ols_car(ret, mkt, es, ee, xs, xe):
    n = ee-es
    if n<MIN_EST or es<0 or xe>len(ret) or xs<0: return None
    er,em=ret[es:ee],mkt[es:ee]
    sm,sr,smm,smr=em.sum(),er.sum(),(em*em).sum(),(em*er).sum()
    denom=n*smm-sm*sm
    if abs(denom)<1e-15: return None
    beta=(n*smr-sm*sr)/denom; alpha=(sr-beta*sm)/n
    sig=float(np.sqrt(np.sum((er-(alpha+beta*em))**2)/max(1,n-2)))
    if sig<1e-10: return None
    vr,vm=ret[xs:xe],mkt[xs:xe]; el=len(vr)
    if el<max(1,int((xe-xs)*0.8)): return None
    car=float((vr-(alpha+beta*vm)).sum())
    mbar=em.mean(); S2=np.sum((em-mbar)**2)
    corr=1+1/n+((vm.mean()-mbar)**2/S2 if S2>1e-15 else 0)
    dp=sig*np.sqrt(el*corr)
    if dp<1e-15: return None
    t=car/dp; p=float(2*stats.t.sf(abs(t),df=max(1,n-2)))
    q="full" if n>=PREF_EST else "marginal"
    return dict(CAR=car, t=t, p=p, quality=q)


def find_t0(cd, shock_month, offset=1):
    try:
        t0 = np.datetime64((pd.Period(shock_month,"M")+offset).to_timestamp().date(),"D")
    except: return None
    pos = np.searchsorted(cd["dates"], t0, side="left")
    return int(pos) if pos < len(cd["dates"]) else None


def build_supply_adjacency(edges_path, cids):
    """공급 방향 엣지만 추출 (SUPPLIES/BUYS_FROM)"""
    ci = set(cids)
    adj = defaultdict(set)  # shock_company → {connected companies}
    edges = pd.read_csv(edges_path)
    for _, r in edges.iterrows():
        s,d = r["src_company_id"], r["dst_company_id"]
        if s not in ci or d not in ci: continue
        if r["rel_type"] in ("SUPPLIES","BUYS_FROM","SOURCES_FROM","OWNS"):
            adj[s].add(d)
            adj[d].add(s)
    return dict(adj)


def compute_car_panel(events, cd_map, connected_map, stage_map, cids):
    """
    각 이벤트에 대해 모든 기업의 CAR 계산 + 연결 여부 레이블
    treatment: 공급망 직접 연결
    control:   같은 레이어이지만 미연결
    """
    rows = []
    for ev in events:
        shock_co = ev["shock_company"]
        connected = connected_map.get(shock_co, set())
        shock_stage = stage_map.get(shock_co, "")

        for cid in cids:
            if cid == shock_co: continue
            cd = cd_map.get(cid)
            if cd is None: continue
            t0 = find_t0(cd, ev["shock_month"], offset=1)
            if t0 is None: continue
            es = max(0, t0+EST_WIN_START); ee = t0+EST_WIN_END
            if ee-es < MIN_EST: continue
            xs, xe = t0+WS, t0+WE+1
            res = ols_car(cd["ret"], cd["mkt"], es, ee, xs, xe)
            if res is None or res["quality"]!="full": continue

            is_connected = cid in connected
            target_stage = stage_map.get(cid, "")

            rows.append(dict(
                event_id       = ev.get("event_id",""),
                shock_company  = shock_co,
                shock_month    = ev["shock_month"],
                shock_type     = ev["shock_type"],
                shock_stage    = shock_stage,
                target_company = cid,
                target_stage   = target_stage,
                treatment      = int(is_connected),  # 1=연결, 0=미연결
                CAR            = res["CAR"],
            ))
    return pd.DataFrame(rows)


def did_estimate(panel: pd.DataFrame, group_col: str = None):
    """
    DID = E[CAR|treatment=1] - E[CAR|treatment=0]
    선택적: group_col로 하위 그룹 분석 가능
    """
    results = []
    groups = panel[group_col].unique() if group_col else ["ALL"]

    for grp in groups:
        sub = panel if grp=="ALL" else panel[panel[group_col]==grp]
        treat = sub[sub["treatment"]==1]["CAR"].values
        ctrl  = sub[sub["treatment"]==0]["CAR"].values

        if len(treat)<3 or len(ctrl)<3:
            continue

        did = treat.mean() - ctrl.mean()
        # Welch t-test (불등분산 가정)
        t, p = stats.ttest_ind(treat, ctrl, equal_var=False)
        n_treat, n_ctrl = len(treat), len(ctrl)

        results.append(dict(
            group     = grp,
            N_treat   = n_treat,
            N_ctrl    = n_ctrl,
            CAAR_treat= float(treat.mean()),
            CAAR_ctrl = float(ctrl.mean()),
            DID       = float(did),
            t_did     = float(t),
            p_did     = float(p),
        ))

    return pd.DataFrame(results)


def main():
    log.info("Phase 4 DID 분석 시작")
    tone = pd.read_parquet(PATH_TONE)
    c18  = pd.read_csv(PATH_C18)
    nm   = dict(zip(c18["company_id"], c18["canonical_name"]))
    stage= dict(zip(c18["company_id"], c18["stage_bucket"]))
    cids = c18["company_id"].tolist()

    cd_map   = build_company_data(cids)
    conn_map = build_supply_adjacency(PATH_EDGES, cids)

    # 이벤트: supply_z NEG (공급망 특화, 가장 외생적)
    events = []
    seen = set()
    mask = tone["reliable_v2"] & tone["neg_shock_supply"]
    for _, r in tone[mask].iterrows():
        eid = f"{r['company_id']}_{r['year_month']}"
        if eid in seen: continue
        seen.add(eid)
        events.append(dict(event_id=eid, shock_company=r["company_id"],
                           shock_month=r["year_month"], shock_type="NEG"))
    log.info("supply_z NEG 이벤트: %d건", len(events))

    log.info("CAR 패널 계산 중 (모든 기업 × 이벤트)...")
    panel = compute_car_panel(events, cd_map, conn_map, stage, cids)
    log.info("패널: %d행", len(panel))

    if len(panel) == 0:
        log.warning("패널 데이터 없음")
        return

    # ── DID 분석 ──
    SEP = "="*68
    lines = [SEP, "Phase 4 DID 분석 — 공급망 효과 vs 섹터 효과 분리", SEP, "",
             "설계:",
             "  Treatment: 충격 기업과 공급망 직접 연결 기업 (SUPPLIES/OWNS 관계)",
             "  Control:   연결되지 않은 나머지 기업",
             "  이벤트:   supply_z NEG (공급망 특화 뉴스 기반 음의 충격)",
             "  DID > 0 → 공급망 연결이 추가적 영향 = 진짜 공급망 효과",
             "  DID ≈ 0 → 섹터/시장 공통 요인으로 설명 가능", ""]

    def sig(p):
        if np.isnan(p): return ""
        return "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.10 else ""

    # (A) 전체 풀드
    did_all = did_estimate(panel)
    lines.append("■ (A) 전체 풀드 DID")
    lines.append(f"  {'그룹':6s}  {'N(treat)':>9}  {'N(ctrl)':>8}  "
                 f"{'CAAR(treat)':>12}  {'CAAR(ctrl)':>11}  {'DID':>8}  {'p':>8}")
    lines.append("  "+"-"*72)
    for _, r in did_all.iterrows():
        p = r["p_did"]
        lines.append(f"  {'ALL':6s}  {int(r['N_treat']):9d}  {int(r['N_ctrl']):8d}  "
                     f"{r['CAAR_treat']:+12.4f}  {r['CAAR_ctrl']:+11.4f}  "
                     f"{r['DID']:+8.4f}  {p:.4f}{sig(p):3s}")

    # (B) 충격 기업 레이어별 DID
    lines += ["", "■ (B) 충격 기업 레이어별 DID"]
    did_stage = did_estimate(panel, group_col="shock_stage")
    lines.append(f"  {'충격 레이어':22s}  {'N_t':>5}  {'N_c':>5}  "
                 f"{'treat':>8}  {'ctrl':>8}  {'DID':>8}  {'p':>8}")
    lines.append("  "+"-"*72)
    for _, r in did_stage.iterrows():
        p = r["p_did"]
        lines.append(f"  {str(r['group']):22s}  {int(r['N_treat']):5d}  {int(r['N_ctrl']):5d}  "
                     f"{r['CAAR_treat']:+8.4f}  {r['CAAR_ctrl']:+8.4f}  "
                     f"{r['DID']:+8.4f}  {p:.4f}{sig(p):3s}")

    # (C) 대상 기업 레이어별 DID
    lines += ["", "■ (C) 대상 기업 레이어별 DID (충격 전달 방향)"]
    did_tgt = did_estimate(panel, group_col="target_stage")
    lines.append(f"  {'대상 레이어':22s}  {'N_t':>5}  {'N_c':>5}  "
                 f"{'treat':>8}  {'ctrl':>8}  {'DID':>8}  {'p':>8}")
    lines.append("  "+"-"*72)
    for _, r in did_tgt.iterrows():
        p = r["p_did"]
        lines.append(f"  {str(r['group']):22s}  {int(r['N_treat']):5d}  {int(r['N_ctrl']):5d}  "
                     f"{r['CAAR_treat']:+8.4f}  {r['CAAR_ctrl']:+8.4f}  "
                     f"{r['DID']:+8.4f}  {p:.4f}{sig(p):3s}")

    # 해석
    lines += ["", "■ 해석",
              "  DID > 0, p<0.10: 공급망 연결이 주가에 추가 영향 → 진짜 공급망 효과 존재",
              "  DID ≈ 0, p>0.10: 공급망 효과가 섹터 공통 효과에 묻혀 있음",
              "  → 결과에 따라 논문의 주장 강도 조절 필요", ""]
    lines.append(SEP)

    rpt = "\n".join(lines)
    OUT_RPT.write_text(rpt, encoding="utf-8")
    panel.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print("\n" + rpt)


if __name__ == "__main__":
    main()
