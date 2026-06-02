"""
Phase 4-B: GDELT 일별 이벤트 직접 CAR
=======================================
현재 파이프라인: 월별 FinBERT z-score → 54 이벤트 (검정력 부족)
이 스크립트:    v9 일별 리스크 이벤트 직접 사용 → 최대 34,722 이벤트

필터 전략:
  공급망 특화 리스크: natural + logistics  (가장 외생적)
  소스 품질: source_tier >= 2
  심각도: severity >= 0.6
  → 예상 이벤트: ~5,000~8,000건

이점:
  - Downstream 전파 효과 검정력 대폭 향상
  - Hop1 > Hop2 감쇠 테스트 가능
  - Sub-sample 분석 가능 (China vs 해외, OEM vs Cell 등)

출력:
  data/processed/car_panel_gdelt_daily.parquet
  reports/phase4_gdelt_daily_summary.txt
"""
from __future__ import annotations
import logging, sys
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd, yaml, pyarrow.parquet as pq
from scipy import stats

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
PATH_V9    = ROOT/"data/processed/risk_events_classified_v9.parquet"
PATH_PRICES= ROOT/"data/processed/prices_daily.parquet"
PATH_MKT   = ROOT/"data/processed/market_proxies.parquet"
PATH_MKTMAP= ROOT/"data/processed/market_proxy_mapping.yaml"
PATH_EDGES = ROOT/"data/seed/seed_edges_18_internal_v3.csv"
PATH_C18   = ROOT/"data/universe/companies_18.csv"
OUT_PANEL  = ROOT/"data/processed/car_panel_gdelt_daily.parquet"
OUT_RPT    = ROOT/"reports/phase4_gdelt_daily_summary.txt"

FINAL_18 = ['TSLA','BMW.DE','VOW3.DE','MBG.DE','7267.T','1211.HK',
            '300750.SZ','373220.KS','3931.HK','300014.SZ','002074.SZ',
            '051910.KS','BAS.DE','247540.KQ','300919.SZ','ALB','GLEN.L','603799.SH']

EST_WIN_START = -250; EST_WIN_END = -20
MIN_EST = 60; PREF_EST = 120

# 이벤트 창: [0,+1], [0,+5], [0,+10], [0,+21]
WINDOWS = [(0,1,"0_1"),(0,5,"0_5"),(0,10,"0_10"),(0,21,"0_21"),(-1,1,"m1_1")]

# 이벤트 필터 (품질 균형)
SUPPLY_TYPES = {"natural","logistics"}
MIN_SEVERITY = 0.6
MIN_TIER     = 2


def load_gdelt_events() -> pd.DataFrame:
    """v9에서 고품질 공급망 특화 이벤트 추출"""
    log.info("v9 이벤트 로딩 (18개 기업, supply+quality 필터)...")
    pf = pq.ParquetFile(PATH_V9)
    batches = []
    for batch in pf.iter_batches(
        batch_size=400_000,
        columns=["event_id","company_id","event_time","risk_types",
                 "severity","source_tier","source_tier_weight","tone"]
    ):
        b = batch.to_pandas()
        b = b[b["company_id"].isin(FINAL_18)].copy()
        if len(b) == 0: continue

        # 공급망 특화 리스크 필터
        def is_supply(rt):
            if not isinstance(rt, str): return False
            types = set(t.strip() for t in rt.split(","))
            return bool(types & SUPPLY_TYPES) and not (types == {"other"})

        b["is_supply"] = b["risk_types"].apply(is_supply)
        b = b[b["is_supply"]].copy()

        # 품질 필터
        b = b[(b["source_tier"].fillna(0) >= MIN_TIER) &
              (b["severity"] >= MIN_SEVERITY)].copy()

        if len(b): batches.append(b)

    df = pd.concat(batches, ignore_index=True)
    df["event_time"] = pd.to_datetime(df["event_time"])
    df["date"] = df["event_time"].dt.normalize()

    # 기업-날짜 중복 제거 (같은 날 여러 기사 → 심각도 최대값 유지)
    df = (df.sort_values("severity", ascending=False)
            .drop_duplicates(["company_id","date"])
            .reset_index(drop=True))

    log.info("  최종 이벤트: %d건", len(df))
    return df


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
        if len(g) < 30: continue
        mkt_s = ir.get(c2i.get(cid,""), ew)
        df = g.set_index("date")[["ret"]].join(mkt_s.to_frame(), how="inner").dropna()
        if len(df) < 30: continue
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
    return dict(CAR=car,t=t,p=p,quality=q)


def build_adjacency(cids):
    ci=set(cids); adj=defaultdict(list)
    edges=pd.read_csv(PATH_EDGES)
    # valid_from/to 무시 (정적 그래프로 우선 실행)
    for _,r in edges.iterrows():
        s,d=r["src_company_id"],r["dst_company_id"]
        if s not in ci or d not in ci: continue
        w=float(r.get("confidence_plink",0.8))*float(r.get("strength",0.8))
        rt=r["rel_type"]
        if rt in("SUPPLIES","BUYS_FROM","SOURCES_FROM","OWNS"):
            adj[s].append(dict(nb=d,dir="DOWNSTREAM",rt=rt,w=w,hops=1))
            adj[d].append(dict(nb=s,dir="UPSTREAM",rt=rt,w=w,hops=1))
        else:
            adj[s].append(dict(nb=d,dir="PEER",rt=rt,w=w,hops=1))
            adj[d].append(dict(nb=s,dir="PEER",rt=rt,w=w,hops=1))
    # 2홉 추가
    for center in list(adj.keys()):
        h1={e["nb"] for e in adj[center]}
        for e1 in list(adj[center]):
            for e2 in adj.get(e1["nb"],[]):
                n2=e2["nb"]
                if n2==center or n2 in h1: continue
                dir2=f"{e1['dir']}_2" if e1["dir"]==e2["dir"] else "MIXED_2"
                adj[center].append(dict(nb=n2,dir=dir2,rt=f"{e1['rt']}->{e2['rt']}",
                                        w=e1["w"]*e2["w"],hops=2))
    return dict(adj)


def caar_stats(vals):
    n=len(vals)
    if n==0: return dict(N=0,CAAR=np.nan,t=np.nan,p=np.nan,p_sign=np.nan)
    cv=vals.mean(); sd=vals.std(ddof=1) if n>1 else np.nan
    t,p=np.nan,np.nan
    if n>1 and not np.isnan(sd) and sd>1e-15:
        t=cv/(sd/np.sqrt(n)); p=float(2*stats.t.sf(abs(t),df=n-1))
    nneg=int((vals<0).sum())
    ps=float(2*min(stats.binom.cdf(nneg,n,0.5),1-stats.binom.cdf(nneg-1,n,0.5)))
    return dict(N=n,CAAR=float(cv),t=float(t) if not np.isnan(t) else np.nan,
                p=float(p) if not np.isnan(p) else np.nan,p_sign=ps)


def main():
    log.info("Phase 4-B: GDELT 일별 이벤트 CAR 시작")
    c18  = pd.read_csv(PATH_C18)
    nm   = dict(zip(c18["company_id"],c18["canonical_name"]))
    stg  = dict(zip(c18["company_id"],c18["stage_bucket"]))
    cids = c18["company_id"].tolist()

    # 이벤트 & 데이터
    events = load_gdelt_events()
    log.info("CompanyData 구축...")
    cd_map = build_company_data(cids)
    adj    = build_adjacency(cids)

    # CAR 계산 루프
    log.info("CAR 계산 시작 (이벤트×기업×창)...")
    rows = []; skip_nodata=0; skip_insuff=0; n_done=0

    for _, ev in events.iterrows():
        cid  = ev["company_id"]
        date = np.datetime64(ev["date"].date(), "D")
        sev  = float(ev["severity"])

        # 대상: 직접 충격 기업 + 1홉 + 2홉 연결 기업
        targets = [dict(target=cid, rel="DIRECT", hops=0)]
        seen = {cid}
        for e in adj.get(cid,[]):
            if e["nb"] not in seen:
                seen.add(e["nb"])
                targets.append(dict(target=e["nb"], rel=e["dir"],
                                    hops=e["hops"], rt=e.get("rt","?")))

        for tgt in targets:
            tcid = tgt["target"]
            cd   = cd_map.get(tcid)
            if cd is None: skip_nodata+=1; continue

            pos = np.searchsorted(cd["dates"], date, side="left")
            if pos >= len(cd["dates"]): skip_nodata+=1; continue

            es = max(0, pos+EST_WIN_START); ee = pos+EST_WIN_END
            if ee-es < MIN_EST: skip_insuff+=1; continue

            for ws,we,wl in WINDOWS:
                xs,xe = pos+ws, pos+we+1
                res = ols_car(cd["ret"],cd["mkt"],es,ee,xs,xe)
                if res is None: continue
                rows.append(dict(
                    shock_company  = cid,
                    event_date     = str(ev["date"].date()),
                    shock_stage    = stg.get(cid,""),
                    severity       = sev,
                    risk_types     = ev.get("risk_types",""),
                    target_company = tcid,
                    target_stage   = stg.get(tcid,""),
                    relationship   = tgt["rel"],
                    hops           = tgt["hops"],
                    window         = wl,
                    CAR            = res["CAR"],
                    t_stat         = res["t"],
                    p_val          = res["p"],
                    quality        = res["quality"],
                ))

        n_done += 1
        if n_done % 1000 == 0:
            log.info("  진행: %d/%d  (CAR %d건)", n_done, len(events), len(rows))

    log.info("CAR 계산 완료: %d행 (skip: nodata=%d insuff=%d)",
             len(rows), skip_nodata, skip_insuff)

    panel = pd.DataFrame(rows)
    panel.to_parquet(OUT_PANEL, index=False)
    log.info("저장: %s", OUT_PANEL)

    # 요약 통계
    def sig(p):
        if np.isnan(p): return ""
        return "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.10 else ""

    SEP="="*72
    lines=[SEP,"Phase 4-B GDELT 일별 이벤트 CAR 결과 요약",SEP,"",
           f"이벤트: {len(events):,}건 (supply+quality 필터)", "",
           "■ 관계 유형별 CAAR [0,+21] (full quality)",
           f"  {'관계':20s}  {'N':>6}  {'CAAR':>9}  {'t':>7}  {'p':>8}"]
    lines.append("  "+"-"*56)

    sub = panel[(panel["window"]=="0_21")&(panel["quality"]=="full")]
    for rel in ["DIRECT","DOWNSTREAM","UPSTREAM","PEER",
                "DOWNSTREAM_2","UPSTREAM_2","MIXED_2"]:
        g = sub[sub["relationship"]==rel]["CAR"].values
        if len(g) < 5: continue
        s = caar_stats(g)
        p=s["p"]
        lines.append(f"  {rel:20s}  {s['N']:6d}  {s['CAAR']:+9.4f}  "
                     f"{s['t']:+7.3f}  {p:.4f}{sig(p):3s}")

    lines += ["","■ 공급망 단계별 CAAR (직접 효과, [0,+21])",
              f"  {'충격 레이어':22s}  {'N':>6}  {'CAAR':>9}  {'p':>8}"]
    lines.append("  "+"-"*50)
    direct = sub[sub["relationship"]=="DIRECT"]
    for s_stg,g in direct.groupby("shock_stage"):
        st=caar_stats(g["CAR"].values)
        p=st["p"]
        lines.append(f"  {s_stg:22s}  {st['N']:6d}  {st['CAAR']:+9.4f}  "
                     f"{p:.4f}{sig(p):3s}")

    lines += ["","■ Hop 거리별 CAAR (전파 감쇠 테스트, [0,+21])",
              "  → 공급망 효과라면 hop1 > hop2 감쇠 예상",
              f"  {'관계':20s}  {'hop':>4}  {'N':>6}  {'CAAR':>9}  {'p':>8}"]
    lines.append("  "+"-"*56)
    for (rel,hops),g in sub.groupby(["relationship","hops"]):
        if len(g["CAR"].values) < 10: continue
        st=caar_stats(g["CAR"].values)
        p=st["p"]
        lines.append(f"  {rel:20s}  {hops:4d}  {st['N']:6d}  "
                     f"{st['CAAR']:+9.4f}  {p:.4f}{sig(p):3s}")

    lines += ["","■ Sub-sample: 중국 기업 vs 해외 기업",
              f"  {'그룹':20s}  {'N':>6}  {'CAAR':>9}  {'p':>8}"]
    lines.append("  "+"-"*46)
    china_cids = {c for c in FINAL_18 if any(x in c for x in [".SZ",".SH",".HK"])}
    direct_c = direct.copy()
    direct_c["region"] = direct_c["shock_company"].apply(
        lambda x: "중국/홍콩" if x in china_cids else "해외")
    for reg,g in direct_c.groupby("region"):
        st=caar_stats(g["CAR"].values)
        p=st["p"]
        lines.append(f"  {reg:20s}  {st['N']:6d}  {st['CAAR']:+9.4f}  "
                     f"{p:.4f}{sig(p):3s}")

    lines+=[f"\n{SEP}\n주: *** p<0.01, ** p<0.05, * p<0.10",SEP]
    rpt="\n".join(lines)
    OUT_RPT.write_text(rpt,encoding="utf-8")
    print("\n"+rpt)


if __name__=="__main__":
    main()
