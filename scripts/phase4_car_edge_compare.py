"""
v1 엣지 vs v2 엣지 — Phase 4 CAR 비교
supply_z 기반 이벤트 사용
"""
from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd, yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PATH_TONE_V2 = ROOT/"data/processed/tone_monthly_zscore_18_v2.parquet"
PATH_PRICES  = ROOT/"data/processed/prices_daily.parquet"
PATH_MKT     = ROOT/"data/processed/market_proxies.parquet"
PATH_MKT_MAP = ROOT/"data/processed/market_proxy_mapping.yaml"
PATH_C18     = ROOT/"data/universe/companies_18.csv"
EDGE_FILES   = {"v1": ROOT/"data/seed/seed_edges_18_internal.csv",
                "v2": ROOT/"data/seed/seed_edges_18_internal_v2.csv"}

EST_WIN_START=-250; EST_WIN_END=-20; MIN_EST=60; PREF_EST=120
WINS=[(0,1,"0_1"),(0,5,"0_5"),(0,10,"0_10"),(0,21,"0_21"),(-1,1,"m1_1")]


def load_events(tone):
    evs, seen = [], set()
    mask = tone["reliable_v2"] & (tone["neg_shock_supply"] | tone["pos_shock_supply"])
    for _, r in tone[mask].iterrows():
        eid = f"{r['company_id']}_{r['year_month']}"
        if eid in seen: continue
        seen.add(eid)
        evs.append(dict(
            event_id      = eid,
            shock_company = r["company_id"],
            shock_month   = r["year_month"],
            shock_type    = "NEG" if r["neg_shock_supply"] else "POS",
            z_score       = float(r.get("supply_z", 0) or 0),
        ))
    return evs


def build_cd(cids):
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
        s = g.set_index("date")["mkt_ret"]; s.name = "mkt_ret"
        ir[str(iname)] = s

    with open(PATH_MKT_MAP) as f:
        c2i = yaml.safe_load(f).get("company_to_market_proxy", {})

    ew = (prices.assign(ret=lambda d: d.groupby("company_id")["price"]
                        .transform(lambda s: s.pct_change()))
          .dropna(subset=["ret"]).groupby("date")["ret"].mean().rename("mkt_ret"))

    cd = {}
    for cid, g in prices.groupby("company_id"):
        g = g.sort_values("date").copy()
        g["ret"] = g["price"].pct_change()
        g = g.dropna(subset=["ret"])
        if len(g) < 30: continue
        mkt_s = ir.get(c2i.get(cid, ""), ew)
        df = g.set_index("date")[["ret"]].join(mkt_s.to_frame(), how="inner").dropna()
        if len(df) < 30: continue
        dates = df.index.values.astype("datetime64[D]")
        cd[cid] = dict(dates=dates,
                       ret=df["ret"].values.astype(np.float64),
                       mkt=df["mkt_ret"].values.astype(np.float64),
                       d2i={d: i for i, d in enumerate(dates)})
    return cd


def ols_car(ret, mkt, es, ee, xs, xe):
    n = ee - es
    if n < MIN_EST or es < 0 or xe > len(ret) or xs < 0:
        return None
    er, em = ret[es:ee], mkt[es:ee]
    sm, sr, smm, smr = em.sum(), er.sum(), (em*em).sum(), (em*er).sum()
    denom = n*smm - sm*sm
    if abs(denom) < 1e-15: return None
    beta  = (n*smr - sm*sr) / denom
    alpha = (sr - beta*sm) / n
    sig   = float(np.sqrt(np.sum((er - (alpha+beta*em))**2) / max(1, n-2)))
    if sig < 1e-10: return None
    vr, vm = ret[xs:xe], mkt[xs:xe]
    el = len(vr)
    if el < max(1, int((xe-xs)*0.8)): return None
    car = float((vr - (alpha + beta*vm)).sum())
    mbar = em.mean(); S2 = np.sum((em - mbar)**2)
    corr = 1 + 1/n + ((vm.mean()-mbar)**2/S2 if S2 > 1e-15 else 0)
    dp = sig * np.sqrt(el * corr)
    if dp < 1e-15: return None
    t = car / dp
    p = float(2 * stats.t.sf(abs(t), df=max(1, n-2)))
    q = "full" if n >= PREF_EST else "marginal"
    return dict(CAR=car, t=t, p=p, quality=q)


def find_t0(cd, sm):
    try:
        t0 = np.datetime64((pd.Period(sm, "M")+1).to_timestamp().date(), "D")
    except Exception:
        return None
    pos = np.searchsorted(cd["dates"], t0, side="left")
    return int(pos) if pos < len(cd["dates"]) else None


def build_adj(edges_df, cids):
    ci  = set(cids)
    adj = defaultdict(list)
    for _, r in edges_df.iterrows():
        s, d = r["src_company_id"], r["dst_company_id"]
        if s not in ci or d not in ci: continue
        w  = float(r.get("confidence_plink", 0.8)) * float(r.get("strength", 0.8))
        rt = r["rel_type"]
        if rt in ("SUPPLIES", "BUYS_FROM", "SOURCES_FROM", "OWNS"):
            adj[s].append(dict(nb=d, dir="DOWNSTREAM", rt=rt, w=w))
            adj[d].append(dict(nb=s, dir="UPSTREAM",   rt=rt, w=w))
        else:  # COMPETES_WITH, PARTNERS_WITH
            adj[s].append(dict(nb=d, dir="PEER", rt=rt, w=w))
            adj[d].append(dict(nb=s, dir="PEER", rt=rt, w=w))
    return dict(adj)


def run_car(events, cd_map, adj, label):
    rows = []
    for ev in events:
        cd_sh = cd_map.get(ev["shock_company"])
        if cd_sh is None: continue
        targets = [dict(cid=ev["shock_company"], rel="DIRECT", rt="SELF")]
        seen = {ev["shock_company"]}
        for e in adj.get(ev["shock_company"], []):
            if e["nb"] not in seen:
                seen.add(e["nb"])
                targets.append(dict(cid=e["nb"], rel=e["dir"], rt=e["rt"]))
        for tgt in targets:
            cd = cd_map.get(tgt["cid"])
            if cd is None: continue
            t0 = find_t0(cd, ev["shock_month"])
            if t0 is None: continue
            es = max(0, t0+EST_WIN_START)
            ee = t0+EST_WIN_END
            if ee - es < MIN_EST: continue
            for ws, we, wl in WINS:
                xs, xe = t0+ws, t0+we+1
                res = ols_car(cd["ret"], cd["mkt"], es, ee, xs, xe)
                if res is None: continue
                rows.append(dict(
                    edge_ver       = label,
                    event_id       = ev["event_id"],
                    shock_company  = ev["shock_company"],
                    shock_month    = ev["shock_month"],
                    shock_type     = ev["shock_type"],
                    z_score        = ev["z_score"],
                    target_company = tgt["cid"],
                    relationship   = tgt["rel"],
                    window         = wl,
                    CAR            = res["CAR"],
                    t_stat         = res["t"],
                    p_val          = res["p"],
                    quality        = res["quality"],
                ))
    return pd.DataFrame(rows)


def caar_stats(vals):
    n = len(vals)
    if n == 0:
        return dict(N=0, CAAR=np.nan, t=np.nan, p=np.nan)
    cv = vals.mean()
    sd = vals.std(ddof=1) if n > 1 else np.nan
    t, p = np.nan, np.nan
    if n > 1 and not np.isnan(sd) and sd > 1e-15:
        t = cv / (sd / np.sqrt(n))
        p = float(2 * stats.t.sf(abs(t), df=n-1))
    return dict(N=n, CAAR=float(cv),
                t=float(t) if not np.isnan(t) else np.nan,
                p=float(p) if not np.isnan(p) else np.nan)


def sig_mark(p):
    if np.isnan(p): return ""
    return "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.10 else ""


def main():
    print("Phase 4 CAR — v1 vs v2 엣지 비교 (supply_z 이벤트)")
    tone   = pd.read_parquet(PATH_TONE_V2)
    c18    = pd.read_csv(PATH_C18)
    nm     = dict(zip(c18["company_id"], c18["canonical_name"]))
    stg    = dict(zip(c18["company_id"], c18["stage_bucket"]))
    cids   = c18["company_id"].tolist()

    print("  CompanyData 구축 중...")
    cd_map = build_cd(cids)
    events = load_events(tone)
    print(f"  이벤트: {len(events)}건  |  CompanyData: {len(cd_map)}기업")

    all_panels = []
    for ver, ep in EDGE_FILES.items():
        adj   = build_adj(pd.read_csv(ep), cids)
        panel = run_car(events, cd_map, adj, ver)
        all_panels.append(panel)
        print(f"  [{ver}] CAR 계산: {len(panel)}행")

    panel = pd.concat(all_panels, ignore_index=True)
    panel["shock_stage"]  = panel["shock_company"].map(stg)
    panel["target_stage"] = panel["target_company"].map(stg)
    panel.to_parquet(ROOT/"data/processed/car_panel_edge_comparison.parquet", index=False)

    SEP = "=" * 72

    # ── 1. 전체 관계별 CAAR 비교 ──────────────────────────────
    print(f"\n{SEP}")
    print("supply_z 이벤트 기반 CAR — v1 엣지 vs v2 엣지 [창 0,+21]")
    print(SEP)

    print(f"\n  {'관계':14s}  {'v1 CAAR':>8} {'p':>7} {'N':>4}  "
          f"{'v2 CAAR':>8} {'p':>7} {'N':>4}  변화")
    print("  " + "-"*70)

    for rel in ["DIRECT", "DOWNSTREAM", "UPSTREAM", "PEER"]:
        res = {}
        for ver in ["v1", "v2"]:
            sub = panel[(panel["edge_ver"]==ver) & (panel["window"]=="0_21") &
                        (panel["quality"]=="full") & (panel["relationship"]==rel)]
            res[ver] = caar_stats(sub["CAR"].values)
        v1r, v2r = res["v1"], res["v2"]
        delta = ""
        if not np.isnan(v1r["CAAR"]) and not np.isnan(v2r["CAAR"]):
            d = (v2r["CAAR"] - v1r["CAAR"]) * 100
            delta = f"{'▲' if d>0 else '▼'}{abs(d):.1f}%p"
        p1, p2 = v1r["p"], v2r["p"]
        print(f"  {rel:14s}  "
              f"{v1r['CAAR']:+8.4f} {p1:6.3f}{sig_mark(p1):3s} {v1r['N']:4d}  "
              f"{v2r['CAAR']:+8.4f} {p2:6.3f}{sig_mark(p2):3s} {v2r['N']:4d}  {delta}")

    # ── 2. Upstream 충격 → 레이어별 반응 ─────────────────────
    print(f"\n■ v2 엣지: Upstream 충격(ALB·Glencore·Huayou) → 공급망 전파")
    up_cos = [c for c in cids if stg.get(c)=="Upstream/Refining"]
    up_sub = panel[(panel["edge_ver"]=="v2") & (panel["shock_type"]=="NEG") &
                   (panel["window"]=="0_21") & (panel["quality"]=="full") &
                   (panel["shock_company"].isin(up_cos))].copy()

    print(f"  {'관계':14s} {'대상 레이어':22s} {'N':>4} {'CAAR':>9} {'p':>8}")
    print("  " + "-"*58)
    for (rel, tstg), g in up_sub.groupby(["relationship","target_stage"]):
        if len(g) < 2: continue
        s = caar_stats(g["CAR"].values)
        p = s["p"]
        pstr = f"{p:.3f}{sig_mark(p):3s}" if not np.isnan(p) else "  N/A"
        print(f"  {rel:14s} {tstg:22s} {s['N']:4d} {s['CAAR']:+9.4f} {pstr:>11}")

    # ── 3. NEG 개별 이벤트 (Upstream 충격) ───────────────────
    print(f"\n■ Upstream 기업 supply_z NEG 이벤트 → DIRECT CAR 상세")
    up_direct = panel[
        (panel["edge_ver"]=="v2") & (panel["relationship"]=="DIRECT") &
        (panel["window"]=="0_21") & (panel["quality"]=="full") &
        (panel["shock_type"]=="NEG") & (panel["shock_company"].isin(up_cos))
    ].copy()
    up_direct["name"] = up_direct["shock_company"].map(nm)
    up_direct = up_direct.sort_values("z_score")
    print(f"  {'기업':22s} {'월':>8} {'z_score':>8} {'CAR':>9} {'p':>8}")
    print("  "+"-"*56)
    for _, r in up_direct.iterrows():
        p = r["p_val"]
        pstr = f"{p:.3f}{sig_mark(p):3s}" if not np.isnan(p) else "  N/A"
        print(f"  {r['name']:22s} {r['shock_month']:>8} "
              f"{r['z_score']:+8.2f} {r['CAR']:+9.4f} {pstr:>11}")

    # ── 4. v2 새로 생긴 전파 경로 ────────────────────────────
    print(f"\n■ v2에서 새로 생긴 경로의 CAR (v1엔 없었던 엣지)")
    new_rel_pairs = [
        ("ALB",      "DOWNSTREAM", "247540.KQ"),  # ALB → EcoPro BM
        ("ALB",      "DOWNSTREAM", "300919.SZ"),  # ALB → CNGR
        ("GLEN.L",   "DOWNSTREAM", "247540.KQ"),  # Glen → EcoPro BM
        ("GLEN.L",   "DOWNSTREAM", "300919.SZ"),  # Glen → CNGR
        ("603799.SH","DOWNSTREAM", "247540.KQ"),  # Huayou → EcoPro BM
    ]
    print(f"  {'충격 기업':20s} {'관계':12s} {'대상 기업':20s} {'N':>3} {'CAAR':>9} {'p':>8}")
    print("  "+"-"*66)
    for shock_co, rel, tgt_co in new_rel_pairs:
        sub = panel[(panel["edge_ver"]=="v2") & (panel["shock_company"]==shock_co) &
                    (panel["relationship"]==rel) & (panel["target_company"]==tgt_co) &
                    (panel["window"]=="0_21") & (panel["quality"]=="full")]
        s = caar_stats(sub["CAR"].values)
        p = s["p"]
        pstr = f"{p:.3f}{sig_mark(p):3s}" if not np.isnan(p) else "  N/A"
        print(f"  {nm.get(shock_co,shock_co):20s} {rel:12s} "
              f"{nm.get(tgt_co,tgt_co):20s} {s['N']:3d} "
              f"{s['CAAR']:+9.4f} {pstr:>11}")

    print(f"\n{SEP}")
    print("저장: data/processed/car_panel_edge_comparison.parquet")
    print(SEP)


if __name__ == "__main__":
    main()
