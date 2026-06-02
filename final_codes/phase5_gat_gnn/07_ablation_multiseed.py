"""
07_ablation_multiseed.py
=========================
Multi-seed Ablation Study (5 seeds × 3 models)

단일 시드(42) ablation의 한계 극복:
  - 5개 독립 시드에서 각 모델 학습
  - 결과: mean ± std, 95% CI (t_{0.025,4} = 2.776)
  - 공정 비교: 시드별 Random Graph도 새로 샘플링

모델:
  ① Real Graph GAT   — seed_edges_44.csv 실제 공급망 그래프
  ② Random Graph GAT — 동일 평균 엣지 수, 시드별 다른 랜덤 엣지
  ③ No Graph GRU     — 그래프 없는 순수 시계열 baseline

데이터: tone_monthly_zscore_all44_v4.parquet (bias-free)
"""
from __future__ import annotations
import logging, random, sys, warnings
from pathlib import Path
from scipy.stats import t as t_dist
from typing import Dict, List

import numpy as np, pandas as pd, yaml, torch
import torch.nn as nn, torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch_geometric.nn import GATConv

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
EXP  = ROOT / "experiments/universe_44"

PATH_TONE   = EXP  / "data/processed/tone_monthly_zscore_all44_v4.parquet"
PATH_PRICES = ROOT / "data/processed/prices_daily.parquet"
PATH_MKT    = ROOT / "data/processed/market_proxies.parquet"
PATH_MKTMAP = ROOT / "data/processed/market_proxy_mapping.yaml"
PATH_EDGES  = EXP  / "data/seed/seed_edges_44.csv"
PATH_C44    = EXP  / "data/universe/companies_44.csv"
OUT_RPT     = EXP  / "reports/phase5_ablation_multiseed.txt"

SEEDS     = [42, 123, 456, 789, 999]
LOOKBACK  = 3; H_GRU=32; H_GAT=16; GAT_HEADS=4
DROPOUT   = 0.1; LR=1e-3; WD=1e-5
EPOCHS    = 300; PATIENCE = 40   # 약간 단축해서 속도 향상
TRAIN_END = "2022-06"; VAL_END = "2024-06"
MIN_VALID = 3; DYN_ALPHA = 0.3; Z_CLIP = 5.0

FEAT_COLS = [
    "z_score_v3","z_lag1_v3","z_lag2_v3","delta_z_v3",
    "rolling_3m_v3","supply_z_v3","z_cross_v3","z_cross_stage",
    "regime_covid","regime_ukraine","regime_ira",
    "regime_china_export","regime_trump_tariff",
]
N_FEAT = len(FEAT_COLS)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def load_market_adj(cids):
    prices = pd.read_parquet(PATH_PRICES)
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices[prices["company_id"].isin(cids)].sort_values(["company_id","date"])
    prices["ym"] = prices["date"].dt.to_period("M")
    co_last = prices.groupby(["company_id","ym"])["adj_close"].last().reset_index()
    co_last["ret"] = co_last.groupby("company_id")["adj_close"].pct_change()
    co_ret = co_last.dropna(subset=["ret"])
    mkt_df = pd.read_parquet(PATH_MKT)
    mkt_df["date"] = pd.to_datetime(mkt_df["date"])
    idx_col = "index_id" if "index_id" in mkt_df.columns else "index_name"
    mkt_df["ym"] = mkt_df["date"].dt.to_period("M")
    idx_last = mkt_df.groupby([idx_col,"ym"])["adj_close"].last().reset_index()
    irm = {}
    for idx, g in idx_last.groupby(idx_col):
        s = g.sort_values("ym").copy(); s["mkt_ret"] = s["adj_close"].pct_change()
        irm[str(idx)] = s[["ym","mkt_ret"]].dropna()
    with open(PATH_MKTMAP) as f:
        c2i = yaml.safe_load(f).get("company_to_market_proxy", {})
    rows = []
    for cid in cids:
        iname = c2i.get(cid)
        if not iname or iname not in irm: continue
        ci = co_ret[co_ret["company_id"]==cid][["ym","ret"]]
        m  = ci.merge(irm[iname], on="ym", how="inner")
        m["ar"] = m["ret"] - m["mkt_ret"]; m["company_id"] = cid
        rows.append(m[["company_id","ym","ar","ret","mkt_ret"]])
    return pd.concat(rows, ignore_index=True)


def build_feature_matrix(ar_df, cids):
    tone = pd.read_parquet(PATH_TONE)
    tone["ym"] = pd.to_datetime(tone["year_month"]).dt.to_period("M")
    tone = tone[tone["company_id"].isin(cids)].copy()
    for c in ["regime_covid","regime_ukraine","regime_ira","regime_china_export","regime_trump_tariff"]:
        if c in tone.columns and tone[c].dtype == bool:
            tone[c] = tone[c].astype("float32")
    for c in FEAT_COLS:
        if c not in tone.columns: tone[c] = 0.0
    reliable_col = "reliable_v3" if "reliable_v3" in tone.columns else "reliable_v2"
    panel = tone[["company_id","ym",reliable_col]+FEAT_COLS].merge(
        ar_df[["company_id","ym","ar"]], on=["company_id","ym"], how="left")
    panel = panel.sort_values(["company_id","ym"])
    Z_COLS = [c for c in FEAT_COLS if "z" in c.lower()]
    for c in FEAT_COLS:
        panel[c] = panel.groupby("company_id")[c].transform(lambda s: s.ffill().bfill().fillna(0.0))
    for c in Z_COLS: panel[c] = panel[c].clip(-Z_CLIP, Z_CLIP)
    panel["ar_next"] = panel.groupby("company_id")["ar"].shift(-1)
    panel["binary_next"] = (panel["ar_next"] > 0).astype("float32")
    panel["valid"] = (panel["ar"].notna() & panel["ar_next"].notna() & panel[reliable_col].fillna(False)).astype("float32")
    return panel, sorted(panel["ym"].unique())


def build_tensors(panel, months, cids):
    T, N = len(months), len(cids)
    cid2i = {c:i for i,c in enumerate(cids)}
    ym2t  = {ym:t for t,ym in enumerate(months)}
    X=np.zeros((T,N,N_FEAT),dtype=np.float32)
    y_bin=np.zeros((T,N),dtype=np.float32)
    y_cont=np.zeros((T,N),dtype=np.float32)
    mask=np.zeros((T,N),dtype=np.float32)
    for _,row in panel.iterrows():
        t=ym2t.get(row["ym"]); n=cid2i.get(row["company_id"])
        if t is None or n is None: continue
        X[t,n,:]=[float(row[c]) if pd.notna(row.get(c,np.nan)) else 0.0 for c in FEAT_COLS]
        if float(row.get("valid",0))>0:
            y_bin[t,n]=float(row["binary_next"]); y_cont[t,n]=float(row["ar_next"]); mask[t,n]=1.0
    return (torch.tensor(X,dtype=torch.float32), torch.tensor(y_bin,dtype=torch.float32),
            torch.tensor(y_cont,dtype=torch.float32), torch.tensor(mask,dtype=torch.float32))


def build_real_graphs(cids, months, panel):
    edges = pd.read_csv(PATH_EDGES)
    edges["valid_from"]=pd.to_datetime(edges["valid_from"],errors="coerce")
    edges["valid_to"]=pd.to_datetime(edges["valid_to"],errors="coerce")
    supply_lookup={}
    if "supply_z_v3" in panel.columns:
        supply_lookup={(row["company_id"],row["ym"]):float(row["supply_z_v3"])
                       for _,row in panel[["company_id","ym","supply_z_v3"]].dropna().iterrows()}
    ci={c:i for i,c in enumerate(cids)}; monthly={}
    for ym in months:
        t_stamp=ym.to_timestamp()
        vm=(edges["valid_from"].notna()&(edges["valid_from"]<=t_stamp)&
            (edges["valid_to"].isna()|(edges["valid_to"]>=t_stamp)))
        ve=edges[vm]; ei,ea=[],[]
        for _,r in ve.iterrows():
            s=ci.get(r["src_company_id"]); d=ci.get(r["dst_company_id"])
            if s is None or d is None: continue
            bw=max(0.01,min(1.0,float(r.get("confidence_plink",0.8))*float(r.get("strength",0.8))))
            sz=supply_lookup.get((r["src_company_id"],ym),0.0)
            dz=supply_lookup.get((r["dst_company_id"],ym),0.0)
            w=float(max(0.01,min(1.0,bw*(1.0+DYN_ALPHA*float(np.tanh((abs(sz)+abs(dz))/2.0))))))
            ei.extend([[s,d],[d,s]]); ea.extend([[w],[w]])
        if ei:
            monthly[ym]=(torch.tensor(ei,dtype=torch.long).t().contiguous(),
                         torch.tensor(ea,dtype=torch.float32))
        else:
            sl=torch.arange(len(cids))
            monthly[ym]=(torch.stack([sl,sl]),torch.ones(len(cids),1)*0.01)
    return monthly


def build_random_graphs(n_nodes, months, real_graphs, seed_offset):
    """시드별 다른 랜덤 그래프 생성."""
    rng = np.random.default_rng(99 + seed_offset)
    monthly = {}
    for ym in months:
        real_ei, _ = real_graphs[ym]
        n_edges = max(real_ei.shape[1]//2, 5)
        pairs = set()
        attempts = 0
        while len(pairs) < n_edges and attempts < 10000:
            s = int(rng.integers(0, n_nodes)); d = int(rng.integers(0, n_nodes))
            if s != d: pairs.add((s,d))
            attempts += 1
        if not pairs:
            sl = torch.arange(n_nodes)
            monthly[ym] = (torch.stack([sl,sl]), torch.ones(n_nodes,1)*0.01)
        else:
            ei = torch.tensor(list(pairs)+[(d,s) for s,d in pairs],dtype=torch.long).t().contiguous()
            ea = torch.ones(ei.shape[1],1,dtype=torch.float32)*0.5
            monthly[ym] = (ei,ea)
    return monthly


def make_windows(X, y_bin, y_cont, mask, lookback, offset=0):
    T=X.shape[0]; windows=[]
    for t in range(lookback-1,T):
        m=mask[t]
        if int(m.sum().item()) < MIN_VALID: continue
        windows.append((offset+t, X[t-lookback+1:t+1], y_bin[t], y_cont[t], m))
    return windows


class GATModel(nn.Module):
    def __init__(self,in_ch,h_gru,h_gat,heads,drop):
        super().__init__()
        self.drop=drop
        self.gru=nn.GRU(in_ch,h_gru,batch_first=True)
        self.gru_norm=nn.LayerNorm(h_gru)
        self.gat1=GATConv(h_gru,h_gat,heads=heads,concat=True,dropout=drop,edge_dim=1,add_self_loops=True)
        self.gat1_norm=nn.LayerNorm(h_gat*heads)
        self.gat2=GATConv(h_gat*heads,h_gat,heads=1,concat=False,dropout=drop,edge_dim=1,add_self_loops=True)
        self.gat2_norm=nn.LayerNorm(h_gat)
        self.head=nn.Sequential(nn.Linear(h_gat,h_gat//2),nn.ELU(),nn.Dropout(drop),nn.Linear(h_gat//2,1))
    def forward(self,x,ei,ea):
        h,_=self.gru(x.permute(1,0,2))
        h=self.gru_norm(F.dropout(h[:,-1,:],p=self.drop,training=self.training))
        h1=self.gat1_norm(F.elu(self.gat1(h,ei,ea))); h1=F.dropout(h1,p=self.drop,training=self.training)
        h2=self.gat2_norm(F.elu(self.gat2(h1,ei,ea))); return self.head(h2).squeeze(-1)


class GRUModel(nn.Module):
    def __init__(self,in_ch,h_gru,drop):
        super().__init__()
        self.drop=drop
        self.gru=nn.GRU(in_ch,h_gru,batch_first=True)
        self.norm=nn.LayerNorm(h_gru)
        self.mlp=nn.Sequential(nn.Linear(h_gru,h_gru//2),nn.ELU(),nn.Dropout(drop),nn.Linear(h_gru//2,1))
    def forward(self,x,ei,ea):
        h,_=self.gru(x.permute(1,0,2)); return self.mlp(self.norm(F.dropout(h[:,-1,:],p=self.drop,training=self.training))).squeeze(-1)


def masked_bce(logits,target,mask):
    n=mask.sum()
    if n<1: return torch.tensor(0.,requires_grad=True)
    return (F.binary_cross_entropy_with_logits(logits,target,reduction="none")*mask).sum()/n


def train_and_eval(model, tr_wins, va_wins, te_wins, graphs, months):
    opt=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=WD)
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="max",patience=10,factor=0.5,min_lr=1e-5)
    best_val,best_state,no_imp=0.,None,0

    for ep in range(1,EPOCHS+1):
        model.train()
        for t_abs,xw,yw_bin,_,mw in tr_wins:
            ei,ea=graphs[months[t_abs]]; opt.zero_grad()
            loss=masked_bce(model(xw,ei,ea),yw_bin,mw)
            if loss.item()==0: continue
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()

        model.eval()
        with torch.no_grad():
            va_lg,va_tb=[],[]
            for t_abs,xw,yw_bin,_,mw in va_wins:
                ei,ea=graphs[months[t_abs]]
                logits=model(xw,ei,ea).numpy(); m=mw.numpy().astype(bool)
                if m.sum()==0: continue
                va_lg.append(logits[m]); va_tb.append(yw_bin.numpy()[m])
        if not va_lg: continue
        va_prob=1/(1+np.exp(-np.concatenate(va_lg))); va_y=np.concatenate(va_tb)
        try: val_score=roc_auc_score(va_y,va_prob)
        except: val_score=0.
        sch.step(val_score)
        if val_score>best_val+1e-5: best_val=val_score; best_state={k:v.clone() for k,v in model.state_dict().items()}; no_imp=0
        else: no_imp+=1
        if no_imp>=PATIENCE: break

    if best_state: model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        lg,tb,tc=[],[],[]
        for t_abs,xw,yw_bin,yw_cont,mw in te_wins:
            ei,ea=graphs[months[t_abs]]
            logits=model(xw,ei,ea).numpy(); m=mw.numpy().astype(bool)
            if m.sum()==0: continue
            lg.append(logits[m]); tb.append(yw_bin.numpy()[m]); tc.append(yw_cont.numpy()[m])
    if not lg: return dict(auc=np.nan,f1=np.nan,acc=np.nan,ic=np.nan)
    lg=np.concatenate(lg); tb=np.concatenate(tb); tc=np.concatenate(tc)
    prob=1/(1+np.exp(-lg)); pred=(prob>0.5).astype(float)
    acc=accuracy_score(tb,pred); f1=f1_score(tb,pred,zero_division=0)
    try: auc=roc_auc_score(tb,prob)
    except: auc=np.nan
    rho,_=spearmanr(prob,tc); ic=float(rho) if not np.isnan(rho) else np.nan
    return dict(auc=auc,f1=f1,acc=acc,ic=ic)


def ci95(vals):
    vals=[v for v in vals if not np.isnan(v)]
    if len(vals)<2: return np.nan,np.nan,np.nan
    m=np.mean(vals); s=np.std(vals,ddof=1); n=len(vals)
    tc=t_dist.ppf(0.975,df=n-1)
    return m,m-tc*s/np.sqrt(n),m+tc*s/np.sqrt(n)


def main():
    log.info("=== Multi-seed Ablation Study (%d seeds) ===", len(SEEDS))
    c44=pd.read_csv(PATH_C44); cids=c44["company_id"].tolist()
    ar_df=load_market_adj(cids)
    panel,months=build_feature_matrix(ar_df,cids)
    months_str=[str(m) for m in months]
    tr_end=next((i for i,m in enumerate(months_str) if m>TRAIN_END),len(months))
    va_end=next((i for i,m in enumerate(months_str) if m>VAL_END),len(months))
    log.info("train=%s~%s val=%s~%s test=%s~%s",
             months[0],months[tr_end-1],months[tr_end],months[va_end-1],months[va_end],months[-1])

    X,y_bin,y_cont,mask=build_tensors(panel,months,cids)
    tr_wins=make_windows(X[:tr_end],y_bin[:tr_end],y_cont[:tr_end],mask[:tr_end],LOOKBACK,0)
    va_wins=make_windows(X[tr_end:va_end],y_bin[tr_end:va_end],y_cont[tr_end:va_end],mask[tr_end:va_end],LOOKBACK,tr_end)
    te_wins=make_windows(X[va_end:],y_bin[va_end:],y_cont[va_end:],mask[va_end:],LOOKBACK,va_end)
    log.info("윈도우: train=%d val=%d test=%d", len(tr_wins),len(va_wins),len(te_wins))

    # 실제 그래프 (공유)
    real_graphs=build_real_graphs(cids,months,panel)

    results = {"real":[], "random":[], "gru":[]}

    for seed_idx, seed in enumerate(SEEDS):
        log.info("\n── Seed %d (%d/%d) ──", seed, seed_idx+1, len(SEEDS))

        # ① Real Graph GAT
        set_seed(seed)
        gat_real=GATModel(N_FEAT,H_GRU,H_GAT,GAT_HEADS,DROPOUT)
        m_real=train_and_eval(gat_real,tr_wins,va_wins,te_wins,real_graphs,months)
        results["real"].append(m_real)
        log.info("  [Real GAT ]  AUC=%.4f  IC=%+.4f", m_real["auc"],m_real["ic"])

        # ② Random Graph GAT (시드별 다른 랜덤 엣지)
        rand_graphs=build_random_graphs(len(cids),months,real_graphs,seed_offset=seed)
        set_seed(seed)
        gat_rand=GATModel(N_FEAT,H_GRU,H_GAT,GAT_HEADS,DROPOUT)
        m_rand=train_and_eval(gat_rand,tr_wins,va_wins,te_wins,rand_graphs,months)
        results["random"].append(m_rand)
        log.info("  [Rand GAT ]  AUC=%.4f  IC=%+.4f", m_rand["auc"],m_rand["ic"])

        # ③ No Graph GRU
        set_seed(seed)
        gru=GRUModel(N_FEAT,H_GRU,DROPOUT)
        m_gru=train_and_eval(gru,tr_wins,va_wins,te_wins,real_graphs,months)
        results["gru"].append(m_gru)
        log.info("  [No-Gr GRU]  AUC=%.4f  IC=%+.4f", m_gru["auc"],m_gru["ic"])

    # ── 집계 ─────────────────────────────────────────────────────
    SEP="="*72
    lines=[SEP,"Phase 5 — Multi-seed Ablation Study (44개 기업, bias-free v4)",SEP,"",
           f"  Seeds: {SEEDS}  (각 시드마다 Real/Random/GRU 학습)",
           f"  데이터: tone_monthly_zscore_all44_v4.parquet (expanding window)",
           f"  Random Graph: seed-dependent (각 시드마다 다른 랜덤 엣지)", "",
           "■ 시드별 Raw 결과","",
           f"  {'Seed':>6}  {'Real_AUC':>9} {'Real_IC':>8}  {'Rand_AUC':>9} {'Rand_IC':>8}  {'GRU_AUC':>8} {'GRU_IC':>8}",
           "  "+"-"*68]
    for i,seed in enumerate(SEEDS):
        r=results["real"][i]; rd=results["random"][i]; g=results["gru"][i]
        lines.append(f"  {seed:>6}  {r['auc']:9.4f} {r['ic']:+8.4f}  "
                     f"{rd['auc']:9.4f} {rd['ic']:+8.4f}  {g['auc']:8.4f} {g['ic']:+8.4f}")

    lines+=["","■ 집계 결과 (mean ± std, 95% CI)","",
            f"  {'모델':<18} {'AUC mean':>9} {'AUC 95%CI':>20}  {'IC mean':>8} {'IC 95%CI':>20}",
            "  "+"-"*76]

    for key,label in [("real","Real Graph GAT"),("random","Random Graph GAT"),("gru","No Graph GRU")]:
        aucs=[r["auc"] for r in results[key]]
        ics =[r["ic"]  for r in results[key]]
        auc_m,auc_lo,auc_hi=ci95(aucs)
        ic_m, ic_lo, ic_hi =ci95(ics)
        lines.append(f"  {label:<18}  {auc_m:9.4f}  [{auc_lo:+.4f}, {auc_hi:+.4f}]  "
                     f"{ic_m:+8.4f}  [{ic_lo:+.4f}, {ic_hi:+.4f}]")

    # 해석
    real_aucs=[r["auc"] for r in results["real"]]
    rand_aucs=[r["auc"] for r in results["random"]]
    gru_aucs =[r["auc"] for r in results["gru"]]
    real_ics =[r["ic"]  for r in results["real"]]
    rand_ics =[r["ic"]  for r in results["random"]]
    gru_ics  =[r["ic"]  for r in results["gru"]]

    real_gt_rand_auc=sum(1 for r,rd in zip(real_aucs,rand_aucs) if r>rd)
    real_gt_gru_auc =sum(1 for r,g  in zip(real_aucs,gru_aucs)  if r>g)
    real_gt_rand_ic =sum(1 for r,rd in zip(real_ics, rand_ics)  if r>rd)

    lines+=[
        "",
        "■ 상대 우위 횟수 (5회 중)",
        f"  Real > Random (AUC): {real_gt_rand_auc}/5회",
        f"  Real > GRU    (AUC): {real_gt_gru_auc}/5회",
        f"  Real > Random (IC):  {real_gt_rand_ic}/5회",
        "",
        "■ 해석",
    ]
    if real_gt_rand_auc >= 4:
        lines.append("  Real > Random AUC 우위: 5회 중 4+회 → 공급망 KG 구조 기여 견고")
    elif real_gt_rand_auc >= 3:
        lines.append("  Real > Random AUC 우위: 5회 중 3회 → 기여 있으나 불안정")
    else:
        lines.append("  Real > Random AUC 우위 불충분 → 그래프 구조 기여 불명확")

    if real_gt_gru_auc >= 4:
        lines.append("  Real > GRU AUC 우위: 5회 중 4+회 → 그래프 사용이 순수 시계열보다 일관되게 우월")
    lines.append("")
    lines.append(SEP)

    rpt="\n".join(lines)
    OUT_RPT.parent.mkdir(parents=True,exist_ok=True)
    OUT_RPT.write_text(rpt,encoding="utf-8")
    print("\n"+rpt)
    log.info("저장: %s",OUT_RPT)


if __name__=="__main__":
    main()
