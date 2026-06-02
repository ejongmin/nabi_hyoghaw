"""
Walk-Forward Cross Validation — universe_44
============================================
고정 train/val/test 분할 대신 확장 윈도우(expanding window) 방식으로
여러 번 반복 학습·평가해 IC 신뢰구간을 제시.

설계:
  Fold 1: train 2017-01~2021-06 → val 2021-07~2022-06 → test 2022-07~2023-06
  Fold 2: train 2017-01~2022-06 → val 2022-07~2023-06 → test 2023-07~2024-06
  Fold 3: train 2017-01~2023-06 → val 2023-07~2024-06 → test 2024-07~2025-06
  Fold 4: train 2017-01~2024-06 → val 2024-07~2025-06 → test 2025-07~2026-05

  각 fold별 AUC, IC 계산 → 평균/표준편차/95% CI → 안정적 성능 추정
"""
from __future__ import annotations
import logging, random, sys, warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np, pandas as pd, yaml, torch
import torch.nn as nn, torch.nn.functional as F
from scipy.stats import spearmanr, t as t_dist
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

try:
    from torch_geometric.nn import GATConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    log.warning("torch_geometric 없음 — GRU baseline만 실행")

ROOT = Path(__file__).resolve().parents[4]   # nabi_hyoghaw/
EXP  = ROOT / "experiments/universe_44"
HYP  = ROOT / "experiments/hypothesis/walk_forward_cv"

PATH_TONE   = EXP  / "data/processed/tone_monthly_zscore_all44_v2.parquet"
PATH_PRICES = ROOT / "data/processed/prices_daily.parquet"
PATH_MKT    = ROOT / "data/processed/market_proxies.parquet"
PATH_MKTMAP = ROOT / "data/processed/market_proxy_mapping.yaml"
PATH_EDGES  = EXP  / "data/seed/seed_edges_44.csv"
PATH_C44    = EXP  / "data/universe/companies_44.csv"
OUT_RPT     = HYP  / "reports/wf_cv_result.txt"

# ── Fold 정의 ──────────────────────────────────────────────────
FOLDS = [
    {"name": "Fold1", "train_end": "2021-06", "val_end": "2022-06", "test_end": "2023-06"},
    {"name": "Fold2", "train_end": "2022-06", "val_end": "2023-06", "test_end": "2024-06"},
    {"name": "Fold3", "train_end": "2023-06", "val_end": "2024-06", "test_end": "2025-06"},
    {"name": "Fold4", "train_end": "2024-06", "val_end": "2025-06", "test_end": "2026-05"},
]

LOOKBACK=3; H_GRU=32; H_GAT=16; GAT_HEADS=4; DROPOUT=0.1
LR=1e-3; WD=1e-5; EPOCHS=300; PATIENCE=40; SEED=42; MIN_VALID=3; DYN_ALPHA=0.3

FEAT_COLS = [
    "z_score_v3","z_lag1_v3","z_lag2_v3","delta_z_v3","rolling_3m_v3",
    "supply_z_v3","z_cross_v3","z_cross_stage",
    "regime_covid","regime_ukraine","regime_ira","regime_china_export","regime_trump_tariff",
]
N_FEAT = len(FEAT_COLS)


def set_seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)


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
        c2i = yaml.safe_load(f).get("company_to_market_proxy",{})
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
        if c in tone.columns and tone[c].dtype == bool: tone[c] = tone[c].astype("float32")
    for c in FEAT_COLS:
        if c not in tone.columns: tone[c] = 0.0
    reliable_col = "reliable_v3" if "reliable_v3" in tone.columns else "reliable_v2"
    panel = tone[["company_id","ym",reliable_col]+FEAT_COLS].merge(
        ar_df[["company_id","ym","ar"]], on=["company_id","ym"], how="left")
    panel = panel.sort_values(["company_id","ym"])
    Z_COLS = [c for c in FEAT_COLS if "z" in c.lower()]
    for c in FEAT_COLS:
        panel[c] = panel.groupby("company_id")[c].transform(lambda s: s.ffill().bfill().fillna(0.0))
    for c in Z_COLS: panel[c] = panel[c].clip(-5.0,5.0)
    panel["ar_next"] = panel.groupby("company_id")["ar"].shift(-1)
    panel["binary_next"] = (panel["ar_next"]>0).astype("float32")
    panel["valid"] = (panel["ar"].notna()&panel["ar_next"].notna()&panel[reliable_col].fillna(False)).astype("float32")
    return panel, sorted(panel["ym"].unique())


def build_tensors(panel, months, cids):
    T,N = len(months),len(cids)
    cid2i={c:i for i,c in enumerate(cids)}; ym2t={ym:t for t,ym in enumerate(months)}
    X=np.zeros((T,N,N_FEAT),dtype=np.float32)
    y_bin=np.zeros((T,N),dtype=np.float32); y_cont=np.zeros((T,N),dtype=np.float32)
    mask=np.zeros((T,N),dtype=np.float32)
    for _,row in panel.iterrows():
        t=ym2t.get(row["ym"]); n=cid2i.get(row["company_id"])
        if t is None or n is None: continue
        X[t,n,:]=[float(row[c]) if pd.notna(row.get(c,np.nan)) else 0.0 for c in FEAT_COLS]
        if float(row.get("valid",0))>0:
            y_bin[t,n]=float(row["binary_next"]); y_cont[t,n]=float(row["ar_next"]); mask[t,n]=1.0
    return (torch.tensor(X,dtype=torch.float32), torch.tensor(y_bin,dtype=torch.float32),
            torch.tensor(y_cont,dtype=torch.float32), torch.tensor(mask,dtype=torch.float32))


def build_dynamic_graphs(cids, months, panel):
    edges = pd.read_csv(PATH_EDGES)
    edges["valid_from"]=pd.to_datetime(edges["valid_from"],errors="coerce")
    edges["valid_to"]=pd.to_datetime(edges["valid_to"],errors="coerce")
    supply_lookup={}
    col="supply_z_v3" if "supply_z_v3" in panel.columns else None
    if col:
        supply_lookup={(row["company_id"],row["ym"]):float(row[col])
                       for _,row in panel[["company_id","ym",col]].dropna().iterrows()}
    ci={c:i for i,c in enumerate(cids)}; monthly={}
    for ym in months:
        t_stamp=ym.to_timestamp()
        valid_mask=(edges["valid_from"].notna()&(edges["valid_from"]<=t_stamp)&
                    (edges["valid_to"].isna()|(edges["valid_to"]>=t_stamp)))
        valid_edges=edges[valid_mask]; ei,ea=[],[]
        for _,r in valid_edges.iterrows():
            s=ci.get(r["src_company_id"]); d=ci.get(r["dst_company_id"])
            if s is None or d is None: continue
            base_w=max(0.01,min(1.0,float(r.get("confidence_plink",0.8))*float(r.get("strength",0.8))))
            src_z=supply_lookup.get((r["src_company_id"],ym),0.0)
            dst_z=supply_lookup.get((r["dst_company_id"],ym),0.0)
            stress=(abs(src_z)+abs(dst_z))/2.0
            w=float(max(0.01,min(1.0,base_w*(1.0+DYN_ALPHA*float(np.tanh(stress))))))
            ei.extend([[s,d],[d,s]]); ea.extend([[w],[w]])
        if ei:
            monthly[ym]=(torch.tensor(ei,dtype=torch.long).t().contiguous(),
                         torch.tensor(ea,dtype=torch.float32))
        else:
            sl=torch.arange(len(cids))
            monthly[ym]=(torch.stack([sl,sl]),torch.ones(len(cids),1)*0.01)
    return monthly


def make_windows(X,y_bin,y_cont,mask,lookback,offset=0):
    T=X.shape[0]; windows=[]
    for t in range(lookback-1,T):
        m=mask[t]
        if int(m.sum().item())<MIN_VALID: continue
        windows.append((offset+t,X[t-lookback+1:t+1],y_bin[t],y_cont[t],m))
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
        h1=self.gat1_norm(F.elu(self.gat1(h,ei,ea)))
        h1=F.dropout(h1,p=self.drop,training=self.training)
        h2=self.gat2_norm(F.elu(self.gat2(h1,ei,ea)))
        return self.head(h2).squeeze(-1)


class GRUModel(nn.Module):
    def __init__(self,in_ch,h_gru,drop):
        super().__init__()
        self.drop=drop
        self.gru=nn.GRU(in_ch,h_gru,batch_first=True)
        self.norm=nn.LayerNorm(h_gru)
        self.mlp=nn.Sequential(nn.Linear(h_gru,h_gru//2),nn.ELU(),nn.Dropout(drop),nn.Linear(h_gru//2,1))
    def forward(self,x,ei,ea):
        h,_=self.gru(x.permute(1,0,2))
        return self.mlp(self.norm(F.dropout(h[:,-1,:],p=self.drop,training=self.training))).squeeze(-1)


def masked_bce(logits,target,mask):
    n=mask.sum()
    if n<1: return torch.tensor(0.,requires_grad=True)
    return (F.binary_cross_entropy_with_logits(logits,target,reduction="none")*mask).sum()/n


def train_one(model,opt,wins,graphs,months):
    model.train(); total=0; cnt=0
    for t_abs,xw,yw_bin,_,mw in wins:
        ei,ea=graphs[months[t_abs]]; opt.zero_grad()
        loss=masked_bce(model(xw,ei,ea),yw_bin,mw)
        if loss.item()==0: continue
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        total+=loss.item(); cnt+=1
    return total/max(1,cnt)


@torch.no_grad()
def evaluate(model,wins,graphs,months):
    model.eval(); lg,tb,tc=[],[],[]
    for t_abs,xw,yw_bin,yw_cont,mw in wins:
        ei,ea=graphs[months[t_abs]]
        logits=model(xw,ei,ea).numpy(); m=mw.numpy().astype(bool)
        if m.sum()==0: continue
        lg.append(logits[m]); tb.append(yw_bin.numpy()[m]); tc.append(yw_cont.numpy()[m])
    if not lg: return dict(acc=np.nan,f1=np.nan,auc=np.nan,ic=np.nan)
    lg=np.concatenate(lg); tb=np.concatenate(tb); tc=np.concatenate(tc)
    prob=1/(1+np.exp(-lg)); pred=(prob>0.5).astype(float)
    acc=accuracy_score(tb,pred); f1=f1_score(tb,pred,zero_division=0)
    try: auc=roc_auc_score(tb,prob)
    except: auc=np.nan
    rho,_=spearmanr(prob,tc); ic=float(rho) if not np.isnan(rho) else np.nan
    return dict(acc=acc,f1=f1,auc=auc,ic=ic)


def train_model(model,tr_wins,va_wins,graphs,months,name):
    opt=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=WD)
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="max",patience=10,factor=0.5,min_lr=1e-5)
    best_val,best_state,no_imp=0.,None,0
    for ep in range(1,EPOCHS+1):
        train_one(model,opt,tr_wins,graphs,months)
        va=evaluate(model,va_wins,graphs,months)
        val_score=va["auc"] if not np.isnan(va["auc"]) else 0.
        sch.step(val_score)
        if val_score>best_val+1e-5: best_val=val_score; best_state={k:v.clone() for k,v in model.state_dict().items()}; no_imp=0
        else: no_imp+=1
        if no_imp>=PATIENCE:
            log.info("    [%s] early stop ep=%d (best_auc=%.4f)",name,ep,best_val); break
    if best_state: model.load_state_dict(best_state)
    return model


def run_fold(fold_cfg, panel, months, cids, graphs):
    name=fold_cfg["name"]
    te=fold_cfg["train_end"]; ve=fold_cfg["val_end"]; xe=fold_cfg["test_end"]
    months_str=[str(m) for m in months]
    tr_end=next((i for i,m in enumerate(months_str) if m>te),len(months))
    va_end=next((i for i,m in enumerate(months_str) if m>ve),len(months))
    te_end=next((i for i,m in enumerate(months_str) if m>xe),len(months))

    X,y_bin,y_cont,mask=build_tensors(panel,months,cids)
    tr_wins=make_windows(X[:tr_end],y_bin[:tr_end],y_cont[:tr_end],mask[:tr_end],LOOKBACK,0)
    va_wins=make_windows(X[tr_end:va_end],y_bin[tr_end:va_end],y_cont[tr_end:va_end],mask[tr_end:va_end],LOOKBACK,tr_end)
    te_wins=make_windows(X[va_end:te_end],y_bin[va_end:te_end],y_cont[va_end:te_end],mask[va_end:te_end],LOOKBACK,va_end)
    log.info("  %s: train=%d월 val=%d월 test=%d월 | wins tr=%d va=%d te=%d",
             name,tr_end,va_end-tr_end,te_end-va_end,len(tr_wins),len(va_wins),len(te_wins))

    set_seed(SEED)
    if HAS_PYG:
        gat=GATModel(N_FEAT,H_GRU,H_GAT,GAT_HEADS,DROPOUT)
        gat=train_model(gat,tr_wins,va_wins,graphs,months,"GAT")
        gat_m=evaluate(gat,te_wins,graphs,months)
    else:
        gat_m=dict(acc=np.nan,f1=np.nan,auc=np.nan,ic=np.nan)

    set_seed(SEED)
    gru=GRUModel(N_FEAT,H_GRU,DROPOUT)
    gru=train_model(gru,tr_wins,va_wins,graphs,months,"GRU")
    gru_m=evaluate(gru,te_wins,graphs,months)

    log.info("  %s TEST → GAT AUC=%.4f IC=%+.4f | GRU AUC=%.4f IC=%+.4f",
             name,gat_m["auc"],gat_m["ic"],gru_m["auc"],gru_m["ic"])
    return {"fold":name,"gat":gat_m,"gru":gru_m,
            "n_train":len(tr_wins),"n_val":len(va_wins),"n_test":len(te_wins)}


def ci95(vals):
    vals=[v for v in vals if not np.isnan(v)]
    if len(vals)<2: return np.nan,np.nan,np.nan
    m=np.mean(vals); s=np.std(vals,ddof=1); n=len(vals)
    t=t_dist.ppf(0.975,df=n-1)
    return m,m-t*s/np.sqrt(n),m+t*s/np.sqrt(n)


def main():
    log.info("=== Walk-Forward CV 시작 (%d folds) ===",len(FOLDS))
    c44=pd.read_csv(PATH_C44); cids=c44["company_id"].tolist()
    ar_df=load_market_adj(cids)
    panel,months=build_feature_matrix(ar_df,cids)
    log.info("전체 데이터: %d기업 %d월",len(cids),len(months))
    graphs=build_dynamic_graphs(cids,months,panel)

    results=[]
    for fold_cfg in FOLDS:
        log.info("\n── %s ──", fold_cfg["name"])
        res=run_fold(fold_cfg,panel,months,cids,graphs)
        results.append(res)

    # 집계
    gat_aucs=[r["gat"]["auc"] for r in results]
    gat_ics =[r["gat"]["ic"]  for r in results]
    gru_aucs=[r["gru"]["auc"] for r in results]
    gru_ics =[r["gru"]["ic"]  for r in results]

    SEP="="*64
    lines=[SEP,"Walk-Forward CV 결과 (universe_44, 4 folds)",SEP,"",
           f"  Fold 정의:",
           *[f"    {f['name']}: train~{f['train_end']} val~{f['val_end']} test~{f['test_end']}" for f in FOLDS],
           ""]

    lines+=["■ Fold별 성능","",
            f"  {'Fold':<8} {'GAT AUC':>8} {'GAT IC':>8} {'GRU AUC':>8} {'GRU IC':>8}",
            "  "+"-"*44]
    for r in results:
        lines.append(f"  {r['fold']:<8} {r['gat']['auc']:8.4f} {r['gat']['ic']:+8.4f} "
                     f"{r['gru']['auc']:8.4f} {r['gru']['ic']:+8.4f}")

    g_auc_m,g_auc_lo,g_auc_hi=ci95(gat_aucs)
    g_ic_m, g_ic_lo, g_ic_hi =ci95(gat_ics)
    lines+=[""," ■ 95% 신뢰구간 (t-distribution, 4 folds)",
            f"  GAT  AUC: {g_auc_m:.4f}  [{g_auc_lo:.4f}, {g_auc_hi:.4f}]",
            f"  GAT  IC : {g_ic_m:+.4f}  [{g_ic_lo:+.4f}, {g_ic_hi:+.4f}]",
            f"  GRU  AUC: {ci95(gru_aucs)[0]:.4f}  [{ci95(gru_aucs)[1]:.4f}, {ci95(gru_aucs)[2]:.4f}]",
            f"  GRU  IC : {ci95(gru_ics)[0]:+.4f}  [{ci95(gru_ics)[1]:+.4f}, {ci95(gru_ics)[2]:+.4f}]",
            "","■ 해석",
            f"  GAT IC 평균 {g_ic_m:+.4f}이 0보다 {'크면 → 유의미한 예측력' if g_ic_m>0 else '작으면 → 신호 없음'}",
            f"  IC 95% CI가 0 포함 여부: {'포함 (비유의)' if g_ic_lo<0<g_ic_hi else '미포함 (유의)'}",
            "",SEP]

    rpt="\n".join(lines)
    OUT_RPT.parent.mkdir(parents=True,exist_ok=True)
    OUT_RPT.write_text(rpt,encoding="utf-8")
    print("\n"+rpt)
    log.info("저장: %s",OUT_RPT)


if __name__=="__main__":
    main()
