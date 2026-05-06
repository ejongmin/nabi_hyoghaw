"""
Phase 5 GNN v2 — v2 엣지 + supply_z 피처 사용
phase5_temporal_gat.py 의 개선판:
  - 엣지: seed_edges_18_internal_v2.csv (88건, 오류 수정 + Upstream→Materials 추가)
  - 피처: z_score_v2, supply_z, z_cross_all 추가 (12개)
  - reliable: reliable_v2 (더 엄격한 기준)
"""
from __future__ import annotations
import logging, random, sys, warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np, pandas as pd, yaml, torch
import torch.nn as nn, torch.nn.functional as F
from scipy.stats import spearmanr
from torch_geometric.nn import GATConv

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]

PATH_TONE  = ROOT/"data/processed/tone_monthly_zscore_18_v2.parquet"
PATH_PRICES= ROOT/"data/processed/prices_daily.parquet"
PATH_MKT   = ROOT/"data/processed/market_proxies.parquet"
PATH_MKTMAP= ROOT/"data/processed/market_proxy_mapping.yaml"
PATH_EDGES = ROOT/"data/seed/seed_edges_18_internal_v2.csv"
PATH_C18   = ROOT/"data/universe/companies_18.csv"

OUT_PRED   = ROOT/"data/processed/phase5_v2_predictions.parquet"
OUT_ATTN   = ROOT/"reports/phase5_v2_attention_weights.csv"
OUT_IC     = ROOT/"reports/phase5_v2_ic_by_month.csv"
OUT_RPT    = ROOT/"reports/phase5_v2_summary.txt"

# ── 하이퍼파라미터 (v1과 동일 유지 → 공정 비교)
LOOKBACK  = 6
H_GRU     = 32
H_GAT     = 16
GAT_HEADS = 4
DROPOUT   = 0.3
LR        = 1e-3
WD        = 1e-4
EPOCHS    = 300
PATIENCE  = 30
SEED      = 42
TRAIN_R   = 0.65
VAL_R     = 0.82

# ── v2 피처 (12개, v1은 10개)
FEAT_COLS = [
    "z_score_v2",    # ★ strict FinBERT z-score (v1: z_score)
    "z_lag1_v2",     # ★ lag1 (strict 기준)
    "z_lag2_v2",     # ★ lag2
    "z_lag3",        # v1과 동일 (v2에 z_lag3_v2 없음)
    "delta_z_v2",    # ★ 전월 대비 변화
    "rolling_3m_v2", # ★ 3개월 이동평균
    "supply_z",      # ★★ 공급망 특화 감성 (신규)
    "z_cross_all",   # ★★ 전체 18개 기업 내 상대 위치 (신규)
    "regime_covid",
    "regime_ukraine",
    "regime_ira",
    "regime_china_export",
]
N_FEAT = len(FEAT_COLS)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


# ── 데이터 로딩 (phase5_temporal_gat.py 와 동일 구조)
def load_market_adj(prices_path, mkt_path, mmap_path, cids):
    prices = pd.read_parquet(prices_path)
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices[prices["company_id"].isin(cids)].sort_values(["company_id","date"])
    prices["ym"] = prices["date"].dt.to_period("M")
    co_last = prices.groupby(["company_id","ym"])["adj_close"].last().reset_index()
    co_last["ret"] = co_last.groupby("company_id")["adj_close"].pct_change()
    co_ret = co_last.dropna(subset=["ret"])

    mkt_df = pd.read_parquet(mkt_path)
    mkt_df["date"] = pd.to_datetime(mkt_df["date"])
    idx_col = "index_id" if "index_id" in mkt_df.columns else "index_name"
    mkt_df["ym"] = mkt_df["date"].dt.to_period("M")
    idx_last = mkt_df.groupby([idx_col,"ym"])["adj_close"].last().reset_index()
    irm = {}
    for idx, g in idx_last.groupby(idx_col):
        s = g.sort_values("ym").copy()
        s["mkt_ret"] = s["adj_close"].pct_change()
        irm[str(idx)] = s[["ym","mkt_ret"]].dropna()

    with open(mmap_path) as f:
        c2i = yaml.safe_load(f).get("company_to_market_proxy",{})

    rows = []
    for cid in cids:
        iname = c2i.get(cid)
        if not iname or iname not in irm: continue
        ci = co_ret[co_ret["company_id"]==cid][["ym","ret"]]
        mi = irm[iname]
        m = ci.merge(mi, on="ym", how="inner")
        m["ar"] = m["ret"] - m["mkt_ret"]
        m["company_id"] = cid
        rows.append(m[["company_id","ym","ar"]])
    return pd.concat(rows, ignore_index=True)


def build_feature_matrix(tone_path, ar_df, cids):
    tone = pd.read_parquet(tone_path)
    tone["ym"] = pd.to_datetime(tone["year_month"]).dt.to_period("M")
    tone = tone[tone["company_id"].isin(cids)].copy()

    for c in ["regime_covid","regime_ukraine","regime_ira","regime_china_export"]:
        if tone[c].dtype == bool:
            tone[c] = tone[c].astype("float32")

    # v1 lag3 사용 (v2에 없음)
    if "z_lag3" not in tone.columns and "z_lag3" not in FEAT_COLS:
        pass

    panel = tone[["company_id","ym","reliable_v2"] + FEAT_COLS].merge(
        ar_df[["company_id","ym","ar"]], on=["company_id","ym"], how="left")

    panel = panel.sort_values(["company_id","ym"])
    for c in FEAT_COLS:
        if c in panel.columns:
            panel[c] = panel.groupby("company_id")[c].transform(
                lambda s: s.ffill().bfill().fillna(0.0))
        else:
            panel[c] = 0.0

    panel["ar_next"] = panel.groupby("company_id")["ar"].shift(-1)
    panel["valid"] = (
        panel["ar"].notna() & panel["ar_next"].notna() &
        panel["reliable_v2"].fillna(False)
    ).astype("float32")

    months = sorted(panel["ym"].unique())
    n_valid = int(panel["valid"].sum())
    log.info("피처 행렬: %d기업, %d월, 유효=%d/%d (%.1f%%)",
             panel["company_id"].nunique(), len(months),
             n_valid, len(panel), 100*n_valid/max(1,len(panel)))
    return panel, months


def build_tensors(panel, months, cids):
    T, N = len(months), len(cids)
    cid2i = {c:i for i,c in enumerate(cids)}
    ym2t  = {ym:t for t,ym in enumerate(months)}
    X = np.zeros((T,N,N_FEAT), dtype=np.float32)
    y = np.zeros((T,N), dtype=np.float32)
    mask = np.zeros((T,N), dtype=np.float32)
    for _, row in panel.iterrows():
        t = ym2t.get(row["ym"]); n = cid2i.get(row["company_id"])
        if t is None or n is None: continue
        X[t,n,:] = [float(row[c]) if c in row.index and pd.notna(row[c]) else 0.0
                    for c in FEAT_COLS]
        if float(row.get("valid",0)) > 0:
            y[t,n] = float(row["ar_next"]); mask[t,n] = 1.0
    return (torch.tensor(X,dtype=torch.float32),
            torch.tensor(y,dtype=torch.float32),
            torch.tensor(mask,dtype=torch.float32))


def build_graph(edges_path, cids):
    edges = pd.read_csv(edges_path)
    ci = {c:i for i,c in enumerate(cids)}
    ei, ea = [], []
    for _, r in edges.iterrows():
        s = ci.get(r["src_company_id"]); d = ci.get(r["dst_company_id"])
        if s is None or d is None: continue
        w = max(0.01, min(1.0, float(r.get("confidence_plink",0.8))*float(r.get("strength",0.8))))
        ei.append([s,d]); ea.append([w])
        ei.append([d,s]); ea.append([w])
    log.info("그래프: %d노드, %d엣지", len(cids), len(ei))
    return (torch.tensor(ei,dtype=torch.long).t().contiguous(),
            torch.tensor(ea,dtype=torch.float32))


def make_windows(X, y, mask, lookback, min_valid=3):
    T = X.shape[0]; wins = []
    for t in range(lookback-1, T):
        m = mask[t]
        if int(m.sum().item()) < min_valid: continue
        wins.append((X[t-lookback+1:t+1], y[t], m))
    return wins


# ── 모델 (v1과 동일 아키텍처, 입력 채널만 N_FEAT로)
class TemporalGAT(nn.Module):
    def __init__(self, in_ch, h_gru, h_gat, heads, drop):
        super().__init__()
        self.drop = drop
        self.gru = nn.GRU(in_ch, h_gru, batch_first=True)
        self.gru_norm = nn.LayerNorm(h_gru)
        self.gat1 = GATConv(h_gru, h_gat, heads=heads, concat=True,
                            dropout=drop, edge_dim=1, add_self_loops=True)
        self.gat1_norm = nn.LayerNorm(h_gat*heads)
        self.gat2 = GATConv(h_gat*heads, h_gat, heads=1, concat=False,
                            dropout=drop, edge_dim=1, add_self_loops=True)
        self.gat2_norm = nn.LayerNorm(h_gat)
        self.head = nn.Sequential(nn.Linear(h_gat, h_gat//2), nn.ELU(),
                                  nn.Dropout(drop), nn.Linear(h_gat//2, 1))

    def forward(self, x, ei, ea, return_attn=False):
        L, N, _ = x.shape
        h, _ = self.gru(x.permute(1,0,2))
        h = self.gru_norm(F.dropout(h[:,-1,:], p=self.drop, training=self.training))
        if return_attn:
            h1,(ei1,aw1) = self.gat1(h,ei,ea,return_attention_weights=True)
        else:
            h1 = self.gat1(h,ei,ea)
        h1 = self.gat1_norm(F.elu(h1))
        h1 = F.dropout(h1, p=self.drop, training=self.training)
        if return_attn:
            h2,(ei2,aw2) = self.gat2(h1,ei,ea,return_attention_weights=True)
        else:
            h2 = self.gat2(h1,ei,ea)
        h2 = self.gat2_norm(F.elu(h2))
        pred = self.head(h2).squeeze(-1)
        if return_attn: return pred,(ei1,aw1),(ei2,aw2)
        return pred


class BaselineGRU(nn.Module):
    def __init__(self, in_ch, h_gru, drop):
        super().__init__()
        self.drop = drop
        self.gru = nn.GRU(in_ch, h_gru, batch_first=True)
        self.norm = nn.LayerNorm(h_gru)
        self.mlp  = nn.Sequential(nn.Linear(h_gru, h_gru//2), nn.ELU(),
                                  nn.Dropout(drop), nn.Linear(h_gru//2, 1))
    def forward(self, x, *a, **kw):
        h, _ = self.gru(x.permute(1,0,2))
        h = self.norm(F.dropout(h[:,-1,:], p=self.drop, training=self.training))
        return self.mlp(h).squeeze(-1)


def masked_mse(pred, tgt, mask):
    n = mask.sum()
    if n < 1: return torch.tensor(0., requires_grad=True)
    return ((pred-tgt)**2 * mask).sum() / n


def train_epoch(model, opt, wins, ei, ea):
    model.train(); total = 0; cnt = 0
    for xw,yw,mw in wins:
        opt.zero_grad()
        pred = model(xw,ei,ea)
        loss = masked_mse(pred,yw,mw)
        if loss.item()==0: continue
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step(); total+=loss.item(); cnt+=1
    return total/max(1,cnt)


@torch.no_grad()
def evaluate(model, wins, ei, ea):
    model.eval(); ap,at=[],[]
    for xw,yw,mw in wins:
        p=model(xw,ei,ea).numpy(); t=yw.numpy(); m=mw.numpy().astype(bool)
        if m.sum()==0: continue
        ap.append(p[m]); at.append(t[m])
    if not ap: return float('inf'),float('inf')
    pa=np.concatenate(ap); ta=np.concatenate(at)
    return float(np.sqrt(np.mean((pa-ta)**2))), float(np.mean(np.abs(pa-ta)))


def train_model(model, tr_wins, va_wins, ei, ea, name):
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5, min_lr=1e-5)
    best_val, best_state, no_imp = float('inf'), None, 0
    for ep in range(1, EPOCHS+1):
        train_epoch(model, opt, tr_wins, ei, ea)
        _, val_rmse = evaluate(model, va_wins, ei, ea)
        sch.step(val_rmse)
        if val_rmse < best_val - 1e-6:
            best_val=val_rmse; best_state={k:v.clone() for k,v in model.state_dict().items()}; no_imp=0
        else:
            no_imp+=1
        if ep%50==0:
            log.info("  [%s] ep=%d val_rmse=%.4f lr=%.6f", name, ep, val_rmse, opt.param_groups[0]['lr'])
        if no_imp>=PATIENCE:
            log.info("  [%s] early stop ep=%d (best=%.4f)", name, ep, best_val); break
    if best_state: model.load_state_dict(best_state)
    return model


@torch.no_grad()
def compute_ic(model, wins, ei, ea, months, lookback, start_t, cids):
    model.eval(); rows=[]
    for i,(xw,yw,mw) in enumerate(wins):
        p=model(xw,ei,ea).numpy(); t=yw.numpy(); m=mw.numpy().astype(bool)
        pv,tv=p[m],t[m]
        if len(pv)<3: continue
        rho,pval=spearmanr(pv,tv)
        if np.isnan(rho): continue
        mi=start_t+i+lookback
        ym=str(months[mi]) if mi<len(months) else "?"
        rows.append(dict(year_month=ym, IC=float(rho), IC_p=float(pval),
                         dir_acc=float(np.mean(np.sign(pv)==np.sign(tv))),
                         n_nodes=int(m.sum())))
    return pd.DataFrame(rows)


@torch.no_grad()
def extract_attn(model, wins, ei, ea, cids, nm):
    model.eval(); s1=None; s2=None; cnt=0
    for xw,_,_ in wins:
        _,(ei1,aw1),(ei2,aw2)=model(xw,ei,ea,return_attn=True)
        a1=aw1.mean(-1).numpy(); a2=aw2.mean(-1).numpy()
        s1=a1.copy() if s1 is None else s1+a1
        s2=a2.copy() if s2 is None else s2+a2
        cnt+=1
    if cnt==0: return pd.DataFrame()
    m1,m2=s1/cnt,s2/cnt
    ei_np=ei.numpy()
    rows=[]
    for e in range(ei_np.shape[1]):
        si,di=ei_np[0,e],ei_np[1,e]
        if si>=len(cids) or di>=len(cids): continue
        rows.append(dict(src=cids[si],src_name=nm.get(cids[si],"?"),
                         dst=cids[di],dst_name=nm.get(cids[di],"?"),
                         attn_l1=float(m1[e]),attn_l2=float(m2[e]),
                         attn_mean=float((m1[e]+m2[e])/2)))
    return pd.DataFrame(rows).sort_values("attn_mean",ascending=False)


def main():
    set_seed(SEED)
    for p in [OUT_PRED.parent, OUT_RPT.parent]: p.mkdir(parents=True, exist_ok=True)

    log.info("Phase 5 GNN v2 시작 — v2 엣지(%s) + supply_z 피처", PATH_EDGES.name)
    log.info("피처(%d개): %s", N_FEAT, FEAT_COLS)

    c18  = pd.read_csv(PATH_C18)
    cids = c18["company_id"].tolist()
    nm   = dict(zip(c18["company_id"], c18["canonical_name"]))
    stg  = dict(zip(c18["company_id"], c18["stage_bucket"]))

    ar_df = load_market_adj(PATH_PRICES, PATH_MKT, PATH_MKTMAP, cids)
    panel, months = build_feature_matrix(PATH_TONE, ar_df, cids)
    X, y, mask = build_tensors(panel, months, cids)
    ei, ea = build_graph(PATH_EDGES, cids)

    T = X.shape[0]
    tr_end = int(T*TRAIN_R); va_end = int(T*VAL_R)
    log.info("분할: train=%s~%s, val=%s~%s, test=%s~%s",
             months[0], months[tr_end-1],
             months[tr_end], months[va_end-1],
             months[va_end], months[-1])

    tr_wins = make_windows(X[:tr_end], y[:tr_end], mask[:tr_end], LOOKBACK)
    va_wins = make_windows(X[tr_end:va_end], y[tr_end:va_end], mask[tr_end:va_end], LOOKBACK)
    te_wins = make_windows(X[va_end:], y[va_end:], mask[va_end:], LOOKBACK)
    log.info("윈도우: train=%d, val=%d, test=%d", len(tr_wins), len(va_wins), len(te_wins))

    gat  = TemporalGAT(N_FEAT, H_GRU, H_GAT, GAT_HEADS, DROPOUT)
    base = BaselineGRU(N_FEAT, H_GRU, DROPOUT)
    log.info("파라미터: GAT=%d, Baseline=%d",
             sum(p.numel() for p in gat.parameters()),
             sum(p.numel() for p in base.parameters()))

    log.info("TemporalGAT 학습...")
    gat  = train_model(gat,  tr_wins, va_wins, ei, ea, "GAT")
    log.info("BaselineGRU 학습...")
    base = train_model(base, tr_wins, va_wins, ei, ea, "GRU")

    metrics = {}
    for name, model in [("TemporalGAT_v2", gat), ("BaselineGRU_v2", base)]:
        rmse, mae = evaluate(model, te_wins, ei, ea)
        ic_df = compute_ic(model, te_wins, ei, ea, months, LOOKBACK, va_end, cids)
        mic   = ic_df["IC"].mean() if len(ic_df) else float('nan')
        icir  = mic/max(ic_df["IC"].std(),1e-6) if len(ic_df)>1 else float('nan')
        dacc  = ic_df["dir_acc"].mean() if len(ic_df) else float('nan')
        metrics[name] = dict(rmse=rmse, mae=mae, mean_ic=mic, icir=icir, dir_acc=dacc)
        log.info("[%s] RMSE=%.4f MAE=%.4f IC=%+.4f ICIR=%.3f Dir=%.1f%%",
                 name, rmse, mae, mic, icir, dacc*100)

    # 어텐션
    attn = extract_attn(gat, te_wins, ei, ea, cids, nm)

    # IC 시계열
    ic_gat  = compute_ic(gat,  te_wins, ei, ea, months, LOOKBACK, va_end, cids)
    ic_base = compute_ic(base, te_wins, ei, ea, months, LOOKBACK, va_end, cids)
    ic_gat["model"]  = "GAT_v2"
    ic_base["model"] = "GRU_v2"

    # 저장
    preds = []
    for split, model, wins, st in [("train",gat,tr_wins,0),
                                    ("val",gat,va_wins,tr_end),
                                    ("test",gat,te_wins,va_end)]:
        model.eval()
        with torch.no_grad():
            for i,(xw,yw,mw) in enumerate(wins):
                p=model(xw,ei,ea).numpy(); t=yw.numpy(); m=mw.numpy().astype(bool)
                mi=st+i+LOOKBACK; ym_str=str(months[mi]) if mi<len(months) else "?"
                for n,cid in enumerate(cids):
                    preds.append(dict(split=split,year_month=ym_str,company_id=cid,
                                      pred=float(p[n]),actual=float(t[n]),valid=bool(m[n])))

    pd.DataFrame(preds).to_parquet(OUT_PRED, index=False)
    if len(attn): attn.to_csv(OUT_ATTN, index=False, encoding="utf-8-sig")
    pd.concat([ic_gat,ic_base]).to_csv(OUT_IC, index=False, encoding="utf-8-sig")

    # 보고서
    lines=[
        "="*68, "Phase 5 GNN v2 — 결과 보고서", "="*68, "",
        f"피처({N_FEAT}개): {FEAT_COLS}", "",
        f"엣지 파일: {PATH_EDGES.name}  (v1 대비 +23엣지, 오류 2건 수정)",
        f"z-score: v2 strict (n_fb>=10) + supply_z + z_cross_all", "",
        "■ 성능 비교 (Test set)",
        f"  {'모델':<22} {'RMSE':>7} {'MAE':>7} {'IC':>8} {'ICIR':>7} {'Dir%':>7}",
        "  "+"-"*56,
    ]
    for mname, m in metrics.items():
        lines.append(
            f"  {mname:<22} {m['rmse']:7.4f} {m['mae']:7.4f} "
            f"{m['mean_ic']:+8.4f} {m['icir']:7.3f} {m['dir_acc']*100:6.1f}%")

    if len(ic_gat):
        lines += ["",
            f"■ GAT_v2 월별 IC (test): mean={ic_gat['IC'].mean():+.3f}  "
            f"std={ic_gat['IC'].std():.3f}  IC>0={( ic_gat['IC']>0).mean():.0%}"]

    if len(attn):
        lines += ["", "■ GAT 어텐션 상위 5 엣지"]
        for _,r in attn.head(5).iterrows():
            lines.append(f"  {r['src_name']:20s} → {r['dst_name']:20s}  {r['attn_mean']:.4f}")

    lines += ["", "="*68]
    rpt = "\n".join(lines)
    OUT_RPT.write_text(rpt, encoding="utf-8")
    print("\n" + rpt)


if __name__ == "__main__":
    main()
