"""
Phase 5 GNN — Ablation Study (3-way 비교)
==========================================

목적: "그래프 구조 자체가 실제로 기여하는가" 검증

① Real Graph GAT   — 기존 seed_edges_18_internal_v3.csv 기반 동적 그래프
② Random Graph GAT — 동일 월별 평균 엣지 수, 랜덤 연결
③ No Graph GRU     — 그래프 없는 순수 시계열 baseline

phase5_dynamic_final.py 의 모든 로직을 그대로 사용하며
3개 모델 학습 · 평가 · 보고서 출력만 추가.
"""
from __future__ import annotations
import logging, random, sys, warnings
from pathlib import Path
from typing import Dict, List, Tuple

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

ROOT     = Path(__file__).resolve().parents[3]   # nabi_hyoghaw/
EXP      = ROOT / "experiments/universe_44"

PATH_TONE  = EXP  / "data/processed/tone_monthly_zscore_all44.parquet"
PATH_PRICES= ROOT / "data/processed/prices_daily.parquet"
PATH_MKT   = ROOT / "data/processed/market_proxies.parquet"
PATH_MKTMAP= ROOT / "data/processed/market_proxy_mapping.yaml"
PATH_EDGES = EXP  / "data/seed/seed_edges_44.csv"
PATH_C18   = EXP  / "data/universe/companies_44.csv"

OUT_RPT    = EXP  / "reports/phase5_ablation_44.txt"

# ── 하이퍼파라미터 (phase5_dynamic_final.py 개선 설정과 동일)
LOOKBACK  = 3
H_GRU     = 32
H_GAT     = 16
GAT_HEADS = 4
DROPOUT   = 0.1
LR        = 1e-3
WD        = 1e-5
EPOCHS    = 400
PATIENCE  = 60
SEED      = 42
TRAIN_END = "2022-06"   # val 22 windows
VAL_END   = "2024-06"
MIN_VALID = 3
DYN_ALPHA = 0.3

FEAT_COLS = [
    "z_score_v3", "z_lag1_v3", "z_lag2_v3", "delta_z_v3",
    "rolling_3m_v3", "supply_z_v3", "z_cross_v3", "z_cross_stage",
    "regime_covid", "regime_ukraine", "regime_ira",
    "regime_china_export", "regime_trump_tariff",
]
N_FEAT = len(FEAT_COLS)


def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


# ─────────────────────────────────────────────────────────────
# 1. 데이터 로딩 (phase5_dynamic_final.py 동일)
# ─────────────────────────────────────────────────────────────

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
    irm: Dict = {}
    for idx, g in idx_last.groupby(idx_col):
        s = g.sort_values("ym").copy()
        s["mkt_ret"] = s["adj_close"].pct_change()
        irm[str(idx)] = s[["ym","mkt_ret"]].dropna()

    with open(mmap_path) as f:
        c2i = yaml.safe_load(f).get("company_to_market_proxy", {})

    rows = []
    for cid in cids:
        iname = c2i.get(cid)
        if not iname or iname not in irm: continue
        ci = co_ret[co_ret["company_id"]==cid][["ym","ret"]]
        m  = ci.merge(irm[iname], on="ym", how="inner")
        m["ar"] = m["ret"] - m["mkt_ret"]
        m["company_id"] = cid
        rows.append(m[["company_id","ym","ar","ret","mkt_ret"]])
    return pd.concat(rows, ignore_index=True)


def build_feature_matrix(tone_path, ar_df, cids):
    tone = pd.read_parquet(tone_path)
    tone["ym"] = pd.to_datetime(tone["year_month"]).dt.to_period("M")
    tone = tone[tone["company_id"].isin(cids)].copy()

    for c in ["regime_covid","regime_ukraine","regime_ira",
              "regime_china_export","regime_trump_tariff"]:
        if c in tone.columns and tone[c].dtype == bool:
            tone[c] = tone[c].astype("float32")

    missing = [c for c in FEAT_COLS if c not in tone.columns]
    if missing:
        log.warning("누락 피처 (0으로 대체): %s", missing)
        for c in missing:
            tone[c] = 0.0

    reliable_col = "reliable_v3" if "reliable_v3" in tone.columns else "reliable_v2"
    panel = tone[["company_id","ym", reliable_col] + FEAT_COLS].merge(
        ar_df[["company_id","ym","ar"]], on=["company_id","ym"], how="left")
    panel = panel.sort_values(["company_id","ym"])

    Z_COLS = [c for c in FEAT_COLS if "z" in c.lower()]
    for c in FEAT_COLS:
        panel[c] = panel.groupby("company_id")[c].transform(
            lambda s: s.ffill().bfill().fillna(0.0))
    for c in Z_COLS:
        panel[c] = panel[c].clip(-5.0, 5.0)

    panel["ar_next"]     = panel.groupby("company_id")["ar"].shift(-1)
    panel["binary_next"] = (panel["ar_next"] > 0).astype("float32")
    panel["valid"] = (
        panel["ar"].notna() &
        panel["ar_next"].notna() &
        panel[reliable_col].fillna(False)
    ).astype("float32")

    months  = sorted(panel["ym"].unique())
    pos_rate = panel[panel["valid"]==1]["binary_next"].mean()
    log.info("피처 행렬: %d기업, %d월, 유효=%d/%d (%.1f%%), Binary 양성=%.1f%%",
             panel["company_id"].nunique(), len(months),
             int(panel["valid"].sum()), len(panel),
             100*panel["valid"].mean(), pos_rate*100)
    return panel, months


def build_tensors(panel, months, cids):
    T, N  = len(months), len(cids)
    cid2i = {c:i for i,c in enumerate(cids)}
    ym2t  = {ym:t for t,ym in enumerate(months)}

    X      = np.zeros((T,N,N_FEAT), dtype=np.float32)
    y_bin  = np.zeros((T,N),         dtype=np.float32)
    y_cont = np.zeros((T,N),         dtype=np.float32)
    mask   = np.zeros((T,N),         dtype=np.float32)

    for _, row in panel.iterrows():
        t = ym2t.get(row["ym"]); n = cid2i.get(row["company_id"])
        if t is None or n is None: continue
        X[t,n,:] = [float(row[c]) if pd.notna(row.get(c,np.nan)) else 0.0
                    for c in FEAT_COLS]
        if float(row.get("valid",0)) > 0:
            y_bin[t,n]  = float(row["binary_next"])
            y_cont[t,n] = float(row["ar_next"])
            mask[t,n]   = 1.0

    return (torch.tensor(X,     dtype=torch.float32),
            torch.tensor(y_bin, dtype=torch.float32),
            torch.tensor(y_cont,dtype=torch.float32),
            torch.tensor(mask,  dtype=torch.float32))


# ─────────────────────────────────────────────────────────────
# 2. 그래프 구축
# ─────────────────────────────────────────────────────────────

def build_supply_lookup(panel: pd.DataFrame) -> Dict:
    col = "supply_z_v3" if "supply_z_v3" in panel.columns else "supply_z"
    if col not in panel.columns:
        return {}
    return {
        (row["company_id"], row["ym"]): float(row[col])
        for _, row in panel[["company_id","ym", col]].dropna().iterrows()
    }


def dynamic_edge_weight(base_w, src_z, dst_z, alpha=DYN_ALPHA):
    stress = (abs(src_z) + abs(dst_z)) / 2.0
    factor = 1.0 + alpha * float(np.tanh(stress))
    return float(max(0.01, min(1.0, base_w * factor)))


def build_dynamic_graphs(edges_path, cids, months, supply_lookup):
    edges = pd.read_csv(edges_path)
    edges["valid_from"] = pd.to_datetime(edges["valid_from"], errors="coerce")
    edges["valid_to"]   = pd.to_datetime(edges["valid_to"],   errors="coerce")

    ci = {c:i for i,c in enumerate(cids)}
    monthly: Dict = {}
    edge_counts = []

    for ym in months:
        t_stamp = ym.to_timestamp()
        valid_mask = (
            edges["valid_from"].notna() &
            (edges["valid_from"] <= t_stamp) &
            (edges["valid_to"].isna() | (edges["valid_to"] >= t_stamp))
        )
        valid_edges = edges[valid_mask]

        ei, ea = [], []
        for _, r in valid_edges.iterrows():
            s = ci.get(r["src_company_id"]); d = ci.get(r["dst_company_id"])
            if s is None or d is None: continue
            base_w = max(0.01, min(1.0,
                float(r.get("confidence_plink", 0.8)) *
                float(r.get("strength", 0.8))))
            src_z = supply_lookup.get((r["src_company_id"], ym), 0.0)
            dst_z = supply_lookup.get((r["dst_company_id"], ym), 0.0)
            w = dynamic_edge_weight(base_w, src_z, dst_z)
            ei.extend([[s,d],[d,s]])
            ea.extend([[w],[w]])

        n_e = len(ei)
        edge_counts.append(n_e)
        if n_e > 0:
            monthly[ym] = (
                torch.tensor(ei, dtype=torch.long).t().contiguous(),
                torch.tensor(ea, dtype=torch.float32),
            )
        else:
            sl = torch.arange(len(cids))
            monthly[ym] = (
                torch.stack([sl, sl]),
                torch.ones(len(cids), 1, dtype=torch.float32) * 0.01,
            )

    log.info("Real 그래프: 최소=%d, 최대=%d, 평균=%.0f 엣지/월",
             min(edge_counts), max(edge_counts), np.mean(edge_counts))
    return monthly


def build_random_graphs(n_nodes: int, months: list, dyn_graphs_real: Dict) -> Dict:
    """
    실제 그래프와 동일한 월별 평균 엣지 수를 가진 랜덤 그래프.
    매 호출마다 새 시드(99)로 생성해 단일 랜덤 시드 편향 방지.
    """
    rng = np.random.default_rng(99)
    monthly: Dict = {}

    for ym in months:
        real_ei, _ = dyn_graphs_real[ym]
        # real_ei 에는 양방향 쌍이 이미 포함됨 → 단방향 엣지 수 복원
        n_real_edges = real_ei.shape[1] // 2
        n_real_edges = max(n_real_edges, 5)

        pairs: set = set()
        attempts = 0
        while len(pairs) < n_real_edges and attempts < 10000:
            s = int(rng.integers(0, n_nodes))
            d = int(rng.integers(0, n_nodes))
            if s != d:
                pairs.add((s, d))
            attempts += 1

        if not pairs:
            sl = torch.arange(n_nodes)
            monthly[ym] = (
                torch.stack([sl, sl]),
                torch.ones(n_nodes, 1) * 0.01,
            )
            continue

        # 무향 그래프로 만들기 위해 양방향 추가
        ei = torch.tensor(
            list(pairs) + [(d, s) for s, d in pairs],
            dtype=torch.long,
        ).t().contiguous()
        ea = torch.ones(ei.shape[1], 1, dtype=torch.float32) * 0.5
        monthly[ym] = (ei, ea)

    edge_counts = [dyn_graphs_real[ym][0].shape[1] for ym in months]
    rand_counts  = [monthly[ym][0].shape[1] for ym in months]
    log.info("Random 그래프: 최소=%d, 최대=%d, 평균=%.0f 엣지/월 (Real: 평균=%.0f)",
             min(rand_counts), max(rand_counts), np.mean(rand_counts),
             np.mean(edge_counts))
    return monthly


# ─────────────────────────────────────────────────────────────
# 3. 슬라이딩 윈도우
# ─────────────────────────────────────────────────────────────

def make_windows(X, y_bin, y_cont, mask, lookback, offset=0):
    T = X.shape[0]
    windows = []
    for t in range(lookback-1, T):
        m = mask[t]
        if int(m.sum().item()) < MIN_VALID: continue
        windows.append((
            offset + t,
            X[t-lookback+1:t+1],
            y_bin[t],
            y_cont[t],
            m,
        ))
    return windows


# ─────────────────────────────────────────────────────────────
# 4. 모델
# ─────────────────────────────────────────────────────────────

class DynamicTemporalGAT(nn.Module):
    def __init__(self, in_ch, h_gru, h_gat, heads, drop):
        super().__init__()
        self.drop = drop
        self.gru      = nn.GRU(in_ch, h_gru, batch_first=True)
        self.gru_norm = nn.LayerNorm(h_gru)
        self.gat1     = GATConv(h_gru, h_gat, heads=heads, concat=True,
                                dropout=drop, edge_dim=1, add_self_loops=True)
        self.gat1_norm= nn.LayerNorm(h_gat * heads)
        self.gat2     = GATConv(h_gat*heads, h_gat, heads=1, concat=False,
                                dropout=drop, edge_dim=1, add_self_loops=True)
        self.gat2_norm= nn.LayerNorm(h_gat)
        self.head = nn.Sequential(
            nn.Linear(h_gat, h_gat//2), nn.ELU(),
            nn.Dropout(drop), nn.Linear(h_gat//2, 1),
        )

    def forward(self, x, ei, ea, return_attn=False):
        h, _ = self.gru(x.permute(1,0,2))
        h    = self.gru_norm(
                   F.dropout(h[:,-1,:], p=self.drop, training=self.training))
        if return_attn:
            h1, (ei1, aw1) = self.gat1(h, ei, ea, return_attention_weights=True)
        else:
            h1 = self.gat1(h, ei, ea)
        h1 = self.gat1_norm(F.elu(h1))
        h1 = F.dropout(h1, p=self.drop, training=self.training)
        if return_attn:
            h2, (ei2, aw2) = self.gat2(h1, ei, ea, return_attention_weights=True)
        else:
            h2 = self.gat2(h1, ei, ea)
        h2   = self.gat2_norm(F.elu(h2))
        pred = self.head(h2).squeeze(-1)
        if return_attn:
            return pred, (ei1, aw1), (ei2, aw2)
        return pred


class BaselineGRU(nn.Module):
    def __init__(self, in_ch, h_gru, drop):
        super().__init__()
        self.drop = drop
        self.gru  = nn.GRU(in_ch, h_gru, batch_first=True)
        self.norm = nn.LayerNorm(h_gru)
        self.mlp  = nn.Sequential(
            nn.Linear(h_gru, h_gru//2), nn.ELU(),
            nn.Dropout(drop), nn.Linear(h_gru//2, 1),
        )

    def forward(self, x, ei, ea, return_attn=False):
        h, _ = self.gru(x.permute(1,0,2))
        h    = self.norm(F.dropout(h[:,-1,:], p=self.drop, training=self.training))
        return self.mlp(h).squeeze(-1)


# ─────────────────────────────────────────────────────────────
# 5. 학습 & 평가
# ─────────────────────────────────────────────────────────────

def masked_bce(logits, target, mask):
    n = mask.sum()
    if n < 1:
        return torch.tensor(0., requires_grad=True)
    return (F.binary_cross_entropy_with_logits(logits, target,
                                               reduction="none") * mask).sum() / n


def train_epoch(model, opt, windows, dyn_graphs, months):
    model.train(); total = 0; cnt = 0
    for t_abs, xw, yw_bin, _, mw in windows:
        ei, ea = dyn_graphs[months[t_abs]]
        opt.zero_grad()
        logits = model(xw, ei, ea)
        loss   = masked_bce(logits, yw_bin, mw)
        if loss.item() == 0: continue
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item(); cnt += 1
    return total / max(1, cnt)


@torch.no_grad()
def evaluate(model, windows, dyn_graphs, months):
    model.eval()
    all_logits, all_true_bin, all_true_cont = [], [], []
    for t_abs, xw, yw_bin, yw_cont, mw in windows:
        ei, ea = dyn_graphs[months[t_abs]]
        logits = model(xw, ei, ea).numpy()
        m      = mw.numpy().astype(bool)
        if m.sum() == 0: continue
        all_logits.append(logits[m])
        all_true_bin.append(yw_bin.numpy()[m])
        all_true_cont.append(yw_cont.numpy()[m])

    if not all_logits:
        return dict(acc=np.nan, f1=np.nan, auc=np.nan, ic=np.nan)

    lg   = np.concatenate(all_logits)
    tb   = np.concatenate(all_true_bin)
    tc   = np.concatenate(all_true_cont)
    prob = 1 / (1 + np.exp(-lg))
    pred = (prob > 0.5).astype(float)

    acc = accuracy_score(tb, pred)
    f1  = f1_score(tb, pred, zero_division=0)
    try:    auc = roc_auc_score(tb, prob)
    except: auc = np.nan
    rho, _ = spearmanr(prob, tc)
    ic = float(rho) if not np.isnan(rho) else np.nan
    return dict(acc=acc, f1=f1, auc=auc, ic=ic)


@torch.no_grad()
def monthly_ic(model, windows, dyn_graphs, months):
    model.eval(); rows = []
    for t_abs, xw, _, yw_cont, mw in windows:
        ei, ea = dyn_graphs[months[t_abs]]
        logits = model(xw, ei, ea).numpy()
        m      = mw.numpy().astype(bool)
        prob   = 1 / (1 + np.exp(-logits))
        pv, tv = prob[m], yw_cont.numpy()[m]
        if len(pv) < 3: continue
        rho, _ = spearmanr(pv, tv)
        rows.append(dict(
            year_month = str(months[t_abs]),
            IC         = float(rho) if not np.isnan(rho) else np.nan,
            n_nodes    = int(m.sum()),
        ))
    return pd.DataFrame(rows)


def train_model(model, tr_wins, va_wins, dyn_graphs, months, name):
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", patience=10, factor=0.5, min_lr=1e-5)
    best_val, best_state, no_imp = 0., None, 0

    for ep in range(1, EPOCHS+1):
        train_epoch(model, opt, tr_wins, dyn_graphs, months)
        va_m = evaluate(model, va_wins, dyn_graphs, months)
        val_score = va_m["auc"] if not np.isnan(va_m["auc"]) else 0.

        sch.step(val_score)
        if val_score > best_val + 1e-5:
            best_val  = val_score
            best_state = {k:v.clone() for k,v in model.state_dict().items()}
            no_imp    = 0
        else:
            no_imp += 1

        if ep % 50 == 0:
            log.info("  [%s] ep=%d  val_auc=%.4f  lr=%.6f",
                     name, ep, val_score, opt.param_groups[0]["lr"])
        if no_imp >= PATIENCE:
            log.info("  [%s] early stop ep=%d (best_auc=%.4f)", name, ep, best_val)
            break

    if best_state:
        model.load_state_dict(best_state)
    return model


# ─────────────────────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────────────────────

def main():
    set_seed(SEED)
    OUT_RPT.parent.mkdir(parents=True, exist_ok=True)

    log.info("=== Phase 5 Ablation Study 시작 ===")
    log.info("LOOKBACK=%d, DROPOUT=%.2f, WD=%.0e, PATIENCE=%d, SEED=%d",
             LOOKBACK, DROPOUT, WD, PATIENCE, SEED)

    c18  = pd.read_csv(PATH_C18)
    cids = c18["company_id"].tolist()
    n_nodes = len(cids)

    ar_df         = load_market_adj(PATH_PRICES, PATH_MKT, PATH_MKTMAP, cids)
    panel, months = build_feature_matrix(PATH_TONE, ar_df, cids)
    supply_lookup = build_supply_lookup(panel)
    X, y_bin, y_cont, mask = build_tensors(panel, months, cids)
    dyn_graphs_real = build_dynamic_graphs(PATH_EDGES, cids, months, supply_lookup)

    T = X.shape[0]
    months_str = [str(m) for m in months]
    tr_end = next((i for i, m in enumerate(months_str) if m > TRAIN_END), T)
    va_end = next((i for i, m in enumerate(months_str) if m > VAL_END),   T)
    log.info("분할: train=%s~%s (%d월), val=%s~%s (%d월), test=%s~%s (%d월)",
             months[0], months[tr_end-1], tr_end,
             months[tr_end], months[va_end-1], va_end-tr_end,
             months[va_end], months[-1], T-va_end)

    tr_wins = make_windows(X[:tr_end],       y_bin[:tr_end],       y_cont[:tr_end],       mask[:tr_end],       LOOKBACK, 0)
    va_wins = make_windows(X[tr_end:va_end], y_bin[tr_end:va_end], y_cont[tr_end:va_end], mask[tr_end:va_end], LOOKBACK, tr_end)
    te_wins = make_windows(X[va_end:],       y_bin[va_end:],       y_cont[va_end:],       mask[va_end:],       LOOKBACK, va_end)
    log.info("윈도우: train=%d, val=%d, test=%d", len(tr_wins), len(va_wins), len(te_wins))

    results: Dict[str, dict] = {}
    ic_tables: Dict[str, pd.DataFrame] = {}

    # ① Real Graph GAT
    log.info("\n[1/3] Real Graph GAT 학습...")
    set_seed(SEED)
    gat_real = DynamicTemporalGAT(N_FEAT, H_GRU, H_GAT, GAT_HEADS, DROPOUT)
    gat_real = train_model(gat_real, tr_wins, va_wins, dyn_graphs_real, months, "RealGAT")
    results["Real Graph GAT"] = evaluate(gat_real, te_wins, dyn_graphs_real, months)
    ic_tables["Real Graph GAT"] = monthly_ic(gat_real, te_wins, dyn_graphs_real, months)
    log.info("[RealGAT] AUC=%.4f  F1=%.4f  IC=%+.4f",
             results["Real Graph GAT"]["auc"],
             results["Real Graph GAT"]["f1"],
             results["Real Graph GAT"]["ic"])

    # ② Random Graph GAT
    log.info("\n[2/3] Random Graph GAT 학습...")
    set_seed(SEED)
    dyn_graphs_rand = build_random_graphs(n_nodes, months, dyn_graphs_real)
    gat_rand = DynamicTemporalGAT(N_FEAT, H_GRU, H_GAT, GAT_HEADS, DROPOUT)
    gat_rand = train_model(gat_rand, tr_wins, va_wins, dyn_graphs_rand, months, "RandGAT")
    results["Random Graph GAT"] = evaluate(gat_rand, te_wins, dyn_graphs_rand, months)
    ic_tables["Random Graph GAT"] = monthly_ic(gat_rand, te_wins, dyn_graphs_rand, months)
    log.info("[RandGAT] AUC=%.4f  F1=%.4f  IC=%+.4f",
             results["Random Graph GAT"]["auc"],
             results["Random Graph GAT"]["f1"],
             results["Random Graph GAT"]["ic"])

    # ③ No Graph GRU
    log.info("\n[3/3] No Graph GRU (baseline) 학습...")
    set_seed(SEED)
    base = BaselineGRU(N_FEAT, H_GRU, DROPOUT)
    base = train_model(base, tr_wins, va_wins, dyn_graphs_real, months, "NoGraphGRU")
    results["No Graph GRU"] = evaluate(base, te_wins, dyn_graphs_real, months)
    ic_tables["No Graph GRU"] = monthly_ic(base, te_wins, dyn_graphs_real, months)
    log.info("[NoGraphGRU] AUC=%.4f  F1=%.4f  IC=%+.4f",
             results["No Graph GRU"]["auc"],
             results["No Graph GRU"]["f1"],
             results["No Graph GRU"]["ic"])

    # ── 보고서 작성
    real_m = results["Real Graph GAT"]
    rand_m = results["Random Graph GAT"]
    gru_m  = results["No Graph GRU"]

    real_ic_df = ic_tables["Real Graph GAT"]
    rand_ic_df = ic_tables["Random Graph GAT"]
    gru_ic_df  = ic_tables["No Graph GRU"]

    def ic_summary(df):
        if df.empty:
            return "mean=N/A  IC>0=N/A"
        mean_ic = df["IC"].mean()
        pct_pos = (df["IC"] > 0).mean()
        return f"mean={mean_ic:+.3f}  IC>0={pct_pos:.0%}"

    lines = [
        "=" * 72,
        "■ Ablation Study — 그래프 구조 기여도 검증",
        "=" * 72,
        "",
        f"  설정: LOOKBACK={LOOKBACK}, DROPOUT={DROPOUT}, WD={WD:.0e},"
        f" PATIENCE={PATIENCE}, SEED={SEED}",
        f"  데이터: {n_nodes}개 기업, {len(months)}개월",
        f"  Train: {months[0]}~{months[tr_end-1]} ({tr_end}월)",
        f"  Val  : {months[tr_end]}~{months[va_end-1]} ({va_end-tr_end}월)",
        f"  Test : {months[va_end]}~{months[-1]} ({T-va_end}월)",
        "",
        "  모델                   AUC      F1     Acc     IC월평균  IC>0%",
        "  " + "-" * 65,
    ]

    model_order = ["Real Graph GAT", "Random Graph GAT", "No Graph GRU"]
    for mname in model_order:
        m  = results[mname]
        ic_df = ic_tables[mname]
        ic_mean = ic_df["IC"].mean() if not ic_df.empty else float("nan")
        ic_pos  = (ic_df["IC"] > 0).mean() if not ic_df.empty else float("nan")
        lines.append(
            f"  {mname:<22}  {m['auc']:6.4f}  {m['f1']:6.4f}"
            f"  {m['acc']*100:5.1f}%  {ic_mean:+7.3f}    {ic_pos:5.0%}"
        )

    lines += [""]

    # 해석
    real_auc  = real_m["auc"]
    rand_auc  = rand_m["auc"]
    gru_auc   = gru_m["auc"]
    real_gt_rand = (real_auc > rand_auc) if not (np.isnan(real_auc) or np.isnan(rand_auc)) else None
    real_gt_gru  = (real_auc > gru_auc)  if not (np.isnan(real_auc) or np.isnan(gru_auc))  else None

    lines += [
        "  해석:",
        f"  Real({real_auc:.4f}) vs Random({rand_auc:.4f}) → "
        + ("Real > Random ✓ 그래프 연결 구조 자체가 기여" if real_gt_rand
           else "Real ≤ Random  그래프 구조 기여 불분명"),
        f"  Real({real_auc:.4f}) vs GRU({gru_auc:.4f})    → "
        + ("Real > GRU ✓ 그래프 사용이 순수 시계열보다 우월" if real_gt_gru
           else "Real ≤ GRU   그래프 추가 효과 불분명"),
        "",
        "  월별 IC 요약:",
        f"  Real Graph GAT  : {ic_summary(real_ic_df)}",
        f"  Random Graph GAT: {ic_summary(rand_ic_df)}",
        f"  No Graph GRU    : {ic_summary(gru_ic_df)}",
    ]

    # 월별 IC 상세 (Real GAT)
    if not real_ic_df.empty:
        lines += ["", "  Real Graph GAT 월별 IC (test):"]
        for _, r in real_ic_df.iterrows():
            lines.append(f"    {r['year_month']}  IC={r['IC']:+.3f}  N={r['n_nodes']}")

    lines += ["", "=" * 72]
    rpt = "\n".join(lines)
    OUT_RPT.write_text(rpt, encoding="utf-8")
    print("\n" + rpt)
    log.info("보고서 저장: %s", OUT_RPT)


if __name__ == "__main__":
    main()
