"""
Phase 5 GNN — Dynamic Graph + Binary Classification
====================================================

개선사항 2가지 동시 적용:

[1] Dynamic Graph (시간 가중 그래프)
    - 매 월마다 valid_from <= t <= valid_to 인 엣지만 사용
    - 2017년: 6엣지 / 2020년: 61엣지 / 2024년: 88엣지
    - Look-ahead bias 제거: 미래에 생긴 계약을 과거 예측에 쓰지 않음

[2] Binary Classification (방향 예측)
    - 목표: 다음 달 시장조정 수익률 > 0 이면 1, 아니면 0
    - 손실: BCEWithLogitsLoss (MSE 대비 분류 문제에 적합)
    - 클래스 밸런스: 49% / 51% (가중치 불필요)

평가 지표:
    - Accuracy, F1, AUC-ROC (분류)
    - IC (Spearman ρ: 예측확률 vs 실제 수익률)
    - Monthly IC 시계열

모델:
    DynamicTemporalGAT  : GRU(시계열) → 동적 GAT(공간) → Sigmoid
    BaselineGRU_binary  : GRU(시계열) → MLP → Sigmoid  [그래프 없는 기준선]
"""
from __future__ import annotations
import logging, random, sys, warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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

ROOT = Path(__file__).resolve().parents[1]

PATH_TONE  = ROOT / "data/processed/tone_monthly_zscore_18_v2.parquet"
PATH_PRICES= ROOT / "data/processed/prices_daily.parquet"
PATH_MKT   = ROOT / "data/processed/market_proxies.parquet"
PATH_MKTMAP= ROOT / "data/processed/market_proxy_mapping.yaml"
PATH_EDGES = ROOT / "data/seed/seed_edges_18_internal_v2.csv"
PATH_C18   = ROOT / "data/universe/companies_18.csv"

OUT_PRED   = ROOT / "data/processed/phase5_dynamic_binary_predictions.parquet"
OUT_ATTN   = ROOT / "reports/phase5_dynamic_binary_attention.csv"
OUT_IC     = ROOT / "reports/phase5_dynamic_binary_ic.csv"
OUT_RPT    = ROOT / "reports/phase5_dynamic_binary_summary.txt"

# ── 하이퍼파라미터
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
MIN_VALID = 3      # 윈도우당 최소 유효 노드 수

FEAT_COLS = [
    "z_score_v2", "z_lag1_v2", "z_lag2_v2", "z_lag3",
    "delta_z_v2", "rolling_3m_v2",
    "supply_z", "z_cross_all",
    "regime_covid", "regime_ukraine", "regime_ira", "regime_china_export",
]
N_FEAT = len(FEAT_COLS)


def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


# ─────────────────────────────────────────────────────────────
# 1. 데이터 로딩
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
    irm: Dict[str, pd.DataFrame] = {}
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

    for c in ["regime_covid","regime_ukraine","regime_ira","regime_china_export"]:
        if tone[c].dtype == bool:
            tone[c] = tone[c].astype("float32")

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

    # ── Binary 타겟: 시장조정 수익률 > 0 이면 1
    panel["binary_next"] = (panel["ar_next"] > 0).astype("float32")

    panel["valid"] = (
        panel["ar"].notna() &
        panel["ar_next"].notna() &
        panel["reliable_v2"].fillna(False)
    ).astype("float32")

    months = sorted(panel["ym"].unique())
    log.info("피처 행렬: %d기업, %d월, 유효=%d/%d (%.1f%%)",
             panel["company_id"].nunique(), len(months),
             int(panel["valid"].sum()), len(panel),
             100*panel["valid"].mean())

    pos_rate = panel[panel["valid"]==1]["binary_next"].mean()
    log.info("Binary 클래스 밸런스: 양성(ar>0)=%.1f%%", pos_rate*100)
    return panel, months


def build_tensors(panel, months, cids):
    """X(피처), y_binary(0/1), y_cont(연속 수익률), mask 반환"""
    T, N = len(months), len(cids)
    cid2i = {c:i for i,c in enumerate(cids)}
    ym2t  = {ym:t for t,ym in enumerate(months)}

    X      = np.zeros((T,N,N_FEAT), dtype=np.float32)
    y_bin  = np.zeros((T,N),         dtype=np.float32)  # binary 타겟
    y_cont = np.zeros((T,N),         dtype=np.float32)  # 연속 타겟 (IC 계산용)
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
# 2. 동적 그래프 구축
# ─────────────────────────────────────────────────────────────

def build_dynamic_graphs(
    edges_path: Path,
    cids: List[str],
    months: list,
) -> Dict:
    """
    월별로 유효한 엣지만 포함하는 동적 그래프 딕셔너리 반환.
    edges[t] = (edge_index, edge_attr)

    valid_from <= month <= valid_to (valid_to=NaN이면 계속 유효)
    """
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
            w = max(0.01, min(1.0,
                float(r.get("confidence_plink",0.8)) *
                float(r.get("strength",0.8))))
            ei.extend([[s,d],[d,s]]); ea.extend([[w],[w]])

        n_e = len(ei)
        edge_counts.append(n_e)
        if n_e > 0:
            monthly[ym] = (
                torch.tensor(ei, dtype=torch.long).t().contiguous(),
                torch.tensor(ea, dtype=torch.float32),
            )
        else:
            # 엣지 없는 달: 자기 루프만 (GNN이 최소한 작동하도록)
            self_loop_idx = torch.arange(len(cids))
            monthly[ym] = (
                torch.stack([self_loop_idx, self_loop_idx]),
                torch.ones(len(cids), 1, dtype=torch.float32) * 0.01,
            )

    log.info("동적 그래프: 최소=%d, 최대=%d, 평균=%.0f 엣지/월",
             min(edge_counts), max(edge_counts),
             np.mean(edge_counts))
    return monthly


# ─────────────────────────────────────────────────────────────
# 3. 슬라이딩 윈도우 (월 인덱스 포함)
# ─────────────────────────────────────────────────────────────

def make_windows(
    X: torch.Tensor,
    y_bin: torch.Tensor,
    y_cont: torch.Tensor,
    mask: torch.Tensor,
    lookback: int,
    offset: int = 0,
) -> List[Tuple]:
    """
    Returns: (t_abs, x_win, y_bin_win, y_cont_win, mask_win)
    t_abs: 전체 시계열 기준 절대 인덱스 (동적 엣지 조회용)
    """
    T = X.shape[0]
    windows = []
    for t in range(lookback-1, T):
        m = mask[t]
        if int(m.sum().item()) < MIN_VALID: continue
        windows.append((
            offset + t,               # 전체 기준 절대 인덱스
            X[t-lookback+1:t+1],      # [L, N, F]
            y_bin[t],                 # [N] binary
            y_cont[t],                # [N] continuous (IC용)
            m,                        # [N] mask
        ))
    return windows


# ─────────────────────────────────────────────────────────────
# 4. 모델
# ─────────────────────────────────────────────────────────────

class DynamicTemporalGAT(nn.Module):
    """
    GRU (시계열 임베딩) → 동적 GAT (해당 월 엣지) → Sigmoid (이진 분류)

    forward() 는 정적 엣지 버전과 달리 per-call로 edge_index, edge_attr 받음
    """

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
        """
        x  : [L, N, F]
        ei : [2, E] — 예측 월의 엣지 인덱스 (동적)
        ea : [E, 1] — 엣지 가중치
        """
        _, N, _ = x.shape
        h, _  = self.gru(x.permute(1,0,2))     # [N, L, H]
        h     = self.gru_norm(
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
        pred = self.head(h2).squeeze(-1)    # [N] logits (sigmoid 적용 안 함)

        if return_attn:
            return pred, (ei1, aw1), (ei2, aw2)
        return pred


class BaselineGRU_binary(nn.Module):
    """그래프 없는 기준선 — GRU + MLP"""

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
    """마스킹된 Binary Cross-Entropy"""
    n = mask.sum()
    if n < 1:
        return torch.tensor(0., requires_grad=True)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * mask).sum() / n


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
    """
    Returns: dict with accuracy, f1, auc, ic_mean
    """
    model.eval()
    all_logits, all_true_bin, all_true_cont = [], [], []

    for t_abs, xw, yw_bin, yw_cont, mw in windows:
        ei, ea = dyn_graphs[months[t_abs]]
        logits = model(xw, ei, ea).numpy()
        tb     = yw_bin.numpy()
        tc     = yw_cont.numpy()
        m      = mw.numpy().astype(bool)
        if m.sum() == 0: continue
        all_logits.append(logits[m])
        all_true_bin.append(tb[m])
        all_true_cont.append(tc[m])

    if not all_logits:
        return dict(acc=np.nan, f1=np.nan, auc=np.nan, ic=np.nan)

    lg   = np.concatenate(all_logits)
    tb   = np.concatenate(all_true_bin)
    tc   = np.concatenate(all_true_cont)
    prob = 1 / (1 + np.exp(-lg))       # sigmoid
    pred = (prob > 0.5).astype(float)

    acc  = accuracy_score(tb, pred)
    f1   = f1_score(tb, pred, zero_division=0)
    try:    auc = roc_auc_score(tb, prob)
    except: auc = np.nan
    rho, _ = spearmanr(prob, tc)
    ic = float(rho) if not np.isnan(rho) else np.nan

    return dict(acc=acc, f1=f1, auc=auc, ic=ic)


def train_model(model, tr_wins, va_wins, dyn_graphs, months, name):
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", patience=10, factor=0.5, min_lr=1e-5)  # maximize AUC
    best_val, best_state, no_imp = 0., None, 0

    for ep in range(1, EPOCHS+1):
        train_epoch(model, opt, tr_wins, dyn_graphs, months)
        va_m = evaluate(model, va_wins, dyn_graphs, months)
        val_score = va_m["auc"] if not np.isnan(va_m["auc"]) else 0.

        sch.step(val_score)
        if val_score > best_val + 1e-5:
            best_val = val_score
            best_state = {k:v.clone() for k,v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1

        if ep % 50 == 0:
            log.info("  [%s] ep=%d  val_auc=%.4f  val_acc=%.1f%%  lr=%.6f",
                     name, ep, val_score, va_m["acc"]*100,
                     opt.param_groups[0]["lr"])
        if no_imp >= PATIENCE:
            log.info("  [%s] early stop ep=%d (best_auc=%.4f)", name, ep, best_val)
            break

    if best_state:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def monthly_ic(model, windows, dyn_graphs, months):
    """월별 IC (예측확률 vs 실제 시장조정 수익률) + 월별 정확도"""
    model.eval(); rows = []
    for t_abs, xw, _, yw_cont, mw in windows:
        ei, ea = dyn_graphs[months[t_abs]]
        logits = model(xw, ei, ea).numpy()
        tc     = yw_cont.numpy()
        m      = mw.numpy().astype(bool)
        prob   = 1 / (1 + np.exp(-logits))
        pv, tv = prob[m], tc[m]
        if len(pv) < 3: continue
        rho, _ = spearmanr(pv, tv)
        tb     = (tv > 0).astype(float)
        pred   = (pv > 0.5).astype(float)
        acc    = float(np.mean(pred == tb))
        rows.append(dict(
            year_month = str(months[t_abs]),
            IC         = float(rho) if not np.isnan(rho) else np.nan,
            dir_acc    = acc,
            n_nodes    = int(m.sum()),
            n_edges    = int(ei.shape[1]),
        ))
    return pd.DataFrame(rows)


@torch.no_grad()
def extract_attn(model, windows, dyn_graphs, months, cids, nm):
    model.eval()
    attn_sum: Optional[np.ndarray] = None; cnt = 0

    for t_abs, xw, _, _, _ in windows:
        ei, ea = dyn_graphs[months[t_abs]]
        _, (_, aw1), (_, aw2) = model(xw, ei, ea, return_attn=True)
        a_mean = ((aw1.mean(-1) + aw2.mean(-1)) / 2).numpy()
        ei_np  = ei.numpy()
        if attn_sum is None:
            attn_sum = np.zeros(ei_np.shape[1])
        if len(a_mean) == len(attn_sum):
            attn_sum += a_mean
            cnt += 1

    if cnt == 0 or attn_sum is None:
        return pd.DataFrame()

    attn_mean = attn_sum / cnt
    rows = []
    last_t, last_xw = windows[-1][0], windows[-1][1]
    ei_last, _ = dyn_graphs[months[last_t]]
    ei_np = ei_last.numpy()
    for e in range(min(ei_np.shape[1], len(attn_mean))):
        si, di = ei_np[0,e], ei_np[1,e]
        if si >= len(cids) or di >= len(cids): continue
        rows.append(dict(
            src      = cids[si], src_name = nm.get(cids[si],"?"),
            dst      = cids[di], dst_name = nm.get(cids[di],"?"),
            attn     = float(attn_mean[e]),
        ))

    df = pd.DataFrame(rows)
    if len(df):
        # 중복 제거 (양방향 엣지 → 한 방향만 유지)
        df["pair"] = df.apply(lambda r: tuple(sorted([r["src"],r["dst"]])), axis=1)
        df = df.groupby("pair").agg(
            src_name=("src_name","first"), dst_name=("dst_name","last"),
            attn=("attn","mean")
        ).reset_index(drop=True).sort_values("attn", ascending=False)
    return df


# ─────────────────────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────────────────────

def main():
    set_seed(SEED)
    for p in [OUT_PRED.parent, OUT_RPT.parent]:
        p.mkdir(parents=True, exist_ok=True)

    log.info("Phase 5 Dynamic GAT + Binary Classification 시작")
    log.info("피처(%d): %s", N_FEAT, FEAT_COLS)

    c18  = pd.read_csv(PATH_C18)
    cids = c18["company_id"].tolist()
    nm   = dict(zip(c18["company_id"], c18["canonical_name"]))
    stg  = dict(zip(c18["company_id"], c18["stage_bucket"]))

    ar_df          = load_market_adj(PATH_PRICES, PATH_MKT, PATH_MKTMAP, cids)
    panel, months  = build_feature_matrix(PATH_TONE, ar_df, cids)
    X, y_bin, y_cont, mask = build_tensors(panel, months, cids)
    dyn_graphs     = build_dynamic_graphs(PATH_EDGES, cids, months)

    T = X.shape[0]
    tr_end = int(T * TRAIN_R); va_end = int(T * VAL_R)
    log.info("분할: train=%s~%s (%d월), val=%s~%s (%d월), test=%s~%s (%d월)",
             months[0], months[tr_end-1], tr_end,
             months[tr_end], months[va_end-1], va_end-tr_end,
             months[va_end], months[-1], T-va_end)

    tr_wins = make_windows(X[:tr_end],       y_bin[:tr_end],       y_cont[:tr_end],       mask[:tr_end],       LOOKBACK, 0)
    va_wins = make_windows(X[tr_end:va_end], y_bin[tr_end:va_end], y_cont[tr_end:va_end], mask[tr_end:va_end], LOOKBACK, tr_end)
    te_wins = make_windows(X[va_end:],       y_bin[va_end:],       y_cont[va_end:],       mask[va_end:],       LOOKBACK, va_end)
    log.info("윈도우: train=%d, val=%d, test=%d", len(tr_wins), len(va_wins), len(te_wins))

    gat  = DynamicTemporalGAT(N_FEAT, H_GRU, H_GAT, GAT_HEADS, DROPOUT)
    base = BaselineGRU_binary(N_FEAT, H_GRU, DROPOUT)
    log.info("파라미터: GAT=%d, Baseline=%d",
             sum(p.numel() for p in gat.parameters()),
             sum(p.numel() for p in base.parameters()))

    log.info("Dynamic GAT 학습...")
    gat  = train_model(gat,  tr_wins, va_wins, dyn_graphs, months, "DynGAT")
    log.info("Baseline GRU 학습...")
    base = train_model(base, tr_wins, va_wins, dyn_graphs, months, "GRU")

    # 최종 평가
    metrics = {}
    for mname, model in [("Dynamic GAT", gat), ("Baseline GRU", base)]:
        m = evaluate(model, te_wins, dyn_graphs, months)
        metrics[mname] = m
        log.info("[%s] Acc=%.1f%% F1=%.4f AUC=%.4f IC=%+.4f",
                 mname, m["acc"]*100, m["f1"], m["auc"], m["ic"])

    # IC 시계열
    ic_gat  = monthly_ic(gat,  te_wins, dyn_graphs, months)
    ic_base = monthly_ic(base, te_wins, dyn_graphs, months)
    ic_gat["model"]  = "DynamicGAT"
    ic_base["model"] = "BaselineGRU"

    # 어텐션
    attn = extract_attn(gat, te_wins, dyn_graphs, months, cids, nm)

    # 예측 패널 저장
    preds = []
    for split, model, wins in [("train",gat,tr_wins),("val",gat,va_wins),("test",gat,te_wins)]:
        model.eval()
        with torch.no_grad():
            for t_abs, xw, yw_bin, yw_cont, mw in wins:
                ei, ea = dyn_graphs[months[t_abs]]
                logits = model(xw, ei, ea).numpy()
                prob   = 1/(1+np.exp(-logits))
                tb     = yw_bin.numpy(); tc = yw_cont.numpy(); m = mw.numpy().astype(bool)
                for n, cid in enumerate(cids):
                    preds.append(dict(
                        split=split, year_month=str(months[t_abs]),
                        company_id=cid, prob=float(prob[n]),
                        pred_binary=int(prob[n]>0.5),
                        actual_binary=int(tb[n]),
                        actual_ar=float(tc[n]),
                        valid=bool(m[n]),
                        n_edges_this_month=int(ei.shape[1]),
                    ))

    pd.DataFrame(preds).to_parquet(OUT_PRED, index=False)
    if len(attn): attn.to_csv(OUT_ATTN, index=False, encoding="utf-8-sig")
    pd.concat([ic_gat, ic_base]).to_csv(OUT_IC, index=False, encoding="utf-8-sig")

    # 보고서
    lines = ["="*68, "Phase 5 Dynamic GAT + Binary Classification 결과", "="*68,
             "",
             f"동적 그래프: 2017년 6엣지 → 2024년 88엣지 (월별 유효 엣지 적용)",
             f"Binary 타겟: 시장조정 수익률 > 0 (클래스 49%/51%)",
             f"피처({N_FEAT}개): supply_z, z_cross_all 포함",
             "",
             "■ 성능 비교 (Test set)",
             f"  {'모델':<20} {'Acc':>7} {'F1':>7} {'AUC':>7} {'IC':>8}",
             "  "+"-"*48,
             ]
    for mname, m in metrics.items():
        lines.append(f"  {mname:<20} {m['acc']*100:6.1f}% {m['f1']:7.4f} "
                     f"{m['auc']:7.4f} {m['ic']:+8.4f}")

    if len(ic_gat) > 0:
        lines += ["",
                  f"■ Dynamic GAT IC 요약 (test): "
                  f"mean={ic_gat['IC'].mean():+.3f}  "
                  f"std={ic_gat['IC'].std():.3f}  "
                  f"IC>0 비율={( ic_gat['IC']>0).mean():.0%}",
                  f"  Dir_Acc: mean={ic_gat['dir_acc'].mean():.1%}  "
                  f"max={ic_gat['dir_acc'].max():.1%}",
                  "",
                  "  월별 상세:"]
        for _, r in ic_gat.iterrows():
            lines.append(f"    {r['year_month']}  IC={r['IC']:+.3f}  "
                         f"acc={r['dir_acc']:.1%}  N={r['n_nodes']}  "
                         f"엣지={r['n_edges']}")

    if len(attn) > 0:
        lines += ["", "■ GAT 어텐션 상위 10 엣지 (테스트 기간, 중복 제거)"]
        for _, r in attn.head(10).iterrows():
            lines.append(f"  {r['src_name']:20s} ↔ {r['dst_name']:20s}  {r['attn']:.4f}")

    lines += ["", "="*68]
    rpt = "\n".join(lines)
    OUT_RPT.write_text(rpt, encoding="utf-8")
    print("\n" + rpt)


if __name__ == "__main__":
    main()
