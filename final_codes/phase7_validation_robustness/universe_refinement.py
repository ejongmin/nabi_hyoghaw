"""
universe_refinement.py
======================
분석 1: 40개 기업 vs 44개 기업 GNN 성능 비교
분석 2: reliable 임계값 민감도 (n_fb2 기준 5, 8, 10, 15)
분석 3: KG 엣지 Temporal Validity 검증

출력:
  experiments/hypothesis/walk_forward_cv/reports/universe_refinement.txt
  experiments/hypothesis/walk_forward_cv/reports/kg_temporal_check.txt
"""
from __future__ import annotations
import logging, random, sys, warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import GATConv

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── 경로 ────────────────────────────────────────────────────────
ROOT = Path("C:/Users/john9/nabi_hyoghaw")
EXP  = ROOT / "experiments/universe_44"

PATH_TONE   = EXP  / "data/processed/tone_monthly_zscore_all44_v2.parquet"
PATH_PRICES = ROOT / "data/processed/prices_daily.parquet"
PATH_MKT    = ROOT / "data/processed/market_proxies.parquet"
PATH_MKTMAP = ROOT / "data/processed/market_proxy_mapping.yaml"
PATH_EDGES  = EXP  / "data/seed/seed_edges_44.csv"
PATH_C44    = EXP  / "data/universe/companies_44.csv"

OUT_DIR     = ROOT / "experiments/hypothesis/walk_forward_cv/reports"
OUT_REFINE  = OUT_DIR / "universe_refinement.txt"
OUT_KG      = OUT_DIR / "kg_temporal_check.txt"

# 취약 4개 기업 (reliable_v3=0)
WEAK_4 = ["300037.SZ", "600110.SH", "603659.SH", "688388.SH"]

# ── 하이퍼파라미터 (04_phase5_gnn_44.py 와 동일) ──────────────
LOOKBACK  = 3
H_GRU     = 32
H_GAT     = 16
GAT_HEADS = 4
DROPOUT   = 0.1
LR        = 1e-3
WD        = 1e-5
EPOCHS    = 400
PATIENCE_FULL = 60   # 분석 1
PATIENCE_SENS = 30   # 분석 2 (시간 절약)
SEED      = 42
TRAIN_END = "2022-06"
VAL_END   = "2024-06"
MIN_VALID = 3
DYN_ALPHA = 0.3

FEAT_COLS = [
    "z_score_v3","z_lag1_v3","z_lag2_v3","delta_z_v3","rolling_3m_v3",
    "supply_z_v3","z_cross_v3","z_cross_stage",
    "regime_covid","regime_ukraine","regime_ira",
    "regime_china_export","regime_trump_tariff",
]
N_FEAT = len(FEAT_COLS)

# ────────────────────────────────────────────────────────────────
# 유틸
# ────────────────────────────────────────────────────────────────
def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


# ────────────────────────────────────────────────────────────────
# 데이터 로딩 (04_phase5_gnn_44.py 에서 그대로 복사)
# ────────────────────────────────────────────────────────────────
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
        if not iname or iname not in irm:
            continue
        ci = co_ret[co_ret["company_id"]==cid][["ym","ret"]]
        m  = ci.merge(irm[iname], on="ym", how="inner")
        m["ar"] = m["ret"] - m["mkt_ret"]
        m["company_id"] = cid
        rows.append(m[["company_id","ym","ar","ret","mkt_ret"]])
    return pd.concat(rows, ignore_index=True)


def build_feature_matrix(tone_path, ar_df, cids, fb2_threshold: int = None):
    """
    fb2_threshold: None이면 원본 reliable_v3 사용.
                  정수이면 n_articles_v3 기준으로 재계산.
    """
    tone = pd.read_parquet(tone_path)
    tone["ym"] = pd.to_datetime(tone["year_month"]).dt.to_period("M")
    tone = tone[tone["company_id"].isin(cids)].copy()

    for c in ["regime_covid","regime_ukraine","regime_ira",
              "regime_china_export","regime_trump_tariff"]:
        if c in tone.columns and tone[c].dtype == bool:
            tone[c] = tone[c].astype("float32")

    # reliable 재계산 (민감도 분석용)
    if fb2_threshold is not None:
        # n_fb2 컬럼 우선 탐색
        n_col_local = None
        for candidate in ["n_fb2", "n_articles_v3", "n_fb2_v3", "n_articles"]:
            if candidate in tone.columns:
                n_col_local = candidate
                break
        if n_col_local is not None:
            tone["reliable_v3"] = (tone[n_col_local].fillna(0) >= fb2_threshold)
            log.info("reliable_v3 재계산: %s >= %d → reliable=%d/%d",
                     n_col_local, fb2_threshold,
                     int(tone["reliable_v3"].sum()), len(tone))
        else:
            log.warning("n_fb2/n_articles 컬럼 없음 — 원본 reliable_v3 사용")

    available = [c for c in FEAT_COLS if c in tone.columns]
    missing   = [c for c in FEAT_COLS if c not in tone.columns]
    if missing:
        for c in missing:
            tone[c] = 0.0

    reliable_col = "reliable_v3" if "reliable_v3" in tone.columns else "reliable_v2"
    panel = tone[["company_id","ym",reliable_col]+FEAT_COLS].merge(
        ar_df[["company_id","ym","ar"]], on=["company_id","ym"], how="left"
    )

    panel = panel.sort_values(["company_id","ym"])
    Z_COLS = [c for c in FEAT_COLS if "z" in c.lower()]
    for c in FEAT_COLS:
        panel[c] = panel.groupby("company_id")[c].transform(
            lambda s: s.ffill().bfill().fillna(0.0)
        )
    for c in Z_COLS:
        panel[c] = panel[c].clip(-5.0, 5.0)

    panel["ar_next"]     = panel.groupby("company_id")["ar"].shift(-1)
    panel["binary_next"] = (panel["ar_next"] > 0).astype("float32")
    panel["valid"] = (
        panel["ar"].notna() &
        panel["ar_next"].notna() &
        panel[reliable_col].fillna(False)
    ).astype("float32")

    months = sorted(panel["ym"].unique())
    return panel, months


def build_tensors(panel, months, cids):
    T, N  = len(months), len(cids)
    cid2i = {c: i for i, c in enumerate(cids)}
    ym2t  = {ym: t for t, ym in enumerate(months)}

    X      = np.zeros((T, N, N_FEAT), dtype=np.float32)
    y_bin  = np.zeros((T, N),         dtype=np.float32)
    y_cont = np.zeros((T, N),         dtype=np.float32)
    mask   = np.zeros((T, N),         dtype=np.float32)

    for _, row in panel.iterrows():
        t = ym2t.get(row["ym"])
        n = cid2i.get(row["company_id"])
        if t is None or n is None:
            continue
        X[t, n, :] = [
            float(row[c]) if pd.notna(row.get(c, np.nan)) else 0.0
            for c in FEAT_COLS
        ]
        if float(row.get("valid", 0)) > 0:
            y_bin[t, n]  = float(row["binary_next"])
            y_cont[t, n] = float(row["ar_next"])
            mask[t, n]   = 1.0

    return (
        torch.tensor(X,      dtype=torch.float32),
        torch.tensor(y_bin,  dtype=torch.float32),
        torch.tensor(y_cont, dtype=torch.float32),
        torch.tensor(mask,   dtype=torch.float32),
    )


def build_supply_lookup(panel):
    col = "supply_z_v3" if "supply_z_v3" in panel.columns else "supply_z"
    if col not in panel.columns:
        return {}
    return {
        (row["company_id"], row["ym"]): float(row[col])
        for _, row in panel[["company_id","ym",col]].dropna().iterrows()
    }


def dynamic_edge_weight(base_w, src_z, dst_z, alpha=DYN_ALPHA):
    stress = (abs(src_z) + abs(dst_z)) / 2.0
    factor = 1.0 + alpha * float(np.tanh(stress))
    return float(max(0.01, min(1.0, base_w * factor)))


def build_dynamic_graphs(edges_path, cids, months, supply_lookup):
    edges = pd.read_csv(edges_path)
    # 4개 취약 기업 엣지 제거 시 cids에 없으면 자동으로 ci.get()이 None 반환 → skip
    edges["valid_from"] = pd.to_datetime(edges["valid_from"], errors="coerce")
    edges["valid_to"]   = pd.to_datetime(edges["valid_to"],   errors="coerce")

    ci = {c: i for i, c in enumerate(cids)}
    monthly = {}
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
            s = ci.get(r["src_company_id"])
            d = ci.get(r["dst_company_id"])
            if s is None or d is None:
                continue
            base_w = max(0.01, min(1.0,
                float(r.get("confidence_plink", 0.8)) *
                float(r.get("strength", 0.8))))
            src_z = supply_lookup.get((r["src_company_id"], ym), 0.0)
            dst_z = supply_lookup.get((r["dst_company_id"], ym), 0.0)
            w = dynamic_edge_weight(base_w, src_z, dst_z)
            ei.extend([[s, d], [d, s]])
            ea.extend([[w], [w]])

        n_e = len(ei)
        edge_counts.append(n_e)
        if n_e > 0:
            monthly[ym] = (
                torch.tensor(ei, dtype=torch.long).t().contiguous(),
                torch.tensor(ea, dtype=torch.float32),
            )
        else:
            self_loop = torch.arange(len(cids))
            monthly[ym] = (
                torch.stack([self_loop, self_loop]),
                torch.ones(len(cids), 1, dtype=torch.float32) * 0.01,
            )

    return monthly, edge_counts


# ────────────────────────────────────────────────────────────────
# 슬라이딩 윈도우
# ────────────────────────────────────────────────────────────────
def make_windows(X, y_bin, y_cont, mask, lookback, offset=0):
    T = X.shape[0]
    windows = []
    for t in range(lookback - 1, T):
        m = mask[t]
        if int(m.sum().item()) < MIN_VALID:
            continue
        windows.append((
            offset + t,
            X[t - lookback + 1: t + 1],
            y_bin[t], y_cont[t], m,
        ))
    return windows


# ────────────────────────────────────────────────────────────────
# 모델
# ────────────────────────────────────────────────────────────────
class DynamicTemporalGAT(nn.Module):
    def __init__(self, in_ch, h_gru, h_gat, heads, drop):
        super().__init__()
        self.drop = drop
        self.gru      = nn.GRU(in_ch, h_gru, batch_first=True)
        self.gru_norm = nn.LayerNorm(h_gru)
        self.gat1     = GATConv(h_gru, h_gat, heads=heads, concat=True,
                                dropout=drop, edge_dim=1, add_self_loops=True)
        self.gat1_norm= nn.LayerNorm(h_gat * heads)
        self.gat2     = GATConv(h_gat * heads, h_gat, heads=1, concat=False,
                                dropout=drop, edge_dim=1, add_self_loops=True)
        self.gat2_norm= nn.LayerNorm(h_gat)
        self.head = nn.Sequential(
            nn.Linear(h_gat, h_gat//2), nn.ELU(),
            nn.Dropout(drop), nn.Linear(h_gat//2, 1),
        )

    def forward(self, x, ei, ea):
        h, _ = self.gru(x.permute(1, 0, 2))
        h    = self.gru_norm(
            F.dropout(h[:, -1, :], p=self.drop, training=self.training))
        h1   = self.gat1(h, ei, ea)
        h1   = self.gat1_norm(F.elu(h1))
        h1   = F.dropout(h1, p=self.drop, training=self.training)
        h2   = self.gat2(h1, ei, ea)
        h2   = self.gat2_norm(F.elu(h2))
        return self.head(h2).squeeze(-1)


def masked_bce(logits, target, mask):
    n = mask.sum()
    if n < 1:
        return torch.tensor(0., requires_grad=True)
    return (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * mask).sum() / n


def train_epoch(model, opt, windows, dyn_graphs, months):
    model.train()
    total, cnt = 0, 0
    for t_abs, xw, yw_bin, _, mw in windows:
        ei, ea = dyn_graphs[months[t_abs]]
        opt.zero_grad()
        logits = model(xw, ei, ea)
        loss   = masked_bce(logits, yw_bin, mw)
        if loss.item() == 0:
            continue
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
        if m.sum() == 0:
            continue
        all_logits.append(logits[m])
        all_true_bin.append(yw_bin.numpy()[m])
        all_true_cont.append(yw_cont.numpy()[m])

    if not all_logits:
        return dict(auc=np.nan, ic=np.nan, n=0, ic_pos_rate=np.nan)

    lg   = np.concatenate(all_logits)
    tb   = np.concatenate(all_true_bin)
    tc   = np.concatenate(all_true_cont)
    prob = 1 / (1 + np.exp(-lg))

    try:
        auc = roc_auc_score(tb, prob)
    except Exception:
        auc = np.nan
    rho, _ = spearmanr(prob, tc)
    ic = float(rho) if not np.isnan(rho) else np.nan
    return dict(auc=auc, ic=ic, n=len(tb), ic_pos_rate=np.nan)


@torch.no_grad()
def monthly_ic_list(model, windows, dyn_graphs, months):
    model.eval()
    ics = []
    for t_abs, xw, _, yw_cont, mw in windows:
        ei, ea  = dyn_graphs[months[t_abs]]
        logits  = model(xw, ei, ea).numpy()
        m       = mw.numpy().astype(bool)
        prob    = 1 / (1 + np.exp(-logits))
        pv, tv  = prob[m], yw_cont.numpy()[m]
        if len(pv) < 3:
            continue
        rho, _ = spearmanr(pv, tv)
        if not np.isnan(rho):
            ics.append(float(rho))
    return ics


def train_model(model, tr_wins, va_wins, dyn_graphs, months, name, patience):
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", patience=10, factor=0.5, min_lr=1e-5)
    best_val, best_state, no_imp = 0.0, None, 0

    for ep in range(1, EPOCHS + 1):
        train_epoch(model, opt, tr_wins, dyn_graphs, months)
        va_m      = evaluate(model, va_wins, dyn_graphs, months)
        val_score = va_m["auc"] if not np.isnan(va_m["auc"]) else 0.0
        sch.step(val_score)
        if val_score > best_val + 1e-5:
            best_val   = val_score
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp     = 0
        else:
            no_imp += 1
        if ep % 100 == 0:
            log.info("  [%s] ep=%d val_auc=%.4f lr=%.6f",
                     name, ep, val_score, opt.param_groups[0]["lr"])
        if no_imp >= patience:
            log.info("  [%s] early stop ep=%d best_auc=%.4f", name, ep, best_val)
            break

    if best_state:
        model.load_state_dict(best_state)
    return model


# ────────────────────────────────────────────────────────────────
# 분석 1: 40개 vs 44개 기업 성능 비교
# ────────────────────────────────────────────────────────────────
def run_analysis1(ar_df_44, c44_df):
    log.info("=" * 60)
    log.info("분석 1: 40개 기업 GNN 학습 시작")
    log.info("=" * 60)

    cids_40 = [c for c in c44_df["company_id"].tolist() if c not in WEAK_4]
    log.info("40개 기업 목록 (취약 4개 제외): %d개", len(cids_40))
    for w in WEAK_4:
        log.info("  제외: %s", w)

    ar_df_40 = ar_df_44[ar_df_44["company_id"].isin(cids_40)].copy()

    panel, months = build_feature_matrix(PATH_TONE, ar_df_40, cids_40)

    # 40개 유니버스에서 reliable 현황
    n_reliable = int(panel["valid"].sum())
    n_total    = len(panel)
    log.info("40개 유니버스 — 유효(valid) 행: %d/%d", n_reliable, n_total)

    supply_lookup = build_supply_lookup(panel)
    X, y_bin, y_cont, mask = build_tensors(panel, months, cids_40)
    dyn_graphs, edge_counts = build_dynamic_graphs(PATH_EDGES, cids_40, months, supply_lookup)

    # 엣지 수 변화 확인 (4개 기업 포함 엣지 제거 효과)
    log.info("40개 그래프 — 월평균 엣지: %.0f (44개는 280)", np.mean(edge_counts))

    T = X.shape[0]
    months_str = [str(m) for m in months]
    tr_end = next((i for i, m in enumerate(months_str) if m > TRAIN_END), T)
    va_end = next((i for i, m in enumerate(months_str) if m > VAL_END),   T)

    tr_wins = make_windows(X[:tr_end],       y_bin[:tr_end],       y_cont[:tr_end],       mask[:tr_end],       LOOKBACK, 0)
    va_wins = make_windows(X[tr_end:va_end], y_bin[tr_end:va_end], y_cont[tr_end:va_end], mask[tr_end:va_end], LOOKBACK, tr_end)
    te_wins = make_windows(X[va_end:],       y_bin[va_end:],       y_cont[va_end:],       mask[va_end:],       LOOKBACK, va_end)

    log.info("윈도우: train=%d, val=%d, test=%d", len(tr_wins), len(va_wins), len(te_wins))

    set_seed(SEED)
    model = DynamicTemporalGAT(N_FEAT, H_GRU, H_GAT, GAT_HEADS, DROPOUT)
    model = train_model(model, tr_wins, va_wins, dyn_graphs, months, "DynGAT-40", PATIENCE_FULL)

    te_m = evaluate(model, te_wins, dyn_graphs, months)
    ics  = monthly_ic_list(model, te_wins, dyn_graphs, months)
    ic_pos_rate = np.mean([i > 0 for i in ics]) if ics else np.nan

    log.info("[40개] test_N=%d AUC=%.4f IC=%+.4f IC>0=%.0f%%",
             te_m["n"], te_m["auc"], te_m["ic"], ic_pos_rate * 100)

    return {
        "버전": "40개",
        "기업수": 40,
        "test_N": te_m["n"],
        "AUC": te_m["auc"],
        "IC": te_m["ic"],
        "IC_pos_rate": ic_pos_rate,
        "mean_edges": np.mean(edge_counts),
    }


# ────────────────────────────────────────────────────────────────
# 분석 2: reliable 임계값 민감도
# ────────────────────────────────────────────────────────────────
def run_analysis2(ar_df_44, cids_44):
    log.info("=" * 60)
    log.info("분석 2: reliable 임계값 민감도 (5, 8, 10, 15)")
    log.info("=" * 60)

    # 원본 tone 데이터에서 n_fb2 컬럼 확인 (tone_monthly_zscore_all44_v2.parquet 기준: n_fb2)
    tone_raw = pd.read_parquet(PATH_TONE)
    log.info("tone 컬럼 수: %d", len(tone_raw.columns))
    n_col = None
    for candidate in ["n_fb2", "n_articles_v3", "n_fb2_v3", "n_articles"]:
        if candidate in tone_raw.columns:
            n_col = candidate
            log.info("기사수 컬럼 사용: %s", n_col)
            break

    results = []
    THRESHOLDS = [5, 8, 10, 15]

    for thr in THRESHOLDS:
        log.info("--- 임계값 %d ---", thr)

        # reliable 행 수 계산 (tone_raw 직접 집계)
        tone_44 = tone_raw[tone_raw["company_id"].isin(cids_44)].copy()
        if n_col is not None:
            n_reliable_rows = int(tone_44[n_col].fillna(0).ge(thr).sum())
            n_reliable_cos  = int(
                tone_44.groupby("company_id")[n_col]
                .apply(lambda s: (s.fillna(0) >= thr).any()).sum()
            )
        else:
            n_reliable_rows = -1
            n_reliable_cos  = -1
            log.warning("n_articles 컬럼 없음 — 원본 reliable_v3 사용")

        # GNN 학습
        panel, months = build_feature_matrix(PATH_TONE, ar_df_44, cids_44,
                                             fb2_threshold=thr if n_col else None)
        supply_lookup = build_supply_lookup(panel)
        X, y_bin, y_cont, mask = build_tensors(panel, months, cids_44)
        dyn_graphs, _ = build_dynamic_graphs(PATH_EDGES, cids_44, months, supply_lookup)

        T = X.shape[0]
        months_str = [str(m) for m in months]
        tr_end = next((i for i, m in enumerate(months_str) if m > TRAIN_END), T)
        va_end = next((i for i, m in enumerate(months_str) if m > VAL_END),   T)

        tr_wins = make_windows(X[:tr_end],       y_bin[:tr_end],       y_cont[:tr_end],       mask[:tr_end],       LOOKBACK, 0)
        va_wins = make_windows(X[tr_end:va_end], y_bin[tr_end:va_end], y_cont[tr_end:va_end], mask[tr_end:va_end], LOOKBACK, tr_end)
        te_wins = make_windows(X[va_end:],       y_bin[va_end:],       y_cont[va_end:],       mask[va_end:],       LOOKBACK, va_end)

        set_seed(SEED)
        model = DynamicTemporalGAT(N_FEAT, H_GRU, H_GAT, GAT_HEADS, DROPOUT)
        model = train_model(model, tr_wins, va_wins, dyn_graphs, months,
                            f"DynGAT-thr{thr}", PATIENCE_SENS)

        te_m = evaluate(model, te_wins, dyn_graphs, months)
        ics  = monthly_ic_list(model, te_wins, dyn_graphs, months)
        ic_pos = np.mean([i > 0 for i in ics]) if ics else np.nan

        results.append({
            "임계값": thr,
            "reliable_행수": n_reliable_rows,
            "reliable_기업수": n_reliable_cos,
            "AUC": te_m["auc"],
            "IC": te_m["ic"],
            "IC>0": ic_pos,
            "test_N": te_m["n"],
        })
        log.info("[임계값=%d] reliable_행=%d 기업=%d AUC=%.4f IC=%+.4f IC>0=%.0f%%",
                 thr, n_reliable_rows, n_reliable_cos,
                 te_m["auc"], te_m["ic"], ic_pos * 100)

    return results


# ────────────────────────────────────────────────────────────────
# 분석 3: KG 엣지 Temporal Validity
# ────────────────────────────────────────────────────────────────
def run_analysis3():
    log.info("=" * 60)
    log.info("분석 3: KG 엣지 Temporal Validity 검증")
    log.info("=" * 60)

    edges = pd.read_csv(PATH_EDGES)
    edges["valid_from"] = pd.to_datetime(edges["valid_from"], errors="coerce")
    edges["valid_to"]   = pd.to_datetime(edges["valid_to"],   errors="coerce")

    n_total     = len(edges)
    n_vf        = int(edges["valid_from"].notna().sum())
    n_vt        = int(edges["valid_to"].notna().sum())

    # valid_from 분포
    vf_years = edges["valid_from"].dropna().dt.year.value_counts().sort_index()
    # valid_to 분포
    vt_years = edges["valid_to"].dropna().dt.year.value_counts().sort_index()

    # test 기간: 2024-07 ~ 2025-11
    test_start = pd.Timestamp("2024-07-01")
    test_end   = pd.Timestamp("2025-11-30")

    # test 기간 내 만료 엣지 (valid_to가 test 기간 안에 있는 엣지)
    expire_mask = (
        edges["valid_to"].notna() &
        (edges["valid_to"] >= test_start) &
        (edges["valid_to"] <= test_end)
    )
    expire_edges = edges[expire_mask][["src_company_id","rel_type","dst_company_id","valid_from","valid_to","evidence"]].copy()
    expire_edges["valid_to_str"] = expire_edges["valid_to"].dt.strftime("%Y-%m")

    # 월별 활성 엣지 수 (test 기간)
    months_test = pd.period_range("2024-07", "2025-11", freq="M")
    monthly_active = []
    for ym in months_test:
        t_stamp = ym.to_timestamp()
        active_mask = (
            edges["valid_from"].notna() &
            (edges["valid_from"] <= t_stamp) &
            (edges["valid_to"].isna() | (edges["valid_to"] >= t_stamp))
        )
        # 방향 포함 (양방향 × 2)
        n_active = int(active_mask.sum()) * 2
        monthly_active.append({"ym": str(ym), "n_active": n_active})

    monthly_active_df = pd.DataFrame(monthly_active)
    avg_active = monthly_active_df["n_active"].mean()

    # 만료 엣지 상세 (전체 valid_to 있는 엣지)
    all_expire = edges[edges["valid_to"].notna()][
        ["src_company_id","rel_type","dst_company_id","valid_from","valid_to","evidence"]
    ].copy()
    all_expire["valid_from_str"] = all_expire["valid_from"].dt.strftime("%Y-%m")
    all_expire["valid_to_str"]   = all_expire["valid_to"].dt.strftime("%Y-%m")

    return {
        "n_total":       n_total,
        "n_vf":          n_vf,
        "n_vt":          n_vt,
        "vf_years":      vf_years,
        "vt_years":      vt_years,
        "expire_edges":  expire_edges,
        "all_expire":    all_expire,
        "monthly_active": monthly_active_df,
        "avg_active":    avg_active,
    }


# ────────────────────────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────────────────────────
def main():
    set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 공통 데이터 로딩
    log.info("공통 데이터 로딩...")
    c44_df  = pd.read_csv(PATH_C44)
    cids_44 = c44_df["company_id"].tolist()
    ar_df_44 = load_market_adj(PATH_PRICES, PATH_MKT, PATH_MKTMAP, cids_44)
    log.info("AR 데이터: %d행", len(ar_df_44))

    # ── 분석 3 (빠름, 먼저 실행) ─────────────────────────────
    kg_result = run_analysis3()

    # ── KG 보고서 작성 ────────────────────────────────────────
    er  = kg_result["expire_edges"]
    ae  = kg_result["all_expire"]
    mdf = kg_result["monthly_active"]

    kg_lines = [
        "=" * 66,
        "=== KG 엣지 Temporal Validity 분석 ===",
        "=" * 66,
        "",
        f"  전체 엣지: {kg_result['n_total']}개",
        f"  valid_from 있는 엣지: {kg_result['n_vf']}개",
        f"  valid_to 있는 엣지 (만료 설정): {kg_result['n_vt']}개",
        f"  영구 활성 엣지 (valid_to=NaT): {kg_result['n_total'] - kg_result['n_vt']}개",
        "",
        "■ valid_from 연도 분포:",
    ]
    for yr, cnt in kg_result["vf_years"].items():
        kg_lines.append(f"    {yr}: {cnt}개")

    kg_lines += ["", "■ valid_to 연도 분포 (만료 설정 엣지):"]
    for yr, cnt in kg_result["vt_years"].items():
        kg_lines.append(f"    {yr}: {cnt}개")

    kg_lines += [
        "",
        "■ 전체 만료 엣지 목록:",
        f"  {'src':20s} {'rel':20s} {'dst':20s} {'from':8s} {'to':8s}",
        "  " + "-" * 80,
    ]
    for _, r in ae.iterrows():
        kg_lines.append(
            f"  {r['src_company_id']:20s} {r['rel_type']:20s} "
            f"{r['dst_company_id']:20s} {r['valid_from_str']:8s} {r['valid_to_str']:8s}"
        )

    kg_lines += [
        "",
        "■ test 기간(2024-07~2025-11) 내 만료 엣지:",
    ]
    if len(er) == 0:
        kg_lines.append("  없음")
    else:
        for _, r in er.iterrows():
            kg_lines.append(
                f"  {r['src_company_id']} → {r['dst_company_id']} "
                f"[{r['rel_type']}]: {r['valid_to_str']} 만료"
            )

    kg_lines += [
        "",
        "■ 월별 활성 엣지 수 (test 기간, 양방향 합산):",
        f"  {'연월':10s}  {'활성엣지':>8}",
        "  " + "-" * 22,
    ]
    for _, r in mdf.iterrows():
        kg_lines.append(f"  {r['ym']:10s}  {r['n_active']:>8d}")

    kg_lines += [
        "",
        f"  월평균 활성 엣지: {kg_result['avg_active']:.1f}개",
        "",
        "■ 해석:",
        "  - 대부분의 엣지는 valid_to=NaT (영구 활성)",
        f"  - 만료 설정 엣지 {kg_result['n_vt']}개 중 test 기간 내 만료: {len(er)}개",
        "  - Ford↔CATL (2024-12 만료): test 기간 2024-07~12 에만 활성",
        "=" * 66,
    ]

    OUT_KG.write_text("\n".join(kg_lines), encoding="utf-8")
    print("\n" + "\n".join(kg_lines))

    # ── 분석 1: 40개 GNN ──────────────────────────────────────
    result_40 = run_analysis1(ar_df_44, c44_df)

    # 44개 기존 결과 (phase5_summary_44.txt 에서 확인된 값)
    result_44 = {
        "버전": "44개 (기존)",
        "기업수": 44,
        "test_N": 540,
        "AUC": 0.5388,
        "IC": 0.0567,
        "IC_pos_rate": 0.80,
        "mean_edges": 280.0,
    }

    # ── 분석 2: 임계값 민감도 ─────────────────────────────────
    threshold_results = run_analysis2(ar_df_44, cids_44)

    # ── 최종 보고서 작성 ──────────────────────────────────────
    lines = [
        "=" * 70,
        "=== 유니버스 정제 효과 및 데이터 품질 민감도 분석 ===",
        "=" * 70,
        "",
        "■ 분석 1: 40개 기업 vs 44개 기업 GNN 성능 비교",
        f"  (취약 4개 제외: {', '.join(WEAK_4)})",
        "",
        f"  {'버전':<14} {'기업수':>6} {'test_N':>7} {'AUC':>7} {'IC':>8} {'IC>0':>7} {'월평균엣지':>10}",
        "  " + "-" * 62,
    ]
    for r in [result_44, result_40]:
        lines.append(
            f"  {r['버전']:<14} {r['기업수']:>6} {r['test_N']:>7} "
            f"{r['AUC']:>7.4f} {r['IC']:>+8.4f} {r['IC_pos_rate']:>6.0%} "
            f"{r['mean_edges']:>10.1f}"
        )

    # 성능 변화 해석
    delta_auc = result_40["AUC"] - result_44["AUC"]
    delta_ic  = result_40["IC"]  - result_44["IC"]
    lines += [
        "",
        f"  AUC 변화: {delta_auc:+.4f} ({'향상' if delta_auc > 0 else '하락'})",
        f"  IC  변화: {delta_ic:+.4f} ({'향상' if delta_ic > 0 else '하락'})",
        "",
        "  해석:",
        "  - 취약 4개 제외 시 유효 데이터 집중 → 노이즈 감소 효과 기대",
        "  - GNN 메시지 전달에서 reliable=False 노드를 완전 제거 vs 유지의 차이",
        "  - AUC 향상 시: 취약 기업 제외가 성능에 유리 → 40개 유니버스 채택 권장",
        "  - AUC 동등/하락 시: KG 구조 유지 이점 > 노이즈 비용 → 44개 유지 권장",
    ]

    lines += [
        "",
        "■ 분석 2: reliable 임계값 민감도 (n_articles 기준)",
        "  현재 설정: FB2_STRICT=10",
        "",
        f"  {'임계값':>6} {'reliable_행수':>13} {'reliable_기업수':>15} {'AUC':>7} {'IC':>8} {'IC>0':>7} {'test_N':>7}",
        "  " + "-" * 65,
    ]
    for r in threshold_results:
        marker = " ← 현재" if r["임계값"] == 10 else ""
        lines.append(
            f"  {r['임계값']:>6} {r['reliable_행수']:>13} {r['reliable_기업수']:>15} "
            f"{r['AUC']:>7.4f} {r['IC']:>+8.4f} {r['IC>0']:>6.0%} {r['test_N']:>7}{marker}"
        )

    # 민감도 해석
    aucs = [r["AUC"] for r in threshold_results]
    ics  = [r["IC"]  for r in threshold_results]
    auc_range = max(aucs) - min(aucs) if all(not np.isnan(a) for a in aucs) else np.nan
    ic_range  = max(ics)  - min(ics)  if all(not np.isnan(i) for i in ics)  else np.nan

    lines += [
        "",
        f"  AUC 범위 (max-min): {auc_range:.4f}" if not np.isnan(auc_range) else "  AUC 범위: N/A",
        f"  IC  범위 (max-min): {ic_range:.4f}"  if not np.isnan(ic_range)  else "  IC  범위: N/A",
        "",
        "  해석:",
        "  - AUC 범위 < 0.02: 임계값에 robust → 현재 설정(10) 적절",
        "  - AUC 범위 >= 0.02: 민감도 존재 → 최적 임계값으로 조정 권장",
        "  - 너무 낮은 임계값(5): 저품질 데이터 포함, 노이즈 증가 위험",
        "  - 너무 높은 임계값(15): 기업-월 커버리지 급감, 학습 데이터 부족",
    ]

    lines += ["", "=" * 70]
    report = "\n".join(lines)
    OUT_REFINE.write_text(report, encoding="utf-8")
    print("\n" + report)

    log.info("완료: %s", OUT_REFINE)
    log.info("완료: %s", OUT_KG)


if __name__ == "__main__":
    main()
