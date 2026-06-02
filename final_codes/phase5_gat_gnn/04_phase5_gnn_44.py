"""
04_phase5_gnn_44.py
===================
Phase 5 GNN — Dynamic Graph (universe_44 버전)

기존 scripts/phase5_dynamic_final.py 와 동일한 로직.
변경 사항:
  - 경로: universe_44 실험 디렉토리 사용
  - 기업 목록: companies_44.csv (44개)
  - 엣지: seed_edges_44.csv
  - 출력: experiments/universe_44/data/processed, reports
  - 하이퍼파라미터: 개선된 버전 (LOOKBACK=3, EPOCHS=400, PATIENCE=60 등)
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
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── 경로 설정 ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]   # nabi_hyoghaw/
EXP  = ROOT / "experiments/universe_44"

PATH_TONE   = EXP  / "data/processed/tone_monthly_zscore_all44_v5.parquet"
PATH_PRICES = ROOT / "data/processed/prices_daily.parquet"        # 공유
PATH_MKT    = ROOT / "data/processed/market_proxies.parquet"      # 공유
PATH_MKTMAP = ROOT / "data/processed/market_proxy_mapping.yaml"   # 공유 (44개 모두 포함)
PATH_EDGES  = EXP  / "data/seed/seed_edges_44.csv"
PATH_C44    = EXP  / "data/universe/companies_44.csv"

OUT_PRED    = EXP  / "data/processed/phase5_predictions_44.parquet"
OUT_ATTN    = EXP  / "reports/phase5_attention_44.csv"
OUT_IC      = EXP  / "reports/phase5_ic_44.csv"
OUT_RPT     = EXP  / "reports/phase5_summary_44.txt"

# ── 하이퍼파라미터 (개선된 버전) ──────────────────────────────
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
TRAIN_END = "2022-06"   # 이 달까지 train (포함) — val 22윈도우 확보
VAL_END   = "2024-06"   # 이 달까지 val (포함), 이후 test
MIN_VALID = 3
DYN_ALPHA = 0.3         # 동적 가중치 강도: w = base × (1 + α × tanh(stress))

# ── 피처 컬럼 (v3 기반, 13개) ─────────────────────────────────
# z_cross_stage: tone v4 이후 미집계(0으로 대체) — 실질 정보 기여 없으나
# 아키텍처 일관성을 위해 유지 (제거 시 초기화 변경으로 seed 민감도 발생)
FEAT_COLS = [
    "z_score_v3",
    "z_lag1_v3",
    "z_lag2_v3",
    "delta_z_v3",
    "rolling_3m_v3",
    "supply_z_v3",
    "z_cross_v3",
    "z_cross_stage",
    "regime_covid",
    "regime_ukraine",
    "regime_ira",
    "regime_china_export",
    "regime_trump_tariff",
]
N_FEAT = len(FEAT_COLS)


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


# ─────────────────────────────────────────────────────────────
# 1. 데이터 로딩
# ─────────────────────────────────────────────────────────────

def load_market_adj(prices_path, mkt_path, mmap_path, cids):
    prices = pd.read_parquet(prices_path)
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices[prices["company_id"].isin(cids)].sort_values(["company_id", "date"])
    prices["ym"] = prices["date"].dt.to_period("M")
    co_last = prices.groupby(["company_id", "ym"])["adj_close"].last().reset_index()
    co_last["ret"] = co_last.groupby("company_id")["adj_close"].pct_change()
    co_ret = co_last.dropna(subset=["ret"])

    mkt_df = pd.read_parquet(mkt_path)
    mkt_df["date"] = pd.to_datetime(mkt_df["date"])
    idx_col = "index_id" if "index_id" in mkt_df.columns else "index_name"
    mkt_df["ym"] = mkt_df["date"].dt.to_period("M")
    idx_last = mkt_df.groupby([idx_col, "ym"])["adj_close"].last().reset_index()
    irm: Dict[str, pd.DataFrame] = {}
    for idx, g in idx_last.groupby(idx_col):
        s = g.sort_values("ym").copy()
        s["mkt_ret"] = s["adj_close"].pct_change()
        irm[str(idx)] = s[["ym", "mkt_ret"]].dropna()

    with open(mmap_path) as f:
        c2i = yaml.safe_load(f).get("company_to_market_proxy", {})

    rows = []
    for cid in cids:
        iname = c2i.get(cid)
        if not iname or iname not in irm:
            continue
        ci = co_ret[co_ret["company_id"] == cid][["ym", "ret"]]
        m  = ci.merge(irm[iname], on="ym", how="inner")
        m["ar"] = m["ret"] - m["mkt_ret"]
        m["company_id"] = cid
        rows.append(m[["company_id", "ym", "ar", "ret", "mkt_ret"]])
    return pd.concat(rows, ignore_index=True)


def build_feature_matrix(tone_path, ar_df, cids):
    tone = pd.read_parquet(tone_path)
    tone["ym"] = pd.to_datetime(tone["year_month"]).dt.to_period("M")
    tone = tone[tone["company_id"].isin(cids)].copy()

    for c in ["regime_covid", "regime_ukraine", "regime_ira",
              "regime_china_export", "regime_trump_tariff"]:
        if c in tone.columns and tone[c].dtype == bool:
            tone[c] = tone[c].astype("float32")

    available = [c for c in FEAT_COLS if c in tone.columns]
    missing   = [c for c in FEAT_COLS if c not in tone.columns]
    if missing:
        log.warning("누락 피처 (0으로 대체): %s", missing)
        for c in missing:
            tone[c] = 0.0

    reliable_col = "reliable_v3" if "reliable_v3" in tone.columns else "reliable_v2"
    panel = tone[["company_id", "ym", reliable_col] + FEAT_COLS].merge(
        ar_df[["company_id", "ym", "ar"]], on=["company_id", "ym"], how="left"
    )

    panel = panel.sort_values(["company_id", "ym"])
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

    months   = sorted(panel["ym"].unique())
    pos_rate = panel[panel["valid"] == 1]["binary_next"].mean()
    log.info(
        "피처 행렬: %d기업, %d월, 유효=%d/%d (%.1f%%), Binary 양성=%.1f%%",
        panel["company_id"].nunique(), len(months),
        int(panel["valid"].sum()), len(panel),
        100 * panel["valid"].mean(), pos_rate * 100,
    )
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


# ─────────────────────────────────────────────────────────────
# 2. 동적 그래프 구축
# ─────────────────────────────────────────────────────────────

def build_supply_lookup(panel: pd.DataFrame) -> Dict:
    col = "supply_z_v3" if "supply_z_v3" in panel.columns else "supply_z"
    if col not in panel.columns:
        return {}
    return {
        (row["company_id"], row["ym"]): float(row[col])
        for _, row in panel[["company_id", "ym", col]].dropna().iterrows()
    }


def dynamic_edge_weight(
    base_w: float,
    src_z: float,
    dst_z: float,
    alpha: float = DYN_ALPHA,
) -> float:
    """
    동적 엣지 가중치:
      stress  = (|src_z| + |dst_z|) / 2
      dynamic = base_w × (1 + α × tanh(stress))
    """
    stress = (abs(src_z) + abs(dst_z)) / 2.0
    factor = 1.0 + alpha * float(np.tanh(stress))
    return float(max(0.01, min(1.0, base_w * factor)))


def build_dynamic_graphs(
    edges_path: Path,
    cids: List[str],
    months: list,
    supply_lookup: Dict,
) -> Dict:
    edges = pd.read_csv(edges_path)
    edges["valid_from"] = pd.to_datetime(edges["valid_from"], errors="coerce")
    edges["valid_to"]   = pd.to_datetime(edges["valid_to"],   errors="coerce")

    ci = {c: i for i, c in enumerate(cids)}
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

    log.info(
        "동적 그래프: 최소=%d, 최대=%d, 평균=%.0f 엣지/월 (동적 가중치 적용)",
        min(edge_counts), max(edge_counts), np.mean(edge_counts),
    )
    return monthly


# ─────────────────────────────────────────────────────────────
# 3. 슬라이딩 윈도우
# ─────────────────────────────────────────────────────────────

def make_windows(
    X: torch.Tensor,
    y_bin: torch.Tensor,
    y_cont: torch.Tensor,
    mask: torch.Tensor,
    lookback: int,
    offset: int = 0,
) -> List[Tuple]:
    T = X.shape[0]
    windows = []
    for t in range(lookback - 1, T):
        m = mask[t]
        if int(m.sum().item()) < MIN_VALID:
            continue
        windows.append((
            offset + t,
            X[t - lookback + 1: t + 1],  # [L, N, F]
            y_bin[t],
            y_cont[t],
            m,
        ))
    return windows


# ─────────────────────────────────────────────────────────────
# 4. 모델
# ─────────────────────────────────────────────────────────────

class DynamicTemporalGAT(nn.Module):
    """GRU(시계열) → 동적 GAT(월별 엣지+동적가중치) → Sigmoid(이진 분류)"""

    def __init__(self, in_ch, h_gru, h_gat, heads, drop):
        super().__init__()
        self.drop = drop
        self.gru       = nn.GRU(in_ch, h_gru, batch_first=True)
        self.gru_norm  = nn.LayerNorm(h_gru)
        self.gat1      = GATConv(h_gru, h_gat, heads=heads, concat=True,
                                 dropout=drop, edge_dim=1, add_self_loops=True)
        self.gat1_norm = nn.LayerNorm(h_gat * heads)
        self.gat2      = GATConv(h_gat * heads, h_gat, heads=1, concat=False,
                                 dropout=drop, edge_dim=1, add_self_loops=True)
        self.gat2_norm = nn.LayerNorm(h_gat)
        self.head = nn.Sequential(
            nn.Linear(h_gat, h_gat // 2), nn.ELU(),
            nn.Dropout(drop), nn.Linear(h_gat // 2, 1),
        )

    def forward(self, x, ei, ea, return_attn=False):
        """x: [L,N,F]  ei: [2,E]  ea: [E,1]"""
        h, _ = self.gru(x.permute(1, 0, 2))       # [N, L, H]
        h    = self.gru_norm(
            F.dropout(h[:, -1, :], p=self.drop, training=self.training))

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
    """그래프 없는 기준선"""

    def __init__(self, in_ch, h_gru, drop):
        super().__init__()
        self.drop = drop
        self.gru  = nn.GRU(in_ch, h_gru, batch_first=True)
        self.norm = nn.LayerNorm(h_gru)
        self.mlp  = nn.Sequential(
            nn.Linear(h_gru, h_gru // 2), nn.ELU(),
            nn.Dropout(drop), nn.Linear(h_gru // 2, 1),
        )

    def forward(self, x, ei, ea, return_attn=False):
        h, _ = self.gru(x.permute(1, 0, 2))
        h    = self.norm(F.dropout(h[:, -1, :], p=self.drop, training=self.training))
        return self.mlp(h).squeeze(-1)


# ─────────────────────────────────────────────────────────────
# 5. 학습 & 평가
# ─────────────────────────────────────────────────────────────

def masked_bce(logits, target, mask):
    n = mask.sum()
    if n < 1:
        return torch.tensor(0., requires_grad=True)
    return (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * mask).sum() / n


def train_epoch(model, opt, windows, dyn_graphs, months):
    model.train()
    total = 0
    cnt   = 0
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
        total += loss.item()
        cnt   += 1
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
        return dict(acc=np.nan, f1=np.nan, auc=np.nan, ic=np.nan)

    lg   = np.concatenate(all_logits)
    tb   = np.concatenate(all_true_bin)
    tc   = np.concatenate(all_true_cont)
    prob = 1 / (1 + np.exp(-lg))
    pred = (prob > 0.5).astype(float)

    acc = accuracy_score(tb, pred)
    f1  = f1_score(tb, pred, zero_division=0)
    try:
        auc = roc_auc_score(tb, prob)
    except Exception:
        auc = np.nan
    rho, _ = spearmanr(prob, tc)
    ic = float(rho) if not np.isnan(rho) else np.nan
    return dict(acc=acc, f1=f1, auc=auc, ic=ic)


def train_model(model, tr_wins, va_wins, dyn_graphs, months, name):
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", patience=10, factor=0.5, min_lr=1e-5
    )
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

        if ep % 50 == 0:
            log.info(
                "  [%s] ep=%d  val_auc=%.4f  val_acc=%.1f%%  lr=%.6f",
                name, ep, val_score, va_m["acc"] * 100,
                opt.param_groups[0]["lr"],
            )
        if no_imp >= PATIENCE:
            log.info("  [%s] early stop ep=%d (best_auc=%.4f)", name, ep, best_val)
            break

    if best_state:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def monthly_ic(model, windows, dyn_graphs, months):
    model.eval()
    rows = []
    for t_abs, xw, _, yw_cont, mw in windows:
        ei, ea  = dyn_graphs[months[t_abs]]
        logits  = model(xw, ei, ea).numpy()
        m       = mw.numpy().astype(bool)
        prob    = 1 / (1 + np.exp(-logits))
        pv, tv  = prob[m], yw_cont.numpy()[m]
        if len(pv) < 3:
            continue
        rho, _ = spearmanr(pv, tv)
        tb      = (tv > 0).astype(float)
        acc     = float(np.mean((pv > 0.5) == tb))
        rows.append(dict(
            year_month=str(months[t_abs]),
            IC=float(rho) if not np.isnan(rho) else np.nan,
            dir_acc=acc,
            n_nodes=int(m.sum()),
            n_edges=int(ei.shape[1]),
        ))
    return pd.DataFrame(rows)


@torch.no_grad()
def extract_attn(model, windows, dyn_graphs, months, cids, nm):
    model.eval()
    attn_by_pair: Dict[Tuple, List] = {}

    for t_abs, xw, _, _, _ in windows:
        ei, ea = dyn_graphs[months[t_abs]]
        _, (_, aw1), (_, aw2) = model(xw, ei, ea, return_attn=True)
        a     = ((aw1.mean(-1) + aw2.mean(-1)) / 2).numpy()
        ei_np = ei.numpy()
        for e in range(min(ei_np.shape[1], len(a))):
            si, di = int(ei_np[0, e]), int(ei_np[1, e])
            if si >= len(cids) or di >= len(cids):
                continue
            key = tuple(sorted([cids[si], cids[di]]))
            attn_by_pair.setdefault(key, []).append(float(a[e]))

    if not attn_by_pair:
        return pd.DataFrame()

    rows = [
        dict(
            src_name=nm.get(k[0], "?"),
            dst_name=nm.get(k[1], "?"),
            attn=float(np.mean(v)),
        )
        for k, v in attn_by_pair.items()
    ]
    return pd.DataFrame(rows).sort_values("attn", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────────────────────

def main():
    set_seed(SEED)
    for p in [OUT_PRED.parent, OUT_RPT.parent]:
        p.mkdir(parents=True, exist_ok=True)

    log.info("Phase 5 Dynamic GAT — universe_44 버전 시작")
    log.info("피처(%d): %s", N_FEAT, FEAT_COLS)
    log.info("LOOKBACK=%d, DYN_ALPHA=%.2f", LOOKBACK, DYN_ALPHA)

    c44  = pd.read_csv(PATH_C44)
    cids = c44["company_id"].tolist()
    nm   = dict(zip(c44["company_id"], c44["canonical_name"])) if "canonical_name" in c44.columns else {}

    ar_df         = load_market_adj(PATH_PRICES, PATH_MKT, PATH_MKTMAP, cids)
    panel, months = build_feature_matrix(PATH_TONE, ar_df, cids)

    supply_lookup = build_supply_lookup(panel)
    log.info("Supply lookup: %d 항목 (기업×월)", len(supply_lookup))

    X, y_bin, y_cont, mask = build_tensors(panel, months, cids)
    dyn_graphs = build_dynamic_graphs(PATH_EDGES, cids, months, supply_lookup)

    T = X.shape[0]
    months_str = [str(m) for m in months]
    tr_end = next((i for i, m in enumerate(months_str) if m > TRAIN_END), T)
    va_end = next((i for i, m in enumerate(months_str) if m > VAL_END),   T)
    log.info(
        "분할(날짜고정): train=%s~%s (%d월), val=%s~%s (%d월), test=%s~%s (%d월)",
        months[0], months[tr_end - 1], tr_end,
        months[tr_end], months[va_end - 1], va_end - tr_end,
        months[va_end], months[-1], T - va_end,
    )

    tr_wins = make_windows(X[:tr_end],       y_bin[:tr_end],       y_cont[:tr_end],       mask[:tr_end],       LOOKBACK, 0)
    va_wins = make_windows(X[tr_end:va_end], y_bin[tr_end:va_end], y_cont[tr_end:va_end], mask[tr_end:va_end], LOOKBACK, tr_end)
    te_wins = make_windows(X[va_end:],       y_bin[va_end:],       y_cont[va_end:],       mask[va_end:],       LOOKBACK, va_end)
    log.info("윈도우: train=%d, val=%d, test=%d", len(tr_wins), len(va_wins), len(te_wins))

    gat  = DynamicTemporalGAT(N_FEAT, H_GRU, H_GAT, GAT_HEADS, DROPOUT)
    base = BaselineGRU(N_FEAT, H_GRU, DROPOUT)
    log.info(
        "파라미터: GAT=%d, Baseline=%d",
        sum(p.numel() for p in gat.parameters()),
        sum(p.numel() for p in base.parameters()),
    )

    log.info("Dynamic GAT 학습 (LOOKBACK=%d, 동적가중치 α=%.2f)...", LOOKBACK, DYN_ALPHA)
    gat  = train_model(gat,  tr_wins, va_wins, dyn_graphs, months, "DynGAT-44")
    log.info("Baseline GRU 학습...")
    base = train_model(base, tr_wins, va_wins, dyn_graphs, months, "GRU-44")

    metrics = {}
    for mname, model in [("Dynamic GAT (44)", gat), ("Baseline GRU (44)", base)]:
        m = evaluate(model, te_wins, dyn_graphs, months)
        metrics[mname] = m
        log.info("[%s] Acc=%.1f%% F1=%.4f AUC=%.4f IC=%+.4f",
                 mname, m["acc"] * 100, m["f1"], m["auc"], m["ic"])

    ic_gat  = monthly_ic(gat,  te_wins, dyn_graphs, months)
    ic_base = monthly_ic(base, te_wins, dyn_graphs, months)
    ic_gat["model"]  = "DynamicGAT-44"
    ic_base["model"] = "BaselineGRU-44"

    attn = extract_attn(gat, te_wins, dyn_graphs, months, cids, nm)

    preds = []
    for split, model, wins in [("train", gat, tr_wins), ("val", gat, va_wins), ("test", gat, te_wins)]:
        model.eval()
        with torch.no_grad():
            for t_abs, xw, yw_bin, yw_cont, mw in wins:
                ei, ea = dyn_graphs[months[t_abs]]
                logits = model(xw, ei, ea).numpy()
                prob   = 1 / (1 + np.exp(-logits))
                m_np   = mw.numpy().astype(bool)
                for n, cid in enumerate(cids):
                    preds.append(dict(
                        split=split,
                        year_month=str(months[t_abs]),
                        company_id=cid,
                        prob=float(prob[n]),
                        pred_binary=int(prob[n] > 0.5),
                        actual_binary=int(yw_bin.numpy()[n]),
                        actual_ar=float(yw_cont.numpy()[n]),
                        valid=bool(m_np[n]),
                        n_edges=int(ei.shape[1]),
                    ))

    pd.DataFrame(preds).to_parquet(OUT_PRED, index=False)
    if len(attn):
        attn.to_csv(OUT_ATTN, index=False, encoding="utf-8-sig")
    pd.concat([ic_gat, ic_base]).to_csv(OUT_IC, index=False, encoding="utf-8-sig")

    n_edges_csv = len(pd.read_csv(PATH_EDGES)) if PATH_EDGES.exists() else 0
    lines = [
        "=" * 72,
        "Phase 5 — Dynamic Graph (universe_44) 결과",
        "=" * 72,
        "",
        "[설정]",
        f"  1. 동적 엣지 가중치: w = base × (1 + {DYN_ALPHA} × tanh(avg_supply_stress))",
        f"  2. tone_monthly_zscore_all44_v6.parquet 사용 (44개 기업)",
        f"  3. seed_edges_44.csv ({n_edges_csv}개 엣지)",
        f"  4. LOOKBACK = {LOOKBACK}개월",
        f"  5. 피처 {N_FEAT}개",
        f"  6. TRAIN_END={TRAIN_END}, VAL_END={VAL_END}",
        "",
        "■ 성능 비교 (Test set)",
        f"  {'모델':<28} {'Acc':>7} {'F1':>7} {'AUC':>7} {'IC':>8}",
        "  " + "-" * 58,
    ]
    for mname, m in metrics.items():
        lines.append(
            f"  {mname:<28} {m['acc']*100:6.1f}% {m['f1']:7.4f} "
            f"{m['auc']:7.4f} {m['ic']:+8.4f}"
        )

    if len(ic_gat) > 0:
        lines += [
            "",
            f"■ Dynamic GAT 월별 IC (test): "
            f"mean={ic_gat['IC'].mean():+.3f}  "
            f"std={ic_gat['IC'].std():.3f}  "
            f"IC>0={(ic_gat['IC']>0).mean():.0%}",
            f"  Dir_Acc: mean={ic_gat['dir_acc'].mean():.1%}  "
            f"max={ic_gat['dir_acc'].max():.1%}",
            "",
            "  월별 상세:",
        ]
        for _, r in ic_gat.iterrows():
            lines.append(
                f"    {r['year_month']}  IC={r['IC']:+.3f}  "
                f"acc={r['dir_acc']:.1%}  N={r['n_nodes']}  엣지={r['n_edges']}"
            )

    if len(attn) > 0:
        lines += ["", "■ GAT 어텐션 상위 10 엣지 (동적 가중치 반영)"]
        for _, r in attn.head(10).iterrows():
            lines.append(f"  {r['src_name']:22s} ↔ {r['dst_name']:22s}  {r['attn']:.4f}")

    lines += ["", "=" * 72]
    rpt = "\n".join(lines)
    OUT_RPT.write_text(rpt, encoding="utf-8")
    print("\n" + rpt)


if __name__ == "__main__":
    main()
