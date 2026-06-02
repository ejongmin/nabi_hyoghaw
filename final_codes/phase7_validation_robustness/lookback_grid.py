"""
lookback_grid.py
================
LOOKBACK = [3, 6, 9, 12] 격자 탐색 — GAT + GRU, universe_44

고정 하이퍼파라미터:
  DROPOUT=0.1, WD=1e-5, PATIENCE=60
  TRAIN_END="2022-06", VAL_END="2024-06"

출력: experiments/hypothesis/walk_forward_cv/reports/lookback_grid.txt
"""
from __future__ import annotations
import logging, random, sys, warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np, pandas as pd, yaml, torch
import torch.nn as nn, torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, roc_auc_score
from torch_geometric.nn import GATConv

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── 경로 설정 ─────────────────────────────────────────────────
ROOT = Path("C:/Users/john9/nabi_hyoghaw")
EXP  = ROOT / "experiments/universe_44"
HYP  = ROOT / "experiments/hypothesis/walk_forward_cv"

PATH_TONE   = EXP  / "data/processed/tone_monthly_zscore_all44_v2.parquet"
PATH_PRICES = ROOT / "data/processed/prices_daily.parquet"
PATH_MKT    = ROOT / "data/processed/market_proxies.parquet"
PATH_MKTMAP = ROOT / "data/processed/market_proxy_mapping.yaml"
PATH_EDGES  = EXP  / "data/seed/seed_edges_44.csv"
PATH_C44    = EXP  / "data/universe/companies_44.csv"
OUT_RPT     = HYP  / "reports/lookback_grid.txt"

# ── 탐색 설정 ─────────────────────────────────────────────────
LOOKBACK_LIST = [3, 6, 9, 12]

# ── 고정 하이퍼파라미터 ───────────────────────────────────────
H_GRU     = 32
H_GAT     = 16
GAT_HEADS = 4
DROPOUT   = 0.1
LR        = 1e-3
WD        = 1e-5
EPOCHS    = 400
PATIENCE  = 60
SEED      = 42
TRAIN_END = "2022-06"
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
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


# ─────────────────────────────────────────────────────────────
# 1. 데이터 로딩 (04_phase5_gnn_44.py 동일 로직)
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

    for c in FEAT_COLS:
        if c not in tone.columns:
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


def build_supply_lookup(panel: pd.DataFrame) -> Dict:
    col = "supply_z_v3" if "supply_z_v3" in panel.columns else "supply_z"
    if col not in panel.columns:
        return {}
    return {
        (row["company_id"], row["ym"]): float(row[col])
        for _, row in panel[["company_id", "ym", col]].dropna().iterrows()
    }


def dynamic_edge_weight(base_w, src_z, dst_z, alpha=DYN_ALPHA):
    stress = (abs(src_z) + abs(dst_z)) / 2.0
    return float(max(0.01, min(1.0, base_w * (1.0 + alpha * float(np.tanh(stress))))))


def build_dynamic_graphs(edges_path, cids, months, supply_lookup):
    edges = pd.read_csv(edges_path)
    edges["valid_from"] = pd.to_datetime(edges["valid_from"], errors="coerce")
    edges["valid_to"]   = pd.to_datetime(edges["valid_to"],   errors="coerce")
    ci = {c: i for i, c in enumerate(cids)}
    monthly: Dict = {}

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
    return monthly


# ─────────────────────────────────────────────────────────────
# 2. 슬라이딩 윈도우
# ─────────────────────────────────────────────────────────────

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
            y_bin[t],
            y_cont[t],
            m,
        ))
    return windows


# ─────────────────────────────────────────────────────────────
# 3. 모델 (04_phase5_gnn_44.py 동일)
# ─────────────────────────────────────────────────────────────

class DynamicTemporalGAT(nn.Module):
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
        h, _ = self.gru(x.permute(1, 0, 2))
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
# 4. 학습 & 평가
# ─────────────────────────────────────────────────────────────

def masked_bce(logits, target, mask):
    n = mask.sum()
    if n < 1:
        return torch.tensor(0., requires_grad=True)
    return (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * mask).sum() / n


def train_epoch(model, opt, windows, dyn_graphs, months):
    model.train()
    total, cnt = 0.0, 0
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
        return dict(f1=np.nan, auc=np.nan, ic=np.nan)

    lg   = np.concatenate(all_logits)
    tb   = np.concatenate(all_true_bin)
    tc   = np.concatenate(all_true_cont)
    prob = 1 / (1 + np.exp(-lg))
    pred = (prob > 0.5).astype(float)

    f1  = f1_score(tb, pred, zero_division=0)
    try:
        auc = roc_auc_score(tb, prob)
    except Exception:
        auc = np.nan
    rho, _ = spearmanr(prob, tc)
    ic = float(rho) if not np.isnan(rho) else np.nan
    return dict(f1=f1, auc=auc, ic=ic)


def train_model(model, tr_wins, va_wins, dyn_graphs, months, name):
    """학습 후 best 상태 반환, early stop epoch도 반환"""
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", patience=10, factor=0.5, min_lr=1e-5
    )
    best_val, best_state, no_imp = 0.0, None, 0
    stop_ep = EPOCHS

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
            log.info("  [%s] ep=%d  val_auc=%.4f  lr=%.6f",
                     name, ep, val_score, opt.param_groups[0]["lr"])

        if no_imp >= PATIENCE:
            stop_ep = ep
            log.info("  [%s] early stop ep=%d (best_auc=%.4f)", name, ep, best_val)
            break

    if best_state:
        model.load_state_dict(best_state)
    return model, stop_ep


# ─────────────────────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────────────────────

def main():
    set_seed(SEED)
    OUT_RPT.parent.mkdir(parents=True, exist_ok=True)

    log.info("LOOKBACK 격자 탐색 시작: %s", LOOKBACK_LIST)

    # 데이터 로딩 (모든 LOOKBACK에 공통)
    c44  = pd.read_csv(PATH_C44)
    cids = c44["company_id"].tolist()

    ar_df         = load_market_adj(PATH_PRICES, PATH_MKT, PATH_MKTMAP, cids)
    panel, months = build_feature_matrix(PATH_TONE, ar_df, cids)
    supply_lookup = build_supply_lookup(panel)
    X, y_bin, y_cont, mask = build_tensors(panel, months, cids)
    dyn_graphs = build_dynamic_graphs(PATH_EDGES, cids, months, supply_lookup)

    # train/val/test 인덱스 (LOOKBACK-독립)
    months_str = [str(m) for m in months]
    tr_end = next((i for i, m in enumerate(months_str) if m > TRAIN_END), len(months))
    va_end = next((i for i, m in enumerate(months_str) if m > VAL_END),   len(months))
    T = X.shape[0]

    log.info("기준 분할: train_end_idx=%d, val_end_idx=%d, total=%d", tr_end, va_end, T)

    results = []

    for lb in LOOKBACK_LIST:
        log.info("=" * 60)
        log.info("LOOKBACK = %d", lb)

        set_seed(SEED)

        tr_wins = make_windows(X[:tr_end],       y_bin[:tr_end],       y_cont[:tr_end],       mask[:tr_end],       lb, 0)
        va_wins = make_windows(X[tr_end:va_end], y_bin[tr_end:va_end], y_cont[tr_end:va_end], mask[tr_end:va_end], lb, tr_end)
        te_wins = make_windows(X[va_end:],       y_bin[va_end:],       y_cont[va_end:],       mask[va_end:],       lb, va_end)

        val_wins_n  = len(va_wins)
        test_wins_n = len(te_wins)
        log.info("  윈도우: train=%d, val=%d, test=%d", len(tr_wins), val_wins_n, test_wins_n)

        if len(tr_wins) == 0 or val_wins_n == 0:
            log.warning("  LOOKBACK=%d: train 또는 val 윈도우 없음 — 스킵", lb)
            results.append(dict(
                LOOKBACK=lb, val_wins=val_wins_n, test_wins=test_wins_n,
                GAT_AUC=np.nan, GAT_F1=np.nan, GAT_IC=np.nan,
                GRU_AUC=np.nan, GRU_F1=np.nan, GRU_IC=np.nan,
                GAT_stop_ep=np.nan,
            ))
            continue

        # GAT 학습
        gat = DynamicTemporalGAT(N_FEAT, H_GRU, H_GAT, GAT_HEADS, DROPOUT)
        log.info("  GAT 학습 중...")
        gat, gat_stop = train_model(gat, tr_wins, va_wins, dyn_graphs, months, f"GAT-LB{lb}")

        # GRU 학습
        gru = BaselineGRU(N_FEAT, H_GRU, DROPOUT)
        log.info("  GRU 학습 중...")
        gru, _ = train_model(gru, tr_wins, va_wins, dyn_graphs, months, f"GRU-LB{lb}")

        # 평가 (test set)
        gat_m = evaluate(gat, te_wins, dyn_graphs, months)
        gru_m = evaluate(gru, te_wins, dyn_graphs, months)

        log.info(
            "  [LB=%d] GAT: AUC=%.4f F1=%.4f IC=%+.4f | GRU: AUC=%.4f F1=%.4f IC=%+.4f | stop_ep=%d",
            lb, gat_m["auc"], gat_m["f1"], gat_m["ic"],
            gru_m["auc"], gru_m["f1"], gru_m["ic"], gat_stop,
        )

        results.append(dict(
            LOOKBACK=lb,
            val_wins=val_wins_n,
            test_wins=test_wins_n,
            GAT_AUC=gat_m["auc"],
            GAT_F1=gat_m["f1"],
            GAT_IC=gat_m["ic"],
            GRU_AUC=gru_m["auc"],
            GRU_F1=gru_m["f1"],
            GRU_IC=gru_m["ic"],
            GAT_stop_ep=gat_stop,
        ))

    # 결과 표 작성
    lines = [
        "=" * 90,
        "LOOKBACK 격자 탐색 결과 — GAT + GRU (universe_44)",
        "=" * 90,
        "",
        f"고정 하이퍼파라미터: DROPOUT={DROPOUT}, WD={WD}, PATIENCE={PATIENCE}",
        f"TRAIN_END={TRAIN_END}, VAL_END={VAL_END}",
        f"H_GRU={H_GRU}, H_GAT={H_GAT}, GAT_HEADS={GAT_HEADS}, DYN_ALPHA={DYN_ALPHA}",
        "",
        f"{'LOOKBACK':>9}  {'val_wins':>9}  {'test_wins':>10}  "
        f"{'GAT_AUC':>8}  {'GAT_F1':>7}  {'GAT_IC':>8}  "
        f"{'GRU_AUC':>8}  {'GRU_F1':>7}  {'GRU_IC':>8}  {'GAT_stop_ep':>12}",
        "-" * 90,
    ]

    for r in results:
        def fmt(v, fmt_str=".4f"):
            return f"{v:{fmt_str}}" if not (v != v) else "   NaN"

        ic_sign = lambda v: f"{v:+.4f}" if not (v != v) else "   NaN"

        lines.append(
            f"{r['LOOKBACK']:>9}  {r['val_wins']:>9}  {r['test_wins']:>10}  "
            f"{fmt(r['GAT_AUC']):>8}  {fmt(r['GAT_F1']):>7}  {ic_sign(r['GAT_IC']):>8}  "
            f"{fmt(r['GRU_AUC']):>8}  {fmt(r['GRU_F1']):>7}  {ic_sign(r['GRU_IC']):>8}  "
            f"{int(r['GAT_stop_ep']) if not (r['GAT_stop_ep'] != r['GAT_stop_ep']) else 'NaN':>12}"
        )

    # 최적 LOOKBACK 찾기
    valid_results = [r for r in results if not (r["GAT_AUC"] != r["GAT_AUC"])]
    if valid_results:
        best = max(valid_results, key=lambda r: r["GAT_AUC"])
        lines += [
            "",
            f"최적 LOOKBACK (GAT_AUC 기준): {best['LOOKBACK']}  (AUC={best['GAT_AUC']:.4f}, IC={best['GAT_IC']:+.4f})",
        ]

    lines += ["", "=" * 90]
    rpt = "\n".join(lines)
    OUT_RPT.write_text(rpt, encoding="utf-8")
    print("\n" + rpt)
    log.info("결과 저장: %s", OUT_RPT)


if __name__ == "__main__":
    main()
