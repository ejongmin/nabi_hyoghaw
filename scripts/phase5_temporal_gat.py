#!/usr/bin/env python3
"""
Phase 5: Temporal GAT — 공급망 리스크 전파 예측 (나비효과 프로젝트)
===================================================================

■ 연구 질문
  감성 시계열(월별 z-score) + 공급망 그래프 구조를 활용하면
  그래프 없이 개별 기업 특성만 쓸 때보다 다음 달 시장 조정 수익률을
  더 잘 예측할 수 있는가?

■ 아키텍처: Temporal GAT (Spatial-Temporal)
  Step 1 [Temporal — GRU]:
    각 노드의 최근 L개월 감성 피처 시퀀스 → GRU → 시계열 임베딩
  Step 2 [Spatial — GAT]:
    18 노드 시계열 임베딩 + 공급망 엣지 → Graph Attention → 그래프 임베딩
  Step 3 [Prediction]:
    Linear → 다음 달 시장 조정 수익률 예측

■ 비교 대상: Baseline (그래프 없는 GRU만)
  동일한 GRU, GAT 단계 생략 → Linear → 예측
  → GAT 유무만 다른 ablation, 그래프 구조의 기여를 정량화

■ 평가 지표
  - RMSE, MAE (절댓값 예측 오차)
  - IC (Information Coefficient): Spearman ρ(예측, 실제) 월평균
  - ICIR = mean_IC / std_IC (신호 일관성)
  - Directional Accuracy (방향 예측 정확도)
  - Hit Rate by Stage (공급망 단계별)

■ 입력
  data/processed/tone_monthly_zscore_18.parquet  → 노드 피처
  data/processed/prices_daily.parquet           → 타겟(시장 조정 수익률)
  data/processed/market_proxies.parquet         → 시장 인덱스
  data/processed/market_proxy_mapping.yaml      → 기업-인덱스 매핑
  data/seed/seed_edges_18_internal.csv          → 그래프 엣지
  data/universe/companies_18.csv                → 노드 순서 고정

■ 출력
  data/processed/phase5_predictions.parquet     → 예측값 패널
  reports/phase5_attention_weights.csv          → GAT 어텐션 가중치 (엣지별)
  reports/phase5_ic_by_month.csv               → 월별 IC 시계열
  reports/phase5_summary.txt                   → 전체 결과 보고서

참고 문헌
  Veličković et al. (2018) Graph Attention Networks, ICLR
  Yu et al. (2018) Spatio-Temporal Graph Convolutional Networks, IJCAI
  Qin et al. (2017) Dual-Stage Attention-Based Recurrent Neural Network
  Grinold & Kahn (2000) Active Portfolio Management — IC/ICIR 정의
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from scipy.stats import spearmanr
from torch_geometric.nn import GATConv

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────────────────────
# 로깅
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 경로
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]

PATH_TONE    = ROOT / "data/processed/tone_monthly_zscore_18.parquet"
PATH_PRICES  = ROOT / "data/processed/prices_daily.parquet"
PATH_MKT     = ROOT / "data/processed/market_proxies.parquet"
PATH_MKT_MAP = ROOT / "data/processed/market_proxy_mapping.yaml"
PATH_EDGES   = ROOT / "data/seed/seed_edges_18_internal.csv"
PATH_C18     = ROOT / "data/universe/companies_18.csv"

OUT_PRED   = ROOT / "data/processed/phase5_predictions.parquet"
OUT_ATTN   = ROOT / "reports/phase5_attention_weights.csv"
OUT_IC     = ROOT / "reports/phase5_ic_by_month.csv"
OUT_REPORT = ROOT / "reports/phase5_summary.txt"

# ─────────────────────────────────────────────────────────────────────────────
# 하이퍼파라미터
# ─────────────────────────────────────────────────────────────────────────────
LOOKBACK    = 6       # 과거 L개월 시퀀스 → 다음 달 예측
HIDDEN_GRU  = 32      # GRU 히든 차원
HIDDEN_GAT  = 16      # GAT 히든 차원
GAT_HEADS   = 4       # GAT 어텐션 헤드 수
DROPOUT     = 0.3     # Dropout 비율
LR          = 1e-3    # Learning rate
WEIGHT_DECAY= 1e-4    # L2 정규화
EPOCHS      = 300     # 최대 에폭
PATIENCE    = 30      # Early stopping 인내
SEED        = 42

# 시간 분할 기준: 전체 97개월 기준
# Train ~65%, Val ~17%, Test ~18%
TRAIN_RATIO = 0.65
VAL_RATIO   = 0.82

FEAT_COLS = [
    "z_score", "z_lag1", "z_lag2", "z_lag3", "delta_z", "rolling_3m",
    "regime_covid", "regime_ukraine", "regime_ira", "regime_china_export",
]
N_FEAT = len(FEAT_COLS)


# ─────────────────────────────────────────────────────────────────────────────
# 재현성 고정
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# 1. 데이터 준비
# ─────────────────────────────────────────────────────────────────────────────

def load_market_adj_returns(
    prices_path: Path,
    mkt_path: Path,
    mkt_map_path: Path,
    company_ids: List[str],
) -> pd.DataFrame:
    """
    월말 종가 기준 시장 조정 수익률 계산.
    market_adj_return = company_monthly_ret - market_index_monthly_ret
    """
    prices = pd.read_parquet(prices_path)
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices[prices["company_id"].isin(company_ids)]
    prices = prices.sort_values(["company_id", "date"])
    prices["ym"] = prices["date"].dt.to_period("M")

    # 월말 종가 → 월별 수익률
    co_last = (
        prices.groupby(["company_id", "ym"])["adj_close"]
        .last()
        .reset_index()
    )
    co_last["ret"] = co_last.groupby("company_id")["adj_close"].pct_change()
    co_ret = co_last.dropna(subset=["ret"]).copy()

    # 시장 인덱스 월별 수익률
    mkt_df = pd.read_parquet(mkt_path)
    mkt_df["date"] = pd.to_datetime(mkt_df["date"])
    idx_col = "index_id" if "index_id" in mkt_df.columns else "index_name"
    mkt_df["ym"] = mkt_df["date"].dt.to_period("M")
    idx_last = mkt_df.groupby([idx_col, "ym"])["adj_close"].last().reset_index()
    idx_ret_map: Dict[str, pd.DataFrame] = {}
    for idx in idx_last[idx_col].unique():
        s = idx_last[idx_last[idx_col] == idx].sort_values("ym").copy()
        s["mkt_ret"] = s["adj_close"].pct_change()
        idx_ret_map[str(idx)] = s[["ym", "mkt_ret"]].dropna()

    with open(mkt_map_path, "r", encoding="utf-8") as f:
        mapping = yaml.safe_load(f)
    c2i: Dict[str, str] = mapping.get("company_to_market_proxy", {})

    rows = []
    for cid in company_ids:
        idxname = c2i.get(cid)
        if not idxname or idxname not in idx_ret_map:
            log.warning("  %s: 마켓 프록시 없음 → 제외", cid)
            continue
        ci = co_ret[co_ret["company_id"] == cid][["ym", "ret"]]
        mi = idx_ret_map[idxname]
        merged = ci.merge(mi, on="ym", how="inner")
        merged["ar"] = merged["ret"] - merged["mkt_ret"]
        merged["company_id"] = cid
        rows.append(merged[["company_id", "ym", "ret", "mkt_ret", "ar"]])

    ar_df = pd.concat(rows, ignore_index=True)
    log.info("시장 조정 수익률: %d행, 기간 %s ~ %s",
             len(ar_df), ar_df["ym"].min(), ar_df["ym"].max())
    return ar_df


def build_feature_matrix(
    tone_path: Path,
    ar_df: pd.DataFrame,
    company_ids: List[str],
) -> Tuple[pd.DataFrame, List]:
    """
    노드 피처 + 타겟 통합 DataFrame 반환.
    - 전체 18개 기업 교집합 월만 쓰지 않고 가용 데이터 최대 활용
    - 기업이 특정 월에 없으면 mask=0으로 처리 (손실 계산 제외)
    - NaN: 기업 내 forward-fill → 남은 NaN 0

    Returns
    -------
    panel: company_id, ym, feat_cols..., ar_next (t+1 타겟), valid (mask 플래그)
    months: tone 데이터 기준 전체 정렬 월 목록
    """
    tone = pd.read_parquet(tone_path)
    tone["ym"] = pd.to_datetime(tone["year_month"]).dt.to_period("M")
    tone = tone[tone["company_id"].isin(company_ids)].copy()

    # 불리언 → float32
    bool_cols = ["regime_covid", "regime_ukraine", "regime_ira", "regime_china_export"]
    for c in bool_cols:
        if tone[c].dtype == bool:
            tone[c] = tone[c].astype("float32")

    # 시장 조정 수익률 left-join (없으면 ar=NaN)
    panel = tone[["company_id", "ym"] + FEAT_COLS + ["reliable"]].merge(
        ar_df[["company_id", "ym", "ar"]], on=["company_id", "ym"], how="left"
    )

    # NaN forward-fill (기업 내) → 0 대체
    panel = panel.sort_values(["company_id", "ym"])
    for c in FEAT_COLS:
        panel[c] = panel.groupby("company_id")[c].transform(
            lambda s: s.ffill().bfill().fillna(0.0)
        )

    # t+1 타겟: 가격 데이터가 있는 행만 ar_next 계산
    panel["ar_next"] = panel.groupby("company_id")["ar"].shift(-1)

    # valid 플래그: 현재 월 ar 있고, 다음 달 ar_next도 있고, reliable=True
    panel["valid"] = (
        panel["ar"].notna() &
        panel["ar_next"].notna() &
        panel["reliable"].fillna(False)
    ).astype("float32")

    # 전체 tone 월 목록 (가장 긴 시계열 기준)
    months = sorted(panel["ym"].unique())

    n_valid = panel["valid"].sum()
    n_total = len(panel)
    log.info("피처 행렬: %d기업, %d개월, 유효 관측치=%d/%d (%.1f%%)",
             panel["company_id"].nunique(), len(months),
             int(n_valid), n_total, 100 * n_valid / max(1, n_total))
    return panel, months


def build_tensors(
    panel: pd.DataFrame,
    months: List,
    company_ids: List[str],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    [T, N, F] 피처 텐서, [T, N] 타겟 텐서, [T, N] 마스크 텐서 반환.
    T = 월 수, N = 기업 수, F = 피처 수
    마스크: 1 = 유효(손실 계산 포함), 0 = 미상장/신뢰 불가(손실 제외)
    """
    T = len(months)
    N = len(company_ids)
    cid_to_idx = {cid: i for i, cid in enumerate(company_ids)}

    X    = np.zeros((T, N, N_FEAT), dtype=np.float32)
    y    = np.zeros((T, N),         dtype=np.float32)
    mask = np.zeros((T, N),         dtype=np.float32)

    ym_to_t = {ym: t for t, ym in enumerate(months)}

    for _, row in panel.iterrows():
        t = ym_to_t.get(row["ym"])
        n = cid_to_idx.get(row["company_id"])
        if t is None or n is None:
            continue
        X[t, n, :] = [float(row[c]) for c in FEAT_COLS]
        if float(row.get("valid", 0)) > 0:
            y[t, n]    = float(row["ar_next"])
            mask[t, n] = 1.0

    X_t    = torch.tensor(X,    dtype=torch.float32)
    y_t    = torch.tensor(y,    dtype=torch.float32)
    mask_t = torch.tensor(mask, dtype=torch.float32)
    return X_t, y_t, mask_t


def build_graph(
    edges_path: Path,
    company_ids: List[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PyG 형식 그래프 구축.
    방향: 원본(공급→수요) + 역방향 모두 추가 (양방향 메시지 패싱)
    edge_attr: confidence × strength (스칼라)

    Returns
    -------
    edge_index: [2, E] LongTensor
    edge_attr:  [E, 1] FloatTensor
    """
    edges = pd.read_csv(edges_path)
    cid_to_idx = {cid: i for i, cid in enumerate(company_ids)}

    ei_rows, ea_rows = [], []
    for _, r in edges.iterrows():
        src = cid_to_idx.get(r["src_company_id"])
        dst = cid_to_idx.get(r["dst_company_id"])
        if src is None or dst is None:
            continue
        w = float(r.get("confidence_plink", 0.8)) * float(r.get("strength", 0.8))
        w = max(0.01, min(1.0, w))  # [0.01, 1.0] 클램핑
        # 원본 방향
        ei_rows.append([src, dst])
        ea_rows.append([w])
        # 역방향 (동일 가중치)
        ei_rows.append([dst, src])
        ea_rows.append([w])

    edge_index = torch.tensor(ei_rows, dtype=torch.long).t().contiguous()
    edge_attr  = torch.tensor(ea_rows, dtype=torch.float32)

    log.info("그래프: %d노드, %d엣지 (양방향 포함, 원본 %d×2)",
             len(company_ids), len(ei_rows), len(edges))
    return edge_index, edge_attr


def make_windows(
    X: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    lookback: int,
    min_valid: int = 3,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    슬라이딩 윈도우 생성.
    X_win: [lookback, N, F], y_win: [N], mask_win: [N]
    미래 데이터 누출 없음 (t-L+1 ~ t 피처 → t+1 타겟, 타겟은 이미 shift된 상태)

    min_valid: 해당 월에 유효 기업이 이 수 미만이면 윈도우 건너뜀
    """
    T = X.shape[0]
    windows = []
    for t in range(lookback - 1, T):
        m_win = mask[t]
        if int(m_win.sum().item()) < min_valid:
            continue
        x_win = X[t - lookback + 1: t + 1]  # [L, N, F]
        y_win = y[t]                          # [N]
        windows.append((x_win, y_win, m_win))
    return windows


# ─────────────────────────────────────────────────────────────────────────────
# 2. 모델 정의
# ─────────────────────────────────────────────────────────────────────────────

class TemporalGAT(nn.Module):
    """
    Temporal GAT: GRU(시계열 임베딩) → GAT(그래프 어그리게이션) → Linear(예측)

    forward()는 하나의 시간 창(window)에 대한 예측을 수행:
      x:          [L, N, F]  — lookback 기간 노드 피처
      edge_index: [2, E]
      edge_attr:  [E, 1]
    Returns:
      pred:       [N]  — 다음 달 시장 조정 수익률 예측
      attn:       (edge_index, [E, heads]) 어텐션 가중치 (return_attention_weights=True 시)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_gru: int,
        hidden_gat: int,
        gat_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.dropout_p = dropout

        # ── Temporal: GRU per node ──
        # input_size = in_channels, batch_first=True → [N, L, F]
        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden_gru,
            num_layers=1,
            batch_first=True,
            dropout=0.0,  # GRU 내부 dropout은 단층이면 0
        )

        # Layer norm after GRU (수치 안정성)
        self.gru_norm = nn.LayerNorm(hidden_gru)

        # ── Spatial: GAT ──
        # Layer 1: hidden_gru → hidden_gat (multi-head, concat=True)
        self.gat1 = GATConv(
            in_channels=hidden_gru,
            out_channels=hidden_gat,
            heads=gat_heads,
            concat=True,
            dropout=dropout,
            edge_dim=1,
            add_self_loops=True,
        )
        self.gat1_norm = nn.LayerNorm(hidden_gat * gat_heads)

        # Layer 2: hidden_gat*heads → hidden_gat (single-head, concat=False)
        self.gat2 = GATConv(
            in_channels=hidden_gat * gat_heads,
            out_channels=hidden_gat,
            heads=1,
            concat=False,
            dropout=dropout,
            edge_dim=1,
            add_self_loops=True,
        )
        self.gat2_norm = nn.LayerNorm(hidden_gat)

        # ── Prediction head ──
        self.head = nn.Sequential(
            nn.Linear(hidden_gat, hidden_gat // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_gat // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | Tuple:
        """
        x: [L, N, in_features]
        Returns pred: [N] or (pred, attn_weights)
        """
        L, N, _in_feat = x.shape  # _in_feat: F 와 충돌 방지 (F = torch.nn.functional)

        # ── Step 1: GRU (temporal) ──
        # reshape: [N, L, F] (batch=N)
        x_gru = x.permute(1, 0, 2)                   # [N, L, F]
        gru_out, _ = self.gru(x_gru)                  # [N, L, H_gru]
        h = gru_out[:, -1, :]                          # [N, H_gru] 마지막 타임스텝
        h = self.gru_norm(h)
        h = F.dropout(h, p=self.dropout_p, training=self.training)

        # ── Step 2: GAT (spatial) ──
        if return_attention_weights:
            h1, (ei1, aw1) = self.gat1(
                h, edge_index, edge_attr, return_attention_weights=True
            )
        else:
            h1 = self.gat1(h, edge_index, edge_attr)

        h1 = self.gat1_norm(F.elu(h1))
        h1 = F.dropout(h1, p=self.dropout_p, training=self.training)

        if return_attention_weights:
            h2, (ei2, aw2) = self.gat2(
                h1, edge_index, edge_attr, return_attention_weights=True
            )
        else:
            h2 = self.gat2(h1, edge_index, edge_attr)

        h2 = self.gat2_norm(F.elu(h2))

        # ── Step 3: Prediction ──
        pred = self.head(h2).squeeze(-1)  # [N]

        if return_attention_weights:
            return pred, (ei1, aw1), (ei2, aw2)
        return pred


class BaselineGRU(nn.Module):
    """
    Baseline: GRU(시계열) → Linear(예측), 그래프 구조 없음
    TemporalGAT 와 동일한 GRU + 동등한 파라미터 수의 Linear head

    ablation: GAT 유무의 기여 분리
    """

    def __init__(
        self,
        in_channels: int,
        hidden_gru: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.dropout_p = dropout

        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden_gru,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        self.gru_norm = nn.LayerNorm(hidden_gru)

        # GAT 없음 → 동일 출력 차원을 MLP로 대체
        self.mlp = nn.Sequential(
            nn.Linear(hidden_gru, hidden_gru // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_gru // 2, 1),
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """x: [L, N, F]"""
        x_gru = x.permute(1, 0, 2)          # [N, L, F]
        gru_out, _ = self.gru(x_gru)
        h = gru_out[:, -1, :]               # [N, H_gru]
        h = self.gru_norm(h)
        h = F.dropout(h, p=self.dropout_p, training=self.training)
        pred = self.mlp(h).squeeze(-1)      # [N]
        return pred


# ─────────────────────────────────────────────────────────────────────────────
# 3. 학습 & 평가
# ─────────────────────────────────────────────────────────────────────────────

def masked_mse_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    유효(mask=1) 노드에 대해서만 MSE 계산.
    mask=0인 노드(미상장/신뢰 불가)는 손실에서 제외.
    """
    n_valid = mask.sum()
    if n_valid < 1:
        return torch.tensor(0.0, requires_grad=True)
    return ((pred - target) ** 2 * mask).sum() / n_valid


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    windows: List[Tuple],
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
) -> float:
    model.train()
    total_loss = 0.0
    n_valid_windows = 0
    for x_win, y_win, m_win in windows:
        optimizer.zero_grad()
        pred = model(x_win, edge_index, edge_attr)
        loss = masked_mse_loss(pred, y_win, m_win)
        if loss.item() == 0.0:
            continue
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_valid_windows += 1
    return total_loss / max(1, n_valid_windows)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    windows: List[Tuple],
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
) -> Tuple[float, float]:
    """유효 노드 기준 RMSE, MAE 반환"""
    model.eval()
    all_pred, all_true = [], []
    for x_win, y_win, m_win in windows:
        pred = model(x_win, edge_index, edge_attr).numpy()
        true = y_win.numpy()
        mask = m_win.numpy().astype(bool)
        if mask.sum() == 0:
            continue
        all_pred.append(pred[mask])
        all_true.append(true[mask])
    if not all_pred:
        return float("inf"), float("inf")
    pred_arr = np.concatenate(all_pred)
    true_arr = np.concatenate(all_true)
    rmse = float(np.sqrt(np.mean((pred_arr - true_arr) ** 2)))
    mae  = float(np.mean(np.abs(pred_arr - true_arr)))
    return rmse, mae


def compute_ic_by_month(
    model: nn.Module,
    windows: List[Tuple],         # (x_win, y_win, m_win)
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    months: List,
    lookback: int,
    test_start: int,
    company_ids: List[str],
) -> pd.DataFrame:
    """
    월별 IC (Spearman ρ(예측, 실제)) 계산.
    IC > 0: 예측 방향 맞음, IC_IR = mean/std 신호 일관성 지표
    """
    model.eval()
    rows = []
    with torch.no_grad():
        for i, (x_win, y_win, m_win) in enumerate(windows):
            pred = model(x_win, edge_index, edge_attr).numpy()
            true = y_win.numpy()
            mask = m_win.numpy().astype(bool)
            pred_v = pred[mask]
            true_v = true[mask]
            if len(pred_v) < 3:
                continue
            # Spearman ρ (유효 노드만)
            rho, p = spearmanr(pred_v, true_v)
            if np.isnan(rho):
                continue
            month_idx = test_start + i + lookback
            ym = str(months[month_idx]) if month_idx < len(months) else "unknown"

            rows.append({
                "year_month": ym,
                "IC": float(rho),
                "IC_p": float(p),
                "dir_acc": float(np.mean(np.sign(pred_v) == np.sign(true_v))),
                "n_nodes": int(mask.sum()),
            })
    df = pd.DataFrame(rows)
    return df


@torch.no_grad()
def extract_attention_weights(
    model: TemporalGAT,
    windows: List[Tuple],
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    company_ids: List[str],
    edges_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    테스트 기간 동안의 GAT 어텐션 가중치 평균 추출.
    각 엣지별 어텐션 집계 → 어떤 공급망 연결이 중요한지 분석
    """
    model.eval()
    attn_sum_l1 = None
    attn_sum_l2 = None
    count = 0

    for x_win, _, _m in windows:
        _, (ei1, aw1), (ei2, aw2) = model(
            x_win, edge_index, edge_attr, return_attention_weights=True
        )
        a1 = aw1.mean(dim=-1).numpy()  # [E] (헤드 평균)
        a2 = aw2.mean(dim=-1).numpy()  # [E]
        if attn_sum_l1 is None:
            attn_sum_l1 = a1.copy()
            attn_sum_l2 = a2.copy()
        else:
            attn_sum_l1 += a1
            attn_sum_l2 += a2
        count += 1

    if count == 0 or attn_sum_l1 is None:
        return pd.DataFrame()

    attn_mean_l1 = attn_sum_l1 / count
    attn_mean_l2 = attn_sum_l2 / count

    # edge_index에는 원본 + 역방향 포함
    ei_np = edge_index.numpy()  # [2, E]
    rows = []
    for e_idx in range(ei_np.shape[1]):
        src_i = ei_np[0, e_idx]
        dst_i = ei_np[1, e_idx]
        if src_i >= len(company_ids) or dst_i >= len(company_ids):
            continue
        rows.append({
            "src_company_id": company_ids[src_i],
            "dst_company_id": company_ids[dst_i],
            "attn_gat1":      float(attn_mean_l1[e_idx]),
            "attn_gat2":      float(attn_mean_l2[e_idx]),
            "attn_combined":  float((attn_mean_l1[e_idx] + attn_mean_l2[e_idx]) / 2),
        })

    attn_df = pd.DataFrame(rows).sort_values("attn_combined", ascending=False)
    return attn_df


# ─────────────────────────────────────────────────────────────────────────────
# 4. 전체 학습 파이프라인
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    train_windows: List[Tuple],
    val_windows: List[Tuple],
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    model_name: str,
) -> Tuple[nn.Module, List[float], List[float]]:
    """
    Early stopping 포함 학습.
    Returns: (best_model, train_losses, val_losses)
    """
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-5
    )

    best_val_rmse = float("inf")
    best_state    = None
    no_improve    = 0
    train_losses  = []
    val_losses    = []

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, optimizer, train_windows, edge_index, edge_attr)
        tr_rmse, _  = evaluate(model, train_windows, edge_index, edge_attr)
        val_rmse, _ = evaluate(model, val_windows,   edge_index, edge_attr)

        scheduler.step(val_rmse)
        train_losses.append(tr_rmse)
        val_losses.append(val_rmse)

        if val_rmse < best_val_rmse - 1e-6:
            best_val_rmse = val_rmse
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1

        if epoch % 50 == 0:
            log.info("  [%s] epoch %3d | tr_rmse=%.4f | val_rmse=%.4f | lr=%.6f",
                     model_name, epoch, tr_rmse, val_rmse,
                     optimizer.param_groups[0]["lr"])

        if no_improve >= PATIENCE:
            log.info("  [%s] Early stop at epoch %d (best val_rmse=%.4f)",
                     model_name, epoch, best_val_rmse)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_losses, val_losses


# ─────────────────────────────────────────────────────────────────────────────
# 5. 예측 패널 생성 & 보고서
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_prediction_panel(
    model: nn.Module,
    windows: List[Tuple],
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    months: List,
    lookback: int,
    start_t: int,
    company_ids: List[str],
    split_label: str,
) -> pd.DataFrame:
    model.eval()
    rows = []
    for i, (x_win, y_win, m_win) in enumerate(windows):
        pred = model(x_win, edge_index, edge_attr).numpy()
        true = y_win.numpy()
        mask = m_win.numpy().astype(bool)
        month_idx = start_t + i + lookback
        ym = str(months[month_idx]) if month_idx < len(months) else "unknown"
        for n, cid in enumerate(company_ids):
            rows.append({
                "split":      split_label,
                "year_month": ym,
                "company_id": cid,
                "pred":       float(pred[n]),
                "actual":     float(true[n]),
                "residual":   float(true[n] - pred[n]),
                "valid":      bool(mask[n]),
            })
    return pd.DataFrame(rows)


def build_report(
    metrics: Dict,
    ic_gat: pd.DataFrame,
    ic_base: pd.DataFrame,
    attn_top: pd.DataFrame,
    c18: pd.DataFrame,
    company_ids: List[str],
) -> str:
    sep = "=" * 72
    lines = [sep, "나비효과 Phase 5 — Temporal GAT 결과 보고서", sep, ""]

    # ── 모델 설명 ──
    lines += [
        "■ 아키텍처",
        f"  TemporalGAT : GRU(H={HIDDEN_GRU}) + GAT(H={HIDDEN_GAT}, heads={GAT_HEADS})",
        f"  BaselineGRU : GRU(H={HIDDEN_GRU}) [그래프 없음]",
        f"  Lookback    : {LOOKBACK}개월",
        f"  Node features: {FEAT_COLS}",
        "",
    ]

    # ── 정량 비교 ──
    lines += ["■ 성능 비교 (Test set)", ""]
    lines.append(
        f"  {'모델':<20} {'RMSE':>8} {'MAE':>8} "
        f"{'mean_IC':>9} {'ICIR':>8} {'Dir_Acc':>9}"
    )
    lines.append("  " + "-" * 60)
    for mname in ["TemporalGAT", "BaselineGRU"]:
        m = metrics.get(mname, {})
        rmse   = m.get("test_rmse", float("nan"))
        mae    = m.get("test_mae",  float("nan"))
        mean_ic = m.get("mean_ic", float("nan"))
        icir   = m.get("icir",    float("nan"))
        dir_a  = m.get("dir_acc", float("nan"))
        lines.append(
            f"  {mname:<20} {rmse:8.4f} {mae:8.4f} "
            f"{mean_ic:+9.4f} {icir:8.3f} {dir_a:9.1%}"
        )
    lines.append("")

    # ── GAT vs Baseline 비교 해석 ──
    gat_ic  = metrics.get("TemporalGAT", {}).get("mean_ic", 0)
    base_ic = metrics.get("BaselineGRU", {}).get("mean_ic", 0)
    delta_ic = gat_ic - base_ic
    lines += [
        "■ 그래프 구조 기여 분석 (GAT - Baseline)",
        f"  IC 개선: {delta_ic:+.4f}  "
        + ("← GAT가 개별 기업 예측을 개선" if delta_ic > 0
           else "← 그래프 구조 추가 효과 없음 (이 샘플 크기에서)"),
        "",
    ]

    # ── 월별 IC 요약 ──
    if len(ic_gat) > 0:
        lines += ["■ TemporalGAT 월별 IC 요약 (Test set)", ""]
        lines.append(f"  기간: {ic_gat['year_month'].iloc[0]} ~ "
                     f"{ic_gat['year_month'].iloc[-1]}")
        lines.append(f"  mean IC  = {ic_gat['IC'].mean():+.4f}")
        lines.append(f"  std  IC  = {ic_gat['IC'].std():.4f}")
        lines.append(f"  ICIR     = {ic_gat['IC'].mean() / max(ic_gat['IC'].std(), 1e-6):.3f}")
        lines.append(f"  IC>0 비율 = {(ic_gat['IC']>0).mean():.1%}")
        lines.append(f"  mean Dir_Acc = {ic_gat['dir_acc'].mean():.1%}")
        lines.append("")

    # ── GAT 어텐션 상위 엣지 ──
    if len(attn_top) > 0:
        nm = dict(zip(c18["company_id"], c18["canonical_name"]))
        lines += ["■ GAT 어텐션 상위 10 공급망 엣지 (테스트 기간 평균)", ""]
        lines.append(f"  {'Src':20s} → {'Dst':20s}   {'Attn':>6}")
        lines.append("  " + "-" * 52)
        for _, r in attn_top.head(10).iterrows():
            src_n = nm.get(r["src_company_id"], r["src_company_id"])[:18]
            dst_n = nm.get(r["dst_company_id"], r["dst_company_id"])[:18]
            lines.append(
                f"  {src_n:<20} → {dst_n:<20}  {r['attn_combined']:6.4f}"
            )
        lines.append("")

    # ── 한계 및 주의사항 ──
    lines += [
        "■ 해석 유의사항",
        "  1. 샘플 크기: N=18 노드 × ~18개월 테스트 → 통계 검정력 제한",
        "  2. IC 해석: |IC|>0.05 이상이면 실무적 의미 있음 (Grinold & Kahn 기준)",
        "  3. 어텐션 가중치: 인과관계 아닌 상관관계 반영",
        "  4. 그래프: 정적 엣지 (시간 변화 미반영), 65엣지는 소규모",
        "",
        sep,
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    set_seed(SEED)
    for p in [OUT_PRED.parent, OUT_REPORT.parent]:
        p.mkdir(parents=True, exist_ok=True)

    log.info("Phase 5 Temporal GAT 시작")
    log.info("Lookback=%d, H_gru=%d, H_gat=%d, heads=%d, dropout=%.1f",
             LOOKBACK, HIDDEN_GRU, HIDDEN_GAT, GAT_HEADS, DROPOUT)

    # ── 기업 순서 고정 ──
    c18 = pd.read_csv(PATH_C18)
    company_ids: List[str] = c18["company_id"].tolist()
    N = len(company_ids)
    log.info("18개 기업: %s", company_ids)

    # ── 데이터 로딩 ──
    ar_df = load_market_adj_returns(PATH_PRICES, PATH_MKT, PATH_MKT_MAP, company_ids)
    panel, months = build_feature_matrix(PATH_TONE, ar_df, company_ids)
    X_full, y_full, mask_full = build_tensors(panel, months, company_ids)

    T = X_full.shape[0]
    n_valid_total = int(mask_full.sum().item())
    log.info("텐서 구성: X=%s, y=%s, mask 유효=%d/%d (%.1f%%)",
             X_full.shape, y_full.shape, n_valid_total, T * len(company_ids),
             100 * n_valid_total / max(1, T * len(company_ids)))

    # ── 그래프 구축 ──
    edge_index, edge_attr = build_graph(PATH_EDGES, company_ids)

    # ── 시간 분할 ──
    train_end = int(T * TRAIN_RATIO)
    val_end   = int(T * VAL_RATIO)
    log.info("시간 분할: train=0~%d (%s~%s), val=%d~%d (%s~%s), test=%d~%d (%s~%s)",
             train_end, months[0], months[train_end - 1],
             train_end, val_end, months[train_end], months[val_end - 1],
             val_end, T - 1, months[val_end], months[-1])

    X_tr, y_tr, m_tr = X_full[:train_end],        y_full[:train_end],        mask_full[:train_end]
    X_va, y_va, m_va = X_full[train_end:val_end], y_full[train_end:val_end], mask_full[train_end:val_end]
    X_te, y_te, m_te = X_full[val_end:],          y_full[val_end:],          mask_full[val_end:]

    tr_wins  = make_windows(X_tr, y_tr, m_tr, LOOKBACK)
    val_wins = make_windows(X_va, y_va, m_va, LOOKBACK)
    te_wins  = make_windows(X_te, y_te, m_te, LOOKBACK)

    log.info("윈도우 수: train=%d, val=%d, test=%d", len(tr_wins), len(val_wins), len(te_wins))

    # ── 모델 초기화 ──
    gat_model  = TemporalGAT(N_FEAT, HIDDEN_GRU, HIDDEN_GAT, GAT_HEADS, DROPOUT)
    base_model = BaselineGRU(N_FEAT, HIDDEN_GRU, DROPOUT)

    n_params_gat  = sum(p.numel() for p in gat_model.parameters())
    n_params_base = sum(p.numel() for p in base_model.parameters())
    log.info("모델 파라미터: GAT=%d, Baseline=%d", n_params_gat, n_params_base)

    # ── 학습 ──
    log.info("TemporalGAT 학습 시작...")
    gat_model, gat_tr_loss, gat_val_loss = train_model(
        gat_model, tr_wins, val_wins, edge_index, edge_attr, "TemporalGAT"
    )

    log.info("BaselineGRU 학습 시작...")
    base_model, base_tr_loss, base_val_loss = train_model(
        base_model, tr_wins, val_wins, edge_index, edge_attr, "BaselineGRU"
    )

    # ── 최종 평가 ──
    metrics: Dict = {}
    for mname, model, te_w in [
        ("TemporalGAT", gat_model, te_wins),
        ("BaselineGRU", base_model, te_wins),
    ]:
        rmse, mae = evaluate(model, te_w, edge_index, edge_attr)
        ic_df = compute_ic_by_month(
            model, te_w, edge_index, edge_attr,
            months, LOOKBACK, val_end, company_ids
        )
        mean_ic = ic_df["IC"].mean() if len(ic_df) > 0 else float("nan")
        std_ic  = ic_df["IC"].std()  if len(ic_df) > 0 else float("nan")
        icir    = mean_ic / max(std_ic, 1e-6) if not np.isnan(std_ic) else float("nan")
        dir_acc = ic_df["dir_acc"].mean() if len(ic_df) > 0 else float("nan")
        metrics[mname] = {
            "test_rmse": rmse, "test_mae": mae,
            "mean_ic": mean_ic, "icir": icir, "dir_acc": dir_acc,
        }
        log.info("[%s] test RMSE=%.4f MAE=%.4f mean_IC=%+.4f ICIR=%.3f dir_acc=%.1f%%",
                 mname, rmse, mae, mean_ic, icir, dir_acc * 100)

    # ── IC 시계열 저장 (GAT) ──
    ic_gat = compute_ic_by_month(
        gat_model, te_wins, edge_index, edge_attr,
        months, LOOKBACK, val_end, company_ids
    )
    ic_base = compute_ic_by_month(
        base_model, te_wins, edge_index, edge_attr,
        months, LOOKBACK, val_end, company_ids
    )

    # ── 어텐션 가중치 추출 ──
    log.info("GAT 어텐션 가중치 추출 (테스트 기간)...")
    edges_df = pd.read_csv(PATH_EDGES)
    attn_df = extract_attention_weights(
        gat_model, te_wins, edge_index, edge_attr, company_ids, edges_df
    )

    # ── 예측 패널 생성 ──
    log.info("예측 패널 생성...")
    all_preds = []
    for split_label, model, w, st in [
        ("train", gat_model,  tr_wins,  0),
        ("val",   gat_model,  val_wins, train_end),
        ("test",  gat_model,  te_wins,  val_end),
    ]:
        df = generate_prediction_panel(
            model, w, edge_index, edge_attr,
            months, LOOKBACK, st, company_ids, split_label
        )
        all_preds.append(df)

    pred_panel = pd.concat(all_preds, ignore_index=True)

    # ── IC 비교 DataFrame ──
    ic_gat["model"]  = "TemporalGAT"
    ic_base["model"] = "BaselineGRU"
    ic_combined = pd.concat([ic_gat, ic_base], ignore_index=True)

    # ── 보고서 생성 ──
    report = build_report(metrics, ic_gat, ic_base, attn_df, c18, company_ids)

    # ── 저장 ──
    pred_panel.to_parquet(OUT_PRED, index=False)
    attn_df.to_csv(OUT_ATTN, index=False, encoding="utf-8-sig")
    ic_combined.to_csv(OUT_IC, index=False, encoding="utf-8-sig")
    OUT_REPORT.write_text(report, encoding="utf-8")

    log.info("저장 완료: %s, %s, %s, %s", OUT_PRED, OUT_ATTN, OUT_IC, OUT_REPORT)
    print("\n" + report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 5: Temporal GAT")
    parser.add_argument("--lookback", type=int, default=LOOKBACK)
    parser.add_argument("--epochs",   type=int, default=EPOCHS)
    parser.add_argument("--seed",     type=int, default=SEED)
    args = parser.parse_args()
    # 커맨드라인 인수로 전역 상수 오버라이드
    LOOKBACK = args.lookback
    EPOCHS   = args.epochs
    SEED     = args.seed
    main(args)
