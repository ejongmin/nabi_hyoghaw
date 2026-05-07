"""
백테스트 재설계 — z<-1.5 신호 기반
=====================================
기존 백테스트: 감성 충격 기업 제외/축소 → 벤치마크 미달
재설계 아이디어:
  z<-1.5 충격 월에 해당 기업 UNDERWEIGHT (감성 나쁜 달 회피)
  z<-1.5 + PEER 동반 하락 신호 → 리스크 경보 포트폴리오
  역투자(Contrarian): z<-1.5 기업 매수 (평균회귀 활용)

비교:
  Bench        : 18개 기업 동일가중
  Exclude      : z<-1.5 충격 기업 다음 달 제외
  Contrarian   : z<-1.5 충격 기업 다음 달 오버웨이트 (역투자)
  PEER_avoid   : PEER 기업 중 z<-1.5 있을 때 해당 레이어 축소
"""
from pathlib import Path
import numpy as np, pandas as pd, yaml

ROOT = Path(__file__).resolve().parents[1]

PATH_PRICES = ROOT/"data/processed/prices_daily.parquet"
PATH_MKT    = ROOT/"data/processed/market_proxies.parquet"
PATH_MKTMAP = ROOT/"data/processed/market_proxy_mapping.yaml"
PATH_TONE   = ROOT/"data/processed/tone_monthly_zscore_18_v2.parquet"
PATH_C18    = ROOT/"data/universe/companies_18.csv"
OUT_RPT     = ROOT/"reports/phase4_backtest_v2.txt"

SHOCK_THR = -1.5   # z < -1.5 충격 신호


def compute_monthly_returns(cids):
    prices = pd.read_parquet(PATH_PRICES)
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices[prices["company_id"].isin(cids)].sort_values(["company_id","date"])
    prices["ym"] = prices["date"].dt.to_period("M")
    last = prices.groupby(["company_id","ym"])["adj_close"].last().reset_index()
    last["ret"] = last.groupby("company_id")["adj_close"].pct_change()
    return last.dropna(subset=["ret"])


def backtest(ret_df: pd.DataFrame, signals: pd.DataFrame,
             strategy: str, shock_thr: float = SHOCK_THR) -> pd.Series:
    """
    strategy: 'bench' | 'exclude' | 'contrarian' | 'peer_avoid'
    signals: company_id, ym, z_score_v2
    """
    cids = ret_df["company_id"].unique()
    months = sorted(ret_df["ym"].unique())

    portfolio_rets = []
    for ym in months:
        # 전월 신호 (이번 달 투자 결정에 사용)
        prev_ym = ym - 1
        prev_sig = signals[signals["ym"]==prev_ym][["company_id","z_score_v2"]]
        sig_map = dict(zip(prev_sig["company_id"], prev_sig["z_score_v2"]))

        # 이번 달 수익률
        month_ret = ret_df[ret_df["ym"]==ym].set_index("company_id")["ret"]
        available = [c for c in cids if c in month_ret.index]
        if not available: continue

        if strategy == "bench":
            weights = {c: 1/len(available) for c in available}

        elif strategy == "exclude":
            shock_cos = {c for c,z in sig_map.items() if z < shock_thr and not np.isnan(z)}
            normal_cos = [c for c in available if c not in shock_cos]
            if not normal_cos: normal_cos = available  # fallback
            weights = {c: 1/len(normal_cos) for c in normal_cos}

        elif strategy == "contrarian":
            # 충격 기업 2x 오버웨이트 (평균회귀 활용)
            base_w = 1/len(available)
            weights = {}
            for c in available:
                z = sig_map.get(c, 0)
                if not np.isnan(z) and z < shock_thr:
                    weights[c] = base_w * 2.0  # 2x 오버웨이트
                else:
                    weights[c] = base_w
            # 정규화
            total = sum(weights.values())
            weights = {c: w/total for c,w in weights.items()}

        elif strategy == "peer_avoid":
            # 같은 레이어에서 충격 기업이 있으면 레이어 전체 0.5x 하향
            c18 = pd.read_csv(PATH_C18)
            stage_map = dict(zip(c18["company_id"], c18["stage_bucket"]))
            shocked_stages = {stage_map.get(c) for c,z in sig_map.items()
                              if z < shock_thr and not np.isnan(z)}
            weights = {}
            for c in available:
                s = stage_map.get(c,"")
                weights[c] = 0.5 if s in shocked_stages else 1.5
            total = sum(weights.values())
            weights = {c: w/total for c,w in weights.items()}

        else:
            weights = {c: 1/len(available) for c in available}

        port_ret = sum(weights.get(c, 0) * month_ret.get(c, 0) for c in available)
        portfolio_rets.append(dict(ym=ym, ret=port_ret, strategy=strategy))

    return pd.DataFrame(portfolio_rets)


def calc_metrics(ret_series: pd.Series) -> dict:
    rets = ret_series.dropna()
    if len(rets) == 0:
        return dict(cagr=np.nan, sharpe=np.nan, mdd=np.nan, vol=np.nan)
    ann_ret = (1 + rets.mean()) ** 12 - 1
    ann_vol = rets.std() * np.sqrt(12)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan
    cum = (1 + rets).cumprod()
    roll_max = cum.cummax()
    dd = (cum / roll_max - 1)
    mdd = float(dd.min())
    return dict(cagr=ann_ret, sharpe=sharpe, mdd=mdd, vol=ann_vol)


def main():
    print("백테스트 v2 시작...")
    c18  = pd.read_csv(PATH_C18)
    cids = c18["company_id"].tolist()

    ret_df  = compute_monthly_returns(cids)
    tone    = pd.read_parquet(PATH_TONE)
    tone["ym"] = pd.to_datetime(tone["year_month"]).dt.to_period("M")
    signals = tone[["company_id","ym","z_score_v2"]].copy()

    # 2019-01 이후 (데이터 안정화 이후)
    start_ym = pd.Period("2019-01","M")
    ret_df  = ret_df[ret_df["ym"] >= start_ym]
    signals = signals[signals["ym"] >= start_ym]

    dfs = []
    for strat in ["bench","exclude","contrarian","peer_avoid"]:
        df = backtest(ret_df, signals, strat)
        dfs.append(df)
    all_bt = pd.concat(dfs, ignore_index=True)

    SEP = "="*60
    lines = [SEP,"백테스트 v2 결과 (2019-01 이후, 18개 기업)",SEP,"",
             f"  {'전략':15s}  {'CAGR':>8}  {'Vol':>8}  {'Sharpe':>8}  {'MDD':>8}",
             "  "+"-"*50]

    for strat, g in all_bt.groupby("strategy"):
        m = calc_metrics(g.set_index("ym")["ret"])
        lines.append(f"  {strat:15s}  {m['cagr']*100:7.2f}%  {m['vol']*100:7.2f}%  "
                     f"{m['sharpe']:8.3f}  {m['mdd']*100:7.2f}%")

    lines += ["","■ 해석",
              "  Contrarian (역투자): z<-1.5 기업 매수가 벤치마크 초과하면",
              "  → '감성 충격 후 주가 반등' 패턴이 실제로 활용 가능함을 의미",
              "  → 이 경우 연구의 투자 전략 함의 재정립 가능",""]
    lines.append(SEP)
    rpt = "\n".join(lines)
    OUT_RPT.write_text(rpt, encoding="utf-8")
    print("\n"+rpt)

    all_bt.to_csv(ROOT/"reports/phase4_backtest_v2_detail.csv",
                  index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
