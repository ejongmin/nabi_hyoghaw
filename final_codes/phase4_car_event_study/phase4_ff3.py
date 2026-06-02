"""
Phase 4 FF3 (Fama-French 3-Factor Model)
==========================================
약점 해결: 단순 시장모형 → FF3 Factor Model로 정상수익률 추정 개선

FF3: R_i = α + β_m×R_mkt + β_smb×SMB + β_hml×HML + ε
  SMB (Small Minus Big): 소형주 - 대형주 수익률 차이
  HML (High Minus Low): 고BM - 저BM 수익률 차이

데이터 소스:
  - 팩터 데이터: yfinance에서 근사 팩터 계산 (실제 FF 데이터 대체)
  - 미국 팩터: ^GSPC(시장), IWM-SPY(SMB 근사), IVE-IVW(HML 근사)
  - 유럽 팩터: ^STOXX, EFV-EFG(HML 근사)
  - 아시아 팩터: 단순 시장모형 유지 (팩터 데이터 제한)

비교:
  - 시장모형 CAR vs FF3 CAR → 정상수익률 추정 품질 차이 확인
  - 주로 미국(TSLA, ALB) + 유럽(BMW, VW, MBG, BAS) 기업에 적용
"""
from __future__ import annotations
import logging, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
from scipy import stats

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

ROOT       = Path(__file__).resolve().parents[1]
PATH_TONE  = ROOT/"data/processed/tone_expanding_compat.parquet"
PATH_PRICES= ROOT/"data/processed/prices_daily.parquet"
PATH_MKT   = ROOT/"data/processed/market_proxies.parquet"
PATH_MKTMAP= ROOT/"data/processed/market_proxy_mapping.yaml"
PATH_C18   = ROOT/"data/universe/companies_18.csv"
PATH_FF3   = ROOT/"data/processed/ff3_factors.parquet"
OUT_RPT    = ROOT/"reports/phase4_ff3.txt"

EST_WIN_START=-250; EST_WIN_END=-20; MIN_EST=60; PREF_EST=120

# 미국·유럽 기업만 FF3 적용 (팩터 데이터 가용)
FF3_COMPANIES = {
    "US":  ["TSLA","ALB","GLEN.L","LCID","RIVN","F","GM"],
    "EU":  ["BMW.DE","VOW3.DE","MBG.DE","BAS.DE"],
}


def fetch_ff3_factors():
    """
    yfinance로 FF3 근사 팩터 데이터 다운로드 및 저장
    미국: SPY(시장), IWM-SPY(SMB 근사), IVE-IVW(HML 근사)
    유럽: ^STOXX, EFV-EFG(HML 근사)
    """
    if PATH_FF3.exists():
        log.info("FF3 팩터 캐시 로드: %s", PATH_FF3)
        return pd.read_parquet(PATH_FF3)

    if not HAS_YF:
        log.warning("yfinance 없음 → FF3 팩터 다운로드 불가")
        return pd.DataFrame()

    log.info("FF3 팩터 다운로드 중...")
    tickers = {
        "SPY": "^GSPC",   # US market
        "IWM": "IWM",     # Small cap (SMB proxy)
        "IVE": "IVE",     # Value (HML proxy)
        "IVW": "IVW",     # Growth (HML proxy)
        "STOXX": "^STOXX",# Europe market
        "EFV": "EFV",     # Europe value
        "EFG": "EFG",     # Europe growth
    }
    dfs = {}
    for name, ticker in tickers.items():
        try:
            raw = yf.download(ticker, start="2015-01-01", end="2026-04-01",
                             auto_adjust=True, progress=False)
            if len(raw) > 100:
                dfs[name] = raw["Close"].pct_change().dropna()
                log.info("  %s: %d일", name, len(dfs[name]))
        except Exception as e:
            log.warning("  %s 다운로드 실패: %s", name, e)

    if not dfs:
        return pd.DataFrame()

    # Series들을 DataFrame으로 합치기 (인덱스 기반 정렬)
    factor_df = pd.concat(dfs, axis=1).dropna()
    factor_df.columns = list(dfs.keys())

    # US 팩터 계산
    if "SPY" in factor_df.columns:
        factor_df["US_mkt"] = factor_df.get("^GSPC", factor_df["SPY"])
    if "IWM" in factor_df.columns and "SPY" in factor_df.columns:
        factor_df["US_smb"] = factor_df["IWM"] - factor_df["SPY"]
    if "IVE" in factor_df.columns and "IVW" in factor_df.columns:
        factor_df["US_hml"] = factor_df["IVE"] - factor_df["IVW"]

    # EU 팩터
    if "STOXX" in factor_df.columns:
        factor_df["EU_mkt"] = factor_df["STOXX"]
    if "EFV" in factor_df.columns and "EFG" in factor_df.columns:
        factor_df["EU_hml"] = factor_df["EFV"] - factor_df["EFG"]

    factor_df.index = pd.to_datetime(factor_df.index)
    PATH_FF3.parent.mkdir(parents=True, exist_ok=True)
    factor_df.to_parquet(PATH_FF3)
    log.info("FF3 팩터 저장: %s (%d일)", PATH_FF3, len(factor_df))
    return factor_df


def ols_car_ff3(ret, factors_arr, es, ee, xs, xe):
    """
    FF3 OLS CAR 계산
    factors_arr: [n_days × n_factors] (mkt, smb, hml)
    """
    n = ee-es
    if n<MIN_EST or es<0 or xe>len(ret) or xs<0: return None
    er = ret[es:ee]; ef = factors_arr[es:ee]
    if ef.shape[0] != len(er): return None

    # OLS with constant
    X_est = np.column_stack([np.ones(n), ef])
    try:
        beta, resid, _, _ = np.linalg.lstsq(X_est, er, rcond=None)
    except: return None

    pred_est = X_est @ beta
    residuals = er - pred_est
    sigma = float(np.sqrt(np.sum(residuals**2) / max(1, n - len(beta))))
    if sigma < 1e-10: return None

    vr = ret[xs:xe]; vf = factors_arr[xs:xe]
    el = len(vr)
    if el < max(1, int((xe-xs)*0.8)): return None

    X_evt = np.column_stack([np.ones(el), vf])
    pred_evt = X_evt @ beta
    ar = vr - pred_evt
    car = float(ar.sum())

    t = car / (sigma * np.sqrt(el)) if sigma > 1e-10 else np.nan
    p = float(2*stats.t.sf(abs(t), df=max(1, n-len(beta)))) if not np.isnan(t) else np.nan
    q = "full" if n>=PREF_EST else "marginal"
    return dict(CAR=car, t=t, p=p, quality=q, n_factors=ef.shape[1])


def find_t0(cd, shock_month, offset=1):
    try:
        t0 = np.datetime64((pd.Period(shock_month,"M")+offset).to_timestamp().date(),"D")
    except: return None
    pos = np.searchsorted(cd["dates"], t0, side="left")
    return int(pos) if pos < len(cd["dates"]) else None


def main():
    log.info("Phase 4 FF3 분석 시작")

    # FF3 팩터 로드
    ff3 = fetch_ff3_factors()
    if ff3.empty:
        log.warning("FF3 팩터 없음 → 시장모형과 비교 불가")
        return

    # 가격 데이터
    prices = pd.read_parquet(PATH_PRICES)
    prices["date"] = pd.to_datetime(prices["date"])
    mkt_df = pd.read_parquet(PATH_MKT)
    mkt_df["date"] = pd.to_datetime(mkt_df["date"])
    idx_col = "index_id" if "index_id" in mkt_df.columns else "index_name"
    mkt_df["mkt_ret"] = mkt_df.groupby(idx_col)["adj_close"].pct_change()
    ir = {}
    for iname, g in mkt_df.groupby(idx_col):
        s = g.set_index("date")["mkt_ret"]; s.name="mkt_ret"
        ir[str(iname)] = s

    with open(PATH_MKTMAP) as f:
        c2i = yaml.safe_load(f).get("company_to_market_proxy",{})

    c18  = pd.read_csv(PATH_C18)
    nm   = dict(zip(c18["company_id"], c18["canonical_name"]))
    cids = c18["company_id"].tolist()
    prices = prices[prices["company_id"].isin(cids)]
    prices["price"] = prices["adj_close"].fillna(prices["close"])

    tone = pd.read_parquet(PATH_TONE)

    # 이벤트
    events = []
    seen = set()
    mask = tone["reliable_v2"] & tone["neg_shock_v2"]
    for _, r in tone[mask].iterrows():
        eid = f"{r['company_id']}_{r['year_month']}"
        if eid in seen: continue
        seen.add(eid)
        events.append(dict(shock_company=r["company_id"],
                           shock_month=r["year_month"]))

    all_us_cids = FF3_COMPANIES["US"]
    all_eu_cids = FF3_COMPANIES["EU"]
    target_cids = [c for c in cids if c in all_us_cids + all_eu_cids]

    # FF3 팩터 정렬
    ff3.index = pd.to_datetime(ff3.index).normalize()

    rows_ff3 = []
    rows_mkt = []

    for cid in target_cids:
        region = "US" if cid in all_us_cids else "EU"
        mkt_col = "US_mkt" if region=="US" else "EU_mkt"

        g = prices[prices["company_id"]==cid].sort_values("date").copy()
        g["ret"] = g["price"].pct_change()
        g = g.dropna(subset=["ret"])
        if len(g) < 100: continue

        # 날짜 정렬된 배열 만들기
        g_idx = g.set_index("date")

        # FF3 팩터와 조인
        mkt_s = ir.get(c2i.get(cid,""), None)
        if mkt_s is None: continue

        factor_cols = [mkt_col]
        if region=="US" and "US_smb" in ff3.columns:
            factor_cols.append("US_smb")
        if region=="US" and "US_hml" in ff3.columns:
            factor_cols.append("US_hml")
        if region=="EU" and "EU_hml" in ff3.columns:
            factor_cols.append("EU_hml")

        available_ff3 = [c for c in factor_cols if c in ff3.columns]
        if not available_ff3:
            log.warning("  %s: FF3 팩터 없음", cid)
            continue

        # 데이터 조인
        df = g_idx[["ret"]].join(ff3[available_ff3], how="inner").join(
             mkt_s.to_frame("mkt"), how="inner").dropna()
        if len(df) < MIN_EST + 25: continue

        dates = df.index.values.astype("datetime64[D]")
        ret_arr  = df["ret"].values.astype(np.float64)
        mkt_arr  = df["mkt"].values.astype(np.float64)
        ff3_arr  = df[available_ff3].values.astype(np.float64)
        d2i = {d:i for i,d in enumerate(dates)}

        cd_obj = dict(dates=dates, ret=ret_arr, mkt=mkt_arr, d2i=d2i)

        for ev in [e for e in events if e["shock_company"]==cid]:
            t0 = find_t0(cd_obj, ev["shock_month"], offset=1)
            if t0 is None: continue
            es = max(0, t0+EST_WIN_START); ee = t0+EST_WIN_END
            if ee-es < MIN_EST: continue

            # 시장모형
            from importlib import import_module
            xs, xe = t0, t0+22
            res_mkt = None
            # 간단한 시장모형 OLS
            n=ee-es
            er,em=ret_arr[es:ee],mkt_arr[es:ee]
            sm,sr,smm,smr=em.sum(),er.sum(),(em*em).sum(),(em*er).sum()
            denom=n*smm-sm*sm
            if abs(denom)>1e-15:
                beta=(n*smr-sm*sr)/denom; alpha=(sr-beta*sm)/n
                sig=float(np.sqrt(np.sum((er-(alpha+beta*em))**2)/max(1,n-2)))
                if sig>1e-10 and xe<=len(ret_arr):
                    vr,vm=ret_arr[xs:xe],mkt_arr[xs:xe]
                    car_mkt=float((vr-(alpha+beta*vm)).sum())
                    rows_mkt.append(dict(company_id=cid,shock_month=ev["shock_month"],
                                        CAR_mkt=car_mkt, model="market"))

            # FF3
            res = ols_car_ff3(ret_arr, ff3_arr, es, ee, xs, xe)
            if res and res["quality"]=="full":
                rows_ff3.append(dict(company_id=cid, shock_month=ev["shock_month"],
                                     CAR_ff3=res["CAR"], n_factors=res["n_factors"],
                                     model="FF3"))

    if not rows_ff3:
        log.warning("FF3 결과 없음")
        return

    df_ff3 = pd.DataFrame(rows_ff3)
    df_mkt = pd.DataFrame(rows_mkt)
    merged = df_ff3.merge(df_mkt, on=["company_id","shock_month"], how="inner")

    SEP = "="*68
    lines = [SEP, "Phase 4 FF3 분석 — 시장모형 vs Fama-French 3-Factor", SEP, ""]

    if len(merged) > 0:
        caar_mkt = merged["CAR_mkt"].mean()
        caar_ff3 = merged["CAR_ff3"].mean()
        diff = caar_ff3 - caar_mkt
        n = len(merged)

        t_mkt,p_mkt = stats.ttest_1samp(merged["CAR_mkt"].values, 0)
        t_ff3,p_ff3 = stats.ttest_1samp(merged["CAR_ff3"].values, 0)

        def sig(p):
            return "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.10 else ""

        lines += [
            f"  대상 기업: {', '.join(target_cids)}",
            f"  이벤트 수: {n}건  (FF3 팩터 가용 기업만)", "",
            f"  {'모델':15s}  {'N':>4}  {'CAAR':>9}  {'t':>7}  {'p':>8}",
            "  "+"-"*46,
            f"  {'시장모형':15s}  {n:4d}  {caar_mkt:+9.4f}  {t_mkt:+7.3f}  "
            f"{p_mkt:.4f}{sig(p_mkt):3s}",
            f"  {'FF3':15s}  {n:4d}  {caar_ff3:+9.4f}  {t_ff3:+7.3f}  "
            f"{p_ff3:.4f}{sig(p_ff3):3s}",
            "",
            f"  CAAR 차이 (FF3 - 시장모형): {diff:+.4f}",
            f"  → {'FF3가 더 음의 CAAR' if diff < 0 else 'FF3가 더 양의 CAAR'}",
            "  → 두 모형의 방향/유의성이 일치하면 결과 robust",
        ]

        # 기업별 비교
        lines += ["", "  기업별 비교:"]
        for cid, g in merged.groupby("company_id"):
            mkt_v = g["CAR_mkt"].mean(); ff3_v = g["CAR_ff3"].mean()
            lines.append(f"  {nm.get(cid,cid):20s}: mkt={mkt_v:+.4f}  ff3={ff3_v:+.4f}  "
                         f"diff={ff3_v-mkt_v:+.4f}")
    else:
        lines.append("  [결과 없음 — 팩터 데이터 또는 가격 데이터 부족]")

    lines += ["", SEP]
    rpt = "\n".join(lines)
    OUT_RPT.write_text(rpt, encoding="utf-8")
    print("\n" + rpt)


if __name__ == "__main__":
    main()
