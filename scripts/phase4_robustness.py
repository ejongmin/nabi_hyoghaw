"""
Phase 4 Robustness Checks
==========================
약점 보완:
  [1] T=0 민감도 분석
      - T0_M1 (기존): 충격월 M+1 첫 거래일
      - T0_M0 (신규): 충격월 M 첫 거래일 (동월 반응 포착)
      비교: 두 설계의 CAAR 방향/크기 차이 → 설계 선택의 근거

  [2] Cluster Standard Error
      이벤트 간 독립 가정 위반 보정
      - 같은 달 여러 기업 동시 충격 → 이벤트-월 클러스터링
      - 비모수 부트스트랩 SE (B=1000)

  [3] 대안적 충격 임계값 민감도
      z < -1.5, z < -2.0(기존), z < -2.5 세 가지 비교
      → 임계값이 결과에 얼마나 민감한지
"""
from __future__ import annotations
import logging, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
from scipy import stats

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]

PATH_TONE  = ROOT/"data/processed/tone_monthly_zscore_18_v2.parquet"
PATH_PRICES= ROOT/"data/processed/prices_daily.parquet"
PATH_MKT   = ROOT/"data/processed/market_proxies.parquet"
PATH_MKTMAP= ROOT/"data/processed/market_proxy_mapping.yaml"
PATH_C18   = ROOT/"data/universe/companies_18.csv"
OUT_RPT    = ROOT/"reports/phase4_robustness.txt"
OUT_CSV    = ROOT/"reports/phase4_robustness_detail.csv"

EST_WIN_START=-250; EST_WIN_END=-20; MIN_EST=60; PREF_EST=120
MAIN_WINDOW = (0, 21)  # 주 분석 창


# ── 공통 유틸 (phase4_car_event_study.py 에서 가져옴) ──────────

def build_company_data(cids):
    prices = pd.read_parquet(PATH_PRICES)
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices[prices["company_id"].isin(cids)].sort_values(["company_id","date"])
    prices["price"] = prices["adj_close"].fillna(prices["close"])

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

    ew = (prices.assign(ret=lambda d: d.groupby("company_id")["price"]
                        .transform(lambda s: s.pct_change()))
          .dropna(subset=["ret"]).groupby("date")["ret"].mean().rename("mkt_ret"))

    cd = {}
    for cid, g in prices.groupby("company_id"):
        g = g.sort_values("date").copy(); g["ret"] = g["price"].pct_change()
        g = g.dropna(subset=["ret"])
        if len(g)<30: continue
        mkt_s = ir.get(c2i.get(cid,""), ew)
        df = g.set_index("date")[["ret"]].join(mkt_s.to_frame(), how="inner").dropna()
        if len(df)<30: continue
        dates = df.index.values.astype("datetime64[D]")
        cd[cid] = dict(dates=dates, ret=df["ret"].values.astype(np.float64),
                       mkt=df["mkt_ret"].values.astype(np.float64),
                       d2i={d:i for i,d in enumerate(dates)})
    return cd


def ols_car(ret, mkt, es, ee, xs, xe):
    n = ee-es
    if n<MIN_EST or es<0 or xe>len(ret) or xs<0: return None
    er,em = ret[es:ee], mkt[es:ee]
    sm,sr,smm,smr = em.sum(),er.sum(),(em*em).sum(),(em*er).sum()
    denom = n*smm - sm*sm
    if abs(denom)<1e-15: return None
    beta = (n*smr-sm*sr)/denom; alpha = (sr-beta*sm)/n
    sig = float(np.sqrt(np.sum((er-(alpha+beta*em))**2)/max(1,n-2)))
    if sig<1e-10: return None
    vr,vm = ret[xs:xe], mkt[xs:xe]; el=len(vr)
    if el<max(1,int((xe-xs)*0.8)): return None
    car = float((vr-(alpha+beta*vm)).sum())
    mbar=em.mean(); S2=np.sum((em-mbar)**2)
    corr=1+1/n+((vm.mean()-mbar)**2/S2 if S2>1e-15 else 0)
    dp=sig*np.sqrt(el*corr)
    if dp<1e-15: return None
    t=car/dp; p=float(2*stats.t.sf(abs(t),df=max(1,n-2)))
    q="full" if n>=PREF_EST else "marginal"
    return dict(CAR=car, t=t, p=p, quality=q)


def find_t0(cd, shock_month, offset_months=1):
    """
    offset_months=1: T=0 = M+1 첫 거래일 (기존)
    offset_months=0: T=0 = M 첫 거래일 (대안)
    """
    try:
        period = pd.Period(shock_month, "M") + offset_months
        t0 = np.datetime64(period.to_timestamp().date(), "D")
    except: return None
    pos = np.searchsorted(cd["dates"], t0, side="left")
    return int(pos) if pos < len(cd["dates"]) else None


def caar_stats(vals):
    n = len(vals)
    if n==0: return dict(N=0, CAAR=np.nan, t=np.nan, p=np.nan)
    cv = vals.mean(); sd = vals.std(ddof=1) if n>1 else np.nan
    t,p = np.nan, np.nan
    if n>1 and not np.isnan(sd) and sd>1e-15:
        t=cv/(sd/np.sqrt(n)); p=float(2*stats.t.sf(abs(t),df=n-1))
    return dict(N=n, CAAR=float(cv), t=float(t) if not np.isnan(t) else np.nan,
                p=float(p) if not np.isnan(p) else np.nan)


# ── [1] T=0 민감도 분석 ────────────────────────────────────────

def run_sensitivity_t0(tone, cd_map, cids):
    """T=0 = M+1 (기존) vs T=0 = M (대안) CAAR 비교"""
    c18 = pd.read_csv(PATH_C18)
    stage = dict(zip(c18["company_id"], c18["stage_bucket"]))

    events = []
    mask = tone["reliable_v2"] & (tone["neg_shock_v2"] | tone["pos_shock_v2"])
    seen = set()
    for _, r in tone[mask].iterrows():
        eid = f"{r['company_id']}_{r['year_month']}"
        if eid in seen: continue
        seen.add(eid)
        events.append(dict(shock_company=r["company_id"], shock_month=r["year_month"],
                           shock_type="NEG" if r["neg_shock_v2"] else "POS",
                           z_score=float(r.get("z_score_v2",0) or 0)))

    ws, we = MAIN_WINDOW
    results = []

    for offset, label in [(1, "T0=M+1 (기존)"), (0, "T0=M (대안)")]:
        cars_neg, cars_pos = [], []
        for ev in events:
            cd = cd_map.get(ev["shock_company"])
            if cd is None: continue
            t0 = find_t0(cd, ev["shock_month"], offset_months=offset)
            if t0 is None: continue
            es = max(0, t0+EST_WIN_START); ee = t0+EST_WIN_END
            if ee-es<MIN_EST: continue
            xs, xe = t0+ws, t0+we+1
            res = ols_car(cd["ret"], cd["mkt"], es, ee, xs, xe)
            if res is None or res["quality"]!="full": continue
            if ev["shock_type"]=="NEG": cars_neg.append(res["CAR"])
            else: cars_pos.append(res["CAR"])

        for shock_type, cars in [("NEG",cars_neg),("POS",cars_pos)]:
            s = caar_stats(np.array(cars))
            results.append(dict(T0_design=label, shock_type=shock_type, **s))

    return pd.DataFrame(results)


# ── [2] Cluster SE (이벤트-월 클러스터 부트스트랩) ──────────────

def clustered_caar(car_df: pd.DataFrame, B: int = 1000, seed: int = 42):
    """
    이벤트-월(shock_month) 클러스터 단위 부트스트랩 SE
    같은 달 여러 기업 동시 충격 → 이벤트 간 상관 존재
    → 클러스터 단위로 리샘플링하면 이 상관을 보정

    car_df: shock_month, CAR 컬럼 필요
    Returns: dict(CAAR, se_cluster, ci_lower, ci_upper, p_bootstrap)
    """
    rng = np.random.default_rng(seed)
    clusters = car_df["shock_month"].unique()
    nc = len(clusters)
    if nc < 5:
        s = caar_stats(car_df["CAR"].values)
        return dict(CAAR=s["CAAR"], se_cluster=np.nan,
                    ci_lo=np.nan, ci_hi=np.nan, p_boot=s["p"], n_cluster=nc)

    # 실제 CAAR
    caar_real = car_df["CAR"].mean()

    # 부트스트랩
    boot_caars = []
    for _ in range(B):
        chosen = rng.choice(clusters, size=nc, replace=True)
        boot_sample = pd.concat([car_df[car_df["shock_month"]==c] for c in chosen])
        boot_caars.append(boot_sample["CAR"].mean())

    boot_arr = np.array(boot_caars)
    se = float(np.std(boot_arr, ddof=1))
    ci_lo = float(np.percentile(boot_arr, 2.5))
    ci_hi = float(np.percentile(boot_arr, 97.5))
    # p-value: 부트스트랩 분포에서 0이 얼마나 극단적인지
    p = float(2 * min((boot_arr >= 0).mean(), (boot_arr <= 0).mean()))

    return dict(CAAR=float(caar_real), se_cluster=se,
                ci_lo=ci_lo, ci_hi=ci_hi, p_boot=p, n_cluster=nc)


def run_cluster_se(tone, cd_map, cids):
    """기존 CAR 결과에 Cluster SE 적용"""
    mask = tone["reliable_v2"] & tone["neg_shock_v2"]
    seen = set()
    events = []
    for _, r in tone[mask].iterrows():
        eid = f"{r['company_id']}_{r['year_month']}"
        if eid in seen: continue
        seen.add(eid)
        events.append(dict(shock_company=r["company_id"], shock_month=r["year_month"]))

    ws, we = MAIN_WINDOW
    car_rows = []
    for ev in events:
        cd = cd_map.get(ev["shock_company"])
        if cd is None: continue
        t0 = find_t0(cd, ev["shock_month"], offset_months=1)
        if t0 is None: continue
        es = max(0, t0+EST_WIN_START); ee = t0+EST_WIN_END
        if ee-es<MIN_EST: continue
        xs, xe = t0+ws, t0+we+1
        res = ols_car(cd["ret"], cd["mkt"], es, ee, xs, xe)
        if res and res["quality"]=="full":
            car_rows.append(dict(shock_month=ev["shock_month"], CAR=res["CAR"]))

    if not car_rows:
        return {}

    car_df = pd.DataFrame(car_rows)
    log.info("Cluster SE: %d개 이벤트, %d개 고유 월",
             len(car_df), car_df["shock_month"].nunique())
    return clustered_caar(car_df, B=1000)


# ── [3] 임계값 민감도 분석 ────────────────────────────────────

def run_threshold_sensitivity(tone, cd_map, cids):
    """z < -1.5, -2.0, -2.5 세 가지 임계값에서 CAAR 비교"""
    ws, we = MAIN_WINDOW
    results = []

    for thr in [-1.5, -2.0, -2.5]:
        mask = (tone["z_score_v2"] < thr) & tone["reliable_v2"] & tone["z_score_v2"].notna()
        cars = []
        seen = set()
        for _, r in tone[mask].iterrows():
            eid = f"{r['company_id']}_{r['year_month']}"
            if eid in seen: continue
            seen.add(eid)
            cd = cd_map.get(r["company_id"])
            if cd is None: continue
            t0 = find_t0(cd, r["year_month"], offset_months=1)
            if t0 is None: continue
            es = max(0, t0+EST_WIN_START); ee = t0+EST_WIN_END
            if ee-es<MIN_EST: continue
            xs, xe = t0+ws, t0+we+1
            res = ols_car(cd["ret"], cd["mkt"], es, ee, xs, xe)
            if res and res["quality"]=="full":
                cars.append(res["CAR"])
        s = caar_stats(np.array(cars))
        results.append(dict(threshold=thr, **s))
        log.info("  z<%.1f: N=%d  CAAR=%+.3f  p=%.3f",
                 thr, s["N"], s["CAAR"] if not np.isnan(s["CAAR"]) else 0, s["p"] if not np.isnan(s["p"]) else 1)

    return pd.DataFrame(results)


# ── Main ──────────────────────────────────────────────────────

def main():
    log.info("Phase 4 Robustness Checks 시작")
    tone = pd.read_parquet(PATH_TONE)
    c18  = pd.read_csv(PATH_C18)
    cids = c18["company_id"].tolist()
    log.info("CompanyData 구축 중...")
    cd_map = build_company_data(cids)

    SEP = "="*68

    # [1] T=0 민감도
    log.info("[1] T=0 민감도 분석...")
    t0_df = run_sensitivity_t0(tone, cd_map, cids)

    # [2] Cluster SE
    log.info("[2] Cluster SE (부트스트랩 B=1000)...")
    clust = run_cluster_se(tone, cd_map, cids)

    # [3] 임계값 민감도
    log.info("[3] 임계값 민감도 분석...")
    thr_df = run_threshold_sensitivity(tone, cd_map, cids)

    # 저장
    detail = pd.concat([t0_df, thr_df], ignore_index=True)
    detail.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    def sig(p):
        if np.isnan(p): return ""
        return "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.10 else ""

    lines = [SEP, "Phase 4 Robustness Checks 결과", SEP, ""]

    # T=0 결과
    lines += ["■ [1] T=0 설계 민감도 (창 [0,+21], NEG shock, full quality)", ""]
    lines.append(f"  {'T=0 설계':20s}  {'충격 유형':8s}  {'N':>4}  {'CAAR':>9}  {'t':>7}  {'p':>8}")
    lines.append("  "+"-"*60)
    for _, r in t0_df.iterrows():
        p = r["p"]
        lines.append(f"  {r['T0_design']:20s}  {r['shock_type']:8s}  "
                     f"{int(r['N']):4d}  {r['CAAR']:+9.4f}  {r['t']:+7.3f}  "
                     f"{p:.4f}{sig(p):3s}" if not np.isnan(p) else
                     f"  {r['CAAR']:+9.4f}  {'N/A':>7}")

    lines += ["",
              "  해석:",
              "  T0=M+1: 월별 감성이 이미 가격에 반영된 후 측정 → 평균회귀 포착",
              "  T0=M:   충격월 내 즉각 반응 포착 → 선행편의 가능성 있지만 동월 효과 확인",
              "  → 두 설계 모두 보고하면 설계 선택의 투명성 확보", ""]

    # Cluster SE 결과
    lines += ["■ [2] Cluster SE (이벤트-월 클러스터 부트스트랩, B=1000)", ""]
    if clust:
        lines.append(f"  CAAR         = {clust['CAAR']:+.4f}")
        lines.append(f"  SE (클러스터) = {clust['se_cluster']:.4f}  "
                     f"(기존 이분산 SE와 비교)")
        lines.append(f"  95% CI       = [{clust['ci_lo']:+.4f}, {clust['ci_hi']:+.4f}]")
        lines.append(f"  p (부트스트랩) = {clust['p_boot']:.4f}{sig(clust['p_boot']):3s}")
        lines.append(f"  클러스터 수   = {clust['n_cluster']}개 고유 이벤트-월")
        contains_zero = clust['ci_lo'] <= 0 <= clust['ci_hi']
        lines += ["",
                  f"  → 95% CI {'0 포함 (통계 비유의)' if contains_zero else '0 미포함 (유의)'}: "
                  f"클러스터링 후에도 {'결론 유지' if not contains_zero else '결론 약화'}"]
    else:
        lines.append("  [결과 없음]")

    # 임계값 민감도
    lines += ["", "■ [3] 충격 임계값 민감도 (직접 효과, [0,+21], NEG)", ""]
    lines.append(f"  {'임계값':10s}  {'N':>4}  {'CAAR':>9}  {'t':>7}  {'p':>8}")
    lines.append("  "+"-"*44)
    for _, r in thr_df.iterrows():
        p = r["p"]
        lines.append(f"  z < {r['threshold']:4.1f}    {int(r['N']):4d}  {r['CAAR']:+9.4f}  "
                     f"{r['t']:+7.3f}  {p:.4f}{sig(p):3s}" if not np.isnan(p) else
                     f"  z < {r['threshold']:4.1f}    {int(r['N']):4d}  {r['CAAR']:+9.4f}")
    lines += ["",
              "  해석: 임계값 변화에 따른 결과 방향/유의성 변화 → 신호 안정성 확인", ""]

    lines += [SEP]
    rpt = "\n".join(lines)
    OUT_RPT.write_text(rpt, encoding="utf-8")
    print("\n" + rpt)


if __name__ == "__main__":
    main()
