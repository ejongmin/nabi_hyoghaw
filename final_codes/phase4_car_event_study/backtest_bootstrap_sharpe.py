"""
Bootstrap Sharpe Test
목적: Contrarian 전략 Sharpe가 Benchmark보다 유의하게 높은지 검증
방법: Block bootstrap (block_size=21 거래일, B=5000)
출력: reports/backtest_bootstrap_sharpe.txt
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "reports" / "backtest_equity.csv"
OUT_PATH = ROOT / "reports" / "backtest_bootstrap_sharpe.txt"

B = 5000
BLOCK_SIZE = 21          # 약 1거래월
TRADING_DAYS = 252
RNG = np.random.default_rng(42)


def sharpe(returns: np.ndarray) -> float:
    """연환산 Sharpe (무위험이자율 0 가정)."""
    if returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(TRADING_DAYS))


def block_bootstrap_sharpe_diff(bench_r: np.ndarray, strat_r: np.ndarray,
                                 b: int, block_size: int) -> np.ndarray:
    """
    bench_r와 strat_r에 동일한 블록 인덱스를 적용해
    샘플 간 상관 구조를 보존하면서 Sharpe 차이 분포를 생성.
    """
    T = len(bench_r)
    starts = np.arange(T - block_size + 1)
    n_blocks = int(np.ceil(T / block_size))
    diffs = np.empty(b)
    for i in range(b):
        chosen = RNG.choice(starts, size=n_blocks, replace=True)
        idx = np.concatenate([np.arange(s, s + block_size) for s in chosen])[:T]
        diffs[i] = sharpe(strat_r[idx]) - sharpe(bench_r[idx])
    return diffs


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # ── 컬럼 확인 ────────────────────────────────────────────────────
    print("컬럼:", df.columns.tolist())
    print("행 수:", len(df))

    # bench_ret / strat_ret이 이미 있으면 그대로 사용,
    # 없으면 equity 컬럼에서 pct_change 계산
    if "bench_ret" in df.columns and "strat_ret" in df.columns:
        df = df.dropna(subset=["bench_ret", "strat_ret"])
        bench_r = df["bench_ret"].values
        strat_r = df["strat_ret"].values
    else:
        bench_col = next(c for c in df.columns if "bench" in c.lower())
        strat_col = next(c for c in df.columns if "strat" in c.lower() or "contrarian" in c.lower())
        df["bench_ret"] = df[bench_col].pct_change()
        df["strat_ret"] = df[strat_col].pct_change()
        df = df.dropna(subset=["bench_ret", "strat_ret"])
        bench_r = df["bench_ret"].values
        strat_r = df["strat_ret"].values

    obs_bench_sharpe = sharpe(bench_r)
    obs_strat_sharpe = sharpe(strat_r)
    obs_diff = obs_strat_sharpe - obs_bench_sharpe

    # ── Block Bootstrap ───────────────────────────────────────────────
    boot_diffs = block_bootstrap_sharpe_diff(bench_r, strat_r, B, BLOCK_SIZE)

    ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
    # 단측: Sharpe 차이 > 0
    pval = (boot_diffs <= 0).sum() / B

    # ── 출력 ──────────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 60)
    lines.append("Bootstrap Sharpe Test 결과")
    lines.append("=" * 60)
    lines.append(f"\n데이터: {DATA_PATH.name}  (N={len(df)} 거래일)")
    lines.append(f"Block Bootstrap: B={B}, block_size={BLOCK_SIZE}거래일")
    lines.append(f"RNG seed: 42\n")

    lines.append("■ 실제 Sharpe 비교")
    lines.append(f"  Benchmark  Sharpe = {obs_bench_sharpe:+.4f}")
    lines.append(f"  Contrarian Sharpe = {obs_strat_sharpe:+.4f}")
    lines.append(f"  차이 (Strat-Bench) = {obs_diff:+.4f}\n")

    lines.append("■ Bootstrap 분포 (Sharpe 차이)")
    lines.append(f"  평균             = {boot_diffs.mean():+.4f}")
    lines.append(f"  표준오차         = {boot_diffs.std():.4f}")
    lines.append(f"  95% CI           = [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    lines.append(f"  p-value (단측, diff>0) = {pval:.4f}")
    sig = "**유의 (p<0.05)**" if pval < 0.05 else "비유의 (p>=0.05)"
    lines.append(f"  판정             → {sig}")
    lines.append(f"\n  ※ p-value = P(boot_diff <= 0): 귀무가설 'Strat Sharpe <= Bench Sharpe'")

    lines.append("\n" + "=" * 60)

    report = "\n".join(lines)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n저장 완료: {OUT_PATH}")


if __name__ == "__main__":
    main()
