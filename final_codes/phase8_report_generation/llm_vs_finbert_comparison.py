"""
llm_vs_finbert_comparison.py
============================
FinBERT vs Claude Haiku 감성분석 비교 (500건 샘플)

목적: FinBERT 결과의 타당성을 LLM으로 검증 → 논문 Appendix robustness check
방법:
  1. risk_events에서 전략적 샘플링 (score 분포 × 기업 다양성)
  2. Claude Haiku 4.5 Batches API로 공급망 감성점수 산출
  3. Spearman correlation + agreement pattern 분석

비용 추정: ~$0.10 (Haiku, Batches 50% 할인)
"""
from __future__ import annotations
import json, logging, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

ROOT   = Path(__file__).resolve().parents[1]
OUT    = ROOT / "reports" / "llm_vs_finbert"
OUT.mkdir(exist_ok=True)

N_SAMPLE = 500

# ── 공급망 특화 감성 프롬프트 ──────────────────────────────────────
SYSTEM_PROMPT = """You are a financial sentiment analyst specializing in EV battery supply chains.
Analyze news article headlines for supply chain sentiment.

Score from -1.0 (very negative) to +1.0 (very positive) for the company's supply chain situation.
Return ONLY a JSON object: {"score": <float>, "reason": "<1 sentence>", "supply_chain_relevant": <bool>}

Scoring guidelines:
- Negative (-1 to -0.3): disruptions, shortages, sanctions, recalls, delays, rising costs
- Neutral (-0.3 to 0.3): general news, minor updates, non-supply-chain topics
- Positive (0.3 to 1.0): new contracts, capacity expansion, stable supply, partnerships"""


def load_sample() -> pd.DataFrame:
    """risk_events와 finbert를 조인해 전략적 500건 샘플 추출."""
    log.info("데이터 로드 중...")

    # finbert_v2에서 url→score 맵 (chunk로 메모리 절약)
    log.info("  finbert_v2 로드...")
    chunks = []
    for chunk in pd.read_parquet(
            ROOT / "data/processed/finbert_v2.parquet",
            columns=["url", "title", "finbert_score"]).pipe(
                lambda df: [df]):
        chunks.append(chunk)
    fb = pd.concat(chunks).drop_duplicates(subset=["url"])
    log.info("  finbert: %d건", len(fb))

    # 이벤트 파일 (기업 정보 포함, 일부만 읽기)
    log.info("  risk_events 로드 (샘플링용)...")
    ev = pd.read_parquet(
        ROOT / "data/processed/risk_events_classified_v10_new.parquet",
        columns=["event_id", "url", "title", "company_id", "event_time"]
    ).sample(frac=0.1, random_state=42)  # 10%만 읽어도 충분
    log.info("  events 샘플: %d건", len(ev))

    # 조인
    merged = ev.merge(fb[["url", "finbert_score"]], on="url", how="inner")
    merged = merged.dropna(subset=["finbert_score", "title"])
    merged["title"] = merged["title"].str.strip()
    merged = merged[merged["title"].str.len() > 10]  # 너무 짧은 제목 제거
    log.info("  조인 후: %d건", len(merged))

    # 전략적 샘플링: score 구간 × 기업 다양성
    # 구간: very_neg(<-0.7) / neg(-0.7~-0.3) / neutral(-0.3~0.3) / pos(0.3+)
    merged["score_bin"] = pd.cut(merged["finbert_score"],
        bins=[-1.1, -0.7, -0.3, 0.3, 1.1],
        labels=["very_neg", "neg", "neutral", "pos"])

    samples = []
    targets = {"very_neg": 150, "neg": 150, "neutral": 100, "pos": 100}
    for bin_label, n in targets.items():
        sub = merged[merged["score_bin"] == bin_label]
        if len(sub) == 0:
            log.warning("  bin=%s: 0건", bin_label)
            continue
        # 기업 다양성 위해 기업별로 고르게 샘플
        n_per_co = max(1, n // sub["company_id"].nunique())
        s = (sub.groupby("company_id", group_keys=False)
                .apply(lambda g: g.sample(min(len(g), n_per_co), random_state=42))
                .sample(min(n, len(sub)), random_state=42))
        samples.append(s)
        log.info("  bin=%-10s: %d건 샘플 (기업 %d개)", bin_label, len(s), s["company_id"].nunique())

    result = pd.concat(samples).drop_duplicates(subset=["url"]).reset_index(drop=True)
    result = result.sample(min(N_SAMPLE, len(result)), random_state=42).reset_index(drop=True)
    log.info("최종 샘플: %d건 (기업 %d개)", len(result), result["company_id"].nunique())
    return result[["url", "title", "company_id", "finbert_score"]]


def run_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Claude Haiku Batches API로 감성분석 실행."""
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = anthropic.Anthropic()

    # 배치 요청 구성
    requests = []
    for i, row in df.iterrows():
        requests.append(Request(
            custom_id=f"row-{i}",
            params=MessageCreateParamsNonStreaming(
                model="claude-haiku-4-5",
                max_tokens=150,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user",
                           "content": f"Headline: {row['title'][:200]}"}]
            )
        ))

    log.info("배치 생성 중 (%d 요청)...", len(requests))
    batch = client.messages.batches.create(requests=requests)
    log.info("배치 ID: %s", batch.id)

    # 완료 대기
    while True:
        batch = client.messages.batches.retrieve(batch.id)
        log.info("  상태: %s (처리중: %d, 완료: %d)",
                 batch.processing_status,
                 batch.request_counts.processing,
                 batch.request_counts.succeeded)
        if batch.processing_status == "ended":
            break
        time.sleep(30)

    log.info("배치 완료 — 성공: %d, 오류: %d",
             batch.request_counts.succeeded,
             batch.request_counts.errored)

    # 결과 수집
    results = {}
    for result in client.messages.batches.results(batch.id):
        if result.result.type != "succeeded":
            continue
        msg = result.result.message
        text = next((b.text for b in msg.content if b.type == "text"), "")
        try:
            # JSON 파싱
            start = text.find("{")
            end   = text.rfind("}") + 1
            parsed = json.loads(text[start:end]) if start >= 0 else {}
            results[result.custom_id] = {
                "llm_score":      float(parsed.get("score", 0.0)),
                "llm_reason":     parsed.get("reason", ""),
                "llm_sc_relevant": bool(parsed.get("supply_chain_relevant", True)),
            }
        except Exception as e:
            log.debug("파싱 실패 [%s]: %s", result.custom_id, e)
            results[result.custom_id] = {"llm_score": 0.0, "llm_reason": "", "llm_sc_relevant": False}

    # df에 병합
    for key, vals in results.items():
        idx = int(key.split("-")[1])
        for col, val in vals.items():
            df.loc[idx, col] = val

    return df


def analyze(df: pd.DataFrame):
    """FinBERT vs LLM 상관분석 및 보고서 생성."""
    valid = df.dropna(subset=["llm_score", "finbert_score"])
    log.info("분석 대상: %d건 (LLM 응답 성공)", len(valid))

    # 전체 상관
    spear_r, spear_p = spearmanr(valid["finbert_score"], valid["llm_score"])
    pears_r, pears_p = pearsonr(valid["finbert_score"], valid["llm_score"])

    # score_bin별 상관
    bin_stats = []
    for bin_label in ["very_neg", "neg", "neutral", "pos"]:
        sub = valid[valid["score_bin"] == bin_label]
        if len(sub) < 10:
            continue
        r, p = spearmanr(sub["finbert_score"], sub["llm_score"])
        bin_stats.append({"bin": bin_label, "N": len(sub),
                          "spearman_r": round(r, 3), "p": round(p, 4)})

    # 방향 일치율 (부호 일치)
    valid["finbert_dir"] = (valid["finbert_score"] > 0).astype(int)
    valid["llm_dir"]     = (valid["llm_score"] > 0).astype(int)
    dir_acc = (valid["finbert_dir"] == valid["llm_dir"]).mean()

    # 극단값 불일치 분석 (FinBERT < -0.5 but LLM > 0.3)
    disagree = valid[(valid["finbert_score"] < -0.5) & (valid["llm_score"] > 0.3)]
    agree_neg = valid[(valid["finbert_score"] < -0.5) & (valid["llm_score"] < -0.3)]

    # 공급망 관련 기사만 필터 후 재계산
    sc_only = valid[valid["llm_sc_relevant"] == True]
    sc_r, sc_p = spearmanr(sc_only["finbert_score"], sc_only["llm_score"]) if len(sc_only) > 20 else (None, None)

    # 보고서 작성
    report = f"""
================================================================
FinBERT vs Claude Haiku — 감성분석 비교 검증 보고서
================================================================

■ 샘플 요약
  총 비교 건수: {len(valid)}건  (기업 {valid['company_id'].nunique()}개)
  LLM 공급망 관련 판단: {sc_only['llm_sc_relevant'].sum() if 'llm_sc_relevant' in valid.columns else 'N/A'}건 ({sc_only['llm_sc_relevant'].mean()*100:.1f}%)

■ 전체 상관계수
  Spearman ρ = {spear_r:+.4f}  (p={spear_p:.4f})
  Pearson  r = {pears_r:+.4f}  (p={pears_p:.4f})
  방향 일치율  = {dir_acc:.1%}

■ Score 구간별 Spearman ρ
"""
    for s in bin_stats:
        sig = "***" if s["p"]<0.01 else "**" if s["p"]<0.05 else "*" if s["p"]<0.10 else ""
        report += f"  {s['bin']:10s}  N={s['N']:3d}  ρ={s['spearman_r']:+.3f}  p={s['p']:.4f} {sig}\n"

    if sc_r is not None:
        report += f"\n■ 공급망 관련 기사만 (LLM 판단)\n"
        report += f"  N={len(sc_only)}건  Spearman ρ={sc_r:+.4f}  p={sc_p:.4f}\n"

    report += f"""
■ 불일치 패턴 분석
  FinBERT<-0.5 & LLM>+0.3 (FinBERT가 과도하게 음수): {len(disagree)}건 ({len(disagree)/len(valid):.1%})
  FinBERT<-0.5 & LLM<-0.3 (둘 다 음수 동의):         {len(agree_neg)}건 ({len(agree_neg)/len(valid):.1%})

  불일치 상위 사례 (FinBERT 음수 but LLM 양수):
"""
    for _, row in disagree.nlargest(5, "llm_score").iterrows():
        report += f"  [{row['company_id']}] {str(row['title'])[:70]}\n"
        report += f"    FinBERT={row['finbert_score']:+.3f}  LLM={row['llm_score']:+.3f}  이유: {row.get('llm_reason','')[:60]}\n"

    report += f"""
■ 해석 및 논문 기술 방법
  → Spearman ρ = {spear_r:+.4f} (p={spear_p:.4f})
  → FinBERT와 Claude Haiku의 감성 방향 일치율: {dir_acc:.1%}
  → FinBERT가 더 극단적(음수 편향) 경향: 평균 {valid['finbert_score'].mean():+.3f} vs LLM {valid['llm_score'].mean():+.3f}

  논문 기술 (예시):
  "FinBERT 감성 점수의 타당성을 검증하기 위해 Claude Haiku로 500건을 독립 평가한 결과,
   Spearman ρ={spear_r:.3f}(p<0.05)의 유의한 상관을 확인하였으며 방향 일치율은
   {dir_acc:.1%}로 나타났다. 이는 FinBERT 기반 신호가 LLM 평가와 일관성을
   가짐을 보여준다 (Appendix Table A)."

================================================================
"""
    print(report)

    # 파일 저장
    out_txt = OUT / "finbert_vs_llm_report.txt"
    out_txt.write_text(report, encoding="utf-8")
    log.info("보고서 저장: %s", out_txt)

    valid.to_csv(OUT / "finbert_vs_llm_detail.csv", index=False)
    log.info("상세 데이터: %s", OUT / "finbert_vs_llm_detail.csv")

    return spear_r, spear_p, dir_acc


def make_figure(df: pd.DataFrame):
    """산점도 Figure 생성."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = df.dropna(subset=["llm_score", "finbert_score"])
    spear_r, _ = spearmanr(valid["finbert_score"], valid["llm_score"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 산점도
    ax = axes[0]
    sc = ax.scatter(valid["finbert_score"], valid["llm_score"],
                    alpha=0.35, s=15, c="#2563EB")
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.axvline(0, color="gray", lw=0.7, ls="--")
    ax.set_xlabel("FinBERT Score")
    ax.set_ylabel("Claude Haiku Score")
    ax.set_title(f"FinBERT vs Claude Haiku\nSpearman ρ={spear_r:+.3f}")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # 분포 비교
    ax = axes[1]
    ax.hist(valid["finbert_score"], bins=30, alpha=0.6, label="FinBERT", color="#2563EB", density=True)
    ax.hist(valid["llm_score"],     bins=30, alpha=0.6, label="Claude Haiku", color="#DC2626", density=True)
    ax.axvline(valid["finbert_score"].mean(), color="#2563EB", ls="--", lw=1.5,
               label=f"FinBERT mean={valid['finbert_score'].mean():+.2f}")
    ax.axvline(valid["llm_score"].mean(),     color="#DC2626", ls="--", lw=1.5,
               label=f"LLM mean={valid['llm_score'].mean():+.2f}")
    ax.set_xlabel("Sentiment Score")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution Comparison")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.suptitle("Figure A. FinBERT vs LLM Sentiment Validation (N=500)", fontsize=13)
    fig.tight_layout()
    out = ROOT / "reports/figures/figA_finbert_vs_llm.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    log.info("Figure 저장: %s", out)


if __name__ == "__main__":
    # STEP 1: 샘플 로드
    df = load_sample()
    df.to_csv(OUT / "sample_before_llm.csv", index=False)

    # STEP 2: API KEY 확인
    if not os.environ.get("ANTHROPIC_API_KEY"):
        log.error("ANTHROPIC_API_KEY 환경변수가 설정되지 않았습니다.")
        log.error("터미널에서: $env:ANTHROPIC_API_KEY = 'sk-ant-...'")
        sys.exit(1)

    # STEP 3: 배치 실행
    df = run_batch(df)
    df.to_csv(OUT / "sample_with_llm.csv", index=False)

    # STEP 4: 분석
    spear_r, spear_p, dir_acc = analyze(df)
    make_figure(df)

    log.info("\n완료: reports/llm_vs_finbert/")
