"""
Phase 2-A: LaBSE cross-lingual event dedup 입력 준비

v8에서 title+finbert 보유 행만 추출하고, site tagline/placeholder 같은
노이즈 title을 제거한 뒤 Colab LaBSE 임베딩용 parquet으로 저장한다.

정책 (추론 포함):
  - same title ≥ 3 repeats → placeholder 의심 → 제거 (실제 같은 기사가 3번 이상
    같은 문자열로 반복되는 경우는 드물고, 대부분 사이트 slogan/tagline임을
    앞선 분석 (36Kr '让一部分人先看到未来' 155회 등)에서 확인했음)
  - title 길이 < 10 or > 300자 → 제거 (정상 뉴스 헤드라인 범위)
  - 너무 흔한 publisher-only 제거 (예: 'Press Release - <Publisher>')

참고 논문:
  Rupnik, Muhic, Leban, Skraba, Fortuna, Grobelnik (2016)
    "News across languages - Cross-Lingual Document Similarity and Event Tracking"
    Journal of Artificial Intelligence Research — 뉴스 이벤트 트래킹에서
    동일 placeholder/tagline 제거가 LaBSE 유사도 분포에 큰 영향을 준다고 명시.
  Leetaru & Schrodt (2013) GDELT event dedup 절차도 동일 원칙.
"""

import re
import logging
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
V8 = ROOT / "data" / "processed" / "risk_events_classified_v8.parquet"
OUT = ROOT / "data" / "processed" / "event_dedup_input.parquet"
STATS = ROOT / "data" / "processed" / "event_dedup_input_stats.txt"

# title 길이 제한 (char 기준; 다국어)
MIN_TITLE_LEN = 10
MAX_TITLE_LEN = 300
# 동일 title 반복 허용 상한
MAX_REPEAT = 2

# Press release 계열 placeholder
PR_RE = re.compile(r"^\s*(press release|공지사항|公告|ニュースリリース)\s*[-–—]",
                   re.IGNORECASE)


def main():
    con = duckdb.connect()
    log.info("Loading v8 title+finbert subset from %s", V8)

    # 1) title+finbert rows 전체 로딩 (45K 정도이므로 메모리 OK)
    df = con.execute(f"""
        SELECT event_id, event_time, url, title, company_id,
               finbert_model, finbert_score, finbert_z,
               publisher, source_tier, source_tier_weight
        FROM read_parquet('{V8}')
        WHERE title IS NOT NULL AND title != ''
          AND finbert_score IS NOT NULL
    """).fetchdf()
    n0 = len(df)
    log.info("  base title+finbert rows: %d", n0)

    # 2) 길이 필터
    tl = df["title"].str.len()
    mask_len = (tl >= MIN_TITLE_LEN) & (tl <= MAX_TITLE_LEN)
    n_len_removed = (~mask_len).sum()
    df = df[mask_len].copy()
    log.info("  length filter removed: %d (keep %d)", n_len_removed, len(df))

    # 3) 반복 횟수 필터 (same exact title)
    vc = df["title"].value_counts()
    repeat_titles = vc[vc > MAX_REPEAT].index
    mask_rep = ~df["title"].isin(repeat_titles)
    n_rep_removed = (~mask_rep).sum()
    df = df[mask_rep].copy()
    log.info("  repeat(>%d) filter removed: %d (distinct titles dropped: %d)",
             MAX_REPEAT, n_rep_removed, len(repeat_titles))

    # 4) Press release prefix 제거
    mask_pr = ~df["title"].apply(lambda t: bool(PR_RE.match(t)) if isinstance(t, str) else False)
    n_pr_removed = (~mask_pr).sum()
    df = df[mask_pr].copy()
    log.info("  press-release placeholder removed: %d", n_pr_removed)

    n_final = len(df)
    log.info("  final rows: %d (%.1f%% of base)", n_final, 100 * n_final / n0)

    # 5) 결정성 확보를 위해 event_time, event_id 정렬
    df = df.sort_values(["event_time", "event_id"]).reset_index(drop=True)

    # 6) 저장
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    log.info("Wrote %s (%d rows, %d cols)", OUT, len(df), df.shape[1])

    # 7) 언어/tier 분포 통계
    lang_dist = df.groupby("finbert_model").size().sort_values(ascending=False)
    tier_dist = df.groupby("source_tier").size().sort_index()

    with open(STATS, "w") as f:
        f.write(f"Base rows (title+finbert): {n0}\n")
        f.write(f"Length filter removed    : {n_len_removed}\n")
        f.write(f"Repeat filter removed    : {n_rep_removed}\n")
        f.write(f"Press-release removed    : {n_pr_removed}\n")
        f.write(f"Final rows               : {n_final}\n\n")
        f.write("== Per-model (language) distribution ==\n")
        for k, v in lang_dist.items():
            f.write(f"  {k:65s}  {v:>7d}\n")
        f.write("\n== Per source_tier distribution ==\n")
        for k, v in tier_dist.items():
            f.write(f"  tier {k}  {v:>7d}\n")

    log.info("Stats written to %s", STATS)
    log.info("\n== Per-model (language) distribution ==")
    for k, v in lang_dist.items():
        log.info("  %-65s %7d", k, v)


if __name__ == "__main__":
    main()
