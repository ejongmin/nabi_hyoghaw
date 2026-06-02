"""
01_collect_news_44.py
=====================
신규 26개 기업 GDELT DOC API 뉴스 수집 (2017-01 ~ 2025-12)

- 기존 18개 기업은 tone_v3에 이미 존재하므로 신규 26개만 수집
- 월별 × 기업별 루프, 캐시 활용 (이미 수집된 월은 스킵)
- rate limit: 1.2초/요청
- 완료 후 모든 기업 통합 parquet 저장

Usage:
    python experiments/universe_44/scripts/01_collect_news_44.py
    python experiments/universe_44/scripts/01_collect_news_44.py --force   # 캐시 무시 재수집
"""
from __future__ import annotations

import argparse
import calendar
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd

# ─────────────────────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parents[2]   # experiments/universe_44/
# ROOT가 올바른 nabi_hyoghaw 레포를 가리키도록 보정
# __file__: .../experiments/universe_44/scripts/01_collect_news_44.py
# parents[2] → experiments/universe_44/
# nabi_hyoghaw/가 필요하므로 parents[3] 사용
ROOT     = Path(__file__).resolve().parents[3]   # nabi_hyoghaw/
DATA_DIR = ROOT / "experiments" / "universe_44" / "data"
RAW_DIR  = DATA_DIR / "raw" / "gdelt_44"
OUT_PATH = RAW_DIR / "articles_raw.parquet"

# ─────────────────────────────────────────────────────────────
# 로깅
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# 신규 26개 기업 검색 키워드
# ─────────────────────────────────────────────────────────────
NEW_COMPANY_SEARCH_TERM: dict[str, str] = {
    "300037.SZ": "Capchem",
    "300207.SZ": "Sunwoda",
    "920185.BJ": "BTR",           # BTR New Material
    "002340.SZ": "GEM",
    "688388.SH": "Jiayuan",
    "600110.SH": "Nuode",
    "603659.SH": "Putailai",
    "600733.SH": "BAIC",          # BAIC BluePark
    "000625.SZ": "Changan",
    "600006.SH": "Dongfeng",
    "601238.SH": "GAC",
    "0175.HK":   "Geely",
    "601633.SH": "GreatWall",     # Great Wall Motor
    "600418.SH": "JAC",
    "603993.SH": "CMOC",
    "002460.SZ": "Ganfeng",
    "600362.SH": "JiangxiCopper",
    "002466.SZ": "Tianqi",
    "3407.T":    "AsahiKasei",
    "066970.KS": "LF",            # L&F
    "003670.KS": "POSCO",         # POSCO Future M
    "F":         "Ford",
    "GM":        "GeneralMotors",
    "LCID":      "Lucid",
    "7203.T":    "Toyota",
    "RIVN":      "Rivian",
}

# 수집 기간: 2017-01 ~ 2025-12
COLLECT_START = (2017, 1)
COLLECT_END   = (2025, 12)


# ─────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────

def _all_year_months() -> list[str]:
    """COLLECT_START ~ COLLECT_END 사이의 모든 'YYYY-MM' 문자열 반환."""
    ym_list: list[str] = []
    y, m = COLLECT_START
    ey, em = COLLECT_END
    while (y, m) <= (ey, em):
        ym_list.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return ym_list


def _month_date_range(ym: str) -> tuple[str, str]:
    """'YYYY-MM' → ('YYYY-MM-01', 'YYYY-MM-DD') 반환."""
    year, month = int(ym[:4]), int(ym[5:7])
    last_day = calendar.monthrange(year, month)[1]
    return f"{ym}-01", f"{ym}-{last_day:02d}"


# ─────────────────────────────────────────────────────────────
# 핵심 수집 함수
# ─────────────────────────────────────────────────────────────

def collect_company_month(
    cid: str,
    keyword: str,
    ym: str,
    raw_dir: Path,
    force: bool = False,
) -> pd.DataFrame:
    """
    단일 기업 × 단일 월 GDELT DOC API 수집.

    캐시 경로: raw_dir / {cid} / gdelt_{ym}.csv
    캐시 존재 시 스킵 (force=True면 재수집).
    반환: 수집된 기사 DataFrame (빈 DF 포함).
    """
    cache_dir  = raw_dir / cid
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"gdelt_{ym}.csv"

    # 캐시 히트
    if cache_file.exists() and not force:
        try:
            df = pd.read_csv(cache_file)
            if len(df) > 0:
                log.debug("캐시 로드  [%s] %s : %d건", cid, ym, len(df))
            return df
        except Exception as e:
            log.warning("캐시 읽기 실패 [%s] %s: %s → 재수집", cid, ym, e)

    # GDELT 쿼리
    try:
        from gdeltdoc import GdeltDoc, Filters
    except ImportError:
        raise RuntimeError(
            "gdeltdoc 패키지 미설치. 'pip install gdeltdoc' 실행 후 재시도."
        )

    start_date, end_date = _month_date_range(ym)
    gd = GdeltDoc()
    filters = Filters(
        keyword=keyword,
        start_date=start_date,
        end_date=end_date,
        language="English",
    )

    df = pd.DataFrame()
    for attempt in range(3):
        try:
            result = gd.article_search(filters)
            if result is not None and len(result) > 0:
                df = result
            break
        except Exception as exc:
            wait = 5 * (2 ** attempt)
            log.warning(
                "  [%s] %s 재시도 %d/3 (%.0fs 대기): %s",
                cid, ym, attempt + 1, wait, exc,
            )
            if attempt < 2:
                time.sleep(wait)

    if not df.empty:
        df["_cid"]     = cid
        df["_keyword"] = keyword
        df["_ym"]      = ym
        # URL 기반 중복 제거
        if "url" in df.columns:
            df = df.drop_duplicates(subset=["url"])

    # 캐시 저장 (빈 DF도 저장해 "수집 완료" 마킹)
    df.to_csv(cache_file, index=False, encoding="utf-8-sig")
    return df


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────

def main(force: bool = False) -> None:
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
        log.warning("tqdm 미설치. 진행바 없이 실행 ('pip install tqdm' 권장).")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    companies = list(NEW_COMPANY_SEARCH_TERM.items())   # [(cid, keyword), ...]
    all_months = _all_year_months()

    total_tasks = len(companies) * len(all_months)
    log.info(
        "수집 시작: %d개 기업 × %d개 월 = %d 태스크",
        len(companies), len(all_months), total_tasks,
    )
    log.info("수집 기간: %s ~ %s", all_months[0], all_months[-1])
    log.info("캐시 경로: %s", RAW_DIR)

    all_frames: list[pd.DataFrame] = []
    done = 0

    # 기업 × 월 루프 (월 외부, 기업 내부 순서)
    company_iter = tqdm(companies, desc="기업", unit="cid") if use_tqdm else companies
    for cid, keyword in company_iter:
        if use_tqdm:
            company_iter.set_postfix(cid=cid)  # type: ignore[union-attr]

        month_iter = (
            tqdm(all_months, desc=f"  {cid}", leave=False, unit="월")
            if use_tqdm
            else all_months
        )
        for ym in month_iter:
            df = collect_company_month(cid, keyword, ym, RAW_DIR, force=force)
            if not df.empty:
                all_frames.append(df)

            done += 1
            if done % 100 == 0:
                log.info(
                    "진행: %d/%d (%.1f%%)  수집 기사 누계: %d건",
                    done, total_tasks, 100 * done / total_tasks,
                    sum(len(f) for f in all_frames),
                )

            # Rate limit
            time.sleep(1.2)

    # 통합 parquet 저장
    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        # 전체 URL 기반 중복 제거
        if "url" in combined.columns:
            before = len(combined)
            combined = combined.drop_duplicates(subset=["url"])
            log.info("URL 중복 제거: %d → %d행", before, len(combined))

        combined.to_parquet(OUT_PATH, index=False)
        log.info("저장 완료: %s  (%d행)", OUT_PATH, len(combined))

        # 기업별 수집 요약
        log.info("\n=== 기업별 수집 결과 ===")
        if "_cid" in combined.columns:
            summary = (
                combined.groupby("_cid")
                .size()
                .reset_index(name="n_articles")
                .sort_values("n_articles", ascending=False)
            )
            for _, row in summary.iterrows():
                log.info("  %-20s %5d건", row["_cid"], row["n_articles"])
    else:
        log.warning("수집된 기사 없음. 네트워크 또는 API 상태를 확인하세요.")
        # 빈 parquet 저장 (파이프라인 연속성 유지)
        pd.DataFrame().to_parquet(OUT_PATH, index=False)

    log.info("완료.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GDELT 신규 26개 기업 뉴스 수집")
    parser.add_argument(
        "--force",
        action="store_true",
        help="캐시 무시하고 전체 재수집",
    )
    args = parser.parse_args()
    main(force=args.force)
