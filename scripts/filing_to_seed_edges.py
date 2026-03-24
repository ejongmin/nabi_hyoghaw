#!/usr/bin/env python3
"""
공시 자료 → Seed Edges 변환기
================================
SEC EDGAR, DART 등에서 추출한 공시 기반 엣지를
기존 seed_edges.csv에 병합.

기존 seed_edges.csv의 "report" source를 공시 기반으로 업그레이드하거나,
새로운 엣지를 추가합니다.

사용:
  python scripts/filing_to_seed_edges.py
  python scripts/filing_to_seed_edges.py --dry-run   # 병합 시뮬레이션만
  python scripts/filing_to_seed_edges.py --replace    # 기존 evidence를 공시로 교체

입력:
  data/raw/filings/sec_edges.parquet   (SEC EDGAR)
  data/raw/filings/dart_edges.parquet  (DART)

출력:
  data/seed/seed_edges.csv             (업데이트된 seed edges)
  data/seed/seed_edges_backup.csv      (백업)
  reports/filing_merge_report.md       (병합 리포트)
"""
import argparse
import logging
import shutil
from pathlib import Path
from typing import List

import pandas as pd

log = logging.getLogger("filing_to_seed")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


def load_filing_edges(filings_dir: Path) -> pd.DataFrame:
    """모든 공시 엣지 파일 로드 + 통합."""
    dfs = []

    for f in filings_dir.glob("*_edges.parquet"):
        df = pd.read_parquet(f)
        source_name = f.stem.replace("_edges", "")  # "sec", "dart"
        df["filing_source"] = source_name
        dfs.append(df)
        log.info(f"  Loaded {f.name}: {len(df)} edges")

    if not dfs:
        log.warning("No filing edge files found")
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    log.info(f"  Total filing edges: {len(combined)}")
    return combined


def normalize_edge(row) -> dict:
    """공시 엣지 → seed_edges.csv 형식으로 변환."""
    # evidence 구성
    filing_type = row.get("filing_type", "Filing")
    fy = row.get("fiscal_year", "")
    evidence = row.get("evidence_text", "")[:200]
    url = row.get("filing_url", "")
    filing_source = row.get("filing_source", "filing")

    source_label = {
        "sec": "SEC_10K",
        "dart": "DART_Annual",
    }.get(filing_source, f"filing_{filing_source}")

    return {
        "src_company_id": row["src_company_id"],
        "rel_type": row["rel_type"],
        "dst_company_id": row.get("dst_company_id", row.get("dst_raw_name", "")),
        "confidence_plink": row.get("confidence", 1.0),
        "strength": row.get("strength", 0.8),
        "evidence": f"{filing_type} FY{fy}: {evidence}",
        "source": source_label,
        "valid_from": f"{fy}-01-01" if fy else "",
        "valid_to": "",
        "risk_rationale": "",
    }


def merge_with_existing(existing: pd.DataFrame, filing_edges: pd.DataFrame,
                        replace: bool = False) -> tuple:
    """
    기존 seed_edges에 공시 엣지 병합.

    Returns:
        (merged_df, stats_dict)
    """
    stats = {
        "existing_total": len(existing),
        "filing_total": len(filing_edges),
        "upgraded": 0,
        "new_added": 0,
        "skipped_duplicate": 0,
    }

    # 공시 엣지 정규화
    normalized = []
    for _, row in filing_edges.iterrows():
        norm = normalize_edge(row)
        # dst_company_id가 company_id 형식인지 확인 (raw name이면 스킵)
        dst = norm["dst_company_id"]
        if not dst or (not any(c in dst for c in [".", "KS", "KQ", "SZ", "SH", "HK", "DE", "T", "L"])
                       and dst not in ["GM", "F", "TSLA", "RIVN", "LCID", "ALB", "LICY"]):
            continue
        normalized.append(norm)

    log.info(f"  Normalized: {len(normalized)} edges (from {len(filing_edges)})")

    # 기존 엣지와 매칭
    existing_keys = set(zip(
        existing["src_company_id"],
        existing["rel_type"],
        existing["dst_company_id"]
    ))

    new_rows = []
    upgrade_indices = {}

    for norm in normalized:
        key = (norm["src_company_id"], norm["rel_type"], norm["dst_company_id"])

        if key in existing_keys:
            if replace:
                # 기존 엣지의 evidence를 공시로 업그레이드
                mask = (
                    (existing["src_company_id"] == key[0]) &
                    (existing["rel_type"] == key[1]) &
                    (existing["dst_company_id"] == key[2])
                )
                idx = existing[mask].index
                if len(idx) > 0:
                    upgrade_indices[idx[0]] = norm
                    stats["upgraded"] += 1
            else:
                stats["skipped_duplicate"] += 1
        else:
            new_rows.append(norm)
            existing_keys.add(key)
            stats["new_added"] += 1

    # 업그레이드 적용
    for idx, norm in upgrade_indices.items():
        existing.loc[idx, "evidence"] = norm["evidence"]
        existing.loc[idx, "source"] = norm["source"]
        if norm["valid_from"]:
            existing.loc[idx, "valid_from"] = norm["valid_from"]

    # 새 엣지 추가
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        # 기존 컬럼과 맞추기
        for col in existing.columns:
            if col not in new_df.columns:
                new_df[col] = ""
        new_df = new_df[existing.columns]
        merged = pd.concat([existing, new_df], ignore_index=True)
    else:
        merged = existing.copy()

    stats["final_total"] = len(merged)
    return merged, stats


def generate_report(stats: dict, reports_dir: Path):
    """병합 리포트 생성."""
    reports_dir.mkdir(parents=True, exist_ok=True)

    md = [
        "# Filing → Seed Edges Merge Report",
        "",
        "## Summary",
        f"- Existing seed edges: {stats['existing_total']}",
        f"- Filing edges found: {stats['filing_total']}",
        f"- Upgraded (evidence replaced): {stats['upgraded']}",
        f"- New edges added: {stats['new_added']}",
        f"- Skipped (duplicate): {stats['skipped_duplicate']}",
        f"- **Final total: {stats['final_total']}**",
        "",
        "## Sources",
        "- SEC EDGAR 10-K (US companies)",
        "- DART Annual Report (Korean companies)",
        "",
        "## Notes",
        "- Confidence = 1.0 for all filing-based edges (legally mandated disclosures)",
        "- Evidence includes filing type, fiscal year, and extracted sentence",
        "- `valid_from` set to fiscal year start date",
        "",
    ]

    path = reports_dir / "filing_merge_report.md"
    path.write_text("\n".join(md), encoding="utf-8")
    log.info(f"Report saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Filing → Seed Edges Converter")
    parser.add_argument("--filings-dir", type=str, default="data/raw/filings",
                        help="Directory with filing edge parquets")
    parser.add_argument("--seed-csv", type=str, default="data/seed/seed_edges.csv",
                        help="Existing seed_edges.csv path")
    parser.add_argument("--replace", action="store_true",
                        help="Replace existing evidence with filing evidence")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate merge without writing")
    args = parser.parse_args()

    filings_dir = Path(args.filings_dir)
    seed_csv = Path(args.seed_csv)

    log.info("=== Filing → Seed Edges Merger ===")

    # Load filing edges
    log.info("Loading filing edges...")
    filing_edges = load_filing_edges(filings_dir)
    if filing_edges.empty:
        log.warning("No filing edges to merge. Run collect_sec_edgar.py and/or collect_dart_kr.py first.")
        return

    # Load existing seed edges
    log.info(f"Loading existing seed edges: {seed_csv}")
    existing = pd.read_csv(seed_csv)
    log.info(f"  Existing: {len(existing)} edges")

    # Merge
    log.info("Merging...")
    merged, stats = merge_with_existing(existing, filing_edges, replace=args.replace)

    log.info(f"\n=== Merge Results ===")
    for k, v in stats.items():
        log.info(f"  {k}: {v}")

    if args.dry_run:
        log.info("[DRY-RUN] No files written")
        return

    # Backup
    backup = seed_csv.with_name("seed_edges_pre_filing_backup.csv")
    shutil.copy2(seed_csv, backup)
    log.info(f"Backup: {backup}")

    # Save
    merged.to_csv(seed_csv, index=False, encoding="utf-8-sig")
    log.info(f"Saved: {seed_csv} ({len(merged)} edges)")

    # Report
    generate_report(stats, Path("reports"))


if __name__ == "__main__":
    main()
