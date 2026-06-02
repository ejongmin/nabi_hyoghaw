"""
seed_edges v3 — seed_edges_appended.xlsx 정밀 통합
===================================================

근본 원인:
  GAT이 GRU를 일관되게 못 이기는 핵심 이유 중 하나는
  그래프 구조(엣지)가 불완전하다는 것.
  특히 Cell/Pack 레이어 내부 관계와
  Materials → Cell 경로의 다양성이 부족했음.

xlsx 처리 전략:
  1. 18개 기업 간 엣지만 추출 (103건)
  2. 노이즈 필터링 (자기루프, 역방향 오류, LLM 환각)
  3. PRODUCES → SUPPLIES 의미 정규화
  4. 기존 v2와 병합, 중복 제거
  5. 신규 유효 엣지만 추가

노이즈 판별 기준:
  - 자기루프 (src == dst) → 삭제
  - Cell → Cell SUPPLIES: 경쟁사끼리 공급 관계 불가 → COMPETES_WITH으로 변환
  - OEM → Cell SUPPLIES: 방향 역전 오류 → 삭제
  - Materials → OEM PRODUCES: Cell 레이어 건너뜀, LLM 추출 오류 → 삭제
  - Cell → Materials SUPPLIES: 역방향 오류 → 삭제
  - BUYS_FROM 역방향 정규화: A BUYS_FROM B → B SUPPLIES A
  - confidence < 0.75 → 삭제
  - 기존에 이미 삭제한 오류 엣지 재등장 방지
"""
from pathlib import Path
import pandas as pd, numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_XLSX = ROOT / "data/seed/seed_edges_appended.xlsx"
SRC_V2   = ROOT / "data/seed/seed_edges_18_internal_v2.csv"
OUT_V3   = ROOT / "data/seed/seed_edges_18_internal_v3.csv"
RPT      = ROOT / "reports/seed_edges_v3_report.txt"

c18   = pd.read_csv(ROOT / "data/universe/companies_18.csv")
nm    = dict(zip(c18["company_id"], c18["canonical_name"]))
stage = dict(zip(c18["company_id"], c18["stage_bucket"]))
FINAL_18 = set(c18["company_id"].tolist())

# 이미 삭제한 오류 엣지 — 재등장 방지
BLACKLIST = {
    ("300750.SZ", "SUPPLIES",      "3931.HK"),   # CATL → CALB (경쟁사)
    ("1211.HK",   "SUPPLIES",      "3931.HK"),   # BYD  → CALB (경쟁사)
    ("300750.SZ", "SUPPLIES",      "373220.KS"), # CATL → LGES (경쟁사)
    ("300014.SZ", "SUPPLIES",      "300750.SZ"), # EVE  → CATL (경쟁사)
    ("002074.SZ", "SUPPLIES",      "300750.SZ"), # Gotion → CATL (경쟁사)
    ("002074.SZ", "BUYS_FROM",     "300750.SZ"), # Gotion BUYS_FROM CATL
}

# 공급망 방향 검증
SUPPLY_FLOW = {
    "Upstream/Refining": 1,
    "Materials":         2,
    "Cell/Pack":         3,
    "OEM":               4,
}

def is_valid_supply_direction(src_cid, dst_cid):
    """SUPPLIES/PRODUCES는 upstream → downstream 방향이어야 함"""
    ss = SUPPLY_FLOW.get(stage.get(src_cid), 0)
    ds = SUPPLY_FLOW.get(stage.get(dst_cid), 0)
    return ss <= ds and ss > 0 and ds > 0 and ss != ds


def normalize_rel_type(rel: str) -> str:
    """PRODUCES 관계를 의미에 맞게 SUPPLIES로 변환"""
    if rel == "PRODUCES":
        return "SUPPLIES"
    if rel == "BUYS_FROM":
        return "SUPPLIES"   # 방향을 뒤집어서 SUPPLIES로 처리
    return rel


def flip_if_buys_from(row):
    """BUYS_FROM은 A→B: "A가 B로부터 구매" → B→A SUPPLIES로 변환"""
    if row["rel_type_orig"] == "BUYS_FROM":
        return {
            "src_company_id": row["dst_company_id"],
            "dst_company_id": row["src_company_id"],
            "rel_type": "SUPPLIES",
        }
    return {
        "src_company_id": row["src_company_id"],
        "dst_company_id": row["dst_company_id"],
        "rel_type": normalize_rel_type(row["rel_type_orig"]),
    }


def process_xlsx():
    df = pd.read_excel(SRC_XLSX)
    df = df.rename(columns={"rel_type": "rel_type_orig"})

    # 1) 18개 기업 간 엣지만
    df = df[df["src_company_id"].isin(FINAL_18) &
            df["dst_company_id"].isin(FINAL_18)].copy()

    # 2) 자기루프 제거
    df = df[df["src_company_id"] != df["dst_company_id"]]

    # 3) confidence 임계값
    df = df[df["confidence_plink"] >= 0.75]

    # 4) 관련 rel_type만
    KEEP_REL = {"SUPPLIES","BUYS_FROM","PRODUCES","COMPETES_WITH",
                "PARTNERS_WITH","OWNS","SOURCES_FROM"}
    df = df[df["rel_type_orig"].isin(KEEP_REL)]

    # 5) BUYS_FROM 방향 반전 + rel 정규화
    flipped = df.apply(flip_if_buys_from, axis=1, result_type="expand")
    df = df.drop(columns=["src_company_id","dst_company_id"])
    df = pd.concat([flipped, df.drop(columns=["rel_type_orig"])], axis=1)

    # 6) 블랙리스트 제거
    df["_key"] = list(zip(df["src_company_id"],
                          df["rel_type"],
                          df["dst_company_id"]))
    df = df[~df["_key"].isin(BLACKLIST)].drop(columns=["_key"])

    # 7) SUPPLIES/SOURCES_FROM: 방향 검증
    supply_types = {"SUPPLIES", "SOURCES_FROM", "OWNS"}
    bad_direction = df["rel_type"].isin(supply_types) & \
                    ~df.apply(lambda r: is_valid_supply_direction(
                        r["src_company_id"], r["dst_company_id"]), axis=1)
    removed_dir = df[bad_direction].copy()
    df = df[~bad_direction]

    # 8) 중복 제거 (같은 src-rel-dst, confidence 높은 것 유지)
    df = df.sort_values("confidence_plink", ascending=False)
    df = df.drop_duplicates(
        subset=["src_company_id","rel_type","dst_company_id"], keep="first")

    return df, removed_dir


def main():
    v2 = pd.read_csv(SRC_V2)
    v2_keys = set(zip(v2["src_company_id"], v2["rel_type"], v2["dst_company_id"]))

    df_clean, removed_dir = process_xlsx()

    # 신규 엣지만 (v2에 없는 것)
    df_clean["_key"] = list(zip(df_clean["src_company_id"],
                                df_clean["rel_type"],
                                df_clean["dst_company_id"]))
    new_edges = df_clean[~df_clean["_key"].isin(v2_keys)].drop(columns=["_key"])
    dup_edges = df_clean[df_clean["_key"].isin(v2_keys)].drop(columns=["_key"])

    print(f"xlsx 정제 결과:")
    print(f"  방향 오류로 제거: {len(removed_dir)}건")
    print(f"  v2에 이미 있음:  {len(dup_edges)}건")
    print(f"  신규 유효 엣지:  {len(new_edges)}건")
    print()
    print("=== 추가될 신규 엣지 ===")
    for _, r in new_edges.sort_values(
            ["confidence_plink"], ascending=False).iterrows():
        sn = nm.get(r["src_company_id"], r["src_company_id"])[:20]
        dn = nm.get(r["dst_company_id"], r["dst_company_id"])[:20]
        ss = stage.get(r["src_company_id"],"?")[:4]
        ds = stage.get(r["dst_company_id"],"?")[:4]
        print(f"  [{ss}→{ds}] {sn:22s} --[{r['rel_type']:13s}]--> "
              f"{dn:22s}  cf={r['confidence_plink']:.2f}")

    # COMPETES_WITH 양방향 보완
    comp_new = new_edges[new_edges["rel_type"]=="COMPETES_WITH"].copy()
    reverse_rows = []
    all_edges_tmp = pd.concat([v2, new_edges], ignore_index=True)
    existing_all = set(zip(all_edges_tmp["src_company_id"],
                           all_edges_tmp["rel_type"],
                           all_edges_tmp["dst_company_id"]))
    for _, r in comp_new.iterrows():
        rev_key = (r["dst_company_id"], "COMPETES_WITH", r["src_company_id"])
        if rev_key not in existing_all:
            rev = r.copy()
            rev["src_company_id"], rev["dst_company_id"] = \
                r["dst_company_id"], r["src_company_id"]
            reverse_rows.append(rev)

    if reverse_rows:
        new_edges = pd.concat([new_edges, pd.DataFrame(reverse_rows)],
                              ignore_index=True)
        print(f"\n  COMPETES_WITH 역방향 추가: {len(reverse_rows)}건")

    # v2 + new → v3
    cols = v2.columns.tolist()
    for c in cols:
        if c not in new_edges.columns:
            new_edges[c] = None
    new_edges = new_edges[cols]

    v3 = pd.concat([v2, new_edges], ignore_index=True)
    v3 = v3.drop_duplicates(
        subset=["src_company_id","rel_type","dst_company_id"],
        keep="first").reset_index(drop=True)

    # 날짜 컬럼 ISO 문자열 통일 (2021-01-01 형식 → parse 오류 방지)
    for dcol in ["valid_from", "valid_to"]:
        v3[dcol] = pd.to_datetime(v3[dcol], errors="coerce").dt.strftime("%Y-%m-%d")
        v3[dcol] = v3[dcol].replace("NaT", "")

    v3.to_csv(OUT_V3, index=False)

    # 통계
    v3["src_stage"] = v3["src_company_id"].map(stage)
    v3["dst_stage"] = v3["dst_company_id"].map(stage)
    mat = v3.groupby(["src_stage","dst_stage"]).size().unstack(fill_value=0)

    lines = ["="*68, "seed_edges v3 통합 보고서", "="*68, "",
             f"v2: {len(v2)}건  →  v3: {len(v3)}건  (신규 +{len(v3)-len(v2)}건)", "",
             "■ 추가된 신규 엣지"]
    for _, r in new_edges.head(30).iterrows():
        sn = nm.get(r["src_company_id"],r["src_company_id"])[:20]
        dn = nm.get(r["dst_company_id"],r["dst_company_id"])[:20]
        lines.append(f"  {sn:22s} --[{r['rel_type']:13s}]--> {dn}")
    lines += ["", "■ 레이어 간 매트릭스 (v3)", mat.to_string(), "", "="*68]

    rpt_text = "\n".join(lines)
    RPT.write_text(rpt_text, encoding="utf-8")
    print("\n" + rpt_text)


if __name__ == "__main__":
    main()
