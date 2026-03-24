# seed_edges 자동 변환(엑셀/문서) 가이드

## 1) 목적
팀 조사자료(공급망/협업/경쟁 등)를 `data/seed/seed_edges.csv`로 자동 변환하여
KG 생성 및 노출도 계산에서 사용합니다.

---

## 2) 엑셀 템플릿
- `data/seed/seed_edges_template.xlsx`를 기준으로 작성하면 매핑 성공률이 높습니다.
- 필수 컬럼 개념:
  - src(공급/원천 기업)
  - dst(수요/대상 기업)
- 나머지는 선택(없으면 기본값 적용)

---

## 3) 컬럼명이 다르면?
`configs/seed_mapping.yaml`에서 alias 목록을 추가하세요.
예: src 컬럼명이 `기업A`라면, `columns.src` 목록에 `기업A`를 추가.

---

## 4) 관계 타입(rel_type) 표준화
`configs/seed_mapping.yaml`의 relation_normalization에
한국어/영어 표현 alias를 추가하면 자동으로 표준 관계로 바뀝니다.

---

## 5) 매핑 실패(회사명/티커가 universe에 없을 때)
- 보고서: `reports/seed_import_report.md`에 unmapped 예시가 출력됩니다.
- 해결:
  1) universe에 회사 alias(약칭/티커)를 `tickers` 또는 `canonical_name`에 추가
  2) seed_mapping.yaml에 컬럼 alias 추가
  3) fuzz score를 조정(config: seed_import.min_fuzzy_score)

